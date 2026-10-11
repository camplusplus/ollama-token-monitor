#!/usr/bin/env python3
"""otop — live terminal dashboard for Ollama and OpenAI-compatible servers.

Shows, refreshed in real time:
  * output tokens / prompt tokens / tok/s per request and session totals
    (via a transparent proxy: point your client at --proxy-port instead of the server)
  * CPU usage and thread count of the ollama server + model runner processes
  * system CPU (overall + per-core), load average, RAM
  * Apple Silicon GPU utilization and GPU memory (via ioreg, no sudo)
  * models currently loaded: size, GPU/CPU split, context length, expiry

Usage:
    python otop.py                     # dashboard + proxy on :11435
    python otop.py --no-proxy          # dashboard only
    python otop.py --proxy-port 9999   # custom proxy port
    python otop.py --upstream http://192.168.0.23:8080/v1 --api-key SECRET

Send traffic through the proxy to get live token counts:
    OLLAMA_HOST=127.0.0.1:11435 ollama run llama3.2
    curl http://127.0.0.1:11435/api/generate -d '{"model":"...","prompt":"..."}'

Cline / any OpenAI-compatible client (the proxy injects the real api key, so
Cline's own key field can be any placeholder):
    ./otop.sh --upstream http://192.168.0.23:8080/v1 --api-key "$(cat key.txt)"
    # then in Cline settings:  openaiBaseUrl = http://127.0.0.1:11435/v1
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
import zlib
from collections import deque
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import psutil
from aiohttp import ClientSession, ClientTimeout, TCPConnector, web
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------- state

@dataclass
class RequestStats:
    id: int
    model: str
    path: str
    started: float
    prompt_tokens: int = 0
    output_tokens: int = 0
    done: bool = False
    ended: float = 0.0
    error: str = ""
    # timestamps of recently emitted tokens, for instantaneous tok/s
    recent: deque = field(default_factory=lambda: deque(maxlen=200))

    @property
    def elapsed(self) -> float:
        return (self.ended or time.time()) - self.started

    @property
    def avg_tps(self) -> float:
        return self.output_tokens / self.elapsed if self.elapsed > 0 else 0.0

    @property
    def live_tps(self) -> float:
        """tok/s over the last 2 seconds."""
        now = time.time()
        n = sum(1 for t in self.recent if now - t <= 2.0)
        return n / 2.0


class State:
    def __init__(self):
        self.requests: dict[int, RequestStats] = {}
        self.next_id = 1
        self.total_output = 0
        self.total_prompt = 0
        self.total_requests = 0
        self.session_start = time.time()
        # rolling history of session-wide output tok/s for the sparkline
        self.tps_history: deque = deque(maxlen=60)
        self.models = []          # /api/ps result
        self.ollama_version = ""
        self.ollama_ok = False
        self.gpu = {}             # ioreg stats
        self.procs = []           # per-process rows
        self.cpu_percent = 0.0
        self.per_core = []
        self.proxy_error = ""
        self.upstream_label = ""       # upstream base url shown in the header
        self.upstream_mode = ""        # "ollama" (native api) or "openai" (compatible)
        self.poll_error = ""           # why the model poll failed, e.g. "401 for /v1/models"
        self.api_key_set = False       # --api-key injected into upstream requests

    def new_request(self, model: str, path: str) -> RequestStats:
        r = RequestStats(self.next_id, model, path, time.time())
        self.next_id += 1
        self.requests[r.id] = r
        self.total_requests += 1
        # keep only the last 8 requests around for display
        for rid in sorted(self.requests)[:-8]:
            del self.requests[rid]
        return r

    def session_live_tps(self) -> float:
        return sum(r.live_tps for r in self.requests.values() if not r.done)


STATE = State()

# ---------------------------------------------------------------- proxy

STREAM_PATHS = ("/api/generate", "/api/chat", "/v1/chat/completions", "/v1/completions")
STREAM_SUFFIXES = ("/chat/completions", "/completions", "/api/chat", "/api/generate")
OPENAI_SUFFIXES = ("/chat/completions", "/completions", "/embeddings", "/models", "/responses")
HOP_HEADERS = {"transfer-encoding", "content-length", "connection", "keep-alive", "host"}
AUTH_HEADERS = {"authorization", "x-api-key", "api-key"}


@dataclass
class Upstream:
    """A base url split into what a proxy request path has to be mapped onto.

    root = "http://host:port"        (native ollama /api/* endpoints live here)
    base = "/v1" or ""               (path prefix the OpenAI-compatible server serves)
    """
    root: str
    base: str
    api_key: str = ""
    key_headers: tuple = ("authorization",)
    label: str = ""


def _split_upstream(url: str) -> tuple[str, str]:
    """("http://192.168.0.23:8080", "/v1") for http://192.168.0.23:8080/v1."""
    url = url.strip()
    if not re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", url):
        url = "http://" + url
    p = urlsplit(url.rstrip("/"))
    return f"{p.scheme}://{p.netloc}", p.path.rstrip("/")


def _endpoint(path: str, base: str) -> str:
    """The path the upstream actually serves, i.e. with the base prefix stripped."""
    if base and (path == base or path.startswith(base + "/")):
        return path[len(base):]
    return path


def _target_url(path: str, up: Upstream) -> str:
    """Absolute upstream url for a proxied request path.

    base "/v1":  /v1/chat/completions -> root/v1/chat/completions
                 /chat/completions    -> root/v1/chat/completions  (client skipped /v1)
                 /api/ps              -> root/api/ps               (native endpoint)
    no base:     everything is just appended to the root.
    """
    ep = _endpoint(path, up.base)
    if ep != path:
        return up.root + path
    if up.base and any(ep.endswith(s) for s in OPENAI_SUFFIXES):
        return up.root + up.base + ep
    return up.root + ep


def _is_stream_path(path: str, base: str) -> bool:
    ep = _endpoint(path, base)
    return ep in STREAM_PATHS or any(ep.endswith(s) for s in STREAM_SUFFIXES)


def _auth_headers(key: str, key_headers: tuple) -> dict:
    """Auth headers we send upstream: Bearer for `authorization`, bare for the rest.

    Shared by the proxy and by the dashboard pollers, so `--api-key-header x-api-key`
    / `both` also decides how /api/ps and /v1/models are authenticated.
    """
    out = {}
    if key:
        for h in key_headers:
            out[h] = f"Bearer {key}" if h == "authorization" else key
    return out


def _extract_model(body: bytes) -> str:
    try:
        return json.loads(body).get("model", "?")
    except Exception:
        return "?"


def _body_wants_stream(body: bytes) -> bool:
    """True when the request asked for a streamed response ("stream": true).

    Needed because llama.cpp / vLLM / some ollama builds answer SSE bodies with
    Content-Type: application/json, so the response header alone is not reliable.
    """
    try:
        return bool(json.loads(body).get("stream", False))
    except Exception:
        return b'"stream"' in body


def _count_ndjson_line(line: bytes, r: RequestStats):
    """One NDJSON line from /api/generate or /api/chat == one token chunk."""
    try:
        obj = json.loads(line)
    except Exception:
        return
    if obj.get("done"):
        r.done = True
        r.ended = time.time()
        # trust ollama's final accounting over our chunk count
        if "eval_count" in obj:
            STATE.total_output += obj["eval_count"] - r.output_tokens
            r.output_tokens = obj["eval_count"]
        if "prompt_eval_count" in obj:
            r.prompt_tokens = obj["prompt_eval_count"]
            STATE.total_prompt += r.prompt_tokens
    else:
        msg = obj.get("message", {})
        # newer ollama builds stream reasoning tokens in a "thinking" field;
        # older builds don't stream them at all (they land in eval_count at done)
        if (obj.get("response") or obj.get("thinking")
                or msg.get("content") or msg.get("thinking")):
            r.output_tokens += 1
            STATE.total_output += 1
            r.recent.append(time.time())


def _count_sse_line(line: bytes, r: RequestStats):
    """One SSE data: line from the OpenAI-compatible endpoints."""
    if not line.startswith(b"data:"):
        return
    payload = line[5:].strip()
    if payload == b"[DONE]":
        r.done = True
        r.ended = time.time()
        return
    try:
        obj = json.loads(payload)
    except Exception:
        return
    usage = obj.get("usage")
    if usage:
        STATE.total_output += usage.get("completion_tokens", r.output_tokens) - r.output_tokens
        r.output_tokens = usage.get("completion_tokens", r.output_tokens)
        r.prompt_tokens = usage.get("prompt_tokens", 0)
        STATE.total_prompt += r.prompt_tokens
    for choice in obj.get("choices", []):
        delta = choice.get("delta", {})
        if (delta.get("content") or delta.get("reasoning_content")
                or delta.get("reasoning") or choice.get("text")):
            r.output_tokens += 1
            STATE.total_output += 1
            r.recent.append(time.time())


def _count_final_json(body: bytes, r: RequestStats):
    """Non-streaming response: read the final token accounting."""
    try:
        obj = json.loads(body)
    except Exception:
        return
    r.done = True
    r.ended = time.time()
    if "eval_count" in obj:                      # ollama native
        r.output_tokens = obj["eval_count"]
        r.prompt_tokens = obj.get("prompt_eval_count", 0)
    elif "usage" in obj:                         # openai-compatible
        r.output_tokens = obj["usage"].get("completion_tokens", 0)
        r.prompt_tokens = obj["usage"].get("prompt_tokens", 0)
    STATE.total_output += r.output_tokens
    STATE.total_prompt += r.prompt_tokens


class _Inflator:
    """Incrementally inflate a gzip/deflate response body for token counting only.

    The bytes written back to the client are never touched (still compressed, with
    Content-Encoding intact). Counting is best-effort: if inflation fails we stop
    counting that response, the passthrough keeps working.
    """
    def __init__(self, enc: str):
        self.enc = enc
        self.obj = None
        self.pending = b""
        self.broken = False

    def feed(self, chunk: bytes) -> bytes:
        if self.broken:
            return b""
        try:
            data = self.pending + chunk
            self.pending = b""
            if self.obj is None:
                if self.enc == "gzip":
                    if len(data) < 10:
                        self.pending = data
                        return b""
                    data = data[10:]          # skip the gzip header
                    self.obj = zlib.decompressobj(-zlib.MAX_WBITS)
                elif data[:1] == b"\x78":     # zlib-wrapped deflate
                    self.obj = zlib.decompressobj()
                else:
                    self.obj = zlib.decompressobj(-zlib.MAX_WBITS)
            return self.obj.decompress(data)
        except Exception:
            self.broken = True
            return b""


def _count_response(body: bytes, r: RequestStats):
    """Whole response body: SSE lines (mislabelled streams) or one final json blob."""
    if b"\ndata:" in body or body.lstrip().startswith(b"data:"):
        for line in body.split(b"\n"):
            if line.strip():
                _count_sse_line(line.strip(), r)
        if not r.done:
            r.done, r.ended = True, time.time()
        return
    _count_final_json(body, r)

async def proxy_handler(request: web.Request, up: Upstream, session: ClientSession):
    body = await request.read()
    path = request.path
    track = request.method == "POST" and _is_stream_path(path, up.base)
    r = STATE.new_request(_extract_model(body), path) if track else None

    # drop hop-by-hop headers, and any client auth we are meant to replace
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in HOP_HEADERS
               and not (up.api_key and k.lower() in AUTH_HEADERS)}
    headers.update(_auth_headers(up.api_key, up.key_headers))

    target = _target_url(path, up)
    if request.url.query:
        target += request.url.query
    try:
        async with session.request(request.method, target, data=body or None,
                                   headers=headers) as upstream:
            ctype = upstream.headers.get("Content-Type", "")
            enc = (upstream.headers.get("Content-Encoding") or "").lower()
            # trust the request's "stream" flag too: llama.cpp / vLLM answer SSE with
            # Content-Type: application/json
            streaming = ("ndjson" in ctype or "event-stream" in ctype
                         or _body_wants_stream(body))
            resp = web.StreamResponse(status=upstream.status)
            for k, v in upstream.headers.items():
                if k.lower() not in HOP_HEADERS:
                    resp.headers[k] = v
            await resp.prepare(request)

            if streaming and r:
                counter = None
                inflator = _Inflator(enc) if enc in ("gzip", "deflate") else None
                buf = b""
                async for chunk in upstream.content.iter_any():
                    await resp.write(chunk)          # forward first, count after
                    if inflator:
                        chunk = inflator.feed(chunk)
                        if not chunk:
                            continue
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        if counter is None:  # sniff once: SSE vs NDJSON
                            counter = (_count_sse_line
                                       if line.startswith(b"data:") else _count_ndjson_line)
                        counter(line, r)
            else:
                collected = b""
                async for chunk in upstream.content.iter_any():
                    await resp.write(chunk)
                    if r:
                        collected += chunk
                if r:
                    _count_response(collected, r)

            await resp.write_eof()
            if r and not r.done:
                r.done, r.ended = True, time.time()
            return resp
    except Exception as e:
        if r:
            r.done, r.ended, r.error = True, time.time(), str(e)[:60]
        if isinstance(e, (ConnectionResetError, asyncio.CancelledError)):
            raise
        return web.json_response({"error": f"otop proxy: {e}"}, status=502)


async def start_proxy(port: int, host: str, up: Upstream):
    session = ClientSession(
        timeout=ClientTimeout(total=None, sock_connect=5),
        connector=TCPConnector(limit=100),
        auto_decompress=False,
    )
    app = web.Application(client_max_size=512 * 1024 * 1024)
    app.router.add_route(
        "*", "/{tail:.*}", lambda req: proxy_handler(req, up, session)
    )
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    try:
        await web.TCPSite(runner, host, port).start()
    except OSError as e:
        STATE.proxy_error = f"proxy failed to bind {host}:{port} ({e.strerror})"
    return runner, session

# ---------------------------------------------------------------- collectors

IOREG_RE = re.compile(
    rb'"PerformanceStatistics"\s*=\s*({.*?})', re.DOTALL
)


async def poll_gpu():
    """Apple Silicon GPU stats from ioreg (no sudo required)."""
    while True:
        try:
            proc = await asyncio.create_subprocess_exec(
                "ioreg", "-r", "-d", "1", "-c", "IOAccelerator",
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
            out, _ = await proc.communicate()
            m = IOREG_RE.search(out)
            if m:
                blob = m.group(1).decode("utf-8", "replace")
                def grab(key):
                    mm = re.search(rf'"{re.escape(key)}"=(\d+)', blob)
                    return int(mm.group(1)) if mm else None
                STATE.gpu = {
                    "device": grab("Device Utilization %"),
                    "renderer": grab("Renderer Utilization %"),
                    "tiler": grab("Tiler Utilization %"),
                    "mem_used": grab("In use system memory"),
                    "mem_alloc": grab("Alloc system memory"),
                }
        except Exception:
            STATE.gpu = {}
        await asyncio.sleep(1.0)


async def poll_procs():
    """CPU%, threads, RSS for ollama server + runner processes."""
    tracked: dict[int, psutil.Process] = {}
    while True:
        rows = []
        found = {}
        for p in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                name = p.info["name"] or ""
                cmd = " ".join(p.info["cmdline"] or [])
                if "ollama" in name.lower() or "ollama" in cmd.lower():
                    if "otop" in cmd:
                        continue
                    found[p.pid] = tracked.get(p.pid) or p
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        tracked = found
        for pid, p in tracked.items():
            try:
                cmd = " ".join(p.cmdline())
                if "llama-server" in cmd or "runner" in cmd:
                    kind = "runner"      # the process actually executing the model
                elif "serve" in cmd:
                    kind = "server"
                else:
                    kind = "app"         # menubar gui etc.
                rows.append({
                    "pid": pid,
                    "kind": kind,
                    "cpu": p.cpu_percent(interval=None),  # since last call
                    "threads": p.num_threads(),
                    "rss": p.memory_info().rss,
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        STATE.procs = sorted(rows, key=lambda r: -r["cpu"])
        STATE.cpu_percent = psutil.cpu_percent(interval=None)
        STATE.per_core = psutil.cpu_percent(interval=None, percpu=True)
        await asyncio.sleep(1.0)


async def poll_ollama(root: str, base: str, api_key: str):
    """Loaded models + version. Native ollama first, OpenAI-compatible fallback."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with ClientSession(timeout=ClientTimeout(total=3), headers=headers) as s:
        while True:
            models, version, err = None, None, ""
            try:
                async with s.get(root + "/api/ps") as resp:
                    if resp.status == 200:
                        models = (await resp.json()).get("models", [])
                    else:
                        err = f"{resp.status} for /api/ps"
            except Exception:
                err = "unreachable"
            if models is not None:
                try:
                    async with s.get(root + "/api/version") as resp:
                        if resp.status == 200:
                            version = (await resp.json()).get("version", "?")
                except Exception:
                    pass
                STATE.models = models
                STATE.ollama_ok = True
                STATE.upstream_mode = "ollama"
                STATE.ollama_version = version or STATE.ollama_version
                STATE.poll_error = ""
                await asyncio.sleep(1.5)
                continue
            # not ollama (or /api/ps missing/unauthorized): ask the OpenAI server
            if base:
                try:
                    async with s.get(root + base + "/models") as resp:
                        if resp.status == 200:
                            models = [(m.get("id") or m.get("model") or "?")
                                      for m in (await resp.json()).get("data", [])]
                        else:
                            err = f"{resp.status} for {base}/models"
                except Exception:
                    err = "unreachable"
                if models is not None:
                    STATE.models = [{"name": n} for n in models]
                    STATE.ollama_ok = True
                    STATE.upstream_mode = "openai"
                    STATE.poll_error = ""
                    await asyncio.sleep(1.5)
                    continue
            STATE.ollama_ok = False
            STATE.models = []
            STATE.upstream_mode = ""
            STATE.poll_error = err
            await asyncio.sleep(1.5)


async def poll_tps_history():
    while True:
        STATE.tps_history.append(STATE.session_live_tps())
        await asyncio.sleep(1.0)

# ---------------------------------------------------------------- rendering

def fmt_bytes(n):
    if not n:
        return "–"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} PB"


def bar(pct, width=18, color="cyan"):
    if pct is None:
        return Text("–")
    pct = max(0.0, min(100.0, pct))
    filled = int(width * pct / 100)
    t = Text()
    t.append("█" * filled, style=color)
    t.append("░" * (width - filled), style="grey35")
    t.append(f" {pct:4.0f}%")
    return t


SPARK = "▁▂▃▄▅▆▇█"


def sparkline(values, width=40):
    vals = list(values)[-width:]
    if not vals:
        return Text("")
    peak = max(max(vals), 1e-9)
    return Text("".join(SPARK[min(7, int(v / peak * 7.999))] for v in vals), style="magenta")


def render_header(proxy_port, no_proxy):
    status = Text()
    if STATE.ollama_ok:
        status.append(" ● ", style="bold green")
        mode = STATE.upstream_mode or "upstream"
        status.append(f"{mode} {STATE.ollama_version}  "
                      if STATE.ollama_version else f"{mode}  ")
    else:
        status.append(" ● ", style="bold red")
        if STATE.poll_error:
            status.append(f"{STATE.poll_error}  ", style="yellow")
        else:
            status.append("upstream unreachable  ")
    if STATE.upstream_label:
        status.append(f"{STATE.upstream_label}  ", style="dim")
    if STATE.api_key_set:
        status.append("api key ✓  ", style="green")
    if no_proxy:
        status.append("proxy off", style="dim")
    elif STATE.proxy_error:
        status.append(STATE.proxy_error, style="bold red")
    else:
        status.append(f"token proxy → 127.0.0.1:{proxy_port}", style="bold yellow")
    up = time.time() - STATE.session_start
    status.append(f"   session {int(up // 60)}m{int(up % 60):02d}s", style="dim")
    return Panel(status, title="[bold]otop – ollama monitor[/]", border_style="blue")


def render_tokens():
    t = Table.grid(expand=True)
    t.add_column(justify="left")
    live = STATE.session_live_tps()
    summary = Text()
    summary.append(f"{STATE.total_output:,}", style="bold magenta")
    summary.append(" output tok   ")
    summary.append(f"{STATE.total_prompt:,}", style="bold cyan")
    summary.append(" prompt tok   ")
    summary.append(f"{STATE.total_requests}", style="bold")
    summary.append(" requests   ")
    summary.append(f"{live:5.1f} tok/s now", style="bold green" if live else "dim")
    t.add_row(summary)
    t.add_row(sparkline(STATE.tps_history))

    reqs = Table(box=None, pad_edge=False, expand=True)
    for col, j in (("#", "right"), ("model", "left"), ("endpoint", "left"),
                   ("out tok", "right"), ("prompt", "right"),
                   ("tok/s", "right"), ("time", "right"), ("", "left")):
        reqs.add_column(col, justify=j, style="dim" if col == "#" else "")
    for r in sorted(STATE.requests.values(), key=lambda x: -x.id)[:6]:
        tps = r.avg_tps if r.done else r.live_tps
        if r.error:
            flag = Text("✗ " + r.error, style="red")
        elif r.done:
            flag = Text("done", style="dim")
        elif r.output_tokens == 0 and r.elapsed > 2.0:
            # active but nothing streamed yet: model load, prompt eval, or a
            # reasoning model thinking on a server that doesn't stream thinking
            flag = Text("● thinking/loading", style="bold yellow")
        else:
            flag = Text("● generating", style="bold green")
        reqs.add_row(
            str(r.id), r.model, r.path.replace("/api/", "").replace("/v1/", "v1/"),
            f"{r.output_tokens:,}", f"{r.prompt_tokens:,}" if r.prompt_tokens else "–",
            f"{tps:.1f}", f"{r.elapsed:5.1f}s", flag,
        )
    if not STATE.requests:
        reqs.add_row("", Text("no requests through proxy yet", style="dim italic"),
                     "", "", "", "", "", "")
    return Panel(Group(t, reqs), title="[bold magenta]tokens[/]", border_style="magenta")


def render_system():
    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column(width=10)
    grid.add_column()
    grid.add_row("cpu total", bar(STATE.cpu_percent, color="cyan"))
    cores = Text()
    for i, c in enumerate(STATE.per_core):
        style = "red" if c > 80 else "yellow" if c > 40 else "green"
        cores.append(SPARK[min(7, int(c / 100 * 7.999))], style=style)
        if i % 2:
            cores.append(" ")
    grid.add_row(f"{len(STATE.per_core)} cores", cores)
    la1, la5, la15 = psutil.getloadavg()
    grid.add_row("load", Text(f"{la1:.2f} {la5:.2f} {la15:.2f}", style="dim"))
    vm = psutil.virtual_memory()
    grid.add_row("ram", bar(vm.percent, color="blue"))
    grid.add_row("", Text(f"{fmt_bytes(vm.used)} / {fmt_bytes(vm.total)}", style="dim"))

    procs = Table(box=None, pad_edge=False, expand=True)
    for col, j in (("pid", "right"), ("proc", "left"), ("cpu%", "right"),
                   ("threads", "right"), ("rss", "right")):
        procs.add_column(col, justify=j)
    for p in STATE.procs[:4]:
        style = "bold" if p["kind"] == "runner" else ""
        procs.add_row(
            Text(str(p["pid"]), style="dim"),
            Text(f"ollama {p['kind']}", style=style),
            Text(f"{p['cpu']:.0f}", style="yellow" if p["cpu"] > 100 else ""),
            str(p["threads"]),
            fmt_bytes(p["rss"]),
        )
    if not STATE.procs:
        procs.add_row("", Text("no ollama processes found", style="dim italic"), "", "", "")
    return Panel(Group(grid, Text(), procs), title="[bold cyan]cpu / system[/]",
                 border_style="cyan")


def render_gpu():
    g = STATE.gpu
    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column(width=10)
    grid.add_column()
    grid.add_row("device", bar(g.get("device"), color="green"))
    grid.add_row("renderer", bar(g.get("renderer"), color="green"))
    grid.add_row("tiler", bar(g.get("tiler"), color="green"))
    used, alloc = g.get("mem_used"), g.get("mem_alloc")
    grid.add_row("gpu mem", Text(f"{fmt_bytes(used)} in use / {fmt_bytes(alloc)} wired",
                                 style="dim"))
    grid.add_row("", Text("unified memory – shared with system ram", style="dim italic"))
    return Panel(grid, title="[bold green]gpu (apple silicon)[/]", border_style="green")


def render_models():
    tbl = Table(box=None, pad_edge=False, expand=True)
    for col, j in (("model", "left"), ("size", "right"), ("vram", "right"),
                   ("placement", "left"), ("ctx", "right"), ("expires", "right")):
        tbl.add_column(col, justify=j)
    now = time.time()
    for m in STATE.models:
        size, vram = m.get("size") or 0, m.get("size_vram") or 0
        gpu_pct = round(vram / size * 100) if size else None
        placement = (f"{gpu_pct}% GPU" if gpu_pct else "100% CPU") if size else "–"
        expires = "–"
        try:
            from datetime import datetime
            # ollama emits rfc3339 with nanoseconds; trim to µs for fromisoformat
            raw = re.sub(r"(\.\d{6})\d+", r"\1", m["expires_at"])
            exp = datetime.fromisoformat(raw).timestamp()
            secs = max(0, exp - now)
            expires = "never" if secs > 3e8 else f"{int(secs // 60)}m{int(secs % 60):02d}s"
        except Exception:
            pass
        tbl.add_row(
            Text(m.get("name", "?"), style="bold"),
            fmt_bytes(size), fmt_bytes(vram),
            Text(placement, style="green" if (gpu_pct or 0) >= 100
                 else "dim" if gpu_pct is None else "yellow"),
            str(m.get("context_length", "–")),
            Text(expires, style="dim"),
        )
    if not STATE.models:
        tbl.add_row(Text("no models loaded", style="dim italic"), "", "", "", "", "")
    return Panel(tbl, title="[bold blue]loaded models[/]", border_style="blue")


def render(proxy_port, no_proxy):
    layout = Layout()
    layout.split_column(
        Layout(render_header(proxy_port, no_proxy), size=3),
        Layout(render_tokens(), size=12),
        Layout(name="mid"),
        Layout(render_models(), size=6),
    )
    layout["mid"].split_row(Layout(render_system()), Layout(render_gpu()))
    return layout

# ---------------------------------------------------------------- main

async def main():
    ap = argparse.ArgumentParser(description="live ollama / openai-compatible monitor")
    ap.add_argument("--upstream", default="http://127.0.0.1:11434",
                    help="upstream base url, may carry a path prefix "
                         "(e.g. http://192.168.0.23:8080/v1) (default: %(default)s)")
    ap.add_argument("--api-key",
                    default=os.environ.get("OTOP_API_KEY") or os.environ.get("OPENAI_API_KEY") or "",
                    help="api key injected into every upstream request "
                         "(default: $OTOP_API_KEY / $OPENAI_API_KEY)")
    ap.add_argument("--api-key-header", choices=("authorization", "x-api-key", "both"),
                    default="authorization",
                    help="header(s) used to send --api-key upstream (default: %(default)s)")
    ap.add_argument("--proxy-port", type=int, default=11435,
                    help="token-counting proxy port (default: %(default)s)")
    ap.add_argument("--proxy-host", default="127.0.0.1",
                    help="proxy bind address (default: %(default)s)")
    ap.add_argument("--no-proxy", action="store_true", help="disable the proxy")
    ap.add_argument("--refresh", type=float, default=4.0,
                    help="ui refreshes per second (default: %(default)s)")
    args = ap.parse_args()

    root, base = _split_upstream(args.upstream)
    key_headers = {"authorization": ("authorization",),
                   "x-api-key": ("x-api-key",),
                   "both": ("authorization", "x-api-key")}[args.api_key_header]
    up = Upstream(root, base, args.api_key, key_headers, f"{root}{base}")
    STATE.upstream_label = up.label
    STATE.api_key_set = bool(args.api_key)

    tasks = [
        asyncio.create_task(poll_gpu()),
        asyncio.create_task(poll_procs()),
        asyncio.create_task(poll_ollama(root, base, args.api_key)),
        asyncio.create_task(poll_tps_history()),
    ]
    runner = session = None
    if not args.no_proxy:
        runner, session = await start_proxy(args.proxy_port, args.proxy_host, up)

    console = Console()
    try:
        with Live(render(args.proxy_port, args.no_proxy), console=console,
                  screen=True, refresh_per_second=args.refresh) as live:
            while True:
                await asyncio.sleep(1.0 / args.refresh)
                live.update(render(args.proxy_port, args.no_proxy))
    finally:
        for t in tasks:
            t.cancel()
        if session:
            await session.close()
        if runner:
            await runner.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
