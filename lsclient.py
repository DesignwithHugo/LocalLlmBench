"""LM Studio client and macOS system probes. Standard library only."""
import json, os, re, subprocess, time, threading, urllib.request, urllib.error, shutil

HOME = os.path.expanduser("~")
LMS = os.path.join(HOME, ".lmstudio", "bin", "lms")
BASE = os.environ.get("LMSTUDIO_URL", "http://127.0.0.1:1234")
# never route localhost through a system proxy
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
TOKEN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lmstudio_token.txt")


def token():
    """LM Studio API token: env LM_API_TOKEN or ~/LocalLlmBench/lmstudio_token.txt (re-read every call)."""
    t = os.environ.get("LM_API_TOKEN", "").strip()
    if not t and os.path.exists(TOKEN_FILE):
        try:
            t = open(TOKEN_FILE).read().strip().split()[0]
        except Exception:
            t = ""
    return t


def headers():
    h = {"Content-Type": "application/json"}
    t = token()
    if t:
        h["Authorization"] = "Bearer " + t
    return h


def sh(cmd, timeout=600):
    """Run a command, return (rc, stdout+stderr). Output goes to a temp file, not a pipe,
    so daemons spawned by the command (e.g. `lms server start`) can't block us."""
    import tempfile
    with tempfile.TemporaryFile(mode="w+") as out:
        try:
            p = subprocess.Popen(cmd, shell=isinstance(cmd, str), stdout=out, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
            try:
                rc = p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(p.pid, 9)
                except Exception:
                    p.kill()
                rc = 124
            out.seek(0)
            return rc, out.read()
        except Exception as e:  # noqa
            return 1, str(e)


def lms(*args, timeout=900):
    exe = LMS if os.path.exists(LMS) else (shutil.which("lms") or LMS)
    return sh([exe, *args], timeout=timeout)


# ---------------------------------------------------------------- HTTP

def http(method, path, body=None, timeout=600):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method, headers=headers())
    try:
        with _OPENER.open(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
            return r.status, (json.loads(raw) if raw.strip().startswith(("{", "[")) else raw)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, raw
    except Exception as e:  # noqa
        return 0, str(e)


def server_status():
    st, j = http("GET", "/v1/models", timeout=10)
    return st, j


def server_up():
    return server_status()[0] == 200


def ensure_server(log):
    """Returns 'ok', 'auth' (server up but needs a valid API token) or 'down'."""
    st, j = server_status()
    if st == 200:
        return "ok"
    if st == 401:
        return "auth"
    log(f"LM Studio server not responding at {BASE} ({st}: {str(j)[:200]}), trying `lms server start`")
    rc, out = lms("server", "start", timeout=90)
    log(f"lms server start -> rc={rc} {out.strip()[-300:]}")
    for _ in range(30):
        st, _ = server_status()
        if st in (200, 401):
            return "ok" if st == 200 else "auth"
        time.sleep(4)
    sh(["open", "-ga", "LM Studio"], timeout=30)
    for _ in range(45):
        st, _ = server_status()
        if st in (200, 401):
            return "ok" if st == 200 else "auth"
        time.sleep(4)
    return "down"


# ---------------------------------------------------------------- models

def _g(d, *names, default=None):
    for n in names:
        if isinstance(d, dict) and n in d and d[n] is not None:
            return d[n]
    return default


def list_models():
    """Return list of dicts: key, type, format, size_bytes, params, arch, max_ctx, vision, tools, path, display."""
    out = []
    rc, txt = lms("ls", "--json", timeout=60)
    items = None
    if rc == 0:
        try:
            items = json.loads(txt[txt.find("["):]) if "[" in txt else None
        except Exception:
            items = None
    if items is None:
        st, j = http("GET", "/api/v1/models", timeout=30)
        if st == 200:
            items = j.get("models", j.get("data", [])) if isinstance(j, dict) else j
    for m in items or []:
        caps = _g(m, "capabilities", default={}) or {}
        out.append({
            "key": _g(m, "modelKey", "key", "id", "model_key"),
            "type": str(_g(m, "type", default="llm")).lower(),
            "format": str(_g(m, "format", "compatibility_type", default="?")).lower(),
            "size_bytes": _g(m, "sizeBytes", "size_bytes", default=0) or 0,
            "params": _g(m, "paramsString", "params_string", default=""),
            "arch": _g(m, "architecture", "arch", default=""),
            "max_ctx": _g(m, "maxContextLength", "max_context_length", default=0) or 0,
            "vision": bool(_g(m, "vision", default=_g(caps, "vision", default=False))),
            "tools": bool(_g(m, "trainedForToolUse", default=_g(caps, "trained_for_tool_use", default=False))),
            "path": _g(m, "path", default=""),
            "display": _g(m, "displayName", "display_name", default=""),
            "quant": ((_g(m, "quantization", default={}) or {}).get("name") if isinstance(_g(m, "quantization"), dict) else _g(m, "quantization", default="")) or "",
            "remote": bool(_g(m, "deviceIdentifier", default=None)),
            "raw": m,
        })
    # skip models that live on other devices linked to LM Studio, and exact duplicates
    seen, res = set(), []
    for m in out:
        if not m["key"] or m["remote"]:
            continue
        sig = (m["key"], m["format"], m["size_bytes"])
        if sig in seen:
            continue
        seen.add(sig)
        res.append(m)
    return res


def ps_instances():
    """Every loaded instance LM Studio knows about (including other Macs' through LM Link), or None if `lms ps` failed."""
    rc, txt = lms("ps", "--json", timeout=60)
    try:
        j = json.loads(txt[txt.find("["):])
        return j if isinstance(j, list) else None
    except Exception:
        return None


def is_remote(inst):
    """An instance running on another Mac through LM Link (it carries the remote device's id)."""
    return isinstance(inst, dict) and bool(inst.get("deviceIdentifier") or inst.get("device_identifier") or inst.get("deviceId"))


def loaded_instances():
    """Instances loaded on this Mac only."""
    return [i for i in (ps_instances() or []) if isinstance(i, dict) and not is_remote(i)]


def unload_all():
    """Unload this Mac's models only. `lms unload --all` would also unload models another Mac loaded through LM Link."""
    insts = ps_instances()
    if insts is None:  # can't tell what is loaded: free this Mac's memory anyway
        lms("unload", "--all", timeout=300)
        time.sleep(3)
        return
    local = [i for i in insts if isinstance(i, dict) and not is_remote(i)]
    for i in local:
        ident = i.get("identifier") or i.get("instanceReference")
        if ident:
            lms("unload", str(ident), timeout=300)
    if local and loaded_instances():  # couldn't unload them one by one
        lms("unload", "--all", timeout=300)
    time.sleep(3)


def remote_load_error(key, ident):
    """After a load: if the model ended up on another Mac through LM Link, unload it there and return an error."""
    insts = ps_instances()
    if not insts:
        return None
    mine = [i for i in insts if isinstance(i, dict) and (i.get("identifier") == ident or i.get("modelKey") == key)]
    if mine and all(is_remote(i) for i in mine):
        for i in mine:
            lms("unload", str(i.get("identifier") or i.get("instanceReference") or key), timeout=300)
        return f"loaded on a remote LM Link device ({mine[0].get('deviceIdentifier') or mine[0].get('deviceId')}), not this Mac"
    return None


def load(key, ctx, ident, gguf):
    """Load a model with a fixed context. Returns dict(ok, load_s, wall_s, ident, config, error)."""
    body = {"model": key, "context_length": int(ctx), "echo_load_config": True}
    if gguf:
        body["flash_attention"] = True
    t0 = time.time()
    st, j = http("POST", "/api/v1/models/load", body, timeout=1800)
    wall = time.time() - t0
    if st == 200 and isinstance(j, dict) and (j.get("status") == "loaded" or j.get("instance_id")):
        remote = remote_load_error(key, j.get("instance_id") or key)
        if remote:
            return {"ok": False, "error": remote, "wall_s": wall}
        return {"ok": True, "load_s": j.get("load_time_seconds", wall), "wall_s": wall,
                "ident": j.get("instance_id") or key, "config": j.get("load_config"), "via": "api"}
    api_err = j if isinstance(j, str) else json.dumps(j)[:600]
    # fallback: CLI
    unload_all()
    t0 = time.time()
    args = ["load", key, "--context-length", str(int(ctx)), "--gpu", "max", "--identifier", ident, "-y"]
    rc, txt = lms(*args, timeout=1800)
    if rc != 0 and re.search(r"unknown|unexpected", txt, re.I):
        rc, txt = lms(*args[:-1], timeout=1800)
    wall = time.time() - t0
    remote = remote_load_error(key, ident) if rc == 0 else None
    if remote:
        return {"ok": False, "error": remote, "wall_s": wall}
    if rc == 0 and any(i for i in loaded_instances()):
        return {"ok": True, "load_s": wall, "wall_s": wall, "ident": ident, "config": None,
                "via": "cli", "api_error": api_err}
    return {"ok": False, "error": (api_err + " | cli: " + txt[-600:]), "wall_s": wall}


def estimate(key, ctx):
    rc, txt = lms("load", key, "--estimate-only", "--context-length", str(int(ctx)), timeout=120)
    return txt.strip()[-400:] if rc == 0 else None


# ---------------------------------------------------------------- chat

THINK_RE = re.compile(r"<think>(.*?)(</think>|$)", re.S)


def split_think(text):
    """Return (reasoning, answer) from text that may contain <think> blocks."""
    reasoning = "".join(m.group(1) for m in THINK_RE.finditer(text))
    answer = THINK_RE.sub("", text).strip()
    return reasoning, answer


def chat_stream(model, messages, max_tokens=256, temperature=0.0, timeout=1800, extra=None):
    """Streaming chat via OpenAI-compatible endpoint with client-side timing.
    Returns dict with ttft, ttfa (first answer token), t_total, decode_tps, prompt_tokens,
    completion_tokens, reasoning_chars, answer, reasoning, finish, error."""
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "stream": True, "stream_options": {"include_usage": True}}
    if extra:
        body.update(extra)
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers=headers(), method="POST")
    t0 = time.time()
    first = first_ans = last = None
    content, reasoning, usage, finish, n_chunks = [], [], None, None, 0
    in_think = False
    try:
        with _OPENER.open(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    j = json.loads(data)
                except Exception:
                    continue
                if j.get("error"):
                    return {"error": str(j["error"])[:500]}
                if j.get("usage"):
                    usage = j["usage"]
                for ch in j.get("choices", []) or []:
                    d = ch.get("delta", {}) or {}
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
                    rc = d.get("reasoning_content") or d.get("reasoning") or ""
                    c = d.get("content") or ""
                    if not (rc or c):
                        continue
                    now = time.time()
                    n_chunks += 1
                    if first is None:
                        first = now
                    last = now
                    if rc:
                        reasoning.append(rc)
                    if c:
                        content.append(c)
                        # handle inline <think> tags
                        s = "".join(content)
                        if s.lstrip().startswith("<think>") and "</think>" not in s:
                            in_think = True
                        else:
                            if in_think and "</think>" in s:
                                in_think = False
                                tail = s.split("</think>", 1)[1]
                                if tail.strip() and first_ans is None:
                                    first_ans = now
                            elif not in_think and first_ans is None and c.strip():
                                first_ans = now
    except Exception as e:  # noqa
        return {"error": f"{type(e).__name__}: {e}"[:500], "t_total": time.time() - t0}
    t_end = time.time()
    text = "".join(content)
    r2, answer = split_think(text)
    reasoning_txt = "".join(reasoning) + r2
    pt = (usage or {}).get("prompt_tokens")
    ct = (usage or {}).get("completion_tokens") or n_chunks
    rt = ((usage or {}).get("completion_tokens_details") or {}).get("reasoning_tokens")
    if rt is None and ct and (reasoning_txt or answer):
        tot = len(reasoning_txt) + len(answer)
        rt = int(round(ct * len(reasoning_txt) / tot)) if tot else 0
    res = {
        "ttft": (first - t0) if first else None,
        "ttfa": (first_ans - t0) if first_ans else None,
        "t_total": t_end - t0,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "reasoning_tokens": rt,
        "usage_reported": usage is not None,
        "decode_tps": ((ct - 1) / (last - first)) if (first and last and last > first and ct and ct > 1) else None,
        "prefill_tps": (pt / (first - t0)) if (pt and first and first > t0) else None,
        "answer": answer,
        "reasoning": reasoning_txt,
        "finish": finish,
    }
    return res


def chat_once(model, messages, max_tokens=1024, temperature=0.0, timeout=900, tools=None, extra=None):
    """Non-streaming call (tool calling). Returns dict(message, usage, t_total, error)."""
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    if tools:
        body["tools"] = tools
    if extra:
        body.update(extra)
    t0 = time.time()
    st, j = http("POST", "/v1/chat/completions", body, timeout=timeout)
    dt = time.time() - t0
    if st != 200 or not isinstance(j, dict) or not j.get("choices"):
        return {"error": (j if isinstance(j, str) else json.dumps(j))[:500], "t_total": dt}
    msg = j["choices"][0].get("message", {})
    return {"message": msg, "usage": j.get("usage"), "t_total": dt, "finish": j["choices"][0].get("finish_reason")}


def chat_v1(model, text, system=None, max_tokens=1024, reasoning="off", timeout=900):
    """LM Studio native v1 chat (used for judges, with reasoning off). Returns (answer, stats, error)."""
    body = {"model": model, "input": text, "max_output_tokens": max_tokens, "temperature": 0,
            "store": False, "reasoning": reasoning}
    if system:
        body["system_prompt"] = system
    st, j = http("POST", "/api/v1/chat", body, timeout=timeout)
    if st != 200 or not isinstance(j, dict):
        return None, None, (j if isinstance(j, str) else json.dumps(j))[:400]
    out = j.get("output", [])
    ans = "".join(o.get("content", "") for o in out if isinstance(o, dict) and o.get("type") == "message")
    return ans, j.get("stats"), None


def embed(model, inputs, timeout=600):
    t0 = time.time()
    st, j = http("POST", "/v1/embeddings", {"model": model, "input": inputs}, timeout=timeout)
    dt = time.time() - t0
    if st != 200 or not isinstance(j, dict):
        return None, dt, (j if isinstance(j, str) else json.dumps(j))[:400]
    return [d["embedding"] for d in j.get("data", [])], dt, None


def chat_v0_stats(model, messages, max_tokens=256, draft=None, timeout=900):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": False}
    if draft:
        body["draft_model"] = draft
    st, j = http("POST", "/api/v0/chat/completions", body, timeout=timeout)
    if st != 200 or not isinstance(j, dict):
        return None, (j if isinstance(j, str) else json.dumps(j))[:400]
    return j.get("stats", {}), None


# ---------------------------------------------------------------- system probes (macOS)

def hid_idle_s():
    rc, txt = sh("ioreg -c IOHIDSystem | awk '/HIDIdleTime/ {print $NF; exit}'", timeout=20)
    try:
        return int(txt.strip().split()[-1]) / 1e9
    except Exception:
        return 1e9  # unknown: assume idle


def mem_used_gb():
    rc, txt = sh("vm_stat", timeout=20)
    if rc != 0:
        return None
    m = re.search(r"page size of (\d+)", txt)
    ps = int(m.group(1)) if m else 16384
    def pages(label):
        mm = re.search(label + r":\s+(\d+)", txt)
        return int(mm.group(1)) if mm else 0
    used = pages("Pages active") + pages("Pages wired down") + pages("Pages occupied by compressor")
    return round(used * ps / 1e9, 2)


def gpu_mem_gb():
    """Memory currently allocated by the GPU (Metal), from the AGX accelerator statistics.
    This captures model weights + KV cache more reliably than vm_stat on Apple Silicon."""
    rc, txt = sh("ioreg -r -d 1 -w 0 -c AGXAccelerator", timeout=20)
    m = re.search(r'"In use system memory"\s*=\s*(\d+)', txt)
    return round(int(m.group(1)) / 1e9, 2) if m else None


def disk_free_gb():
    try:
        st = os.statvfs(HOME)
        return round(st.f_bavail * st.f_frsize / 1e9, 1)
    except Exception:
        return None


class Guard:
    """Background watchdog: while armed, if swap grows by more than `swap_gb` or free disk drops
    below `disk_gb`, unload all models (which aborts the running request) and remember why."""
    def __init__(self, swap_gb=6.0, disk_gb=12.0, every=3.0):
        self.swap_gb, self.disk_gb, self.every = swap_gb, disk_gb, every
        self.armed, self.swap0, self.tripped = False, 0.0, None
        threading.Thread(target=self._run, daemon=True).start()
    def arm(self):
        self.swap0, self.tripped, self.armed = (swap_used_mb() or 0.0), None, True
    def disarm(self):
        self.armed = False
        return self.tripped
    def _run(self):
        while True:
            time.sleep(self.every)
            if not self.armed:
                continue
            s, d = swap_used_mb() or 0.0, disk_free_gb()
            reason = None
            if s - self.swap0 > self.swap_gb * 1000:
                reason = f"swap grew {round((s - self.swap0) / 1000, 1)} GB"
            elif d is not None and d < self.disk_gb:
                reason = f"free disk fell to {d} GB"
            if reason:
                self.tripped, self.armed = reason, False
                unload_all()


def swap_used_mb():
    rc, txt = sh("sysctl -n vm.swapusage", timeout=20)
    m = re.search(r"used = ([\d.]+)M", txt)
    return float(m.group(1)) if m else None


class MemSampler:
    """Samples used memory every `every` seconds in a thread; .peak after stop()."""
    def __init__(self, every=2.0):
        self.every, self.peak, self.gpu_peak, self._stop = every, 0.0, 0.0, threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)
    def _run(self):
        while not self._stop.is_set():
            v = mem_used_gb()
            if v and v > self.peak:
                self.peak = v
            g = gpu_mem_gb()
            if g and g > self.gpu_peak:
                self.gpu_peak = g
            self._stop.wait(self.every)
    def __enter__(self):
        self.t.start(); return self
    def __exit__(self, *a):
        self._stop.set(); self.t.join(timeout=5)


class PowerSampler:
    """Optional: uses `macmon pipe` if installed. Average system watts during the block."""
    def __init__(self):
        self.exe = shutil.which("macmon") or (os.path.exists("/opt/homebrew/bin/macmon") and "/opt/homebrew/bin/macmon")
        self.samples, self.p = [], None
    def __enter__(self):
        if self.exe:
            try:
                self.p = subprocess.Popen([self.exe, "pipe", "-i", "500"], stdout=subprocess.PIPE,
                                          stderr=subprocess.DEVNULL, text=True)
                self.t = threading.Thread(target=self._read, daemon=True); self.t.start()
            except Exception:
                self.p = None
        return self
    def _read(self):
        for line in self.p.stdout:
            try:
                j = json.loads(line)
                w = j.get("sys_power") or j.get("all_power")
                if w:
                    self.samples.append(float(w))
            except Exception:
                pass
    def __exit__(self, *a):
        if self.p:
            self.p.terminate()
    @property
    def watts(self):
        return round(sum(self.samples) / len(self.samples), 1) if self.samples else None


def environment():
    env = {}
    rc, t = sh("sw_vers -productVersion", 20); env["macos"] = t.strip()
    rc, t = sh("sysctl -n machdep.cpu.brand_string", 20); env["chip"] = t.strip()
    rc2, t2 = sh("system_profiler SPHardwareDataType | grep 'Model Name'", 30)
    if "MacBook Air" in (t2 or ""):  # label results from the Air so they never pass for Studio numbers
        env["chip"] = "MacBook Air " + env["chip"].replace("Apple ", "")
    rc, t = sh("sysctl -n hw.memsize", 20)
    try: env["ram_gb"] = round(int(t.strip()) / 2**30)
    except Exception: pass
    rc, t = sh("sysctl -n hw.model", 20); env["model"] = t.strip()
    rc, t = sh("system_profiler SPHardwareDataType | grep -i 'Total Number of Cores\\|Chip'", 60); env["hw"] = " ".join(t.split())
    rc, t = sh("sysctl -n iogpu.wired_limit_mb", 20); env["gpu_wired_limit_mb"] = t.strip()
    rc, t = lms("version", timeout=30)
    t = re.sub(r"\x1b\[[0-9;]*m", "", t)
    m = re.search(r"\d+\.\d+\.\d+[\w.+-]*", t)
    env["lms_version"] = m.group() if m else ""
    try:
        hv = json.load(open(os.path.join(HOME, ".lmstudio/.internal/historical-version-info.json")))
        env["lmstudio_app"] = hv["targetHistories"][0]["lastRecordedAppVersion"]
    except Exception:
        pass
    try:
        env["engines"] = json.load(open(os.path.join(HOME, ".lmstudio/.internal/backend-preferences-v1.json")))
    except Exception:
        pass
    return env
