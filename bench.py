#!/usr/bin/env python3
"""LocalLlmBench: unattended LM Studio benchmark for Apple Silicon.

Loads, tests and unloads every installed chat model (plus embeddings), records speed, memory,
context limits and auto-graded quality, then writes an Obsidian report and a community package.
Resumable: every finished test is appended to data/results.jsonl and never repeated.

Usage:  ./run.sh            (waits for the night window and an idle Mac)
        ./run.sh --now      (start immediately, ignore idle/night rules)
        ./run.sh --smoke    (quick check on the smallest model, then report)
        ./run.sh --report   (only rebuild the report)
"""
import argparse, base64, collections, copy, hashlib, json, os, random, re, subprocess, sys, threading, time, traceback
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lsclient as L  # noqa: E402
import tasks as T  # noqa: E402
import agent as A  # noqa: E402

DATA = os.path.join(HERE, "data")
RESULTS = os.path.join(DATA, "results.jsonl")
LOG = os.path.join(DATA, "run.log")
os.makedirs(DATA, exist_ok=True)

DEFAULTS = {
    "night_start_hour": 23, "night_end_hour": 8,
    "idle_to_start_s": 900, "idle_to_resume_s": 600, "active_if_idle_below_s": 60,
    "ctx_standard": 65536, "max_fill_cap_tokens": 131072,
    "short_reps": 5, "reps_8k": 3, "reps_32k": 2, "reps_60k": 1,
    "gen_tokens_short": 256, "gen_tokens_long": 512,
    "exclude_keywords": ["flux", "qwen-image", "ltx", "minimax-h3", "dspark", "dflash", "mmproj"],
    "priority": ["qwen3.8-27b", "qwen3.6-35b", "holo4", "lfm2.5-24b", "muse-glimmer", "nemotron", "mimo",
                 "minicpm-v-4.7-35b", "bonsai", "qwen3-vl-4b", "minicpm-v-4.7-1b", "minicpm5", "d1-3b"],
    "judges": ["qwen3.8-27b", "muse-glimmer", "lfm2.5-24b", "nemotron-3.5-bolt", "qwen3.6-35b"],
    "families": {"holo": "qwen", "underdog": "qwen", "sage": "qwen", "qwen": "qwen", "bonsai": "qwen", "mimo": "xiaomi", "lfm": "liquid", "d1-": "liquid",
                 "glimmer": "meta", "muse": "meta", "nemotron": "nvidia", "minicpm": "openbmb",
                 "gemma": "google", "gpt-oss": "openai", "mistral": "mistral", "granite": "ibm"},
    "download_quiet_s": 180, "download_wait_max_min": 240,
    # agent suite: phase 1 runs trial 0 of every task on every model, phase 2 adds the reliability trials
    "agent_trials": 3, "agent_temperature": 0.6, "agent_max_tokens": 8192, "agent_budget_s": 1800,
    "cache_reps": 3, "linger_after_done_h": 6,
    # with --now, optionally pause while the Mac is in heavy use (input in most samples of the window plus other apps' CPU)
    "heavy_use_pause": False, "heavy_window_s": 300, "heavy_sample_s": 30, "heavy_active_frac": 0.6, "heavy_cpu_pct": 150,
    # max-context fill: if set, step the context down until LM Studio's memory estimate fits under this many GB
    "dmax_mem_cap_gb": None,
}
CFG = dict(DEFAULTS)
_cfg_path = os.path.join(HERE, "config.json")
if os.path.exists(_cfg_path):
    CFG.update(json.load(open(_cfg_path)))

ARGS = None
QUIJOTE = open(os.path.join(HERE, "assets", "quijote.txt"), encoding="utf-8").read()


class Paused(Exception):
    pass


# ---------------------------------------------------------------- utils

def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    try:
        print(line, flush=True)
        with open(LOG, "a") as f:
            f.write(line + "\n")
    except OSError:  # a full disk must not stop the benchmark; it pauses itself until there is room again
        pass


def invalidations():
    """data/invalidate.json: [{"model": key or "*", "test_prefix": "", "before": ISO ts, "why": ...}].
    Matching records are ignored (append-only way to redo tests, e.g. after an engine update)."""
    try:
        return json.load(open(os.path.join(DATA, "invalidate.json")))
    except Exception:
        return []


def is_invalid(r, inv):
    for x in inv:
        if x.get("model") in ("*", r.get("model")) and str(r.get("test", "")).startswith(x.get("test_prefix", "")) \
                and str(r.get("ts", "")) < x.get("before", ""):
            return True
    return False


def load_results():
    out = []
    inv = invalidations()
    if os.path.exists(RESULTS):
        for line in open(RESULTS, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if inv and is_invalid(r, inv):
                continue
            out.append(r)
    return out


def record(rec):
    rec.setdefault("ts", datetime.now().isoformat(timespec="seconds"))
    with open(RESULTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _retryable(r):
    """Failed max-context fills that overshot the context window get one more try with the safer margin."""
    return (not r.get("ok") and r.get("test") == "dmax" and not r.get("retry_v2")
            and re.search(r"context (length|size)|exceeds", str(r.get("error", "")), re.I))


def done_set():
    return {(r.get("model"), r.get("test"), r.get("rep", 0)) for r in load_results() if r.get("final") and not _retryable(r)}


def family(key):
    k = key.lower()
    for kw, fam in CFG["families"].items():
        if kw in k:
            return fam
    return k.split("/")[0]


def prio(key):
    k = key.lower()
    for i, kw in enumerate(CFG["priority"]):
        if kw in k:
            return i
    return 100


def seed_for(*parts):
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def filler(n_chars, seed):
    start = seed % (len(QUIJOTE) - 10)
    s = QUIJOTE[start:start + n_chars]
    while len(s) < n_chars:
        s += "\n" + QUIJOTE[: n_chars - len(s)]
    return s


# ---------------------------------------------------------------- gating (night window, idle, downloads)

def in_window():
    h = datetime.now().hour
    a, b = CFG["night_start_hour"], CFG["night_end_hour"]
    return (h >= a or h < b) if a > b else (a <= h < b)


def user_active():
    return (not ARGS.now) and L.hid_idle_s() < CFG["active_if_idle_below_s"]


HEAVY = {"on": False, "seen": 0.0}
_NOT_OTHER = ("LM Studio", ".lmstudio", "lmstudio", "llmster", "bench.py", "caffeinate", "Claude", "ps -A -o")


def other_cpu_pct():
    """Summed %CPU of everything except LM Studio, this benchmark and Claude."""
    rc, txt = L.sh("ps -A -o %cpu=,args=", timeout=20)
    tot = 0.0
    for line in (txt or "").splitlines():
        parts = line.strip().split(None, 1)
        if not parts:
            continue
        try:
            c = float(parts[0])
        except ValueError:
            continue
        if not any(x in (parts[1] if len(parts) > 1 else "") for x in _NOT_OTHER):
            tot += c
    return tot


def heavy_loop():
    """Flags heavy use: input in most samples of the window and other apps above heavy_cpu_pct on average."""
    step = CFG["heavy_sample_s"]
    n = max(2, int(CFG["heavy_window_s"] / step))
    dq = collections.deque(maxlen=n)
    while True:
        try:
            dq.append((L.hid_idle_s() < CFG["active_if_idle_below_s"], other_cpu_pct()))
            act = sum(1 for a, _ in dq if a) / len(dq)
            avg = sum(c for _, c in dq) / len(dq)
            heavy = len(dq) == n and act >= CFG["heavy_active_frac"] and avg >= CFG["heavy_cpu_pct"]
            if heavy:
                HEAVY["seen"] = time.time()
            if heavy != HEAVY["on"]:
                HEAVY["on"] = heavy
                log(f"{'heavy use detected, pausing' if heavy else 'heavy use over, resuming'}: input in {act:.0%} of the last "
                    f"{CFG['heavy_window_s'] // 60} min, other apps' CPU {avg:.0f}%")
        except Exception as e:  # noqa
            log(f"heavy-use monitor error: {e}")
        time.sleep(step)


def check_pause():
    """Called between tests. Raises Paused if the user is back or the window closed."""
    if ARGS.now:
        if CFG.get("heavy_use_pause") and HEAVY["on"]:
            raise Paused("heavy use of the Mac")
        return
    if not in_window():
        raise Paused("night window closed")
    if user_active():
        raise Paused("Mac in use")


def wait_slot(first=False):
    if ARGS.now:
        last_note = 0
        while CFG.get("heavy_use_pause") and HEAVY["on"]:
            if time.time() - last_note > 1800:
                log("waiting: heavy use of the Mac")
                last_note = time.time()
            time.sleep(60)
        return
    need = CFG["idle_to_start_s"] if first else CFG["idle_to_resume_s"]
    last_note = 0
    while True:
        idle = L.hid_idle_s()
        if in_window() and idle >= need:
            return
        if time.time() - last_note > 1800:
            log(f"waiting: in_window={in_window()} idle={int(idle)}s (need {need}s)")
            last_note = time.time()
        time.sleep(60)


def active_downloads():
    root = os.path.join(L.HOME, ".lmstudio", "models")
    now, act = time.time(), []
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.endswith(".part"):
                p = os.path.join(dp, f)
                if now - os.path.getmtime(p) < CFG["download_quiet_s"]:
                    act.append(os.path.relpath(p, root))
    return act


def wait_downloads():
    t0 = time.time()
    while True:
        act = active_downloads()
        if not act:
            return True
        if (time.time() - t0) / 60 > CFG["download_wait_max_min"]:
            log(f"downloads still active after {CFG['download_wait_max_min']} min, continuing anyway: {act}")
            return False
        log(f"waiting for downloads to finish: {act}")
        time.sleep(300)


# ---------------------------------------------------------------- model plan

QBITS = {"iq1": 1.8, "iq2": 2.6, "q2_k": 2.9, "iq3": 3.4, "q3_k_s": 3.5, "q3_k_m": 3.9, "q3_k_l": 4.3, "iq4_xs": 4.3, "q4_0": 4.5,
         "q4_k_s": 4.6, "q4_k_m": 4.85, "q4_k_xl": 4.9, "q5_k_s": 5.5, "q5_k_m": 5.7, "q6_k": 6.56, "q8_0": 8.5, "bf16": 16, "f16": 16,
         "3bit": 3.5, "4bit": 4.5, "5bit": 5.5, "6bit": 6.5, "8bit": 8.5}


def _qbits(tag):
    t = str(tag or "").lower().replace("-", "_")
    return next((v for k, v in sorted(QBITS.items(), key=lambda kv: -len(kv[0])) if k in t), None)


def _alnum(x):
    return re.sub(r"[^a-z0-9]", "", str(x or "").lower())


def variant_files(m, tag):
    """(bytes, paths relative to the models folder) for one quant of a multi-variant model; (0, []) if not fully downloaded."""
    root = os.path.expanduser("~/.lmstudio/models")
    stem = _alnum(m.get("display") or m["key"].split("/")[-1].split("@")[0])
    t = _alnum(tag)
    total, paths = 0, []
    for dp, ds, fs in os.walk(root):
        if t in _alnum(os.path.basename(dp)) and stem in _alnum(os.path.basename(dp)) and any(f.endswith(".safetensors") for f in fs):
            return sum(os.path.getsize(os.path.join(dp, f)) for f in fs), [os.path.relpath(dp, root)]
        for f in sorted(fs):
            n = _alnum(f)
            if f.endswith(".gguf") and "mmproj" not in f.lower() and stem in n and t in n:
                total += os.path.getsize(os.path.join(dp, f))
                paths.append(os.path.relpath(os.path.join(dp, f), root))
    return total, paths


def alias_keys(models):
    """LM Studio renames a model to 'name@quant' once a second quant of it is downloaded. Keep the old key for the quant
    that was already tested, so its results still count and it isn't tested twice."""
    metas = {}
    for r in load_results():
        if r.get("type") == "meta":
            metas[r.get("model")] = str((r.get("meta") or {}).get("quant") or "").lower().replace("-", "_")
    listed = {m["key"] for m in models}
    for m in models:
        k = m["key"]
        if "@" not in k or k in metas:
            continue
        base, tag = k.split("@", 1)
        for o, q in metas.items():
            if o not in listed and o.split("/")[-1] == base.split("/")[-1] and q == tag.lower().replace("-", "_"):
                m["load_key"], m["key"] = m.get("load_key") or k, o
                listed.add(o)
                break
    return models


def expand_variants(models):
    """LM Studio can group several quants under one model key ('flint' with variants q4_k_m, q6_k, q8_0). Test each
    downloaded quant as its own model. The quant tested first keeps the plain key, so its results still count."""
    tested = {}
    for r in load_results():
        if r.get("type") == "meta":
            tested[r.get("model")] = str((r.get("meta") or {}).get("quant") or "").lower().replace("-", "_")
    out = []
    for m in models:
        raw = m.get("raw") or {}
        vs = raw.get("variants") or []
        if len(vs) <= 1:
            q = str(m.get("quant") or "").lower().replace("-", "_")
            t = tested.get(m["key"])
            if t and q and t != q:
                # the file behind this key is a different quant from the one tested under it (the tested quant was deleted
                # and another downloaded, or a variant group collapsed): use 'key@quant' so the two quants' results stay apart
                m = dict(m, key=f'{m["key"]}@{q}', load_key=m.get("load_key") or m["key"])
            out.append(m)
            continue
        sel = raw.get("selectedVariant") or vs[0]
        base = tested.get(m["key"]) or sel.split("@")[-1].lower().replace("-", "_")
        entries, matched = [], False
        for v in vs:
            tag = v.split("@")[-1]
            size, paths = variant_files(m, tag)
            # identifiers to try when loading: the variant id, the file path, and the plain key when it points at this variant
            alts = [v] + paths[:1] + ([m["key"]] if v == sel else [])
            if tag.lower().replace("-", "_") == base:
                entries.append(dict(m, load_key=alts[0], alt_keys=alts[1:])); matched = True
            elif size:  # skip variants that are listed but not fully on disk
                entries.append(dict(m, key=v, load_key=alts[0], alt_keys=alts[1:], quant=tag.upper(), size_bytes=size))
        if not matched:  # the tested quant goes by another name: keep the plain key on the selected variant
            entries = [dict(m, load_key=m["key"])] + [e for e in entries if e["load_key"] != sel]
        out += entries
    return out


def eligible(models):
    out = []
    for m in models:
        k = m["key"].lower() + " " + (m.get("path") or "").lower()
        if any(x in k for x in CFG["exclude_keywords"]) or str(m.get("arch", "")).lower() == "clip":
            continue
        if ARGS.only and ARGS.only.lower() not in m["key"].lower():
            continue
        out.append(m)
    return out


def native_ctx(m):
    """Reported max context; values under 16k are treated as missing metadata (probe instead)."""
    mx = m.get("max_ctx") or 0
    return mx if mx >= 16384 else 0


def std_ctx(m):
    mx = native_ctx(m)
    return min(CFG["ctx_standard"], mx) if mx else CFG["ctx_standard"]


def plan(m):
    """Ordered list of (test, rep)."""
    if m["type"].startswith("embed"):
        return [("emb_load", 0), ("emb_speed", 0), ("emb_retrieval", 0)]
    p = [("load_std", 0), ("calib", 0)]
    p += [("short", r) for r in range(CFG["short_reps"] + 1)]  # rep 0 = warmup
    p += [("d8k", r) for r in range(CFG["reps_8k"])]
    if std_ctx(m) >= 40000:
        p += [("d32k", r) for r in range(CFG["reps_32k"])]
    if std_ctx(m) >= 65536:
        p += [("d60k", r) for r in range(CFG["reps_60k"])]
    if m.get("vision") or "vl" in m["key"].lower() or "minicpm-v" in m["key"].lower():
        p += [("vis_encode", 0)]
    for t in T.tasks():
        if t["kind"] == "vision" and not (m.get("vision") or "vl" in m["key"].lower() or "minicpm-v" in m["key"].lower()):
            continue
        p.append(("q:" + t["id"], 0))
    p += [("load_repeat", 0), ("load_max", 0), ("dmax", 0)]
    p += [("ag_cache", r) for r in range(CFG["cache_reps"])]
    for t in A.TASKS:
        p += [("ag:" + t["id"], r) for r in range(CFG["agent_trials"])]
    for t in A.FIELD_TASKS:  # real-cases tier, scored separately
        p += [("fx:" + t["id"], r) for r in range(CFG["agent_trials"])]
    if ARGS.smoke:
        keep = {"load_std", "calib", "short", "d8k", "q:if_json_only", "q:tool_weather_es", "q:code_contrast",
                "q:write_email_es", "vis_encode", "q:vis_ui_es", "load_repeat", "ag_cache", "ag:ag_room_recovery",
                "ag:ag_big_toolset", "ag:ag_code_fix", "fx:fx_shared_log"}
        p = [(t, r) for t, r in p if t in keep and r <= (0 if t.startswith(("ag", "fx")) else 2)]
    return p


def phase_of(test, rep):
    """0 = base speed/quality tests, 1 = agent suite first pass + prefix cache, 2 = agent reliability trials,
    3 = real-cases tier first pass, 4 = real-cases reliability trials."""
    if test == "ag_cache":
        return 1
    if test.startswith("ag:"):
        return 1 if rep == 0 else 2
    if test.startswith("fx:"):
        return 3 if rep == 0 else 4
    return 0


def pending(m, done, max_phase=4):
    return [(t, r) for t, r in plan(m) if phase_of(t, r) <= max_phase and (m["key"], t, r) not in done]


# ---------------------------------------------------------------- running a model

class Ctx:
    def __init__(self, m):
        self.m, self.key = m, m["key"]
        self.load_key = m.get("load_key") or m["key"]  # a specific quant when one key groups several
        self.alt_keys = list(m.get("alt_keys") or [])
        self.gguf = "gguf" in (m.get("format") or "") or (m.get("path") or "").lower().endswith(".gguf")
        self.ident, self.ctx = None, None
        self.ratio, self.overhead = 3.6, 30

    def ensure(self, ctx):
        if self.ident and self.ctx == ctx:
            return True
        L.unload_all()
        r = load_m(self, ctx, "bench-" + re.sub(r"[^a-z0-9]+", "-", self.key.lower())[-40:], self.gguf)
        if r["ok"]:
            self.ident, self.ctx = r["ident"], ctx
            return True
        self.ident = None
        return False


def load_m(c, ctx, ident, gguf):
    """Load c's model. A quant variant may only be loadable by its file path, so try the alternatives on 'not found'."""
    r = L.load(c.load_key, ctx, ident, gguf)
    if r["ok"] or not re.search(r"not found|no model found", str(r.get("error", "")), re.I):
        return r
    for k in list(c.alt_keys):
        r2 = L.load(k, ctx, ident, gguf)
        if r2["ok"]:
            log(f"    loaded as {k} ({c.load_key} wasn't accepted)")
            c.alt_keys = [x for x in [c.load_key] + c.alt_keys if x != k]
            c.load_key = k
            return r2
        r["error"] = str(r.get("error", ""))[:300] + f" | as {k}: " + str(r2.get("error", ""))[:300]
    return r


def env_short():
    return {"lmstudio": ENV.get("lmstudio_app"), "engines": ENV.get("engines")}


def finish(m, test, rep, data, ok=True, final=True):
    rec = {"model": m["key"], "test": test, "rep": rep, "ok": ok, "final": final, **data}
    record(rec)
    return rec


def timed(fn):
    i0 = L.hid_idle_s()
    t0 = time.time()
    res = fn()
    dur = time.time() - t0
    i1 = L.hid_idle_s()
    disturbed = (not ARGS.now) and i1 < dur - 5
    return res, disturbed, i0


DISTURBED = {}
GUARD = None
CURRENT_CTX_RESET = [False]
LOW_DISK_GB = 20


def wait_for_disk():
    """Don't start tests while the disk is nearly full (swap needs room)."""
    while True:
        d = L.disk_free_gb()
        if d is None or d >= LOW_DISK_GB:
            return
        msg = f"Paused: only {d} GB free on disk. Free some space; the benchmark resumes on its own above {LOW_DISK_GB} GB."
        log(msg)
        import report as _r
        _r.NOTE = msg
        try:
            _r.build()
        except Exception:
            pass
        time.sleep(300)


def run_with_retry(m, test, rep, fn, max_disturbed=2):
    """fn() -> (ok, data). If the user touched the Mac during the test, pause and redo it later
    (up to max_disturbed times, then keep the result with a 'disturbed' flag)."""
    check_pause()
    wait_for_disk()
    if GUARD:
        GUARD.arm()
    (ok, data), disturbed, _ = timed(fn)
    tripped = GUARD.disarm() if GUARD else None
    if tripped:
        log(f"  {test}#{rep} aborted by safety guard: {tripped}")
        data = {"error": f"aborted by safety guard: {tripped}", "guard": tripped, "ctx": data.get("ctx")}
        ok = False
        CURRENT_CTX_RESET[0] = True
    k = (m["key"], test, rep)
    if disturbed and DISTURBED.get(k, 0) < max_disturbed:
        DISTURBED[k] = DISTURBED.get(k, 0) + 1
        log(f"  {test}#{rep} disturbed by user activity, will redo later")
        raise Paused("Mac in use during test")
    data["disturbed"] = disturbed
    return finish(m, test, rep, data, ok=ok)


def speed_prompt(kind, c, rep, target_tokens=None):
    nonce = f"[ref {seed_for(c.key, kind, rep, time.time()) % 10**8}]\n"
    if kind == "short":
        body = filler(1600, seed_for(c.key, kind, rep))
        text = (nonce + "Contexto de referencia:\n" + body + "\n\nInstrucción: Escribe un ensayo largo y muy detallado "
                "(al menos 1500 palabras) sobre la historia del diseño gráfico en México.")
        return [{"role": "user", "content": text}], None
    code = str(100000 + seed_for(c.key, kind, rep) % 900000)
    n_chars = int(max(500, target_tokens - c.overhead - 120) * c.ratio)
    body = filler(n_chars, seed_for(c.key, kind, rep, "f"))
    cut = len(body) // 2
    needle = f"\n\nDato importante: la contraseña del archivo de Rocinante es {code}.\n\n"
    text = (nonce + body[:cut] + needle + body[cut:] + "\n\nPregunta: según el texto anterior, ¿cuál es la contraseña del "
            "archivo de Rocinante? Responde primero con el número y después escribe un resumen detallado del texto.")
    return [{"role": "user", "content": text}], code


def needle_score(res, code):
    """1.0 if the model surfaced the hidden code anywhere (answer or reasoning). Thinking models often
    find it but spend the whole token cap reasoning, which is a length issue, not a recall failure."""
    if code in (res.get("answer") or "") or code in (res.get("reasoning") or ""):
        return 1.0
    return 0.0


def slim(res, keep_answer=False):
    d = {k: v for k, v in res.items() if k not in ("answer", "reasoning")}
    d["reasoning_chars"] = len(res.get("reasoning") or "")
    if keep_answer:
        d["answer"] = res.get("answer")
        d["reasoning_head"] = (res.get("reasoning") or "")[:1500]
    return d


def run_model(m, max_phase=2):
    c = Ctx(m)
    done = done_set()
    todo = pending(m, done, max_phase)
    if not todo:
        return
    log(f"== {m['key']}  ({m.get('format')}, {round((m.get('size_bytes') or 0)/1e9,1)} GB, max ctx {m.get('max_ctx')}) {len(todo)} tests left")
    record({"type": "meta", "model": m["key"], "test": "meta", "final": False,
            "meta": {k: v for k, v in m.items() if k != "raw"}, "family": family(m["key"])})
    if m["type"].startswith("embed"):
        return run_embedding(m, c, todo)
    ctx = std_ctx(m)
    tasks = {t["id"]: t for t in T.tasks()}
    calib_rec = next((r for r in load_results() if r.get("model") == c.key and r.get("test") == "calib" and r.get("ok")), None)
    if calib_rec:
        c.ratio, c.overhead = calib_rec["ratio"], calib_rec["overhead"]
    std_rec = next((r for r in load_results() if r.get("model") == c.key and r.get("test") == "load_std" and r.get("ok")), None)
    if std_rec:
        ctx = std_rec.get("ctx", ctx)

    for test, rep in todo:
        check_pause()
        if time.time() - _QUEUE_T[0] > 300:  # keep starting queued downloads while a long model run is in progress
            _QUEUE_T[0] = time.time()
            try:
                process_download_queue()
            except Exception as e:  # noqa
                log(f"download queue error: {e}")
        if CURRENT_CTX_RESET[0]:
            c.ident, c.ctx, CURRENT_CTX_RESET[0] = None, None, False
        log(f"  {test} #{rep}")
        try:
            if test == "load_std":
                def f():
                    L.unload_all(); time.sleep(5)
                    base, swap0, gbase = L.mem_used_gb(), L.swap_used_mb(), L.gpu_mem_gb()
                    tried = []
                    for cx in sorted({ctx, 32768, 16384, 8192}, reverse=True):
                        if cx > ctx:
                            continue
                        r = load_m(c, cx, "bench-" + re.sub(r"[^a-z0-9]+", "-", c.key.lower())[-40:], c.gguf)
                        tried.append({"ctx": cx, "ok": r["ok"], "error": r.get("error", "")[:300]})
                        if r["ok"]:
                            time.sleep(3)
                            c.ident, c.ctx = r["ident"], cx
                            after, gafter = L.mem_used_gb(), L.gpu_mem_gb()
                            return True, {"ctx": cx, "load_s": r["load_s"], "wall_s": r["wall_s"], "via": r["via"],
                                          "mem_base_gb": base, "mem_after_gb": after,
                                          "mem_model_gb": round(after - base, 2) if (after and base) else None,
                                          "gpu_base_gb": gbase, "gpu_after_gb": gafter,
                                          "gpu_model_gb": round(gafter - (gbase or 0), 2) if gafter else None,
                                          "swap_delta_mb": (L.swap_used_mb() or 0) - (swap0 or 0),
                                          "estimate": L.estimate(c.load_key, cx), "load_config": r.get("config"),
                                          "tried": tried, "env": env_short()}
                    return False, {"tried": tried}
                rec = run_with_retry(m, test, rep, f)
                if rec and not rec["ok"]:
                    log("  could not load at any context, skipping model")
                    for t2, r2 in pending(m, done_set()):
                        finish(m, t2, r2, {"skipped": "model failed to load"}, ok=False)
                    return
                ctx = rec["ctx"] if rec else ctx
                continue

            if not c.ensure(ctx):
                raise RuntimeError("load failed on resume")

            if test == "calib":
                def f():
                    a = L.chat_stream(c.ident, [{"role": "user", "content": "Hola"}], max_tokens=1)
                    txt = filler(8000, 12345)
                    b = L.chat_stream(c.ident, [{"role": "user", "content": txt}], max_tokens=1)
                    if a.get("error") or b.get("error") or not b.get("prompt_tokens"):
                        return False, {"error": a.get("error") or b.get("error") or "no usage reported"}
                    inst = L.loaded_instances()
                    try:  # diagnostics: what `lms ps --json` looks like with this Mac's model loaded (LM Link checks)
                        open(os.path.join(DATA, "lms_ps_loaded.json"), "w").write(L.lms("ps", "--json", timeout=60)[1])
                    except Exception:
                        pass
                    c.overhead = max(1, (a["prompt_tokens"] or 10) - 2)
                    c.ratio = 8000 / max(1, b["prompt_tokens"] - c.overhead)
                    return True, {"ratio": round(c.ratio, 3), "overhead": c.overhead, "instances": len(inst) if isinstance(inst, list) else None}
                rec = run_with_retry(m, test, rep, f)
                if rec and isinstance(rec.get("instances"), int) and rec["instances"] > 1:
                    log("  WARNING: more than one instance loaded (JIT?). Results may be off.")
                continue

            if test in ("short", "d8k", "d32k", "d60k"):
                target = {"d8k": 8192, "d32k": 32768, "d60k": 60000}.get(test)
                gen = CFG["gen_tokens_short"] if test == "short" else CFG["gen_tokens_long"]
                def f(test=test, rep=rep, target=target, gen=gen):
                    msgs, code = speed_prompt(test, c, rep, target)
                    with L.MemSampler() as ms, L.PowerSampler() as pw:
                        res = L.chat_stream(c.ident, msgs, max_tokens=gen, timeout=3600)
                    if res.get("error"):
                        return False, {"error": res["error"]}
                    d = slim(res)
                    d.update({"mem_peak_gb": ms.peak, "gpu_peak_gb": ms.gpu_peak or None, "watts": pw.watts, "ctx": c.ctx, "target_tokens": target,
                              "warmup": test == "short" and rep == 0, "gen_limit": gen,
                              "hit_limit": (res.get("completion_tokens") or 0) >= gen - 1})
                    if code:
                        d["needle"] = needle_score(res, code)
                    if pw.watts and res.get("decode_tps"):
                        d["tokens_per_joule"] = round(res["decode_tps"] / pw.watts, 3)
                    return True, d
                run_with_retry(m, test, rep, f)
                continue

            if test == "vis_encode":
                def f():
                    img = base64.b64encode(open(os.path.join(HERE, "assets", "images", "ui_es.png"), "rb").read()).decode()
                    q = "Describe brevemente esta pantalla."
                    t_txt, t_img = [], []
                    for i in range(3):
                        a = L.chat_stream(c.ident, [{"role": "user", "content": f"[{i}{time.time()}] {q}"}], max_tokens=16)
                        b = L.chat_stream(c.ident, [{"role": "user", "content": [
                            {"type": "text", "text": f"[{i}{time.time()}] {q}"},
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}}]}], max_tokens=16)
                        if b.get("error"):
                            return False, {"error": b["error"]}
                        t_txt.append(a.get("ttft") or 0); t_img.append(b.get("ttft") or 0)
                    med = lambda xs: sorted(xs)[len(xs) // 2]
                    return True, {"ttft_text": med(t_txt), "ttft_image": med(t_img), "image_overhead_s": round(med(t_img) - med(t_txt), 3)}
                run_with_retry(m, test, rep, f)
                continue

            if test.startswith("q:"):
                t = tasks[test[2:]]
                def f(t=t):
                    msgs_content = t["prompt"]
                    if t["kind"] == "vision":
                        img = base64.b64encode(open(os.path.join(HERE, "assets", "images", t["image"]), "rb").read()).decode()
                        msgs_content = [{"type": "text", "text": t["prompt"]},
                                        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}}]
                    msgs = [{"role": "user", "content": msgs_content}]
                    if t["kind"] == "tool":
                        res = L.chat_once(c.ident, msgs, max_tokens=t["max_tokens"], tools=T.TOOLS)
                        if res.get("error"):
                            return True, {"cat": t["cat"], "score": 0.0, "note": "error: " + res["error"][:200], "t_total": res.get("t_total")}
                        score, note = t["grader"](res["message"])
                        msg = res["message"]
                        return True, {"cat": t["cat"], "score": round(score, 3), "note": note, "t_total": res["t_total"],
                                      "answer": (msg.get("content") or "")[:3000],
                                      "tool_calls": msg.get("tool_calls"), "usage": res.get("usage")}
                    res = L.chat_stream(c.ident, msgs, max_tokens=t["max_tokens"], temperature=t["temperature"], timeout=3600)
                    if res.get("error"):
                        return True, {"cat": t["cat"], "score": 0.0, "note": "error: " + res["error"][:200]}
                    try:
                        score, note = t["grader"](res.get("answer") or "")
                    except Exception as e:  # noqa
                        score, note = 0.0, f"grader error {e}"
                    d = slim(res, keep_answer=True)
                    d.update({"cat": t["cat"], "score": round(score, 3), "note": note, "judged": bool(t.get("judge"))})
                    return True, d
                run_with_retry(m, test, rep, f)
                continue

            if test == "ag_cache":
                def f(rep=rep):
                    # an agent-framework sized prefix (system prompt + 40 tools + project notes, ~12k tokens);
                    # cold = new prefix, warm = same prefix + different question, cont = same chat one turn later
                    nonce = f"[session {seed_for(c.key, 'cache', rep, time.time()) % 10**8}]"
                    sysmsg = nonce + "\n" + A.BIG_SYSTEM + "\n\n# Project knowledge base\n" + filler(30000, seed_for(c.key, "kb", rep))
                    tools = copy.deepcopy(A.task_tools(A.TASK_BY_ID["ag_big_toolset"]))
                    tools[0]["function"]["description"] = nonce + " " + tools[0]["function"]["description"]
                    base, ex = [{"role": "system", "content": sysmsg}], {"tools": tools}
                    q1 = base + [{"role": "user", "content": "Reply with just the word OK."}]
                    a = L.chat_stream(c.ident, q1, max_tokens=16, extra=ex, timeout=1800)
                    if a.get("error"):
                        return False, {"error": a["error"]}
                    b = L.chat_stream(c.ident, base + [{"role": "user", "content": "Reply with just the word READY."}], max_tokens=16, extra=ex, timeout=1800)
                    d = L.chat_stream(c.ident, q1 + [{"role": "assistant", "content": "OK"}, {"role": "user", "content": "Thanks. Now reply with just the word DONE."}],
                                      max_tokens=16, extra=ex, timeout=1800)
                    t_ = lambda x: x.get("ttft") or x.get("t_total")
                    return True, {"prompt_tokens": a.get("prompt_tokens"), "cold_ttft": t_(a), "warm_ttft": t_(b), "cont_ttft": t_(d),
                                  "speedup": round(t_(a) / t_(b), 2) if t_(a) and t_(b) else None, "ctx": c.ctx,
                                  "errors": [x.get("error") for x in (b, d) if x.get("error")]}
                run_with_retry(m, test, rep, f)
                continue

            if test.startswith(("ag:", "fx:")):
                t = A.TASK_BY_ID.get(test[3:])
                base_rec = {"cat": t["cat"] if t else None, "score": 0.0, "solved": False, "grader_v": A.version(t) if t else None}
                if not t:
                    finish(m, test, rep, dict(base_rec, skipped="task no longer exists"), ok=True); continue
                if t.get("min_ctx") and (c.ctx or ctx) < t["min_ctx"]:
                    finish(m, test, rep, dict(base_rec, skipped=f"needs {t['min_ctx']} context, model runs at {c.ctx or ctx}"), ok=True); continue
                if rep > 0:
                    first = [r for r in load_results() if r.get("model") == c.key and r.get("test") == test and r.get("rep") == 0 and r.get("final")]
                    if first and first[-1].get("stop") == "timeout":
                        finish(m, test, rep, dict(base_rec, skipped="timed out in trial 0", stop="timeout"), ok=True); continue
                def f(t=t, rep=rep):
                    def chat(msgs, tools, max_tokens, temperature, seed, timeout):
                        return L.chat_once(c.ident, msgs, max_tokens=max_tokens, temperature=temperature, timeout=timeout,
                                           tools=tools, extra={"seed": seed})
                    r = A.run_trial(t, chat, temperature=CFG["agent_temperature"], seed=1000 * (rep + 1),
                                    max_tokens=CFG["agent_max_tokens"], budget_s=CFG["agent_budget_s"])
                    r["ctx"] = c.ctx
                    return True, r
                rec = run_with_retry(m, test, rep, f)
                if rec:
                    log(f"    score {rec.get('score')} {rec.get('stop')} {rec.get('t_total')}s steps={rec.get('steps')} calls={rec.get('tool_calls')} "
                        f"invalid={(rec.get('bad_json') or 0) + (rec.get('schema_errors') or 0) + (rec.get('unknown_tool') or 0)} | {str(rec.get('note'))[:80]}"
                        + (f" | ERROR {str(rec.get('error'))[:200]}" if rec.get("error") else ""))
                continue

            if test == "load_repeat":
                def f():
                    L.unload_all(); c.ident = None; time.sleep(5)
                    r = load_m(c, ctx, "bench-" + re.sub(r"[^a-z0-9]+", "-", c.key.lower())[-40:], c.gguf)
                    if r["ok"]:
                        c.ident, c.ctx = r["ident"], ctx
                    return r["ok"], {"ctx": ctx, "load_s": r.get("load_s"), "wall_s": r.get("wall_s")}
                run_with_retry(m, test, rep, f)
                continue

            if test == "load_max":
                def f():
                    mx = native_ctx(m) or 262144
                    if mx <= ctx:
                        return True, {"ctx": ctx, "same_as_standard": True, "native_max": mx}
                    cands = [mx] + [x for x in (1048576, 524288, 262144, 196608, 131072, 98304) if ctx < x < mx]
                    L.unload_all(); c.ident = None; time.sleep(5)
                    base, gbase = L.mem_used_gb(), L.gpu_mem_gb()
                    tried = []
                    for cx in cands:
                        r = load_m(c, cx, "bench-max", c.gguf)
                        tried.append({"ctx": cx, "ok": r["ok"], "error": (r.get("error") or "")[:200]})
                        if r["ok"]:
                            time.sleep(3)
                            c.ident, c.ctx = r["ident"], cx
                            after, gafter = L.mem_used_gb(), L.gpu_mem_gb()
                            return True, {"ctx": cx, "native_max": mx, "load_s": r.get("load_s"), "mem_base_gb": base,
                                          "mem_after_gb": after, "mem_model_gb": round(after - base, 2) if (after and base) else None,
                                          "gpu_model_gb": round(gafter - (gbase or 0), 2) if gafter else None,
                                          "tried": tried, "estimate": L.estimate(c.load_key, cx)}
                        L.unload_all()
                    return True, {"ctx": ctx, "native_max": mx, "tried": tried, "note": "no larger context could be loaded"}
                rec = run_with_retry(m, test, rep, f)
                continue

            if test == "dmax":
                lm = next((r for r in load_results() if r.get("model") == c.key and r.get("test") == "load_max" and r.get("final")), None)
                mx_ctx = (lm or {}).get("ctx") or ctx
                cap_gb = CFG.get("dmax_mem_cap_gb")
                if cap_gb and mx_ctx > ctx:
                    def _est_gb(cx):
                        t = L.estimate(c.load_key, cx) or ""
                        mm = re.search(r"Estimated Total Memory:\s*([\d.]+)\s*GiB", t)
                        return float(mm.group(1)) * 1.0737 if mm else None
                    for cx in [mx_ctx] + [x for x in (524288, 262144, 196608, 131072, 98304) if ctx < x < mx_ctx]:
                        g = _est_gb(cx)
                        if g is not None and g <= cap_gb:
                            if cx != mx_ctx:
                                log(f"  dmax: {mx_ctx // 1024}k needs about {_est_gb(mx_ctx)} GB, over the {cap_gb} GB cap; filling at {cx // 1024}k")
                            mx_ctx = cx
                            break
                    else:
                        finish(m, test, rep, {"skipped": f"no context above standard fits under {cap_gb} GB"}, ok=True)
                        continue
                # tokens-per-char calibration can be off by a few %, so leave a 10% margin
                fill = min(int(mx_ctx * 0.9) - CFG["gen_tokens_long"], CFG["max_fill_cap_tokens"])
                if mx_ctx <= ctx or fill <= 64000:
                    finish(m, test, rep, {"skipped": "max context not larger than standard"}, ok=True)
                    continue
                if not c.ensure(mx_ctx):
                    finish(m, test, rep, {"error": "could not reload at max context"}, ok=False)
                    continue
                def f(fill=fill, mx_ctx=mx_ctx):
                    msgs, code = speed_prompt("dmax", c, rep, fill)
                    swap0 = L.swap_used_mb()
                    with L.MemSampler() as ms:
                        res = L.chat_stream(c.ident, msgs, max_tokens=CFG["gen_tokens_long"], timeout=7200)
                    if res.get("error"):
                        return False, {"error": res["error"], "ctx": mx_ctx, "target_tokens": fill, "retry_v2": True}
                    d = slim(res)
                    d.update({"ctx": mx_ctx, "target_tokens": fill, "needle": needle_score(res, code),
                              "mem_peak_gb": ms.peak, "gpu_peak_gb": ms.gpu_peak or None, "swap_delta_mb": (L.swap_used_mb() or 0) - (swap0 or 0)})
                    return True, d
                run_with_retry(m, test, rep, f)
                continue
        except Paused:
            raise
        except Exception as e:  # noqa
            log(f"  ERROR in {test}#{rep}: {e}\n{traceback.format_exc()[-800:]}")
            finish(m, test, rep, {"error": str(e)[:400]}, ok=False)
    L.unload_all()


def run_embedding(m, c, todo):
    paras = [p.strip().replace("\n", " ") for p in QUIJOTE.split("\n\n") if 300 < len(p) < 1500][:400]
    is_gemma = "gemma" in c.key.lower()
    doc = (lambda t: "title: none | text: " + t) if is_gemma else (lambda t: t)
    qry = (lambda t: "task: search result | query: " + t) if is_gemma else (lambda t: t)
    for test, rep in todo:
        check_pause()
        log(f"  {test}")
        if test == "emb_load":
            def f():
                L.unload_all(); base, gbase = L.mem_used_gb(), L.gpu_mem_gb()
                r = load_m(c, 2048, "bench-emb", False)
                if r["ok"]:
                    c.ident = r["ident"]
                after, gafter = L.mem_used_gb(), L.gpu_mem_gb()
                return r["ok"], {"load_s": r.get("load_s"), "mem_model_gb": round(after - base, 2) if (after and base) else None,
                                 "gpu_model_gb": round(gafter - (gbase or 0), 2) if gafter else None, "error": r.get("error")}
            run_with_retry(m, test, rep, f)
            continue
        if not c.ident:
            L.unload_all(); r = load_m(c, 2048, "bench-emb", False); c.ident = r.get("ident")
        if test == "emb_speed":
            def f():
                chunks = [doc(p[:800]) for p in paras[:256]]
                t0 = time.time(); n_ok = 0
                for i in range(0, len(chunks), 32):
                    v, dt, err = L.embed(c.ident, chunks[i:i + 32])
                    if err:
                        return False, {"error": err}
                    n_ok += len(v)
                dt = time.time() - t0
                chars = sum(len(x) for x in chunks)
                return True, {"chunks": n_ok, "seconds": round(dt, 2), "chunks_per_s": round(n_ok / dt, 1),
                              "approx_tokens_per_s": round(chars / 4 / dt), "dims": len(v[0]) if v else None}
            run_with_retry(m, test, rep, f)
        if test == "emb_retrieval":
            def f():
                docs = paras[:200]
                rng = random.Random(7)
                queries = []
                for i, p in enumerate(docs):
                    sents = [s for s in re.split(r"(?<=[.;:])\s+", p) if len(s) > 40]
                    if sents:
                        queries.append((i, rng.choice(sents)))
                dv = []
                for i in range(0, len(docs), 32):
                    v, _, err = L.embed(c.ident, [doc(x) for x in docs[i:i + 32]])
                    if err:
                        return False, {"error": err}
                    dv += v
                hits = 0
                for i in range(0, len(queries), 32):
                    batch = queries[i:i + 32]
                    qv, _, err = L.embed(c.ident, [qry(q) for _, q in batch])
                    if err:
                        return False, {"error": err}
                    for (gold, _), v in zip(batch, qv):
                        sims = [sum(a * b for a, b in zip(v, d)) / ((sum(a * a for a in v) ** .5) * (sum(b * b for b in d) ** .5) + 1e-9) for d in dv]
                        hits += max(range(len(sims)), key=sims.__getitem__) == gold
                return True, {"recall_at_1": round(hits / max(1, len(queries)), 3), "queries": len(queries)}
            run_with_retry(m, test, rep, f)
    L.unload_all()


# ---------------------------------------------------------------- judging

def run_judges(models):
    res = load_results()
    judged_tasks = {t["id"]: t for t in T.tasks() if t.get("judge")}
    answers = [r for r in res if r.get("final") and str(r.get("test", "")).startswith("q:") and r["test"][2:] in judged_tasks and r.get("answer")]
    have = {}
    for r in res:
        if r.get("type") == "judge":
            have.setdefault((r["model"], r["task"]), set()).add(r["judge"])
    keys = [m["key"] for m in models if not m["type"].startswith("embed")]
    judges = []
    for kw in CFG["judges"]:
        k = next((k for k in keys if kw in k.lower()), None)
        if k and k not in judges:
            judges.append(k)
    if not judges:
        log("no judge models available")
        return
    # assignment: 2 judges from other families
    work = {}
    for a in answers:
        tid = a["test"][2:]
        got = have.get((a["model"], tid), set())
        cands = [j for j in judges if family(j) != family(a["model"])] or [j for j in judges if j != a["model"]]
        for j in cands[:2]:
            if j not in got:
                work.setdefault(j, []).append(a)
    for j, items in work.items():
        check_pause()
        log(f"== judging with {j}: {len(items)} answers")
        c = Ctx({"key": j, "format": next((m.get("format") for m in models if m["key"] == j), ""), "path": ""})
        if not c.ensure(32768):
            log(f"  judge {j} failed to load"); continue
        for a in items:
            check_pause()
            t = judged_tasks[a["test"][2:]]
            prompt = T.judge_prompt(t, a["answer"])
            ans, stats, err = L.chat_v1(c.ident, prompt, system=T.JUDGE_SYSTEM, max_tokens=400, reasoning="off")
            if err or not ans or T.parse_judge(ans) is None:
                r2 = L.chat_stream(c.ident, [{"role": "system", "content": T.JUDGE_SYSTEM}, {"role": "user", "content": prompt}], max_tokens=3000)
                ans, err = r2.get("answer"), r2.get("error")
            score = T.parse_judge(ans or "")
            record({"type": "judge", "model": a["model"], "task": t["id"], "judge": j, "score": score,
                    "note": (ans or err or "")[:300], "final": True, "test": f"judge:{t['id']}:{j}", "rep": 0})
        L.unload_all()


# ---------------------------------------------------------------- main

ENV = {}
JUDGED = [False]
SRC = [os.path.join(HERE, f) for f in ("bench.py", "lsclient.py", "tasks.py", "agent.py", "report.py", "config.json")]
SRC_MTIME = {p: os.path.getmtime(p) for p in SRC if os.path.exists(p)}


_QUEUE_T = [0.0]
_QUEUE_LOCK = threading.Lock()


def _queue_loop():
    """Background thread: start queued downloads every 5 minutes, even during long tests."""
    n = 0
    while True:
        time.sleep(300)
        n += 1
        try:
            process_download_queue()
        except Exception as e:  # noqa
            log(f"download queue error: {e}")
        if n % 12 == 0:
            disk_diag()


_DISK_WAIT_LOGGED = {}


def process_download_queue():
    if not _QUEUE_LOCK.acquire(blocking=False):
        return
    try:
        _process_download_queue()
    finally:
        _QUEUE_LOCK.release()


def _process_download_queue():
    """data/download_queue.json: [{"model": "<Hugging Face link or catalog id>", "quantization": "Q8_0"}, ...].
    Each entry is sent once to LM Studio's download API (up to 3 tries), so downloads can be added while nobody is at the Mac.
    Results go to data/download_queue_done.json and the log."""
    qp, dp = os.path.join(DATA, "download_queue.json"), os.path.join(DATA, "download_queue_done.json")
    if not os.path.exists(qp):
        return
    try:
        queue = json.load(open(qp))
    except Exception as e:  # noqa
        log(f"download queue unreadable: {e}")
        return
    # download_queue_pending.json: entries that should only start once there is disk room for them
    pp = os.path.join(DATA, "download_queue_pending.json")
    if os.path.exists(pp) and isinstance(queue, list):
        try:
            queue = queue + [dict(q, wait_for_disk=True) for q in json.load(open(pp)) if isinstance(q, dict)]
        except Exception as e:  # noqa
            log(f"pending download queue unreadable: {e}")
    try:
        done = json.load(open(dp))
    except Exception:
        done = []
    changed = False
    # at most max_parallel_downloads at a time (paused ones don't count), smallest first, so each one finishes sooner
    slots = CFG.get("max_parallel_downloads", 5) - len(active_downloads())
    same = lambda d, q: d.get("model") == q["model"] and d.get("quantization") == q.get("quantization") and d.get("round", 1) == q.get("round", 1)
    pending_q = [q for q in (queue if isinstance(queue, list) else []) if isinstance(q, dict) and q.get("model")]
    committed = 0.0
    for q in sorted(pending_q, key=lambda q: q.get("need_gb") or q.get("size_gb") or 999):
        prev = [d for d in done if same(d, q)]
        if any(d.get("ok") for d in prev) or len(prev) >= 3:
            continue
        if slots <= 0:
            break
        if q.get("wait_for_disk") and (committed or active_downloads()):
            continue  # their remaining size isn't known, so a waiting download only starts once the others are done
        need = q.get("need_gb") or q.get("size_gb")
        if need:
            free = L.disk_free_gb()
            free = None if free is None else round(free - committed, 1)  # downloads started in this pass haven't written yet
            floor = q.get("min_free_gb", CFG.get("download_min_free_gb", 30))
            if free is not None and free - need < floor:
                k = (q["model"], q.get("quantization"), q.get("round", 1))
                if time.time() - _DISK_WAIT_LOGGED.get(k, 0) > 3600:
                    _DISK_WAIT_LOGGED[k] = time.time()
                    log(f"download waiting for disk room: {q['model']} {q.get('quantization') or ''} needs {need} GB, {free} GB free (keeps {floor} GB spare)")
                continue
        slots -= 1
        committed += need or 0
        body = {"model": q["model"], **({"quantization": q["quantization"]} if q.get("quantization") else {})}
        st, j = L.http("POST", "/api/v1/models/download", body, timeout=120)
        ok = st == 200 and isinstance(j, dict) and j.get("status") in ("downloading", "paused", "completed", "already_downloaded")
        log(f"download requested: {q['model']} {q.get('quantization') or ''} -> HTTP {st} {json.dumps(j, ensure_ascii=False)[:240]}")
        done.append(dict(q, ok=ok, http=st, response=j, ts=datetime.now().isoformat(timespec="seconds")))
        changed = True
    if changed:
        json.dump(done, open(dp, "w"), indent=1, ensure_ascii=False)


def cleanup_keys():
    """Model keys listed in data/delete_when_done.json entries that haven't been deleted yet."""
    try:
        return {k for e in json.load(open(os.path.join(DATA, "delete_when_done.json"))) if not e.get("done") for k in e.get("after") or []}
    except Exception:
        return set()


def delete_when_done(models, done):
    """data/delete_when_done.json: [{"after": [model keys], "paths": [files or folders under ~/.lmstudio/models], "why": ...}].
    The user approved each entry. Once every listed model has finished all phases, its files are emptied and removed here,
    so the space comes back right away (an open file handle elsewhere can't hold it) and waiting downloads can start."""
    p = os.path.join(DATA, "delete_when_done.json")
    if not os.path.exists(p):
        return
    try:
        entries = json.load(open(p))
    except Exception as e:  # noqa
        log(f"delete list unreadable: {e}")
        return
    root = os.path.realpath(os.path.expanduser("~/.lmstudio/models"))
    by_key = {m["key"]: m for m in models}
    changed = False
    for e in entries:
        after = e.get("after") or []
        if e.get("done") or not (after or e.get("now")):
            continue
        # every listed model must still be there with nothing left to run; a missing one means something changed, so wait.
        # Entries with "now": true are deleted at the next pause between models.
        if any(k not in by_key or pending(by_key[k], done) for k in after):
            continue
        L.unload_all()  # never truncate a file a loaded model still maps
        freed = 0
        for rel in e.get("paths") or []:
            full = os.path.realpath(os.path.join(root, rel))
            if not full.startswith(root + os.sep) or not os.path.exists(full):
                continue
            files = [full] if os.path.isfile(full) else [os.path.join(d, f) for d, _, fs in os.walk(full) for f in fs]
            for f in files:
                try:
                    size = os.path.getsize(f)
                    with open(f, "r+b") as fh:
                        fh.truncate(0)
                    os.remove(f)
                    freed += size
                except OSError as x:
                    log(f"could not delete {f}: {x}")
            if os.path.isdir(full):
                for d in sorted((d for d, _, _ in os.walk(full)), key=len, reverse=True):
                    try:
                        os.rmdir(d)
                    except OSError:
                        pass
        e["done"], e["freed_gb"] = datetime.now().isoformat(timespec="seconds"), round(freed / 1e9, 1)
        log(f"deleted (approved): {', '.join(after) or ', '.join(e.get('paths') or [])}, {e['freed_gb']} GB freed")
        changed = True
    if changed:
        json.dump(entries, open(p, "w"), indent=1, ensure_ascii=False)


def maybe_restart():
    """If the harness files were updated, restart with the same arguments (between models only)."""
    changed = [p for p in SRC if os.path.exists(p) and os.path.getmtime(p) != SRC_MTIME.get(p)]
    if changed:
        log(f"code updated ({', '.join(os.path.basename(p) for p in changed)}), restarting")
        L.unload_all()
        os.execv(sys.executable, [sys.executable] + sys.argv)


def disk_diag():
    """Read-only disk snapshot (free space, purgeable, local snapshots, deleted-but-open files) for remote diagnosis."""
    cmds = ["df -h /", "diskutil info / | grep -iE 'space|purgeable'",
            "tmutil listlocalsnapshots / 2>&1", "lsof -nP +L1 2>/dev/null | awk 'NR==1 || $7>100000000'",
            "for p in $(lsof -nP +L1 2>/dev/null | awk '$7>100000000 {print $2}' | sort -u); do ps -o pid=,etime=,command= -p $p; done"]
    out = [time.strftime("== %Y-%m-%d %H:%M:%S")]
    for c in cmds:
        try:
            r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=120)
            out += ["$ " + c, (r.stdout + r.stderr).strip()[:20000]]
        except Exception as e:
            out += ["$ " + c, f"error: {e}"]
    try:
        open(os.path.join(DATA, "disk_diag.txt"), "w").write("\n".join(out) + "\n")
    except Exception:
        pass


def main():
    global ARGS, ENV
    ap = argparse.ArgumentParser()
    ap.add_argument("--now", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--only")
    ap.add_argument("--no-judge", action="store_true")
    ARGS = ap.parse_args()
    global RESULTS
    if ARGS.smoke:
        ARGS.now = True
        RESULTS = os.path.join(DATA, "smoke_results.jsonl")
        if os.path.exists(RESULTS):
            os.remove(RESULTS)
    import report
    report.RESULTS = RESULTS
    report.CFG = CFG
    report.OUT = os.path.join(DATA, "smoke_report") if ARGS.smoke else report.OUT
    if ARGS.report:
        report.build(); return
    if not ARGS.smoke:
        import fcntl
        global _LOCK
        _LOCK = open(os.path.join(DATA, "bench.lock"), "w")
        try:
            fcntl.flock(_LOCK, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            log("another benchmark instance is already running; exiting")
            return
        _LOCK.write(str(os.getpid())); _LOCK.flush()
    if not ARGS.smoke and not ARGS.report:
        threading.Thread(target=_queue_loop, daemon=True).start()
        threading.Thread(target=disk_diag, daemon=True).start()
        if ARGS.now and CFG.get("heavy_use_pause"):
            threading.Thread(target=heavy_loop, daemon=True).start()
    log(f"start args={vars(ARGS)}")
    # model discovery works through the CLI even without a token; log it for diagnostics
    try:
        ms = alias_keys(expand_variants(L.list_models()))
        log(f"lms ls: {len(ms)} models")
        rc, raw = L.lms("ls", "--json", timeout=60)
        open(os.path.join(DATA, "lms_ls_raw.json"), "w").write(raw)
        rc, raw = L.lms("ps", "--json", timeout=60)  # diagnostics: loaded instances, including other Macs' over LM Link
        open(os.path.join(DATA, "lms_ps_raw.json"), "w").write(raw)
        el = eligible(ms)
        json.dump({"models": [{"key": m["key"], "type": m["type"], "format": m["format"], "size_gb": round((m.get("size_bytes") or 0) / 1e9, 1),
                               "max_ctx": m.get("max_ctx"), "vision": m.get("vision"), "excluded": m not in el, "path": m.get("path")} for m in ms],
                   "downloading": active_downloads(), "ts": datetime.now().isoformat(timespec="seconds")},
                  open(os.path.join(DATA, "inventory.json"), "w"), indent=1)
    except Exception as e:  # noqa
        log(f"lms ls failed: {e}")
    while True:
        st = L.ensure_server(log)
        if st == "ok":
            break
        if st == "auth":
            msg = ("LM Studio requires an API token. Create one in LM Studio (Developer > Server Settings > "
                   "Manage Tokens) and paste it into ~/LocalLlmBench/lmstudio_token.txt. The benchmark starts by itself once it's there.")
        else:
            msg = "LM Studio's server isn't reachable. Open LM Studio; the benchmark retries every 10 minutes."
        log(msg)
        report.NOTE = msg
        report.build()
        if ARGS.smoke:
            return
        time.sleep(600)
        maybe_restart()
    report.NOTE = ""
    global GUARD
    GUARD = L.Guard(swap_gb=6.0, disk_gb=12.0)
    ENV = L.environment()
    record({"type": "env", "env": ENV, "test": "env", "model": "_", "final": False})
    first = True
    linger_until = None
    while True:
        try:
            maybe_restart()
            try:
                process_download_queue()
            except Exception as e:  # noqa
                log(f"download queue error: {e}")
            allm = alias_keys(expand_variants(L.list_models()))
            models = eligible(allm)
            try:
                inv = [{"key": m["key"], "type": m["type"], "format": m["format"], "size_gb": round((m.get("size_bytes") or 0) / 1e9, 1),
                        "max_ctx": m.get("max_ctx"), "vision": m.get("vision"), "arch": m.get("arch"),
                        "excluded": m not in models, "path": m.get("path")} for m in allm]
                json.dump({"models": inv, "downloading": active_downloads(), "ts": datetime.now().isoformat(timespec="seconds")},
                          open(os.path.join(DATA, "inventory.json"), "w"), indent=1)
            except Exception:
                pass
            if not models:
                log("no models found"); break
            if ARGS.smoke:
                llms = [m for m in models if not m["type"].startswith("embed")]
                models = sorted(llms, key=lambda m: m.get("size_bytes") or 9e12)[:1]
            done = done_set()
            try:
                delete_when_done(models, done)
            except Exception as e:  # noqa
                log(f"delete-after-testing error: {e}")
            phase, todo = 0, []
            for ph in (0, 1, 2, 3, 4):
                todo = [m for m in models if pending(m, done, ph)]
                if todo:
                    phase = ph
                    break
            if todo and phase >= 1 and not JUDGED[0] and not ARGS.no_judge and not ARGS.smoke:
                # score the new writing answers before the long agent phases
                run_judges(models); JUDGED[0] = True
                report.build()
            if phase == 0:
                todo.sort(key=lambda m: (m["type"].startswith("embed"), prio(m["key"]), m.get("size_bytes") or 0))
            else:  # smallest first, so the agent leaderboard fills in quickly
                todo.sort(key=lambda m: m.get("size_bytes") or 0)
            # config priority_small_gb: models up to that size finish every phase (one load each) before bigger ones continue
            small_gb = CFG.get("priority_small_gb") or 0
            small = [m for m in models if small_gb and (m.get("size_bytes") or 0) <= small_gb * 1e9 and pending(m, done, 4)]
            if small and not ARGS.smoke:
                # brand-new small models (base tests still pending) first, then smallest first
                todo, phase = sorted(small, key=lambda m: (not pending(m, done, 0), m.get("size_bytes") or 0)), 4
            elif not ARGS.smoke:
                # next, models that get deleted once tested: finishing them frees disk for the waiting downloads
                cl = [m for m in models if m["key"] in cleanup_keys() and pending(m, done, 4)]
                if cl:
                    todo, phase = sorted(cl, key=lambda m: m.get("size_bytes") or 0), 4
            if not todo:
                if not ARGS.no_judge and not ARGS.smoke:
                    wait_slot(first); first = False
                    run_judges(models)
                report.build()
                if active_downloads():
                    log("all current models done; downloads still running, waiting 15 min")
                    time.sleep(900); continue
                # stay around a while so new models or harness updates are picked up without a manual restart
                hours = CFG.get("linger_after_done_h", 6)
                if not ARGS.smoke and hours:
                    if linger_until is None:
                        linger_until = time.time() + hours * 3600
                        log(f"all done; checking for new models or harness updates for the next {hours} h")
                    if time.time() < linger_until:
                        time.sleep(300); continue
                log("all done"); break
            linger_until = None
            wait_downloads()
            wait_slot(first); first = False
            maybe_restart()
            run_model(todo[0], phase)
            report.build()
        except Paused as p:
            log(f"paused: {p}. Unloading and waiting.")
            L.unload_all()
            report.build()
            wait_slot()
            maybe_restart()
        except KeyboardInterrupt:
            log("stopped by user"); L.unload_all(); break
        except Exception as e:  # noqa
            log(f"main loop error: {e}\n{traceback.format_exc()[-1500:]}")
            L.unload_all(); time.sleep(120)
    report.build()


if __name__ == "__main__":
    main()
