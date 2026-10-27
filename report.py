"""Builds the Obsidian report (hub note, per-model notes, base, chart) and the community package
from data/results.jsonl. Pure standard library; safe to run any time on partial data."""
import csv, json, os, re, shutil, statistics as st, zipfile
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "data", "results.jsonl")
OUT = os.path.join(HERE, "report")
CFG = {}
NOTE = ""

import tasks as T  # noqa: E402
import agent as A  # noqa: E402

CATS = ["instructions", "json", "tools", "coding", "reasoning", "writing", "summary", "long_context"]
CAT_LABEL = {"instructions": "Instructions", "json": "JSON", "tools": "Tools", "coding": "Coding",
             "reasoning": "Reasoning", "writing": "Writing (ES)", "summary": "Summaries", "long_context": "Long ctx recall",
             "vision": "Vision"}


# ---------------------------------------------------------------- helpers

def est_gb(text):
    """LM Studio's own memory estimate ('Estimated Total Memory: 20.97 GiB') in GB."""
    m = re.search(r"Estimated Total Memory:\s*([\d.]+)\s*GiB", text or "")
    return round(float(m.group(1)) * 1.0737, 1) if m else None


def pos(x):
    return x if isinstance(x, (int, float)) and x > 0 else None


def med(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return st.median(xs) if xs else None


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else None


def spread(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    if len(xs) < 2:
        return None
    m = st.median(xs)
    return st.pstdev(xs) / m * 100 if m else None


def f(x, nd=1, suf=""):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:,.{nd}f}{suf}"
    return f"{x}{suf}"


def ctxk(x):
    return f"{int(x) // 1024}k" if x else "-"


def short_name(key, fmt=""):
    k = key.split("/")[-1].split("@")[0]
    k = re.sub(r"(?i)[-_](gguf|mlx|instruct|it)\b", "", k)
    k = re.sub(r"(?i)[-_]?(\d+bit|q\d_k_\w+|ud-q\d\w*)$", "", k)
    tag = "MLX" if ("safetensors" in fmt or "mlx" in fmt or "mlx" in key.lower()) else ("GGUF" if "gguf" in fmt or "gguf" in key.lower() else "")
    return f"{k} ({tag})" if tag else k


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def load():
    """All result rows, minus the ones listed in data/invalidate.json (tests that were redone)."""
    rows = []
    try:
        inv = json.load(open(os.path.join(os.path.dirname(RESULTS), "invalidate.json")))
    except Exception:
        inv = []
    def bad(r):
        return any(x.get("model") in ("*", r.get("model")) and str(r.get("test", "")).startswith(x.get("test_prefix", ""))
                   and str(r.get("ts", "")) < x.get("before", "") for x in inv)
    if os.path.exists(RESULTS):
        for line in open(RESULTS, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not (inv and bad(r)):
                rows.append(r)
    return rows


def engine_str(eng, fmt):
    """'llama.cpp 2.63.0' / 'MLX 1.13.1' for the engine a model ran on."""
    for e in eng or []:
        if isinstance(e, dict) and e.get("model_format") == ("safetensors" if "MLX" in (fmt or "") else "gguf"):
            return ("MLX " if "mlx" in e.get("name", "") else "llama.cpp ") + str(e.get("version", ""))
    return None


# ---------------------------------------------------------------- aggregation

def aggregate(rows):
    env = next((r["env"] for r in reversed(rows) if r.get("type") == "env"), {})
    meta = {}
    for r in rows:
        if r.get("type") == "meta":
            meta[r["model"]] = dict(r["meta"], family=r.get("family"))
    by = {}
    for r in rows:
        if not r.get("final") or r.get("type") in ("judge", "env", "meta"):
            continue
        # keep latest record per (test, rep)
        by.setdefault(r["model"], {})[(r["test"], r.get("rep", 0))] = r
    judges = {}
    for r in rows:
        if r.get("type") == "judge" and r.get("score") is not None:
            judges.setdefault((r["model"], r["task"]), []).append(r["score"])
    task_info = {t["id"]: t for t in T.tasks()}
    M = {}
    for key, tests in by.items():
        mt = meta.get(key, {})
        def recs(name, skip_warm=False):
            return [v for (t, rep), v in sorted(tests.items()) if t == name and v.get("ok") and not (skip_warm and v.get("warmup"))]
        fmt = ("MLX" if mt.get("format") == "safetensors" else str(mt.get("format") or "").upper())
        fmt = (fmt + " " + str(mt.get("quant") or "")).strip()
        a = {"key": key, "name": short_name(key, mt.get("format", "")), "format": fmt,
             "size_gb": round((mt.get("size_bytes") or 0) / 1e9, 1) or None, "params": mt.get("params"),
             "arch": mt.get("arch"), "native_ctx": mt.get("max_ctx"), "family": mt.get("family"),
             "type": mt.get("type", "llm")}
        if str(a["type"]).startswith("embed"):
            lo = recs("emb_load"); sp = recs("emb_speed"); rt = recs("emb_retrieval")
            a.update({"embedding": True, "load_s": lo[0].get("load_s") if lo else None,
                      "mem_gb": pos(lo[0].get("mem_model_gb")) if lo else None,
                      "chunks_per_s": sp[0].get("chunks_per_s") if sp else None,
                      "tok_s": sp[0].get("approx_tokens_per_s") if sp else None,
                      "dims": sp[0].get("dims") if sp else None,
                      "recall1": rt[0].get("recall_at_1") if rt else None})
            M[key] = a
            continue
        ls = recs("load_std")
        failed = [v for (t, _), v in tests.items() if t == "load_std" and not v.get("ok")]
        a["load_failed"] = bool(failed and not ls)
        if failed and not ls:
            a["load_error"] = json.dumps(failed[0].get("tried", ""))[:300]
        if ls:
            l0 = ls[0]
            est = est_gb(l0.get("estimate"))
            a.update({"ctx_std": l0.get("ctx"), "load_first_s": l0.get("load_s"),
                      "mem_gb": est or pos(l0.get("mem_model_gb")),
                      "mem_src": "estimate" if est else "measured",
                      "mem_measured_gb": pos(l0.get("mem_model_gb")),
                      "estimate": l0.get("estimate")})
            eng = (l0.get("env") or {}).get("engines")
            a["engine"] = engine_str(eng, fmt)
        lr = recs("load_repeat")
        a["load_repeat_s"] = lr[0].get("load_s") if lr else None
        sh = recs("short", skip_warm=True)
        a["tg_short"] = med([r.get("decode_tps") for r in sh])
        a["tg_short_spread"] = spread([r.get("decode_tps") for r in sh])
        a["ttft_short"] = med([r.get("ttft") for r in sh])
        a["watts"] = med([r.get("watts") for r in sh])
        a["tok_per_j"] = med([r.get("tokens_per_joule") for r in sh])
        for d in ("d8k", "d32k", "d60k"):
            rr = recs(d)
            a[f"ttft_{d}"] = med([r.get("ttft") for r in rr])
            a[f"pp_{d}"] = med([r.get("prefill_tps") for r in rr])
            a[f"tg_{d}"] = med([r.get("decode_tps") for r in rr])
            a[f"needle_{d}"] = mean([r.get("needle") for r in rr])
            a[f"mem_peak_{d}"] = max([r.get("mem_peak_gb") or 0 for r in rr], default=None) or None
        lm = recs("load_max")
        if lm:
            a["ctx_max_loaded"] = lm[0].get("ctx")
            a["mem_max_gb"] = est_gb(lm[0].get("estimate")) or pos(lm[0].get("mem_model_gb"))
            a["ctx_max_note"] = lm[0].get("note")
        dm = recs("dmax")
        if dm and not dm[0].get("skipped"):
            a.update({"dmax_tokens": dm[0].get("prompt_tokens") or dm[0].get("target_tokens"), "dmax_ttft": dm[0].get("ttft"),
                      "dmax_tg": dm[0].get("decode_tps"), "dmax_needle": dm[0].get("needle"),
                      "dmax_mem_peak": dm[0].get("mem_peak_gb"), "dmax_swap": dm[0].get("swap_delta_mb")})
        ve = recs("vis_encode")
        a["img_overhead_s"] = ve[0].get("image_overhead_s") if ve else None
        # quality
        cat_scores, task_rows, times, rtoks, ttfas = {}, {}, [], [], []
        for (t, _), v in tests.items():
            if not t.startswith("q:"):
                continue
            tid = t[2:]
            ti = task_info.get(tid)
            if not ti:
                continue
            auto = v.get("score") or 0.0
            try:
                if ti["kind"] == "tool" and ("answer" in v or "tool_calls" in v):
                    auto, v["note"] = ti["grader"]({"content": v.get("answer") or "", "tool_calls": v.get("tool_calls")})
                elif ti["kind"] != "tool" and v.get("answer") is not None:
                    auto, v["note"] = ti["grader"](v.get("answer") or "")
            except Exception:
                pass
            js = judges.get((key, tid))
            score = auto
            jm = None
            if ti.get("judge") and js:
                jm = sum(js) / len(js)
                score = 0.3 * auto + 0.7 * jm / 10
            task_rows[tid] = {"cat": ti["cat"], "auto": auto, "judge": jm, "score": score, "note": v.get("note"),
                              "t_total": v.get("t_total"), "reasoning_tokens": v.get("reasoning_tokens"),
                              "answer": v.get("answer")}
            cat_scores.setdefault(ti["cat"], []).append(score)
            times.append(v.get("t_total")); rtoks.append(v.get("reasoning_tokens")); ttfas.append(v.get("ttfa"))
        needles = [a.get("needle_d8k"), a.get("needle_d32k"), a.get("needle_d60k")]
        if any(n is not None for n in needles):
            cat_scores["long_context"] = [n for n in needles if n is not None]
        a["tasks"] = task_rows
        a["capped"] = sum(1 for (t, _), v in tests.items() if t.startswith("q:") and v.get("finish") == "length" and not (v.get("answer") or "").strip()
                          and not v.get("tool_calls"))
        a["cats"] = {c: mean(v) for c, v in cat_scores.items()}
        text_cats = [a["cats"][c] for c in CATS if a["cats"].get(c) is not None]
        a["quality"] = mean(text_cats) * 100 if text_cats and task_rows else None
        a["n_cats"] = len(text_cats)
        a["vision"] = a["cats"].get("vision") * 100 if a["cats"].get("vision") is not None else None
        a["sec_per_task"] = med([x for x, (tid, tr) in zip(times, task_rows.items()) if tr["cat"] != "vision"])
        a["reason_tokens"] = mean(rtoks)
        a["ttfa"] = med(ttfas)
        a["judge_pending"] = any(task_info[t].get("judge") and r["judge"] is None and (r.get("answer") or "").strip() for t, r in task_rows.items())
        agent_metrics(a, tests, recs)
        a["disturbed"] = sum(1 for v in tests.values() if v.get("disturbed"))
        a["errors"] = sum(1 for v in tests.values() if not v.get("ok"))
        a["early_stop"] = sum(1 for (t, _), v in tests.items() if t in ("short", "d8k", "d32k") and v.get("ok") and not v.get("hit_limit"))
        a["tests_done"] = len(tests)
        M[key] = a
    # several quants of one model: keep them apart by name, and remember the shared base name for the quant table
    groups = {}
    for key, a in M.items():
        q = str(meta.get(key, {}).get("quant") or (key.split("@")[1] if "@" in key else ""))
        a["base_name"], a["quant"] = a["name"], (q.lower() if "bit" in q.lower() else q.upper())
        groups.setdefault(a["name"], []).append(a)
    for name, xs in groups.items():
        if len(xs) > 1:
            for a in xs:
                if a["quant"]:
                    a["name"] = name.replace(" (", f" {a['quant']} (") if " (" in name else f"{name} {a['quant']}"
    return env, M


_REGRADED = {}


def collect_trials(a, tests, prefix):
    """task id -> list of run records for one suite ('ag:' main, 'fx:' real cases). Re-grades old traces when a grader changed."""
    ag = {}
    for (t, rep), v in sorted(tests.items()):
        if not t.startswith(prefix):
            continue
        ti = A.TASK_BY_ID.get(t[3:])
        if not ti:
            continue
        if v.get("trace") is not None and not v.get("skipped") and v.get("grader_v") != A.version(ti):
            ck = (a["key"], t, rep, v.get("ts"), A.version(ti))
            try:
                if ck not in _REGRADED:  # replaying is cheap, except the code tasks, so remember results between builds
                    _REGRADED[ck] = A.regrade(ti, v)
                v["score"], v["note"] = _REGRADED[ck]
                v["solved"] = v["score"] >= A.PASS
            except Exception:
                pass
        if v.get("skipped") or (v.get("score") is None and v.get("trace") is None):  # never ran (model unloadable): not a 0
            continue
        ag.setdefault(ti["id"], []).append(v)
    return ag


def field_metrics(a, tests):
    """Real-cases tier: score, solved every time, serious slips, by category."""
    fx = collect_trials(a, tests, "fx:")
    a["field_tasks"], a["field_n"] = fx, len(fx)
    if not fx:
        a["field_cats"] = {}
        return
    k = CFG.get("agent_trials", 3)
    allv = [x for xs in fx.values() for x in xs]
    run = [x for x in allv if not x.get("skipped")]
    a["field_score"] = mean([task_mean(xs) for xs in fx.values()]) * 100
    full = [xs for xs in fx.values() if len(xs) >= k]
    a["field_pass_k"] = sum(all(x.get("solved") for x in xs) for xs in full) / len(full) * 100 if full else None
    a["field_k_tasks"] = len(full)
    a["field_runs"] = len(run)
    a["field_slips"] = sum(1 for tid, xs in fx.items() for x in xs if A.slipped(tid, x.get("note")))
    a["field_sec"] = med([x.get("t_total") for x in run])
    a["field_timeouts"] = sum(1 for x in allv if x.get("stop") == "timeout")
    cs = {}
    for tid, xs in fx.items():
        cs.setdefault(A.TASK_BY_ID[tid]["cat"], []).append(task_mean(xs))
    a["field_cats"] = {c: mean(v) for c, v in cs.items()}


def agent_metrics(a, tests, recs):
    """Agent-suite and prefix-cache numbers for one model."""
    field_metrics(a, tests)
    ag = collect_trials(a, tests, "ag:")
    a["agent_tasks"] = ag
    a["agent_n"] = len(ag)
    if ag:
        allv = [x for xs in ag.values() for x in xs]
        run = [x for x in allv if not x.get("skipped")]
        a["agent_score"] = mean([mean([x.get("score") or 0 for x in xs]) for xs in ag.values()]) * 100
        a["agent_pass1"] = mean([1.0 if x.get("solved") else 0.0 for x in allv]) * 100
        k = CFG.get("agent_trials", 3)
        full = [xs for xs in ag.values() if len(xs) >= k]
        a["agent_pass_k"] = sum(all(x.get("solved") for x in xs) for xs in full) / len(full) * 100 if full else None
        a["agent_k_tasks"] = len(full)
        calls = sum(x.get("tool_calls") or 0 for x in run)
        invalid = sum((x.get("bad_json") or 0) + (x.get("schema_errors") or 0) + (x.get("unknown_tool") or 0) for x in run)
        a["agent_calls"], a["agent_invalid"] = calls, invalid
        a["agent_valid_pct"] = (1 - invalid / calls) * 100 if calls else None
        a["agent_text_calls"] = sum(x.get("text_tool_calls") or 0 for x in run)
        a["agent_steps"] = med([x.get("steps") for x in run])
        a["agent_sec"] = med([x.get("t_total") for x in run])
        a["agent_tokens"] = med([x.get("completion_tokens") for x in run])
        a["agent_timeouts"] = sum(1 for x in allv if x.get("stop") == "timeout")
        errs = [x for x in run if x.get("stop") == "error"]
        a["agent_errors"], a["agent_error_msg"] = len(errs), (str(errs[0].get("error"))[:160] if errs else None)
        pw = [x for x in ag.get("ag_parallel_weather", []) if not x.get("skipped")]
        a["agent_parallel"] = max([x.get("parallel_max") or 0 for x in pw]) if pw else None
        inj = [x for x in ag.get("ag_injection", []) if not x.get("skipped")]
        a["agent_inj"] = (sum(1 for x in inj if "followed the injected" not in str(x.get("note"))), len(inj)) if inj else None
        dr = ag.get("ag_draft_not_send", [])
        a["agent_unauth"] = sum(1 for x in dr if "without permission" in str(x.get("note")))
        cl = ag.get("ag_clarify", [])
        a["agent_asked"] = (sum(1 for x in cl if "asked=True" in str(x.get("note"))), len(cl)) if cl else None
        cs = {}
        for tid, xs in ag.items():
            cs.setdefault(A.TASK_BY_ID[tid]["cat"], []).append(mean([x.get("score") or 0 for x in xs]))
        a["agent_cats"] = {c: mean(v) for c, v in cs.items()}
        per_run = [x * 100 for x in (mean([xs[r].get("score") or 0 for xs in ag.values() if len(xs) > r]) for r in range(k)) if x is not None]
        a["agent_run_sd"] = st.pstdev(per_run) if len(ag) == len(A.TASKS) and all(len(xs) >= k for xs in ag.values()) else None
    else:
        a["agent_cats"] = {}
    cache = recs("ag_cache")
    if cache:
        a["cache_tokens"] = med([r.get("prompt_tokens") for r in cache])
        a["cache_cold"], a["cache_warm"], a["cache_cont"] = (med([r.get(k) for r in cache]) for k in ("cold_ttft", "warm_ttft", "cont_ttft"))
        a["cache_speedup"] = a["cache_cold"] / a["cache_warm"] if a["cache_cold"] and a["cache_warm"] else None


def pareto(models):
    pts = [(m["key"], m.get("tg_d8k") or m.get("tg_short"), m.get("quality")) for m in models]
    pts = [p for p in pts if p[1] and p[2] is not None]
    out = set()
    for k, s, q in pts:
        if not any((s2 >= s and q2 >= q) and (s2 > s or q2 > q) for k2, s2, q2 in pts if k2 != k):
            out.add(k)
    return out


def rank(models, field, reverse=True):
    xs = [m for m in models if m.get(field) is not None]
    xs.sort(key=lambda m: m[field], reverse=reverse)
    return {m["key"]: i for i, m in enumerate(xs)}


def best(models, fn, cond=lambda m: True):
    """Best model by fn among those passing cond; models with a source caveat are measured but never picked."""
    xs = [m for m in models if cond(m) and fn(m) is not None and not provenance_note(m)]
    return max(xs, key=fn) if xs else None


# ---------------------------------------------------------------- narrative

def pros_cons(m, chat):
    pros, cons, uses = [], [], []
    n = len(chat)
    r_speed = rank(chat, "tg_d8k")
    r_q = rank(chat, "quality")
    if m.get("load_failed"):
        return [], ["Doesn't load in LM Studio on this Mac"], ["Not usable here"]
    tg = m.get("tg_d8k") or m.get("tg_short")
    if n >= 4 and m["key"] in r_speed and r_speed[m["key"]] < 3:
        pros.append(f"Among the fastest here ({f(tg)} tok/s at 8k)")
    if m["key"] in r_speed and n >= 5 and r_speed[m["key"]] >= n - 3:
        cons.append(f"Among the slowest here ({f(tg)} tok/s at 8k)")
    if n >= 4 and m["key"] in r_q and r_q[m["key"]] < 3 and m.get("n_cats", 0) >= 5:
        pros.append(f"Top-3 quality score ({f(m['quality'], 0)}%)")
    if m["key"] in r_q and n >= 5 and r_q[m["key"]] >= n - 3 and m.get("n_cats", 0) >= 5:
        cons.append(f"Bottom-3 quality score ({f(m['quality'], 0)}%)")
    c = m.get("cats", {})
    if (c.get("tools") or 0) >= 0.9: pros.append("Reliable tool calling")
    elif c.get("tools") is not None and c["tools"] < 0.5: cons.append("Unreliable tool calling")
    if (c.get("json") or 0) >= 0.9: pros.append("Clean structured JSON")
    elif c.get("json") is not None and c["json"] < 0.6: cons.append("Sloppy JSON output")
    if (c.get("writing") or 0) >= 0.8: pros.append("Good Spanish writing")
    elif c.get("writing") is not None and c["writing"] < 0.55: cons.append("Weak Spanish writing")
    if (c.get("coding") or 0) >= 0.9: pros.append("Passes the coding tests")
    elif c.get("coding") is not None and c["coding"] < 0.5: cons.append("Fails most coding tests")
    if m.get("vision") is not None:
        if m["vision"] >= 80: pros.append(f"Strong vision ({f(m['vision'], 0)}%)")
        elif m["vision"] < 50: cons.append(f"Weak vision ({f(m['vision'], 0)}%)")
    nd = [m.get("needle_d8k"), m.get("needle_d32k"), m.get("needle_d60k")]
    if all(x == 1 for x in nd if x is not None) and any(x is not None for x in nd):
        pros.append("Found the hidden fact at every context depth")
    elif any(x is not None and x < 1 for x in nd):
        cons.append("Lost the hidden fact in long context")
    if (m.get("ctx_max_loaded") or 0) >= 131072:
        pros.append(f"Loads {ctxk(m['ctx_max_loaded'])} context")
    if m.get("ctx_std") and m["ctx_std"] < 65536:
        cons.append(f"Can't run a 64k agent context (max {ctxk(m['ctx_std'])})")
    mem = m.get("mem_gb")
    if mem is not None and mem <= 6: pros.append(f"Small footprint (~{f(mem)} GB)")
    if mem is not None and mem >= 30: cons.append(f"Heavy (~{f(mem)} GB at 64k)")
    if (m.get("reason_tokens") or 0) > 800:
        cons.append(f"Thinks a lot: ~{int(m['reason_tokens'])} reasoning tokens and {f(m.get('sec_per_task'))} s per task")
    # agent suite
    agents = [x for x in chat if x.get("agent_n", 0) >= 12]
    r_ag = rank(agents, "agent_score")
    if m.get("agent_n", 0) >= 12:
        if len(agents) >= 4 and r_ag.get(m["key"], 99) < 3:
            pros.append(f"Top-3 agent score ({f(m['agent_score'], 0)}%)")
        if len(agents) >= 5 and r_ag.get(m["key"], -1) >= len(agents) - 3:
            cons.append(f"Bottom-3 agent score ({f(m['agent_score'], 0)}%)")
    if m.get("agent_inj") and m["agent_inj"][0] < m["agent_inj"][1]:
        cons.append(f"Obeyed a prompt injection hidden in an email ({m['agent_inj'][1] - m['agent_inj'][0]}/{m['agent_inj'][1]} runs)")
    if m.get("agent_unauth"):
        cons.append(f"Sent an email it was told only to draft ({m['agent_unauth']} run(s))")
    if m.get("agent_text_calls"):
        cons.append(f"Wrote tool calls as plain text {m['agent_text_calls']} time(s), so they never ran")
    if m.get("agent_valid_pct") is not None and m["agent_valid_pct"] < 90:
        cons.append(f"Sloppy tool arguments ({100 - m['agent_valid_pct']:.0f}% of calls rejected by the schema)")
    if (m.get("agent_parallel") or 0) >= 3:
        pros.append("Makes parallel tool calls")
    if m.get("cache_speedup"):
        if m["cache_speedup"] >= 3:
            pros.append(f"Reuses the prompt cache: follow-up turns start {m['cache_speedup']:.0f}x faster")
        elif m["cache_speedup"] < 1.5:
            cons.append("Re-reads the whole agent prompt every turn (no prompt-cache reuse)")
    if m.get("agent_timeouts"):
        cons.append(f"{m['agent_timeouts']} agent run(s) hit the 30-minute limit")
    # recommended use: categories where top-2
    for cat in CATS + ["vision"]:
        xs = sorted([x for x in chat if x.get("cats", {}).get(cat) is not None], key=lambda x: x["cats"][cat], reverse=True)
        if xs and m in xs[:2] and m["cats"][cat] >= 0.6:
            uses.append(CAT_LABEL[cat])
    if m.get("agent_n", 0) >= 12 and r_ag.get(m["key"], 99) < 2 and (m.get("agent_score") or 0) >= 60:
        uses.append("Agents")
    cod = sorted([x for x in agents if x.get("agent_cats", {}).get("coding") is not None], key=lambda x: (-x["agent_cats"]["coding"], x.get("agent_sec") or 0))
    if m in cod[:2] and m["agent_cats"]["coding"] >= 0.8:
        uses.append("Coding agent")
    if mem is not None and mem <= 14 and (m.get("quality") or 0) >= 50:
        uses.append("MacBook Air (24 GB)")
    if (m.get("tg_d8k") or 0) and m["key"] in r_speed and r_speed[m["key"]] < 3 and (m.get("quality") or 0) >= 50:
        uses.append("Quick drafts")
    return pros, cons, uses or ["General backup"]


# ---------------------------------------------------------------- SVG chart

def scatter_svg(chat, par, path):
    pts = [(m["name"], m.get("tg_d8k") or m.get("tg_short"), m.get("quality"), m["key"] in par) for m in chat]
    pts = [p for p in pts if p[1] and p[2] is not None]
    W, H, P = 900, 560, 70
    if not pts:
        return False
    xmax = max(p[1] for p in pts) * 1.1
    ymin = max(0, min(p[2] for p in pts) - 10); ymax = min(100, max(p[2] for p in pts) + 5)
    X = lambda v: P + (W - 2 * P) * v / xmax
    Y = lambda v: H - P - (H - 2 * P) * (v - ymin) / max(1, ymax - ymin)
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="-apple-system,Helvetica,Arial" font-size="13">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         f'<text x="{P}" y="32" font-size="17" font-weight="600" fill="#222">Quality vs speed (Mac Studio M1 Max 64 GB, 8k context)</text>',
         f'<line x1="{P}" y1="{H-P}" x2="{W-P}" y2="{H-P}" stroke="#999"/><line x1="{P}" y1="{P}" x2="{P}" y2="{H-P}" stroke="#999"/>',
         f'<text x="{W/2}" y="{H-25}" text-anchor="middle" fill="#444">Generation speed at 8k context (tokens/s)</text>',
         f'<text x="20" y="{H/2}" transform="rotate(-90 20 {H/2})" text-anchor="middle" fill="#444">Quality score (%)</text>']
    for i in range(6):
        xv = xmax * i / 5; yv = ymin + (ymax - ymin) * i / 5
        s.append(f'<text x="{X(xv)}" y="{H-P+18}" text-anchor="middle" fill="#666">{xv:.0f}</text>')
        s.append(f'<text x="{P-8}" y="{Y(yv)+4}" text-anchor="end" fill="#666">{yv:.0f}</text>')
        s.append(f'<line x1="{P}" y1="{Y(yv)}" x2="{W-P}" y2="{Y(yv)}" stroke="#eee"/>')
    fr = sorted([p for p in pts if p[3]], key=lambda p: p[1])
    if len(fr) > 1:
        s.append('<polyline fill="none" stroke="#e07a1f" stroke-dasharray="5,4" points="' + " ".join(f"{X(p[1]):.0f},{Y(p[2]):.0f}" for p in fr) + '"/>')
    picked = {m["name"] for m in chat if m["key"] in PICKED}
    for name, x, y, on in pts:  # dots first, labels on top; only frontier models and picks get a label
        col = "#e07a1f" if (on or name in picked) else "#d9d9d9"
        r = 7 if (on or name in picked) else 4.5
        s.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="{r}" fill="{col}"' + (' stroke="#222" stroke-width="2"' if name in picked else "") + "/>")
    s += svg_labels([(label_name(n), X(x), Y(y)) for n, x, y, on in pts if on or n in picked], W, color="#e07a1f")
    s.append(f'<text x="{W-P}" y="{H-P-8}" text-anchor="end" fill="#e07a1f">orange = no other model is both faster and better; black ring = a pick</text>')
    s.append("</svg>")
    open(path, "w").write("\n".join(s))
    return True


def label_name(n):
    return re.sub(r" \((GGUF|MLX)\)$", "", n)


def svg_labels(items, W, font_w=7.4, line_h=15, color="#222"):
    """Place text labels next to points without overlapping each other (greedy: shift down, then try the left side)."""
    boxes, out = [], []
    for text, x, y in sorted(items, key=lambda t: (t[2], t[1])):
        w = len(text) * font_w
        placed = None
        for side in (1, -1):
            lx = x + 10 if side > 0 else x - 10 - w
            if lx < 4 or lx + w > W - 4:
                continue
            for k in range(0, 6):
                ly = y - 8 + k * line_h
                box = (lx, ly - 11, lx + w, ly + 3)
                if not any(not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3]) for b in boxes):
                    placed = (lx, ly); break
            if placed:
                break
        if not placed:
            placed = (x + 10, y - 8)
        boxes.append((placed[0], placed[1] - 11, placed[0] + w, placed[1] + 3))
        out.append(f'<text x="{placed[0]:.1f}" y="{placed[1]:.1f}" fill="{color}" font-weight="600">{text}</text>')
    return out


def agent_svg(models, path):
    """Agent score vs median seconds per task (log scale): up and to the left is better."""
    import math
    pts = [(m["name"], m.get("agent_sec"), m.get("agent_score")) for m in models if m.get("agent_n", 0) >= 12]
    pts = [p for p in pts if p[1] and p[2] is not None]
    if len(pts) < 2:
        return False
    W, H, P = 900, 560, 70
    lo, hi = math.log10(max(1, min(p[1] for p in pts) / 1.5)), math.log10(max(p[1] for p in pts) * 1.5)
    ymin = max(0, min(p[2] for p in pts) - 10); ymax = min(100, max(p[2] for p in pts) + 5)
    X = lambda v: P + (W - 2 * P) * (math.log10(v) - lo) / max(1e-9, hi - lo)
    Y = lambda v: H - P - (H - 2 * P) * (v - ymin) / max(1, ymax - ymin)
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="-apple-system,Helvetica,Arial" font-size="13">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         f'<text x="{P}" y="32" font-size="17" font-weight="600" fill="#222">Agent score vs time per task (M1 Max 64 GB, 16 tool-use tasks)</text>',
         f'<line x1="{P}" y1="{H-P}" x2="{W-P}" y2="{H-P}" stroke="#999"/><line x1="{P}" y1="{P}" x2="{P}" y2="{H-P}" stroke="#999"/>',
         f'<text x="{W/2}" y="{H-25}" text-anchor="middle" fill="#444">Median seconds per agent task (log scale, lower is better)</text>',
         f'<text x="20" y="{H/2}" transform="rotate(-90 20 {H/2})" text-anchor="middle" fill="#444">Agent score (%)</text>']
    for v in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000):
        if lo <= math.log10(v) <= hi:
            s.append(f'<text x="{X(v):.0f}" y="{H-P+18}" text-anchor="middle" fill="#666">{v}</text>')
            s.append(f'<line x1="{X(v):.0f}" y1="{P}" x2="{X(v):.0f}" y2="{H-P}" stroke="#f1f1f1"/>')
    for i in range(6):
        yv = ymin + (ymax - ymin) * i / 5
        s.append(f'<text x="{P-8}" y="{Y(yv)+4}" text-anchor="end" fill="#666">{yv:.0f}</text>')
        s.append(f'<line x1="{P}" y1="{Y(yv)}" x2="{W-P}" y2="{Y(yv)}" stroke="#eee"/>')
    picked = {m["name"] for m in models if m["key"] in PICKED}
    # label the picks plus the frontier of this chart (no other model is both faster and better)
    front = {n for n, x, y in pts if not any(x2 < x and y2 >= y and (x2, y2) != (x, y) for _, x2, y2 in pts)}
    for name, x, y in pts:
        lab = name in picked or name in front
        s.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="{7 if lab else 4.5}" fill="{"#2f8f6b" if lab else "#d9d9d9"}"'
                 + (' stroke="#222" stroke-width="2"' if name in picked else "") + "/>")
    s += svg_labels([(label_name(n), X(x), Y(y)) for n, x, y in pts if n in picked or n in front], W, color="#2f8f6b")
    s.append(f'<text x="{W-P}" y="{H-P-8}" text-anchor="end" fill="#2f8f6b">dark = no other model is both faster and better; black ring = a pick</text>')
    s.append("</svg>")
    open(path, "w").write("\n".join(s))
    return True


PICKED = {}  # model key -> pick labels, filled while the picks list is built; bold in tables and labelled in charts


def nm(m):
    """Model name for a table cell, bold when the model is one of the picks."""
    return f"**{m['name']}**" if m["key"] in PICKED else m["name"]


def lk(m):
    """Wikilinked model name for the Obsidian report, bold when picked."""
    t = f"[[{slug(m['name'])}|{m['name']}]]"
    return f"**{t}**" if m["key"] in PICKED else t


def qcell(m):
    """Quality cell; '*' when tasks ran out of tokens before answering and could not be rerun."""
    return f(m.get("quality"), 0) + ("*" if m.get("capped") else "")


CAPPED_NOTE = ("\\* Ran out of output tokens before answering on some quality tasks (those score 0). Installed models were rerun with "
               "a 16,384-token cap (8,192 for vision); deleted models only had the earlier 4,096 cap, so their quality is understated.")


def agent_cell(m):
    if not m.get("agent_n"):
        return "-"
    v = f(m.get("agent_score"), 0)
    return v if m["agent_n"] >= len(A.TASKS) else f"{v} ({m['agent_n']}/{len(A.TASKS)})"


def task_mean(xs):
    return mean([x.get("score") or 0 for x in xs])


def agent_sections(ok, agents, K):
    agm = sorted([m for m in ok if m.get("agent_n")], key=lambda m: -(m.get("agent_score") or 0))
    if not agm:
        return []
    n = len(A.TASKS)
    L = ["## Agents and tool use", "",
         f"{n} multi-step tasks where the model works through tools (notes, calendar, email, files, a small code repo...) and is graded on what "
         "actually happened: the event that got created, the email that got sent or not, the tests that pass. "
         f"Each task runs {K} times. 'Solved every time' is the share of tasks solved in all {K} runs, which is what matters when an agent runs unattended."]
    sds = [m["agent_run_sd"] for m in agm if m.get("agent_run_sd") is not None]
    if len(sds) >= 3:
        L[-1] += (f" The same model's score moves about {med(sds):.0f} point{'s' if round(med(sds)) != 1 else ''} from one pass to the next (up to {max(sds):.0f} for the least stable), "
                  "so treat gaps of 3 points or less as ties.")
    L += [""]
    hdr = ["Model", "Agent score", f"Solved all {K} runs", "Valid tool calls", "Median steps", "s per task",
           "Tokens per task", "Parallel calls", "Ignored injection", "Notes"]
    rws = []
    for m in agm:
        notes = []
        if m["agent_n"] < n:
            notes.append(f"{m['agent_n']}/{n} tasks so far")
        elif (m.get("agent_k_tasks") or 0) < m["agent_n"]:
            notes.append(f"repeat runs done for {m.get('agent_k_tasks') or 0}/{n} tasks")
        if m.get("agent_timeouts"):
            notes.append(f"{m['agent_timeouts']} timeouts")
        if m.get("agent_errors"):
            notes.append(f"{m['agent_errors']} API errors")
        if m.get("agent_text_calls"):
            notes.append(f"{m['agent_text_calls']} tool calls written as text")
        if m.get("agent_unauth"):
            notes.append(f"sent instead of drafting x{m['agent_unauth']}")
        inj = m.get("agent_inj")
        pk = m.get("agent_pass_k")
        rws.append([nm(m), f(m.get("agent_score"), 0), (f(pk, 0) + "%") if pk is not None else "-",
                    (f(m["agent_valid_pct"], 0) + "%") if m.get("agent_valid_pct") is not None else "-", f(m.get("agent_steps"), 0),
                    f(m.get("agent_sec"), 0), f(m.get("agent_tokens"), 0),
                    ("yes" if (m.get("agent_parallel") or 0) >= 2 else "no") if m.get("agent_parallel") is not None else "-",
                    f"{inj[0]}/{inj[1]}" if inj else "-", "; ".join(notes)])
    L += [table(hdr, rws), ""]
    cats = list(A.CATS)
    L += ["### By category", "", "Two tasks per category, percent of points.", "",
          table(["Model"] + [A.CATS[c] for c in cats],
                [[m["name"]] + [f(m["agent_cats"][c] * 100, 0) if m["agent_cats"].get(c) is not None else "-" for c in cats] for m in agm]), ""]
    rows = []
    for t in A.TASKS:
        runs = [x for m in agm for x in m["agent_tasks"].get(t["id"], [])]
        if not runs:
            continue
        rate = mean([1.0 if x.get("solved") else 0.0 for x in runs])
        have = [m for m in agm if m["agent_tasks"].get(t["id"])]
        top = [m["name"] for m in sorted(have, key=lambda m: m.get("agent_sec") or 0) if task_mean(m["agent_tasks"][t["id"]]) >= 0.999][:3]
        rows.append((rate, [t["id"].replace("ag_", ""), A.CATS[t["cat"]], t["desc"], f"{rate * 100:.0f}%", ", ".join(top) or "-"]))
    rows.sort(key=lambda x: x[0])
    L += ["### The tasks, hardest first", "", table(["Task", "Category", "What it checks", "Runs solved (all models)", "Perfect every run (fastest first)"],
                                                   [r for _, r in rows]), ""]
    cm = [m for m in ok if m.get("cache_cold")]
    if cm:
        L += ["## Prompt cache (agent follow-up speed)", "",
              "Agent apps resend a long system prompt and tool list on every turn. If the engine keeps that prefix cached, only the new part "
              "is read and the next turn starts fast; if not, every turn pays the full wait again.", "",
              table(["Model", "Engine", "Prefix (tokens)", "Cold start (s)", "Same prefix, new question (s)", "Next turn of the chat (s)", "Speedup"],
                    [[m["name"], m.get("engine") or "-", f"{int(m['cache_tokens']):,}" if m.get("cache_tokens") else "-", f(m.get("cache_cold"), 1), f(m.get("cache_warm"), 2),
                      f(m.get("cache_cont"), 2), (f(m.get("cache_speedup"), 0) + "x") if m.get("cache_speedup") else "-"]
                     for m in sorted(cm, key=lambda m: m.get("cache_warm") or 9e9)]), ""]
    return L


def field_cell(m):
    if not m.get("field_n"):
        return "-"
    v = f(m.get("field_score"), 0)
    n = len(A.FIELD_TASKS)
    return v if m["field_n"] >= n else f"{v} ({m['field_n']}/{n})"


def field_table(fm, K, md_links=False):
    n = len(A.FIELD_TASKS)
    cats = list(A.FIELD_CATS)
    hdr = ["Model", "Real-cases score", f"Solved all {K} runs", "Serious slips"] + [A.FIELD_CATS[c] for c in cats] + ["s per task", "Notes"]
    rws = []
    for m in fm:
        notes = []
        if m["field_n"] < n:
            notes.append(f"{m['field_n']}/{n} tasks so far")
        elif (m.get("field_k_tasks") or 0) < m["field_n"]:
            notes.append(f"repeat runs done for {m.get('field_k_tasks') or 0}/{n}")
        if m.get("field_timeouts"):
            notes.append(f"{m['field_timeouts']} timeouts")
        pk = m.get("field_pass_k")
        rws.append([nm(m), f(m.get("field_score"), 0), (f(pk, 0) + "%") if pk is not None else "-",
                    f"{m.get('field_slips', 0)}/{m.get('field_runs', 0)}"]
                   + [f(m["field_cats"][c] * 100, 0) if m["field_cats"].get(c) is not None else "-" for c in cats]
                   + [f(m.get("field_sec"), 0), "; ".join(notes)])
    return table(hdr, rws)


def field_task_rows(fm):
    rows = []
    for t in A.FIELD_TASKS:
        runs = [x for m in fm for x in m["field_tasks"].get(t["id"], [])]
        if not runs:
            continue
        rate = mean([1.0 if x.get("solved") else 0.0 for x in runs])
        slips = sum(1 for x in runs if A.slipped(t["id"], x.get("note")))
        have = [m for m in fm if m["field_tasks"].get(t["id"])]
        top = [m["name"] for m in sorted(have, key=lambda m: m.get("field_sec") or 0) if task_mean(m["field_tasks"][t["id"]]) >= 0.999][:3]
        rows.append((rate, t, slips, len(runs), top))
    rows.sort(key=lambda x: x[0])
    return rows


def field_sections(ok, K):
    fm = sorted([m for m in ok if m.get("field_n")], key=lambda m: -(m.get("field_score") or 0))
    if not fm:
        return []
    n = len(A.FIELD_TASKS)
    L = ["## Real-world failure tier", "",
         f"{n} more agent tasks, each rebuilt from a mistake a real assistant made on this owner's setup (names, clients and numbers "
         "changed). They test judgment more than tool mechanics: checking before an irreversible step, staying inside the request, "
         "trusting the live system over old notes, and not claiming more than was done. Scored separately, so the main agent "
         f"leaderboard stays comparable. 'Serious slips' counts runs that {A.SLIP_TEXT}.", "",
         field_table(fm, K), ""]
    rows = field_task_rows(fm)
    L += ["### The real cases, hardest first", "",
          table(["Task", "Category", "What went wrong originally", "What it checks", "Runs solved", "Serious slips", "Perfect every run (fastest first)"],
                [[t["id"].replace("fx_", ""), A.FIELD_CATS[t["cat"]], t["origin"], t["desc"], f"{rate * 100:.0f}%", f"{sl}/{nr}", ", ".join(top) or "-"]
                 for rate, t, sl, nr, top in rows]), ""]
    return L


# models built for a narrow job, so their general scores are not a verdict on them
SPECIAL = {"d1-3b": "a decision model (it scores given options), not built for chat or agents",
           "hy-memory": "a memory-writing component of a larger system, not a general assistant"}
# models that fail the source check (named publisher, paper/site/GitHub, real adoption): measured, but not recommended
PROVENANCE = {"spark-x2.6": "comes from an anonymous publisher (\"SparkLLM Team\", no named company). The scores are our own "
                            "measurements, but we don't recommend it for real use"}


def provenance_note(m):
    return next((v for k, v in PROVENANCE.items() if k in m["key"].lower()), None)


def special_note(m):
    return next((v for k, v in SPECIAL.items() if k in m["key"].lower()), None)


# keys that are another build of the same base model, grouped with it in the quantization table (with a label)
SAME_WEIGHTS = {"underdog-saluki-27b": "qwen3.8-27b",  # Saluki is Conway Research's 2-bit (IQ2) quant of Qwen3.8-27B
                "ternary-bonsai-2-27b": "qwen3.8-27b",  # Bonsai 2 27B: PrismML's ternary (~1.7-bit) build of Qwen3.8-27B
                "gemma-4-12b-it-qat": "gemma-4-12b-it"}  # Google's quantization-aware-trained 4-bit Gemma 4 12B
SAME_LABEL = {"saluki": "Saluki", "bonsai-2": "Bonsai 2", "-qat": "QAT"}


def quant_family(m):
    base = re.sub(r" \((GGUF|MLX)\)$", "", m.get("base_name") or m["name"])
    return next((v for k, v in SAME_WEIGHTS.items() if k in m["key"].lower()), base)


def quant_sections(ok, K):
    """Same model at different quantizations (and formats), side by side."""
    groups = {}
    for m in ok:
        groups.setdefault(quant_family(m), []).append(m)
    groups = {k: sorted(v, key=lambda m: m.get("size_gb") or 0) for k, v in groups.items() if len(v) > 1}
    if not groups:
        return []
    rows = []
    for name, xs in sorted(groups.items()):
        for m in xs:
            fmt = "MLX" if "MLX" in (m.get("format") or "") else "GGUF"
            tag = next((v for k, v in SAME_LABEL.items() if k in m["key"].lower()), "")
            label = name + (f" ({tag})" if tag else "")
            rows.append([f"**{label}**" if m["key"] in PICKED else label, fmt, m.get("quant") or "-", f(m.get("size_gb")), f(m.get("mem_gb")), f(m.get("tg_d8k")), qcell(m), agent_cell(m),
                         field_cell(m), f"{m.get('field_slips', 0)}/{m.get('field_runs', 0)}" if m.get("field_n") else "-", f(m.get("agent_sec"), 0)])
    sds = [m["agent_run_sd"] for m in ok if m.get("agent_run_sd") is not None]
    noise = (f" The same model's agent score moves about {med(sds):.0f} point{'s' if round(med(sds)) != 1 else ''} between passes, "
             "and single tasks flip more than that, so read small gaps as ties.") if len(sds) >= 3 else ""
    return ["## Quantization: same model, different sizes", "",
            "Each quant is tested as its own model with the same tests and settings. Bigger quants need more memory and generate more slowly; "
            "the question is whether they also decide better." + noise, "",
            table(["Model", "Format", "Quant", "File (GB)", "Memory @64k (GB)", "Gen tok/s @8k", "Quality %", "Agent %", "Real cases %", "Serious slips", "s per task"], rows), ""]


def findings(ok, agents):
    """A few data-driven one-liners for the README, the Reddit draft and the hub note."""
    out = []
    if agents:
        rates = []
        for t in A.TASKS:
            runs = [x for m in agents for x in m["agent_tasks"].get(t["id"], [])]
            if runs:
                rates.append((mean([1.0 if x.get("solved") else 0.0 for x in runs]), t))
        rates.sort(key=lambda x: x[0])
        if rates:
            out.append("Hardest agent tasks: " + "; ".join(f"{t['id'].replace('ag_', '')} ({r * 100:.0f}% of runs solved)" for r, t in rates[:3]) + ".")
        inj = [m for m in agents if m.get("agent_inj") and m["agent_inj"][0] < m["agent_inj"][1]]
        if any(m.get("agent_inj") for m in agents):
            out.append(f"{len(inj)} of {len(agents)} models obeyed a prompt injection hidden in an email at least once"
                       + (f" ({', '.join(m['name'] for m in inj)})." if inj else "."))
        tc = [m for m in agents if m.get("agent_text_calls")]
        if tc:
            out.append(f"{len(tc)} model(s) sometimes wrote tool calls as plain text that LM Studio couldn't parse: {', '.join(m['name'] for m in tc[:4])}.")
        b, fast = agents[0], best(agents, lambda m: m.get("agent_score"), lambda m: (m.get("agent_sec") or 999) <= 30)
        if fast and fast is not b:
            out.append(f"Best agent score: {b['name']} ({b['agent_score']:.0f}%, {f(b.get('agent_sec'), 0)} s per task). "
                       f"Best under 30 s per task: {fast['name']} ({fast['agent_score']:.0f}%, {f(fast.get('agent_sec'), 0)} s).")
    fm = sorted([m for m in ok if (m.get("field_n") or 0) >= len(A.FIELD_TASKS)], key=lambda m: -(m.get("field_score") or 0))
    if len(fm) >= 2:
        clean = [m for m in fm if not m.get("field_slips")]
        x = (f"Real-world failure tier: best {fm[0]['name']} ({fm[0]['field_score']:.0f}%); {len(fm) - len(clean)} of {len(fm)} models made at least one "
             f"serious slip (something hard to undo, or a false claim)")
        rows = field_task_rows(fm)
        if rows:
            x += f"; hardest case: {rows[0][1]['id'].replace('fx_', '')} ({rows[0][0] * 100:.0f}% of runs solved)"
        gap = [m for m in fm if m.get("agent_score") is not None and m["agent_score"] - m["field_score"] >= 15]
        if gap:
            x += f". Biggest drops from the main agent score: " + ", ".join(f"{m['name']} ({m['agent_score']:.0f} to {m['field_score']:.0f})" for m in sorted(gap, key=lambda m: m["field_score"] - m["agent_score"])[:3])
        out.append(x + ".")
    for fmt in ("GGUF", "MLX"):
        xs = [m["cache_speedup"] for m in ok if m.get("cache_speedup") and (m.get("format") or "").startswith(fmt)]
        if len(xs) >= 2:
            out.append(f"Prompt cache: on {fmt} models a follow-up turn on the same long agent prefix started {med(xs):.0f}x faster than the cold start (median of {len(xs)} models).")
    return out


HARNESS_FILES = ["bench.py", "lsclient.py", "tasks.py", "agent.py", "report.py", "run.sh", "README.md"]


HOME_RE = re.compile(r"(/|\\\\/)Users(?:/|\\\\/)[^/\\\\\"]+|(?:/|\\\\/)private(?:/|\\\\/)var(?:/|\\\\/)folders(?:/|\\\\/)[^/\\\\\"]+(?:(?:/|\\\\/)[^/\\\\\"]+(?:/|\\\\/)T)?")  # home folder and per-user temp folder, plain or JSON-escaped
SCRUB = []  # (text, replacement) pairs applied to the published raw data


def package_harness(cdir):
    """Zip the harness (code + assets, no data or token) so others can run the same tests."""
    out = os.path.join(cdir, "LocalLlmBench-harness.zip")
    srcs = [os.path.join(HERE, f) for f in HARNESS_FILES if os.path.exists(os.path.join(HERE, f))]
    adir = os.path.join(HERE, "assets")
    for dp, _, fs in os.walk(adir):
        srcs += [os.path.join(dp, x) for x in fs if not x.startswith(".")]
    newest = max(os.path.getmtime(x) for x in srcs)
    if os.path.exists(out) and os.path.getmtime(out) >= newest:
        mirror_harness(out)
        return out
    with zipfile.ZipFile(out + ".tmp", "w", zipfile.ZIP_DEFLATED) as z:
        for x in srcs:
            z.write(x, os.path.join("LocalLlmBench", os.path.relpath(x, HERE)))
        z.writestr("LocalLlmBench/config.json", json.dumps({"night_start_hour": 0, "night_end_hour": 24, "ctx_standard": 65536}, indent=1))
    os.replace(out + ".tmp", out)
    mirror_harness(out)
    return out


def mirror_harness(out):
    """config harness_mirror: also keep a copy of the zip there (e.g. the synced vault), so another Mac runs the same code.
    harness_mirror_b64: the same zip as base64 text in a Markdown note (md5 of the zip on line 1), for syncs that skip .zip."""
    import base64, hashlib
    for key in ("harness_mirror", "harness_mirror_b64"):
        dst = os.path.expanduser(str((CFG or {}).get(key) or ""))
        if not dst or not os.path.isdir(os.path.dirname(dst)):
            continue
        try:
            if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(out):
                continue
            if key == "harness_mirror":
                shutil.copy2(out, dst)
            else:
                raw = open(out, "rb").read()
                b64 = base64.b64encode(raw).decode()
                lines = [b64[i:i + 76] for i in range(0, len(b64), 76)]
                open(dst + ".tmp", "w").write(hashlib.md5(raw).hexdigest() + "\n\n```text\n" + "\n".join(lines) + "\n```\n")
                os.replace(dst + ".tmp", dst)
        except OSError:
            pass


# ---------------------------------------------------------------- build

LOWER_IS_BETTER = ("s per task", "ttft", "mem", "memory", "s to ", "seconds", "tokens per task", "slips", "median steps", "cold", "warm", "cont",
                   "reasoning", "time", "latency", "ms")
HIGHER_IS_BETTER = ("%", "tok/s", "t/s", "quality", "agent", "real", "solved", "valid", "score", "max ctx", "prefill", "gen", "recall", "injection",
                    "chunks", "needle", "speedup", "writing", "coding", "vision", "parallel", "context")
NOT_RANKED = ("model", "format", "quant", "notes", "engine", "file", "size", "category", "task", "what", "tier", "frontier", "pareto")


def _num(cell):
    """Numeric value of a table cell for ranking: '12.3', '94%', '3/30', '152x', '1,138', '99*', '256k'; None if not a number."""
    t = str(cell).strip().replace("**", "").replace(",", "").rstrip("*").strip()
    if not t or t == "-":
        return None
    if "/" in t and all(x.strip().isdigit() for x in t.split("/")[:2]):
        a, b = t.split("/")[:2]
        return int(a) / int(b) if int(b) else None
    m = re.match(r"^-?\d+(\.\d+)?", t)
    if not m:
        return None
    v = float(m.group(0))
    return v * 1000 if t[m.end():m.end() + 1] == "k" else v


def _best_cols(headers):
    out = {}
    for i, h in enumerate(headers):
        hl = h.lower()
        if any(k in hl for k in NOT_RANKED) and not any(k in hl for k in ("score", "%")):
            continue
        if any(k in hl for k in LOWER_IS_BETTER):
            out[i] = min
        elif any(k in hl for k in HIGHER_IS_BETTER):
            out[i] = max
    return out


def table(headers, rows, rank=True):
    """Markdown table. With rank, the best value in each numeric column is bold (lower is better for times, memory and slips)."""
    rows = [[str(c) for c in r] for r in rows]
    if rank and len(rows) > 1:
        for i, fn in _best_cols(headers).items():
            vals = [(_num(r[i]) if i < len(r) else None) for r in rows]
            good = [v for v in vals if v is not None]
            if len(good) < 2:
                continue
            b = fn(good)
            if sum(1 for v in good if v == b) > 3:  # a tie across many rows says nothing
                continue
            for r, v in zip(rows, vals):
                if v is not None and v == b and not r[i].startswith("**"):
                    r[i] = f"**{r[i]}**"
    # numbers right-aligned (GitHub and Obsidian honour the ---: marker), text left
    aligns = []
    for i in range(len(headers)):
        cells = [r[i] for r in rows if i < len(r) and r[i].strip() not in ("", "-")]
        aligns.append("---:" if cells and sum(_num(c) is not None for c in cells) >= 0.8 * len(cells) and i > 0 else "---")
    if rank and len(rows) > 3:
        # per row: the metric where this model ranks best against the others (if it is in the top third there) is bold too
        cols = _best_cols(headers)
        ranks = {}
        for i, fn in cols.items():
            vals = [(_num(r[i]) if i < len(r) else None) for r in rows]
            good = sorted({v for v in vals if v is not None}, reverse=(fn is max))
            if len(good) < 4:
                continue
            ranks[i] = {v: k / (len(good) - 1) for k, v in enumerate(good)}  # 0 = best, 1 = worst
        for r in rows:
            best = min(((ranks[i][_num(r[i])], i) for i in ranks if i < len(r) and _num(r[i]) in ranks[i]), default=None)
            if best and best[0] <= 0.34 and not r[best[1]].startswith("**"):
                r[best[1]] = f"**{r[best[1]]}**"
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(aligns) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def build():
    rows = load()
    env, M = aggregate(rows)
    os.makedirs(os.path.join(OUT, "models"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "charts"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "community"), exist_ok=True)
    chat = [m for m in M.values() if not m.get("embedding")]
    ok = [m for m in chat if not m.get("load_failed")]
    emb = [m for m in M.values() if m.get("embedding")]
    par = pareto(ok)
    ok.sort(key=lambda m: -(m.get("quality") or 0))
    agents = sorted([m for m in ok if m.get("agent_n", 0) >= 12], key=lambda m: -(m.get("agent_score") or 0))
    K = CFG.get("agent_trials", 3)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    inv = {}
    try:
        inv = json.load(open(os.path.join(HERE, "data", "inventory.json")))
    except Exception:
        pass

    # picks
    picks = []
    bq = max([m.get("quality") or 0 for m in ok] or [0])
    PICKED.clear()
    def pick(label, m, why):
        if m:
            PICKED.setdefault(m["key"], []).append(label)
        if m:
            picks.append(f"- **{label}:** [[{slug(m['name'])}|{m['name']}]] ({why(m)})")
    def overall(m):  # text quality and agent score count equally once the agent suite has run
        if m.get("quality") is None:
            return None
        return (m["quality"] + m["agent_score"]) / 2 if m.get("agent_n", 0) >= 12 else m["quality"]
    pick("Best overall", best(ok, overall, lambda m: m.get("n_cats", 0) >= 5),
         lambda m: f"{f(m['quality'],0)}% quality" + (f", {f(m['agent_score'],0)}% agent" if m.get("agent_score") is not None else "") + f", {f(m.get('tg_d8k'))} tok/s")
    def everyday(m):  # quality, agent and real-cases scores count equally; the base quality set alone is nearly saturated
        xs = [m.get("quality"), m.get("agent_score") if m.get("agent_n", 0) >= 12 else None,
              m.get("field_score") if (m.get("field_n") or 0) >= len(A.FIELD_TASKS) else None]
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None
    pick("Best everyday pick (40+ tok/s)", best(ok, everyday, lambda m: m.get("n_cats", 0) >= 5 and (m.get("tg_d8k") or 0) >= 40),
         lambda m: f"{f(m['quality'],0)}% quality" + (f", {f(m['agent_score'],0)}% agent" if m.get("agent_score") is not None else "")
         + (f", {f(m['field_score'],0)}% real cases" if m.get("field_score") is not None else "") + f", {f(m.get('tg_d8k'))} tok/s")
    pick("Spanish writing", best(ok, lambda m: m["cats"].get("writing")), lambda m: f"writing {f(m['cats']['writing']*100,0)}%")
    pick("Coding", best(ok, lambda m: (m["cats"].get("coding") or 0) * 100 + (m.get("tg_d8k") or 0) / 100 if m["cats"].get("coding") is not None else None), lambda m: f"coding {f(m['cats']['coding']*100,0)}%")
    if agents:
        def agent_why(m):
            w = f"agent score {f(m['agent_score'], 0)}%, {f(m.get('agent_sec'), 0)} s per task"
            return w + (f", {f(m['agent_pass_k'], 0)}% of tasks solved in all {K} runs" if m.get("agent_pass_k") is not None else "")
        pick("Agents (Hermes, OpenCode, n8n style)", best(agents, lambda m: (m.get("agent_score") or 0) + (m.get("agent_pass_k") or 0) / 1000), agent_why)
        top_agent = best(agents, lambda m: (m.get("agent_score") or 0) + (m.get("agent_pass_k") or 0) / 1000)
        snappy = best(agents, lambda m: m.get("agent_score"), lambda m: (m.get("agent_sec") or 999) <= 30)
        if snappy is not top_agent:
            pick("Snappy agent (30 s or less per task)", snappy, agent_why)
        fdone = [m for m in agents if (m.get("field_n") or 0) >= len(A.FIELD_TASKS)]
        if len(fdone) >= 3:
            pick("Unattended agent (real-world cases)", best(fdone, lambda m: (m.get("field_score") or 0) - 5 * (m.get("field_slips") or 0) + (m.get("field_pass_k") or 0) / 1000),
                 lambda m: f"{f(m['field_score'], 0)}% on the real cases, {m.get('field_slips', 0)} serious slips in {m.get('field_runs', 0)} runs, {f(m.get('field_sec'), 0)} s per task")
        pick("Coding agent", best(agents, lambda m: (m["agent_cats"]["coding"] * 100 - (m.get("agent_sec") or 0) / 1000) if m["agent_cats"].get("coding") is not None else None,
                                  lambda m: (m.get("agent_score") or 0) >= 95),
             lambda m: f"fixed the repos in {f(m['agent_cats']['coding'] * 100, 0)}% of runs, {f(m.get('agent_sec'), 0)} s per task")
    else:
        pick("Agents and tools", best(ok, lambda m: ((m["cats"].get("tools") or 0) + (m["cats"].get("json") or 0)) if m["cats"].get("tools") is not None else None), lambda m: f"tools {f(m['cats']['tools']*100,0)}%, JSON {f((m['cats'].get('json') or 0)*100,0)}%")
    pick("Fastest that's still good", best(ok, lambda m: m.get("tg_d8k"), lambda m: (m.get("quality") or 0) >= 0.75 * bq), lambda m: f"{f(m['tg_d8k'])} tok/s, {f(m.get('quality'),0)}% quality")
    def _recall(m):
        xs = [m.get(k) for k in ("needle_d8k", "needle_d32k", "needle_d60k", "dmax_needle")]
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if len(xs) >= 3 else None
    pick("Long documents", best(ok, lambda m: (_recall(m) * 1000 + (m.get("pp_d32k") or 0) / 10) if _recall(m) is not None else None,
                                lambda m: (m.get("quality") or 0) >= 0.8 * bq and (m.get("dmax_tokens") or 0) >= 100000),
         lambda m: f"found the hidden fact up to {ctxk(m.get('dmax_tokens'))}, reads {f(m.get('pp_d32k'),0)} tok/s at 32k")
    pick("Vision and OCR", best(ok, lambda m: m.get("vision")), lambda m: f"vision {f(m['vision'],0)}%")
    # fits a 24 GB MacBook Air with room for apps, and generates at 25+ tok/s here (the Air's bandwidth is about a quarter of this Mac's)
    air = lambda m: (m.get("mem_gb") or 99) <= 14 and (m.get("tg_d8k") or 0) >= 25
    pick("MacBook Air (24 GB), chat and writing", best(ok, everyday, air),
         lambda m: f"{f(m.get('mem_gb'))} GB, {f(m.get('quality'),0)}% quality, {f(m.get('tg_d8k'))} tok/s here")
    pick("MacBook Air (24 GB), agents and coding", best(agents, lambda m: (m.get("agent_score") or 0) + (m.get("field_score") or 0) + (m.get("agent_pass_k") or 0) / 1000, air),
         lambda m: f"{f(m.get('mem_gb'))} GB, {f(m.get('agent_score'),0)}% agent, {f(m.get('field_score'),0)}% real cases, {f(m.get('tg_d8k'))} tok/s here")
    if emb:
        e = best(emb, lambda m: m.get("recall1"))
        if e:
            picks.append(f"- **Vault search embeddings:** {e['name']} (recall@1 {f((e.get('recall1') or 0)*100,0)}%, {f(e.get('chunks_per_s'))} chunks/s)")

    has_chart = scatter_svg(ok, par, os.path.join(OUT, "charts", "speed_vs_quality.svg"))
    has_agent_chart = agent_svg(ok, os.path.join(OUT, "charts", "agent_vs_time.svg"))

    # status
    excluded = [m for m in inv.get("models", []) if m.get("excluded")]
    tested_keys = set(M)
    pending = [m for m in inv.get("models", []) if not m.get("excluded") and m["key"] not in tested_keys]
    status = ([f"**Action needed:** {NOTE}"] if NOTE else []) + [f"Last update: {now}. Models with results: {len(M)}."]
    if pending:
        status.append("Not tested yet: " + ", ".join(m["key"] for m in pending) + ".")
    if inv.get("downloading"):
        status.append("Still downloading: " + ", ".join(inv["downloading"]) + ".")
    if any(m.get("judge_pending") for m in ok):
        status.append("Writing and summary scores are auto-checks only until the local judges finish.")

    L = []
    L += ["---", "tags: [llm, benchmark]", f"updated: {now}", f"machine: {env.get('chip','')} {env.get('ram_gb','')} GB",
          f"lmstudio: {env.get('lmstudio_app','')}", "---", "", "# Local LLM benchmark", ""]
    L += ["> [!info] Status", *["> " + s for s in status], ""]
    L += ["## Picks", "", *(picks or ["- Not enough results yet."]), ""]
    fnd = findings(ok, agents)
    if fnd:
        L += ["## Findings", "", *[f"- {x}" for x in fnd], ""]
    if has_chart:
        L += ["## Quality vs speed", "", "![[charts/speed_vs_quality.svg]]", ""]
    if has_agent_chart:
        L += ["## Agent score vs time", "", "![[charts/agent_vs_time.svg]]", ""]
    # headline table
    hdr = ["Model", "Format", "Memory needed @64k (GB)", "Gen tok/s @8k", "Time to first token @8k (s)", "Quality %", "Agent %", "Real cases %", "Vision %", "Max context loaded", "Frontier"]
    rws = []
    for m in ok:
        rws.append([lk(m), (m.get("format") or ""), f(m.get("mem_gb")),
                    f(m.get("tg_d8k")), f(m.get("ttft_d8k"), 2), qcell(m), agent_cell(m), field_cell(m), f(m.get("vision"), 0),
                    ctxk(m.get("ctx_max_loaded") or m.get("ctx_std")), "yes" if m["key"] in par else ""])
    L += ["## Summary", "", table(hdr, rws), "",
          "**Bold** model = one of the picks above; **bold** number = best in its column, or the metric where that model stands out most. Quality is the average of the text categories below (0 to 100). Agent is the multi-step tool-use suite; Real cases is the tier rebuilt from real agent failures (both below). Frontier means no other model here is both faster and better."]
    if any(m.get("capped") for m in ok):
        L += ["", CAPPED_NOTE]
    sp = [m for m in ok if special_note(m)]
    if sp:
        L += ["", "Specialized models, so read their scores with that in mind: " + "; ".join(f"{m['name']} is {special_note(m)}" for m in sp) + "."]
    pv = [m for m in ok if provenance_note(m)]
    if pv:
        L += ["", "Source caveat: " + "; ".join(f"{m['name']} {provenance_note(m)}" for m in pv) + "."]
    L += [""]
    L += agent_sections(ok, agents, K)
    L += field_sections(ok, K)
    L += quant_sections(ok, K)
    # speed detail
    hdr = ["Model", "Load first (s)", "Load repeat (s)", "TTFT short (s)", "Gen short", "Spread %", "Prefill @8k", "Gen @8k", "Prefill @32k", "Gen @32k", "TTFT @60k (s)", "Gen @60k", "Watts", "Tok/J"]
    rws = [[m["name"], f(m.get("load_first_s")), f(m.get("load_repeat_s")), f(m.get("ttft_short"), 2), f(m.get("tg_short")),
            f(m.get("tg_short_spread"), 0), f(m.get("pp_d8k"), 0), f(m.get("tg_d8k")), f(m.get("pp_d32k"), 0), f(m.get("tg_d32k")),
            f(m.get("ttft_d60k")), f(m.get("tg_d60k")), f(m.get("watts")), f(m.get("tok_per_j"), 2)] for m in ok]
    if not any(m.get("watts") for m in ok):
        hdr, rws = hdr[:-2], [r_[:-2] for r_ in rws]
    L += ["## Speed", "", "Tokens per second unless noted. Prefill is how fast the prompt is read; Gen is how fast the answer is written. Medians of repeated runs at a 64k context.", "", table(hdr, rws), ""]
    # context
    hdr = ["Model", "Native max", "Standard ctx", "Mem @std (GB)", "Largest that loaded", "Mem @max (GB)", "Filled to", "TTFT at fill (s)", "Gen at fill", "Recall 8k / 32k / 60k / max"]
    rws = []
    for m in ok:
        rec = " / ".join("-" if m.get(k) is None else f"{m[k]*100:.0f}%" for k in ("needle_d8k", "needle_d32k", "needle_d60k", "dmax_needle"))
        rws.append([m["name"], ctxk(m.get("native_ctx")), ctxk(m.get("ctx_std")), f(m.get("mem_gb")), ctxk(m.get("ctx_max_loaded")),
                    f(m.get("mem_max_gb")), ctxk(m.get("dmax_tokens")), f(m.get("dmax_ttft"), 0), f(m.get("dmax_tg")), rec])
    L += ["## Context", "", "Every model runs at a 64k agent context (65,536 tokens), then at the largest context that loads, filled up to 128k. Recall checks whether the model finds a fact hidden in the middle of the text. Memory is LM Studio's estimate; for MLX models that estimate doesn't grow with context, so expect more than shown when you actually fill a long context.", "", table(hdr, rws), ""]
    # thinking
    hdr = ["Model", "Median seconds per task", "Avg reasoning tokens", "Time to first answer token (s)", "Ran out of tokens before answering", "Agent: tokens per task", "Agent: s per task"]
    rws = [[m["name"], f(m.get("sec_per_task")), f(m.get("reason_tokens"), 0), f(m.get("ttfa"), 1), m.get("capped") or "", f(m.get("agent_tokens"), 0), f(m.get("agent_sec"), 0)] for m in ok]
    L += ["## Thinking cost", "", "Thinking models look fast in tokens per second but spend many tokens before answering. This is the wait you actually feel. "
          "Quality tasks allow 16,384 output tokens (8,192 for vision); a task that runs out before answering scores 0.", "", table(hdr, rws), ""]
    # quality detail
    hdr = ["Model"] + [CAT_LABEL[c] for c in CATS] + ["Vision"]
    rws = [[m["name"]] + [f(m["cats"].get(c) * 100, 0) if m["cats"].get(c) is not None else "-" for c in CATS + ["vision"]] for m in ok]
    L += ["## Quality by category", "", "Percent of points. Most tasks are graded automatically (rules, JSON checks, unit tests, exact answers). Writing and summaries also get scores from two local judge models of a different family than the model being graded.", "", table(hdr, rws), ""]
    # pros/cons
    L += ["## Pros, cons and recommended use", ""]
    for m in ok + [m for m in chat if m.get("load_failed")]:
        pros, cons, uses = pros_cons(m, ok)
        L += [f"### {m['name']}", f"- **Use for:** {', '.join(uses)}"]
        L += [f"- **Pro:** {p}" for p in pros] + [f"- **Con:** {c}" for c in cons] + [""]
    if emb:
        hdr = ["Model", "Memory (GB)", "Chunks/s", "~Tokens/s", "Dims", "Recall@1"]
        rws = [[m["name"], f(m.get("mem_gb")), f(m.get("chunks_per_s")), f(m.get("tok_s"), 0), m.get("dims") or "-", f((m.get("recall1") or 0) * 100, 0) + "%"] for m in emb]
        L += ["## Embeddings", "", table(hdr, rws), "", "Recall@1 is a simple check: find the right Quijote paragraph from one of its sentences among 200.", ""]
    nt = [m for m in chat if m.get("load_failed")]
    if excluded or nt:
        L += ["## Not tested", ""]
        for m in nt:
            L.append(f"- {m['name']}: failed to load in LM Studio.")
        for m in excluded:
            k = (m["key"] + " " + str(m.get("path") or "")).lower()
            if any(x in k for x in ("dspark", "dflash")):
                why = "speculative-decoding draft file"
            elif "mmproj" in k or str(m.get("arch", "")).lower() == "clip":
                why = "vision projector file (part of another model, not a model itself)"
            else:
                why = "image or video generator (needs ComfyUI)"
            L.append(f"- {m['key']}: {why}.")
        L.append("")
    L += ["## Method", "",
          f"- Machine: {env.get('hw') or env.get('chip')} , {env.get('ram_gb')} GB, macOS {env.get('macos')}. GPU memory limit left at default ({env.get('gpu_wired_limit_mb') or 'default'}).",
          f"- LM Studio {env.get('lmstudio_app','?')}; engines: " + ", ".join(f"{e.get('name','')} {e.get('version','')}" for e in (env.get('engines') or []) if isinstance(e, dict)) + ".",
          "- Each model is loaded alone, tested, then unloaded. Loads set the context explicitly; GGUF loads use flash attention; everything else is LM Studio's default.",
          f"- Speed: 1 warmup + {CFG.get('short_reps',5)} runs with a short prompt, {CFG.get('reps_8k',3)} at 8k, {CFG.get('reps_32k',2)} at 32k, {CFG.get('reps_60k',1)} at 60k; temperature 0; output capped at 256/512 tokens. Every prompt starts with a unique tag so nothing is served from cache.",
          "- Prompts and long-context filler come from Don Quijote (public domain). Vision tests use synthetic images.",
          "- Timings are measured client-side from the streaming API on the same machine. Memory is LM Studio's own estimate for loading the model at that context (`lms load --estimate-only`, weights plus KV cache). Measured system memory was also logged, but it undercounts because model files are memory-mapped, so it's only a fallback.",
          "- Runs only at night with the Mac idle; any test interrupted by keyboard or mouse activity is redone.",
          "- Load times: 'first' is the first load in the run, 'repeat' is a reload right after. Neither is a true cold load because macOS may already have the file cached.",
          "- Quality: 26 tasks (21 text, 5 vision), mostly auto-graded. Small sample, so treat differences under ~10 points as ties.",
          f"- Agent suite: {len(A.TASKS)} multi-step tool-use tasks against a simulated workspace (notes, contacts, calendar, email, files, rooms, sales data, two small Python repos). The harness runs the tool loop through LM Studio's OpenAI-compatible API (native tool calls, non-streaming), returns simulated tool results, and grades the final state of the workspace plus the final answer. Each task runs {K} times at temperature {CFG.get('agent_temperature', 0.6)} with different seeds; the score is the mean, and 'solved in all runs' counts tasks where every run scored at least {int(A.PASS * 100)}%. Limits: {CFG.get('agent_max_tokens', 8192)} output tokens per turn, 30 minutes per run.",
          f"- Real-world failure tier: {len(A.FIELD_TASKS)} more tasks in the same harness, each rebuilt (anonymized) from a mistake a real assistant made: stamping invoices without cross-checking, moving data points while 'tidying' a chart, trusting stale setup notes, saving into an archived folder, singling out a student in a group message, overwriting another session's log entry, inventing references, deleting without a confirmed backup, claiming all files were fixed, and moving files before the user approved the plan. Same runs and limits as the agent suite; reported separately.",
          "- Prefix cache: a ~12k-token agent prefix (system prompt + 40 tool definitions + project notes) is sent cold, then again with a different question, then as the next turn of the same chat. A big speedup means the engine reuses the cached prefix instead of re-reading it.",
          "- Engines: models were tested over several days, and LM Studio updated its runtimes in between; the engine each model ran on is in its note.", ""]
    swapped = [m for m in ok if (m.get("dmax_swap") or 0) > 500]
    if swapped:
        L += ["> [!note] Swap at max context"] + [f"> - {m['name']}: used {m['dmax_swap']/1000:.1f} GB of swap during the max-context fill, so its numbers at that length are a bit slower than RAM-only." for m in swapped] + [""]
    flags = [m for m in ok if m.get("disturbed") or m.get("errors") or m.get("early_stop") or m.get("capped")]
    if flags:
        def note(m):
            x = f"> - {m['name']}: {m.get('errors', 0)} failed tests, {m.get('disturbed', 0)} disturbed, {m.get('early_stop', 0)} speed runs ended before the token cap"
            return x + (f", {m['capped']} quality tasks ran out of tokens before answering" if m.get("capped") else "")
        L += ["> [!warning] Data notes"] + [note(m) for m in flags] + [""]
    L += ["Raw data: `~/LocalLlmBench/data/results.jsonl`. Community package: [[community/README|community/README]].", ""]
    open(os.path.join(OUT, "LLM Benchmark.md"), "w", encoding="utf-8").write("\n".join(L))

    # per-model notes
    for m in ok + [m for m in chat if m.get("load_failed")]:
        pros, cons, uses = pros_cons(m, ok)
        props = {"model_key": m["key"], "format": m.get("format"), "family": m.get("family"), "params": m.get("params"),
                 "size_gb": m.get("size_gb"), "mem_gb_64k": m.get("mem_gb"), "gen_tps_8k": round(m["tg_d8k"], 1) if m.get("tg_d8k") else None,
                 "gen_tps_short": round(m["tg_short"], 1) if m.get("tg_short") else None, "ttft_8k_s": round(m["ttft_d8k"], 2) if m.get("ttft_d8k") else None,
                 "quality_pct": round(m["quality"]) if m.get("quality") is not None else None, "vision_pct": round(m["vision"]) if m.get("vision") is not None else None,
                 "max_ctx_loaded": m.get("ctx_max_loaded") or m.get("ctx_std"), "frontier": m["key"] in par,
                 "agent_pct": round(m["agent_score"]) if m.get("agent_score") is not None else None,
                 "agent_solved_every_run_pct": round(m["agent_pass_k"]) if m.get("agent_pass_k") is not None else None,
                 "agent_s_per_task": round(m["agent_sec"]) if m.get("agent_sec") else None,
                 "real_cases_pct": round(m["field_score"]) if m.get("field_score") is not None else None,
                 "real_cases_slips": m.get("field_slips") if m.get("field_n") else None,
                 "prompt_cache_speedup": round(m["cache_speedup"], 1) if m.get("cache_speedup") else None,
                 "engine": m.get("engine"), "use_for": uses, "updated": now}
        fm = ["---"] + [f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in props.items() if v is not None] + ["tags: [llm-model]", "---"]
        body = [f"# {m['name']}", "", "Back to [[LLM Benchmark]]", "", "## Verdict", f"- **Use for:** {', '.join(uses)}"]
        if special_note(m):
            body.insert(4, f"> [!note] This is {special_note(m)}, so the general tests here don't measure what it was made for.\n")
        if provenance_note(m):
            body.insert(4, f"> [!warning] This model {provenance_note(m)}.\n")
        body += [f"- **Pro:** {p}" for p in pros] + [f"- **Con:** {c}" for c in cons]
        trs = m.get("tasks", {})
        if trs:
            body += ["", "## Task results", "", table(["Task", "Category", "Score", "Judge /10", "Seconds", "Note"],
                     [[t, r["cat"], f(r["score"] * 100, 0), f(r["judge"], 1), f(r["t_total"]), (r.get("note") or "").replace("|", "/")[:60]] for t, r in sorted(trs.items())])]
            body += ["", "## Sample answers", ""]
            agt = m.get("agent_tasks") or {}
            if agt:
                ins = len(body) - 2
                arows = []
                for t in A.TASKS:
                    xs = agt.get(t["id"])
                    if not xs:
                        continue
                    arows.append([t["id"].replace("ag_", ""), A.CATS[t["cat"]], f"{sum(1 for x in xs if x.get('solved'))}/{len(xs)}",
                                  f(task_mean(xs) * 100, 0), f(med([x.get("t_total") for x in xs]), 0), f(med([x.get("steps") for x in xs]), 0),
                                  str(xs[-1].get("skipped") or xs[-1].get("note") or "").replace("|", "/")[:70]])
                body[ins:ins] = ["", "## Agent tasks", "", table(["Task", "Category", "Solved", "Score", "Median s", "Steps", "Last run"], arows), ""]
            fxt = m.get("field_tasks") or {}
            if fxt:
                ins = len(body) - 2
                frows = []
                for t in A.FIELD_TASKS:
                    xs = fxt.get(t["id"])
                    if not xs:
                        continue
                    frows.append([t["id"].replace("fx_", ""), A.FIELD_CATS[t["cat"]], f"{sum(1 for x in xs if x.get('solved'))}/{len(xs)}",
                                  f(task_mean(xs) * 100, 0), str(sum(1 for x in xs if A.slipped(t["id"], x.get("note")))),
                                  f(med([x.get("t_total") for x in xs]), 0), str(xs[-1].get("skipped") or xs[-1].get("note") or "").replace("|", "/")[:70]])
                body[ins:ins] = ["", "## Real-world cases", "", table(["Task", "Category", "Solved", "Score", "Slips", "Median s", "Last run"], frows), ""]
            for t in ("write_email_es", "write_post_es", "summary_es"):
                if trs.get(t, {}).get("answer"):
                    body += [f"> [!quote]- {t}", *["> " + l for l in trs[t]["answer"].strip().splitlines()], ""]
        open(os.path.join(OUT, "models", slug(m["name"]) + ".md"), "w", encoding="utf-8").write("\n".join(fm + [""] + body) + "\n")

    # base file (Obsidian Bases)
    base = """filters:
  and:
    - file.hasTag("llm-model")
views:
  - type: table
    name: Models
    order:
      - file.name
      - format
      - mem_gb_64k
      - gen_tps_8k
      - ttft_8k_s
      - quality_pct
      - agent_pct
      - agent_s_per_task
      - real_cases_pct
      - vision_pct
      - max_ctx_loaded
      - frontier
      - use_for
    sort:
      - property: quality_pct
        direction: DESC
"""
    open(os.path.join(OUT, "Benchmark.base"), "w").write(base)

    # community package
    cdir = os.path.join(OUT, "community")
    cols = ["key", "name", "format", "engine", "size_gb", "mem_gb", "ctx_std", "load_first_s", "ttft_short", "tg_short", "pp_d8k", "tg_d8k",
            "ttft_d8k", "pp_d32k", "tg_d32k", "ttft_d60k", "tg_d60k", "ctx_max_loaded", "dmax_tokens", "dmax_tg", "needle_d8k", "needle_d32k",
            "needle_d60k", "dmax_needle", "quality", "vision", "sec_per_task", "reason_tokens", "img_overhead_s",
            "agent_score", "agent_pass_k", "agent_pass1", "agent_valid_pct", "agent_steps", "agent_sec", "agent_tokens", "agent_parallel",
            "agent_text_calls", "agent_timeouts", "field_score", "field_pass_k", "field_slips", "field_runs", "field_sec",
            "cache_tokens", "cache_cold", "cache_warm", "cache_cont", "cache_speedup"]
    with open(os.path.join(cdir, "results.csv"), "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(cols)
        for m in ok:
            w.writerow([m.get(c) if not isinstance(m.get(c), float) else round(m[c], 3) for c in cols])
    with open(os.path.join(cdir, "agent_tasks.csv"), "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["model", "tier", "task", "category", "runs", "solved", "mean_score", "serious_slips", "median_seconds", "median_steps", "median_completion_tokens"])
        for m in ok:
            for tier, d in (("main", m.get("agent_tasks") or {}), ("real_cases", m.get("field_tasks") or {})):
                for tid, xs in sorted(d.items()):
                    w.writerow([m["key"], tier, tid, A.TASK_BY_ID[tid]["cat"], len(xs), sum(1 for x in xs if x.get("solved")), round(task_mean(xs), 3),
                                sum(1 for x in xs if A.slipped(tid, x.get("note"))) if tier == "real_cases" else "",
                                med([x.get("t_total") for x in xs]), med([x.get("steps") for x in xs]), med([x.get("completion_tokens") for x in xs])])
    try:
        with open(RESULTS, encoding="utf-8") as src, open(os.path.join(cdir, "results.jsonl"), "w", encoding="utf-8") as dst:
            for line in src:  # crash tracebacks can carry the user's home folder
                line = HOME_RE.sub("~", line)
                for a, b in SCRUB:
                    line = line.replace(a, b)
                dst.write(line)
        for name, flag in (("speed_vs_quality.svg", has_chart), ("agent_vs_time.svg", has_agent_chart)):
            if flag:
                shutil.copy(os.path.join(OUT, "charts", name), os.path.join(cdir, name))
    except Exception:
        pass
    try:
        package_harness(cdir)
    except Exception:
        pass
    strip = lambda p: re.sub(r"\[\[[^|\]]*\|([^\]]*)\]\]", r"\1", p)
    t_agent = table(["Model", "Format", "Agent %", f"Solved all {K} runs", "s per task", "Valid tool calls", "Ignored injection"],
                    [[nm(m), m.get("format") or "", f(m.get("agent_score"), 0), (f(m["agent_pass_k"], 0) + "%") if m.get("agent_pass_k") is not None else "-",
                      f(m.get("agent_sec"), 0), (f(m["agent_valid_pct"], 0) + "%") if m.get("agent_valid_pct") is not None else "-",
                      "{}/{}".format(*m["agent_inj"]) if m.get("agent_inj") else "-"] for m in agents]) if agents else ""
    tbl = table(["Model", "Format", "Mem @64k GB", "Gen t/s @8k", "TTFT @8k s", "Prefill t/s @32k", "Max ctx loaded", "Quality %", "Agent %", "Real cases %"],
                [[nm(m), (m.get("format") or ""), f(m.get("mem_gb")), f(m.get("tg_d8k")), f(m.get("ttft_d8k"), 2),
                  f(m.get("pp_d32k"), 0), ctxk(m.get("ctx_max_loaded") or m.get("ctx_std")), qcell(m), agent_cell(m), field_cell(m)] for m in ok])
    fms = sorted([m for m in ok if m.get("field_n")], key=lambda m: -(m.get("field_score") or 0))
    t_field = field_table(fms, K) if fms else ""
    t_ftasks = table(["Task", "Category", "What went wrong originally", "What it checks"],
                     [[t["id"].replace("fx_", ""), A.FIELD_CATS[t["cat"]], t["origin"], t["desc"]] for t in A.FIELD_TASKS])
    if any(m.get("capped") for m in ok):
        tbl += "\n\n" + CAPPED_NOTE
    if agents and len(agents) < len(ok):
        tbl += ("\n\nModels with '-' under Agent were deleted before the agent suite was added, so they only have speed and quality numbers.")
    t_tasks = table(["Task", "Category", "What it checks"], [[t["id"].replace("ag_", ""), A.CATS[t["cat"]], t["desc"]] for t in A.TASKS])
    cm = sorted([m for m in ok if m.get("cache_cold")], key=lambda m: m.get("cache_warm") or 9e9)
    t_cache = table(["Model", "Engine", "Cold start s", "Next turn s", "Speedup"],
                    [[m["name"], m.get("engine") or "-", f(m.get("cache_cold"), 1), f(m.get("cache_cont"), 2),
                      (f(m.get("cache_speedup"), 0) + "x") if m.get("cache_speedup") else "-"] for m in cm]) if cm else ""
    engines = sorted({m.get("engine") for m in ok if m.get("engine")})
    fnd = findings(ok, agents)
    # the short version: what to run for what, and a compact leaderboard sorted by the hardest tier
    ranked = sorted([m for m in ok if (m.get("field_n") or 0) >= len(A.FIELD_TASKS)], key=lambda m: -(m.get("field_score") or 0))
    t_top = table(["Model", "Mem @64k GB", "Real cases %", "Agent %", "Quality %", "tok/s"],
                  [[nm(m), f(m.get("mem_gb")), f(m.get("field_score"), 0), f(m.get("agent_score"), 0), qcell(m), f(m.get("tg_d8k"), 0)]
                   for m in ranked[:12]]) if len(ranked) >= 5 else ""
    quick = [strip(p) for p in picks[:8]]
    readme = [f"# M1 Max 64 GB: {len(ok)} local models in LM Studio, {len(agents)} of them also tested as agents ({datetime.now():%B %Y})", "",
              f"Hardware: {env.get('chip', 'Apple M1 Max')} (32-core GPU), {env.get('ram_gb', 64)} GB, macOS {env.get('macos', '')}. "
              f"LM Studio {env.get('lmstudio_app', '')}; engines used: {', '.join(engines) or '-'}.", ""]
    if quick:
        readme += ["## What to run", "", "Picks from the tables below. Memory is at a 64k context; real cases is the tier rebuilt from real agent failures and the one that separates models most.", ""] + quick + [""]
    if t_top:
        readme += ["## Top 12 by real cases", "", t_top, "", "Scores are out of 100. Differences under about 3 points are noise.", ""]
    if fnd:
        readme += ["## Findings", ""] + [f"- {x}" for x in fnd] + [""]
    if agents:
        readme += ["## Agent leaderboard", "", f"{len(A.TASKS)} multi-step tool-use tasks, {K} runs each, graded on the resulting state "
                   "(what got created, sent or fixed), not on how the answer sounds.", "", t_agent, "", "![](agent_vs_time.svg)", ""]
    if t_field:
        readme += ["## Real-world failure tier", "", f"{len(A.FIELD_TASKS)} more tasks, each rebuilt (anonymized) from a mistake a real assistant made "
                   "on my setup: checking before irreversible steps, staying in scope, trusting the live system over old notes, and not "
                   f"claiming more than was done. 'Serious slips' = runs that {A.SLIP_TEXT}.", "", t_field, ""]
    qs = quant_sections(ok, K)
    if qs:
        readme += qs
    readme += ["## Speed, memory and context", "", tbl, "", "![](speed_vs_quality.svg)", ""]
    if t_cache:
        readme += ["## Prompt cache", "", "Agent apps resend the same long prefix every turn (system prompt + tool list). "
                   "Cold start = first time a ~12k-token prefix is read; next turn = the same chat one turn later.", "", t_cache, ""]
    readme += ["## The agent tasks", "", "A simulated workspace (notes, contacts, calendar, email inbox, files, meeting rooms, sales data, two small "
               "Python repos). Tool results are deterministic, so every run can be replayed and re-graded. Half the tasks are in Spanish.", "", t_tasks, "",
               "## The real-world cases", "", "Same harness, a drive of documents, slides and an assistant config added to the workspace.", "", t_ftasks, "",
               "## How it was measured", "",
               "- One model loaded at a time with an explicit 64k context (65,536). GGUF with flash attention, default KV cache.",
               "- Speed: short prompt (1 warmup + 5 runs), 8k (3 runs), 32k (2), 60k (1). Medians. Temperature 0, output capped at 256 or 512 tokens, unique prefix per run so nothing is cached.",
               "- Then each model is loaded at the largest context that fits (native max, stepping down) and filled up to 128k tokens once. Recall: a 6-digit code hidden halfway through Don Quijote text.",
               "- Quality: 26 short tasks (instruction following, JSON extraction, single tool calls, Python with unit tests, reasoning, Spanish writing and summaries, 5 vision tasks). Mostly auto-graded; writing is also scored by two local judge models from other families.",
               f"- Agents: native tool calls through LM Studio's OpenAI-compatible API, temperature {CFG.get('agent_temperature', 0.6)}, {CFG.get('agent_max_tokens', 8192)} output tokens per turn, "
               f"30 minutes per run, {K} runs per task with different seeds. A run counts as solved at {int(A.PASS * 100)}% or more of its checks.",
               "- Memory is LM Studio's estimate for loading at that context (weights plus KV cache). The GPU wired limit was left at the macOS default.",
               "- Ran unattended with the Mac idle; any test interrupted by user input was repeated.",
               "- It's one person's test set on one machine. Differences of a few points are noise; read it as tiers.", "",
               "## Run it on your own Mac", "",
               "`LocalLlmBench-harness.zip` has everything (Python standard library only, no pip installs):", "",
               "1. In LM Studio, start the local server (Developer tab). If it asks for an API token, create one and paste it into `lmstudio_token.txt`.",
               "2. Unzip, then `cd LocalLlmBench && ./run.sh --smoke` for a 2-minute check on your smallest model.",
               "3. `./run.sh --background --now` runs every downloaded model (it's resumable; `tail -f data/run.log` to watch).",
               "4. Your results end up in `report/community/` in this same format. Post them; numbers from other Macs are the most useful part.", "",
               "## Files", "", "- `results.csv`: one row per model.", "- `agent_tasks.csv`: one row per model and agent task.",
               "- `agent_tasks.csv` has a `tier` column: `main` or `real_cases`.",
               "- `results.jsonl`: every individual run, including answers and full agent traces.", "- `LocalLlmBench-harness.zip`: the harness.", ""]
    open(os.path.join(cdir, "README.md"), "w", encoding="utf-8").write("\n".join(readme))
    title = (f"I tested {len(agents)} local models on an M1 Max 64GB as agents ({len(A.TASKS)} multi-step tool tasks"
             + (f" + {len(A.FIELD_TASKS)} rebuilt from real agent failures" if fms else "") + f", x{K} runs), plus speed and context for {len(ok)}"
             if agents else f"M1 Max 64GB benchmark: {len(ok)} models in LM Studio at 64k context, plus max context and a small quality test")
    repo = CFG.get("repo_url") or "[link to repo]"
    post = [f"Title: {title}", "", "Body:", "",
            "tok/s doesn't tell you whether a model can run an agent, so I built an unattended harness and ran my whole LM Studio library through it "
            f"on an M1 Max 64 GB: speed and memory at the same 64k context, max context, a quality set, {len(A.TASKS)} multi-step tool tasks in a "
            f"simulated workspace, and {len(A.FIELD_TASKS)} tasks rebuilt from mistakes my own assistants actually made (stamping invoices without "
            "checking the order, moving data while 'tidying' a chart, deleting when the backup check timed out, acting before the plan was approved). "
            f"Every agent task ran {K} times and is graded on what happened in the workspace, not on how the answer sounds.", ""]
    post += ["**What to run**", ""] + quick + [""]
    if t_top:
        post += ["**Top 12 by the real-world tier** (memory at 64k context; scores out of 100, under ~3 points apart is noise; bold model = a pick above, bold number = best in column or that model's standout metric)", "", t_top, ""]
    if fnd:
        post += ["**What stood out**", ""] + [f"- {x}" for x in fnd] + [""]
    post += ["**Caveats**", "",
             "- One person's tasks on one machine. Half the tasks are in Spanish because that's what I use.",
             "- Writing quality is scored by local judge models from another family; everything else is deterministic code checking the final state.",
             "- The base quality set is nearly saturated for good models; the real-world tier is where they separate.", "",
             f"Full tables ({len(ok)} models: speed, memory, context, quant comparisons, prompt cache, every agent task), the harness and every trace: {repo}", "",
             "If you have a different Mac, run it and open a PR with your results folder; numbers from other machines are the most useful part. "
             "Model requests welcome if they fit in 64 GB."]
    open(os.path.join(cdir, "reddit_post.md"), "w", encoding="utf-8").write("\n".join(post) + "\n")
    return True


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        RESULTS = sys.argv[1]
    build()
