import glob
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

from .common import BASELINES, banner, read_json, resolve_model, results_path, wpath, write_json, write_jsonl
from .data import chat_prompt, eval_sets, subset
from .grade import grade_many

MAIN = ["math500", "gsm8k"]


def metrics(rows):
    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    toks = sum(r["ntok"] for r in rows)
    return {
        "n": n,
        "acc": correct / max(1, n),
        "mean_tokens": toks / max(1, n),
        "tokens_per_correct": toks / max(1, correct),
        "trunc_rate": sum(not r["closed"] for r in rows) / max(1, n),
        "format_rate": sum(r["boxed"] for r in rows) / max(1, n),
    }


def by_level(rows):
    groups = defaultdict(list)
    for r in rows:
        if r.get("level"):
            groups[str(r["level"])].append(r)
    return {k: metrics(v) for k, v in sorted(groups.items())}


def run_suite(cfg, eng, tag, variant, mode):
    e = cfg["eval"]
    sets = eval_sets(cfg, mode)
    thinking = variant != "nothink"
    concise = variant == "concise"
    out = {"tag": tag, "model": eng.model, "variant": variant, "mode": mode, "benchmarks": {}}
    total_tok, total_time = 0, 0.0
    samples_dump = []
    for bname, items in sets.items():
        n = 1 if mode == "quick" else (e["aime_samples"] if bname == "aime24" else e["samples"])
        prompts = [chat_prompt(eng.tok, it["question"], thinking=thinking, concise=concise) for it in items]
        banner(f"[{tag} | {variant} | {mode}] {bname}: {len(items)} problems x {n} samples")
        if variant == "budget":
            res, dt = eng.budget(prompts, n, e["budget_tokens"], e["budget_tail"])
        else:
            res, dt = eng.run(prompts, eng.params(n, e["max_tokens"], thinking))
        texts, golds, meta = [], [], []
        for it, outs in zip(items, res):
            for text, ntok, fr in outs:
                texts.append(text)
                golds.append(it["gold"])
                meta.append((it, ntok, fr))
        grades = grade_many(texts, golds)
        rows = []
        for (it, ntok, fr), (ok, closed, boxed), text in zip(meta, grades, texts):
            rows.append(
                {
                    "id": it["id"],
                    "level": it["level"],
                    "ntok": ntok,
                    "finish": fr,
                    "correct": ok,
                    "closed": bool(closed and fr != "length"),
                    "boxed": boxed,
                }
            )
            if len(samples_dump) < 300 and len([s for s in samples_dump if s["bench"] == bname]) < 15:
                samples_dump.append({"bench": bname, "id": it["id"], "gold": it["gold"], "correct": ok, "ntok": ntok, "text": text})
        m = metrics(rows)
        if bname == "math500":
            m["by_level"] = by_level(rows)
        m["wall_s"] = dt
        out["benchmarks"][bname] = m
        total_tok += sum(r["ntok"] for r in rows)
        total_time += dt
        print(json.dumps({k: v for k, v in m.items() if k != "by_level"}, indent=1), flush=True)
    main = [out["benchmarks"][b] for b in MAIN if b in out["benchmarks"]]
    out["summary"] = {
        "acc": sum(b["acc"] for b in main) / len(main),
        "mean_tokens": sum(b["mean_tokens"] for b in main) / len(main),
        "tokens_per_correct": sum(b["tokens_per_correct"] for b in main) / len(main),
    }
    out["throughput_tok_s"] = total_tok / max(1e-9, total_time)
    out["cost_per_1k_queries_usd"] = (
        out["summary"]["mean_tokens"] * 1000 / max(1e-9, out["throughput_tok_s"]) / 3600 * cfg["gpu_cost_per_hour"]
    )
    if mode == "full" and e.get("latency_n"):
        lat_items = subset(sets["math500"], e["latency_n"], cfg["seed"] + 1)
        lat_prompts = [chat_prompt(eng.tok, it["question"], thinking=thinking, concise=concise) for it in lat_items]
        banner(f"[{tag} | {variant}] latency probe at batch size 1 on {len(lat_prompts)} problems")
        out["latency"] = eng.latency(
            lat_prompts, e["max_tokens"], thinking, budget=e["budget_tokens"] if variant == "budget" else None, tail=e["budget_tail"]
        )
    out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(results_path(cfg, tag, mode), out)
    write_jsonl(wpath(cfg, "results", f"{tag}_{mode}_samples.jsonl"), samples_dump)
    banner(f"[{tag}] summary {json.dumps(out['summary'])}")
    return out


def run_baseline(cfg):
    from .engine import Engine

    eng = Engine(cfg, cfg["base_model"])
    run_suite(cfg, eng, "base_think", "think", "quick")
    for v in BASELINES:
        tag = f"base_{v}"
        if read_json(results_path(cfg, tag, "full")):
            print(f"skip {tag}, already evaluated", flush=True)
            continue
        run_suite(cfg, eng, tag, v, "full")


def quick_eval(cfg, model_path, tag):
    from .engine import Engine

    eng = Engine(cfg, model_path)
    return run_suite(cfg, eng, tag, "think", "quick")


def run_model_eval(cfg, name):
    from .engine import Engine

    eng = Engine(cfg, resolve_model(cfg, name))
    return run_suite(cfg, eng, name, "think", "full")


def run_mmlu(cfg, name):
    if not cfg["eval"].get("mmlu"):
        print("mmlu disabled in config", flush=True)
        return
    tag = "base" if name == "base" else name
    path = resolve_model(cfg, name)
    out_dir = wpath(cfg, "results", f"mmlu_{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    margs = (
        f"pretrained={path},dtype=bfloat16,gpu_memory_utilization={cfg['gen']['gpu_memory_utilization']},"
        f"max_model_len={cfg['eval']['mmlu_max_model_len']},seed={cfg['seed']}"
    )
    lm_eval_bin = os.path.join(os.path.dirname(sys.executable), "lm_eval")
    cmd = [lm_eval_bin, "--model", "vllm", "--model_args", margs, "--tasks", "mmlu", "--batch_size", "auto", "--output_path", str(out_dir)]
    banner("running " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    files = sorted(glob.glob(str(out_dir / "**" / "results_*.json"), recursive=True), key=os.path.getmtime)
    if not files:
        raise RuntimeError("lm_eval produced no results file")
    res = json.loads(open(files[-1]).read())["results"]["mmlu"]
    acc = res.get("acc,none", res.get("acc"))
    write_json(wpath(cfg, "results", f"{tag}_mmlu.json"), {"tag": tag, "mmlu_acc": acc, "raw": res})
    banner(f"[{tag}] MMLU acc {acc:.4f}")
