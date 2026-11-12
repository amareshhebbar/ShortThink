import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

from st.common import MODELS, dataset_repo, get_meta, read_json, repo_id, results_path, wpath, write_json

MAX_GSM8K_DROP = 1.0


def change(points):
    if abs(points) < 0.05:
        return "same"
    return f"{abs(points):.1f} points {'higher' if points > 0 else 'lower'}"


def row(base, r):
    bs, rs = base["summary"], r["summary"]
    bl, rl = base.get("latency", {}).get("p50_s"), r.get("latency", {}).get("p50_s")
    return {
        "fewer_tokens_pct": (1 - rs["mean_tokens"] / bs["mean_tokens"]) * 100,
        "speedup": bl / rl if bl and rl else None,
        "gsm8k_acc": r["benchmarks"]["gsm8k"]["acc"] * 100,
        "gsm8k_change": (r["benchmarks"]["gsm8k"]["acc"] - base["benchmarks"]["gsm8k"]["acc"]) * 100,
        "math500_acc": r["benchmarks"]["math500"]["acc"] * 100,
        "math500_change": (r["benchmarks"]["math500"]["acc"] - base["benchmarks"]["math500"]["acc"]) * 100,
        "mean_tokens": rs["mean_tokens"],
        "tokens_per_correct": rs["tokens_per_correct"],
        "latency_p50_s": rl,
    }


def build(cfg):
    base = read_json(results_path(cfg, "base_think", "full"))
    if not base:
        raise SystemExit("base_think results missing, run the baseline step first")
    rows = {}
    for m in MODELS:
        r = read_json(results_path(cfg, m, "full"))
        if r:
            rows[m] = row(base, r)
    if not rows:
        raise SystemExit("no model results yet, run the pipeline first")
    keep = {m: v for m, v in rows.items() if v["gsm8k_change"] >= -MAX_GSM8K_DROP} or rows
    best = max(keep, key=lambda m: keep[m]["fewer_tokens_pct"])
    return base, rows, best


def sentence(cfg, best, v):
    speed = f" and answer {v['speedup']:.1f}x faster" if v["speedup"] else ""
    return (
        f"ShortThink {best.upper()} makes {cfg['base_label']} reason in {v['fewer_tokens_pct']:.0f}% fewer tokens{speed}, "
        f"with GSM8K accuracy {change(v['gsm8k_change'])} and MATH500 accuracy {change(v['math500_change'])} compared with the base model."
    ).replace("accuracy same", "accuracy unchanged")


def markdown(cfg, base, rows, best):
    b = base["summary"]
    coll = get_meta(cfg, "collection_slug")
    lines = [
        f"# ShortThink results ({cfg['base_label']})",
        "",
        f"**Headline:** {sentence(cfg, best, rows[best])}",
        "",
        f"Baseline: `{cfg['base_model']}` in thinking mode, {b['mean_tokens']:.0f} tokens per answer on average, "
        f"GSM8K {base['benchmarks']['gsm8k']['acc'] * 100:.1f}%, MATH500 {base['benchmarks']['math500']['acc'] * 100:.1f}%. "
        f"Every row uses the same prompts, the same sampling settings and the same math_verify grading, with {cfg['eval']['samples']} samples per problem. "
        "Speed is the median time per answer at batch size 1 on vLLM.",
        "",
        "| Model | Fewer tokens | Faster | GSM8K accuracy | GSM8K vs base | MATH500 accuracy | MATH500 vs base | Tokens per correct answer |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m, v in sorted(rows.items(), key=lambda kv: -kv[1]["fewer_tokens_pct"]):
        name = f"[{m.upper()}](https://huggingface.co/{repo_id(cfg, m)})"
        if m == best:
            name = f"**{name}**"
        sp = f"{v['speedup']:.2f}x" if v["speedup"] else ""
        lines.append(
            f"| {name} | {v['fewer_tokens_pct']:.1f}% | {sp} | {v['gsm8k_acc']:.1f}% | {change(v['gsm8k_change'])} | {v['math500_acc']:.1f}% | {change(v['math500_change'])} | {v['tokens_per_correct']:.0f} |"
        )
    lines += [
        "",
        f"The headline model is the one with the largest token reduction whose GSM8K accuracy stays within {MAX_GSM8K_DROP:.0f} point of the base model.",
        "",
        "## Check it yourself",
        "",
        f"* Every model card has the full table, charts and training settings: [collection](https://huggingface.co/collections/{coll})" if coll else "* Every model card has the full table, charts and training settings.",
        f"* Raw evaluation files and sample answers are in each model repo under `eval/`, for example [{best.upper()} eval](https://huggingface.co/{repo_id(cfg, best)}/tree/main/eval).",
        f"* Training data: [{dataset_repo(cfg)}](https://huggingface.co/datasets/{dataset_repo(cfg)}).",
        "* Code to reproduce every number: [github.com/amareshhebbar/ShortThink](https://github.com/amareshhebbar/ShortThink).",
        "",
        f"Generated {time.strftime('%Y-%m-%d')} from the pipeline's own result files.",
        "",
    ]
    return "\n".join(lines)


def main():
    from huggingface_hub import HfApi

    from st.common import load_cfg
    from st.hub import login

    cfg = load_cfg()
    base, rows, best = build(cfg)
    for m, v in sorted(rows.items(), key=lambda kv: -kv[1]["fewer_tokens_pct"]):
        sp = f"{v['speedup']:.2f}x faster" if v["speedup"] else "speed n/a"
        print(f"{m:6s} {v['fewer_tokens_pct']:5.1f}% fewer tokens  {sp}  GSM8K {change(v['gsm8k_change'])}  MATH500 {change(v['math500_change'])}")
    md = markdown(cfg, base, rows, best)
    out = wpath(cfg, "report", "HEADLINE.md")
    out.write_text(md)
    write_json(wpath(cfg, "report", "headline.json"), {"best": best, "sentence": sentence(cfg, best, rows[best]), "rows": rows})
    print("\n" + sentence(cfg, best, rows[best]) + "\n")
    if "--local" in sys.argv:
        print(f"written to {out}, not uploaded")
        return
    login()
    api = HfApi()
    ds = dataset_repo(cfg)
    for f in ["HEADLINE.md", "headline.json"]:
        api.upload_file(path_or_fileobj=str(wpath(cfg, "report", f)), path_in_repo=f"report/{f}", repo_id=ds, repo_type="dataset", commit_message=f"Add {f}")
        api.upload_file(path_or_fileobj=str(wpath(cfg, "report", f)), path_in_repo=f, repo_id=repo_id(cfg, best), commit_message=f"Add {f}")
    print("proof links:")
    print(f"  https://huggingface.co/{repo_id(cfg, best)}/blob/main/HEADLINE.md")
    print(f"  https://huggingface.co/datasets/{ds}/blob/main/report/HEADLINE.md")
    print(f"  https://huggingface.co/{repo_id(cfg, best)}")


if __name__ == "__main__":
    main()
