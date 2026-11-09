from huggingface_hub import HfApi

from .card import results_table
from .common import BASELINES, MODELS, banner, dataset_repo, read_json, repo_id, results_path, wpath, write_json


def collect(cfg):
    out = {}
    for v in BASELINES:
        r = read_json(results_path(cfg, f"base_{v}", "full"))
        if r:
            out[f"base_{v}"] = r
    for m in MODELS:
        r = read_json(results_path(cfg, m, "full"))
        if r:
            out[m] = r
    return out


def plots(cfg, res):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for tag, r in res.items():
        x, y = r["summary"]["mean_tokens"], r["summary"]["acc"] * 100
        marker = "s" if tag.startswith("base_") else "o"
        ax.scatter([x], [y], s=70, marker=marker)
        label = tag.replace("base_", "base ") if tag.startswith("base_") else tag.upper()
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("Mean generated tokens (MATH500 and GSM8K)")
    ax.set_ylabel("Accuracy % (MATH500 and GSM8K)")
    ax.set_title("ShortThink: accuracy vs tokens")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(wpath(cfg, "report", "pareto.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for tag, r in res.items():
        lv = r["benchmarks"].get("math500", {}).get("by_level", {})
        if not lv or tag in ("base_concise", "base_budget"):
            continue
        ks = sorted(lv, key=int)
        ax.plot([int(k) for k in ks], [lv[k]["mean_tokens"] for k in ks], marker="o", label=tag)
    ax.set_xlabel("MATH difficulty level")
    ax.set_ylabel("Mean generated tokens")
    ax.set_title("Tokens by difficulty (MATH500)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(wpath(cfg, "report", "levels.png"), dpi=140)
    plt.close(fig)


def run_report(cfg):
    from .hub import login, publish

    login()
    res = collect(cfg)
    if not res:
        raise RuntimeError("no full results to report")
    plots(cfg, res)
    summary = {tag: {**r["summary"], "cost_per_1k_queries_usd": r.get("cost_per_1k_queries_usd"), "latency": r.get("latency")} for tag, r in res.items()}
    write_json(wpath(cfg, "report", "summary.json"), summary)
    html = results_table(cfg)
    wpath(cfg, "report", "table.html").write_text(html)
    api = HfApi()
    repo = dataset_repo(cfg)
    for f in ["summary.json", "table.html", "pareto.png", "levels.png"]:
        api.upload_file(path_or_fileobj=str(wpath(cfg, "report", f)), path_in_repo=f"report/{f}", repo_id=repo, repo_type="dataset", commit_message=f"Add report {f}")
    for m in MODELS:
        if read_json(results_path(cfg, m, "full")):
            publish(cfg, m)
    banner("report done\n" + "\n".join(f"{k}: acc {v['acc']:.3f} tokens {v['mean_tokens']:.0f} tpc {v['tokens_per_correct']:.0f}" for k, v in summary.items()))
    for m in MODELS:
        print(f"https://huggingface.co/{repo_id(cfg, m)}", flush=True)
