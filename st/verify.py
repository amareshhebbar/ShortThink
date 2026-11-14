import json
import shutil

from huggingface_hub import HfApi, snapshot_download

from .common import banner, read_json, repo_id, results_path, wpath, write_json


def verify_model(cfg, name):
    from transformers import AutoConfig, AutoTokenizer

    from .evaluate import quick_eval
    from .hub import login

    login()
    repo = repo_id(cfg, name)
    banner(f"verifying {repo} straight from the hub")
    files = HfApi().list_repo_files(repo)
    checks = {
        "config.json": "config.json" in files,
        "generation_config.json": "generation_config.json" in files,
        "tokenizer": "tokenizer.json" in files or "tokenizer_config.json" in files,
        "safetensors": any(f.endswith(".safetensors") and "/" not in f for f in files),
        "README.md": "README.md" in files,
    }
    for k, v in checks.items():
        print(f"  {k}: {'OK' if v else 'MISSING'}", flush=True)
    if not all(checks.values()):
        raise SystemExit(f"verify {name}: repo is missing files")
    local = wpath(cfg, "verify", name)
    if local.exists():
        shutil.rmtree(local)
    snapshot_download(repo, local_dir=str(local), ignore_patterns=["adapter/*", "training/*", "assets/*", "eval/*"], force_download=True)
    conf = AutoConfig.from_pretrained(str(local))
    tok = AutoTokenizer.from_pretrained(str(local))
    print(f"  model_type {conf.model_type}, eos {tok.eos_token}, vocab {len(tok)}", flush=True)
    res = quick_eval(cfg, str(local), name)
    base = read_json(results_path(cfg, "base_think", "quick"))
    mine = res["summary"]
    fmt = min(b["format_rate"] for b in res["benchmarks"].values())
    base_b = base.get("benchmarks", {}) if base else {}
    need = {}
    for k, b in res["benchmarks"].items():
        t = cfg["eval"]["gate_format_rate"]
        if k in base_b:
            t = min(t, base_b[k]["format_rate"] - cfg["eval"]["gate_acc_drop"])
        need[k] = round(t, 4)
    gates = {"format_rate": all(b["format_rate"] >= need[k] for k, b in res["benchmarks"].items())}
    if base:
        gates["accuracy"] = mine["acc"] >= base["summary"]["acc"] - cfg["eval"]["gate_acc_drop"]
    report = {
        "repo": repo,
        "quick": mine,
        "base_quick": base["summary"] if base else None,
        "format_rate": fmt,
        "format_needed": need,
        "gates": gates,
        "passed": all(gates.values()),
    }
    if base:
        report["token_change_vs_base"] = mine["mean_tokens"] / base["summary"]["mean_tokens"] - 1
    write_json(wpath(cfg, "results", f"{name}_verify.json"), report)
    banner(f"verify {name}\n" + json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(f"verify {name} FAILED, see {wpath(cfg, 'results', name + '_verify.json')}")
    print(f"verify {name} PASSED", flush=True)
