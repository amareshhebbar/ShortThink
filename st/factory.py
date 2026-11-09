import statistics
from collections import defaultdict
from pathlib import Path

from .common import banner, dataset_repo, read_jsonl, wpath, write_json, write_jsonl
from .data import chat_prompt, train_questions
from .grade import grade_many


def generate_shards(cfg):
    from .engine import Engine

    f = cfg["factory"]
    items = train_questions(cfg)
    shards = wpath(cfg, "data", "shards")
    shards.mkdir(parents=True, exist_ok=True)
    chunk = f["chunk_questions"]
    todo = [ci for ci in range(0, len(items), chunk) if not (shards / f"shard_{ci // chunk:04d}.jsonl").exists()]
    banner(f"factory: {len(items)} questions, {len(todo)} of {(len(items) + chunk - 1) // chunk} shards left")
    if not todo:
        return
    eng = Engine(cfg, cfg["base_model"])
    sp = eng.params(f["samples"], f["max_tokens"], True)
    for ci in todo:
        part = items[ci : ci + chunk]
        prompts = [chat_prompt(eng.tok, it["question"], thinking=True) for it in part]
        banner(f"shard {ci // chunk}: {len(part)} questions x {f['samples']} samples")
        res, dt = eng.run(prompts, sp)
        texts, golds, meta = [], [], []
        for it, p, outs in zip(part, prompts, res):
            for s, (text, ntok, fr) in enumerate(outs):
                texts.append(text)
                golds.append(it["gold"])
                meta.append((it, p, s, ntok, fr))
        grades = grade_many(texts, golds)
        rows = []
        for (it, p, s, ntok, fr), (ok, closed, boxed), text in zip(meta, grades, texts):
            rows.append(
                {
                    "id": it["id"],
                    "source": it["source"],
                    "level": it["level"],
                    "question": it["question"],
                    "gold": it["gold"],
                    "prompt": p,
                    "sample": s,
                    "text": text,
                    "ntok": ntok,
                    "finish": fr,
                    "correct": bool(ok and fr == "stop"),
                }
            )
        acc = sum(r["correct"] for r in rows) / max(1, len(rows))
        mean_tok = sum(r["ntok"] for r in rows) / max(1, len(rows))
        print(f"shard {ci // chunk}: acc {acc:.3f} mean tokens {mean_tok:.0f} time {dt / 60:.1f} min", flush=True)
        write_jsonl(shards / f"shard_{ci // chunk:04d}.jsonl", rows)


def build_sets(cfg, eos):
    f = cfg["factory"]
    rows = []
    for p in sorted(Path(wpath(cfg, "data", "shards")).glob("shard_*.jsonl")):
        rows.extend(read_jsonl(p))
    by_q = defaultdict(list)
    for r in rows:
        by_q[r["id"]].append(r)
    sft, pairs, prompts = [], [], []
    all_len, short_len, pass_rates = [], [], []
    for qid, samples in by_q.items():
        k = len(samples)
        correct = [s for s in samples if s["correct"]]
        pass_rate = len(correct) / max(1, k)
        pass_rates.append(pass_rate)
        all_len.extend(s["ntok"] for s in samples)
        if not correct:
            continue
        head = samples[0]
        prompts.append(
            {"id": qid, "source": head["source"], "level": head["level"], "prompt": head["prompt"], "gold": head["gold"], "pass_rate": pass_rate}
        )
        med = statistics.median(s["ntok"] for s in correct)
        valid = sorted([s for s in correct if s["ntok"] >= f["min_len_ratio"] * med], key=lambda s: s["ntok"])
        if not valid:
            continue
        best = valid[0]
        if best["ntok"] <= f["max_sft_tokens"]:
            short_len.append(best["ntok"])
            sft.append(
                {
                    "id": qid,
                    "source": head["source"],
                    "level": head["level"],
                    "pass_rate": pass_rate,
                    "prompt": head["prompt"],
                    "completion": best["text"] + eos,
                    "ntok": best["ntok"],
                    "median_correct_ntok": med,
                }
            )
        worst = valid[-1]
        if len(valid) >= 2 and worst["ntok"] >= f["pair_min_ratio"] * best["ntok"] and worst["ntok"] <= f["max_pair_tokens"]:
            pairs.append(
                {
                    "id": qid,
                    "source": head["source"],
                    "level": head["level"],
                    "pass_rate": pass_rate,
                    "prompt": head["prompt"],
                    "chosen": best["text"] + eos,
                    "rejected": worst["text"] + eos,
                    "chosen_ntok": best["ntok"],
                    "rejected_ntok": worst["ntok"],
                }
            )
    stats = {
        "questions": len(by_q),
        "samples": len(rows),
        "sample_acc": sum(r["correct"] for r in rows) / max(1, len(rows)),
        "mean_tokens_all_samples": statistics.mean(all_len) if all_len else 0,
        "mean_tokens_shortest_correct": statistics.mean(short_len) if short_len else 0,
        "solved_questions": len(prompts),
        "sft_rows": len(sft),
        "pair_rows": len(pairs),
        "prompt_rows": len(prompts),
        "pass_rate_hist": {str(b): sum(1 for p in pass_rates if round(p * 8) == b) for b in range(9)},
    }
    if stats["mean_tokens_all_samples"]:
        stats["compression_headroom"] = 1 - stats["mean_tokens_shortest_correct"] / stats["mean_tokens_all_samples"]
    write_jsonl(wpath(cfg, "data", "sft.jsonl"), sft)
    write_jsonl(wpath(cfg, "data", "pairs.jsonl"), pairs)
    write_jsonl(wpath(cfg, "data", "prompts.jsonl"), prompts)
    write_json(wpath(cfg, "data", "stats.json"), stats)
    banner("factory stats\n" + "\n".join(f"{k}: {v}" for k, v in stats.items()))
    return stats


def dataset_card_body(cfg, stats):
    return f"""
# ShortThink Traces ({cfg['base_label']})

Self generated reasoning traces used to train the ShortThink collection. Every question from the GSM8K and MATH train splits was sampled {cfg['factory']['samples']} times from `{cfg['base_model']}` in thinking mode, and every answer was checked with `math_verify`.

## Configs

<table>
<tr><th>Config</th><th>Rows</th><th>What it holds</th></tr>
<tr><td>sft</td><td>{stats['sft_rows']}</td><td>Shortest valid correct trace per question</td></tr>
<tr><td>pairs</td><td>{stats['pair_rows']}</td><td>Chosen is the shortest correct trace, rejected is the longest correct trace (at least {cfg['factory']['pair_min_ratio']}x longer)</td></tr>
<tr><td>prompts</td><td>{stats['prompt_rows']}</td><td>Every question the base model solved at least once, with its pass rate, used for GRPO</td></tr>
</table>

## Statistics

<table>
<tr><td>Questions sampled</td><td>{stats['questions']}</td></tr>
<tr><td>Total samples</td><td>{stats['samples']}</td></tr>
<tr><td>Sample accuracy</td><td>{stats['sample_acc']:.3f}</td></tr>
<tr><td>Mean tokens over all samples</td><td>{stats['mean_tokens_all_samples']:.0f}</td></tr>
<tr><td>Mean tokens of shortest correct trace</td><td>{stats['mean_tokens_shortest_correct']:.0f}</td></tr>
<tr><td>Compression headroom</td><td>{stats.get('compression_headroom', 0) * 100:.1f}%</td></tr>
</table>

Traces shorter than {cfg['factory']['min_len_ratio']} of the median correct length for their question were dropped as likely lucky guesses. No test split was used.

Only train splits are included: `openai/gsm8k` (train) and `EleutherAI/hendrycks_math` (train). Evaluation uses MATH500, the GSM8K test split and AIME 2024.
"""


def push_dataset(cfg, stats):
    from datasets import Dataset, load_dataset
    from huggingface_hub import HfApi, hf_hub_download

    from .hub import ensure_collection, login

    login()
    api = HfApi()
    repo = dataset_repo(cfg)
    api.create_repo(repo, repo_type="dataset", private=cfg["private"], exist_ok=True)
    for name in ["sft", "pairs", "prompts"]:
        rows = read_jsonl(wpath(cfg, "data", f"{name}.jsonl"))
        banner(f"pushing {name}: {len(rows)} rows to {repo}")
        Dataset.from_list(rows).push_to_hub(repo, config_name=name, private=cfg["private"], commit_message=f"Add {name} config")
    readme = Path(hf_hub_download(repo, "README.md", repo_type="dataset", force_download=True)).read_text()
    front = ""
    if readme.startswith("---"):
        end = readme.find("\n---", 3)
        front = readme[: end + 4]
    body = dataset_card_body(cfg, stats)
    api.upload_file(path_or_fileobj=(front + "\n" + body).encode(), path_in_repo="README.md", repo_id=repo, repo_type="dataset", commit_message="Update dataset card")
    api.upload_file(path_or_fileobj=str(wpath(cfg, "data", "stats.json")), path_in_repo="stats.json", repo_id=repo, repo_type="dataset", commit_message="Add stats")
    slug = ensure_collection(cfg)
    api.add_collection_item(slug, repo, "dataset", note="Training traces shared by every model in this collection", exists_ok=True)
    banner("verifying dataset from the hub")
    for name, key in [("sft", "sft_rows"), ("pairs", "pair_rows"), ("prompts", "prompt_rows")]:
        ds = load_dataset(repo, name, split="train", download_mode="force_redownload")
        if len(ds) != stats[key]:
            raise RuntimeError(f"{name}: hub has {len(ds)} rows, expected {stats[key]}")
        print(f"{name}: {len(ds)} rows OK", flush=True)


def run_factory(cfg):
    from transformers import AutoTokenizer

    generate_shards(cfg)
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    stats = build_sets(cfg, tok.eos_token)
    if stats["sft_rows"] < 100 or stats["pair_rows"] < 50:
        raise RuntimeError(f"too little data: {stats}")
    push_dataset(cfg, stats)
