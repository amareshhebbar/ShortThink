from pathlib import Path

from .common import BASELINES, MODELS, dataset_repo, get_meta, read_json, repo_id, results_path, wpath

METHOD = {
    "sft": "Supervised finetuning on the shortest valid correct reasoning trace per question, sampled from the base model itself (rejection sampling finetuning).",
    "dpo": "Direct Preference Optimization on top of the SFT model. Chosen is the shortest correct trace, rejected is the longest correct trace for the same question, with the SFT model as the frozen reference.",
    "orpo": "Odds Ratio Preference Optimization trained from the base model in a single stage: SFT loss on the short trace plus an odds ratio penalty against the long trace. No reference model.",
    "simpo": "SimPO on top of the SFT model: a reference free, length normalized preference loss with a target margin. Length normalization removes the built in short sequence advantage of DPO, so any compression comes from the preference itself.",
    "grpo": "Online GRPO from the base model with a group relative length reward. Correct answers earn 1 plus a bonus that grows as they get shorter than their siblings; wrong answers never earn a positive reward. Uses the Dr. GRPO loss with no reward scaling, so long wrong answers are not favored.",
    "ties": "TIES merge of the two strongest ShortThink models: each task vector is trimmed to its largest entries, a sign is elected per parameter, and only agreeing entries are averaged.",
}

BASE_LABEL = {
    "think": "thinking",
    "concise": "thinking + concise prompt",
    "nothink": "no thinking",
    "budget": "budget forcing",
}


def _rows(cfg):
    rows = []
    base_mmlu = read_json(wpath(cfg, "results", "base_mmlu.json"), {}).get("mmlu_acc")
    for v in BASELINES:
        r = read_json(results_path(cfg, f"base_{v}", "full"))
        if r:
            label = f"{cfg['base_label']} ({BASE_LABEL[v]})"
            if v == "budget":
                label = f"{cfg['base_label']} (budget forcing {cfg['eval']['budget_tokens']})"
            rows.append((f"base_{v}", label, r, base_mmlu))
    for m in MODELS:
        r = read_json(results_path(cfg, m, "full"))
        if r:
            rows.append((m, f"ShortThink {m.upper()}", r, read_json(wpath(cfg, "results", f"{m}_mmlu.json"), {}).get("mmlu_acc")))
    return rows


def _pct(x):
    return "" if x is None else f"{x * 100:.1f}"


def _change(new, old):
    r = (new / old - 1) * 100
    if abs(r) < 0.05:
        return "same"
    return f"{abs(r):.1f}% {'fewer' if r < 0 else 'more'}"


def _num(x, fmt="{:.0f}"):
    return "" if x is None else fmt.format(x)


def results_table(cfg, highlight=None):
    rows = _rows(cfg)
    if not rows:
        return "<p>Evaluation results are added after the full evaluation step.</p>"
    ref = read_json(results_path(cfg, "base_think", "full"))
    ref_tok = ref["summary"]["mean_tokens"] if ref else None
    head = (
        "<tr><th>Model</th><th>MATH500 acc</th><th>MATH500 tokens</th><th>GSM8K acc</th><th>GSM8K tokens</th>"
        "<th>Tokens per correct</th><th>Token change vs thinking base</th><th>AIME24 acc</th><th>MMLU</th>"
        "<th>Latency p50 (s)</th><th>Cost per 1k queries (USD)</th></tr>"
    )
    body = []
    for tag, label, r, mmlu in rows:
        b = r["benchmarks"]
        m5, gs, ai = b.get("math500", {}), b.get("gsm8k", {}), b.get("aime24", {})
        change = ""
        if ref_tok:
            change = _change(r["summary"]["mean_tokens"], ref_tok)
        cells = [
            label,
            _pct(m5.get("acc")),
            _num(m5.get("mean_tokens")),
            _pct(gs.get("acc")),
            _num(gs.get("mean_tokens")),
            _num(r["summary"]["tokens_per_correct"]),
            change,
            _pct(ai.get("acc")),
            _pct(mmlu),
            _num(r.get("latency", {}).get("p50_s"), "{:.1f}"),
            _num(r.get("cost_per_1k_queries_usd"), "{:.3f}"),
        ]
        if tag == highlight:
            cells = [f"<b>{c}</b>" for c in cells]
        body.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    return "<table>\n" + head + "\n" + "\n".join(body) + "\n</table>"


def level_table(cfg, name):
    base = read_json(results_path(cfg, "base_think", "full"))
    mine = read_json(results_path(cfg, name, "full"))
    if not base or not mine:
        return ""
    bl, ml = base["benchmarks"]["math500"].get("by_level", {}), mine["benchmarks"]["math500"].get("by_level", {})
    rows = []
    for lv in sorted(set(bl) | set(ml), key=int):
        b, m = bl.get(lv, {}), ml.get(lv, {})
        red = ""
        if b.get("mean_tokens") and m.get("mean_tokens"):
            red = _change(m["mean_tokens"], b["mean_tokens"])
        rows.append(
            f"<tr><td>Level {lv}</td><td>{_pct(b.get('acc'))}</td><td>{_pct(m.get('acc'))}</td>"
            f"<td>{_num(b.get('mean_tokens'))}</td><td>{_num(m.get('mean_tokens'))}</td><td>{red}</td></tr>"
        )
    return (
        "<table>\n<tr><th>MATH level</th><th>Base acc</th><th>This model acc</th><th>Base tokens</th><th>This model tokens</th><th>Token change</th></tr>\n"
        + "\n".join(rows)
        + "\n</table>"
    )


def hyperparams(cfg, name):
    if name == "ties":
        info = read_json(wpath(cfg, "models", "ties", "merge_info.json"), {})
        return "<table>\n" + "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in info.items()) + "\n</table>"
    t = dict(cfg["train"][name])
    if name != "grpo" and t.get("mode") == "lora":
        t.update({f"lora_{k}": v for k, v in cfg["lora"].items()})
    if name == "grpo":
        t.update({f"lora_{k}": v for k, v in cfg["lora"].items()})
        t["loss_type"] = "dr_grpo"
        t["scale_rewards"] = False
    return "<table>\n" + "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in t.items()) + "\n</table>"


def base_model_of(cfg, name):
    if name == "ties":
        info = read_json(wpath(cfg, "models", "ties", "merge_info.json"), {})
        return [repo_id(cfg, m) for m in info.get("models", [])] or [cfg["base_model"]], "merge"
    init = cfg["train"][name].get("init", "base")
    return [cfg["base_model"] if init == "base" else repo_id(cfg, init)], "finetune"


def card(cfg, name):
    bases, relation = base_model_of(cfg, name)
    repo = repo_id(cfg, name)
    slug = get_meta(cfg, "collection_slug")
    coll = f"https://huggingface.co/collections/{slug}" if slug else f"https://huggingface.co/{cfg['hf_user']}"
    base_yaml = "\n".join(f"  - {b}" for b in bases)
    has_pareto = Path(wpath(cfg, "report", "pareto.png")).exists()
    pareto = "\n![Accuracy vs tokens](assets/pareto.png)\n" if has_pareto else ""
    has_levels = Path(wpath(cfg, "report", "levels.png")).exists()
    levels_img = "\n![Tokens by MATH level](assets/levels.png)\n" if has_levels else ""
    train_desc = "merged, no gradient steps" if name == "ties" else "LoRA on all linear layers, merged into full bf16 safetensors"
    return f"""---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
base_model:
{base_yaml}
base_model_relation: {relation}
datasets:
  - {dataset_repo(cfg)}
language:
  - en
tags:
  - shortthink
  - reasoning
  - concise_reasoning
  - qwen3
  - {name}
---

# ShortThink {name.upper()} ({cfg['base_label']})

Reasoning models spend far more tokens than they need on easy problems. ShortThink teaches `{cfg['base_model']}` to keep its accuracy while thinking less, and to compress easy problems more than hard ones. This model is one of six in the [ShortThink collection]({coll}), all trained on the same self generated data with a different algorithm, and all evaluated on the same table.

## Method

{METHOD[name]}

Training data: [{dataset_repo(cfg)}](https://huggingface.co/datasets/{dataset_repo(cfg)}). Weights: {train_desc}.

## Why finetune instead of prompting

A prompt can ask for brevity, and the base model can switch thinking off entirely. The table below includes both of those, plus budget forcing (cap the thinking at {cfg['eval']['budget_tokens']} tokens and force an answer), so you can see exactly what training buys over prompting.

## Results

All rows use the same prompts, sampling settings and `math_verify` grading. MATH500 and GSM8K use {cfg['eval']['samples']} samples per problem, AIME 2024 uses {cfg['eval']['aime_samples']}. Tokens are generated tokens including the thinking. Tokens per correct is total tokens divided by correct answers, averaged over MATH500 and GSM8K. Latency is measured at batch size 1 with vLLM; cost assumes ${cfg['gpu_cost_per_hour']}/hour GPU at the measured batched throughput.

{results_table(cfg, highlight=name)}
{pareto}
### Compression by difficulty (MATH500)

{level_table(cfg, name)}
{levels_img}
## Training hyperparameters

{hyperparams(cfg, name)}

## Usage

```python
from vllm import LLM, SamplingParams

llm = LLM("{repo}")
tok = llm.get_tokenizer()
question = "What is 15% of 80?"
msgs = [{{"role": "user", "content": question + "\\nPlease reason step by step, and put your final answer within \\\\boxed{{}}."}}]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
out = llm.generate([prompt], SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=8192))
print(out[0].outputs[0].text)
```

Use the same prompt format and sampling settings as the base model's thinking mode.

## Files

The repository root holds merged bf16 safetensors. `adapter/` holds the LoRA adapter where one was trained, `training/` holds loss curves and logs, and `eval/` holds raw evaluation results and sample generations.
"""


def write_card(cfg, name, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(card(cfg, name))
