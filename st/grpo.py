import dataclasses
import json
import random
from collections import defaultdict

import torch

from .common import banner, init_repo_for, model_dir, resolve_model, write_json
from .grade import score
from .train import load_rows, plot_log, save_merged


def make_rewards(alpha):
    def length_reward(prompts, completions, completion_ids=None, gold=None, **kwargs):
        ntoks = [len(c) for c in completion_ids] if completion_ids else [len(c) for c in completions]
        ok = [score((c, g))[0] for c, g in zip(completions, gold)]
        groups = defaultdict(list)
        for i, p in enumerate(prompts):
            groups[p].append(i)
        rewards = [0.0] * len(completions)
        for idx in groups.values():
            lens = [ntoks[i] for i in idx]
            lo, hi = min(lens), max(lens)
            for i in idx:
                lam = 0.0 if hi == lo else 0.5 - (ntoks[i] - lo) / (hi - lo)
                rewards[i] = 1.0 + alpha * lam if ok[i] else alpha * min(0.0, lam)
        return rewards

    def accuracy(prompts, completions, gold=None, **kwargs):
        return [1.0 if score((c, g))[0] else 0.0 for c, g in zip(completions, gold)]

    return [length_reward, accuracy]


def build_config(cls, values):
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(k for k in values if k not in names)
    if unknown:
        print(f"ignoring config keys not in {cls.__name__}: {unknown}", flush=True)
    return cls(**{k: v for k, v in values.items() if k in names})


def grpo_args(cfg, g, out_dir, max_steps, smoke=False):
    from trl import GRPOConfig

    return build_config(
        GRPOConfig,
        {
            "output_dir": str(out_dir),
            "learning_rate": g["lr"],
            "lr_scheduler_type": "cosine",
            "warmup_ratio": g["warmup_ratio"],
            "max_steps": max_steps,
            "per_device_train_batch_size": g["per_device_batch"],
            "gradient_accumulation_steps": g["grad_accum"],
            "num_generations": g["num_generations"],
            "max_prompt_length": g["max_prompt_length"],
            "max_completion_length": g["max_completion_length"],
            "temperature": g["temperature"],
            "top_p": g["top_p"],
            "top_k": g["top_k"],
            "use_vllm": True,
            "vllm_mode": "colocate",
            "vllm_gpu_memory_utilization": g["vllm_gpu_memory_utilization"],
            "loss_type": "dr_grpo",
            "scale_rewards": False,
            "beta": g["beta"],
            "mask_truncated_completions": True,
            "reward_weights": [1.0, 0.0],
            "bf16": True,
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "logging_steps": 1,
            "save_strategy": "no",
            "report_to": "none",
            "seed": cfg["seed"],
            "remove_unused_columns": False,
            "log_completions": False,
            "disable_dropout": True,
        },
    )


def lora_cfg(cfg):
    from peft import LoraConfig

    return LoraConfig(r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"], lora_dropout=cfg["lora"]["dropout"], target_modules="all-linear", task_type="CAUSAL_LM")


def train_grpo(cfg, smoke_rows=None, smoke_overrides=None):
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOTrainer

    g = dict(cfg["train"]["grpo"])
    if smoke_overrides:
        g.update(smoke_overrides)
    init = g.get("init", "base")
    init_path = resolve_model(cfg, init)
    tok = AutoTokenizer.from_pretrained(init_path)
    if smoke_rows is None:
        rows = load_rows(cfg, "prompts")
        prompt_ids_len = [len(tok(r["prompt"], add_special_tokens=False)["input_ids"]) for r in rows]
        rows = [r for r, n in zip(rows, prompt_ids_len) if n <= g["max_prompt_length"]]
        rng = random.Random(cfg["seed"])
        rng.shuffle(rows)
        prompts_per_step = g["per_device_batch"] * g["grad_accum"] // g["num_generations"]
        need = g["max_steps"] * prompts_per_step
        cap = g.get("max_prompts") or need
        rows = rows[: min(len(rows), cap)]
        out_dir = model_dir(cfg, "grpo") / "trainer"
        max_steps = g["max_steps"]
    else:
        rows = smoke_rows
        out_dir = model_dir(cfg, "_smoke_grpo") / "trainer"
        max_steps = g["max_steps"]
    ds = Dataset.from_list([{"prompt": r["prompt"], "gold": r["gold"]} for r in rows])
    banner(f"grpo: {len(ds)} prompts, {max_steps} steps, {g['num_generations']} samples per prompt, alpha {g['alpha']}")
    model = AutoModelForCausalLM.from_pretrained(init_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    args = grpo_args(cfg, g, out_dir, max_steps)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=make_rewards(g["alpha"]),
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=lora_cfg(cfg),
    )
    trainer.train()
    history = trainer.state.log_history
    if smoke_rows is not None:
        return history
    logs = model_dir(cfg, "grpo") / "training"
    logs.mkdir(parents=True, exist_ok=True)
    rows_log = []
    for h in history:
        if "loss" in h or "reward" in h:
            rec = {"step": h.get("step")}
            for k, v in h.items():
                if isinstance(v, (int, float)):
                    rec[k.replace("/", "_")] = v
            rows_log.append(rec)
    with open(logs / "log.jsonl", "w") as f:
        for r in rows_log:
            f.write(json.dumps(r) + "\n")
    write_json(logs / "summary.json", {"prompts": len(ds), "steps": max_steps, "config": g})
    plot_grpo(logs / "log.jsonl", logs / "curves.png")
    model = trainer.model
    model.eval()
    save_merged(cfg, "grpo", model, tok, init_path, init_repo_for(cfg, init))


def plot_grpo(log_path, png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .common import read_jsonl

    rows = read_jsonl(log_path)
    keys = [k for k in ("reward", "rewards_accuracy_mean", "completions_mean_length", "loss") if any(k in r for r in rows)]
    if not keys:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 3.5), squeeze=False)
    for ax, k in zip(axes[0], keys):
        pts = [(r["step"], r[k]) for r in rows if k in r and r.get("step") is not None]
        ax.plot([p[0] for p in pts], [p[1] for p in pts])
        ax.set_title(f"grpo {k}")
        ax.set_xlabel("step")
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    plt.close(fig)
