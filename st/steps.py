import os

from .common import ACTIONS, MODELS


def selected_models():
    only = os.environ.get("ONLY_MODELS", "").strip()
    if not only:
        return list(MODELS)
    want = [m.strip().lower() for m in only.split(",") if m.strip()]
    bad = [m for m in want if m not in MODELS]
    if bad:
        raise SystemExit(f"ONLY_MODELS has unknown models {bad}, choose from {MODELS}")
    return [m for m in MODELS if m in want]


def pipeline_steps():
    steps = ["baseline", "baseline_mmlu", "factory"]
    for m in selected_models():
        steps += [f"{m}:{a}" for a in ACTIONS]
    steps.append("report")
    return steps


def run_step(cfg, step):
    if step == "baseline":
        from .evaluate import run_baseline

        return run_baseline(cfg)
    if step == "baseline_mmlu":
        from .evaluate import run_mmlu

        return run_mmlu(cfg, "base")
    if step == "factory":
        from .factory import run_factory

        return run_factory(cfg)
    if step == "report":
        from .report import run_report

        return run_report(cfg)
    if ":" not in step:
        raise SystemExit(f"unknown step {step}")
    name, action = step.split(":", 1)
    if name not in MODELS or action not in ACTIONS:
        raise SystemExit(f"unknown step {step}")
    if action == "train":
        if name == "grpo":
            from .grpo import train_grpo

            return train_grpo(cfg)
        if name == "ties":
            from .ties import run_ties

            return run_ties(cfg)
        from .train import train_offline

        return train_offline(cfg, name)
    if action == "push":
        from .hub import push_model

        return push_model(cfg, name)
    if action == "verify":
        from .verify import verify_model

        return verify_model(cfg, name)
    if action == "evalgen":
        from .evaluate import run_model_eval

        return run_model_eval(cfg, name)
    if action == "mmlu":
        from .evaluate import run_mmlu

        return run_mmlu(cfg, name)
    if action == "publish":
        from .hub import publish

        return publish(cfg, name)
