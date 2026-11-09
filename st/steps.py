from .common import ACTIONS, MODELS, POLICIES


def pipeline_steps():
    steps = ["baseline", "baseline_mmlu", "factory"]
    for m in POLICIES + ["ties"]:
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
