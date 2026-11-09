from pathlib import Path

import torch
from safetensors import safe_open

from .common import POLICIES, banner, merged_dir, read_json, repo_id, resolve_model, results_path, write_json


def pick_models(cfg):
    m = cfg["merge"]
    if m.get("models"):
        return list(m["models"])
    base = read_json(results_path(cfg, "base_think", "full"))
    scored = []
    for name in POLICIES:
        r = read_json(results_path(cfg, name, "full"))
        if r:
            scored.append((name, r["summary"]["acc"], r["summary"]["tokens_per_correct"]))
    if len(scored) < 2:
        raise RuntimeError(f"need at least two evaluated models to merge, have {[s[0] for s in scored]}")
    floor = base["summary"]["acc"] - cfg["eval"]["merge_acc_drop"] if base else -1
    ok = sorted([s for s in scored if s[1] >= floor], key=lambda s: s[2])
    if len(ok) < 2:
        ok = sorted(scored, key=lambda s: (-s[1], s[2]))
    print("merge candidates (name, acc, tokens per correct):", ok, flush=True)
    return [ok[0][0], ok[1][0]]


def load_tensors(path):
    files = sorted(Path(path).glob("*.safetensors"))
    index = {}
    for f in files:
        with safe_open(str(f), framework="pt") as h:
            for k in h.keys():
                index[k] = f
    return index


def get_tensor(index, key):
    with safe_open(str(index[key]), framework="pt") as h:
        return h.get_tensor(key)


def ties(base, deltas, weights, density):
    trimmed = []
    for d in deltas:
        flat = d.abs().flatten()
        k = max(1, int(round(density * flat.numel())))
        if k >= flat.numel():
            trimmed.append(d)
            continue
        thresh = torch.topk(flat, k, sorted=False).values.min()
        trimmed.append(torch.where(d.abs() >= thresh, d, torch.zeros_like(d)))
    w = torch.tensor(weights, dtype=torch.float32, device=base.device).view(-1, *([1] * base.dim()))
    stack = torch.stack(trimmed) * w
    sign = torch.sign(stack.sum(0))
    agree = (torch.sign(stack) == sign.unsqueeze(0)) & (stack != 0)
    num = (stack * agree).sum(0)
    den = (w * agree).sum(0).clamp(min=1e-12)
    merged = torch.where(agree.any(0), num / den, torch.zeros_like(num))
    return base + merged


def run_ties(cfg):
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    names = pick_models(cfg)
    weights = cfg["merge"]["weights"][: len(names)]
    density = cfg["merge"]["density"]
    paths = [resolve_model(cfg, n) for n in names]
    banner(f"TIES merge of {names} with weights {weights}, density {density}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], torch_dtype=torch.bfloat16)
    indices = [load_tensors(p) for p in paths]
    sd = model.state_dict()
    seen = set()
    changed = 0
    for key, base_t in sd.items():
        ptr = base_t.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        if not all(key in ix for ix in indices):
            continue
        b = base_t.to(dev, torch.float32)
        deltas = [get_tensor(ix, key).to(dev, torch.float32) - b for ix in indices]
        if all(float(d.abs().max()) == 0.0 for d in deltas):
            continue
        out = ties(b, deltas, weights, density)
        base_t.copy_(out.to(base_t.device, base_t.dtype))
        changed += 1
    print(f"merged {changed} tensors", flush=True)
    out_dir = merged_dir(cfg, "ties")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.config.use_cache = True
    model.save_pretrained(str(out_dir), safe_serialization=True, max_shard_size="2GB")
    AutoTokenizer.from_pretrained(paths[0]).save_pretrained(str(out_dir))
    try:
        GenerationConfig.from_pretrained(cfg["base_model"]).save_pretrained(str(out_dir))
    except Exception:
        pass
    write_json(
        merged_dir(cfg, "ties").parent / "merge_info.json",
        {"method": "ties", "models": names, "repos": [repo_id(cfg, n) for n in names], "weights": weights, "density": density, "changed_tensors": changed},
    )
    banner(f"saved TIES merge to {out_dir}")
