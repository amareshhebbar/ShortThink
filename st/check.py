import importlib
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid

PARTS = ["sys", "hub", "data", "vllm", "train", "grpo"]

PACKAGES = [
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "datasets",
    "trl",
    "vllm",
    "math_verify",
    "huggingface_hub",
    "lm_eval",
    "safetensors",
    "yaml",
    "pandas",
    "matplotlib",
    "hf_transfer",
]


def ok(msg):
    print(f"  [PASS] {msg}", flush=True)


def fail(msg):
    print(f"  [FAIL] {msg}", flush=True)
    raise SystemExit(1)


def check_sys(cfg):
    import torch

    print(f"  python {platform.python_version()} at {sys.executable}", flush=True)
    for name in PACKAGES:
        try:
            m = importlib.import_module(name)
            ok(f"import {name} {getattr(m, '__version__', '')}")
        except Exception as e:
            fail(f"import {name}: {e!r}")
    if shutil.which("nvidia-smi"):
        print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(), flush=True)
    if not torch.cuda.is_available():
        fail("torch.cuda.is_available() is False")
    ok(f"CUDA available, torch CUDA {torch.version.cuda}, {torch.cuda.device_count()} GPU(s)")
    p = torch.cuda.get_device_properties(0)
    vram = p.total_memory / 2**30
    ok(f"GPU 0: {p.name}, {vram:.1f} GiB, compute capability {p.major}.{p.minor}")
    if p.major < 8:
        fail("bf16 needs compute capability 8.0 or newer (A100, A6000, L40S, H100, RTX 3090/4090)")
    if not torch.cuda.is_bf16_supported():
        fail("bf16 not supported on this GPU")
    ok("bf16 supported")
    if vram < 40:
        print(f"  [WARN] {vram:.0f} GiB VRAM. Defaults target 48 GiB+. Lower micro_batch, max_len and grpo per_device_batch in configs/config.yaml.", flush=True)
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(20):
        a @ a
    torch.cuda.synchronize()
    tflops = 20 * 2 * 4096**3 / (time.time() - t) / 1e12
    ok(f"bf16 matmul {tflops:.1f} TFLOPS")
    work = cfg["paths"]["work"]
    free = shutil.disk_usage(work).free / 2**30
    print(f"  work dir {work}, free disk {free:.0f} GiB", flush=True)
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    os.makedirs(hf_home, exist_ok=True)
    print(f"  HF_HOME {hf_home}, free disk {shutil.disk_usage(hf_home).free / 2**30:.0f} GiB", flush=True)
    if free < 60:
        print("  [WARN] under 60 GiB free. The full run stores 8 models of about 3.5 GiB plus traces. Use a bigger volume.", flush=True)
    else:
        ok("disk space")


def check_hub(cfg):
    from huggingface_hub import HfApi

    from .hub import login

    try:
        login()
    except Exception as e:
        fail(f"login: {e}")
    api = HfApi()
    me = api.whoami()
    name = me.get("name")
    ok(f"logged in as {name}")
    if name != cfg["hf_user"] and cfg["hf_user"] not in [o.get("name") for o in me.get("orgs", [])]:
        fail(f"token belongs to {name} but config hf_user is {cfg['hf_user']}. Set HF_USER in .env")
    role = me.get("auth", {}).get("accessToken", {}).get("role")
    print(f"  token role: {role}", flush=True)
    if role == "read":
        fail("token is read only. Create a write token at huggingface.co/settings/tokens")
    tag = uuid.uuid4().hex[:6]
    repo = f"{cfg['hf_user']}/shortthink_write_test_{tag}"
    try:
        api.create_repo(repo, private=True)
        api.upload_file(path_or_fileobj=b"ok", path_in_repo="probe.txt", repo_id=repo, commit_message="write probe")
        files = api.list_repo_files(repo)
        if "probe.txt" not in files:
            fail("uploaded file not visible")
        ok(f"create repo and upload file ({repo})")
        c = api.create_collection(title=f"shortthink write test {tag}", namespace=cfg["hf_user"], private=True)
        api.add_collection_item(c.slug, repo, "model", note="probe")
        ok("create collection and add item")
        api.delete_collection(c.slug)
        ok("delete test collection")
    finally:
        try:
            api.delete_repo(repo)
            ok("delete test repo")
        except Exception as e:
            print(f"  [WARN] could not delete {repo}: {e}", flush=True)


def check_data(cfg):
    from math_verify import parse, verify
    from transformers import AutoTokenizer

    from . import data
    from .grade import score

    sets = {
        "gsm8k train": lambda: data.gsm8k("train"),
        "gsm8k test": lambda: data.gsm8k("test"),
        "MATH train": data.math_train,
        "MATH500": data.math500,
        "AIME 2024": data.aime24,
    }
    for label, fn in sets.items():
        try:
            items = fn()
        except Exception as e:
            fail(f"load {label}: {e!r}")
        if not items:
            fail(f"{label} is empty")
        ok(f"{label}: {len(items)} rows, sample gold {items[0]['gold'][:40]!r}")
    if not bool(verify(parse("\\boxed{\\frac{1}{2}}"), parse("The answer is $\\boxed{0.5}$"))):
        fail("math_verify did not match 1/2 with 0.5")
    if bool(verify(parse("\\boxed{3}"), parse("\\boxed{4}"))):
        fail("math_verify matched 3 with 4")
    if not score(("<think>\nhmm\n</think>\n\nSo $\\boxed{18}$", "18"))[0]:
        fail("grader rejected a correct thinking answer")
    if score(("<think>\nstill thinking", "18"))[0]:
        fail("grader accepted an unfinished thinking answer")
    ok("math_verify and grader")
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    p_think = data.chat_prompt(tok, "What is 2+2?", thinking=True)
    p_no = data.chat_prompt(tok, "What is 2+2?", thinking=False)
    if "<think>" in p_think.split("<|im_start|>assistant")[-1]:
        fail("thinking prompt already contains a think block")
    if "</think>" not in p_no:
        fail("no thinking prompt is missing the empty think block, enable_thinking not supported by this template")
    ok("chat template thinking and no thinking modes")
    if tok.eos_token != "<|im_end|>":
        fail(f"unexpected eos token {tok.eos_token!r}")
    ok(f"eos token {tok.eos_token}")


def check_vllm(cfg):
    from .data import chat_prompt, gsm8k
    from .engine import Engine
    from .grade import grade_many

    t = time.time()
    eng = Engine(cfg, cfg["base_model"], gpu_util=0.6, max_model_len=4096)
    ok(f"vLLM loaded {cfg['base_model']} in {time.time() - t:.0f}s")
    items = gsm8k("test")[:4]
    prompts = [chat_prompt(eng.tok, it["question"], thinking=True) for it in items]
    res, dt = eng.run(prompts, eng.params(2, 3072, True))
    texts = [o[0] for outs in res for o in outs]
    golds = [it["gold"] for it in items for _ in range(2)]
    grades = grade_many(texts, golds)
    toks = [o[1] for outs in res for o in outs]
    print(f"  thinking: {sum(g[0] for g in grades)}/{len(grades)} correct, mean tokens {sum(toks) / len(toks):.0f}, {sum(toks) / dt:.0f} tok/s", flush=True)
    print("  sample:\n" + texts[0][-400:], flush=True)
    if not all(t.strip() for t in texts):
        fail("empty generations")
    nres, _ = eng.run([chat_prompt(eng.tok, it["question"], thinking=False) for it in items], eng.params(1, 512, False))
    print(f"  no thinking: mean tokens {sum(o[0][1] for o in nres) / len(nres):.0f}", flush=True)
    bres, _ = eng.budget(prompts[:2], 1, 128, 128)
    if "</think>" not in bres[0][0][0]:
        fail("budget forcing did not close the think block")
    ok("vLLM thinking, no thinking and budget forcing generation")


def check_train(cfg):
    import torch
    from transformers import AutoTokenizer

    from .data import chat_prompt
    from .train import build_model, collate, encode, pref_loss, save_merged, sft_loss

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    prompt = chat_prompt(tok, "What is 3+5?", thinking=True)
    short = "<think>\n3+5=8.\n</think>\n\nThe answer is $\\boxed{8}$." + tok.eos_token
    long = "<think>\nLet me think. 3 plus 5. Count up from 3: 4, 5, 6, 7, 8. So 8. Double check: 5+3=8. Yes.\n</think>\n\nThe answer is $\\boxed{8}$." + tok.eos_token
    model = build_model(cfg, cfg["base_model"], "lora")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    tc = {"logits_chunk": 16, "beta": 0.1, "lam": 0.1, "gamma_beta_ratio": 0.5}
    sft_batch = collate([encode(tok, prompt, short)], pad)
    losses = []
    for _ in range(4):
        loss, st = sft_loss(model, sft_batch, tc)
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
    print(f"  sft losses {[round(x, 4) for x in losses]}", flush=True)
    if not losses[-1] < losses[0]:
        fail("SFT loss did not go down")
    ok("SFT step with chunked logits and LoRA")
    pair = collate([encode(tok, prompt, short), encode(tok, prompt, long)], pad)
    for name in ["dpo", "orpo", "simpo"]:
        tc["beta"] = 0.1 if name == "dpo" else 2.0
        loss, st = pref_loss(name)(model, pair, tc)
        loss.backward()
        opt.zero_grad()
        if not torch.isfinite(loss):
            fail(f"{name} loss not finite")
        ok(f"{name} loss {loss.item():.4f} {st}")
    print(f"  peak memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    model.eval()
    smoke_cfg = dict(cfg)
    save_merged(smoke_cfg, "_smoke_train", model, tok, cfg["base_model"], cfg["base_model"])
    from .common import merged_dir

    out = merged_dir(cfg, "_smoke_train")
    if not list(out.glob("*.safetensors")):
        fail("merged safetensors not written")
    from transformers import AutoModelForCausalLM

    AutoModelForCausalLM.from_pretrained(str(out), torch_dtype=torch.bfloat16)
    ok("merge LoRA, save safetensors, reload")
    shutil.rmtree(out.parent, ignore_errors=True)


def check_grpo(cfg):
    from transformers import AutoTokenizer

    from .common import model_dir
    from .data import chat_prompt, gsm8k
    from .grpo import train_grpo

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    rows = [{"prompt": chat_prompt(tok, it["question"], thinking=True), "gold": it["gold"]} for it in gsm8k("test")[:4]]
    hist = train_grpo(
        cfg,
        smoke_rows=rows,
        smoke_overrides={
            "max_steps": 2,
            "num_generations": 2,
            "per_device_batch": 2,
            "grad_accum": 2,
            "max_completion_length": 256,
            "vllm_gpu_memory_utilization": 0.25,
        },
    )
    rewards = [h.get("reward") for h in hist if "reward" in h]
    print(f"  rewards {rewards}", flush=True)
    if not rewards:
        fail("GRPO logged no rewards")
    ok("GRPO with vLLM colocate, LoRA and the length reward")
    shutil.rmtree(model_dir(cfg, "_smoke_grpo"), ignore_errors=True)


FUNCS = {"sys": check_sys, "hub": check_hub, "data": check_data, "vllm": check_vllm, "train": check_train, "grpo": check_grpo}


def run_part(cfg, part):
    print(f"\nCHECK {part}", flush=True)
    FUNCS[part](cfg)
    print(f"CHECK {part} DONE", flush=True)
