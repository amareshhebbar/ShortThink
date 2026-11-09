# ShortThink

Reasoning models that stop overthinking. One base model (Qwen3 1.7B), one self generated dataset, six training algorithms, and one honest comparison table against prompting baselines.

Every model is pushed to the Hugging Face Hub as merged bf16 safetensors, added to the ShortThink collection, downloaded back from the Hub and tested before the next model starts. No external APIs are used anywhere: generation, grading, training and evaluation all run on the pod.

## Quick start on RunPod

Use a pod with a 48 GB GPU or bigger (L40S, A6000, A100, H100) and a network volume of at least 150 GB mounted at `/workspace`.

```bash
cd /workspace
git clone https://github.com/amareshhebbar/ShortThink.git
cd ShortThink
bash setup.sh
nano .env
source .venv/bin/activate
python run.py check
tmux new -s st
python run.py pipeline
```

In `.env`, set `HF_TOKEN` to a Hugging Face token with write access. Detach from tmux with `Ctrl+b` then `d`, and reattach later with `tmux attach -t st`.

## What the check covers

`python run.py check` runs six isolated checks and prints PASS or FAIL for each.

<table>
<tr><th>Check</th><th>What it proves</th></tr>
<tr><td>sys</td><td>Every library imports, CUDA works, bf16 works, VRAM and disk are enough, a bf16 matmul runs</td></tr>
<tr><td>hub</td><td>Token is valid and has write access: creates a private repo, uploads a file, creates a collection, adds the repo, then deletes both</td></tr>
<tr><td>data</td><td>GSM8K, MATH, MATH500 and AIME 2024 load; math_verify grades correctly; the Qwen3 chat template switches thinking on and off</td></tr>
<tr><td>vllm</td><td>Base model loads in vLLM and generates in thinking, no thinking and budget forcing modes</td></tr>
<tr><td>train</td><td>LoRA SFT loss goes down, DPO, ORPO and SimPO losses are finite, the adapter merges into safetensors and reloads</td></tr>
<tr><td>grpo</td><td>Two real GRPO steps with vLLM colocated, LoRA and the length reward</td></tr>
</table>

Run a single check with `python run.py check grpo`.

## Pipeline

`python run.py pipeline` runs these steps in order. Finished steps are recorded in `/workspace/shortthink_runs/state.json`, so if the pod stops, run the same command again and it continues where it left off.

1. `baseline`: base model in four modes (thinking, thinking with a concise prompt, no thinking, budget forcing), full evaluation.
2. `baseline_mmlu`: MMLU for the base model.
3. `factory`: samples every GSM8K and MATH train question 8 times, grades every answer, builds the `sft`, `pairs` and `prompts` sets, pushes them as one dataset to the Hub, adds it to the collection and checks the row counts from the Hub.
4. For each of SFT, DPO, ORPO, SimPO, GRPO and the TIES merge:
   1. `train`: trains and merges the LoRA adapter into full safetensors
   2. `push`: uploads weights, adapter, training logs and a model card, adds the repo to the collection
   3. `verify`: deletes the local copy, downloads from the Hub, checks the files, runs a quick evaluation and stops the pipeline if the model is broken
   4. `evalgen`: full evaluation on MATH500, GSM8K and AIME 2024 with latency and cost
   5. `mmlu`: MMLU to confirm general ability did not break
   6. `publish`: updates the model card with the results table
5. `report`: comparison charts, results table, and final cards for every model.

Useful commands:

```bash
python run.py status
python run.py steps
python run.py run sft:verify
python run.py pipeline from=dpo:train to=dpo:publish
python run.py reset grpo:train
```

## Repository names

<table>
<tr><th>Item</th><th>Hub id</th></tr>
<tr><td>Dataset</td><td>AmareshHebbar/ShortThink_Traces_Qwen3_1.7B</td></tr>
<tr><td>Models</td><td>AmareshHebbar/ShortThink_Qwen3_1.7B_SFT, _DPO, _ORPO, _SIMPO, _GRPO, _TIES</td></tr>
<tr><td>Collection</td><td>ShortThink</td></tr>
</table>

Change names, hyperparameters and budgets in `configs/config.yaml`.

## Smaller GPUs or a test run

For a cheap end to end dry run, set these in `configs/config.yaml` before `python run.py pipeline`:

```yaml
factory:
  max_questions: 600
eval:
  samples: 1
  aime: false
  mmlu: false
train:
  grpo:
    max_steps: 20
```

On a 24 GB GPU, set `micro_batch: 1` for SFT, lower `max_len` for the preference methods to 4096, and set GRPO `per_device_batch: 2` with `grad_accum: 32`.

## Layout

<table>
<tr><td>run.py</td><td>Entry point: check, pipeline, run, status, reset</td></tr>
<tr><td>st/check.py</td><td>Environment checks</td></tr>
<tr><td>st/data.py</td><td>Dataset loaders and the shared prompt format</td></tr>
<tr><td>st/engine.py</td><td>vLLM generation, budget forcing, latency probe</td></tr>
<tr><td>st/grade.py</td><td>math_verify grading in parallel</td></tr>
<tr><td>st/factory.py</td><td>Trace generation, SFT and preference sets, dataset push</td></tr>
<tr><td>st/train.py</td><td>SFT, DPO, ORPO and SimPO with chunked logits and LoRA</td></tr>
<tr><td>st/grpo.py</td><td>GRPO with a group relative length reward (TRL, vLLM colocate)</td></tr>
<tr><td>st/ties.py</td><td>TIES merge of the two best models</td></tr>
<tr><td>st/evaluate.py</td><td>Quick and full evaluation, MMLU through lm_eval</td></tr>
<tr><td>st/verify.py</td><td>Hub download and quick evaluation gate</td></tr>
<tr><td>st/hub.py, st/card.py</td><td>Uploads, collection, model cards</td></tr>
<tr><td>st/report.py</td><td>Charts and the final comparison</td></tr>
</table>
