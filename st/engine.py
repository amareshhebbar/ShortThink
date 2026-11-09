import statistics
import time

from vllm import LLM, SamplingParams

FORCE = "\n\nI have spent enough time thinking, so I will give the final answer now.\n</think>\n\n"


class Engine:
    def __init__(self, cfg, model, gpu_util=None, max_model_len=None):
        self.cfg = cfg
        self.model = model
        self.llm = LLM(
            model=model,
            dtype="bfloat16",
            seed=cfg["seed"],
            gpu_memory_utilization=gpu_util or cfg["gen"]["gpu_memory_utilization"],
            max_model_len=max_model_len or cfg["gen"]["max_model_len"],
            enable_prefix_caching=True,
        )
        self.tok = self.llm.get_tokenizer()
        self.force_len = len(self.tok(FORCE, add_special_tokens=False)["input_ids"])

    def params(self, n, max_tokens, thinking=True):
        g = self.cfg["gen"]["think" if thinking else "nothink"]
        return SamplingParams(n=n, temperature=g["temperature"], top_p=g["top_p"], top_k=g["top_k"], max_tokens=max_tokens)

    def run(self, prompts, sp, progress=True):
        t = time.time()
        outs = self.llm.generate(prompts, sp, use_tqdm=progress)
        dt = time.time() - t
        res = [[(c.text, len(c.token_ids), c.finish_reason) for c in o.outputs] for o in outs]
        return res, dt

    def budget(self, prompts, n, budget, tail):
        first, dt1 = self.run(prompts, self.params(n, budget, True))
        conts, where = [], []
        for i, (p, outs) in enumerate(zip(prompts, first)):
            for j, (t, k, fr) in enumerate(outs):
                if "</think>" not in t and fr == "length":
                    conts.append(p + t + FORCE)
                    where.append((i, j))
        res = [list(o) for o in first]
        dt2 = 0.0
        if conts:
            second, dt2 = self.run(conts, self.params(1, tail, True))
            for (i, j), o in zip(where, second):
                t, k, _ = res[i][j]
                t2, k2, fr2 = o[0]
                res[i][j] = (t + FORCE + t2, k + self.force_len + k2, fr2)
        return res, dt1 + dt2

    def latency(self, prompts, max_tokens, thinking=True, budget=None, tail=512):
        times, toks = [], []
        for p in prompts:
            t = time.time()
            if budget:
                r, _ = self.budget([p], 1, budget, tail)
            else:
                r, _ = self.run([p], self.params(1, max_tokens, thinking), progress=False)
            times.append(time.time() - t)
            toks.append(r[0][0][1])
        times_sorted = sorted(times)
        p95 = times_sorted[min(len(times_sorted) - 1, int(round(0.95 * (len(times_sorted) - 1))))]
        return {
            "n": len(times),
            "p50_s": statistics.median(times),
            "p95_s": p95,
            "mean_s": statistics.mean(times),
            "decode_tok_s": sum(toks) / max(1e-9, sum(times)),
        }
