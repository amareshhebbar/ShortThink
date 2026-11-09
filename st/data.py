import random

from datasets import load_dataset

SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
CONCISE = "\nBe concise: think only as much as this problem needs."
MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def user_text(question, concise=False):
    return question.strip() + SUFFIX + (CONCISE if concise else "")


def chat_prompt(tok, question, thinking=True, concise=False):
    msgs = [{"role": "user", "content": user_text(question, concise)}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)


def last_boxed(s):
    if not s:
        return None
    i = s.rfind("\\boxed")
    if i < 0:
        i = s.rfind("\\fbox")
    if i < 0:
        return None
    j = s.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(s)):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[j + 1 : k].strip()
    return None


def _level(v):
    if v is None:
        return None
    if isinstance(v, int):
        return v
    digits = "".join(ch for ch in str(v) if ch.isdigit())
    return int(digits) if digits else None


def gsm8k(split):
    ds = load_dataset("openai/gsm8k", "main", split=split)
    out = []
    for i, r in enumerate(ds):
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        out.append({"id": f"gsm8k_{split}_{i}", "source": "gsm8k", "level": None, "question": r["question"], "gold": gold})
    return out


def math_train():
    out = []
    for subj in MATH_SUBJECTS:
        ds = load_dataset("EleutherAI/hendrycks_math", subj, split="train")
        for i, r in enumerate(ds):
            gold = last_boxed(r["solution"])
            if not gold:
                continue
            out.append(
                {
                    "id": f"math_train_{subj}_{i}",
                    "source": "math",
                    "level": _level(r.get("level")),
                    "question": r["problem"],
                    "gold": gold,
                }
            )
    return out


def math500():
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [
        {
            "id": f"math500_{r.get('unique_id', i)}",
            "source": "math500",
            "level": _level(r.get("level")),
            "question": r["problem"],
            "gold": str(r["answer"]),
        }
        for i, r in enumerate(ds)
    ]


def aime24():
    ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
    return [
        {"id": f"aime24_{r.get('id', i)}", "source": "aime24", "level": None, "question": r["problem"], "gold": str(r["answer"])}
        for i, r in enumerate(ds)
    ]


def train_questions(cfg):
    items = gsm8k("train") + math_train()
    n = cfg["factory"]["max_questions"]
    if n and n < len(items):
        rng = random.Random(cfg["seed"])
        rng.shuffle(items)
        items = items[:n]
    return items


def subset(items, n, seed):
    if not n or n >= len(items):
        return list(items)
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(items)), n))
    return [items[i] for i in idx]


def eval_sets(cfg, mode):
    e = cfg["eval"]
    m5 = math500()
    gs = gsm8k("test")
    if mode == "quick":
        return {"math500": subset(m5, e["quick_n"], cfg["seed"]), "gsm8k": subset(gs, e["quick_n"], cfg["seed"])}
    sets = {"math500": m5, "gsm8k": gs}
    if e.get("aime"):
        sets["aime24"] = aime24()
    return sets
