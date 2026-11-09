import multiprocessing as mp
import os

from math_verify import parse, verify


def split_answer(text):
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1], True
    if "<think>" in text:
        return "", False
    return text, True


def score(pair):
    text, gold = pair
    ans, closed = split_answer(text or "")
    boxed = "\\boxed" in ans
    if not closed or not ans.strip():
        return (False, closed, boxed)
    try:
        g = parse("\\boxed{" + str(gold) + "}")
        p = parse(ans)
        ok = bool(g) and bool(p) and bool(verify(g, p))
    except Exception:
        ok = False
    return (ok, closed, boxed)


def grade_many(texts, golds, workers=None):
    pairs = list(zip(texts, golds))
    if len(pairs) < 64:
        return [score(p) for p in pairs]
    n = workers or max(1, min(32, (os.cpu_count() or 2) - 1))
    ctx = mp.get_context("spawn")
    with ctx.Pool(n) as pool:
        return list(pool.imap(score, pairs, chunksize=64))
