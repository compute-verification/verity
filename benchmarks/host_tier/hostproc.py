"""A stand-in for vLLM's host process, timed per step: a scheduler step over a running batch, then tokenizing a prompt.

    hostproc.py --seconds 5 --label alone [--wait-for FILE]

The scheduler step is pure Python as vLLM's is: each of 256 running requests takes a token, allocates a 16-token KV block
from a free list when it fills one, and the step flattens input ids, positions and slot mapping for the batch. The
tokenizer is a greedy longest match of a 4 KiB prompt over a 128k-entry vocabulary, in pure Python. Prints one JSON
line: steps and the p50/p90/p99 of each part in microseconds.
"""

import argparse
import json
import os
import random
import time


def percentiles(xs):
    xs = sorted(xs)
    return {f"p{q}": round(xs[min(len(xs) - 1, len(xs) * q // 100)] * 1e6, 1) for q in (50, 90, 99)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=5)
    ap.add_argument("--label", default="")
    ap.add_argument("--wait-for")
    a = ap.parse_args()
    rng = random.Random(1)
    alphabet = "etaoinshrdlucmfwypvbgkqjxz     "
    vocab = {}
    while len(vocab) < 128_000:
        w = "".join(rng.choice(alphabet) for _ in range(rng.randint(2, 8)))
        vocab.setdefault(w, len(vocab))
    for c in alphabet:
        vocab.setdefault(c, len(vocab))
    prompts = ["".join(rng.choice(alphabet) for _ in range(4096)) for _ in range(64)]
    free = list(range(200_000))
    reqs = [{"id": i, "tokens": [rng.randrange(128_000) for _ in range(rng.randint(64, 2048))], "blocks": []} for i in range(256)]
    for r in reqs:
        r["blocks"] = [free.pop() for _ in range((len(r["tokens"]) + 15) // 16)]

    def schedule():
        ids, pos, slots = [], [], []
        for r in reqs:
            t = r["tokens"]
            t.append(t[-1] * 31 % 128_000)
            if len(t) % 16 == 1:
                r["blocks"].append(free.pop())
            if len(t) > 4096:
                free.extend(r["blocks"])
                r["tokens"], r["blocks"] = t[:64], [free.pop() for _ in range(4)]
                t = r["tokens"]
            n = len(t) - 1
            ids.append(t[-1])
            pos.append(n)
            slots.append(r["blocks"][n // 16] * 16 + n % 16)
        reqs.sort(key=lambda r: (len(r["tokens"]) & 7, r["id"]))
        return len(ids) + len(pos) + len(slots)

    def tokenize(s):
        out, i = [], 0
        while i < len(s):
            for n in range(8, 0, -1):
                v = vocab.get(s[i : i + n])
                if v is not None:
                    out.append(v)
                    i += n
                    break
        return out

    if a.wait_for:
        while not os.path.exists(a.wait_for):
            time.sleep(0.01)
    sched, tok = [], []
    t_end = time.perf_counter() + a.seconds
    k = 0
    while time.perf_counter() < t_end:
        t0 = time.perf_counter()
        schedule()
        t1 = time.perf_counter()
        tokenize(prompts[k % len(prompts)])
        t2 = time.perf_counter()
        sched.append(t1 - t0)
        tok.append(t2 - t1)
        k += 1
    print(json.dumps({"label": a.label, "cpu": sorted(os.sched_getaffinity(0)), "steps": k, "schedule_us": percentiles(sched),
                      "tokenize_us": percentiles(tok)}))


if __name__ == "__main__":
    main()
