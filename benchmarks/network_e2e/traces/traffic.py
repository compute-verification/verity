"""The corpus's traffic: one seeded plan of requests (prompts, caps), and each trace's Poisson arrivals over it.

Every trace serves the same requests in the same order; only the arrival times differ, the same unit-rate exponential gaps
scaled by 1 / rate. A request's prompt is token ids:

* its length L from a mixture over [16, 1024] (`LENGTHS`: log-uniform short, uniform medium and long);
* with probability `SHARED`, one of `PREFIXES` shared prefixes in front (a group drawn with weight 1 / (g + 1), the longest
  that leaves at least `MIN_BODY` tokens), so later requests of a group hit the prefix cache: BOS, the start of the group's
  own document, then a blank line, `PREFIXES[g]` tokens in all, a multiple of the 16-token block;
* then a body: BOS when there is no prefix, then the L - p tokens of a document that end `gap` tokens before its end, where
  `gap` is 0 with probability `AT_END` and else uniform in [4, 240]. A base model that is a few sentences from a
  document's end tends to finish it and emit its end-of-text token: the outputs end on EOS or the cap as the model chooses.
  At the very end (gap 0) it emits EOS at once, a one-token output; `AT_END` keeps those few.

Caps are `CAPS` (64 or 256), each with probability 1/2. Temperature is 0.
"""
from __future__ import annotations

import math
import random
from collections.abc import Sequence

BOS = 128000
BLANK_LINE = 271
BLOCK = 16
#: (weight, kind, lo, hi) of the prompt-length mixture, in tokens
LENGTHS = ((0.35, "log", 16, 128), (0.40, "uniform", 128, 512), (0.25, "uniform", 512, 1024))
#: each shared prefix's length in tokens, BOS and the blank line included
PREFIXES = (32, 64, 128, 128, 256, 256, 384, 512)
SHARED = 0.6
MIN_BODY = 16
AT_END = 0.05
GAP = (4, 240)
CAPS = (64, 256)


def prompt_length(rng: random.Random) -> int:
    u, acc = rng.random(), 0.0
    for w, kind, lo, hi in LENGTHS:
        acc += w
        if u < acc:
            break
    if kind == "log":
        return min(hi, int(math.exp(rng.uniform(math.log(lo), math.log(hi + 1)))))
    return rng.randint(lo, hi)


def prefix_sources(docs: Sequence[Sequence[int]]) -> list[int]:
    """The first len(PREFIXES) documents long enough for the longest prefix."""
    out = [d for d, doc in enumerate(docs) if len(doc) >= max(PREFIXES) - 2][:len(PREFIXES)]
    if len(out) < len(PREFIXES):
        raise ValueError(f"the prefixes need {len(PREFIXES)} documents of {max(PREFIXES) - 2} tokens, not {len(out)}")
    return out


def prefixes(docs: Sequence[Sequence[int]]) -> list[list[int]]:
    """Group g's prefix from its source document (`prefix_sources`): BOS, its first PREFIXES[g] - 2 tokens, a blank line."""
    return [[BOS, *docs[d][:p - 2], BLANK_LINE] for d, p in zip(prefix_sources(docs), PREFIXES)]


def plan(docs: Sequence[Sequence[int]], n: int, seed: int = 0) -> list[dict]:
    """n requests over `docs` (token ids, add_special_tokens off): every document but the prefixes' sources is some
    request's body at most once. Raises when the documents run out."""
    rng = random.Random(seed)
    heads = prefixes(docs)
    sources = set(prefix_sources(docs))
    pool = sorted((d for d in range(len(docs)) if d not in sources), key=lambda d: (len(docs[d]), d))
    used: set[int] = set()
    weights = [1 / (g + 1) for g in range(len(PREFIXES))]
    out = []
    for i in range(n):
        length = prompt_length(rng)
        group = rng.choices(range(len(PREFIXES)), weights)[0] if rng.random() < SHARED else None
        if group is not None and PREFIXES[group] + MIN_BODY > length:
            fits = [g for g in range(len(PREFIXES)) if PREFIXES[g] + MIN_BODY <= length]
            group = max(fits, key=lambda g: (PREFIXES[g], -g)) if fits else None
        head = heads[group] if group is not None else [BOS]
        body_len = length - len(head)
        gap = 0 if rng.random() < AT_END else rng.randint(*GAP)
        doc = _take(docs, pool, used, body_len + gap, rng)
        end = len(docs[doc]) - gap
        body = list(docs[doc][end - body_len:end])
        cap = rng.choice(CAPS)
        out.append({"i": i, "prompt_ids": head + body, "prompt_len": length, "prefix": group,
                    "prefix_len": len(head) if group is not None else 0, "doc": doc, "gap": gap, "max_tokens": cap,
                    "gap_draw": rng.expovariate(1.0)})
    return out


def _take(docs, pool: list[int], used: set[int], need: int, rng: random.Random) -> int:
    """An unused document of at least `need` tokens, uniformly among the shortest 64 that qualify."""
    lo, hi = 0, len(pool)
    while lo < hi:
        mid = (lo + hi) // 2
        if len(docs[pool[mid]]) < need:
            lo = mid + 1
        else:
            hi = mid
    free = [d for d in pool[lo:lo + 4096] if d not in used][:64]
    if not free:
        raise ValueError(f"no unused document of {need} tokens is left")
    d = rng.choice(free)
    used.add(d)
    return d


def arrivals(requests: Sequence[dict], rate: float) -> list[float]:
    """Each request's arrival in seconds from the trace's start: the running sum of its unit-rate gap over `rate`."""
    t, out = 0.0, []
    for r in requests:
        t += r["gap_draw"] / rate
        out.append(t)
    return out
