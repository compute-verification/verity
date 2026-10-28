"""The bench's fixed methodology: eval sets, contexts and their windows, seeds, calibration, batching and the significance
rule. Nothing here is a parameter of a run; `run.py eval` refuses any other window list, context or seed."""
from __future__ import annotations

import math

#: the held-out text each eval set reads (datasets.json pins the files)
EVAL_SETS = ("c4", "wikitext2")
#: context length -> how many windows: windows 0..n-1 of the set's token stream cut at that length
CONTEXTS = {2048: 16, 8192: 16, 32768: 4}
#: every arm whose result depends on a seed runs all of them; a deterministic arm runs the first
SEEDS = (0, 1, 2)
#: SmoothQuant's maxima and GPTQ's second moments: C4 windows 100..115 at 2048, disjoint from every eval window of C4 at
#: every context (16 x 8191 and 4 x 32767 tokens both end before window 100 at 2048 starts, at 100 x 2047)
CALIBRATION = {"set": "c4", "ctx": 2048, "windows": tuple(range(100, 116))}
#: C4: the first 3000 documents of the validation shard (enough for every window above)
C4_DOCS = 3000
#: the bf16 reference and calibration streams run this many consecutive windows per forward; arms run one window per
#: forward, since their coins are drawn per window
REF_BATCH = 4
#: the window on which every exactness gate that is too costly for every window runs (peels, recomputed products, the
#: HF reference layer), on every layer
CHECK_WINDOW = 0
#: one-sided level of the paired sign test over windows
SIGN_ALPHA = 0.025
#: how far apart two measurements of one arm on one (model, set, context) may be and still be the same measurement,
#: relative, beyond their seed ranges: kernel shapes and RoPE tables move a deterministic arm by up to ~0.07%
REPRODUCE_FLOOR = 0.001


def windows(ctx: int) -> tuple[int, ...]:
    if ctx not in CONTEXTS:
        raise ValueError(f"context {ctx}: the bench's contexts are {sorted(CONTEXTS)}")
    return tuple(range(CONTEXTS[ctx]))


def scored_tokens(ctx: int) -> int:
    """Tokens a run's perplexity averages over: every position of every window but the first."""
    return CONTEXTS[ctx] * (ctx - 1)


def sign_threshold(n: int) -> int | None:
    """The fewest of n windows that must favour one arm for the paired sign test to reject equality at SIGN_ALPHA
    (one-sided, Binomial(n, 1/2)), or None when even n of n doesn't (n < 6)."""
    tail = 0.0
    for k in range(n, -1, -1):
        tail += math.comb(n, k) / 2 ** n
        if tail > SIGN_ALPHA:
            return k + 1 if k < n else None
    return None


def compare(a_seeds: list[float], b_seeds: list[float], a_windows: list[float], b_windows: list[float]) -> str:
    """The significance rule for two arms on one (model, set, context): `<` (a is better) or `>` when both hold:
    their seed ranges of perplexity are disjoint, and the paired sign test over windows on the seed-mean NLL of each
    window favours the same arm (at least `sign_threshold(n)` of n windows; with fewer than 6 windows, every window,
    and the verdict is marked weak). Otherwise `≈`. Each seed's perplexity varies the coins; each window's NLL varies
    the text, so a verdict needs both to agree."""
    if len(a_windows) != len(b_windows):
        raise ValueError("the arms ran different windows")
    n = len(a_windows)
    lo = max(a_seeds) < min(b_seeds)
    hi = min(a_seeds) > max(b_seeds)
    wins = sum(x < y for x, y in zip(a_windows, b_windows))
    losses = sum(x > y for x, y in zip(a_windows, b_windows))
    need = sign_threshold(n)
    weak = need is None
    need = n if need is None else need
    if lo and wins >= need:
        return "< (weak)" if weak else "<"
    if hi and losses >= need:
        return "> (weak)" if weak else ">"
    return "≈"


def reproduces(lo: float, hi: float, other_lo: float, other_hi: float) -> bool:
    """Whether two measurements of one arm agree: their perplexity ranges over seeds (a single value for one seed),
    the first widened by REPRODUCE_FLOOR on each side, overlap."""
    return lo * (1 - REPRODUCE_FLOOR) <= other_hi and other_lo <= hi * (1 + REPRODUCE_FLOOR)
