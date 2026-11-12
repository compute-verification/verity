"""What recorded arrivals carry: the bits an interval a server's bit moves through when that interval reaches the
client.

A sample is one release interval: how long after the interval's release time its bytes reached the observer, in
seconds, and the bit the server chose for it. `measure` fits a decoder on the even samples and scores it on the odd
ones, so the score is of a decoder fixed before the bits it reads.

**The number.** A decoder wrong at rate e on uniform bits leaves H(bit | arrivals) ≤ H(e) (Fano), so the arrivals carry
at least 1 − H(e) bits an interval, 0 at e ≥ 1/2: `bits`. It bounds from below what this server's writes move, under
this decoder; nothing here bounds a channel from above. `lower` and `upper` are of e's 95% Wilson interval over the
scored samples, which takes the scored intervals' errors as independent: `lower` is `bits` at its upper end, and
`upper` is 1 − H at its end farther from one half (`carried`), since a decoder wrong more often than not carries as
much read the other way up. So an error rate that comes out over one half never makes `upper` small. For a server
whose bits move nothing the interval is exact: the bits are uniform and independent of every arrival, so each scored
guess is wrong with probability one half whatever the arrivals did.

**The decoder** is one threshold on one feature, both chosen on the even samples: the offset itself, or `step`, the
offset less the mean of the `STEP_WINDOW` offsets around it without their `STEP_DROP` largest. `step` drops what moves
slower than its window: a drift between two hosts' clocks, which an even window cancels, and a host's slow changes;
the dropped largest keep one late interval out of its neighbours' steps. `information` is the read that isn't a
threshold: the plug-in mutual information of the bit and the offset's bin (`BINS` equal-frequency bins) less its mean
over `SHUFFLES` shuffles of the bits, with the share of the shuffles that reach it. A spread that differs by bit shows
there and not in `bits`.

`target` and `samples_to_show` state the budget's side: what the whole timing channel may carry an interval, and how
many scored intervals it takes for a decoder at chance to have its upper bound under that. The module imports only
the standard library.
"""

from __future__ import annotations

import bisect
import math
import random
import statistics
from collections import Counter

#: the two-sided 95% normal quantile, `wilson`'s and `samples_to_show`'s
Z95 = 1.959963984540054
#: the two-sided normal quantile of `samples_to_show`'s power, nine runs in ten
Z_POWER = 1.6448536269514722
#: the offsets around a sample, half before and half after, whose mean `step` subtracts, and how many of their
#: largest it leaves out
STEP_WINDOW, STEP_DROP = 16, 2
#: `information`'s bins of the offset, and its shuffles of the bits
BINS, SHUFFLES = 8, 200
#: the budget (a world-year's exfiltration against the compute it takes): the seconds one node's whole timing channel
#: may take to carry a bit, at the budget's two ends
SECONDS_A_BIT = (240.0, 600.0)


def entropy(p: float) -> float:
    """H(p), in bits."""
    return 0.0 if p <= 0.0 or p >= 1.0 else -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def bits(error: float) -> float:
    """1 − H(error), the bits an interval a binary decoder wrong at rate `error` shows; 0 from one half on."""
    return 1.0 - entropy(error) if error < 0.5 else 0.0


def carried(low: float, high: float) -> float:
    """The most bits an interval a binary decoder's answers carry, read either way up, when its error rate is between
    `low` and `high`: 1 − H at the end farther from one half."""
    return 1.0 - entropy(low if 0.5 - low >= high - 0.5 else high)


def wilson(wrong: int, n: int, z: float = Z95) -> tuple[float, float]:
    """The Wilson interval of an error rate of `wrong` in `n`."""
    if n <= 0:
        return 0.0, 1.0
    p, zz = wrong / n, z * z / n
    mid, half = (p + zz / 2) / (1 + zz), z * math.sqrt(p * (1 - p) / n + zz / (4 * n)) / (1 + zz)
    return max(0.0, mid - half), min(1.0, mid + half)


def steps(offsets: list[float]) -> list[float]:
    """Each offset less the mean of the `STEP_WINDOW` offsets around it, half before and half after (those there
    are, at an end), without their `STEP_DROP` largest; 0 for an offset alone."""
    half, out = STEP_WINDOW // 2, []
    for i, x in enumerate(offsets):
        near = sorted(offsets[max(0, i - half):i] + offsets[i + 1:i + 1 + half])
        kept = near[:max(1, len(near) - STEP_DROP)]
        out.append(x - sum(kept) / len(kept) if kept else 0.0)
    return out


def fit(pairs: list[tuple[float, int]]) -> tuple[float, int, int]:
    """The threshold and side (1: bit 1 above it, −1: below) that read the most of `pairs`' bits right, and how many
    they read wrong. With no threshold better than one answer for all, the threshold is −inf."""
    xs = sorted(pairs)
    ones, n = sum(b for _, b in xs), len(xs)
    best, below_ones = (max(ones, n - ones), -math.inf, 1 if ones >= n - ones else -1), 0
    for i in range(n - 1):
        below_ones += xs[i][1]
        if xs[i][0] == xs[i + 1][0]:
            continue
        above = (i + 1 - below_ones) + (ones - below_ones)
        cut = (xs[i][0] + xs[i + 1][0]) / 2
        best = max(best, (above, cut, 1), (n - above, cut, -1))
    return best[1], best[2], n - best[0]


def decode(x: float, cut: float, side: int) -> int:
    return int(side * (x - cut) > 0)


def _plug_in(bins: list[int], coins: list[int]) -> float:
    n, joint = len(coins), Counter(zip(bins, coins))
    pb, px = Counter(coins), Counter(bins)
    return sum(c / n * math.log2(c * n / (px[x] * pb[b])) for (x, b), c in joint.items())


def information(samples: list[tuple[float, int]], seed: int = 0) -> dict:
    """The mutual information of the bit and the offset's bin, in bits an interval: the plug-in estimate less its
    mean over the shuffles (never under 0), and the share of the shuffles that reach the plug-in estimate (`p`)."""
    xs = sorted(x for x, _ in samples)
    cuts = [xs[len(xs) * k // BINS] for k in range(1, BINS)]
    bins, coins = [bisect.bisect_right(cuts, x) for x, _ in samples], [b for _, b in samples]
    raw, rng, shuffled = _plug_in(bins, coins), random.Random(seed), coins[:]
    null = []
    for _ in range(SHUFFLES):
        rng.shuffle(shuffled)
        null.append(_plug_in(bins, shuffled))
    return {"bits": max(0.0, raw - statistics.fmean(null)), "p": (1 + sum(x >= raw for x in null)) / (1 + SHUFFLES)}


def measure(samples: list[tuple[float, int]], seed: int = 0) -> dict:
    """What `samples`, (offset in seconds, bit) in interval order, carry (the module docstring): `bits` with `lower`
    and `upper`, the scored samples `n` of `samples` in all, the decoder (its feature, threshold in µs, side, its
    errors on the even samples it was fitted on and on the odd ones it is scored on), bit 1's median offset less bit
    0's in µs (`shift_us`), and `information`. A decoder with no threshold, one answer for every interval, reads
    nothing: its `bits` is 0 whatever the scored bits' balance. Fewer than four samples, or one bit only among the
    even ones, measure nothing: `bits` is None."""
    out: dict = {"samples": len(samples), "ones": sum(b for _, b in samples), "n": 0, "bits": None, "lower": None,
                 "upper": None}
    train, test = samples[0::2], samples[1::2]
    if len(test) < 2 or len({b for _, b in train}) < 2:
        return out
    offsets = [x for x, _ in samples]
    features = {"offset": offsets, "step": steps(offsets)}
    fitted = {name: fit([(xs[i], samples[i][1]) for i in range(0, len(samples), 2)]) for name, xs in features.items()}
    name = min(fitted, key=lambda k: (fitted[k][2], k))
    cut, side, train_wrong = fitted[name]
    xs = features[name]
    wrong = sum(decode(xs[i], cut, side) != samples[i][1] for i in range(1, len(samples), 2))
    low, high = wilson(wrong, len(test))
    by_bit = {b: [x for x, c in samples if c == b] for b in (0, 1)}
    read = not math.isinf(cut)
    return out | {
        "n": len(test), "bits": bits(wrong / len(test)) if read else 0.0, "lower": bits(high) if read else 0.0,
        "upper": carried(low, high),
        "decoder": {"feature": name, "threshold_us": None if math.isinf(cut) else round(cut * 1e6, 3), "side": side,
                    "train_n": len(train), "train_wrong": train_wrong, "wrong": wrong,
                    "error": wrong / len(test), "error_95": [low, high]},
        "shift_us": round((statistics.median(by_bit[1]) - statistics.median(by_bit[0])) * 1e6, 3),
        "information": information(samples, seed)}


def pool(measures: list[dict]) -> dict:
    """Several runs of one case as one: each run's decoder is its own, their scored samples and errors add."""
    scored = [m for m in measures if m.get("bits") is not None]
    n, wrong = sum(m["n"] for m in scored), sum(m["decoder"]["wrong"] for m in scored)
    if not n:
        return {"runs": len(measures), "n": 0, "bits": None, "lower": None, "upper": None}
    low, high = wilson(wrong, n)
    return {"runs": len(measures), "n": n, "wrong": wrong, "error": wrong / n, "bits": bits(wrong / n),
            "lower": bits(high), "upper": carried(low, high)}


def target(interval_s: float) -> dict:
    """The budget at a release interval of `interval_s` seconds: the bits an interval the certifier's whole timing
    channel may carry, at the budget's two ends (`SECONDS_A_BIT`)."""
    return {"interval_s": interval_s, "seconds_a_bit": list(SECONDS_A_BIT),
            "bits": [interval_s / s for s in SECONDS_A_BIT]}


def _margin(value: float) -> float:
    """d with 1 − H(1/2 − d) = `value`: how far under one half an error rate shows `value` bits."""
    low, high = 0.0, 0.5
    for _ in range(200):
        mid = (low + high) / 2
        low, high = (mid, high) if bits(0.5 - mid) < value else (low, mid)
    return high


def samples_to_show(value: float) -> dict:
    """The scored intervals it takes to show a number of at most `value` bits an interval, for a decoder truly at
    chance: `least`, with which a run whose error rate comes out at exactly one half has its upper bound at `value`
    (every other run shows more, so no run of fewer shows it), and `nine_in_ten`, with which nine runs in ten show
    it. A run scores half its intervals, so it lasts twice either count. A channel at or over `value` is never shown
    under it, and a channel between 0 and `value` takes more."""
    d = _margin(value)
    return {"bits": value, "least": math.ceil(max(0.0, Z95 ** 2 / (4 * d * d) - Z95 ** 2)),
            "nine_in_ten": math.ceil(((Z95 + Z_POWER) / (2 * d)) ** 2)}
