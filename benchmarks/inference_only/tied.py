"""The served class under the value tie, priced for `run.py`: the bound over a monitoring period at `tied.toml`'s
policy (`min_cut.bound_charged`), each figure beside the share of a block's work that no PoUW check credits.

Under the tie a block reads as fresh input only its scan state row and the execution phase's salt, so the count gives a
number far above the served policy's 1. Two things stand between that number and a claim, and `priced` gives it with
both:

* **the untied words** (`policy.Tied`): outputs of the linears that PoUW's sample accepted without tying, at most a
  node's bound for each block of one of its execution phases. A fleet's period holds `Tied.untied` of them, fed by at
  most `macs_a_word` MACs each, and one block may hold as many as it has output words of the linears. They enter the
  count as `bound_charged` takes them, under each of its readings (`Charged.reading`): one `β_x` with the threshold at
  which the figure is largest (`best_w`), one `β_x` at the top budget with no block charged, and a `β_x` a block;
* **the uncredited share** `s`: the share of a block's work that no PoUW check credits, which the caller states. A
  developer gets that share whole, so each figure `ρ` is quoted with its limit `1 / (s + 1/ρ)` (`min_cut.capped`). The
  words that work computes are tied by nothing either: a fourth reading counts them as fresh input, `k + word · u` with
  `u` the most words `s` of a block's work feeds (a word of a linear's output is fed by at least the least fan-in of
  the model's linears), under one `β_x` with its own best threshold.

Every row is scaled: the blocks are a fleet's at its data sheet's rate (`policy.fleet_periods`), and a node's bound on
its untied words is one execution phase's, scaled by the blocks the node serves in a phase. That bound doesn't grow
with the period, so the figure does: each fleet's one execution phase, where a block's untied words are the most, is
priced first, beside its periods.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import policy as P
from rho import min_cut

#: the fourth reading's name: `Charged.reading` at the `k` that holds the uncredited words, and this
IN_K = "the uncredited words as fresh input"
#: the span that is one execution phase, beside the monitoring periods
PHASE = "execution phase"


def output_words(p: P.Policy, sh: P.Shape) -> int:
    """The output words of the layers' linears in one block of ``p``'s class: the most untied words one block reads."""
    return P.class_of(p.cls).positions * sh.layers * sum(n for _, n, _ in sh.linears)


def spans(t: P.Tied, work: float) -> Iterator[dict[str, Any]]:
    """`policy.fleet_periods`' rows, each fleet's one execution phase before its monitoring periods."""
    seen = set()
    for r in P.fleet_periods(work):
        if r["where"] not in seen:
            seen.add(r["where"])
            yield {**r, "period": PHASE, "seconds": t.phase_seconds,
                   "blocks": r["blocks"] * t.phase_seconds / r["seconds"]}
        yield r


def priced(t: P.Tied, sh: P.Shape, s: float) -> dict[str, Any]:
    """The tied policy's figures at the uncredited share ``s``: one block's, then each of the calculator's fleets over
    one execution phase and over each monitoring period under each reading, every figure with its limit at ``s`` and
    the pooled total in gates."""
    p, N = t.policy, t.policy.work[P.TOTAL]
    work = {"N": N, "O": N * sh.outputs * min_cut.BF16 / sh.macs, "size": p.c}
    most, fan_in = output_words(p, sh), min(k for _, _, k in sh.linears)
    words = min(s * N / fan_in, most)
    k_words = p.k + t.word_bits * words
    one, with_words = min_cut.bound(p.c, p.k, p.n, **work), min_cut.bound(p.c, k_words, p.n, **work)
    rows = []
    for r in spans(t, N):
        untied = t.untied(r["blocks"], r["seconds"])
        terms = {"blocks": r["blocks"], "untied": untied, "untied_macs": t.macs_a_word * untied, "most": most,
                 "word": t.word_bits, **work}
        readings = [(min_cut.bound_charged(p.c, p.k, w=min_cut.best_w(p.c, p.k, **terms), **terms), False),
                    (min_cut.bound_charged(p.c, p.k, **terms), False),
                    (min_cut.bound_charged(p.c, p.k, rows=True, **terms), False),
                    (min_cut.bound_charged(p.c, k_words, w=min_cut.best_w(p.c, k_words, **terms), **terms), True)]
        for got, in_k in readings:
            limit, gates = min_cut.capped(got.slowdown, s), r["blocks"] * p.c
            rows.append({
                "where": r["where"], "fleet": r["fleet"], "period": r["period"], "blocks": r["blocks"],
                "untied": untied, "untied_a_block": untied / r["blocks"], "untied_macs": got.untied_macs,
                "reading": f"{got.reading}, {IN_K}" if in_k else got.reading, "uncredited_words_in_k": in_k,
                "k": got.k, "w": None if got.w == float("inf") else got.w, "charged": got.charged,
                "number": got.slowdown, "limit": limit,
                "gates": {"blocks": gates, "training_by_the_count": got.gates, "training_at_the_limit": gates / limit},
                "share": 1 / limit, "statement": got.statement, "premises": list(got.premises),
                "rests_on": got.rests_on})
    return {
        "label": p.label, "model": p.model, "class": list(p.cls), "source": p.source, "c": p.c, "k": p.k, "n": p.n,
        "units": P.UNITS, "k_by_kind": [{"kind": kind, "bits": bits, "how": how} for kind, bits, how in p.parts],
        "work": {**{x: p.work[x] for x in P.WORK}, "N": N}, "work_is": P.WORK_IS,
        "untied": {"words": t.words, "blocks": t.blocks, "phase_seconds": t.phase_seconds,
                   "macs_a_word": t.macs_a_word, "word_bits": t.word_bits, "most_a_block": most},
        "uncredited": {"share": s, "from": "the command line", "pouw_verdict": None, "macs_a_block": s * N,
                       "words_a_block": words, "least_fan_in": fan_in, "k_with_words": k_words},
        "block": {"number": one.slowdown, "limit": min_cut.capped(one.slowdown, s), "rests_on": one.rests_on,
                  "with_words": {"k": k_words, "number": with_words.slowdown,
                                 "limit": min_cut.capped(with_words.slowdown, s), "rests_on": with_words.rests_on}},
        "periods": rows,
    }
