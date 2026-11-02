"""The policy plane: the (c, k) values the formalism scores every record at, and a start's reference shapes
(note:training-20261102T1817Z-report-training-formalism-v1, section 2; c in Boolean gates, k in bits).

    python -m te_core.policy [START ...]    # each start's reference shapes and grid

A start's shape (`starts_store.json`'s `shape`: d_model, n_layer, vocab, ctx, parameters, the tokens it was trained
on, its forward MACs per token as the harness counts them) gives the two reference shapes and the reference policy:

- **inference block** (one token): c = one token's forward at full context, w_fwd = forward MACs per token at the
  bf16 GEMM price (te_core.gates); k = what that block reads beyond θ: its token id, ⌈log2 V⌉ bits, and its context's
  keys and values, (ctx − 1)·2·n_layer·d_model bf16 values;
- **reference** policy (the experiments draft's E0, note:training-20261102T1715Z-draft-training-efficiency-experiments):
  c = 4096·w_fwd, one 4,096-token session per block, and k = 10 kB, 80,000 bits, where honest serving still passes Fit
  (Δ = 0 and s·τ = 4096·⌈log2 V⌉ ≤ k);
- **server block**: the start's whole training in one block, c = C0 (its training's gates: `c0_gates` when its shape
  has the count, else 3 forward MACs per token per training token at the bf16 GEMM price, labelled), k = its training
  state beyond θ, P·(16 + 2·32) bits (new bf16 weights and AdamW's two f32 moments).

The grid is every decade of c and of k from the least to the greatest of those shapes' (`grid`), keyed
`c1e<i>/k1e<j>` (`key`). A start is scored at the grid's cells and at the three shapes exactly, keyed by name
(`cells`). P1, P2 and P4 read the reference policy (`REFERENCE`); P3 marks all three.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from .gates import per_mac

ACT_BITS = 16
#: bits per parameter of the training state beyond θ: the new weights in bf16 and AdamW's two f32 moments
STATE_BITS = 16 + 2 * 32
#: the reference policy's session, in tokens, and its k
SESSION_TOKENS = 4096
REFERENCE_K_BITS = 8 * 10_000
MARKS = ("inference", "reference", "server")
REFERENCE = "reference"
MANIFEST = Path(__file__).resolve().parents[1] / "starts_store.json"


def id_bits(vocab: int) -> int:
    return math.ceil(math.log2(vocab))


def c0(shape: dict) -> tuple[float, str]:
    """The start's training compute in gates, and where it comes from."""
    if shape.get("c0_gates"):
        return float(shape["c0_gates"]), "counted"
    if shape.get("c0_macs"):
        return shape["c0_macs"] * per_mac("gemm/bf16"), "the start's counted MACs, every MAC at the bf16 GEMM price"
    macs = 3 * shape["forward_macs_per_token"] * shape["tokens"]
    return macs * per_mac("gemm/bf16"), "stub: 3 x forward MACs per token x training tokens, bf16 GEMM"


def shapes(shape: dict) -> dict[str, dict]:
    """The start's reference shapes (module docstring), each {c, k, why}."""
    w_fwd = shape["forward_macs_per_token"] * per_mac("gemm/bf16")
    ids, ctx = id_bits(shape["vocab"]), shape["ctx"]
    kv = (ctx - 1) * 2 * shape["n_layer"] * shape["d_model"] * ACT_BITS
    c_s, how = c0(shape)
    return {
        "inference": {"c": w_fwd, "k": ids + kv, "why": "one token's forward at full context: its id and its context's K, V"},
        "reference": {"c": SESSION_TOKENS * w_fwd, "k": REFERENCE_K_BITS,
                      "why": "the draft's reference policy: one 4,096-token session per block, k = 10 kB"},
        "server": {"c": c_s, "k": shape["parameters"] * STATE_BITS,
                   "why": f"the start's training in one block (C0 {how}) and its state beyond theta"},
    }


def decades(lo: float, hi: float) -> list[float]:
    return [10.0 ** i for i in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1)]


def grid(shape: dict) -> list[tuple[float, float]]:
    """Every decade of c and k spanning the start's reference shapes."""
    ref = shapes(shape).values()
    cs = decades(min(r["c"] for r in ref), max(r["c"] for r in ref))
    ks = decades(min(r["k"] for r in ref), max(r["k"] for r in ref))
    return [(c, k) for c in cs for k in ks]


def cells(shape: dict) -> dict[str, tuple[float, float]]:
    """Every (c, k) a start is scored at: the grid's, keyed by decade (`key`), and its reference shapes', exactly, keyed
    by name (`MARKS`)."""
    return {key(c, k): (c, k) for c, k in grid(shape)} | {m: (s["c"], s["k"]) for m, s in shapes(shape).items()}


def key(c: float, k: float) -> str:
    return f"c1e{round(math.log10(c))}/k1e{round(math.log10(k))}"


def of_key(name: str) -> tuple[float, float]:
    c, k = name.split("/")
    return 10.0 ** int(c[3:]), 10.0 ** int(k[3:])


def starts(manifest: Path = MANIFEST) -> dict[str, dict]:
    """Each start's shape, from the starts manifest."""
    return {n: s["shape"] for n, s in json.loads(manifest.read_text())["starts"].items() if s.get("shape")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="*")
    a = ap.parse_args(argv)
    known = starts()
    for name in a.names or sorted(known):
        g = grid(known[name])
        cs, ks = sorted({c for c, _ in g}), sorted({k for _, k in g})
        print(f"{name}: c 1e{math.log10(cs[0]):.0f}..1e{math.log10(cs[-1]):.0f} gates, k 1e{math.log10(ks[0]):.0f}.."
              f"1e{math.log10(ks[-1]):.0f} bits ({len(g)} cells)")
        for m, s in shapes(known[name]).items():
            print(f"  {m:9} c {s['c']:.3g} gates, k {s['k']:.3g} bits: {s['why']}" + (" (reference)" if m == REFERENCE else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
