"""Strassen-type schedules against the lead's committed words (HT-PREMISE round 4).

The lead (`verity/core/protocols/pouw/circuit/lead.py`, PR #6614; `benchmarks/pouw/np_kernel/`, PR #6630) commits every
k32 word of the stock int8 kernel: word(i, t, j) = word(i, t − 1, j) + Σ_{ℓ<32} A′[i, 32t + ℓ] B′[32t + ℓ, j], with
B′ the private weights at lanes 0–27 of each step, the certifier's odd coins s_t[·, j] in [−127, 127] at lanes 28–30
and 1 at lane 31, and A′ the call's activations at lanes 0–27, rotWeight(i) at 28–30 and 1 + b_t[i] at 31 (every step
is laid out as a step t ≥ 1; step 0's 21 data lanes are one step in S). This executes, on HT-PUBLIC's natural weights
at the lead's shapes, every schedule family the per-word commitment leaves open, and checks each committed word against
the stock kernel's at every step:

- `step`: Strassen or Winograd inside each k32 step, rows and columns split in two and the step's 32 lanes in two k16
  halves, from the previous committed word (the only k-split a step has: s8 MMA has nothing below k16);
- `cross`: Strassen or Winograd over two steps (k32 halves), from the word two steps back, with the middle step's word
  computed by the stock step (it is committed, and the scheme's products don't give it);
- an M,N-only split inside a step is `<a,1,c>`, whose rank is ac, the stock count, so it isn't executed.

Every operand form is split lo + 256·hi with lo in int8, and the products of hi run as their own correction, so the
words are exact by construction and asserted. Prices, per register written (W1; `H100Assumptions.lean`, the H200's
measured table): `h200` takes s8 m16n8k16 `mma.sync` at 32 a register as measured (32.010), wgmma k32 at 32, so two
k16 products chained into one accumulator pack into one k32 at 32; `half` is a hypothetical part with k16 s8 at 16 a
register (below `H32`), the attacker's best case. Adds: `fast` IADD3 (two terms a write), `plain` IADD (one).

Each scheme is realized at its cheapest within the family searched (`realizations`): per output and product, the
product either chains into that output's accumulator (an MMA from the running word, recomputed per output) or is
computed once and added. Forms of data lanes are free (both are fixed before the coins: B is registered, A committed
before the release); forms over the coin lanes are runtime and cost one add an entry (four int8 lanes a register
under `fast`, two under `plain`). Overflowing B-forms are corrected at the cheaper of one IMAD per output row and
nonzero (`alu`) and a gathered s8 product (natural.py's `corr`); A-forms (the activations, uniform int8 here, not a
natural sample) are priced free and unbounded, the attacker's best case, and only counted.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

from execute import SCHEMES
from natural import HF, MODELS, load_tensor, quantize, safetensors_header

LEAD_SHAPES = {  # the lead's four calls (PR #6630) on Llama-3.2-1B: name -> (block, projections fused along n)
    "qkv": ("self_attn", ("q_proj", "k_proj", "v_proj")),
    "o": ("self_attn", ("o_proj",)),
    "gate_up": ("mlp", ("gate_proj", "up_proj")),
    "down": ("mlp", ("down_proj",)),
}
DATA = 28
TABLES = {"h200": {"k16": 32, "k32": 32, "pack": True}, "half": {"k16": 16, "k32": 32, "pack": False}}
READINGS = {"fast": (2, 4), "plain": (1, 2)}  # (add terms a write, int8 lanes a register on the forming side)


def realizations(W, k_cost, pack, terms_per_write, k32=32):
    """The cheapest way to add each output's products to its running word, per quadrant word: each (output, product)
    incidence either chains (an MMA from the accumulator, `k_cost`; with `pack`, two chained products share one k32
    at `k32`) or adds a product computed once. Returns (cost, choice)."""
    inc = [(o, i) for o in range(4) for i in range(7) if W[o][i]]
    best = (math.inf, None)
    for bits in itertools.product((0, 1), repeat=len(inc)):
        chained = [0] * 4
        added = [0] * 4
        standalone = set()
        for (o, i), b in zip(inc, bits, strict=True):
            if b:
                chained[o] += 1
            else:
                added[o] += 1
                standalone.add(i)
        cost = len(standalone) * k_cost
        for o in range(4):
            cost += k32 * math.ceil(chained[o] / 2) if pack else k_cost * chained[o]
            cost += 32 * math.ceil(added[o] / terms_per_write)
        if cost < best[0]:
            best = (cost, bits)
    return best


def lead_weights(layer):
    repo = MODELS["llama3.2-1b"]
    url = f"{HF}/{repo}/resolve/main/model.safetensors"
    base, h = safetensors_header(url)
    out = {}
    for name, (blk, projs) in LEAD_SHAPES.items():
        parts = [quantize(load_tensor(url, base, h[f"model.layers.{layer}.{blk}.{p}.weight"])) for p in projs]
        out[name] = np.concatenate(parts, axis=0)
    return out


def operands(Wq, rows, rng):
    """A′ (rows × 32S) and B′ (32S × n) in the lead's step layout, with live-coin stand-ins drawn from `rng`."""
    n, K = Wq.shape
    S = -(-K // DATA)
    Ap = np.zeros((rows, 32 * S), np.int64)
    Bp = np.zeros((32 * S, n), np.int64)
    A = rng.integers(-127, 128, (rows, K))
    rot = np.where((np.arange(rows) % 2 == 0)[:, None], [127, 1, 1], [1, 127, 1])
    for t in range(S):
        k0, k1 = DATA * t, min(DATA * t + DATA, K)
        Ap[:, 32 * t : 32 * t + k1 - k0] = A[:, k0:k1]
        Ap[:, 32 * t + 28 : 32 * t + 31] = rot
        Ap[:, 32 * t + 31] = 1 + rng.integers(0, 2, rows)
        Bp[32 * t : 32 * t + k1 - k0] = Wq[:, k0:k1].T
        Bp[32 * t + 28 : 32 * t + 31] = 2 * rng.integers(0, 128, (3, n)) - 127
        Bp[32 * t + 31] = 1
    return Ap, Bp, S


def split8(F):
    hi = (F + 128) >> 8
    return F - 256 * hi, hi


def corr_cost(hi, m, alu=1.0):
    nz = hi != 0
    nnz = int(nz.sum())
    if not nnz:
        return 0.0
    s, t = int(nz.any(axis=1).sum()), int(nz.any(axis=0).sum())
    return min(32 * alu * m * nnz, m * s * t + 32 * m * t)


def mm(X, Y):
    return (X.astype(np.float64) @ Y.astype(np.float64)).astype(np.int64)


def runtime_adds(coeffs, live):
    """Adds a form needs after the coins, per entry of a live line: the pre-coin terms' sum is a table, so each live
    term is one add (all terms live: terms − 1)."""
    terms = sum(1 for c in coeffs if c)
    n_live = sum(1 for q, c in enumerate(coeffs) if c and live[q])
    return n_live if n_live < terms else terms - 1


def scheme_step(Ablk, Bblk, scheme, a_live, b_live, alu):
    """One 2 × 2 × 2 scheme over the blocks: the four quadrant increments (exact, through lo + 256·hi), the B-forms'
    correction work, the runtime form entries (adds), and the A-forms' overflowing entries. `a_live` and `b_live` say
    per block how many of its columns (A) or rows (B) carry coins."""
    U, V, Wm = SCHEMES[scheme]
    m2, c2 = Ablk[0].shape[0], Bblk[0].shape[1]
    M = []
    corr = 0.0
    adds = a_over = 0
    for u, v in zip(U, V, strict=True):
        Sa = sum(c * Ablk[q] for q, c in enumerate(u) if c)
        Tb = sum(c * Bblk[q] for q, c in enumerate(v) if c)
        lo, hi = split8(Tb)
        a_over += int(((Sa < -128) | (Sa > 127)).sum())
        M.append(mm(Sa, lo) + 256 * mm(Sa, hi))
        corr += corr_cost(hi, m2, alu)
        adds += runtime_adds(v, b_live) * max(b_live) * c2 + runtime_adds(u, a_live) * max(a_live) * m2
    C = [sum(Wm[o][i] * M[i] for i in range(7) if Wm[o][i]) for o in range(4)]
    return C, corr, adds, a_over


def run_matrix(name, layer, Wq, rows, seed, alu):
    rng = np.random.default_rng(seed)
    Ap, Bp, S = operands(Wq, rows, rng)
    n = Bp.shape[1]
    r2, c2 = rows // 2, n // 2
    stock_inc = [mm(Ap[:, 32 * t : 32 * t + 32], Bp[32 * t : 32 * t + 32]) for t in range(S)]
    words = np.cumsum(np.stack(stock_inc), axis=0)
    stock = 32.0 * rows * n * S
    out = {"matrix": name, "layer": layer, "rows": rows, "n": n, "K": Wq.shape[1], "steps": S, "stock": stock}
    for scheme in SCHEMES:
        # inside each k32 step: k16 halves
        corr = adds = a_over = 0
        word = np.zeros((rows, n), np.int64)
        for t in range(S):
            A = Ap[:, 32 * t : 32 * t + 32]
            B = Bp[32 * t : 32 * t + 32]
            Ablk = [A[:r2, :16], A[:r2, 16:], A[r2:, :16], A[r2:, 16:]]
            Bblk = [B[:16, :c2], B[:16, c2:], B[16:, :c2], B[16:, c2:]]
            C, c, ad, ao = scheme_step(Ablk, Bblk, scheme, (0, 1, 0, 1), (0, 0, 3, 3), alu)
            corr, adds, a_over = corr + c, adds + ad, a_over + ao
            new = word.copy()
            for qi, X in enumerate(C):
                rs = slice(0, r2) if qi < 2 else slice(r2, rows)
                cs = slice(0, c2) if qi % 2 == 0 else slice(c2, n)
                new[rs, cs] += X
            assert np.array_equal(new, words[t]), (name, scheme, "step", t)
            word = new
        for tab, pr in TABLES.items():
            for r, (tpw, lanes) in READINGS.items():
                rc, choice = realizations(SCHEMES[scheme][2], pr["k16"], pr["pack"], tpw)
                total = rc * r2 * c2 * S + corr + adds * 32 / lanes
                out[f"step_{scheme}_{tab}_{r}"] = total / stock
                out[f"step_{scheme}_{tab}_{r}_products_adds"] = rc / 128
                out[f"step_{scheme}_{tab}_{r}_choice"] = "".join(map(str, choice))
        out[f"step_{scheme}_corr"] = corr / stock
        out[f"step_{scheme}_form_fast"] = adds * 8 / stock
        out[f"step_{scheme}_a_overflow"] = a_over / (7 * r2 * 16 * S)

        # across two steps: k32 halves, the middle word by the stock step
        corr = adds = 0
        for t in range(0, S - 1, 2):
            A = Ap[:, 32 * t : 32 * t + 64]
            B = Bp[32 * t : 32 * t + 64]
            Ablk = [A[:r2, :32], A[:r2, 32:], A[r2:, :32], A[r2:, 32:]]
            Bblk = [B[:32, :c2], B[:32, c2:], B[32:, :c2], B[32:, c2:]]
            C, c, ad, _ = scheme_step(Ablk, Bblk, scheme, (1, 1, 1, 1), (3, 3, 3, 3), alu)
            corr, adds = corr + c, adds + ad
            prev = words[t - 1] if t else np.zeros((rows, n), np.int64)
            new = prev.copy()
            for qi, X in enumerate(C):
                rs = slice(0, r2) if qi < 2 else slice(r2, rows)
                cs = slice(0, c2) if qi % 2 == 0 else slice(c2, n)
                new[rs, cs] += X
            assert np.array_equal(new, words[t + 1]), (name, scheme, "cross", t)
            assert np.array_equal(prev + stock_inc[t], words[t]), (name, scheme, "cross-mid", t)
        npairs = len(range(0, S - 1, 2))
        for r, (tpw, lanes) in READINGS.items():
            rc, _choice = realizations(SCHEMES[scheme][2], 32, False, tpw)
            mid = 32.0 * rows * n * npairs
            rest = 32.0 * rows * n * (S - 2 * npairs)
            out[f"cross_{scheme}_{r}"] = (rc * r2 * c2 * npairs + mid + rest + corr + adds * 32 / lanes) / stock
        out[f"cross_{scheme}_corr"] = corr / stock
    return out


def work(layer, rows_list, seed, alu):
    W = lead_weights(layer)
    return [run_matrix(name, layer, Wq, rows, seed, alu) for name, Wq in W.items() for rows in rows_list]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--layers", default="0 8 15")
    ap.add_argument("--rows", default="64 16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alu", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for scheme, (_U, _V, W) in SCHEMES.items():
        for tab, pr in TABLES.items():
            for r, (tpw, _lanes) in READINGS.items():
                rc, choice = realizations(W, pr["k16"], pr["pack"], tpw)
                print(f"realization {scheme} {tab} {r}: {rc / 128:.4f} of stock (choice {''.join(map(str, choice))})")
    layers = [int(x) for x in args.layers.split()]
    rows_list = [int(x) for x in args.rows.split()]
    with (args.out / "lead.jsonl").open("w") as f, concurrent.futures.ProcessPoolExecutor(args.workers) as ex:
        for res in ex.map(work, layers, [rows_list] * len(layers), [args.seed] * len(layers), [args.alu] * len(layers)):
            for row in res:
                f.write(json.dumps(row) + "\n")
                f.flush()
                keys = [k for k in row if (k.startswith("step_") and k.count("_") == 3)
                        or (k.startswith("cross_") and k.endswith(tuple(READINGS)))]
                print(row["matrix"], row["layer"], row["rows"], " ".join(f"{k}={row[k]:.4f}" for k in keys), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
