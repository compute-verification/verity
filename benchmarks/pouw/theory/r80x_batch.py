"""R80X and RDX in ShaWrites-R80's batch form: what a window's chains share at zero data, on FOLD-ROUNDS' W1 model.

The batch form is about a window's m chains, whose words at zero data are row walks plus column walks: chain τ holds
rows 2τ and 2τ + 1, and its message word k of step s is (lo, hi) = (w(2τ, k), w(2τ + 1, k)) with
w(i, k) = x_s + R_s(i) + C_s(k) mod 2^32 (`hash_rounds.decomp_r80`'s layout; walks of 14-bit increments, as
`fold_dynamic.fold32`'s). RDX's V word of the same position reads a second set of columns. Every chain reads the same
column walks, so a cross-chain prover can compute any value of the columns, the coins and a public start once a window
and share it. Checked bit for bit on 64 chains of 160 positions:

- **Shared writes** (`hash_rounds.Regs(fold_consts=True)`: an op whose inputs are equal on every chain is done once a
  window). From a public start the chains share only round 0's writes that read the start state alone (as
  `hash_rounds.head_saving` counts on random words), for R80X and RDX alike; from a live or carried start, none. The
  zero-data words add no shared write.
- **RDX's raw V, the word skip on e' = d + T1 + V.** V's column part C_V (both halves) goes into U's K slot, so
  T1~ = T1 + C_V; e' = d + T1~ + R_V, with V's lo carry corrected by predicates on IADD3.X's carry-ins; and
  a' = T1~ + S0 + Maj − C_V, a fourth term, one more IADD3 pair. V is never formed (its 2 MMA writes saved) and the
  round pays 2 more: a tie with honest, and worse where the corrected carry-in needs more than IADD3.X's two
  carry-ins. No other slot takes V's parts: U's three terms and T1's are full, a and e feed S0, Maj, S1 and Ch (no
  offset passes them), and d and h are earlier rounds' a and e.
- **R80X's XORed W, the same skip on U = h + W' + K.** `hash_rounds.decomp_r80(xor=True)`'s program is wrong (the XOR
  doesn't pass through the add) and its raw-word control is exact; AddXor (FOLD-ROUNDS, checked for one write) says
  no instruction merges the XOR with either add.

What it doesn't see: a shared part combined with a per-chain part in some other way than the two skips above (the
census counts only values equal across chains), and routes that change the write count of a round itself
(ShaWrites-R80's single-chain form, FOLD-ROUNDS' `falsify`).

Run: `uv run --no-project --with numpy --with-editable ../exact_kit python -B r80x_batch.py --rounds=PATH/hash_rounds.py
[--mutations] [--json=OUT]` (from this directory; PATH holds FOLD-ROUNDS' `hash_rounds.py`).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exact_kit"))
from exact_kit.checks import Checks

M32 = (1 << 32) - 1
M64 = (1 << 64) - 1
CHAINS, COLS, STEPS = 64, 16, 10


def load(path: str):
    spec = importlib.util.spec_from_file_location("hash_rounds", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def batch(rng: np.random.Generator, v_words: bool, span: int = 14) -> dict:
    """A window's zero-data words: per chain the (lo, hi) pairs of W (and V) at each position, and their parts. The
    walks take increments of `span` bits (the lead's 14; 32 makes V's lo half wrap about half the time)."""
    n_cols = 2 * COLS if v_words else COLS
    R = np.cumsum(rng.integers(0, 2**span, (STEPS, 2 * CHAINS), dtype=np.uint64), axis=0) & M32
    C = np.cumsum(rng.integers(0, 2**span, (STEPS, n_cols), dtype=np.uint64), axis=0) & M32
    x = rng.integers(0, 2**32, STEPS, dtype=np.uint64)

    def pair(s: int, k: int) -> np.ndarray:
        lo = (x[s] + R[s, 0::2] + C[s, k]) & M32
        hi = (x[s] + R[s, 1::2] + C[s, k]) & M32
        return hi << np.uint64(32) | lo

    pos = [(s, k) for s in range(STEPS) for k in range(COLS)]
    ws = [pair(s, k) for s, k in pos]
    vs = [pair(s, COLS + k) for s, k in pos] if v_words else None
    return {"R": R, "C": C, "x": x, "pos": pos, "ws": ws, "vs": vs}


def census(hr, design: dict, b: dict, st0: list[np.ndarray]) -> tuple[int, int]:
    """(honest writes a chain, of them the writes every chain shares): `Regs` counts one SIMT program over the chains."""
    pairs = lambda xs: [hr.to_pair(v) for v in xs] if xs is not None else None  # noqa: E731
    full, folded = hr.Regs(), hr.Regs(fold_consts=True)
    a = hr.fold_w1(full, design, pairs(st0), pairs(b["ws"]), pairs(b["vs"]))
    c = hr.fold_w1(folded, design, pairs(st0), pairs(b["ws"]), pairs(b["vs"]))
    assert all((x[0] == y[0]).all() and (x[1] == y[1]).all() for x, y in zip(a, c))
    return full.total(), full.total() - folded.total()


def rdx_honest(hr, b: dict, st0: list[np.ndarray]) -> tuple[list[np.ndarray], int]:
    r = hr.Regs()
    st = hr.fold_w1(r, hr.DESIGNS["RDX"], [hr.to_pair(v) for v in st0], [hr.to_pair(v) for v in b["ws"]],
                    [hr.to_pair(v) for v in b["vs"]])
    return [hr.from_pair(p) for p in st], r.total() + 4 * len(b["ws"])


def rdx_vskip(hr, b: dict, st0: list[np.ndarray], mutate: str | None = None):
    """RDX with V never formed: (state, writes a chain with W's MMA writes, shared writes, carries seen)."""
    r, shared = hr.Regs(), hr.Regs()
    st = [hr.to_pair(v) for v in st0]
    carries: set[int] = set()
    R, C, x = b["R"], b["C"], b["x"]
    for i, (s, k) in enumerate(b["pos"]):
        p = i % 80
        cv = int((x[s] + C[s, COLS + k]) & M32)
        cv64 = cv << 32 | cv
        kc = (hr.K[p] + cv64) & M64
        neg = (-cv64) & M64
        shared.n["IADD3"] += 4
        a_, b_, c_, d_, e_, f_, g_, h_ = st
        w = hr.xor_w1(r, hr.to_pair(b["ws"][i]), hr.xc(p))
        u = r.add(h_, w, kc)
        t1 = r.add(u, hr.sig_w1(r, "S1", e_), hr.ch_w1(r, e_, f_, g_))
        s0, mj = hr.sig_w1(r, "S0", a_), hr.maj_w1(r, a_, b_, c_)
        r_lo, r_hi = R[s, 0::2], R[s, 1::2]
        lo = d_[0].astype(np.uint64) + t1[0] + r_lo
        borrow = (r_lo + np.uint64(cv)) >> np.uint64(32) if mutate != "no carry fix" else np.uint64(0)
        carry = (lo >> np.uint64(32)).astype(np.int64) - borrow.astype(np.int64)
        carries |= {int(v) for v in np.unique(carry)}
        hi = (d_[1].astype(np.int64) + t1[1].astype(np.int64) + r_hi.astype(np.int64) + carry) & M32
        r.n["IADD3"] += 2
        e2 = ((lo & np.uint64(M32)).astype(np.uint32), hi.astype(np.uint32))
        a2 = r.add(t1, s0, mj)
        if mutate != "no a' fix":
            a2 = r.add(a2, neg)
        st = [a2, a_, b_, c_, e2, e_, f_, g_]
    return [hr.from_pair(p) for p in st], r.total() + 2 * len(b["ws"]), shared.total(), carries


def main() -> int:
    checks = Checks()
    arg = lambda key: next((a.split("=", 1)[1] for a in checks.args if a.startswith(f"--{key}=")), None)  # noqa: E731
    hr = load(arg("rounds") or str(Path(__file__).with_name("hash_rounds.py")))
    rng = np.random.default_rng(80)
    report: dict = {"chains": CHAINS, "positions": STEPS * COLS, "census": {}, "rdx_v": {}, "r80x_w": {}}

    public = [np.full(CHAINS, v, dtype=np.uint64) for v in hr.IV]
    live = [rng.integers(0, 2**64, CHAINS, dtype=np.uint64) for _ in range(8)]
    for name in ("R80X", "RDX"):
        d = hr.DESIGNS[name]
        b = batch(rng, v_words=name == "RDX")
        head = hr.head_saving(d, np.random.default_rng(1))
        rows = {}
        for start, st0 in (("public", public), ("live", live)):
            total, sh = census(hr, d, b, st0)
            rows[start] = {"writes_a_chain": total, "shared_a_chain": sh}
        rows["head_saving_random_words"] = head
        report["census"][name] = rows
        checks.check(f"{name}: from a public start a window's zero-data chains share only the head (round 0's writes "
                     "on the start state alone), as on random words; from a live start, nothing",
                     rows["public"]["shared_a_chain"] == head and rows["live"]["shared_a_chain"] == 0,
                     f"public {rows['public']['shared_a_chain']} a chain (random words {head}), "
                     f"live {rows['live']['shared_a_chain']}")

    for span in (14, 32):
        b = batch(rng, v_words=True, span=span)
        hs, hw = rdx_honest(hr, b, live)
        ks, kw, ksh, carries = rdx_vskip(hr, b, live)
        exact = all((x == y).all() for x, y in zip(hs, ks))
        report["rdx_v"][f"span{span}"] = {"exact": exact, "honest_writes": hw, "skip_writes": kw,
                                          "skip_shared_writes": ksh, "carries": sorted(carries),
                                          "per_round_honest": hw / len(b["ws"]), "per_round_skip": kw / len(b["ws"])}
        checks.check(f"RDX, {span}-bit walk increments: the V skip (C_V in U's K slot, R_V in e', C_V taken back out "
                     "of a') equals honest on every chain, with V's lo carry corrected by predicates", exact,
                     f"carry-ins {sorted(carries)}")
        checks.check(f"RDX, {span}-bit walk increments: the V skip writes as much as honest a round (2 MMA writes "
                     "saved, 2 IADD3 writes paid) before its shared writes, so raw V ties", kw == hw,
                     f"{kw / len(b['ws']):g} against {hw / len(b['ws']):g} a round a chain")
    checks.check("RDX: the corrected carry-in fits IADD3.X's two carry-ins (0..2) at the lead's walks and takes four "
                 "values (-1..2) at full-width ones, one more than two carry-ins hold after any bias, so there the V "
                 "skip pays past the tie on some positions",
                 set(report["rdx_v"]["span14"]["carries"]) <= {0, 1, 2}
                 and set(report["rdx_v"]["span32"]["carries"]) == {-1, 0, 1, 2},
                 {k: v["carries"] for k, v in report["rdx_v"].items()})

    xok, _ = hr.decomp_r80(np.random.default_rng(2), threads=2, cols=16, steps=5, xor=True)
    rok, _ = hr.decomp_r80(np.random.default_rng(3), threads=2, cols=16, steps=5, xor=False)
    report["r80x_w"] = {"decomp_r80_xor_exact": xok, "decomp_r80_raw_exact": rok}
    checks.check("R80X: the word skip on U = h + W' + K is wrong with the XOR and exact on raw words (decomp_r80)",
                 rok and not xok)

    if checks.mutations:
        b = batch(np.random.default_rng(4), v_words=True, span=32)
        hs = rdx_honest(hr, b, live)[0]
        for m in ("no carry fix", "no a' fix"):
            ms = rdx_vskip(hr, b, live, mutate=m)[0]
            checks.mutation(f"RDX V skip with {m}", not all((x == y).all() for x, y in zip(hs, ms)))
    out = arg("json")
    if out:
        Path(out).write_text(json.dumps(report, indent=1, default=int) + "\n")
    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
