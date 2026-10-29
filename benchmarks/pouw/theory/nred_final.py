"""N-RED final: A2r (the cancelling lane with revisits), a certifier-side distinctness check for R1, and grinding.

**A2r.** A2c (interim 2) cancels the public part of each later step's noise with one chosen data lane, so a row's
word repeats when b_t = 0 (51.6% of stock). A2r alternates the cancel: at odd steps the increment is b_t ∈ {0, 1}, at
even steps b_t − 1 ∈ {−1, 0}. Each row's words then walk lazily around their step-1 value and revisit a few values.
Under model P (a price per fresh word, moves free) a revisited value is a copy of a register still held, so only the
distinct values are written.

- `ssn-1l-tag`: later noise (1 + b_t)·1 is column-constant, so every column of a row walks together. A 32-lane (or
  4-lane) register at walk position p is the same vector each time, so models W and C save as much as P.
- `ssn-1l-hw`: later noise (1 + 2b_t)·σ_t[j] with σ public. A per-column cancel B_t[j] = −σ_t[j]·(1 + 2e_t[j]), with
  e_t[j] picked from σ_t[j], makes every column's increment 2·(b_t − ẽ_t)·(±1 fixed per column). Model P runs per
  entry. Models W and C need every lane's position to repeat, which a fixed per-column sign gives too.

Each adversary decides copy-or-write from the coins and its own data only (the walk position), and its words must
equal the honest k32 chain's (`np.array_equal`).

**R1's check.** In R1 the certifier holds every word in the clear, so it can count what DistinctWrites assumes
instead of assuming it:
- (a) credit only words that differ from the entry's previous word. That refuses A2c, but not A2r.
- (b) credit only an entry's distinct values. That refuses both, and costs the honest only its chance revisits.
(b) is sound against copies within an entry; copies across entries are the premise `ssn-tag-distinct` keeps (G
private), and copies from advice are A5's.

**Grinding.** A redrawn coin draw lets the adversary keep the best of g. The count it maximizes is the call's zero
b-bits (A2c) or its unchanged entries (A8, round 3, model P).
- Re-commit tries: the lead's per-matmul seeds are Fiat–Shamir in root(A_j). A try re-salts A_j's hiding leaf:
  c = SHA-512(SP ‖ y) is 2 compressions, the leaf 2 and the seed 1, so 5 compressions at SHA-512's floor (an
  estimate, with M·y's XORs uncounted, so the net gain is an upper bound).
- Free tries: a certifier that issues a second draw for one slot (R1b's "a second A_j"), or any retry that redraws.
  The protocol forbids both; these rows price the rule.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import noise_sweep as ns  # noqa: E402
import nred_round3 as r3  # noqa: E402

H32 = ns.H32
SHA_FLOOR_UNITS_BYTE = 908.0
C_TRY = 5 * 128 * SHA_FLOOR_UNITS_BYTE
MS = (1, 8, 64, 512)
GS = (1, 2, 16, 256, 4096)


def a2r_walk(s: int, m: int, rng: np.random.Generator, pattern: str):
    """Per-row increments a later step under the chosen cancel: `a2c` gives b_t; `a2r` alternates b_t and b_t − 1."""
    b = rng.integers(0, 2, (m, s - 1))
    off = np.zeros(s - 1, dtype=np.int64)
    if pattern == "a2r":
        off[1::2] = -1
    return b, off


def tag_instance(m: int, n: int, k: int, rng: np.random.Generator, pattern: str):
    """The honest k32 chain on chosen data: step 1's word W1 (data + tags, arbitrary), then at each later step a
    noise lane (A = 1 + b_t, B = 1) and a cancel lane (A = 1, B = −1 + off_t); 30 more data lanes, all zero."""
    s = ns.steps1l(k)
    b, off = a2r_walk(s, m, rng, pattern)
    w1 = rng.integers(-(1 << 20), 1 << 20, (m, n))
    a_noise = 1 + b                                   # m × (S−1)
    b_cancel = np.broadcast_to((-1 + off)[:, None], (s - 1, n))
    inc = a_noise[:, :, None] * 1 + 1 * b_cancel[None, :, :]
    honest = np.concatenate([w1[:, None, :], w1[:, None, :] + np.cumsum(inc, axis=1)], axis=1)
    return {"s": s, "b": b, "off": off, "w1": w1, "honest": honest}


def hw_instance(m: int, n: int, k: int, rng: np.random.Generator, pattern: str):
    """`ssn-1l-hw`'s later noise (1 + 2b_t)·σ_t[j], σ public ±1, against a per-column cancel B_t[j] =
    −σ_t[j](1 + 2e_t[j]) chosen from σ so each column's increment is 2·sgn_j·(b_t − ẽ_t), sgn_j fixed per column."""
    s = ns.steps1l(k)
    b, off = a2r_walk(s, m, rng, pattern)
    sigma = rng.choice(np.array([-1, 1]), (s - 1, n))
    sgn = rng.choice(np.array([-1, 1]), n)
    et = -off                                                     # ẽ_t ∈ {0, 1}
    # want σ(b − e) = sgn(b − ẽ) for b ∈ {0, 1}: σ = sgn → e = ẽ; σ = −sgn → e = 1 − ẽ ... only when it is consistent
    e = np.where(sigma == sgn[None, :], et[:, None], 1 - et[:, None])
    flip = sigma != sgn[None, :]
    w1 = rng.integers(-(1 << 20), 1 << 20, (m, n))
    inc = (1 + 2 * b)[:, :, None] * sigma[None, :, :] - (sigma * (1 + 2 * e))[None, :, :]
    honest = np.concatenate([w1[:, None, :], w1[:, None, :] + np.cumsum(inc, axis=1)], axis=1)
    return {"s": s, "b": b, "off": off, "w1": w1, "honest": honest, "sigma": sigma, "sgn": sgn, "flip": flip}


def build_walk(inst: dict, design: str):
    """The adversary: its walk position from the coins and its data only; a new position is written (W1 + its
    value), a visited one is copied. Returns its words, fresh writes a row (per entry for hw), and the words."""
    b, off, w1, s = inst["b"], inst["off"], inst["w1"], inst["s"]
    m, n = w1.shape
    words = np.empty((m, s, n), dtype=np.int64)
    words[:, 0, :] = w1
    writes = np.ones(m, dtype=np.int64)                         # step 1, a row
    for i in range(m):
        if design == "tag":
            pos = np.concatenate([[0], np.cumsum(b[i] + off)])   # increment b_t + off_t
            held = {0: w1[i].copy()}
            for t in range(1, s):
                p = int(pos[t])
                if p not in held:
                    held[p] = w1[i] + p
                    writes[i] += 1
                words[i, t] = held[p]
        else:
            # hw: each column's increment is 2·sgn_j·(b_t − ẽ_t) when σ = sgn; where σ = −sgn the chosen e = 1 − ẽ
            # gives σ(b − 1 + ẽ) = sgn(1 − ẽ − b), the walk's mirror. Track the entry's value directly.
            sigma, sgn, flip = inst["sigma"], inst["sgn"], inst["flip"]
            et = -off
            step = np.where(flip, sgn[None, :] * (1 - et[:, None] - b[i][:, None]),
                            sgn[None, :] * (b[i][:, None] - et[:, None])) * 2
            pos = np.concatenate([np.zeros((1, n), dtype=np.int64), np.cumsum(step, axis=0)])
            words[i] = w1[i][None, :] + pos
            writes[i] = sum(len(np.unique(pos[:, j])) for j in range(n))   # per entry
    return words, writes


def model_p_share(writes: np.ndarray, m: int, n: int, k: int, s: int, per_entry: bool) -> float:
    """Saving under model P as a share of stock: the later words not written, less one pointer a row-step."""
    later = m * n * (s - 1)
    if per_entry:
        saved = later - (writes.sum() - m * n)
    else:
        saved = later - n * (writes.sum() - m)
    return H32 * (saved - m * (s - 1)) / (m * n * k)


def checks_r1(words: np.ndarray, k: int):
    """Credit under (a) consecutive-distinct and (b) per-entry distinct, as words a row-step-entry; returns the
    counts summed over entries."""
    m, s, n = words.shape
    consec = int((np.diff(words, axis=1) != 0).sum()) + m * n
    distinct = sum(len(np.unique(words[i, :, j])) for i in range(m) for j in range(n))
    return consec, distinct


def honest_revisits(m: int, n: int, k: int, rng: np.random.Generator) -> dict:
    """The honest's words on int8 data (A ~ N(0, 30), B ~ N(0, 25), clipped), `ssn-1l-tag`'s noise: the share of
    later words a consecutive-distinct check (a) and a per-entry distinct check (b) would not credit."""
    s = ns.steps1l(k)
    b = rng.integers(0, 2, (m, s - 1))
    a = np.clip(np.rint(rng.normal(0, 30, (m, s - 1, 31))), -127, 127).astype(np.int64)
    bb = np.clip(np.rint(rng.normal(0, 25, (s - 1, 31, n))), -127, 127).astype(np.int64)
    inc = np.einsum("mtl,tln->mtn", a, bb) + (1 + b)[:, :, None]
    w1 = rng.integers(-(1 << 20), 1 << 20, (m, n))
    words = np.concatenate([w1[:, None, :], w1[:, None, :] + np.cumsum(inc, axis=1)], axis=1)
    consec, distinct = checks_r1(words, k)
    total = m * n * s
    return {"uncredited_a": 1 - consec / total, "uncredited_b": 1 - distinct / total}


def a2r_runs(seed: int = 67) -> dict:
    """A2c and A2r on `ssn-1l-tag` and `ssn-1l-hw` at m 16, n 256, K 2,048 and 8,192: constructive, equality-checked,
    model P (and W = C for tag), and what R1's checks (a) and (b) would credit."""
    rng = np.random.default_rng(seed)
    out = {}
    for design, make in (("tag", tag_instance), ("hw", hw_instance)):
        for pattern in ("a2c", "a2r"):
            for k in (2048, 8192):
                m, n = 16, 256
                inst = make(m, n, k, rng, pattern)
                words, writes = build_walk(inst, design)
                equal = bool(np.array_equal(words, inst["honest"]))
                s = inst["s"]
                share = model_p_share(writes, m, n, k, s, per_entry=(design == "hw"))
                consec, distinct = checks_r1(words, k)
                paid = int(writes.sum()) * (1 if design == "hw" else n)
                out[f"{design}/{pattern}/k{k}"] = {
                    "equal_to_honest": equal, "model_p_share": share,
                    "fresh_writes_per_entry": paid / (m * n), "words_per_entry": s,
                    "over_credit_check_a_share": H32 * (consec - paid) / (m * n * k),
                    "over_credit_check_b_share": H32 * (distinct - paid) / (m * n * k)}
    return out


_trapz = getattr(np, "trapezoid", None) or np.trapz
_X = np.linspace(-12, 12, 96001)
_PHI = np.exp(-_X * _X / 2) / math.sqrt(2 * math.pi)
_CDF = 0.5 * (1 + np.vectorize(math.erf)(_X / math.sqrt(2)))


def e_max_normal(g: int) -> float:
    """E[max of g iid N(0, 1)]."""
    if g == 1:
        return 0.0
    return float(_trapz(_X * g * _PHI * _CDF ** (g - 1), _X))


def e_max_binom(nn: int) -> dict:
    """E[max of g iid Bin(nn, 1/2)] for every g in GS, exactly: Σ_z P(max > z)."""
    z = np.arange(nn + 1)
    lg = np.vectorize(math.lgamma)
    pmf = np.exp(math.lgamma(nn + 1) - lg(z + 1) - lg(nn - z + 1) - nn * math.log(2))
    sf = np.clip(np.concatenate([np.cumsum(pmf[::-1])[::-1][1:], [0.0]]), 0, 1 - 1e-15)   # P(Z > z)
    l1 = np.log1p(-sf)
    return {g: float(np.sum(-np.expm1(g * l1))) for g in GS}


def a8_row_stats(design: str, n: int, k: int, rng: np.random.Generator):
    """Round 3's A8 (model P): with w drawn once a window, a row's unchanged entries at step t are c_t[X_t]; returns
    the per-row mean and variance of Σ_t c_t[X_t], and the table for a Monte Carlo check."""
    s = ns.steps1l(k)
    xs, _ = r3.DESIGNS[design]
    v, _ = r3.point_mass(design)
    _, w = r3.draw(design, 1, n, s, rng)
    table = np.array([[int((w[t] == v // x).sum()) if v % x == 0 else 0 for x in xs] for t in range(s - 1)])
    return float(table.mean(1).sum()), float(table.var(1).sum()), table, s


def grinding(seed: int = 71) -> dict:
    rng = np.random.default_rng(seed)
    eg = {g: e_max_normal(g) for g in GS}
    out = {"e_max_normal": eg}
    # A2c on ssn-1l-tag / hw: zero row-steps Bin(m(S − 1), 1/2), each worth n words
    for m in MS:
        rows = {}
        stock = sum(m * n * k for n, k in ns.LLAMA.values())
        base = {g: 0.0 for g in GS}
        best_net = 0.0
        for n, k in ns.LLAMA.values():
            s = ns.steps1l(k)
            ez = e_max_binom(m * (s - 1))
            for g in GS:
                base[g] += H32 * (n * ez[g] - m * (s - 1))
            best_net += max(H32 * n * (ez[g] - ez[1]) - (g - 1) * C_TRY for g in GS)
        rows["free_tries_share"] = {str(g): base[g] / stock for g in GS}
        rows["recommit_best_net_share"] = (base[1] + best_net) / stock
        out[f"a2c/m{m}"] = rows
    # A8 model P on round 3's designs
    stats = {d: {lin: a8_row_stats(d, n, k, rng) for lin, (n, k) in ns.LLAMA.items()}
             for d in ("fix2", "fix2w2", "uw4", "uw6")}
    for d, per in stats.items():
        for m in MS:
            stock = sum(m * n * k for n, k in ns.LLAMA.values())
            share = {g: 0.0 for g in GS}
            best_net = 0.0
            for lin, (n, k) in ns.LLAMA.items():
                mu, var, _, s = per[lin]
                vals = {g: m * mu + math.sqrt(m * var) * eg[g] for g in GS}
                for g in GS:
                    share[g] += H32 * (vals[g] - m * (s - 1))
                best_net += max(H32 * (vals[g] - vals[1]) - (g - 1) * C_TRY for g in GS)
            out[f"a8/{d}/m{m}"] = {"free_tries_share": {str(g): share[g] / stock for g in GS},
                                   "recommit_best_net_share": (share[1] + best_net) / stock}
    # Monte Carlo check of the normal order statistic at m = 1, g = 16 (uw4, gate-up)
    mu, var, table, s = stats["uw4"]["gate_up"]
    reps, g = 400, 16
    xi = rng.integers(0, table.shape[1], (reps, g, s - 1))
    sums = table[np.arange(s - 1)[None, None, :], xi].sum(-1)
    out["mc_check_uw4_gate_up_m1_g16"] = {"mc": float(sums.max(1).mean()), "normal": mu + math.sqrt(var) * eg[16]}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    a2 = a2r_runs()
    hon = {f"k{k}": honest_revisits(16, 256, k, np.random.default_rng(73 + k)) for k in (2048, 8192)}
    gr = grinding()
    mc = gr["mc_check_uw4_gate_up_m1_g16"]
    ck = [
        ("A2c and A2r words equal the honest chain's (tag and hw, K 2,048 and 8,192)",
         all(v["equal_to_honest"] for v in a2.values()), str({k: v["equal_to_honest"] for k, v in a2.items()})),
        ("A2c on tag reproduces interim 2's 51.6% at K 2,048 (model P, m 16, n 256)",
         abs(a2["tag/a2c/k2048"]["model_p_share"] - 0.516) < 0.03, f"{a2['tag/a2c/k2048']['model_p_share']:.4f}"),
        ("A2r saves more than A2c on tag and hw",
         all(a2[f"{d}/a2r/k{k}"]["model_p_share"] > a2[f"{d}/a2c/k{k}"]["model_p_share"]
             for d in ("tag", "hw") for k in (2048, 8192)),
         ", ".join(f"{d}/k{k} {a2[f'{d}/a2c/k{k}']['model_p_share']:.3f}->{a2[f'{d}/a2r/k{k}']['model_p_share']:.3f}"
                   for d in ("tag", "hw") for k in (2048, 8192))),
        ("check (a) refuses A2c's saving (over-credit ≈ 0) but not A2r's",
         abs(a2["tag/a2c/k2048"]["over_credit_check_a_share"]) < 1e-3
         and a2["tag/a2r/k2048"]["over_credit_check_a_share"] > 0.1,
         f"a2c {a2['tag/a2c/k2048']['over_credit_check_a_share']:.4f}, a2r "
         f"{a2['tag/a2r/k2048']['over_credit_check_a_share']:.4f}"),
        ("check (b) over-credits neither (≤ 0 everywhere)",
         all(v["over_credit_check_b_share"] <= 1e-12 for v in a2.values()),
         ", ".join(f"{k} {v['over_credit_check_b_share']:.2e}" for k, v in a2.items())),
        ("the honest loses < 1% of its credit under check (b)",
         all(v["uncredited_b"] < 0.01 for v in hon.values()), json.dumps(hon)),
        ("A2c's g = 1 grinding share equals its no-grinding share (51.6% band) at m 64",
         abs(gr["a2c/m64"]["free_tries_share"]["1"] - 0.516) < 0.005, f"{gr['a2c/m64']['free_tries_share']['1']:.4f}"),
        ("normal order statistic within 3% of Monte Carlo (uw4 gate-up, m 1, g 16)",
         abs(mc["mc"] / mc["normal"] - 1) < 0.03, f"mc {mc['mc']:.1f} / normal {mc['normal']:.1f}"),
    ]
    ck = [(c[0], bool(c[1]), c[2]) for c in ck]
    ok = all(c[1] for c in ck)
    stock = sum(n * k for n, k in ns.LLAMA.values())
    llama = {f"{d}/{p}": sum(n * k * a2[f"{d}/{p}/k{k}"]["model_p_share"] for n, k in ns.LLAMA.values()) / stock
             for d in ("tag", "hw") for p in ("a2c", "a2r")}
    import nred_egress as eg
    bounds = {"a2r_tag_llama": llama["tag/a2r"], "a2r_hw_llama": llama["hw/a2r"],
              "a2r_check_a_overcredit_tag_k2048": a2["tag/a2r/k2048"]["over_credit_check_a_share"],
              "a2c_m1_free_g4096": gr["a2c/m1"]["free_tries_share"]["4096"],
              "a2c_m1_recommit_net": gr["a2c/m1"]["recommit_best_net_share"],
              "uw6_m1_free_g4096": gr["a8/uw6/m1"]["free_tries_share"]["4096"]}
    rep = {"a2r": a2, "llama_weighted": llama, "honest_revisits": hon, "grinding": gr, "c_try_units": C_TRY,
           "per_period": {k: eg.periods(v) for k, v in bounds.items()}, "bounds": bounds,
           "checks": [{"check": c[0], "pass": c[1], "got": c[2]} for c in ck], "all_pass": ok}
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rep, f, indent=1)
    for k, v in a2.items():
        print(f"{k}: equal {v['equal_to_honest']}, model P {100*v['model_p_share']:.1f}%, writes/entry "
              f"{v['fresh_writes_per_entry']:.1f} of {v['words_per_entry']}, over-credit (a) "
              f"{100*v['over_credit_check_a_share']:.1f}% (b) {100*v['over_credit_check_b_share']:.2f}%")
    print("honest uncredited:", json.dumps(hon))
    print("Llama-weighted:", ", ".join(f"{k} {100*v:.1f}%" for k, v in llama.items()))
    for k, v in bounds.items():
        p = rep["per_period"][k]
        print(f"periods {k} ({100*v:.4g}%): H200 " + " / ".join(f"{x:.2e}" for x in p["one_h200"].values())
              + "; DC " + " / ".join(f"{x:.2e}" for x in p["datacenter"].values())
              + "; world " + " / ".join(f"{x:.2e}" for x in p["world"].values()))
    for key in sorted(k for k in gr if k.startswith(("a2c/", "a8/"))):
        v = gr[key]
        print(f"{key}: free " + " / ".join(f"g{g} {100*x:.2f}%" for g, x in v["free_tries_share"].items())
              + f"; re-commit best net {100*v['recommit_best_net_share']:.2f}%")
    for c in ck:
        print(f"{'PASS' if c[1] else 'FAIL'}  {c[0]}  [{c[2][:300]}]")
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
