#!/usr/bin/env python3
"""TT_OUT's gate at k = 256 on the H100 record: the two families the ticket-chain census does not cover, on the served
scheme ``pearl-c-h100-v1-h3`` itself (its forming, chain, peel and debit, bit for bit).

At k = 256 a word is 8 atoms in two promotion groups of 4, so both searches are exhaustive:

* **Cross-group cancellation.** For every word, every one of the 2^8 sets of skipped atoms (each group from +0 under its
  mask, promoted into the total) that leaves the ticket word unchanged.  The served debit (`pearl_c_debit.word_debit`, P1's
  `jointFlags`) flags only sets inside one group; an atom of a kept set it doesn't flag is the residual.
* **Peel shortcuts.**  U is C̃ through four BF16 k16 steps (`pearl_c.peel`); for every word, every one of the 2^4 sets of
  skipped steps that leaves U unchanged (U = C̃ is the whole peel skipped).  The debit charges none of them.  Also counted:
  products with a zero factor, and k16 slices of a fragment whose BF16 factors all fit E4M3 (an FP8 route's precondition).

TT_OUT's cost model charges a tensor-core instruction its whole fragment, never per output word, so a skip saves work only
when it holds for every word of a fragment: the smallest, mma.sync m16n8 (16 rows x 8 columns).  Each cell reports the
saving per word (an existence count, P1's convention) and per fragment (what the cost model prices), as shares of the
cells' credit (`PearlC.credit_of` at n = 6,144, GLM-5.3's down projections); an undebited atom costs `atom_units` (40), a
peel step 16 BF16 MACs (32).  A cell is the tile's credited words (rows passing the noise floor and liveness).  Each cell
also reports whether the verifier accepts it: its replayed debit within ρ of its credit (`tile_cap_ok`), and every
credited row's within its row cap of its own (-h3's, `clear_work.check_opened`): 4ρ at k = 256 (`PearlC.row_cap_at`),
under seven flagged atom-words over a 64-column tile, where ρ would allow under two.

The gate passes when every in-domain cell's fragment saving (cross-group residual plus peel), accepted or not, is at most
1/400 of its credit and the chain's exact runs (14 or more atoms) are none.  ``--b-samples FILE`` (an .npz of 64 x 256
weight slices) adds a family ``glm-w2-NAME`` per slice: those rows as B, with t(4) activations as A.

    uv run python benchmarks/pouw/pearlc_k256_gate.py --out gate.json [--runs 1,2,3] [--families ...] [--jobs 32]
    uv run python benchmarks/pouw/pearlc_k256_gate.py --verdict DIR

``--verdict DIR`` is the run's status (`pearlc_k256_gate.sh`): it passes only when every ``census_*.json`` under DIR (the
census with the replayed debit) passes its own criterion, every ``gate*.json`` passes, and ``slices.json``
(`pearl_c_sm90/glm53_slices.py`) covers every down projection.  A weight row failing the per-row rules is excluded (bound,
no credit), so the slices' failing share is reported, not gated.  ``verdict.json`` also reports the honest tiles' acceptance
(Gaussian and the w2 samples): how many of them -h3's row cap rejects before any row is given up.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from fractions import Fraction
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pearlc_census as K  # noqa: E402
import run_result  # noqa: E402
from verity.core.protocols.pouw.protocol import Context, Matmul, Shape, Tile  # noqa: E402
from verity.core.protocols.pouw.schemes import get  # noqa: E402
from verity.core.protocols.pouw.schemes import pearl_c as C  # noqa: E402
from verity.core.protocols.pouw.schemes import pearl_c_debit as D  # noqa: E402
from verity.core.protocols.pouw.schemes import pearl_kw as P  # noqa: E402

SCHEME = "pearl-c-h100-v1-h3"
K_ = 256
N_CREDIT = 6144
FRAG = (16, 8)
THRESHOLD = Fraction(1, 400)
ALL_FAMILIES = K.FAMILIES + K.ACTIVATION_FAMILIES + K.OUTLIER_FAMILIES + K.FREEZE_FAMILIES + K.XCANCEL_FAMILIES
B_SAMPLES = ""
#: GLM-5.3's down projections: 76 MoE layers (3-77, and the MTP layer 78), each 256 routed experts' w2 and a shared one
SLICE_TENSORS = 76 * 257
POP = np.array([bin(m).count("1") for m in range(256)], dtype=np.int64)


def _h(*parts: object) -> bytes:
    return hashlib.sha512("|".join(["verity/pouw/pearl-c/k256-gate/v1", *map(str, parts)]).encode()).digest()


def words_of(X: np.ndarray) -> list[bytes]:
    return [C.f32_row([int(w) for w in row]) for row in np.ascontiguousarray(X, dtype=np.float32).view(np.uint32)]


def row_ok(side) -> np.ndarray:
    return np.array([P.f32_of_bits(r) * P.f32_of_bits(a) >= P.SIGMA_MIN and C.live_row(x)
                     for r, a, x in zip(side.rho, side.alpha, side.x)])


def frag_all(x: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """x: (masks, rows*cols) bool, words row-major; True where it holds for every word of each m16n8 fragment."""
    fr, fc = FRAG
    return x.reshape(len(x), rows // fr, fr, cols // fc, fc).all(axis=(2, 4)).reshape(len(x), -1)


def frag_sum(x: np.ndarray, rows: int, cols: int) -> np.ndarray:
    fr, fc = FRAG
    return x.reshape(len(x), rows // fr, fr, cols // fc, fc).sum(axis=(2, 4)).reshape(len(x), -1)


def chain_masks(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Every word's ticket under each of the 2^8 skip sets (bit t: atom t passed through): (256, words) FP32 values."""
    G, T = 4, A.shape[1] // 32
    assert T == 2 * G, "the exhaustive search is k = 256's: two groups of four atoms"
    W = [np.stack([K.group_run(A, B, list(range(g * G, g * G + G)), frozenset(g * G + i for i in range(G) if m >> i & 1))[0]
                   for m in range(16)]).astype(np.float32) for g in range(2)]
    final = (W[0][None, :, :] + W[1][:, None, :]).astype(np.float32)   # [m1, m0]: the second promotion into W0's total
    return final.reshape(256, -1)


def peel_masks(c: list[list[int]], pa: list[list[int]], pb: list[list[int]], step) -> np.ndarray:
    """Every word's U under each of the 2^4 sets of skipped k16 steps: (16, words) FP32 words."""
    out = np.zeros((16, len(pa) * len(pb)), dtype=np.uint32)
    for i, ra in enumerate(pa):
        for j, rb in enumerate(pb):
            states = {0: c[i][j]}
            for s in range(4):
                nxt = {}
                for m, acc in states.items():
                    nxt[m] = step(acc, ra[16 * s:16 * s + 16], rb[16 * s:16 * s + 16])
                    nxt[m | 1 << s] = acc
                states = nxt
            out[:, i * len(pb) + j] = [states[m] for m in range(16)]
    return out


def e4m3_fits(word: int) -> bool:
    """A BF16 word whose value is an E4M3 significand up to a power of two (its 4 low mantissa bits zero)."""
    return (word & 0x7FFF) == 0 or (word & 0xF) == 0


def _init(path: str) -> None:
    global B_SAMPLES
    B_SAMPLES = path


def cell(fam: str, run: int, rows: int = 64, cols: int = 64) -> dict:
    t0 = time.time()
    s = get(SCHEME)
    k = K_
    rng = K._rng(run, fam, k, 1.0)
    if fam.startswith("glm-w2-"):
        XA, XB = rng.standard_t(4, (rows, k)), np.load(B_SAMPLES)[fam.removeprefix("glm-w2-")].astype(np.float64)
    else:
        XA, XB = K.family(fam, rows, cols, k, K.lines(k, K._rng(run, "FA", k)), rng)
    ctx = Context(_h(run, fam, "salt")[:32], Matmul(0, Shape(rows, k, N_CREDIT), 1), _h(run, fam, "root-a"), _h(run, fam, "root-b"))
    noise = s.noise(ctx)
    ops = s.encode(ctx, noise, words_of(XA), words_of(XB), Tile(tuple(range(rows)), tuple(range(cols))))
    row = {"family": fam, "run": run, "k": k, "tile": f"{rows}x{cols}", "fragment": f"{FRAG[0]}x{FRAG[1]}"}
    if ops is None:
        return {**row, "in_domain": False, "rejected_by": ["non-finite forming"]}
    ok_a, ok_b = row_ok(ops.a), row_ok(ops.b)
    cred = (ok_a[:, None] & ok_b[None, :]).ravel()
    row.update({"rows_ok_A": int(ok_a.sum()), "rows_ok_B": int(ok_b.sum()), "credited_words": int(cred.sum()),
                "in_domain": bool(cred.any())})
    checked = s.checked(ctx, ops)
    A8, B8 = np.array(ops.a.codes, dtype=np.uint8), np.array(ops.b.codes, dtype=np.uint8)
    ii, jj = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    A, B = A8[ii.ravel()], B8[jj.ravel()]
    nw = rows * cols
    credit = Fraction(s.credit_of(Shape(1, k, N_CREDIT)), N_CREDIT)

    # ---- the ticket chain: every skip set, the served debit, the cap
    final = chain_masks(A, B)
    c_words = np.array(checked.c, dtype=np.uint32).ravel()
    assert (final[0] == c_words.view(np.float32)).all(), "the census's atom disagrees with the served chain"
    keeps = final == final[0]
    flags_k, _ = K.debit(A, B, 4)
    served = [D.word_debit(ops.a.codes[i], ops.b.codes[j], s.device) for i in range(rows) for j in range(cols)]
    flags = np.array([f for f, _ in served], dtype=bool)
    zeros = np.array([z for _, z in served], dtype=np.int64)
    assert (flags == flags_k).all(), "pearlc_census.debit disagrees with the served word_debit"
    au = D.atom_units(s.device, s.prices)
    word_debit = [D.debit_units(int(f.sum()), 0, s.prices, s.device, zeros=int(z)) for f, z in zip(flags, zeros)]
    debit = sum((d for d, c in zip(word_debit, cred) if c), Fraction(0))
    cap = s.cap * credit * int(cred.sum())
    row_over = []
    for i in range(rows):
        sel = cred.reshape(rows, cols)[i]
        if sel.any():
            d_i = sum((word_debit[i * cols + j] for j in np.flatnonzero(sel)), Fraction(0))
            row_over.append(d_i > s.row_cap_at(k) * credit * int(sel.sum()))
    tile_ok = bool(cred.any()) and debit <= cap
    accepted = tile_ok and not any(row_over)
    fb = (flags.astype(np.int64) << np.arange(8)).sum(1)
    masks = np.arange(256)
    und = POP[(masks[:, None] & ~fb[None, :]) & 0xFF]                     # (256, words): the set's unflagged atoms
    und = np.where(cred[None, :], und, 0)
    word_best = np.where(keeps, und, 0).max(0)
    kept_f = frag_all(keeps, rows, cols)
    frag_best = np.where(kept_f, frag_sum(und, rows, cols), 0).max(0)
    total_set = np.where(keeps, POP[masks][:, None], 0).max(0)
    nc = int(cred.sum()) or 1
    tid, tex, _, _ = K.chain(A, B, 4)

    # ---- the peel: every set of skipped k16 steps
    U = peel_masks(checked.c, ops.pa, ops.pb, s.device.peel.step)
    assert (U[0] == np.array(checked.u, dtype=np.uint32).ravel()).all(), "the peel replay disagrees with the served U"
    keeps_u = U == U[0]
    pmasks = np.arange(16)
    pk = np.where(keeps_u & cred[None, :], POP[pmasks][:, None], 0)
    peel_word = pk.max(0)
    kept_uf = frag_all(keeps_u, rows, cols)
    peel_frag = np.where(kept_uf, POP[pmasks][:, None] * frag_sum(np.broadcast_to(cred, keeps_u.shape), rows, cols), 0).max(0)
    step_units = s.prices.bf16_mac * 16
    zero_a = np.array([[(w & 0x7FFF) == 0 for w in r] for r in ops.pa])
    zero_b = np.array([[(w & 0x7FFF) == 0 for w in r] for r in ops.pb])
    zero_products = float((zero_a[ii.ravel()] | zero_b[jj.ravel()]).mean())
    fits_a = np.array([[e4m3_fits(w) for w in r] for r in ops.pa])
    fits_b = np.array([[e4m3_fits(w) for w in r] for r in ops.pb])
    fr, fc = FRAG
    fp8_slices = 0
    for bi in range(rows // fr):
        for bj in range(cols // fc):
            for sl in range(4):
                cut = slice(16 * sl, 16 * sl + 16)
                if fits_a[bi * fr:(bi + 1) * fr, cut].all() and fits_b[bj * fc:(bj + 1) * fc, cut].all():
                    fp8_slices += 1

    chain_word = au * int(word_best.sum()) / (credit * nc)
    chain_frag = au * int(frag_best.sum()) / (credit * nc)
    peel_w = Fraction(step_units * int(peel_word.sum())) / (credit * nc)
    peel_f = Fraction(step_units * int(peel_frag.sum())) / (credit * nc)
    exact_runs = K.exact_run_atoms(tex)
    row.update({
        "accepted": accepted, "tile_cap_ok": tile_ok, "debit_share_of_credit": float(debit / (credit * nc)), "rows_over_row_cap": int(sum(row_over)),
        "debit_atom_share": round(float(flags[cred].mean()), 6) if cred.any() else 0.0,
        "later_zero_promotions": int(zeros[cred].sum()),
        "xg_words_with_undebited_set": int((word_best > 0).sum()),
        "xg_undebited_atoms_word_max": int(word_best.max()), "xg_kept_set_atoms_word_max": int(total_set.max()),
        "xg_word_share_of_credit": float(chain_word), "xg_fragment_share_of_credit": float(chain_frag),
        "xg_fragments_with_undebited_set": int((frag_best > 0).sum()), "xg_fragment_kept_atoms_max":
            int(np.where(kept_f, POP[masks][:, None], 0).max()),
        "tile_identity_atoms": int(tid.sum()), "exact_run_atoms": exact_runs,
        "peel_words_with_skippable_step": int((peel_word > 0).sum()), "peel_words_u_equals_c": int((keeps_u[15] & cred).sum()),
        "peel_word_share_of_credit": float(peel_w), "peel_fragment_share_of_credit": float(peel_f),
        "peel_fragments_with_skippable_step": int((peel_frag > 0).sum()),
        "peel_zero_product_share": round(zero_products, 6), "peel_fp8_route_slices": fp8_slices,
        "peel_bf16_entries_fitting_e4m3": round(float(np.concatenate([fits_a[:, 32:].ravel(), fits_b[:, :32].ravel()]).mean()), 6),
        "fragment_share_of_credit": float(chain_frag + peel_f), "word_share_of_credit": float(chain_word + peel_w),
        "seconds": round(time.time() - t0, 1)})
    row["falsifier"] = bool(row["in_domain"] and (chain_frag + peel_f > THRESHOLD or exact_runs))
    return row


def verdict(d: Path) -> None:
    """The run's status over DIR's census, gate and slice reports (module docstring)."""
    parts, ok = [], True
    for f in sorted(d.glob("census_*.json")):
        c = json.loads(f.read_text())
        good = c["criterion"] == "residual_share" and c["selftest_mismatches"] == 0 and c["worst_in_domain_skippable_share"] <= 1 / 400
        cross = [r for r in c["cells"] if r.get("in_domain") and "cross_group_beyond_debit_max" in r]
        w = max(cross, key=lambda r: r["cross_group_beyond_debit_mean"], default=None)
        parts.append({"file": f.name, "pass": good, "residual_worst": c["worst_in_domain_skippable_share"],
                      "cross_group_beyond_debit_mean_worst": w and (w["family"], w["layout"], w["cross_group_beyond_debit_mean"],
                                                                    w.get("debit_share"))})
        ok &= good
    gates = sorted(d.glob("gate*.json"))
    for f in gates:
        g = json.loads(f.read_text())
        good = g["selftest_mismatches"] == 0 and not g["falsifiers"] and g["in_domain_cells"] > 0
        parts.append({"file": f.name, "pass": good, "in_domain_cells": g["in_domain_cells"],
                      "accepted_in_domain_cells": g["accepted_in_domain_cells"], "worst_fragment_share": g["worst_fragment_share"],
                      "worst_word_share_tile_cap_ok": g["worst_word_share"],
                      "worst_word_share_accepted": g.get("worst_word_share_accepted"), "falsifiers": g["falsifiers"]})
        ok &= good
    honest = {}
    for f in gates:
        for r in json.loads(f.read_text())["cells"]:
            if r["in_domain"] and (r["family"] == "gaussian" or r["family"].startswith("glm-w2-")):
                h = honest.setdefault("glm-w2" if r["family"].startswith("glm-w2-") else r["family"],
                                      {"tiles": 0, "accepted": 0, "rows": 0, "rows_over_row_cap": 0})
                h["tiles"] += 1
                h["accepted"] += r["accepted"]
                h["rows"] += r["rows_ok_A"]
                h["rows_over_row_cap"] += r["rows_over_row_cap"]
    sl = d / "slices.json"
    if sl.exists():
        x = json.loads(sl.read_text())
        good = x["rows"] > 0 and x["tensors"] == SLICE_TENSORS
        parts.append({"file": sl.name, "pass": good, "tensors": x["tensors"], "rows": x["rows"], "either_fail": x["either_fail"],
                      "fail_share": x["fail_share"],
                      "roles": {r: {y: v[y] for y in ("rows", "floor_fail", "live_fail", "either_fail", "fail_share", "worst_slice")}
                                for r, v in x["roles"].items()}})
        ok &= good
    ok &= bool(gates) and sl.exists() and any(p["file"].startswith("census_") for p in parts)
    out = d / "verdict.json"
    out.write_text(json.dumps({"pass": ok, "parts": parts, "honest_acceptance": honest}, indent=1) + "\n")
    failed = [p["file"] for p in parts if not p["pass"]]
    run_result.write(out, ok, "pearlc-k256-gate-verdict", "every part passes" if ok else f"failing: {', '.join(failed) or 'a part is missing'}",
                     [("parts", len(parts), "count"), ("failing_parts", len(failed), "count")])
    print(json.dumps({"pass": ok, "failing": failed}))
    sys.exit(0 if ok else 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default=",".join(ALL_FAMILIES))
    ap.add_argument("--runs", default="1,2,3")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--b-samples", default="")
    ap.add_argument("--verdict", default="")
    ap.add_argument("--out", default="pearlc_k256_gate.json")
    a = ap.parse_args()
    if a.verdict:
        return verdict(Path(a.verdict))
    global B_SAMPLES
    B_SAMPLES = a.b_samples
    families = a.families.split(",")
    if a.b_samples:
        families += [f"glm-w2-{x}" for x in np.load(a.b_samples).files]
    bad = K.selftest()
    work = [(f, int(r)) for r in a.runs.split(",") for f in families if f]
    t0 = time.time()
    if a.jobs > 1:
        with Pool(a.jobs, initializer=_init, initargs=(a.b_samples,)) as pool:
            rows = pool.starmap(cell, work, chunksize=1)
    else:
        rows = [cell(*w) for w in work]
    for r in rows:
        print(json.dumps(r), flush=True)
    scored = [r for r in rows if r["in_domain"]]
    worst = max(scored, key=lambda r: r["fragment_share_of_credit"], default=None)
    worst_word = max((r for r in scored if r["tile_cap_ok"]), key=lambda r: r["word_share_of_credit"], default=None)
    worst_acc = max((r for r in scored if r["accepted"]), key=lambda r: r["word_share_of_credit"], default=None)
    by_family = {}
    for r in scored:
        f = by_family.setdefault(r["family"], {"cells": 0, "tile_cap_ok": 0, "accepted": 0, "rows_over_row_cap": 0})
        f["cells"] += 1
        f["tile_cap_ok"] += r["tile_cap_ok"]
        f["accepted"] += r["accepted"]
        f["rows_over_row_cap"] += r["rows_over_row_cap"]
    falsified = [r for r in rows if r.get("falsifier")]
    doc = {"tool": "pouw_pearlc_k256_gate", "scheme": SCHEME, "k": K_, "n_credit": N_CREDIT, "fragment": list(FRAG),
           "tile_cap": str(get(SCHEME).cap), "row_cap": str(get(SCHEME).row_cap_at(K_)),
           "threshold": float(THRESHOLD), "selftest_mismatches": bad, "cells": rows, "in_domain_cells": len(scored),
           "accepted_in_domain_cells": sum(r["accepted"] for r in scored), "acceptance_by_family": by_family,
           "worst_fragment_share": worst and {x: worst[x] for x in ("family", "run", "fragment_share_of_credit",
                                                                     "xg_fragment_share_of_credit", "peel_fragment_share_of_credit")},
           "worst_word_share": worst_word and {x: worst_word[x] for x in ("family", "run", "word_share_of_credit",
                                                                          "xg_word_share_of_credit", "peel_word_share_of_credit")},
           "worst_word_share_accepted": worst_acc and {x: worst_acc[x] for x in ("family", "run", "word_share_of_credit")},
           "falsifiers": [{x: r[x] for x in ("family", "run", "fragment_share_of_credit")} for r in falsified],
           "seconds": round(time.time() - t0, 1)}
    out = Path(a.out)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    ok = bad == 0 and not falsified and bool(scored)
    w = worst["fragment_share_of_credit"] if worst else 0.0
    run_result.write(out, ok, "pearlc-k256-gate", f"worst accepted fragment share {w:.6f} (threshold 0.0025), "
                     f"{len(falsified)} falsifiers", [("worst_fragment_share", w, "fraction")])
    print(json.dumps({x: v for x, v in doc.items() if x != "cells"}))


if __name__ == "__main__":
    main()
