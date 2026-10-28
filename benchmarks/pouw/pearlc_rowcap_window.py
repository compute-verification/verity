#!/usr/bin/env python3
"""Option (b) of card 82c1e288: how far -h3's per-row debit cap must widen at k = 256 for honest tiles to pass, and what
the widening gives away, on the served scheme ``pearl-c-h100-v1-h3`` (Hopper's atom, forming s5, ρ = 1/400).

At k = 256 and n = 6,144 (GLM-5.3's routed w2 and shared down_proj, one TP8 rank) a credited row's cap over a 64-column
tile is ρ·64·420 = 67 units and a flagged atom-word costs 40, so a row with two flagged atoms in a tile rejects it
(`clear_work.check_opened`'s row cap; `pearlCTilesDevRev1RowCap`).  Each cell here is one 64-row band over the whole n:
A′ rows from t(4) (or Gaussian) activations, B̃ rows a whole weight slice, every word's served debit (`pearlc_census.debit`
on all 64 x 6,144 words, spot-checked against `pearl_c_debit.word_debit`).  Every candidate keeps the tile-wide cap
(the tile's debit at most ρ of its credited cells' credit) and differs in its row rule:

* ``tile``: no row cap (the -h1/-h2 rule), the ceiling;
* ``row-x1``: the row cap per 64 x 64 tile at ρ (-h3 now); ``row-xC``: the row cap per tile at C·ρ;
* ``window-W``: the drawn tile is 64 x 64, and each of its credited rows' debit over its aligned window of W tiles
  (64·W columns) is at most ρ of that row's credit there;
* ``draw-W``: the audit tile is the window itself, 64 x 64·W, with the tile-wide and row caps over it; a 64 x 64 tile
  counts as accepted when its window is;
* ``served``: the scheme's own rule, its row cap `PearlC.row_cap_at(256)` per 64 x 64 tile (4ρ), with the
  rows an honest prover gives up to pass it (``served_rows_over``) and the debit share of the tiles it accepts.

``window-96`` and ``draw-96`` are the row's whole n.  For ``window-W`` the adversary's best concentration is measured too:
within each window, the 64 columns with the most debit packed into one tile (a weight order the prover may choose), which
keeps every row's window sum, hence the window cap, and reports that tile's debit and its worst row's, as shares of their
credit in units of ρ.

    uv run python benchmarks/pouw/pearlc_rowcap_window.py --out DIR [--snapshot SNAP] [--gaussian-runs 8] [--bands 2]
        [--slices 20] [--jobs 40]

`pearlc_rowcap_window.sh` is the `research run --cwd source` job on a CPU node holding GLM-5.3's snapshot.

Writes DIR/rowcap_window.json and DIR/cells.jsonl; under `research run` its status is the run's: it passes when every planned
cell is in domain and every spot-checked word's flags and +0 promotions agree with the served `word_debit`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from fractions import Fraction
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "pearl_c_sm90"))
import glm53_slices as SL  # noqa: E402
import pearlc_census as K  # noqa: E402
import pearlc_k256_gate as G  # noqa: E402
import run_result  # noqa: E402
from verity.core.protocols.pouw.protocol import Context, Matmul, Shape, Tile  # noqa: E402
from verity.ml.protocols.pouw.schemes import get  # noqa: E402
from verity.ml.protocols.pouw.schemes import pearl_c_debit as D  # noqa: E402

SCHEME, K_, N, ROWS, TN = "pearl-c-h100-v1-h3", 256, 6144, 64, 64
NT = N // TN
ROW_X = (Fraction(1), Fraction(3, 2), Fraction(2), Fraction(3), Fraction(4), Fraction(6), Fraction(8))
WINDOWS = (2, 4, 8, 16, 32, 96)
SPOT = 400
#: (layer, expert or None for the shared expert, rank): the gate's eight w2 samples' slices, then twelve more across depth
SLICES = SL.SAMPLES + ((7, 33, 1), (15, 99, 5), (25, None, 0), (30, 250, 3), (35, 64, 6), (45, None, 2), (55, 11, 7),
                       (65, 180, 0), (70, None, 4), (74, 90, 2), (78, 5, 1), (78, None, 6))
SNAP = ""


def slice_name(layer: int, expert: int | None) -> str:
    mid = "shared_experts" if expert is None else f"experts.{expert}"
    return f"model.layers.{layer}.mlp.{mid}.down_proj.weight"


def load_slice(layer: int, expert: int | None, rank: int) -> np.ndarray:
    """Rank `rank`'s 256-wide slice of the down projection, n x 256 FP32 (`glm53_slices`'s widening)."""
    snap = Path(SNAP)
    wm = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    name = slice_name(layer, expert)
    w = SL.widen(SL.tensor(snap / wm[name], name), SL.tensor(snap / wm[name + "_scale_inv"], name + "_scale_inv"))
    assert w.shape[0] == N, (name, w.shape)
    return np.ascontiguousarray(w[:, K_ * rank:K_ * rank + K_])


def le_cap(debit: np.ndarray, count: np.ndarray, c: Fraction, rho: Fraction, credit: int) -> np.ndarray:
    """debit ≤ c·ρ·(credit/N)·count, exactly, on integer debits."""
    q = c * rho
    return (debit.astype(object) * (N * q.denominator) <= count.astype(object) * (q.numerator * credit)).astype(bool)


def share(debit: int, count: int, rho: Fraction, credit: int) -> float:
    """debit as a share of `count` cells' credit, in units of ρ."""
    return float(Fraction(int(debit) * N, int(count) * credit) / rho) if count else 0.0


def cell(fam: str, run: int, sl: tuple | None) -> dict:
    t0 = time.time()
    s = get(SCHEME)
    rho, credit = s.cap, s.credit_of(Shape(1, K_, N))
    au, fadd = D.atom_units(s.device, s.prices), s.prices.fadd
    assert au.denominator == 1 and int(fadd) == fadd
    rng = K._rng(run, fam, K_, 1.0)
    if sl is None:
        XA, XB = K.family("gaussian", ROWS, N, K_, K.lines(K_, K._rng(run, "FA", K_)), rng)
    else:
        XA, XB = rng.standard_t(4, (ROWS, K_)), load_slice(*sl).astype(np.float64)
    ctx = Context(G._h(run, fam, "salt")[:32], Matmul(0, Shape(ROWS, K_, N), 1), G._h(run, fam, "root-a"),
                  G._h(run, fam, "root-b"))
    ops = s.encode(ctx, s.noise(ctx), G.words_of(XA), G.words_of(XB), Tile(tuple(range(ROWS)), tuple(range(N))))
    row = {"family": fam, "run": run, "slice": sl}
    if ops is None:
        return {**row, "in_domain": False}
    ok_a, ok_b = G.row_ok(ops.a), G.row_ok(ops.b)
    cred = ok_a[:, None] & ok_b[None, :]
    A8, B8 = np.array(ops.a.codes, dtype=np.uint8), np.array(ops.b.codes, dtype=np.uint8)
    ii, jj = np.meshgrid(np.arange(ROWS), np.arange(N), indexing="ij")
    A, B = A8[ii.ravel()], B8[jj.ravel()]
    flags, _ = K.debit(A, B, 4)
    first, _ = K.group_run(A, B, [0, 1, 2, 3])
    zeros = first == 0
    del A, B
    spot = np.random.default_rng(run * 7919 + len(fam)).choice(ROWS * N, SPOT, replace=False)
    mismatches = 0
    for w in spot:
        i, j = divmod(int(w), N)
        f, z = D.word_debit(ops.a.codes[i], ops.b.codes[j], s.device)
        mismatches += f != list(flags[w]) or z != int(zeros[w])
    word = (int(au) * flags.sum(1) + int(fadd) * zeros).reshape(ROWS, N) * cred
    dt = word.reshape(ROWS, NT, TN).sum(2)                       # (row, tile) debit over credited words
    nt = cred.reshape(ROWS, NT, TN).sum(2)                       # (row, tile) credited words
    agg_ok = le_cap(dt.sum(0), nt.sum(0), Fraction(1), rho, credit) & (nt.sum(0) > 0)
    rows_live = ok_a
    acc: dict[str, list[bool]] = {"tile": agg_ok.tolist()}
    over: dict[str, int] = {}
    for c in ROW_X:
        r_ok = le_cap(dt, nt, c, rho, credit) | ~rows_live[:, None]
        acc[f"row-x{c}"] = (agg_ok & r_ok.all(0)).tolist()
        over[f"row-x{c}"] = int((~r_ok).sum())
    served = s.row_cap_at(K_) / rho
    served_r_ok = le_cap(dt, nt, served, rho, credit) | ~rows_live[:, None]
    served_ok = agg_ok & served_r_ok.all(0)
    acc["served"], over["served"] = served_ok.tolist(), int((~served_r_ok).sum())
    conc = {}
    for W in WINDOWS:
        dw = dt.reshape(ROWS, NT // W, W).sum(2)
        nw = nt.reshape(ROWS, NT // W, W).sum(2)
        r_ok = le_cap(dw, nw, Fraction(1), rho, credit) | ~rows_live[:, None]
        w_rows = np.repeat(r_ok.all(0), W)
        acc[f"window-{W}"] = (agg_ok & w_rows).tolist()
        over[f"window-{W}"] = int((~r_ok).sum())
        w_agg = le_cap(dw.sum(0), nw.sum(0), Fraction(1), rho, credit) & (nw.sum(0) > 0)
        acc[f"draw-{W}"] = np.repeat(w_agg & r_ok.all(0), W).tolist()
        worst_tile = worst_row = 0.0
        packed_agg_ok = 0
        for b in range(NT // W):
            if not r_ok[:, b].all():
                continue
            cols = slice(b * W * TN, (b + 1) * W * TN)
            top = np.argsort(-word[:, cols].sum(0), kind="stable")[:TN]
            pd, pn = word[:, cols][:, top], cred[:, cols][:, top]
            t_share = share(pd.sum(), pn.sum(), rho, credit)
            r_share = max((share(pd[i].sum(), pn[i].sum(), rho, credit) for i in range(ROWS) if rows_live[i]), default=0.0)
            worst_tile, worst_row = max(worst_tile, t_share), max(worst_row, r_share)
            packed_agg_ok += t_share <= 1
        conc[W] = {"windows_under_row_cap": int(r_ok.all(0).sum()), "packed_tile_share_max": worst_tile,
                   "packed_row_share_max": worst_row, "packed_tiles_under_tile_cap": packed_agg_ok}
    tile_share = [share(dt[:, t].sum(), nt[:, t].sum(), rho, credit) for t in range(NT)]
    row_tile_share = [share(dt[i, t], nt[i, t], rho, credit) for i in range(ROWS) if rows_live[i] for t in range(NT)]
    hist = np.bincount((flags.sum(1).reshape(ROWS, N) * cred).reshape(ROWS, NT, TN).sum(2)[rows_live].ravel())
    row.update({"in_domain": bool(cred.any()), "rows_ok_A": int(ok_a.sum()), "rows_ok_B": int(ok_b.sum()),
                "spot_words": SPOT, "spot_mismatches": int(mismatches), "flag_share": float(flags[cred.ravel()].mean()),
                "later_zero_promotions": int((zeros & cred.ravel()).sum()),
                "band_debit_share_rho": share(dt.sum(), nt.sum(), rho, credit),
                "tile_share_rho_max": max(tile_share), "row_tile_share_rho_max": max(row_tile_share, default=0.0),
                "rows_over_row_cap_somewhere": int(((~(le_cap(dt, nt, Fraction(1), rho, credit))).any(1) & rows_live).sum()),
                "row_tile_flagged_atoms_hist": hist.tolist(), "accepted": acc, "row_tiles_over": over,
                "served_rows_over": int(((~served_r_ok).any(1) & rows_live).sum()),
                "served_accepted_tile_share_rho_max": max((tile_share[t] for t in range(NT) if served_ok[t]), default=0.0),
                "concentration": conc, "seconds": round(time.time() - t0, 1)})
    return row


def _init(snap: str) -> None:
    global SNAP
    SNAP = snap


def _run(job):
    return cell(*job)


def gamma_shift(s) -> dict:
    """The sampled γ's growth when the split game's per-row cap is ρ_row instead of ρ (TileCapRev1Gamma's transfer, run
    the other way): 1 − γ₀' = (1 − 1/400)·(1 − ρ_row)/(1 − ρ), γ = 1 − (1 − γ₀')/ω, ω = W_ref/((1 − ρ)·credit)."""
    sh = Shape(1, K_, N)
    rho = s.cap
    omega = Fraction(s.wref(sh)) / ((1 - rho) * s.credit_of(sh))
    out = {"omega": float(omega)}
    base = 1 - Fraction(399, 400) / omega
    for c in sorted({*ROW_X, *map(Fraction, WINDOWS)}):
        g = 1 - Fraction(399, 400) * (1 - c * rho) / ((1 - rho) * omega)
        out[str(c)] = {"rho_row": float(c * rho), "gamma": float(g), "gamma_minus_h3": float(g - base)}
    rr = s.row_cap_at(K_)
    g = 1 - Fraction(399, 400) * (1 - rr) / ((1 - rho) * omega)
    out["served"] = {"rho_row": str(rr), "gamma": str(g), "gamma_float": float(g)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--snapshot", default="")
    ap.add_argument("--gaussian-runs", type=int, default=8)
    ap.add_argument("--bands", type=int, default=2)
    ap.add_argument("--slices", type=int, default=len(SLICES))
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    work = [("gaussian", r, None) for r in range(1, a.gaussian_runs + 1)]
    if a.snapshot:
        work += [(f"glm-w2-L{sl[0]}-{'s' if sl[1] is None else f'e{sl[1]}'}-r{sl[2]}", r, sl)
                 for sl in SLICES[:a.slices] for r in range(1, a.bands + 1)]
    t0 = time.time()
    with Pool(max(1, a.jobs), initializer=_init, initargs=(a.snapshot,)) as pool:
        cells = pool.map(_run, work, chunksize=1)
    with (out / "cells.jsonl").open("w") as f:
        for c in cells:
            f.write(json.dumps(c) + "\n")
    s = get(SCHEME)
    fams: dict = {}
    for c in cells:
        if not c["in_domain"]:
            continue
        f = fams.setdefault("gaussian" if c["family"] == "gaussian" else "glm-w2",
                            {"bands": 0, "tiles": 0, "rows": 0, "rows_over_row_cap_somewhere": 0, "accepted": {},
                             "row_tiles_over": {}, "band_debit_share_rho_max": 0.0, "tile_share_rho_max": 0.0,
                             "row_tile_share_rho_max": 0.0, "flagged_atoms_per_row_tile": [], "concentration": {},
                             "served_rows_over": 0, "served_accepted_tile_share_rho_max": 0.0, "band_debit_share_rho_sum": 0.0})
        f["bands"] += 1
        f["tiles"] += NT
        f["rows"] += c["rows_ok_A"]
        f["rows_over_row_cap_somewhere"] += c["rows_over_row_cap_somewhere"]
        f["served_rows_over"] += c["served_rows_over"]
        f["band_debit_share_rho_sum"] += c["band_debit_share_rho"]
        f["served_accepted_tile_share_rho_max"] = max(f["served_accepted_tile_share_rho_max"],
                                                      c["served_accepted_tile_share_rho_max"])
        for k, v in c["accepted"].items():
            f["accepted"][k] = f["accepted"].get(k, 0) + sum(v)
        for k, v in c["row_tiles_over"].items():
            f["row_tiles_over"][k] = f["row_tiles_over"].get(k, 0) + v
        for k in ("band_debit_share_rho", "tile_share_rho", "row_tile_share_rho"):
            f[k + "_max"] = max(f[k + "_max"], c[k + "_max" if k != "band_debit_share_rho" else k])
        h = f["flagged_atoms_per_row_tile"]
        for i, n in enumerate(c["row_tile_flagged_atoms_hist"]):
            h.extend([0] * (i + 1 - len(h)))
            h[i] += n
        for W, v in c["concentration"].items():
            g = f["concentration"].setdefault(str(W), {"windows_under_row_cap": 0, "packed_tile_share_max": 0.0,
                                                       "packed_row_share_max": 0.0, "packed_tiles_under_tile_cap": 0})
            g["windows_under_row_cap"] += v["windows_under_row_cap"]
            g["packed_tiles_under_tile_cap"] += v["packed_tiles_under_tile_cap"]
            g["packed_tile_share_max"] = max(g["packed_tile_share_max"], v["packed_tile_share_max"])
            g["packed_row_share_max"] = max(g["packed_row_share_max"], v["packed_row_share_max"])
    for f in fams.values():
        f["acceptance"] = {k: round(v / f["tiles"], 4) for k, v in f["accepted"].items()}
        f["served_rows_over_share"] = round(f["served_rows_over"] / max(1, f["rows"]), 6)
        f["band_debit_share_rho_mean"] = f.pop("band_debit_share_rho_sum") / max(1, f["bands"])
    planned = len(work)
    done = sum(c["in_domain"] for c in cells)
    bad = sum(c.get("spot_mismatches", 0) for c in cells)
    ok = done == planned and bad == 0 and (not a.snapshot or "glm-w2" in fams)
    doc = {"tool": "pouw_pearlc_rowcap_window", "scheme": SCHEME, "k": K_, "n": N, "tile": [ROWS, TN], "rho": str(s.cap),
           "served_row_cap": str(s.row_cap_at(K_)),
           "credit_per_word": str(Fraction(s.credit_of(Shape(1, K_, N)), N)), "atom_units": str(D.atom_units(s.device, s.prices)),
           "cells_planned": planned, "cells_in_domain": done, "spot_mismatches": bad, "families": fams,
           "gamma_shift": gamma_shift(s), "seconds": round(time.time() - t0, 1)}
    path = out / "rowcap_window.json"
    path.write_text(json.dumps(doc, indent=1) + "\n")
    m = [(f"{fam}_acceptance_{k}", v, "fraction") for fam, f in fams.items() for k, v in f["acceptance"].items()]
    run_result.write(path, ok, "pearlc-rowcap-window",
                     f"{done}/{planned} cells in domain, {bad} spot mismatches against the served word_debit", m)
    print(json.dumps({x: v for x, v in doc.items() if x != "families"}, indent=1))
    for fam, f in fams.items():
        print(fam, json.dumps(f["acceptance"]))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
