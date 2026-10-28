"""Whether a retained pass's credited rows meet `rhoD_atomMixP_flat`'s domain conditions (`FormingP.lean`), CPU only.

Lean's ρ is free, while the served chain fixes α, β and X (`pearl_c.s5_stats`).  The largest ρ that `hsat`
(α·(|x| + 4ρ) ≤ 448) allows is ρ_max = (448/α − s)/4, with equality at the row's largest element, and it is the best ρ for
`hfloor` and `hlive`, which only get easier as ρ grows.  At ρ_max, per credited row i (δ = `Params.pi.δ` = 1):

- `hfloor` (1 ≤ α·ρ) holds iff α·s ≤ 444; the margin is 444 − α·s;
- `hlive` (`LiveOperand 1 ρ X`: 64·#{(i, l) : 8ρ_i ≤ |X i l|} ≤ m·k) counts x_l loud iff |x_l|·α ≥ 2·(448 − α·s), over a
  unit's credited rows (m their count); the slack is m·k − 64·loud;
- `hN`, `hc`, `hc4`: some t in 1…16 puts c = 5.5·t·V/(α·ρ_max) in [2667/5000, 4], V = e4m3(f32(β·44)) (the flat line
  with 16 − t negative entries).

So a row failing `hfloor` is outside the domain for every ρ, and so is a unit failing `hlive`.  The rows are BF16, so
x² is exact in FP32 and the stride chain is FP32 adds, done here in numpy (IEEE, nearest-even); a row with a nonzero stride
element below 2^-60 (a possibly subnormal square) goes through `pearl_c.s5_stats` instead, and a sample of every split's
rows is checked word for word against it, and against an all-Fraction recount, before anything is reported.  The products
α·s and |x|·α are exact in float64; a row whose 448 − α·s is not, or whose `hN` interval ends within 1e-9 of an integer t,
is recounted exactly.  Read-only on the pass.

  python benchmarks/pouw/pearl_c_vllm/forming_domain.py PASS_DIR [--splits prefill decode] [--jobs J] --out FILE
"""
from __future__ import annotations

import argparse
import json
import math
import mmap
import multiprocessing
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

from verity.ml.protocols.pouw.schemes import pearl_c as C
from verity.ml.protocols.pouw.schemes import pearl_kw as P

QMAX, FLOOR_AS, TILE = 448, 444, 64
C_LO, C_HI = Fraction(2667, 5000), Fraction(4)
TINY = 2.0 ** -60
CHUNK = 512
SAMPLES = 400
F32 = np.float32


def _f(w: int) -> Fraction:
    return Fraction(P.f32_of_bits(w))


def _v(beta_w: int) -> Fraction:
    return Fraction(P.fp8_to_f32(P.f32_to_fp8(P.f32_of_bits(P.f32_bits(float(F32(P.f32_of_bits(beta_w)) * F32(44.0)))))))


def _hn(v: Fraction, ar: Fraction) -> bool:
    if ar <= 0 or v <= 0:
        return False
    lo, hi = C_LO * ar / (Fraction(11, 2) * v), C_HI * ar / (Fraction(11, 2) * v)
    return max(1, math.ceil(lo)) <= min(16, math.floor(hi))


def exact_row(words: list[int]) -> dict:
    """One row by `pearl_c.s5_stats` and Fractions: the reference every vectorised row is held to."""
    s_w, rho_w, alpha_w, beta_w = C.s5_stats(words)
    alpha, s = _f(alpha_w), _f(s_w)
    a_s = alpha * s
    rhs = 2 * (QMAX - a_s)
    loud = sum(1 for w in words if abs(_f(w)) * alpha >= rhs)
    return {"words": (s_w, rho_w, alpha_w, beta_w), "a_s": a_s, "loud": loud, "hn": _hn(_v(beta_w), (QMAX - a_s) / 4),
            "halpha": alpha > 0, "scheme_floor": P.f32_of_bits(rho_w) * P.f32_of_bits(alpha_w) >= P.SIGMA_MIN}


def stats(u32: np.ndarray) -> tuple[np.ndarray, ...]:
    """(s, ρ, α, β) of each row of FP32 words in `pearl_c.s5_stats`'s order, and the rows it must take instead."""
    x = u32.view(F32)
    xs = x[:, ::C.S5_STRIDE]
    k8 = xs.shape[1]
    ax = np.abs(xs)
    fallback = ((ax < TINY) & (ax > 0)).any(axis=1) | ~np.isfinite(x).all(axis=1)
    lanes = np.concatenate([xs, np.zeros((xs.shape[0], -k8 % 32), F32)], axis=1).reshape(xs.shape[0], -1, 32)
    part = np.zeros((xs.shape[0], 32), F32)
    for g in range(lanes.shape[1]):
        part = part + lanes[:, g, :] * lanes[:, g, :]
    for o in (16, 8, 4, 2, 1):
        part = part + part[:, np.arange(32) ^ o]
    s = np.abs(x).max(axis=1)
    with np.errstate(over="ignore", invalid="ignore"):
        rho = np.sqrt(part[:, 0] / F32(k8))
        t = s + F32(4) * rho
        alpha = F32(448) / np.maximum(t, F32(2.0 ** -32))
        beta = (rho * alpha) * P.f32_of_bits(C.S5_LINE)
        fallback |= ~(np.isfinite(part).all(axis=1) & np.isfinite(t) & np.isfinite(beta))
    return s, rho, alpha, beta.astype(F32), fallback


def _words(a: np.ndarray) -> np.ndarray:
    return a.astype(F32).view(np.uint32)


def check_matmul(job: tuple) -> dict:
    """One unit's credited rows: their counts, the unit's liveness slack, and its sampled rows held to `exact_row`."""
    base, rec, sample = job
    m, k = rec["m"], rec["k"]
    with open(Path(base) / rec["rows"]["path"], "rb") as fh:
        data = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    if len(data) != rec["rows"]["bytes"] or len(data) != 2 * m * k:
        raise ValueError(f"{rec['rows']['path']}: {len(data)} bytes; the manifest says {rec['rows']['bytes']} for {m}×{k}")
    rows = np.frombuffer(data, "<u2").reshape(m, k)
    credited = np.setdiff1d(np.arange(rec["real_rows"]), np.asarray(rec["excluded_a"], dtype=np.int64))
    out = {"index": rec["index"], "m_cred": int(credited.size), "k": k, "loud": 0, "rows_out": 0, "min_margin": None,
           "min_margin_row": None, "scheme_floor_fail": 0, "halpha_fail": 0, "hn_fail": 0, "fallback": 0, "exact": 0,
           "max_out_per_tile": 0, "sampled": 0}
    out_rows = []
    for lo in range(0, credited.size, CHUNK):
        idx = credited[lo:lo + CHUNK]
        u32 = rows[idx].astype("<u4") << 16
        s, rho, alpha, beta, fallback = stats(u32)
        s64, a64 = s.astype(np.float64), alpha.astype(np.float64)
        a_s = a64 * s64
        rhs = 2 * (QMAX - a_s)
        ar = rhs / 8
        loud = (np.abs(u32.view(F32)).astype(np.float64) * a64[:, None] >= rhs[:, None]).sum(axis=1)
        v = _beta_values(beta)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_lo, t_hi = float(C_LO) * ar / (5.5 * v), float(C_HI) * ar / (5.5 * v)
            hn = (np.maximum(1, np.ceil(t_lo)) <= np.minimum(16, np.floor(t_hi))) & (ar > 0) & (v > 0)
            near = (np.abs(t_lo - np.round(t_lo)) < 1e-9 * np.maximum(1, t_lo)) | \
                   (np.abs(t_hi - np.round(t_hi)) < 1e-9 * np.maximum(1, t_hi))
        recount = fallback | near | ~np.isfinite(t_lo) | ~np.isfinite(t_hi)
        for r in range(idx.size):
            if not recount[r] and Fraction(float(rhs[r])) == 2 * (QMAX - Fraction(float(a64[r])) * Fraction(float(s64[r]))):
                continue
            ex = exact_row(u32[r].tolist())
            out["exact"] += 1
            out["fallback"] += int(fallback[r])
            s_w, rho_w, alpha_w, beta_w = ex["words"]
            s[r], rho[r], alpha[r], beta[r] = (P.f32_of_bits(w) for w in (s_w, rho_w, alpha_w, beta_w))
            a_s[r], loud[r], hn[r] = float(ex["a_s"]), ex["loud"], ex["hn"]
            if Fraction(a_s[r]) != ex["a_s"]:
                raise AssertionError("α·s not exact in float64")
        out["loud"] += int(loud.sum())
        out["halpha_fail"] += int((~(np.isfinite(alpha) & (alpha > 0))).sum())
        out["scheme_floor_fail"] += int((rho.astype(np.float64) * alpha.astype(np.float64) < P.SIGMA_MIN).sum())
        out["hn_fail"] += int((~hn).sum())
        bad = a_s > FLOOR_AS
        out["rows_out"] += int(bad.sum())
        out_rows.extend(int(i) for i in idx[bad])
        j = int(np.argmin(FLOOR_AS - a_s))
        margin = float(FLOOR_AS - a_s[j])
        if out["min_margin"] is None or margin < out["min_margin"]:
            out["min_margin"], out["min_margin_row"] = margin, int(idx[j])
        for r in np.nonzero(np.isin(idx, sample))[0]:
            ex = exact_row(u32[r].tolist())
            got = (int(_words(s)[r]), int(_words(rho)[r]), int(_words(alpha)[r]), int(_words(beta)[r]))
            if got != ex["words"] or int(loud[r]) != ex["loud"] or Fraction(float(a_s[r])) != ex["a_s"] or bool(hn[r]) != ex["hn"]:
                raise AssertionError(f"unit {rec['index']} row {int(idx[r])}: the vectorised row is not s5_stats' "
                                     f"({got} vs {ex['words']}, loud {int(loud[r])} vs {ex['loud']})")
            out["sampled"] += 1
    if out_rows:
        out["max_out_per_tile"] = int(np.bincount(np.asarray(out_rows) // TILE).max())
    out["slack"] = out["m_cred"] * k - 64 * out["loud"]
    del rows
    data.close()
    return out


def _beta_values(beta: np.ndarray) -> np.ndarray:
    """V = e4m3(f32(β·44)) of each row, through Pearl's cast once per distinct value."""
    y = beta * F32(44.0)
    uniq, inv = np.unique(y, return_inverse=True)
    vals = np.array([P.fp8_to_f32(P.f32_to_fp8(float(u))) if math.isfinite(u) else 0.0 for u in uniq], np.float64)
    return vals[inv.reshape(-1)]


def sample_rows(matmuls: list[dict], n: int = SAMPLES) -> list[list[int]]:
    """About n credited rows of a split, spread over its units: every unit's first, middle and last when there are few."""
    every = max(1, len(matmuls) // n)
    per = max(1, -(-n // len(matmuls)))
    picks = []
    for j, rec in enumerate(matmuls):
        cred = sorted(set(range(rec["real_rows"])) - set(rec["excluded_a"]))
        if j % every or not cred:
            picks.append([])
            continue
        picks.append(sorted({cred[(len(cred) - 1) * q // max(1, per - 1)] for q in range(per)}))
    return picks


def check_split(path: Path, jobs: int) -> dict:
    man = json.loads((path / "manifest.json").read_text())
    mats = man["matmuls"]
    work = [(str(path), rec, s) for rec, s in zip(mats, sample_rows(mats))]
    with multiprocessing.get_context("fork").Pool(jobs) as pool:
        units = list(pool.imap(check_matmul, work, chunksize=1 if len(work) < 1000 else 8))
    worst_slack = sorted(units, key=lambda u: u["slack"])[:5]
    margins = [u for u in units if u["min_margin"] is not None]
    worst_margin = sorted(margins, key=lambda u: u["min_margin"])[:5]
    def total(key):
        return sum(u[key] for u in units)
    return {"units": len(units), "rows_checked": total("m_cred"), "rows_out_of_domain": total("rows_out"),
            "min_margin": worst_margin[0]["min_margin"] if worst_margin else None,
            "min_margin_at": [{"unit": u["index"], "row": u["min_margin_row"], "margin": u["min_margin"]} for u in worst_margin],
            "units_failing_live": sum(1 for u in units if u["slack"] < 0), "min_slack": worst_slack[0]["slack"],
            "min_slack_at": [{"unit": u["index"], "slack": u["slack"], "m_cred": u["m_cred"], "k": u["k"], "loud": u["loud"]}
                             for u in worst_slack],
            "max_out_rows_per_tile": max(u["max_out_per_tile"] for u in units),
            "hn_fail": total("hn_fail"), "halpha_fail": total("halpha_fail"), "scheme_floor_fail": total("scheme_floor_fail"),
            "fallback_rows": total("fallback"), "exact_rows": total("exact"), "sampled_rows": total("sampled")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pass_dir", type=Path)
    ap.add_argument("--splits", nargs="+", default=["prefill", "decode"])
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    res = {"pass": str(a.pass_dir), "rule": "rhoD_atomMixP_flat at rho_max = (448/alpha - s)/4", "splits": {}}
    for split in a.splits:
        res["splits"][split] = r = check_split(a.pass_dir / split, a.jobs)
        print(split, json.dumps({k: r[k] for k in ("rows_checked", "rows_out_of_domain", "min_margin", "units",
                                                   "units_failing_live", "min_slack", "hn_fail", "sampled_rows")}), flush=True)
    res["hfloor_holds"] = all(r["rows_out_of_domain"] == 0 for r in res["splits"].values())
    res["hlive_holds"] = all(r["units_failing_live"] == 0 for r in res["splits"].values())
    res["hn_holds"] = all(r["hn_fail"] == 0 for r in res["splits"].values())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("hfloor_holds", "hlive_holds", "hn_holds")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
