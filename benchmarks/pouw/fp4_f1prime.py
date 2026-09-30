"""F1' (theory-pearl-c4-domain.md section 6.2, 13:35Z), the base split's per-window debit, on ``fp4_census.family`` rows,
next to the split's measured cost on the same rows under Pearl-C4's exact forming.

The rule runs as its author ran it: ``dnc2.py`` and the modules it imports, from ``--rule-dir``
(``internal/pouw/cheap-binding/pearlc4-fix`` in the project store), with ``fp4_emulation.py`` from ``--emu-dir``; their
sha256 go in the output.  Per row window w of 128 on each side: S_w (the split's expected saving at the cheater's best
within-block layout), M_w (the window's blocks whose realised scale byte isn't their modal byte) and
debit = max(0, S_w - min(C * M_w, 0.125)), at the rule's C = 16.92/128 and at ``--fp32-op``/128.  A tile's debit is its A
rows' mean plus its B rows' mean (``dnc2``'s own census), against the cap.

The measured side forms each operand once with GPU 5's twin (``--twin-tree``) and prices the split over aligned 16 x 64
fragments (``fp4_formed_census.frag_cost``: skip, one ``mma.sp`` pass, dense, or a CUDA-core correction at ``--fp32-op``
FP4 MACs per changed code), against the clean codes and scales, on two change maps:

- ``value``: every code of a block whose scale moved counts as changed, so A0 = the clean values and the decomposition is
  exact on both sides (#545's route);
- ``code_scale``: changed codes only, plus F1''s own scale correction min(C * M_w, 0.125) on the blocks whose scale
  moved.  That takes A0's per-block products as free, which the rule grants and a full-rank row with a salt-keyed B
  doesn't, so it bounds the split from the cheater's side.

Saving = 1 - cost per side; two-sided, the cheater pays both sides' corrections (full-rank A0 and a salt-keyed B), so its
saving is 1 - (cost_A + cost_B).  A tile over the cap is rejected (coverage); one within it nets a gain when a route saves
more than the tile's debit.

Usage:  fp4_f1prime.py --twin-tree PEARL_C4_CHECKOUT --rule-dir DNC2_DIR --emu-dir FP4_EMULATION_DIR [--families F,G]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fp4_census as C  # noqa: E402
import fp4_formed_census as FC  # noqa: E402

RULE_FILES = ("dnc2.py", "dnc.py", "count_real.py", "coverage.py", "dnc_attack.py", "fix_check.py")
FAMILIES = "stride-row-grid,stride-row-grid-edge,stride-row-tight,stride-row,spiky-row,gaussian"


def load_rule(rule_dir: Path, emu_dir: Path):
    for p in (emu_dir, rule_dir):
        sys.path.insert(0, str(p))
    mods = {m: importlib.import_module(m) for m in ("dnc2", "dnc", "count_real", "coverage")}
    digests = {f: hashlib.sha256((rule_dir / f).read_bytes()).hexdigest() for f in RULE_FILES if (rule_dir / f).exists()}
    digests["fp4_emulation.py"] = hashlib.sha256((emu_dir / "fp4_emulation.py").read_bytes()).hexdigest()
    return mods, digests


def f1prime(rule, X: np.ndarray, rng, c_values: dict) -> dict:
    """``dnc2``'s F1' on the rows X (BF16 values), as its ``__main__`` runs it on a family, at each C."""
    d2, dnc, cr, cov = rule["dnc2"], rule["dnc"], rule["count_real"], rule["coverage"]
    alpha, _, dnf, rho2 = cov.stats(X)
    p, scr = cr.p_mode(X)
    p = np.where(scr, 0.0, p)
    mod, rea = dnc.modal_and_realised(X, alpha, rho2, rng)
    off = (mod != rea).astype(int)
    R, k = X.shape
    mw = off[:, : k // 128 * 8].reshape(R, -1, 8).sum(-1)
    out = {"rows_dnf": float(np.mean(dnf)), "screened_blocks": float(scr[:, ::16].mean()),
           "modal_byte_share": float(1 - off.mean()), "M_w_mean": float(mw.mean())}
    c0, vx0 = d2.C, d2.VX
    try:
        for name, c in c_values.items():
            d2.C, d2.VX = c, np.array([max(0.0, 0.5 - c * x) for x in range(d2.XM + 1)])
            s = d2.window_saving(p[:, : k // 128 * 128].reshape(-1, 128)).reshape(R, -1)
            deb = d2.debit(p, off)
            out[name] = {"C": c, "S_w_mean": float(s.mean()), "scale_term_mean": float(np.minimum(c * mw, 0.125).mean()),
                         "debit_mean": float(deb.mean()), "debit_row_max": float(deb.mean(1).max()),
                         "windows_debited": float((deb > 0).mean())}
    finally:
        d2.C, d2.VX = c0, vx0
    return out


def measured(twin, F, fmt, words: np.ndarray, seed: bytes, role_e: int, role_f: int, fp32_op: float, c: float) -> dict:
    """The split's cost per MAC on one salt's exact forming of one side."""
    rows, k = words.shape
    e = np.array([F.line(seed, role_e, i) for i in range(rows)], np.uint8)
    f1 = twin.form(fmt, words, e, np.array(F.basis(seed, role_f, k), np.uint8))
    c0, s0 = twin.quantize(fmt, twin.u32(twin.f32(words) * twin.f32(f1.alpha)[:, None]))
    val = lambda cc, s: (C.E2M1_VALUES[cc & 7] * np.where(cc & 8, -1, 1)).reshape(rows, -1, fmt.block) \
        * FC.scale_values(F, fmt, twin, s)[..., None]  # noqa: E731
    z = lambda cc: np.where(cc & 7, cc, 0)  # noqa: E731
    live_code = z(f1.codes) != z(c0)
    live_value = (val(f1.codes, f1.scales) != val(c0, s0)).reshape(rows, k)
    moved = f1.scales != s0
    mw = moved[:, : k // 128 * 8].reshape(rows, -1, 8).sum(-1)
    code, value = FC.frag_cost(live_code, fp32_op=fp32_op), FC.frag_cost(live_value, fp32_op=fp32_op)
    scale = float(np.minimum(c * mw, 0.125).mean())
    return {"code_change": float(live_code.mean()), "value_change": float(live_value.mean()),
            "scale_moved": float(moved.mean()), "M_w_mean": float(mw.mean()),
            "cost_value": value["cost_per_mac_with_cuda_core"], "cost_value_2of4_only": value["cost_per_mac"],
            "cost_code": code["cost_per_mac_with_cuda_core"], "scale_term": scale,
            "cost_code_scale": code["cost_per_mac_with_cuda_core"] + scale}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--twin-tree", type=Path, required=True)
    ap.add_argument("--rule-dir", type=Path, required=True)
    ap.add_argument("--emu-dir", type=Path, required=True)
    ap.add_argument("--families", default=FAMILIES)
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--k", type=int, default=8192)
    ap.add_argument("--fp32-op", type=float, default=2 * 1047 / 125,
                    help="FP4 MACs per FP32 op: twice the in-loop FADD's 1047/125 = 8.376 FP8 MACs (13:17Z ruling)")
    ap.add_argument("--rule-fp32-op", type=float, default=16.92, help="the FP32 price dnc2.py's C was run at")
    ap.add_argument("--cap", type=float, default=1 / 400)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    twin, F = FC.load_twin(args.twin_tree)
    rule, digests = load_rule(args.rule_dir, args.emu_dir)
    c_values = {"as_run": args.rule_fp32_op / 128, "ruled": args.fp32_op / 128}
    rng, rule_rng = np.random.default_rng(args.seed), np.random.default_rng(args.seed + 1)
    seed = hashlib.sha256(f"fp4-f1prime/{args.seed}".encode()).digest()
    res = {"twin_tree": str(args.twin_tree), "rule_sha256": digests, "fp32_op": args.fp32_op,
           "rule_fp32_op": args.rule_fp32_op, "cap": args.cap, "rows": args.rows, "k": args.k, "seed": args.seed,
           "results": []}
    for fam in args.families.split(","):
        A, B = C.family(fam, args.rows, args.k, rng)
        cell = {"family": fam}
        for side, M, re, rf in (("A", A, F.ROLE_EA, F.ROLE_FA), ("B", B, F.ROLE_EB, F.ROLE_FB)):
            w = FC.bf16_words(M)
            cell[side] = {"f1prime": f1prime(rule, twin.f32(w).astype(np.float64), rule_rng, c_values),
                          "measured": measured(twin, F, F.NVFP4, w, seed, re, rf, args.fp32_op, c_values["ruled"])}
        for cn in c_values:
            tile = cell["A"]["f1prime"][cn]["debit_mean"] + cell["B"]["f1prime"][cn]["debit_mean"]
            cell[cn] = {"tile_debit": tile, "rejected": tile > args.cap}
        m = {s: cell[s]["measured"] for s in "AB"}
        routes = {"value": (m["A"]["cost_value"], m["B"]["cost_value"]),
                  "code_scale": (m["A"]["cost_code_scale"], m["B"]["cost_code_scale"])}
        cell["saving"] = {r: {"one_sided_A": 1 - a, "one_sided_B": 1 - b, "two_sided": 1 - a - b}
                          for r, (a, b) in routes.items()}
        best = max(v["two_sided"] for v in cell["saving"].values())
        t = cell["ruled"]
        cell["net_gain_two_sided"] = 0.0 if t["rejected"] else max(0.0, best - t["tile_debit"])
        res["results"].append(cell)
        print(json.dumps({"family": fam, **{f"{s}_{q}": round(v, 4) for s in "AB" for q, v in (
            ("S_w", cell[s]["f1prime"]["ruled"]["S_w_mean"]), ("modal", cell[s]["f1prime"]["modal_byte_share"]),
            ("debit", cell[s]["f1prime"]["ruled"]["debit_mean"]), ("cost_value", m[s]["cost_value"]),
            ("cost_code_scale", m[s]["cost_code_scale"]))},
            "tile_debit": round(t["tile_debit"], 4), "tile_debit_as_run": round(cell["as_run"]["tile_debit"], 4),
            "rejected": t["rejected"], **{f"saving2_{r}": round(v["two_sided"], 4) for r, v in cell["saving"].items()},
            "net_gain": cell["net_gain_two_sided"]}), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
