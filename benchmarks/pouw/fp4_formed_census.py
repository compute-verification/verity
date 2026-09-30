"""The base-split falsifier and the debit's salt-dead statistics on Pearl-C4's exact forming, on real rows.

``fp4_census.py`` salts operands with stand-in row noise.  This script forms them as the design does (GPU 5's numpy twin,
``twin.form``: the noise atom per element on alpha * x, then the noisy block scale and the cast), from a checkout of the
Pearl-C4 branch that ``--twin-tree`` names, and reports, per (layer, linear, side) of ``fp4_quality.py --dump-acts``
rows, next to the stand-in's figures on the same rows:

- the clean-versus-salt code and value change densities, and salt versus salt;
- the share of 4-groups with >= 2 codes one salt left unchanged, and the base-split cheater's cost per MAC over aligned
  16 x 64 fragments (``fp4_census.base_split``'s rule);
- D-24's and the instruction's 2:4 predicates on the formed 16-row spans and row windows;
- on ``--witness-rows`` rows, the reference's own ``salt_dead`` against ``fp4_census.witness_salt_dead``.

Usage:  fp4_formed_census.py --twin-tree /path/to/pearl-c4-checkout --acts DUMPS [--fmt nvf4] [--out result.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fp4_census as C  # noqa: E402


def load_twin(tree: Path):
    for p in (tree / "packages/verity/src", tree / "protocols/pouw", tree / "benchmarks/pouw/pearl_c4"):
        sys.path.insert(0, str(p))
    import twin
    from verity_pouw.schemes import pearl_c4 as F
    return twin, F


def bf16_words(X: np.ndarray) -> np.ndarray:
    w = np.asarray(X, np.float32).view(np.uint32).astype(np.uint64)
    return (((w + 0x7FFF + ((w >> 16) & 1)) >> 16) << 16).astype(np.uint32)


def scale_values(F, fmt, twin, s: np.ndarray) -> np.ndarray:
    if fmt is F.NVFP4:
        return twin.ue4m3_value(s).astype(np.float64)
    return np.ldexp(1.0, s.astype(np.int64) - 127)


def frag_cost(live: np.ndarray, rows: int = 16) -> dict:
    """``fp4_census.base_split``'s cheater cost over aligned 16 x 64 fragments of a change map: 0 when nothing changed,
    0.5 when every (row, 4-group) has <= 2 changes (one mma.sp pass), else 1.0."""
    r, K = live.shape
    d = live[: r - r % rows, : K - K % 64].reshape(r // rows, rows, K // 64, 16, 4).transpose(0, 2, 1, 3, 4).sum(-1)
    skip = d.reshape(d.shape[0], d.shape[1], -1).sum(-1) == 0
    one = ~skip & (d.max((2, 3)) <= 2)
    return {"cost_per_mac": float(np.where(skip, 0.0, np.where(one, 0.5, 1.0)).mean()),
            "frag_one_pass": float(one.mean()), "frag_skip": float(skip.mean())}


def side_stats(twin, F, fmt, fmt_name: str, words: np.ndarray, seeds: list[bytes], role_e: int, role_f: int,
               witness_rows: int, rng) -> dict:
    rows, k = words.shape
    formed = []
    for sd in seeds:
        e = np.array([F.line(sd, role_e, i) for i in range(rows)], np.uint8)
        formed.append(twin.form(fmt, words, e, np.array(F.basis(sd, role_f, k), np.uint8)))
    alpha = formed[0].alpha
    c0, s0 = twin.quantize(fmt, twin.u32(twin.f32(words) * twin.f32(alpha)[:, None]))
    val = lambda c, s: (C.E2M1_VALUES[c & 7] * np.where(c & 8, -1, 1)).reshape(rows, -1, fmt.block) \
        * scale_values(F, fmt, twin, s)[..., None]
    z = lambda c: np.where(c & 7, c, 0)
    f1 = formed[0]
    live_code = z(f1.codes) != z(c0)
    live_value = (val(f1.codes, f1.scales) != val(c0, s0)).reshape(rows, k)
    g4 = lambda live: float((live[:, : k - k % 4].reshape(rows, -1, 4).sum(-1) <= 2).mean())
    sp = C.span_2of4((f1.codes & 7) != 0)
    out = {"rows": rows, "k": k,
           "exact": {"code_change": float(live_code.mean()), "value_change": float(live_value.mean()),
                     "salt_vs_salt_code": float((z(formed[0].codes) != z(formed[1].codes)).mean()),
                     "scale_change": float((f1.scales != s0).mean()),
                     "g4_ge2_dead_code": g4(live_code), "g4_ge2_dead_value": g4(live_value),
                     "basesplit_code": frag_cost(live_code), "basesplit_value": frag_cost(live_value),
                     "formed_span_d24": float(sp["span_d24"].mean()), "formed_span_pairs": float(sp["span_pairs"].mean()),
                     "formed_windows_d24": float(sp["g4"].reshape(rows, -1, 32).all(-1).mean()),
                     "formed_windows_pairs": float(sp["p8"].reshape(rows, -1, 16).all(-1).mean())}}
    X = twin.f32(words).astype(np.float64)
    sb = C.base_split(fmt_name, X, "row", 0.25, rng)
    out["stand_in"] = {"code_change": sb["code_change_density"], "g4_ge2_dead_code": sb["share_4groups_ge2_salt_dead"],
                       "basesplit_cost": sb.get("cheater_A_cost_per_dense_mac")}
    if witness_rows:
        idx = np.linspace(0, rows - 1, witness_rows).astype(int)
        ref = np.array([F.salt_dead(fmt, [int(v) for v in words[i]], int(f1.alpha[i]), int(f1.ae[i])) for i in idx])
        mine = C.witness_salt_dead(X[idx], fmt_name)
        out["witness"] = {"rows": idx.tolist(), "reference_salt_dead": float(ref.mean()),
                          "census_salt_dead": float(mine.mean()), "agree": float((ref == mine).mean()),
                          "census_dead_reference_live": float((mine & ~ref).mean()),
                          "census_live_reference_dead": float((~mine & ref).mean())}
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--twin-tree", type=Path, required=True)
    ap.add_argument("--acts", type=Path, required=True)
    ap.add_argument("--fmt", choices=["nvf4", "mxf4"], default="nvf4")
    ap.add_argument("--rows", type=int, default=128)
    ap.add_argument("--witness-rows", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    twin, F = load_twin(args.twin_tree)
    fmt = F.NVFP4 if args.fmt == "nvf4" else F.MXFP4
    rng = np.random.default_rng(args.seed)
    seeds = [hashlib.sha256(f"fp4-formed-census/{args.seed}/{i}".encode()).digest() for i in range(2)]
    res = {"twin_tree": str(args.twin_tree), "fmt": args.fmt, "acts": str(args.acts), "results": []}
    files = [args.acts] if args.acts.is_file() else sorted(args.acts.glob("*.npz"))
    for f in files:
        d = np.load(f)
        for lin in sorted({key.rsplit("/", 1)[0] for key in d.files}):
            cell = {"source": f.stem, "linear": lin}
            for side, key, re, rf in (("A", "x", F.ROLE_EA, F.ROLE_FA), ("B", "w", F.ROLE_EB, F.ROLE_FB)):
                words = bf16_words(d[f"{lin}/{key}"][: args.rows])
                cell[side] = side_stats(twin, F, fmt, args.fmt, words, seeds, re, rf, args.witness_rows, rng)
            res["results"].append(cell)
            print(json.dumps({"src": f.stem, "linear": lin, **{
                f"{s}_{q}": round(v, 4) for s in "AB" for q, v in (
                    ("code", cell[s]["exact"]["code_change"]), ("code_standin", cell[s]["stand_in"]["code_change"]),
                    ("ge2dead", cell[s]["exact"]["g4_ge2_dead_code"]),
                    ("ge2dead_standin", cell[s]["stand_in"]["g4_ge2_dead_code"]),
                    ("cost", cell[s]["exact"]["basesplit_code"]["cost_per_mac"]),
                    ("value_cost", cell[s]["exact"]["basesplit_value"]["cost_per_mac"]),
                    ("witness_agree", cell[s].get("witness", {}).get("agree", float("nan"))))}}), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
