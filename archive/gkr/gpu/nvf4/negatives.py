"""The fp4-nvf4 negative battery as ``gpu.bb_export`` directories: ``DIR`` (the statement files + an honest witness of the
first ``--vus`` frozen VUs: ``units.bin`` / ``epilogue.bin`` / ``public.bin``) and ``DIR/neg/<id>/`` (the same three files
for one dishonest statement or witness each).  ``python -m gpu.run negatives DIR --proof-dir PD`` proves every one (the
proofs a dishonest prover would send) and must reject each; ``verity-gkr-verify negatives --dir DIR --proofs PD`` must
reject every proof too.

    python -m gpu.nvf4.negatives build --stmt STMT --out DIR [--vus 14] [--variant-vus 512] [--row-picks 60]

Cases (every one must be rejected):
  claim.*       the honest witness, a wrong public word: each of the 32 bits of VU 3's final FP32 word flipped (the bit's
                component: sign, exponent field, fraction), and the fraction / exponent field +-1
  link.*        a chain link broken (unit 10 of VU 2 takes an accumulator component the previous unit did not output), a
                nonzero c_0 component
  variant.*     a prover following a WRONG model rule (``fp4/witness.VARIANTS``) feeding its own accumulators forward and
                claiming its own final word, on a VU where that word differs from the model's (``fp4/chain._variant_base``)
  row.*         one committed column of one unit + 1 or + 2, for the first column of every (scope, kind) class
  domain.*      an operand code off the domain (a NaN scale byte 0x7F in a or b, an E2M1 code >= 16) in a code column
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
import torch

from archive.direct.ligero.fp4 import relation as R

from ..field import P
from .circuit import _pair_prod, _scale_x, statement
from .witness import NVF4Generator, load_instances

STATEMENT_FILES = ("circuit.txt", "epilogue.txt", "chain.txt", "manifest.json")


def write_rows_np(a, path: Path) -> None:
    a = np.asarray(a, dtype=np.int64) % P
    if a.ndim == 1:
        a = a[:, None]
    with open(path, "wb") as f:
        f.write(struct.pack("<QQ", a.shape[0], a.shape[1]))
        f.write(a.astype("<u4").tobytes())


def _write(d: Path, units, epi, pub) -> None:
    d.mkdir(parents=True, exist_ok=True)
    write_rows_np(units, d / "units.bin")
    write_rows_np(epi, d / "epilogue.bin")
    write_rows_np(pub, d / "public.bin")


def _words(y: np.ndarray) -> np.ndarray:
    return np.stack([(y >> 31) & 1, (y >> 23) & 0xFF, y & (2**23 - 1)], axis=1)


def build(stmt: Path, out: Path, vus: int, variant_vus: int, row_picks: int, device: str) -> dict:
    from archive.direct.ligero.fp4.chain import _variant_base, instances_fp4
    from archive.direct.ligero.fp4.witness import VARIANTS

    st = statement()
    if st.ue.text != (stmt / "circuit.txt").read_text():
        raise SystemExit(f"{stmt}: not this tree's gpu.nvf4.circuit export")
    out.mkdir(parents=True, exist_ok=True)
    for n in STATEMENT_FILES:
        shutil.copyfile(stmt / n, out / n)
    gen = NVF4Generator(st, device)
    dev = gen.dev
    steps = R.STEPS
    names = st.ue.col_names
    col = {n: i for i, n in enumerate(names)}

    def rows_of(A, B, Y, variant=None):
        r = gen.run(torch.as_tensor(A).to(dev).long(), torch.as_tensor(B).to(dev).long(), torch.as_tensor(Y).to(dev), variant=variant)
        return r.units.cpu().numpy(), r.epilogue.cpu().numpy(), r.public.cpu().numpy(), r.bad.cpu().numpy()

    A, B, Y = load_instances(0, vus, 1)
    U, E, W, bad = rows_of(A, B, Y)
    if bad.any():
        raise SystemExit(f"honest VUs rejected at witness generation: {np.nonzero(bad)[0].tolist()}")
    _write(out, U, E, W)
    cases: dict[str, dict] = {}                   # neg/<id> (numeric: the Rust verifier's `negatives` reads <id>.bin)

    def neg(name: str, why: str, units=None, epi=None, pub=None):
        i = str(len(cases))
        _write(out / "neg" / i, U if units is None else units, E if epi is None else epi, W if pub is None else pub)
        cases[i] = {"name": name, "why": why}

    # claims
    v = 3
    for bit in range(32):
        y = int(Y[v]) ^ (1 << bit)
        pub = W.copy()
        pub[v] = _words(np.array([y], dtype=np.int64))[0]
        neg(f"claim.bit{bit}", f"VU {v}'s claimed word, bit {bit} flipped", pub=pub)
    for comp, ci in (("t", 1), ("f", 2)):
        for d in (1, -1):
            pub = W.copy()
            pub[v, ci] = (pub[v, ci] + d) % P
            neg(f"claim.{comp}{'+' if d > 0 else '-'}1", f"VU {v}'s claimed {comp} {d:+d}", pub=pub)
    # links
    for comp in ("acc.s", "acc.t", "acc.f"):
        u = U.copy()
        u[2 * steps + 10, col[comp]] = (u[2 * steps + 10, col[comp]] + 1) % P
        neg(f"link.{comp}", f"VU 2 unit 10 {comp} != unit 9's output (+1)", units=u)
        u = U.copy()
        u[1 * steps, col[comp]] = 1
        neg(f"link.c0.{comp}", f"VU 1 unit 0 {comp} = 1 (c_0 != 0)", units=u)
    # wrong-rule witnesses, own claim
    data = instances_fp4(variant_vus)
    not_exercised = []
    for var in VARIANTS:
        vb = _variant_base(data, var)
        if vb is None:
            not_exercised.append(var)
            continue
        vus_v, vi, step = vb
        Av = np.stack([np.asarray(x[0], dtype=np.int64) for x in vus_v])
        Bv = np.stack([np.asarray(x[1], dtype=np.int64) for x in vus_v])
        Yv = np.array([int(x[3]) for x in vus_v], dtype=np.int64)
        try:
            u, e, w, b = rows_of(Av, Bv, Yv, variant=var)
        except (AssertionError, ValueError) as ex:
            not_exercised.append(f"{var} (no witness in the relation's shape: {type(ex).__name__})")
            continue
        neg(f"variant.{var}", f"wrong rule {var}, own claim (VU {vi}, first differing step {step})", units=u, epi=e, pub=w)
    # one committed column + delta, first column of every (scope, kind)
    rng = np.random.default_rng(11)
    seen, picks = set(), []
    for c, r in enumerate(st.ue.col_rows):
        if r < 0:
            continue
        row = st.unit.sys.rows[r]
        key = (".".join(row.name.split(".")[:3]), row.kind)
        if key not in seen:
            seen.add(key)
            picks.append((c, row))
    for c, row in picks[:row_picks]:
        unit = int(rng.integers(0, vus * steps))
        u = U.copy()
        old = int(u[unit, c])
        if re.fullmatch(r"x\.[ps]\d+", names[c]):
            # an operand code is witness (the statement is the chain's endpoints): a code with the same decoded outputs is
            # another witness of the same statement, so the mutation takes the nearest code whose table row differs
            fn, n = (_pair_prod, 256) if names[c].startswith("x.p") else (_scale_x, 128 * 127)
            dom = (lambda k: True) if n == 256 else (lambda k: (k & 127) < 127)
            new = next(k for d in range(1, n) for k in ((old + d) % n,) if dom(k) and fn(k) != fn(old))
            u[unit, c] = new
            neg(f"row.{c}", f"operand code {names[c]} {old} -> {new} (decode {fn(old)} -> {fn(new)}) at unit {unit}", units=u)
            continue
        delta = int(rng.integers(1, 3))
        u[unit, c] = (old + delta) % P
        neg(f"row.{c}", f"column {names[c]} ({row.kind}) +{delta} at unit {unit}", units=u)
    # the operand domain (pair-code / scale-code columns)
    for name, cname, fn in (("domain.nan_scale_a", "x.s0", lambda k: 0x7F * 128 + (k & 127)),
                            ("domain.nan_scale_b", "x.s1", lambda k: (k & ~127) + 0x7F),
                            ("domain.code16_a", "x.p5", lambda k: 16 * 16 + (k & 15))):
        u = U.copy()
        uu = 5
        u[uu, col[cname]] = fn(int(u[uu, col[cname]])) % P
        neg(name, f"unit {uu}: {cname} off the domain", units=u)
    man = json.loads((out / "manifest.json").read_text())
    man["negatives"] = {"vus": vus, "cases": cases, "variants_not_exercised": not_exercised, "variant_vus": variant_vus}
    (out / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    return man["negatives"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m gpu.nvf4.negatives")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--stmt", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--vus", type=int, default=14)
    b.add_argument("--variant-vus", type=int, default=512)
    b.add_argument("--row-picks", type=int, default=60)
    b.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    r = build(args.stmt, args.out, args.vus, args.variant_vus, args.row_picks, args.device)
    print(json.dumps({"cases": len(r["cases"]), "variants_not_exercised": r["variants_not_exercised"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
