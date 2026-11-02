#!/usr/bin/env python3
"""Theorem C's per-opcode creation cap κ, from each device's census: an issue slot that writes d words per lane creates at
most κ = 65·d incidences.

The cap per word is Lean's: `Pouw.SecurityProofs.Eps1.ClassPSlots` (Security/Proofs/Pouw/Eps1/ClassP.lean,
`created_le_65`): a class-P tuple word creates only incidences of its fields' own entries, a field's range has at least
128|E| + 1 values, and `Π_k (128 e_k + 1) ≤ 2^32` with `e_k ≤ 32` caps `Σ_k e_k` at 65 (two 32-entry fields and a
one-entry field). The guarantee charges a slot of d words `65·d`, so a word costs at least `1 + (32 − w)/κ` slots when a
writer creates at most w (the q* analysis takes 4; a writer that takes two segments' fields from one tuple creates 8).

Each census kind the current epoch holds is classified: a CUDA lane kind (registers from registers, the vector route),
a register load (the table route: its address may be data), a tensor kind (MMA forms and tensor-memory loads, the
writer route, whose words the count doesn't charge), uniform, or a kind with no destination register (stores,
reductions, async copies). A kind no rule names fails the run, so a new census kind gets a destination width before it
gets a κ.

  uv run python benchmarks/pouw/theory/kappa_count.py [--device sm_NN ...] [--json PATH]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from fractions import Fraction

from verity_machine_model.model import census, device, devices, model

WORD_CAP = 65

LANE = {"ALU": 1, "IMAD": 1, "FFMA": 1, "HFMA2": 1, "VIMNMX": 1, "F2FS": 1, "XU": 1, "SHFL": 1, "NOROW": 1, "SEL": 1,
        "IDP4A": 1, "LEA": 1, "FLO": 1, "VIADD": 1,
        "ALU64": 2, "IMADW": 2, "DMUL": 2, "W2": 2, "MOV64": 2, "CVT64": 2,
        # a predicate, charged a whole word
        "ISETP": 1,
        # the DC measured readings' catch-all (A2's): an instruction no class names, one word as NOROW on sm_120
        "UNLISTED": 1}
# REDUX writes a uniform register, ahead of NO_DEST's RED (a reduction into memory)
UNIFORM = {"UNI": 1, "UNI64": 2, "REDUX": 1}
LOADS = {"LDG32": 1, "LDG64": 2, "LDG128": 4, "LDG128CG": 4, "LDG128NA": 4, "LDG256": 8, "LDG256CG": 8,
         "LDS32": 1, "LDS64": 2, "LDS128": 4, "LDC64": 2, "LDCUNI": 1, "LDCDIV": 1,
         "TLD1": 1, "TLD2": 2, "TLD2D1": 1, "TLD2D4": 4, "TLD4w": 4, "TLD4G": 4,
         "TEX1": 1, "TEX2": 2, "TEX4": 4, "TEXBIL": 4, "SULD1": 1, "SULD4": 4,
         "ATOMG32": 1, "ATOMG64": 2, "ATOMS32": 1, "ATOMS64": 2, "ATOME32": 1, "ATOME32P": 1, "ATOME64": 2}
# ldmatrix: .m8n8 and .m8n16 shapes write n registers a lane at .xn; .m16n16 (the sub-byte and 8-bit `T` forms) write 2n
LDSM = re.compile(r"LDSM(?P<sub>[468]?)(?P<t>T?)_X(?P<n>\d+)(?:T)?")
TENSOR = re.compile(r".*MMA.*|LDTM_.*")
# TMA box loads (UTMA*, TMA512) and gather4 (G4_*) write shared memory, never a register
NO_DEST = re.compile(r"(ST|RED|SURED|TMA|UTMA|G4_|UBLKCP|DSMST|LDGSTS).*")


def classify(kind: str) -> tuple[str, int]:
    """(class, destination 32-bit words a lane) of a census kind; raises on a kind no rule names."""
    if kind in LANE:
        return "lane", LANE[kind]
    if kind in UNIFORM:
        return "uniform", UNIFORM[kind]
    if kind in LOADS:
        return "load", LOADS[kind]
    if m := LDSM.fullmatch(kind):
        n = int(m["n"])
        return "load", 2 * n if m["sub"] and m["t"] else n
    if TENSOR.fullmatch(kind):
        return "tensor", 0
    if NO_DEST.fullmatch(kind):
        return "none", 0
    raise SystemExit(f"census kind {kind!r} has no destination width: add it to kappa_count.py's tables")


def floor(kappa: int, w: int) -> Fraction:
    """`ClassPSlots`' issue floor a word: `1 + (32 − w)/κ`."""
    return 1 + Fraction(32 - w, kappa)


def count(name: str) -> dict:
    data = device(name)
    m = model(data)
    kinds = sorted(census(m, m.current, data))
    table = {k: dict(zip(("class", "words"), classify(k))) for k in kinds}
    for v in table.values():
        v["kappa"] = WORD_CAP * v["words"] if v["class"] in ("lane", "load", "uniform") else None
    vec = [k for k in m.cuda if k in table]
    missing = [k for k in m.cuda if k not in table]
    if missing:
        raise SystemExit(f"{name}: model.toml's lane classes {missing} have no census line at {m.current}")
    out = {"device": name, "epoch": m.current, "kinds": table,
           "kappa_vector": max(table[k]["kappa"] for k in vec),
           "kappa_lane": max((v["kappa"] for v in table.values() if v["class"] == "lane"), default=None),
           "kappa_table": max((v["kappa"] for v in table.values() if v["class"] == "load"), default=None)}
    out["floors"] = {key: {f"w{w}": float(floor(out[key], w)) for w in (4, 8)}
                     for key in ("kappa_vector", "kappa_lane", "kappa_table") if out[key]}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--device", action="append", help="sm_NN (default: every priced device)")
    ap.add_argument("--json", help="write the counts here")
    a = ap.parse_args(argv)
    res = [count(d) for d in (a.device or devices())]
    for r in res:
        print(f"{r['device']} ({r['epoch']}): κ = {WORD_CAP}·d")
        for k, v in r["kinds"].items():
            if v["kappa"] is not None:
                print(f"  {k:12s} {v['class']:8s} d = {v['words']}  κ = {v['kappa']}")
        for key, fl in r["floors"].items():
            print(f"  {key} = {r[key]}: floor {fl['w4']:.4f} at w = 4, {fl['w8']:.4f} at w = 8")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
