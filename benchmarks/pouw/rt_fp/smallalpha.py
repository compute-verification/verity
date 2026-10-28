"""A Dither-2S word on the FP forms when the committed data keeps every operand on the FP grid.

The machine model's FP forms (data/forms.toml) take `split` passes an s8 product, a bound for operands that range over
all 256 s8 values (OMMASF16864 3, QMMA16832 3). Dither-2S puts one fresh {0, 1} coin on each operand and nothing
wider, so on data whose dithered floors are 0, 1 or 2 every operand is 0..3: an E2M1 and an E4M3 value, products at
most 9, a 32-step word at most 288, exact in FP32. A cheater who commits such data writes every word on FP4 at split
1. The model converts an FP accumulator to s32 for free (forms.toml: "the cheater's favour"); fx1's fold XORs the word
right after its keyed IMAD, so the cheater pays the conversion, about one ALU a word. At L = 32 every gain is the k64
form's (OMMASF16864, kind::mxf4nvf4), whose one output a cell holds two 32-deep words only packed (the second block at
scale 2^10: 288·1025 < 2^24, exact), so a mask and a shift a pair cost it one ALU a word even with the conversion free.

It prices, at the floor credit with operands free:
  sm_120 (data/): Dither-2S, fpx_design's fpx-g1s-fx1 (dither-2s with one more IMAD a word of fold, as fpx_design
    31899b9c10's fpx_eps.py writes it) and the record dither-c-peel-fx1 (H1 + fx1: its co32c and cohalf words carry
    H1's U, 32 values, which keeps them off split 1 (`h1_grid.py`), but its P·V is Dither-2S; the model gives every
    word of a design one set of forms, so `fp1` lets all of the record's words take them), as data/ has them (`base`),
    with the FP forms at split 1 (`fp1`), and at split 1 paying 4 ALU warp instructions an MMA (one a word; `fp1c`,
    charged only on the lines a form has of its own, which undercharges it);
  sm_100 (data/devices/sm_100/): the record dither-2s-licpw-redux-heavy-l32 with honest's s8 form alone (`base`) and
    beside kind::mxf4nvf4 at k64 (`fp4`), priced an instruction as the s8 form (the sheet's FP4 at twice s8).
H200 has no FP4 kind, and its FP8 kind runs at the s8 rate, so it isn't priced.

Run: `uv run --project benchmarks/pouw/machine_model python benchmarks/pouw/rt_fp/smallalpha.py OUT`. It builds the
data copies in a temporary directory and writes only OUT/results.jsonl and OUT/table.txt, one line a reading.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MM = Path(__file__).resolve().parents[1] / "machine_model"
DATA = MM / "data"
FP = ("QMMA16832", "QMMASF16832", "QMMASP16864", "OMMASF16864")
CONV = 4  # one ALU a word: an m16n8 form writes 128 words, 4 warp instructions
G1S_FX1 = """base = "dither-2s"
name = "fpx-g1s-fx1"
title = "g1 on int8 units (RTX PRO 6000): Dither-2S on R + b with fx1's keyed IMAD a word"
summary = "FPX-DESIGN 31899b9c10's fpx-g1s-fx1, as its fpx_eps.py writes it"
plain = 0.0

[fold]
name = "duo+x+IMAD1"
ALU = 2
IMAD = 2.0
"""
FP4_FORMS = """[UTCIMMA_M128_N256]
shape = [128, 256, 32]
split = 1
exact = "s8 into s32"

[UTCOMMA_M128_N256]
shape = [128, 256, 64]
split = 1
exact = "operands in 0..3 are E2M1 values; products <= 9 and a 32-step word <= 288 are exact in FP32"
"""


def set_splits(d: Path, split: int) -> None:
    t = (d / "forms.toml").read_text()
    for kind in FP:
        t, n = re.subn(rf"(\[{kind}\]\nshape = [^\n]*\n(?:sparse = [^\n]*\n)?split = )\d+", rf"\g<1>{split}", t)
        assert n == 1, kind
    (d / "forms.toml").write_text(t)


def surcharge(d: Path) -> None:
    """Each FP form pays CONV ALU warp instructions an MMA on every cycles line it has of its own: ALU's coefficient
    on that line's row and epoch, else the generic vector-pipe @V's."""
    cyc = lambda f: f is not None and f[4] != "" and f[2] == "" and f[5] == ""  # noqa: E731
    for csv in (d / "census").glob("*.csv"):
        lines = csv.read_text().splitlines()
        head, body = lines[0], [x.split(",") for x in lines[1:] if x]
        at: dict[tuple, dict] = {}
        for f in body:
            at.setdefault((f[1], f[6], f[7]), {})[f[0]] = f
        out = [f for f in body if f[0] not in FP]
        for kinds in at.values():
            alu = kinds.get("ALU") or kinds.get("@V")
            for form in FP:
                own = kinds.get(form)
                if own is None:
                    continue
                if cyc(alu) and cyc(own):
                    own = [*own[:4], repr(float(own[4]) + CONV * float(alu[4])), *own[5:]]
                out.append(own)
        csv.write_text("\n".join([head] + [",".join(f) for f in out]) + "\n")


def sm120(out: Path) -> list[Path]:
    dirs = []
    for tag in ("base", "fp1", "fp1c"):
        d = out / "sm_120" / tag
        shutil.copytree(DATA, d, ignore=shutil.ignore_patterns("devices"))
        (d / "designs" / "fpx-g1s-fx1.toml").write_text(G1S_FX1)
        if tag != "base":
            set_splits(d, 1)
        if tag == "fp1c":
            surcharge(d)
        dirs.append(d)
    return dirs


def sm100(out: Path) -> list[Path]:
    dirs = []
    for tag in ("base", "fp4"):
        d = out / "sm_100" / tag
        shutil.copytree(DATA / "devices" / "sm_100", d)
        if tag == "fp4":
            (d / "forms.toml").write_text(FP4_FORMS)
            census = d / "census.csv"
            imma = [x for x in census.read_text().splitlines() if x.startswith("UTCIMMA_M128_N256,")]
            fp4 = [x.replace("UTCIMMA_M128_N256,", "UTCOMMA_M128_N256,", 1) for x in imma]
            census.write_text(census.read_text() + "\n".join(fp4) + "\n")
            for p in (d / "honest").glob("*.toml"):
                t = p.read_text()
                p.write_text(re.sub(r'^(class = "([A-Z0-9_]+)")$', r'\1\nform = "\2"', t, count=1, flags=re.M))
        dirs.append(d)
    return dirs


def eps(data: Path, design: str, vector: str, q: float) -> dict:
    cmd = [sys.executable, "-m", "verity_machine_model", "--data", str(data), "eps", design, "--vector", vector,
           "--credit", "floor", "--operands", "free", "--q", str(q), "--json"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=MM, env={**os.environ, "PYTHONPATH": str(MM)},
                       check=True)
    j = json.loads(r.stdout)
    return {"core": j["core"], "forms": j["cheat"]["forms"], "binding": j["cheat"]["binding"]}


def main(out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp())
    try:
        runs = [(d, design, v) for d in sm120(scratch) for design in ("dither-2s", "fpx-g1s-fx1", "dither-c-peel-fx1")
                for v in ("DS-120", "VF3r-16")]
        runs += [(d, "dither-2s-licpw-redux-heavy-l32", v) for d in sm100(scratch) for v in ("DS-100", "DC-100-ri")]
        rows, table = [], []
        for d, design, vector in runs:
            for q in (0.5, 1.22):
                e = eps(d, design, vector, q)
                row = {"device": d.parent.name, "data": d.name, "design": design, "vector": vector, "q": q, **e}
                rows.append(row)
                forms = ", ".join(f"{k} {v:.3f}" for k, v in e["forms"].items())
                table.append(f"{row['device']:6} {d.name:4} {design:32} {vector:9} q*={q:<4} core "
                             f"{100 * e['core']:7.3f}%  {forms}  [{', '.join(e['binding'])}]")
                print(table[-1])
    finally:
        shutil.rmtree(scratch)
    (out / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "table.txt").write_text("\n".join(table) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
