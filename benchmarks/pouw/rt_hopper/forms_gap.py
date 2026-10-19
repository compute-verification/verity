"""RT-HOPPER's CPU witnesses on the H200 (sm_90): the machine model's ε when the cheater may use a wgmma form the census
already prices, and the data-sheet vector's cliffs in A2u and the fold.

    python forms_gap.py            RT-HOPPER-1: the measured reading with IGMMA_S8_N256_RS offered beside the honest form
    python forms_gap.py --sheet    RT-HOPPER-2/3: DS-90's ε against the fold's cost and against A2u's uniform issue cost
    python forms_gap.py --b1       RT-HOPPER-1/2: BGMMA_B1_N256 offered at split 8 (the least: t binary products separate
                                   at most 2^t values of an s8 operand) and 64 (bit planes); DS-90 prices it on issue alone
    python forms_gap.py --rs-fit   RT-HOPPER-1: RS offered at SS's joint coefficients times the H200's matched-pair ratio
    python forms_gap.py --honest-true  RT-HOPPER-8: γ with honest timed by the loop that folds the wgmma's accumulators

sm_90 has no data/forms.toml, so `eps` restricts the cheater's tensor route to the honest's form, IGMMA_S8_N256 (wgmma
m64n256k32 s8, A and B from shared memory). The census also prices IGMMA_S8_N256_RS, the same shape with A from
registers: the same fresh words and s8 depth, the same tensor line (M 0.0312468), but on the async-joint rows JI90_0 and
JI90_1 the default a lane instruction gets (0.98, 0.92) where the honest form's measured lines are 62.74 and 66.94. Each
witness copies the device's data into a scratch directory, edits only the copy and runs `machine-model eps` on it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = HERE.parent / "machine_model"
SM90 = MODEL / "data" / "devices" / "sm_90"
DESIGN = "dither-2s-licpw-redux-l32"
FORMS = """[IGMMA_S8_N256]
shape = [16, 256, 32]
split = 1
exact = "s8.s8 into s32, A and B from shared memory by descriptor"

[IGMMA_S8_N256_RS]
shape = [16, 256, 32]
split = 1
exact = "s8.s8 into s32, A from registers, B from shared memory"
"""


def b1(split: int) -> str:
    return f"""[IGMMA_S8_N256]
shape = [16, 256, 32]
split = 1
exact = "s8.s8 into s32"

[BGMMA_B1_N256]
shape = [16, 256, 256]
split = {split}
exact = "b1 and.popc: s8 operands in bit planes, {split} b1 products an s8 product"
"""


def scratch(root: Path, *, rs: bool, forms: str = FORMS) -> Path:
    """A copy of sm_90's data under `root`; with `rs`, a forms.toml (`forms`, by default the honest form and the RS form)
    and each honest naming its form."""
    d = root / "sm_90"
    shutil.copytree(SM90, d)
    if rs:
        (d / "forms.toml").write_text(forms)
        for p in (d / "honest").glob("*.toml"):
            t = p.read_text()
            p.write_text(re.sub(r'^(class = "IGMMA_S8_N256")$', r'\1\nform = "IGMMA_S8_N256"', t, flags=re.M))
    return d


def eps(data: Path, vector: str, q: float, *extra: str) -> dict:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(MODEL), os.environ.get("PYTHONPATH")]))}
    cmd = [sys.executable, "-m", "verity_machine_model.cli", "--data", str(data), "eps", DESIGN, "--credit", "floor",
           "--operands", "free", "--vector", vector, "--q", str(q), "--json", *extra]
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def line(label: str, d: dict) -> str:
    c = d["cheat"]
    forms = {k: round(v, 4) for k, v in (c.get("forms") or {}).items()}
    words = {k: round(v, 4) for k, v in c["words"].items() if v}
    return (f"{label:<34} core {d['core'] * 100:7.3f}%  ε_machine {d['terms']['machine'] * 100:7.3f}%  "
            f"cheater {c['cycles_per_mma']:7.2f} SMSP-cycles a MMA on {','.join(c['binding'])}  forms {forms}  words {words}")


def forms_gap() -> None:
    with tempfile.TemporaryDirectory() as t:
        base, rs = scratch(Path(t) / "base", rs=False), scratch(Path(t) / "rs", rs=True)
        for vector in ("DS-90", "DC-90", "DC-90-ri"):
            for q in (1.22, 0.5):
                print(line(f"{vector} q*={q} honest form only", eps(base, vector, q)))
                print(line(f"{vector} q*={q} + IGMMA_S8_N256_RS", eps(rs, vector, q)))


def set_uni_issue(data: Path, cost: float) -> None:
    p = data / "census.csv"
    p.write_text(re.sub(r"^UNI,DS_issue,,,[^,]*,", f"UNI,DS_issue,,,{cost},", p.read_text(), flags=re.M))


def sheet() -> None:
    with tempfile.TemporaryDirectory() as t:
        d = scratch(Path(t), rs=False)
        for r in (1.0546875, 0.95, 0.93, 0.92, 0.91, 0.9, 0.75):
            print(line(f"DS-90 q*=0.5 fold {r}", eps(d, "DS-90", 0.5, "--fold-cost", str(r))))
        for r in (0.9, 0.75):
            print(line(f"DS-90 q*=1.22 fold {r}", eps(d, "DS-90", 1.22, "--fold-cost", str(r))))
        for c in (1, 0.05, 0.031, 0.025, 0.02, 0.0001862, 0):
            set_uni_issue(d, c)
            print(line(f"DS-90 q*=0.5 UNI on DS_issue {c}", eps(d, "DS-90", 0.5)))
        v = d / "vectors" / "DS-90.toml"
        v.write_text(v.read_text().replace('"DS_shfl"]', '"DS_shfl", "UP"]'))
        for qu in (0.5, 8):
            print(line(f"DS-90+UP q*=0.5 UNI 0 q*_U {qu}", eps(d, "DS-90", 0.5, "--q-uniform", str(qu))))


SS_JI = {"JI90_0": 62.7407638, "JI90_1": 66.93849, "JI90_2": 121.566}
# RS's coefficient over SS's on each joint row, the least over the matched pairs of r20261019-055823-616f (rs.sh at
# d6e2f80cf0): the LOP3-only mix beside the wgmma, where RS runs 1.2% faster than SS. Beside the honest's LI fold RS
# runs 1.9% slower (ratios 1.032, 1.030, 1.019).
RS_FIT = {"JI90_0": 0.9666, "JI90_1": 0.9686, "JI90_2": 0.9868}


def set_line(data: Path, kind: str, row: str, u: float) -> None:
    p = data / "census.csv"
    t, n = re.subn(rf"^{kind},{row},,,[^,]*,", f"{kind},{row},,,{u},", p.read_text(), flags=re.M)
    assert n == 1, (kind, row, n)
    p.write_text(t)


def rs_fit() -> None:
    with tempfile.TemporaryDirectory() as t:
        for scale in (1.0, None, 0.95, 0.9):
            d = scratch(Path(t) / f"fit-{scale}", rs=True)
            for row, u in SS_JI.items():
                set_line(d, "IGMMA_S8_N256_RS", row, u * (RS_FIT[row] if scale is None else scale))
            label = "the run's matched-pair ratios" if scale is None else f"{scale}× SS's"
            for vector, q in (("DC-90", 1.22), ("DC-90-ri", 1.22), ("DC-90-ri", 0.5)):
                print(line(f"{vector} q*={q} RS at {label}", eps(d, vector, q)))


TRUE_LOOP = 290.36   # wgli_true_w2 in r20261019-055823-616f: the plain GEMM's 122.41 over its tensor fraction 0.42158


def honest_true() -> None:
    """γ and ε when honest's measured time is the loop that folds the wgmma's own accumulators (RT-HOPPER-8)."""
    with tempfile.TemporaryDirectory() as t:
        d = scratch(Path(t), rs=False)
        for cyc in (None, TRUE_LOOP):
            if cyc:
                p = d / "honest" / "dc90-licpw-redux-l32.toml"
                p.write_text(re.sub(r"^cycles_per_mma = [0-9.]+", f"cycles_per_mma = {cyc}", p.read_text(), flags=re.M))
            for vector, q in (("DS-90", 0.5), ("DC-90-ri", 1.22)):
                r = eps(d, vector, q)
                print(line(f"{vector} q*={q} honest {cyc or 204.25}", r) + f"  γ {r['gamma']:.4f}  honest {r['honest_cycles_per_mma']:.2f} ({r['honest_time']})")


def b1_gap() -> None:
    with tempfile.TemporaryDirectory() as t:
        for split in (8, 64):
            d = scratch(Path(t) / f"b1-{split}", rs=True, forms=b1(split))
            for r in (None, 0.9, 0.75, 0.5):
                fold = ("--fold-cost", str(r)) if r else ()
                print(line(f"DS-90 q*=0.5 +b1 split {split} fold {r or 'today'}", eps(d, "DS-90", 0.5, *fold)))
            print(line(f"DC-90-ri q*=1.22 +b1 split {split}", eps(d, "DC-90-ri", 1.22)))
        d = scratch(Path(t) / "base", rs=False)
        for r in (0.75, 0.5):
            print(line(f"DS-90 q*=0.5 honest form only fold {r}", eps(d, "DS-90", 0.5, "--fold-cost", str(r))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sheet", action="store_true", help="DS-90's cliffs in the fold's cost and in A2u")
    ap.add_argument("--b1", action="store_true", help="ε with BGMMA_B1_N256, which DS-90 prices on issue alone, offered")
    ap.add_argument("--rs-fit", action="store_true", help="ε with RS offered at SS's joint coefficients times the H200's ratio")
    ap.add_argument("--honest-true", action="store_true", help="γ with honest timed by the loop that folds the accumulators")
    a = ap.parse_args()
    (sheet() if a.sheet else b1_gap() if a.b1 else rs_fit() if a.rs_fit else honest_true() if a.honest_true
     else forms_gap())
    return 0


if __name__ == "__main__":
    sys.exit(main())
