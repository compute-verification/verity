"""RT-HOPPER's CPU witnesses on the H200 (sm_90): the machine model's ε when the cheater may use a wgmma form the census
already prices, and the data-sheet vector's cliffs in A2u and the fold.

    python forms_gap.py            RT-HOPPER-1: the measured reading with IGMMA_S8_N256_RS offered beside the honest form
    python forms_gap.py --sheet    RT-HOPPER-2/3: DS-90's ε against the fold's cost and against A2u's uniform issue cost

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


def scratch(root: Path, *, rs: bool) -> Path:
    """A copy of sm_90's data under `root`; with `rs`, a forms.toml offering the RS form and each honest naming its form."""
    d = root / "sm_90"
    shutil.copytree(SM90, d)
    if rs:
        (d / "forms.toml").write_text(FORMS)
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sheet", action="store_true", help="DS-90's cliffs in the fold's cost and in A2u")
    a = ap.parse_args()
    sheet() if a.sheet else forms_gap()
    return 0


if __name__ == "__main__":
    sys.exit(main())
