"""theory/fold_wide3.c, FOLD-OPT's two-word search, agrees with fold_opt_fx.py on a small case."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

THEORY = Path(__file__).resolve().parents[1] / "theory"
sys.path.insert(0, str(THEORY))

import fold_opt_fx as fx


def test_fold_wide3_matches_fold_opt_fx(tmp_path):
    """Over three inputs at width 4 the kernel's first instructions are fold_opt_fx.py's (every form, by count), and it
    finds every control of `wide3_controls` (w, kx and m1 read as y, z and a): each two-word form as a first and as a
    last instruction, each way a last one is reached, a SEL.64 no IMNMX.64 is, and a word of a second instruction."""
    exe = tmp_path / "fold_wide3"
    subprocess.run(["gcc", "-O2", "-fopenmp", "-o", str(exe), str(THEORY / "fold_wide3.c")], check=True)
    g = fx.lcg(0xBEEF)
    y, z = ([next(g) & 15 for _ in range(10)] for _ in range(2))
    a = [(fx.key_mul(0, 0) & 15) | 1] * 10
    inputs = {"y": y, "z": z, "a": a}
    jobs = fx.wide3_controls(4, {"y": y, "z": z, "w": y, "kx": z, "a": a, "m1": a})
    assert "sel_free" in jobs
    res = fx.width_wide3(exe, 4, inputs, jobs)
    assert {r["firsts"] for r in res.values()} == {fx._new_word_sets(4, inputs)} == {663}
    count = {"ctl": "found", "sel": "sel", "le2": "le2"}
    assert not [lab for lab, r in res.items() if r[count[lab.split("_")[0]]] == 0], res
