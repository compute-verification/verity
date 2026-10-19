"""theory/fold_wide3.c, FOLD-OPT's 3-instruction two-word search, agrees with fold_opt_fx.py on a small case."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

THEORY = Path(__file__).resolve().parents[1] / "theory"
sys.path.insert(0, str(THEORY))

import fold_opt_fx as fx


def test_fold_wide3_matches_fold_opt_fx(tmp_path):
    """Over three inputs at width 4 the kernel's first instructions are fold_opt_fx.py's (every form, by count), and it
    finds a pair a 3-instruction program ending on an IMAD.WIDE writes, with p, q a second-instruction word and with
    p, q before it."""
    exe = tmp_path / "fold_wide3"
    subprocess.run(["gcc", "-O2", "-fopenmp", "-o", str(exe), str(THEORY / "fold_wide3.c")], check=True)
    g = fx.lcg(0xBEEF)
    y, z = ([next(g) & 15 for _ in range(10)] for _ in range(2))
    a = [(fx.key_mul(0, 0) & 15) | 1] * 10
    v1 = [y[i] ^ z[i] ^ a[i] for i in range(10)]
    v2 = [(v1[i] * a[i] + z[i]) & 15 for i in range(10)]
    late = [fx.wide(v2[i], a[i], v1[i], z[i], True, 4) for i in range(10)]
    early = [fx.wide(v1[i], a[i], z[i], v2[i], False, 4) for i in range(10)]
    jobs = {"late": ([x[0] for x in late], [x[1] for x in late]), "early": ([x[1] for x in early], [x[0] for x in early])}
    inputs = {"y": y, "z": z, "a": a}
    res = fx.width_wide3(exe, 4, inputs, jobs)
    assert res["late"][2] == res["early"][2] == fx._new_word_sets(4, inputs) == 477
    assert res["late"][0] > 0 and res["early"][0] > 0, res
