"""theory/fold_wide3.c at the full 32-bit word (fold_wide32.py) agrees with fold_opt_fx.py on a small case."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import fold_opt_fx as fx
import fold_wide32 as w32

THEORY = Path(__file__).resolve().parents[3] / "calibration" / "pouw" / "theory"


def test_fold_wide3_at_32_bits_firsts_and_one_control(tmp_path):
    """The quick tier's share of the test below: at 32 bits the kernel's first instructions are fold_opt_fx.py's, and it
    finds fold_wide32's first control."""
    exe = tmp_path / "fold_wide3"
    subprocess.run(["gcc", "-O2", "-fopenmp", "-o", str(exe), str(THEORY / "fold_wide3.c")], check=True)
    ins, _, _, _ = w32.fx1_case()
    three = {"y": ins["y"], "z": ins["z"], "a": ins["a"]}
    lab, pair = next(iter(w32.controls({**three, "w": ins["y"], "kx": ins["z"], "m1": ins["a"]}).items()))
    r = fx.width_wide3(exe, 32, three, {lab: pair})[lab]
    assert r["firsts"] == fx.new_word_sets(32, three) and r[w32.kind(lab)] > 0, r


@pytest.mark.slow
def test_fold_wide3_at_32_bits(tmp_path):
    """Over three inputs at two 32-bit points, the kernel's first instructions are fold_opt_fx.py's (by count), and it
    finds every control (w, kx and m1 read as y, z and a), fold_wide32's SEL.64 control included."""
    exe = tmp_path / "fold_wide3"
    subprocess.run(["gcc", "-O2", "-fopenmp", "-o", str(exe), str(THEORY / "fold_wide3.c")], check=True)
    ins, _, _, _ = w32.fx1_case()
    y, z, a = ins["y"], ins["z"], ins["a"]
    three = {"y": y, "z": z, "a": a}
    jobs = w32.controls({"y": y, "z": z, "w": y, "kx": z, "a": a, "m1": a})
    res = fx.width_wide3(exe, 32, three, jobs)
    assert {r["firsts"] for r in res.values()} == {fx.new_word_sets(32, three)}
    assert not [lab for lab, r in res.items() if r[w32.kind(lab)] == 0], res
