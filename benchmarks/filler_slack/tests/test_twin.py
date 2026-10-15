"""The filler's twin (`twin.py`), its gate (`gate.py`) and the frontier arithmetic (`analyze.py`).

`vectors.txt` holds gate lines the sm_120 filler printed (run r20261015-075411-2497): the twin must reproduce them word
for word, so a twin change that drifts from the kernel fails here rather than on a node.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import analyze  # noqa: E402
import twin  # noqa: E402


def vector_lines() -> list[str]:
    return [ln for ln in (HERE / "vectors.txt").read_text().splitlines() if ln.startswith("G ")]


def test_fragment_layouts_are_bijections():
    assert sorted(twin.a_pos(l, i) for l in range(32) for i in range(16)) == [(r, c) for r in range(16) for c in range(32)]
    assert sorted(twin.b_pos(l, i) for l in range(32) for i in range(8)) == [(k, n) for k in range(32) for n in range(8)]
    assert sorted(twin.c_pos(l, r) for l in range(32) for r in range(4)) == [(m, n) for m in range(16) for n in range(8)]


def test_closed_form_matches_iterated_accumulation():
    seed, kseed, iters, ftid = 7, 99, 5, 37
    P = twin.warp_products(seed, ftid // 32)
    lane = ftid % 32
    acc = [[0] * 4 for _ in range(twin.FT)]
    st = [(twin.gen(seed, ftid, j, 100), twin.gen(seed, ftid, j, 101)) for j in range(twin.FT)]
    for _ in range(iters):
        for j in range(twin.FT):
            acc[j] = [(acc[j][r] + P[j & 1][twin.c_pos(lane, r)[0]][twin.c_pos(lane, r)[1]]) & twin.M32 for r in range(4)]
            st[j] = twin.duo(st[j], acc[j], 4 * j, kseed)
    assert twin.thread_words(seed, kseed, iters, ftid) == [twin.finish(s) for s in st] + [w for a in acc for w in a]


def test_receipts_depend_on_salt_and_key():
    w = twin.thread_words(7, 99, 3, 0)
    assert twin.thread_words(8, 99, 3, 0)[:4] != w[:4]
    assert twin.thread_words(7, 98, 3, 0)[:4] != w[:4]
    assert twin.thread_words(7, 98, 3, 0)[4:] == w[4:]


def test_twin_reproduces_recorded_gpu_lines():
    lines = vector_lines()
    assert len(lines) >= 8
    for ln in lines:
        f = ln.split()
        seed, kseed, iters, ftid = int(f[2]), int(f[3]), int(f[4]), int(f[5])
        assert twin.thread_words(seed, kseed, iters, ftid) == [int(w, 16) for w in f[6:]], ln


def run_gate(text: str) -> tuple[int, dict]:
    p = subprocess.run([sys.executable, str(HERE.parent / "gate.py")], input=text, capture_output=True, text=True)
    return p.returncode, json.loads(p.stdout)


def test_gate_passes_recorded_lines():
    rc, rep = run_gate("\n".join(vector_lines()) + "\n")
    assert rc == 0 and rep["ok"] and rep["kernels"]["base-u1"]["bad"] == 0


@pytest.mark.parametrize("word", [0, 3, 4, 19])
def test_gate_catches_one_flipped_word(word):
    lines = vector_lines()
    f = lines[1].split()
    f[6 + word] = f"{int(f[6 + word], 16) ^ 1:08x}"
    rc, rep = run_gate("\n".join([lines[0], " ".join(f)] + lines[2:]) + "\n")
    k = rep["kernels"]["base-u1"]
    assert rc == 1 and not rep["ok"] and k["bad"] == 1 and k["first"]["words"] == [word]


def test_gate_fails_on_no_lines():
    rc, rep = run_gate("no gate lines here\n")
    assert rc == 1 and not rep["ok"]


POINTS = [(0, 0.001, 1.0), (0, 0.002, 0.9995), (0, 0.004, 0.996), (0, 0.008, 0.99), (0, 0.016, 0.98),
          (1, 0.001, 1.0), (1, 0.002, 0.998), (1, 0.004, 0.9992), (1, 0.008, 0.97)]


def test_frontier_takes_largest_passing_point_per_rep():
    assert analyze.frontier(POINTS, 0.001) == [0.002, 0.004]
    assert analyze.frontier(POINTS, 0.005) == [0.004, 0.004]
    assert analyze.frontier(POINTS, 0.0001) == [0.001, 0.001]
    assert analyze.frontier([(0, 0.001, 0.5)], 0.01) == [0.0]


def test_bracket_spans_largest_pass_and_next_failure():
    b = analyze.bracket(POINTS, 0.001)
    assert b["lo"]["min"] == 0.002 and b["lo"]["max"] == 0.004
    assert b["hi"]["min"] == 0.004 and b["hi"]["max"] == 0.008
    top = analyze.bracket(POINTS, 0.5)
    assert top["hi"]["n"] == 0 and top["lo"]["max"] == 0.016 and top["interp"]["n"] == 0


def test_bracket_interpolates_the_loss_line():
    b = analyze.bracket([(0, 0.002, 0.9995), (0, 0.004, 0.9975), (1, 0.001, 1.0), (1, 0.003, 0.996)], 0.001)
    assert b["interp"]["min"] == pytest.approx(0.0015) and b["interp"]["max"] == pytest.approx(0.0025)


def test_probe_overhead_reads_the_probe_body():
    """The body is the smallest sleeping loop that holds the PB ops: not the span through the filler's loop, nor the
    probe-first kernel's 4-instruction start-flag wait (run r20261015-080548-a359's sass-loops.json)."""
    wait = {"n": 4, "ops": {"NANOSLEEP": 1, "LDG.E.STRONG.SYS": 1, "ISETP.NE.AND": 1, "BRA": 1}}
    sass = {"3,0,0,1,16": {"loops": [{"n": 40, "ops": {"NANOSLEEP": 1, "BRA": 1}}, {"n": 19, "ops": {"NANOSLEEP": 1}}]},
            "pk:1,16": {"loops": [wait, {"n": 23, "ops": {"IADD3": 16, "NANOSLEEP": 1}}]},
            "pk:6,32": {"loops": [wait, {"n": 72, "ops": {"IMMA.16832.S8.S8": 32, "NANOSLEEP": 1}}]},
            "0,0,0,1,1": {"loops": [{"n": 30, "ops": {"IMMA.16832.S8.S8": 4}}]}}
    assert analyze.probe_overhead(sass) == {"p-imad-b16": 19 / 16, "s-iadd3-b16": 23 / 16, "s-imma-b32": 72 / 32}


def test_ops_follow_filler_enum():
    """analyze.OPS names filler.cu's Op values in order (sass_loops keys carry the numbers, run rows the names)."""
    src = (HERE.parent / "filler.cu").read_text()
    names = re.search(r"OP_NAME\[NOPS\] = \{(.*?)\};", src, re.S).group(1)
    assert tuple(re.findall(r'"(\w+)"', names)) == analyze.OPS


def test_sass_loops_counts_register_reads(tmp_path):
    """A loop's vector reads skip RZ, the destination and reuse-cache hits; UR and c[..][..] sources count apart; a
    backward `BRA.U` closes a loop (the filler's own loop)."""
    body = [
        "IADD3 R9, PT, PT, R22.reuse, R9, RZ", "IADD3 R8, PT, PT, R22.reuse, R8, RZ", "IADD3 R7, PT, PT, R22, R7, RZ",
        "IADD3 R6, PT, PT, R6, UR4, RZ", "IADD3 R5, PT, PT, R5, c[0x3][0x10], RZ", "BRA.U UP0, 0x0000",
    ]
    lines = ["Function : _Z2pkILi8ELi16EE"] + [f"/*{16 * i:04x}*/ {t} ;" for i, t in enumerate(body)]
    (tmp_path / "f.sass").write_text("\n".join(lines) + "\n")
    out = json.loads(subprocess.run([sys.executable, str(HERE.parent / "sass_loops.py"), str(tmp_path / "f.sass")],
                                    check=True, capture_output=True, text=True).stdout)
    (lp,) = out["pk:8,16"]["loops"]
    assert (lp["n"], lp["vreads"], lp["reuse_hits"], lp["ureads"], lp["creads"]) == (6, 6, 2, 1, 1)
    assert lp["shapes"]["IADD3 R, PT, PT, R.reuse, R, RZ"] == 2 and lp["shapes"]["IADD3 R, PT, PT, R, c[0x3][.], RZ"] == 1
