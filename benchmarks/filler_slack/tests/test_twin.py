"""The filler's twin (`twin.py`), its gate (`gate.py`) and the frontier arithmetic (`analyze.py`).

`vectors.txt` holds gate lines the sm_120 filler printed (run r20261015-075411-2497): the twin must reproduce them word
for word, so a twin change that drifts from the kernel fails here rather than on a node.
"""

from __future__ import annotations

import json
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
    assert top["hi"]["n"] == 0 and top["lo"]["max"] == 0.016


def test_probe_overhead_reads_the_sleep_loop():
    sass = {"3,0,0,1,16": {"loops": [{"n": 40, "ops": ["NANOSLEEP", "BRA"]}, {"n": 19, "ops": ["NANOSLEEP"]}]},
            "pk:1,16": {"loops": [{"n": 20, "ops": ["NANOSLEEP"]}]},
            "0,0,0,1,1": {"loops": [{"n": 30, "ops": ["IMMA"]}]}}
    assert analyze.probe_overhead(sass) == {"p-imad-b16": 19 / 16, "s-iadd3-b16": 20 / 16}
