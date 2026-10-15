"""`trails.c` against `trails.py`'s model and against duo itself: its arithmetic self-test, the duo trail both find with
the key secret (the rotation pair at t = 12), that trail measured on duo as the 2^-4 differential its four sign
patterns make, and duo2's joint search excluding what one of its lanes alone allows."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import trails as T  # noqa: E402

PAIR = [(12, 1 << 31), (13, (1 << 31) | (1 << 27)), (14, 1 << 31)]


@pytest.fixture(scope="module")
def binary(tmp_path_factory) -> str:
    cc = shutil.which("gcc") or shutil.which("cc")
    if cc is None:
        pytest.skip("no C compiler")
    out = tmp_path_factory.mktemp("trails") / "trails"
    subprocess.run([cc, "-O2", "-o", str(out), str(HERE / "trails.c")], check=True)
    return str(out)


def run(binary: str, *args, kind: str = "duo") -> dict:
    sched = "\n".join(T.schedule_lines(kind, 0xC0FFEE11)) + "\n"
    return json.loads(subprocess.run([binary, *map(str, args)], input=sched, capture_output=True, text=True,
                                     check=True).stdout)


def test_selftest(binary: str) -> None:
    out = subprocess.run([binary, "selftest"], capture_output=True, text=True, check=True).stdout
    assert json.loads(out)["selftest"]


def test_the_rotation_pair_with_the_key_secret(binary: str) -> None:
    d = run(binary, "secret", 1, 7, 10, 14)
    tr = d["trail"]
    assert d["complete"] and tr["cost"] == 6 and tr["end"] == "collision"
    assert [(s["t"], s["dp"]) for s in tr["steps"]] == PAIR
    py, _ = T.search(T.Schedule(), [12], 7, "secret")
    assert py["cost"] == 6 and T.word_diffs(py) == PAIR
    m = T.simulate_lanes("duo", 0xC0FFEE11, PAIR, 1 << 14, 1, True, "collision")
    assert 0.04 < m["hits"] / m["samples"] < 0.085


def test_duo2_lanes_take_one_word_difference(binary: str) -> None:
    assert run(binary, "secret", 1, 8, 0, 60, kind="duo2-lane1")["trail"] is not None
    d = run(binary, "secret", 2, 9, 0, 60, kind="duo2")
    assert d["complete"] and d["trail"] is None
    m = T.simulate_lanes("duo2", 0xC0FFEE11, PAIR, 1 << 14, 1, True, "collision")
    assert m["hits"] == 0
