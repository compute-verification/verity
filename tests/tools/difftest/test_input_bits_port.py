"""A declared circuit's input bits by binding (`Flock.Program.inputBits`, `program-difftest`'s `input_bits`) against its
Python mirror `program.input_bits_of`, on `tests/verity/circuits/input_bits_cases.py`'s cases: one input node of each
binding, a batch, a scan at the root, no input node, and the two refusals, each on a rebound callee no decoder gives."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
from hypothesis import strategies as st

import verity_difftest as D

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tests" / "verity" / "circuits"))
import input_bits_cases as C  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

PORT = D.Port("program-difftest", "input_bits", C.python, st.sampled_from([C.instance(n) for n in C.CASES]),
              package=REPO / "verity" / "circuits" / "difftest", need_rejects=False)
server = D.server_fixture(PORT)


@pytest.mark.parametrize("name", list(C.CASES))
def test_input_bits_agree(server, name):
    assert D.replay(PORT, C.instance(name), server)[0] == "output"
