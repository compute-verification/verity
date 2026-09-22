"""Undefined names in the adapter are a merge-time defect, not a runtime one.

The TP merge (1bd8af4) renamed `prod_tp` to `tp` in `replay.py` on one side of a conflict and kept the use on the
other; nothing on the laptop runs a replay, so every replay on the head then failed writing `replay.json`, and two
lanes (INT3, QZ) each found and fixed it on their pods. `ruff --select F821` on that file reports exactly that name.
This test runs the same check over the code that only executes on a GPU host, so the full suite a merge is checked
with fails on the laptop instead.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]  # vllm-poc/
TARGETS = ["verity_vllm_adapter", "verity_vllm", "e2e"]


def test_no_undefined_names_in_gpu_only_code():
    probe = subprocess.run([sys.executable, "-m", "ruff", "--version"], capture_output=True, text=True)
    if probe.returncode != 0:
        pytest.skip("ruff is not installed in this interpreter")
    r = subprocess.run([sys.executable, "-m", "ruff", "check", "--select", "F821,F823", "--no-cache", "--output-format", "concise", *TARGETS],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
