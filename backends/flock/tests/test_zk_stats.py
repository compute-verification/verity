"""The M1 statistics (``zk_stats.py``) on synthetic dumps: uniform real and simulated transcripts pass, and a real
transcript whose masked values do not depend on the prover's randomness fails. The dumps from ``flock-circuit zkstat`` run
on a pod or VM; this checks the analysis itself."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")
sys.path.insert(0, os.path.dirname(__file__))
import zk_stats  # noqa: E402

SIZES = {"exposed": 60, "s_ab": 16, "s_c": 16, "inner_scalars": 3, "inner_y": 40, "inner_open_h": 20, "inner_open_mu": 20,
         "lig_exposed": 10, "lig_tprime_ybar": 20, "lig_rows0": 20, "lig_deep": 20}
#: positions of lig_deep that are public structure (always 0 in every kind)
ZERO_AT = {"lig_deep": [3, 7]}


def _row(kind, run, rep, rng, fixed=None):
    row = {"kind": kind, "run": run, "rep": rep}
    for cls, n in SIZES.items():
        if fixed is not None and cls in fixed:
            row[cls] = fixed[cls]
        else:
            row[cls] = [rng.integers(0, 256, 16, dtype=np.uint8).tobytes().hex() for _ in range(n)]
        for i in ZERO_AT.get(cls, []):
            row[cls][i] = bytes(16).hex()
    return row


def _dump(n, leak=False):
    rng = np.random.default_rng(7)
    const = {cls: [bytes([i % 7] * 16).hex() for i in range(SIZES[cls])] for cls in zk_stats.MASKED}
    lines = []
    for run in range(n):
        for rep in range(2):
            lines.append("ZKREAL\t" + json.dumps(_row("real", run, rep, rng, const if leak else None)))
            lines.append("ZKSIM\t" + json.dumps(_row("sim", run, rep, rng)))
            lines.append("ZKCONTROL\t" + json.dumps(_row("control", run, rep, rng, const)))
    return lines


def test_uniform_transcripts_pass_and_the_control_is_distinguished():
    res = zk_stats.analyse(zk_stats.load(_dump(24)))
    assert res["pass"], json.dumps(res, indent=1)
    assert all(res["classes"][c]["control_distinguished"] for c in zk_stats.MASKED)
    assert res["classes"]["lig_deep"]["structural_constants"] == 2


def test_a_witness_dependent_real_transcript_fails():
    res = zk_stats.analyse(zk_stats.load(_dump(24, leak=True)))
    assert not res["pass"]
    assert not res["classes"]["exposed"]["real_indistinguishable_from_sim"]
