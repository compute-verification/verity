"""The C-Flock ZK transcript statistics (``zk_stats.py``) on synthetic dumps: uniform real and simulated transcripts
pass, and a real transcript whose masked values do not depend on the prover's randomness fails. The dumps from
``flock-circuit zkstat`` run on a pod or VM; this checks the analysis itself."""
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


REL_SIZES = {"rel_level0": 40, "rel_level1": 12}


def _rel(kind, run, rng, fixed=False):
    row = {"kind": kind, "run": run, "rep": 0}
    for cls, n in REL_SIZES.items():
        row[cls] = [bytes([i % 5] * 16).hex() for i in range(n)] if fixed else \
            [rng.integers(0, 256, 16, dtype=np.uint8).tobytes().hex() for _ in range(n)]
    return row


def _dump(n, leak=False, relation_leak=False):
    rng = np.random.default_rng(7)
    const = {cls: [bytes([i % 7] * 16).hex() for i in range(SIZES[cls])] for cls in zk_stats.MASKED}
    lines = []
    for run in range(n):
        for rep in range(2):
            lines.append("ZKREAL\t" + json.dumps(_row("real", run, rep, rng, const if leak else None)))
            lines.append("ZKSIM\t" + json.dumps(_row("sim", run, rep, rng)))
            lines.append("ZKCONTROL\t" + json.dumps(_row("control", run, rep, rng, const)))
        lines.append("ZKREL\t" + json.dumps(_rel("real", run, rng, fixed=relation_leak)))
        lines.append("ZKREL\t" + json.dumps(_rel("sim", run, rng)))
        lines.append("ZKREL\t" + json.dumps(_rel("control", run, rng, fixed=True)))
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


def test_a_relation_across_fields_and_reps_fails_though_every_field_passes():
    """red-team-m1-zk F1's shape: every field uniform, a cross-rep functional of them the same in every session."""
    res = zk_stats.analyse(zk_stats.load(_dump(24, relation_leak=True)))
    assert not res["pass"]
    assert all(res["classes"][c]["real_indistinguishable_from_sim"] for c in zk_stats.CLASSES)
    assert not res["classes"]["rel_level0"]["real_indistinguishable_from_sim"]


def test_a_dump_without_relations_fails():
    lines = [x for x in _dump(24) if not x.startswith("ZKREL")]
    assert not zk_stats.analyse(zk_stats.load(lines))["pass"]


def _copy_rep0(cls, n, q=None):
    """A dump whose rep 1 carries rep 0's first ``n`` values of ``cls`` in every run; ``q``: each rep's ``lig_rows0_q``."""
    lines = []
    for line in _dump(24):
        tag, _, body = line.partition("\t")
        if tag in ("ZKREAL", "ZKSIM"):
            v = json.loads(body)
            if v["rep"] == 1:
                twin = next(json.loads(x.partition("\t")[2]) for x in lines if x.startswith(tag + "\t")
                            and json.loads(x.partition("\t")[2])["run"] == v["run"] and json.loads(x.partition("\t")[2])["rep"] == 0)
                v[cls][:n] = twin[cls][:n]
            if q is not None:
                v["lig_rows0_q"] = q[v["rep"]]
            line = tag + "\t" + json.dumps(v)
        lines.append(line)
    return zk_stats.analyse(zk_stats.load(lines))


def test_a_position_both_reps_open_counts_once():
    """Both reps open the one level-0 codeword: a position both open holds the same row in both, which is one observation."""
    res = _copy_rep0("lig_rows0", 5, q=[[10, 11, 12, 13], [10, 21, 22, 23]])
    assert res["pass"], json.dumps(res["classes"]["lig_rows0"], indent=1)
    assert res["classes"]["lig_rows0"]["shared_by_the_reps"] == 5


def test_a_row_equal_across_reps_at_positions_not_both_opened_fails():
    res = _copy_rep0("lig_rows0", 5, q=[[10, 11, 12, 13], [20, 21, 22, 23]])
    assert not res["pass"]
    assert res["classes"]["lig_rows0"]["reused_across_reps"] == 5


def test_randomness_reused_across_reps_fails_outside_the_level0_rows():
    res = _copy_rep0("exposed", 3, q=[[10, 11, 12, 13], [10, 21, 22, 23]])
    assert not res["pass"]
    assert res["classes"]["exposed"]["reused_across_reps"] == 3
    assert not res["classes"]["exposed"]["real_indistinguishable_from_sim"]


def test_an_inner_column_opened_twice_counts_once():
    """The inner proof's queries repeat: a column at a position already opened by the same rep is one observation."""
    q = list(range(100, 100 + SIZES["inner_open_mu"]))
    q[7] = q[2]
    lines = []
    for line in _dump(24):
        tag, _, body = line.partition("\t")
        if tag in ("ZKREAL", "ZKSIM", "ZKCONTROL"):
            v = json.loads(body)
            v["inner_open_q"] = q
            for cls in zk_stats.INNER_OPENED:
                v[cls][7] = v[cls][2]
            line = tag + "\t" + json.dumps(v)
        lines.append(line)
    res = zk_stats.analyse(zk_stats.load(lines))
    assert res["pass"], json.dumps(res["classes"]["inner_open_mu"], indent=1)
    assert all(res["classes"][c]["repeated_positions"] == 1 for c in zk_stats.INNER_OPENED)
