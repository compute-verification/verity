"""``circuit-check --batch`` (`cli.batch`): each template's batch-invariance verdict (`verity_vllm.check.batch`) is held to the one
declared beside it, and the command fails on one that isn't as declared."""
from __future__ import annotations

import json

import pytest
from circuit_check import cli

from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.refs import array_of
from verity.ml.circuits.types import Array, Value
from verity_vllm.check.batch import SHARED, Rows, Template

V16 = Value(16)
ADD = PrimitiveDefinition("CircuitCheckBatchAdd16", 1, (("a", V16), ("b", V16)), V16, lambda a, b: (a + b) & 0xFFFF,
                          register=False)


def _sig(S):
    return (("x", Array(S.M, V16)), ("w", V16)), Array(S.M, V16)


ROWWISE = CompositeDefinition("CircuitCheckBatchRowwise", 1, ("M",), _sig,
                              lambda B, S, x, w: array_of([B.call(ADD, x[i], w) for i in range(S.M)]), register=False)
#: every row plus row 0
CROSS = CompositeDefinition("CircuitCheckBatchCross", 1, ("M",), _sig,
                            lambda B, S, x, w: array_of([B.call(ADD, x[i], x[0]) for i in range(S.M)]), register=False)
#: a static the launch's row count picks
BY_ROWS = CompositeDefinition("CircuitCheckBatchByRows", 1, ("M", "K"), _sig,
                              lambda B, S, x, w: array_of([B.call(ADD, x[i], w) for i in range(S.M)]) if S.K == 1
                              else array_of([B.call(ADD, B.call(ADD, x[i], w), w) for i in range(S.M)]), register=False)

ROWS = Rows(("a", "b"))
SHAPES = tuple(((), m) for m in (1, 2, 4))


def _templates(declared_cross=(False, None)):
    return lambda: [
        Template("rowwise", bind(ROWWISE, M=2), (ROWS, SHARED), ROWS, bind_at=lambda req, m: bind(ROWWISE, M=2), shapes=SHAPES),
        Template("cross", bind(CROSS, M=2), (ROWS, SHARED), ROWS, expect=declared_cross),
        Template("by-rows", None, bind_at=lambda req, m: bind(BY_ROWS, M=2, K=1 if m < 4 else 2), shapes=SHAPES,
                 expect=(None, False))]


def test_the_report_gives_each_templates_verdict_against_its_declared_one():
    rep = cli.batch(_templates())
    assert rep["schema"] == "verity-circuit-check/batch/v1"
    assert [(v["template"], v["got"], v["ok"]) for v in rep["verdicts"]] == [
        ("rowwise", [True, True], True), ("cross", [False, None], True), ("by-rows", [None, False], True)]
    assert "| `cross` | **fail**" in rep["table"]


def test_the_command_fails_a_verdict_that_is_not_as_declared(tmp_path, monkeypatch, capsys):
    out = tmp_path / "batch.json"
    real = cli.batch
    monkeypatch.setattr(cli, "batch", lambda: real(_templates()))
    assert cli.main(["--batch", "--out", str(out)]) == 0
    assert json.loads(out.read_text())["verdicts"][1]["crossings"]
    monkeypatch.setattr(cli, "batch", lambda: real(_templates(declared_cross=(True, None))))
    assert cli.main(["--batch"]) == 1
    assert "FAIL cross" in capsys.readouterr().out
    for argv in (["--batch", "--ambient"], ["--batch", "F32Mul_v4"]):
        with pytest.raises(SystemExit):
            cli.main(argv)
