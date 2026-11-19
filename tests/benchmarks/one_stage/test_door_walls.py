"""`door_walls.py`: the census reports the door's refusals by class and owner (Lean's compose refusal of lean's case on a
stub of `flock-lower-check` that answers as Lean does, and the door's own refusal of a template of no gate, which no
lowering reaches), and exits 1 when a block is refused; the fuzzer's draws are a function of the seed; and, on Lean's own
lowering, an honest block passes the row check, the same block flags a planted wrong evaluation of its rows
(`tmpl.evaluate` monkeypatched), and the fuzzer's findings are a function of its seed."""
from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "benchmarks" / "one_stage"))
import door_walls as W  # noqa: E402

COMPOSE_WHY = "no row has a compression before its last block"


def _stub(tmp_path: Path) -> Path:
    """An executable that refuses every `compose` as Lean refuses lean's case."""
    exe = tmp_path / "flock-lower-check"
    exe.write_text(f"#!/bin/sh\necho \"flock-lower-check compose: unit/$3: {COMPOSE_WHY}\" >&2\nexit 1\n")
    exe.chmod(0o755)
    return exe


def _zero_gate() -> tuple[bytes, dict]:
    """A template that passes its parameter through: no gate."""
    from verity.circuits.types import Value
    ps = (("a", Value(8)),)
    return W._rooted(W._composite("WallsPass", ps, Value(8), lambda B, S, a: a), ps, Value(8))


def test_the_census_reports_leans_compose_refusal_and_the_doors_by_class(tmp_path):
    W._catalog()
    exe = _stub(tmp_path)
    c, q = W.lean_case()
    (r,) = W.census(c, q, "one-per-template", lambda: exe)
    assert (r.step, r.templates) == ("lower", ("WallsLeanC_v1",))
    assert r.klass == f"lower: Refused: E_LOWER: flock-lower-check compose (<n>): flock-lower-check compose: unit/<T>: {COMPOSE_WHY}"
    c, q = _zero_gate()
    (z,) = W.census(c, q, "one-per-template", lambda: exe)
    assert (z.step, z.klass) == ("lower", "lower: Refused: E_LOWER: C_S's root calls [..], and a block's units are instances of one template")
    rows = W.table([r, z])
    assert sorted((x["step"], x["count"], x["owner"]) for x in rows) == [
        ("lower", 1, "lean's compose (Flock.Compose)"), ("lower", 1, "proofs' door (one_circuit.lower, proofs.tmpl_why)")]
    assert W.main(["census", "--circuit", "lean_case", "--lower", str(exe)]) == 1


def test_one_cause_is_one_class():
    a = "E_LOWER: flock-lower-check compose (1): unit/Gemm_v2{K=16,N=16}: no row at 1234 of 'x_v1'"
    b = "E_LOWER: flock-lower-check compose (2): unit/SiluMul_v2{I=128}: no row at 7 of 'y_v3'"
    assert W.normalise(a) == W.normalise(b)
    assert W.normalise("digest " + "ab" * 32) == "digest <H>"


def test_the_fuzzers_draws_are_the_seeds():
    W._catalog()
    for seed in range(20):
        assert W.draw(random.Random(seed)).doc() == W.draw(random.Random(seed)).doc()
    assert len({repr(W.draw(random.Random(s)).doc()) for s in range(20)}) > 15


def _xor_spec() -> W.Spec:
    """One template x0 ^ x1 on 32-bit words, two units in one block."""
    xor = next(i for i, (fn, _ts, _r) in enumerate(W._prims()) if fn.id == "PouwXor32_v1")
    return W.Spec([{"params": [32, 32], "steps": [[xor, [0, 1]]], "ret": [2], "nested": False}],
                  [{"t": 0, "form": "batch", "n": 2, "args": [[["p", 0], ["p", 1]], [["p", 2], ["p", 3]]]}],
                  [32, 32, 32, 32], [], [[0, 2]])


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_the_row_check_holds_an_honest_block_and_flags_a_planted_wrong_one(monkeypatch):
    from verity_flock import tmpl as T
    W._catalog()
    exe = W._exe(None)
    honest = W.check_case(_xor_spec(), exe, 7)
    assert not [f.klass for f in honest if f.step == "rows"], [f.klass for f in honest]
    evaluate = T.evaluate

    def flipped(t, inputs, out_cols=None, **kw):
        y = evaluate(t, inputs, out_cols, **kw)
        y[0] ^= 1
        return y

    monkeypatch.setattr(T, "evaluate", flipped)
    planted = {f.klass for f in W.check_case(_xor_spec(), exe, 7)}
    assert any(k.startswith("rows: honest output refused") for k in planted), planted
    monkeypatch.undo()
    first = W.fuzz(3, 4, 5.0, exe)
    again = W.fuzz(3, 4, 5.0, exe)
    assert [c | {"example": None} for c in first["classes"]] == [c | {"example": None} for c in again["classes"]]

