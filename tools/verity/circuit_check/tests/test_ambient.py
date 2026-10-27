"""`circuit_check.ambient` (``circuit-check --ambient``): a Definition's flat lowering is one text in a plain interpreter and a
perturbed one and reads no environment, clock or unseeded draw, so it depends only on its builders' operands; a builder that
reads anything else is refused by name. And no word primitive is lowered by whichever of two builders was imported first."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from circuit_check import ambient as A
from circuit_check import cli
from verity_flock import ir_lower as IL

from verity.ml.circuits.defs import REGISTRY

FIXTURES = Path(__file__).parent / "fixtures"

#: the Boolean Definitions the PoUW H1 tile's proof block (`h1unit.Shape(32).unit()`) reaches that lower in under a second each
H1 = ["PouwLow8_v2", "PouwSx8_v2", "PouwAndBit_v2", "PouwI8AddBit_v2", "PouwI8Neg_v2", "PouwI8Add_v2", "PouwU129_v2",
      "PouwZyCell_v2", "F2fpBf16_v2", "PouwXinFinal_v2", "S32ToF32Rn_v2", "F32Mul_v4", "PouwSegDot0_v2", "PouwSegDot_v2",
      "BlackwellS8ImmaDot32_v2", "GemmAccumulatorS8_v2{K=32,DOT=BlackwellS8ImmaDot32_v2}", "PouwZyDequant_v2"]


def _verdicts(specs: list[str], **kw) -> dict[str, str]:
    return {s: r["ambient"]["verdict"] for s, r in zip(specs, A.check(specs, **kw))}


def test_the_h1_units_definitions_lower_to_one_text_in_any_interpreter():
    assert _verdicts(H1) == dict.fromkeys(H1, "same")


@pytest.mark.slow
def test_the_h1_units_larger_definitions_lower_to_one_text_in_any_interpreter():
    """PouwFx1Absorb and PouwNoiseBlock, the H1 tile's largest Definitions that lower in seconds (about 4 and 6 s a lowering,
    three lowerings each): their builders (the absorb's fold, the noise chunk's SHA-512) are none of the small ones'."""
    big = ["PouwFx1Absorb_v2", "PouwNoiseBlock_v2"]
    assert _verdicts(big) == dict.fromkeys(big, "same")


BAD = {"Counted": "differs", "SetOrder": "differs", "Environ": "reads", "Clocked": "reads", "Drawn": "reads"}


def test_a_builder_that_reads_state_besides_its_operands_is_refused_by_name():
    """A count of earlier lowerings, the order of a set of strings, the environment, a clock, an unseeded draw
    (`fixtures/ambient_builders.py`): each refused, and the read named where the builder made it."""
    specs = [f"ambient_builders:{n}" for n in [*BAD, "Plain"]]
    rs = dict(zip(specs, A.check(specs, path=(str(FIXTURES),))))
    assert {s.split(":")[1]: r["ambient"]["verdict"] for s, r in rs.items()} == {**BAD, "Plain": "same"}
    assert {s.split(":")[1]: [f["code"] for f in r["failures"]] for s, r in rs.items()} == {
        **{n: [v] for n, v in BAD.items()}, "Plain": []}
    reads = {s.split(":")[1]: r["failures"][0]["reads"] for s, r in rs.items() if r["ambient"]["verdict"] == "reads"}
    assert [x.split(" at ")[0] for x in reads["Environ"]] == ["os.environ['VERITY_AMBIENT_TEST']"]
    assert [x.split(" at ")[0] for x in reads["Clocked"]] == ["time.perf_counter"]
    assert [x.split(" at ")[0] for x in reads["Drawn"]] == ["random.random"]
    assert all(x.split(" at ")[1].startswith("tools/verity/circuit_check/tests/fixtures/ambient_builders.py:")
               for xs in reads.values() for x in xs)


def test_a_target_that_does_not_lower_is_an_error_by_name():
    (r,) = A.check(["ambient_builders:Missing"], path=(str(FIXTURES),))
    assert r["ambient"]["verdict"] == "error" and {f["code"] for f in r["failures"]} == {"error"}
    assert "AttributeError" in r["failures"][0]["name"]


def test_no_word_primitive_is_lowered_by_whichever_builder_registered_first():
    assert A.ambiguous_words() == {}, (
        "these word primitives are built by two registered Boolean Definitions with different builders, so "
        "`ir_lower._boolean_build` lowers each by whichever module was imported first: give each word one Boolean "
        "Definition, or its own piece in `ir_lower.PIECES`")


def test_a_word_two_builders_build_is_ambiguous_unless_it_has_a_piece_of_its_own(monkeypatch):
    word = SimpleNamespace(id="AmbientTestWord_v1")
    for i, build in (("AmbientTestOne_v1", lambda C, a: a), ("AmbientTestTwo_v1", lambda C, a: [a[0].inv().inv()])):
        monkeypatch.setitem(REGISTRY.defs, i, SimpleNamespace(id=i, word=word, build=build))
    try:
        assert A.ambiguous_words() == {"AmbientTestWord_v1": ["AmbientTestOne_v1", "AmbientTestTwo_v1"]}
        dict.__setitem__(IL.PIECES, word.id, lambda C, a: a)
        assert A.ambiguous_words() == {}
    finally:
        dict.pop(IL.PIECES, word.id, None)


def test_the_command_reports_each_target_and_fails_one_that_reads(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PYTHONPATH", str(FIXTURES))
    out = tmp_path / "r.json"
    assert cli.main(["--ambient", "PouwU129_v2", "ambient_builders:Plain", "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert [(t["kind"], t["target"], t["ambient"]["verdict"]) for t in rep["targets"]] == [
        ("ambient", "PouwU129_v2", "same"), ("ambient", "ambient_builders:Plain", "same")]
    assert "circuit-check: PouwU129_v2: same over 3 lowerings, sha256 " in capsys.readouterr().out
    assert cli.main(["--ambient", "ambient_builders:Environ"]) == 1
    assert "FAIL ambient/reads ambient_builders:Environ" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["--ambient"])
