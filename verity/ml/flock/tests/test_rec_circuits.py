"""`rec_circuits`: a V* statement's circuit text built once per shape and read back after."""
from __future__ import annotations

import pytest
from test_rec_outer import PROGRAM, _coefs, _queries, _top
from test_rec_residuals import S_SMALL
from verity_flock import circuit as FC
from verity_flock import rec_circuits as RC
from verity_flock import rec_outer as O
from verity_flock import rec_residuals as RR


def test_a_text_is_kept_under_its_kind_and_key_alone_and_nothing_without_the_cache(tmp_path, monkeypatch):
    assert RC.load("rec-open", {"n": 2}) is None
    RC.store("rec-open", {"n": 2}, "text", {"unit_rows": 1})
    monkeypatch.setenv(RC.ENV, str(tmp_path))
    RC.store("rec-open", {"n": 2}, "text", {"unit_rows": 1})
    assert RC.load("rec-open", {"n": 2}) == ("text", {"unit_rows": 1, "kind": "rec-open", "key": {"n": 2}})
    assert RC.load("rec-open", {"n": 3}) is None and RC.load("rec-residuals", {"n": 2}) is None
    assert len(list(tmp_path.iterdir())) == 1


def test_a_kept_residuals_circuit_is_the_one_built(tmp_path, monkeypatch):
    monkeypatch.setenv(RC.ENV, str(tmp_path))
    built = RR.statement(S_SMALL, "toy-residuals/small", 2)
    kept = RR.statement(S_SMALL, "toy-residuals/small", 2)
    assert built.low is not None and not built.circuit_kept and kept.low is None and kept.circuit_kept
    assert kept.text == built.text and kept.unit_rows == built.low.layout.useful
    assert kept.digests == (FC.digest(built.text), FC.class_digest(built.text))


@pytest.mark.slow
def test_a_level_staged_from_its_kept_circuit_writes_the_verifiers_files_it_wrote_when_built(tmp_path, monkeypatch):
    """The second staging of one level shape reads its circuit back and skips its own replay of the unit (`opened` and
    `summed` unknown); the circuit, program, partition and reads files are byte for byte the first staging's. Staging a level
    takes about a minute; no cheaper statement reaches `rec_outer.stage`'s kept path."""
    monkeypatch.setenv(RC.ENV, str(tmp_path / "cache"))
    qs, layers = _queries(level=1)
    (level,) = O.levels(qs)
    coefs = _coefs(qs)
    tops = O.TopValue.build(PROGRAM, 1, {0: _top(layers[1], 2), 1: _top(layers[1], 3)})
    chain = O.Chain.build(PROGRAM, qs, coefs)
    s = [O.stage(level, tmp_path / d, chain, coefs, tops) for d in ("built", "kept")]
    assert not s[0]["circuit_kept"] and s[0]["opened"] == 3 and s[0]["not_opened"] == [(1, 1)]
    assert s[1]["circuit_kept"] and s[1]["opened"] is None and s[1]["not_opened"] is None
    assert {k: v for k, v in s[1].items() if k not in ("circuit_kept", "opened", "not_opened", "summed", "not_summed")} == \
        {k: v for k, v in s[0].items() if k not in ("circuit_kept", "opened", "not_opened", "summed", "not_summed")}
    for f in ("circuit.txt", "program.json", "partition.json", "registered.json"):
        assert (tmp_path / "kept" / f).read_bytes() == (tmp_path / "built" / f).read_bytes(), f
