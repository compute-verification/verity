"""`program.inputs`: every Program parameter, `run` field and check bit is declared, or listed free in `ALLOWED` with its
reason, and each refusal names the fix; each relation's evidence exists; `ALLOWED` only shrinks and holds no stale entry."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from verity.circuits.defs import Builder, PrimitiveDefinition, bind
from verity.circuits.refs import Affine, Coll
from verity.circuits.types import Array, Value
from verity.definitions.registry import dsa_topk
from verity_vllm.program import inputs as IN
from verity_vllm.program.frontend.registration import RunInputs

ROOT = Path(__file__).resolve().parents[4]
FIX = "serve it canonically"

#: the list only shrinks: lower the cap with the entry that goes
ALLOWED_CAP = 2

#: where each `ALLOWED` entry is still made or read, so an entry nothing makes is stale
MADE_AT = {
    "splits": ("adapters/vllm/verity_vllm/program/predict/build.py", '("splits", '),
    "run.survey_hole": ("adapters/vllm/verity_vllm/program/frontend/rules/training_survey.py", '"survey_hole"'),
}
PACKAGE = ROOT / "adapters/vllm/verity_vllm"


@pytest.mark.parametrize("name,kind", [("prompt", "certified"), ("seed", "witness"), ("seed_salt", "certified"), ("prefix_ik", "block"),
                                       ("peers_12", "block"), ("weights", "registered"), ("rank3.weights[7]", "registered")])
def test_a_declared_parameter_has_its_kind(name, kind):
    assert IN.parameter(name).kind == kind


def test_an_allowed_parameter_returns_its_reason_and_an_undeclared_one_is_refused_with_the_fix():
    assert "SplitsFor_v1" in IN.parameter("rank0.splits")
    with pytest.raises(IN.Free, match=FIX):
        IN.parameter("tile_choice")
    with pytest.raises(IN.Free, match="'rank0.schedule'"):
        IN.parameter("rank0.schedule")
    with pytest.raises(IN.Free, match="'rank1.tile_choice'"):
        IN.check_params([("prompt", None), ("run", None), ("rank1.tile_choice", None), ("weights", None)])
    IN.check_params([("prompt", None), ("seed", None), ("run", None), ("peers_0", None), ("weights", None)])


def test_a_run_field_is_read_only_under_a_declaration():
    r = RunInputs.declare([])
    hole = r.value("hole0.1:add_3", Array(3, Value(1)), "survey_hole")
    assert hole.type == Array(3, Value(1))
    assert "check_identity" in IN.run_field("hole0.1:add_3", "survey_hole")
    with pytest.raises(IN.Free, match=FIX):
        r.value("model.layers.3.mlp/align_key[t=7]", Value(32), "moe_align_key")
    with pytest.raises(IN.Free, match=FIX):
        r.value("model.layers.3.self_attn.attn/topk_switches[t=2049]", Array(3, Value(1)), "dsa_topk_switches")
    assert [f["name"] for f in r.fields] == ["hole0.1:add_3"]


def test_run_fields_are_laid_out_in_first_read_order_and_read_once():
    r = RunInputs.declare([])
    names = [f"hole{i}.0:add_{i}" for i in range(3)]
    for name, n in zip(names, (3, 5, 2)):
        r.value(name, Array(n, Value(1)), "survey_hole")
    assert [(f["name"], f["offset"]) for f in r.fields] == list(zip(names, (0, 3, 8)))
    with pytest.raises(ValueError, match="read twice"):
        r.value(names[1], Array(5, Value(1)), "survey_hole")
    assert r.value("hole9.0:add_9", Value(1), "survey_hole").refs[0] == ("p", 0, 10)


@pytest.mark.parametrize("declared", sorted(IN.RUN))
def test_every_run_declaration_is_read(declared):
    readers = [p for p in PACKAGE.rglob("*.py") if p.name != "inputs.py" and f'"{declared}"' in p.read_text()]
    assert readers, f"no rule reads a run field under {declared!r}: remove its declaration from RUN"


#: a relation `RELATIONS` doesn't list: any permutation of the top-3 set of 5 scores (log2(3!) free bits), a test's own
ANY_ORDER = PrimitiveDefinition("TAnyTopKOrder", 1, (("scores", Array(5, Value(32))), ("order", Array(3, Value(32)))), Value(1),
                                lambda *w: dsa_topk.order_ok(w[:5], w[5:], 3), register=False)


def _call(fn):
    params, ret = fn.signature()
    B = Builder(tuple(params), ret)
    return B, B.call(fn, *(Coll(t, Affine("p", i, 0, t.leaves)) for i, (_, t) in enumerate(params)))


def test_a_check_bit_is_a_declared_relations_verdict():
    B, ok = _call(bind(dsa_topk.DsaTopKLongRow, N=5, TOPK=3))
    IN.check(B, ok, "the long row")
    B, ok = _call(ANY_ORDER)
    with pytest.raises(IN.Free, match="TAnyTopKOrder_v1, which RELATIONS doesn't list"):
        IN.check(B, ok, "any permutation")
    B = Builder((("b", Value(1)),), Value(1))
    with pytest.raises(IN.Free, match="no call's output"):
        IN.check(B, Coll(Value(1), Affine("p", 0, 0, 1)), "a parameter bit")


def test_advice_needs_a_charging_term_and_a_witness_a_listed_relation():
    with pytest.raises(ValueError, match="charged by a term of the window charge"):
        IN.Kind("advice", "a choice", charge="decisions")
    with pytest.raises(ValueError, match="names its relation in RELATIONS"):
        IN.Kind("witness", "an order", relation="TAnyTopKOrder_v1")


@pytest.mark.parametrize("relation", sorted(IN.RELATIONS))
def test_every_relation_names_evidence_that_exists(relation):
    path, name = IN.RELATIONS[relation].split("::")
    text = (ROOT / path).read_text()
    form = r"def {}\(" if path.endswith(".py") else r"(theorem|lemma) {}\b"
    assert re.search(form.format(re.escape(name)), text), f"{relation}: {path} defines no {name}"


def test_allowed_only_shrinks_and_holds_no_stale_entry():
    assert len(IN.ALLOWED) <= ALLOWED_CAP, f"ALLOWED holds {len(IN.ALLOWED)} entries, over its cap {ALLOWED_CAP}: it only shrinks"
    assert set(MADE_AT) == set(IN.ALLOWED), "each ALLOWED entry says where its name is made (MADE_AT)"
    stale = [k for k, (path, made) in MADE_AT.items() if made not in (ROOT / path).read_text()]
    assert not stale, f"ALLOWED entries nothing makes any more (remove them, and lower ALLOWED_CAP): {stale}"
