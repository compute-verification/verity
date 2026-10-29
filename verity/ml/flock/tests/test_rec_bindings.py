"""`rec_bindings.bindings` of a statement whose program is `population_program` of a registered template, at v1 (the
builders' default, `TemplateInstances_v1{template, N}`) and at an old statement's v0 (`TemplateInstances_v0{N}`): each is
rebuilt at the version its program names; given V*'s Definition of the statement (`vstar_definition`), a template other
than it is refused (`Flock.RecVStar.checkTemplate`'s mirror). The statements V* stages are checked against the Lean
verifier in `test_rec_vstar_checks`; this needs no Lean."""
from __future__ import annotations

import json

import pytest
from verity.core.protocols.sampled_proofs.one_stage import partition as P
from verity.ml.circuits.codec import descriptor_id, encode_program, program_digest
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops.boolean import elementwise
from verity.ml.circuits.program import Program

from verity_flock import rec_bindings as RB

I32Add = bind(elementwise.I32Add)


def _statement(version: int, n: int = 3) -> tuple[dict, dict]:
    prog = P.population_program(I32Add, n, version=version)
    return P.partition(prog, P.template_instance_query(I32Add)), json.loads(json.dumps(encode_program(prog)))


@pytest.mark.parametrize("version", P.POPULATION_VERSIONS)
def test_the_bindings_of_a_population_program_at_either_version(version):
    part, desc = _statement(version)
    assert RB.POPULATION[version].fullmatch(desc["program_fn"])
    assert RB.bindings(part, desc) == {"partition": P.digest(part), "program": program_digest(desc),
                                       "program_digests": {descriptor_id(I32Add): program_digest(Program(I32Add))}}


def _staged(defn, n: int = 2) -> tuple[dict, dict]:
    prog = P.population_program(defn, n)
    return P.partition(prog, P.template_instance_query(defn)), json.loads(json.dumps(encode_program(prog)))


def test_a_part_relabelled_under_another_template_is_refused_against_v_stars():
    """`Flock.RecVStar.checkTemplate`'s mirror: a part of the private inner shape (22, 10, 2) staged under its own S, the
    partition object and program consistent, has V*'s bindings as that statement's, digest V*'s Definition's; checked as
    the plain shape's part of the same unit, whose S differs, it is refused by the template, and so is RecCap's statement
    checked as a level's."""
    unit = "rec-residuals/m22-k10-c2-p0"
    holo = RB.vstar_definition(unit, holo=True)
    assert descriptor_id(holo) != descriptor_id(RB.vstar_definition(unit))
    part, desc = _staged(holo)
    assert RB.bindings(part, desc, holo)["program_digests"] == {descriptor_id(holo): program_digest(Program(holo))}
    with pytest.raises(ValueError, match="^the partition's template 'InnerRepCheck_v2.*is not V\\*'s template of the "
                                         "statement 'InnerRepCheck_v2"):
        RB.bindings(part, desc, RB.vstar_definition(unit))
    cap, cdesc = _staged(RB.vstar_definition("rec-cap/n128"))
    assert list(RB.bindings(cap, cdesc, RB.vstar_definition("rec-cap/n4-l2"))["program_digests"]) == ["RecCap_v1"]
    with pytest.raises(ValueError, match="^the partition's template 'RecCap_v1' is not V\\*'s template of the statement "
                                         "'RecOpen_v4\\{LANES=16,H=1\\}'$"):
        RB.bindings(cap, cdesc, RB.vstar_definition("rec-open-stacked/l16-h1"))


def test_v_stars_definition_of_each_unit():
    """Each of V*'s unit names gives its family at the statics the name and the inner session's holo flag give."""
    assert descriptor_id(RB.vstar_definition("rec-open/l16-h1")) == "RecOpen_v4{LANES=16,H=1}"
    assert descriptor_id(RB.vstar_definition("rec-open-stacked/l64-h8")) == "RecOpen_v4{LANES=64,H=8}"
    assert descriptor_id(RB.vstar_definition("rec-sparse/n4-e6-s2-t1")) == "RecSparse_v1{NU=4,ELL=6,S_LOG=2,T_LOG=1}"
    assert descriptor_id(RB.vstar_definition("rec-cap/n128")) == "RecCap_v1"
    assert descriptor_id(RB.vstar_definition("rec-opening/m22-c2-p0")).startswith('InnerRepCheck_v2{S={"dict":{"gf"')
    for bad in ("rec-open/l16", "rec-residuals/m22-k10-c2-p9", "I32Add_v1"):
        with pytest.raises(ValueError):
            RB.vstar_definition(bad)


def test_a_program_of_another_function_is_refused():
    """A `TemplatePopulations` program batches the template too, but is no `population_program` of it."""
    prog = P.mixed_population_program([(I32Add, 3)])
    part = {**P.partition(prog, P.template_instances_query([I32Add])), "query": P.template_instance_query(I32Add)}
    with pytest.raises(ValueError, match="is not TemplateInstances_v1"):
        RB.bindings(part, json.loads(json.dumps(encode_program(prog))))
