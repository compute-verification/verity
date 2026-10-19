"""`rec_bindings.bindings` of a statement whose program is `population_program` of a registered template, at v1 (the
builders' default, `TemplateInstances_v1{template, N}`) and at an old statement's v0 (`TemplateInstances_v0{N}`): each is
rebuilt at the version its program names. The statements V* stages are checked against the Lean verifier in
`test_rec_vstar_checks`; this needs no Lean."""
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


def test_a_program_of_another_function_is_refused():
    """A `TemplatePopulations` program batches the template too, but is no `population_program` of it."""
    prog = P.mixed_population_program([(I32Add, 3)])
    part = {**P.partition(prog, P.template_instances_query([I32Add])), "query": P.template_instance_query(I32Add)}
    with pytest.raises(ValueError, match="is not TemplateInstances_v1"):
        RB.bindings(part, json.loads(json.dumps(encode_program(prog))))
