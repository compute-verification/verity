"""A TP rank's stochastic Program is the TP1 Program under the rank wrapper's `inner` scope (`vllm_meta.WithPeers` calls the TP1 wrapper
as its submodule `inner`): its selector is `inner/gumbel_topp_token_select[0]`, not `gumbel_topp_token_select[0]` at the root.  The
selector's seed and splits reads are the sampler's (`runner.sampler/seed`, `runner.sampler/splits`), never token ids a module body
consumes: a stochastic TP2 B8 manifest (phi4-14b) required an `input_token_ids` identity `inner/in` per sampling event, which
no collector binds, and the composition took it as its reference for the token ids."""

from __future__ import annotations

import dataclasses

import pytest

from tests.query.test_literal_operands import RES, constant_s_program, sampler_program
from verity_vllm.query.units import ValueRef
from verity_vllm.correspondence.reader_for_query import Correspondence
from verity_vllm.query.program_view import ProgramView
from verity_vllm.query.required import request_manifest

IMPL = ("embed", "lm_head")


def _rank_name(name: str) -> str:
    mod = name.partition("/")[0]
    return f"inner.{name}" if "." in mod else f"inner/{name}"


def under_rank_wrapper(P: ProgramView) -> ProgramView:
    """`P` as a rank Program names it: every module path and weight field under `inner.`, the root's Calls under `inner/`."""
    W = {p.name: f"inner.{p.name}" for p in P.params[0].port_table}
    weights = dataclasses.replace(P.params[0], port_table=tuple(dataclasses.replace(p, name=W[p.name]) for p in P.params[0].port_table))
    calls = tuple(dataclasses.replace(c, name=_rank_name(c.name), reads=tuple(ValueRef(r.call, W[r.port]) if r.call == -1 else r for r in c.reads))
                  for c in P.calls)
    return dataclasses.replace(P, calls=calls, params=(weights, *P.params[1:]))


def _identities(P: ProgramView, res: dict, **kw) -> list[tuple]:
    man = request_manifest(P, Correspondence(P, implementation_paths=IMPL), res, request_id="r0", **kw).manifest
    assert man["complete"] and man["unmodelled"] == {}
    return sorted((x["family"], x["step"], x["op_path"], x["output_member"], tuple(x["element_range"])) for x in man["identities"])


@pytest.mark.parametrize("program", [sampler_program, constant_s_program])
def test_a_tp_ranks_sampler_requires_the_tp1_identities(program):
    P = program()
    R = under_rank_wrapper(P)
    assert Correspondence(R).module_of(5 if program is sampler_program else 6) == "inner"
    tp1 = _identities(P, RES)
    assert not any(op == "inner" for _f, _s, op, _m, _r in tp1)
    assert _identities(R, dict(RES, tp={"world": 2, "rank": 0}), rank=0, ranks=2) == tp1
    assert [x for x in tp1 if x[0] == "input_token_ids"] == [("input_token_ids", s, "embed", "in", (0, 1)) for s in (0, 1)]
