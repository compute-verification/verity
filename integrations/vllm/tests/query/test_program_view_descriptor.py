"""`program_view.from_descriptor` on a request Build's descriptor, whose program function is inlined into the root (circuits,
VLLM: the lookup of `program_fn` among the definitions raised KeyError)."""
from __future__ import annotations

import pytest

pytest.importorskip("torch")


def test_a_request_build_descriptor_views_its_roots_body_with_its_inputs_as_parameters():
    from tests.pipeline import test_single_request_build_path as T
    from verity.ml.circuits.codec import encode_program
    from verity.ml.circuits.program import is_input
    from verity_vllm.query import program_view as PV

    rep = T._derive(V=64, S=2, LP=4, STEPS=2)
    desc = encode_program(rep.program)
    assert desc["program_fn"] not in desc["definitions"]
    nodes = rep.program.root.body.nodes
    view = PV.from_descriptor(desc)
    assert len(view.calls) == sum(not is_input(n.fn) for n in nodes) > 0
    assert len(view.params) == sum(is_input(n.fn) for n in nodes) > 0 and view.outputs
    ids = {p.id for p in view.params}
    for c in view.calls:
        assert all(r.call in ids or 0 <= r.call < c.id for r in c.reads), c.id
