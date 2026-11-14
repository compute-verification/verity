"""`partition_by` over a Definition key is the structural query `group(nodes(".", "@*"), key)`: on one Program, the units
`partition_by` makes of `from_descriptor`'s Calls are the classes `query_ast.group` makes of the root's nodes, with the same
ids, Calls and order (the root's Input nodes are the view's parameters, so they are no Call)."""
from __future__ import annotations

import gzip
import json

import pytest
from verity.circuits import query_ast as G
from verity.circuits.codec import encode_program
from verity.circuits.defs import REGISTRY, CompositeDefinition, bind
from verity.circuits.program import Program, is_input
from verity.circuits.refs import tuple_of
from verity.circuits.types import Array, Tuple
from verity.definitions.registry import b1
from verity.definitions.registry import prims as P
from verity.definitions.registry.prims import BF16, F32

from verity_vllm.query.program_view import from_descriptor
from verity_vllm.query.units import ValueRef, partition_by


def _mix(B, S, x, xb, w):
    a = B.call(bind(b1.Gemm, K=32, N=4), xb, w)
    b = B.call(bind(b1.Gemm, K=32, N=2), xb, w[0:2])
    c = B.call(bind(b1.Gemm, K=32, N=4), xb, w)
    d = B.batch(P.F32Mul, x, x, axes=(0, 0))
    e = B.call(P.F32Add, B.call(b1.ZERO32), d[0])
    return tuple_of(a, b, c, d, e)


_Mix = CompositeDefinition(
    "QgroupMix", 1, (),
    lambda S: ((("x", Array(4, F32)), ("xb", Array(32, BF16)), ("w", Array(4, Array(32, BF16)))),
               Tuple(Array(4, BF16), Array(2, BF16), Array(4, BF16), Array(4, F32), F32)),
    _mix, register=False)

KEYS = {("base",): lambda c: c.family, ("definition",): lambda c: c.definition, ("name",): lambda c: c.family.rsplit("_v", 1)[0],
        ("static:N",): lambda c: str(c.statics.get("N", "")), ("base", "static:K"): lambda c: f"{c.family}|{c.statics.get('K', '')}",
        (): lambda c: ""}


@pytest.fixture(scope="module")
def mix():
    program = Program(bind(_Mix))
    return program, from_descriptor(encode_program(program), registry=REGISTRY)


@pytest.mark.parametrize("key", sorted(KEYS))
def test_group_over_the_roots_nodes_is_partition_by(mix, key):
    program, view = mix
    nodes = program.root.body.nodes
    call_of = {j: k for k, j in enumerate(j for j, n in enumerate(nodes) if not is_input(n.fn))}
    fam = G.evaluate(program, G.group(G.nodes(".", "@*"), list(key)))
    classes = []
    for i, s in enumerate(fam):
        calls = frozenset(call_of[j] for p in s.parts for j, _ in p.items if j in call_of)
        if calls:
            classes.append(("|".join(fam.label_of(i)), calls))
    classes.sort(key=lambda u: min(u[1]))
    units = [(u.id, u.calls) for u in partition_by(KEYS[key])(view)]
    assert classes == units
    assert len(units) > 1 or key == ()


def test_a_descriptor_file_views_as_its_document_and_the_checks_are_the_last_leaves_it_returns(mix, tmp_path):
    program, view = mix
    desc = encode_program(program)
    for name, opener in (("descriptor.json", open), ("descriptor.json.gz", gzip.open)):
        with opener(tmp_path / name, "wt") as f:
            json.dump(desc, f)
        assert from_descriptor(str(tmp_path / name), registry=REGISTRY) == view and from_descriptor(str(tmp_path / name)) == view
    # the Program returns 4 + 2 + 4 + 4 + 1 leaves: its last is Call 5's, the four before it the batch's (Call 3)
    last, batch = ValueRef(5, "out"), ValueRef(3, "out")
    assert view.checks == () and view.outputs[-2:] == (batch, last)
    checks = {n: from_descriptor(desc, registry=REGISTRY, checks=n).checks for n in (1, 2, 5, 15)}
    assert checks == {1: (last,), 2: (batch, last), 5: (batch, last), 15: view.outputs}
    with pytest.raises(ValueError, match="^the Program returns fewer than its 16 checks' leaves$"):
        from_descriptor(desc, registry=REGISTRY, checks=16)
