"""SHARED-CODE (lane ov-moe): `bench.resolve_log.short_label` labels a `resolver.Group` root (D70) instead of raising
`AttributeError: 'Group' object has no attribute 'reason'` — which made the resolve stage (and so G2) fail for any profile
whose scope pattern returns a Group, e.g. `MoEFusedExperts` on `vllm.moe_forward`."""

from __future__ import annotations

from verity_vllm.observe.fold.resolve_log import short_label
from verity_vllm.observe.events import ViewDesc
from verity_vllm.observe.fold.resolver import Compute, Group, PerRow, Transparent, Unsupported
from verity.ml.circuits.defs import bind
from verity_vllm.program.registry import b1


def _v(alloc, shape):
    strides, s = [], 1
    for n in reversed(shape):
        strides.append(s)
        s *= n
    return ViewDesc(alloc, 0, tuple(shape), tuple(reversed(strides)), "bfloat16", "cuda:0")


def test_short_label_of_a_group_counts_its_parts():
    x, y = _v(1, (4, 32)), _v(2, (4, 16))
    c = Compute(bind(b1.SiluMul, I=16), 4, (PerRow(x),), (PerRow(y),), (3,), "mlp")
    t = Transparent("noop", (4,), hint="schedule")
    g = Group((t, c, c), (2, 3, 4), hint="scope", pattern="p")
    assert short_label(g) == "Group of 3: 2 Compute, 1 Transparent"


def test_short_label_of_the_other_spellings_is_unchanged():
    x, y = _v(1, (4, 32)), _v(2, (4, 16))
    c = Compute(bind(b1.SiluMul, I=16), 4, (PerRow(x),), (PerRow(y),), (3,), "mlp")
    assert short_label(c) == f"Compute {c.spec.id}"
    assert short_label(Transparent("alias", (5,))) == "Transparent alias"
    assert short_label(Unsupported(7, "aten.foo", "no pattern", {})) == "Unsupported: no pattern"
