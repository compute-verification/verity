"""Hybrid KV cache: with several KV-cache groups (`StepStart.kv_groups`, vLLM's hybrid KV-cache
manager -- Gemma-2's sliding-window vs global layers) a layer's KV rows are addressed by ITS group's block table and slot mapping;
a single group (old logs: `kv_groups is None`) is byte-identical to the legacy fields; an unknown layer is Unsupported, never a guess."""
from __future__ import annotations

import dataclasses

import pytest

from verity_vllm.observe.events import Note, StepStart, ViewDesc, from_json, to_json
from verity_vllm.observe.fold.patterns import GeometryError, kv_group_of, kv_slots
from verity_vllm.observe.fold.resolver import Compute, Gather, GatherMany, Transparent, Unsupported, external_buffers, resolve_step
from tests.observe.test_patterns_synthetic import BS, by_spec, context, decode_step
from verity_vllm.observe.fold.tree import step_events

LAYER = "model.layers.0.self_attn.attn"
OTHER = "model.layers.1.self_attn.attn"
ATTN = "Attention_v3{T=33,NH=9,KVH=3,D=64,BN=128}"


def _groups(start: StepStart, this_layer_group: int, shift: int = 8) -> list[dict]:
    """Two groups: group 0 = the legacy tables; group 1 = the same request's blocks shifted by `shift` (disjoint cells of the
    shared tensor).  `this_layer_group` says which group holds the synthetic step's layer 0."""
    req = start.rows[0][0]
    g0 = {"layers": [OTHER if this_layer_group else LAYER], "block_size": start.block_size,
          "block_tables": dict(start.block_tables), "slot_mapping": list(start.slot_mapping)}
    g1 = {"layers": [LAYER if this_layer_group else OTHER], "block_size": start.block_size,
          "block_tables": {req: [b + shift for b in start.block_tables[req]]},
          "slot_mapping": [s + shift * start.block_size for s in start.slot_mapping]}
    return [g0, g1]


def _attention_and_moves(res):
    (c,) = by_spec(res)[ATTN]
    assert isinstance(c, Compute)
    moves = []
    for r in res:
        if isinstance(r, Transparent) and r.effect == "move" and isinstance(r.dst, Gather):
            moves.append(r)
            moves.extend(m for m in getattr(r, "also", ()) if isinstance(m.dst, Gather))     # TransparentGroup: k + v
    return c, moves


def test_kv_group_of_is_none_for_a_single_group_and_selects_by_layer_name():
    events, start = decode_step()
    ctx = context(start, None)
    assert kv_group_of(ctx, LAYER) is None and kv_group_of(ctx, None) is None
    one = dataclasses.replace(start, kv_groups=[{"layers": [LAYER], "block_size": BS, "block_tables": {}, "slot_mapping": []}])
    assert kv_group_of(context(one, None), "anything") is None          # one group: legacy fields are complete
    two = dataclasses.replace(start, kv_groups=_groups(start, this_layer_group=1))
    ctx2 = context(two, None)
    assert kv_group_of(ctx2, LAYER) is two.kv_groups[1] and kv_group_of(ctx2, OTHER) is two.kv_groups[0]
    with pytest.raises(GeometryError, match="unknown"):
        kv_group_of(ctx2, None)
    with pytest.raises(GeometryError, match="none of the 2"):
        kv_group_of(ctx2, "model.layers.7.self_attn.attn")


def test_kv_slots_follow_the_layers_group():
    events, start = decode_step()
    two = dataclasses.replace(start, kv_groups=_groups(start, this_layer_group=1, shift=8))
    ctx = context(two, None)
    legacy = kv_slots(context(start, None), "r0", 33)
    assert kv_slots(ctx, "r0", 33, layer=OTHER) == legacy                   # group 0 == the legacy table
    assert kv_slots(ctx, "r0", 33, layer=LAYER) == tuple(s + 8 * BS for s in legacy)
    with pytest.raises(GeometryError, match="unknown"):                       # two groups, no layer: never a guess
        kv_slots(ctx, "r0", 33)


def test_single_group_resolution_is_unchanged_with_or_without_kv_groups():
    events, start = decode_step()
    se = step_events(events)
    base = resolve_step(se, context(start, se.sample))
    one = dataclasses.replace(start, kv_groups=[{"layers": [LAYER], "block_size": BS, "block_tables": dict(start.block_tables),
                                                "slot_mapping": list(start.slot_mapping)}])
    res = resolve_step(se, context(one, se.sample))
    assert not [r for r in res if isinstance(r, Unsupported)]
    cb, mb = _attention_and_moves(base)
    c1, m1 = _attention_and_moves(res)
    assert (c1.operands, c1.outputs, c1.spec) == (cb.operands, cb.outputs, cb.spec)
    assert [m.dst for m in m1] == [m.dst for m in mb]


def test_layer_in_group_1_reads_and_writes_group_1_cells():
    events, start = decode_step()
    se = step_events(events)
    base = resolve_step(se, context(start, se.sample))
    two = dataclasses.replace(start, kv_groups=_groups(start, this_layer_group=1, shift=8))
    res = resolve_step(se, context(two, se.sample))
    assert not [r for r in res if isinstance(r, Unsupported)], [r.reason for r in res if isinstance(r, Unsupported)]
    cb, mb = _attention_and_moves(base)
    c1, m1 = _attention_and_moves(res)
    # the attention gathers K/V at the shifted slots ...
    gb = [o for o in cb.operands if isinstance(o, GatherMany)]
    g1 = [o for o in c1.operands if isinstance(o, GatherMany)]
    assert len(gb) == len(g1) == 2
    for a, b in zip(gb, g1):
        assert b.view == a.view and b.indices == tuple(tuple(s + 8 * BS for s in row) for row in a.indices)
    # ... and the KV moves write there (k and v)
    assert len(mb) == len(m1) == 2
    for a, b in zip(mb, m1):
        assert isinstance(a.dst, Gather) and isinstance(b.dst, Gather) and b.dst.view == a.dst.view
        assert b.dst.indices == tuple(s + 8 * BS for s in a.dst.indices)
    # everything else in the step is the same
    others = lambda d: sorted(s for s in d if not s.startswith("Attention_v3"))
    assert others(by_spec(res)) == others(by_spec(base))


def test_layer_in_group_0_of_two_is_the_legacy_addressing():
    events, start = decode_step()
    se = step_events(events)
    base = resolve_step(se, context(start, se.sample))
    two = dataclasses.replace(start, kv_groups=_groups(start, this_layer_group=0, shift=8))
    res = resolve_step(se, context(two, se.sample))
    assert not [r for r in res if isinstance(r, Unsupported)]
    cb, mb = _attention_and_moves(base)
    c0, m0 = _attention_and_moves(res)
    assert c0.operands == cb.operands and [m.dst for m in m0] == [m.dst for m in mb]


def test_layer_of_no_group_is_unsupported_not_guessed():
    events, start = decode_step()
    se = step_events(events)
    groups = _groups(start, this_layer_group=1)
    groups[1]["layers"] = ["model.layers.9.self_attn.attn"]          # layer 0 is in neither group
    two = dataclasses.replace(start, kv_groups=groups)
    res = resolve_step(se, context(two, se.sample))
    uns = [r for r in res if isinstance(r, Unsupported)]
    assert uns and all("none of the 2 KV-cache groups" in u.reason for u in uns)
    assert ATTN not in by_spec(res)


def test_kv_layer_of_prefers_the_string_argument_then_the_module_hint():
    from verity_vllm.observe.fold.patterns import kv_layer_of
    events, start = decode_step()
    se = step_events(events)
    att = next(n for n in se.roots if n.op.startswith("vllm.unified_attention_with_output"))
    kvu = next(n for n in se.roots if n.op.startswith("vllm.unified_kv_cache_update"))
    assert kv_layer_of(att) == LAYER and kv_layer_of(kvu) == LAYER          # the synthetic step passes the string
    # the real vLLM d9105ea80 log: `PyObject layer_name` is recorded as {} -> the observer's module path decides
    real = dataclasses.replace(att.enter, args=[*att.enter.args[:4], {}, *att.enter.args[5:]], module="model.layers.6.self_attn.attn")
    assert kv_layer_of(dataclasses.replace(att, enter=real)) == "model.layers.6.self_attn.attn"
    nothing = dataclasses.replace(att.enter, args=[*att.enter.args[:4], {}, *att.enter.args[5:]], module=None)
    assert kv_layer_of(dataclasses.replace(att, enter=nothing)) is None


def test_bare_reshape_and_cache_flash_uses_the_module_hint_for_its_group():
    """This vLLM issues `reshape_and_cache_flash` bare (no `unified_kv_cache_update` parent): the module hint keys the group."""
    from verity_vllm.observe.fold.patterns import KVCacheUpdate
    events, start = decode_step()
    se = step_events(events)
    rc = next(n for n in se.roots for n in n.walk() if n.op.startswith("_C_cache_ops.reshape_and_cache_flash"))
    two = dataclasses.replace(start, kv_groups=_groups(start, this_layer_group=1, shift=8))
    ctx = context(two, se.sample)
    bare = dataclasses.replace(rc, enter=dataclasses.replace(rc.enter, module="model.layers.0.self_attn.attn", parent=None), children=[])
    res = KVCacheUpdate().match(bare, ctx)
    assert not isinstance(res, Unsupported), getattr(res, "reason", None)
    assert res.dst.indices == tuple(s + 8 * BS for s in start.slot_mapping)
    none = dataclasses.replace(rc, enter=dataclasses.replace(rc.enter, module=None, parent=None), children=[])
    res = KVCacheUpdate().match(none, ctx)
    assert isinstance(res, Unsupported) and "unknown" in res.reason


def test_step_start_kv_groups_round_trip_and_old_logs_default_to_none():
    events, start = decode_step()
    two = dataclasses.replace(start, kv_groups=_groups(start, this_layer_group=1))
    d = to_json(two)
    back = from_json(d)
    assert back.kv_groups == two.kv_groups and back.block_tables == start.block_tables
    d.pop("kv_groups")                                                  # a log written before the field existed
    assert from_json(d).kv_groups is None


def test_the_external_buffers_note_gives_each_modules_kv_cache_as_a_view_on_its_allocations_device():
    note = Note("external buffers", {"buffers": {
        "kv_cache[model.layers.1.attn]": {"alloc": 7, "shape": [40, 2, 1, 584], "dtype": "uint8", "offset": 1728, "strides": [1435968, 584, 584, 1]},
        "runner.kv_caches[0]": {"alloc": 7, "shape": [57454080], "dtype": "uint8"}}})
    assert external_buffers(note, {7: "cuda:3"}.get) == {
        "kv_cache[model.layers.1.attn]": ViewDesc(7, 1728, (40, 2, 1, 584), (1435968, 584, 584, 1), "uint8", "cuda:3")}
    assert external_buffers(Note("another note"), {7: "cuda:3"}.get) == {}
