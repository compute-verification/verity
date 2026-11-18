"""The rows cut by key block as the window circuit cuts them (`serving_templates.KEY_BLOCKS`, `attention_cuts.units`): a
row's unit templates past itself are the Calls of the lowering's plan of its heads (`window_lowering._Program._plan`), each
times the row's heads, by name and count, whatever its length, phase or kBlockN."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from verity.circuits.call_scope import spec_base, spec_statics
from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind

from verity_vllm.definitions.numerics import fa3_model
from verity_vllm.definitions.registry import attention_cuts as AC
from verity_vllm.definitions.registry import attention_vocab, b1, hopper
from verity_vllm.definitions.registry import fa3_check_inf as F3
from verity_vllm.pipeline import serving_rows as SRW
from verity_vllm.program.predict import window_lowering as WL

#: Llama-3.2-1B's attention on the H200: heads, KV heads, head dim and the query heads a KV head
NH, KVH, D, G = 32, 8, 64, 4
FA3 = {"DOT": hopper.HopperBF16WgmmaDot16, "INV": hopper.Fa3InvSum}


def fa3_row(T: int, seqlen_q: int, row: int):
    """The FA3 attention row of T keys, query row ``row`` of a launch of ``seqlen_q`` query rows: its tile's kBlockN and the
    lowest key block it guards, as the target binds them (`targets.attention_spec`)."""
    BN, BM = fa3_model.kblock_n_for(D, seqlen_q, G), fa3_model.kblock_m_for(D, seqlen_q, G)
    return bind(F3.AttentionV4, T=T, NH=NH, KVH=KVH, D=D, BN=BN, MASKED_FROM=attention_vocab.fa3_masked_from(T, BN, row, BM, G), **FA3)


def planned(fn) -> list[tuple[str, int]]:
    """The lowering's plan of ``fn``'s heads: each Call's Definition with how many of it the row's heads make."""
    lowering = object.__new__(WL._Program)
    lowering.b = SimpleNamespace(plans={})
    out: dict[str, int] = {}
    for f, _ports in lowering._plan(fn, NH):
        out[descriptor_id(f)] = out.get(descriptor_id(f), 0) + NH
    return list(out.items())


@pytest.mark.parametrize("T, seqlen_q, row, bn", [
    (5, 159, 4, 128), (128, 159, 127, 128), (130, 159, 129, 128), (159, 159, 158, 128), (300, 300, 299, 128),
    (11, 11, 10, 192), (400, 16, 15, 192), (160, 1, 0, 192), (257, 1, 0, 192), (600, 1, 0, 192)])
def test_a_rows_units_are_the_lowerings_plan_of_its_heads(T, seqlen_q, row, bn):
    """Prefill rows (a query row of a launch of 159, 300, 11 or 16) and decode rows (one query row), one key block or
    several, at kBlockN 128 and 192, guarded below their first block or not."""
    fn = fa3_row(T, seqlen_q, row)
    spec = descriptor_id(fn)
    assert fn.bindings["BN"] == bn and spec_base(spec) in SRW.KEY_BLOCKS
    assert SRW.unit_templates(spec) == ((spec, 1), *planned(fn))


@pytest.mark.parametrize("T, seqlen_q, row, first, below", [
    (5, 159, 4, 5, []), (130, 159, 129, 2, [True]), (159, 159, 158, 31, [False]), (300, 300, 299, 44, [False, False]),
    (257, 1, 0, 65, [False]), (600, 1, 0, 24, [False] * 3)])
def test_a_rows_units_are_its_first_visited_block_the_blocks_below_it_and_its_end(T, seqlen_q, row, first, below):
    """The first visited block holds the keys past the last whole block; each block below it at kBlockN, guarded (``CHECK``)
    when at or above its tile's `MASKED_FROM` (130 keys: the tile's first query row is 96, in block 0); one end a head."""
    bn = 128 if seqlen_q > 16 else 192
    got = [(spec_base(t), spec_statics(t).get("NVIS"), spec_statics(t).get("CHECK"), n)
           for t, n in SRW.unit_templates(descriptor_id(fa3_row(T, seqlen_q, row)))[1:]]
    want = [("AttnBlockFirst_v1", first, None, NH), *(("AttnBlock_v4", bn, c, NH * below.count(c)) for c in dict.fromkeys(below)),
            ("AttnHeadEnd_v1", None, None, NH)]
    assert got == want


def test_the_lowering_and_the_census_read_one_cut_and_a_version_with_none_is_its_row():
    """`Attention_v2` (FA3 guarding every block) has no cut: no units, and its row is its one template."""
    assert WL.CUTS is AC.CUTS
    row = bind(b1.AttentionV2, T=5, NH=NH, KVH=KVH, D=D, BN=128, **FA3)
    with pytest.raises(KeyError):
        AC.units(row)
    assert SRW.unit_templates(descriptor_id(row)) == ((descriptor_id(row), 1),)
