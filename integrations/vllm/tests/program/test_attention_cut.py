"""The key-block cut: `dead_free.AttentionHead_v21` / `Attention_v21` compute v18's function and `fa3_check_inf.AttentionHead_v22` /
`Attention_v22` v4's, word for word, each key block its softmax step and its P·V groups (`AttnBlockSoftmax_v1`, `_v2`,
`AttnBlockPv_v1`) a Call each; and with those Calls and `AttnHeadEnd_v1` written in place each is its function's circuit node for
node (v18's with `AttnBlock_v5` written in place, v20's with `AttnBlockFirst_v1`, `AttnBlock_v4` and `AttnHeadEnd_v1`), at
Llama-3.2-1B's heads over kBlockN 192, the value GEMM's slices of 32 columns and groups of 96 keys.  A constant (a gate of no
operand, such as a zero word) is compared by its Definition, so a zero emitted in another step of the same function is the same
leaf.
"""
from __future__ import annotations

import pytest
from tests.program.test_dead_free import FA2, RNG, _attention_cases, _bf16, _dead, _same

from verity.ml.circuits.defs import bind
from verity.ml.definitions.registry import dead_free as DF, fa2_check_inf as F2, fa3_check_inf as F3, hopper

FA3 = dict(DOT=hopper.HopperBF16WgmmaDot16, INV=hopper.Fa3InvSum)
#: the Calls a cut head's body is, and the steps the old heads Call
CUT_STEPS = frozenset({"AttnBlockSoftmax_v1", "AttnBlockSoftmax_v2", "AttnBlockPv_v1", "AttnHeadEnd_v1"})


def _inlined(fn, inline: frozenset[str]) -> tuple[list[tuple], list[tuple]]:
    """``fn``'s nodes with the Calls of the families ``inline`` names written in place, each (form, Definition id, axes, n, n_xs,
    operands), every operand leaf a parameter leaf of ``fn``, a constant's (a primitive of no operand, by its id) or an output leaf
    of an earlier node here; and the return's leaves.  Equal for two Definitions, they are one circuit."""
    nodes: list[tuple] = []

    def run(body, params: list) -> list:
        outs: list = []

        def src(r):
            s, i, j = r
            return params[i][j] if s == "p" else outs[i][j]

        for node in body.nodes:
            args = [tuple(map(src, a.refs)) for a in node.args]
            if node.form == "call" and node.fn.base_id in inline:
                outs.append(run(node.fn.body, args))
            elif node.form == "primitive" and not args:
                outs.append([("c", node.fn.id, j) for j in range(node.out.leaves)])
            else:
                nodes.append((node.form, node.fn.id, node.axes, node.n, node.n_xs, args))
                outs.append([("n", len(nodes) - 1, j) for j in range(node.out.leaves)])
        return [src(r) for r in body.ret]

    ret = run(fn.body, [[("p", i, j) for j in range(t.leaves)] for i, (_, t) in enumerate(fn.body.params)])
    return nodes, ret


@pytest.mark.parametrize("T,BN,VS,PK,MF", [(1, 16, 4, 16, 0), (15, 32, 8, 16, 0), (16, 32, 16, 32, 0), (17, 32, 4, 16, 0),
                                           (40, 32, 8, 16, 1), (48, 48, 4, 32, 0), (97, 48, 16, 48, 2), (100, 32, 4, 32, 1),
                                           (65, 16, 8, 16, 3)])
def test_attention_heads_v21_and_v22_equal_v18_and_v4(T, BN, VS, PK, MF):
    """A first block of 1 ... BN keys, its last group short of a k-step or not, one group a block or several (the last short of
    PK at PK 32 over BN 48), blocks below it guarded and not."""
    for new, old, kw in ((DF.AttentionHeadV21, DF.AttentionHeadV18, FA2), (F3.AttentionHeadV22, F3.AttentionHeadV4, FA3)):
        new, old = bind(new, T=T, D=16, BN=BN, MASKED_FROM=MF, VS=VS, PK=PK, **kw), bind(old, T=T, D=16, BN=BN, MASKED_FROM=MF, **kw)
        assert _dead(new) == 0
        for q, k, v in _attention_cases(16, T):
            _same(new, old, q, k, v)


def test_attentions_v21_and_v22_equal_v18_and_v4():
    for new, old, kw in ((DF.AttentionV21, DF.AttentionV18, FA2), (F3.AttentionV22, F3.AttentionV4, FA3)):
        new, old = (bind(fn, T=48, NH=4, KVH=2, D=16, BN=32, MASKED_FROM=1, **kw) for fn in (bind(new, VS=8, PK=16), old))
        for _ in range(2):
            q, kc, vc = (_bf16(RNG.normal(0, 1, n)) for n in (4 * 16, 48 * 2 * 16, 48 * 2 * 16))
            _same(new, old, q, kc, vc)


@pytest.mark.parametrize("T", [1, 16, 17, 95, 96, 97, 191, 192, 193, 288, 384, 385, 576])
def test_attention_heads_v21_and_v22_are_their_functions_with_calls_inlined_at_llama_geometry(T):
    """Llama-3.2-1B's heads (D 64) over kBlockN 192, a key's value 2 slices of 32 columns, groups of 96 keys: each cut head's body
    is Calls only, a softmax and its groups a block and the end, and with them written in place it is v18's circuit (`AttnBlock_v5`
    in place) and v20's (`AttnBlockFirst_v1`, `AttnBlock_v4` and `AttnHeadEnd_v1` in place) node for node, at every MASKED_FROM's
    extremes."""
    D, BN, VS, PK = 64, 192, 32, 96
    nb = -(-T // BN)
    calls = 1 + len(F2.pv_groups(T - (nb - 1) * BN, PK)) + (nb - 1) * (1 + BN // PK) + 1
    for MF in {0, nb - 1}:
        for new, old, kw, steps in ((DF.AttentionHeadV21, DF.AttentionHeadV18, FA2, {"AttnBlock_v5"}),
                                    (F3.AttentionHeadV22, F3.AttentionHeadV20, FA3,
                                     {"AttnBlockFirst_v1", "AttnBlock_v4", "AttnHeadEnd_v1"})):
            new = bind(new, T=T, D=D, BN=BN, MASKED_FROM=MF, VS=VS, PK=PK, **kw)
            old = bind(old, T=T, D=D, BN=BN, MASKED_FROM=MF, **kw)
            assert {(n.form, n.fn.base_id) for n in new.body.nodes} <= {("call", f) for f in CUT_STEPS}
            assert len(new.body.nodes) == calls
            assert _inlined(new, CUT_STEPS) == _inlined(old, frozenset(steps))


def test_the_cut_refuses_what_its_steps_refuse():
    for kw in ({"VS": 24}, {"PK": 40}):
        with pytest.raises(ValueError, match="refuses this binding"):
            bind(DF.AttentionHeadV21, T=48, D=64, BN=48, MASKED_FROM=0, **{"VS": 32, "PK": 16, **kw}, **FA2)
    with pytest.raises(ValueError, match="MASKED_FROM 1 outside blocks 0..0"):
        _ = bind(F3.AttentionHeadV22, T=48, D=16, BN=48, MASKED_FROM=1, VS=4, PK=16, **FA3).body
