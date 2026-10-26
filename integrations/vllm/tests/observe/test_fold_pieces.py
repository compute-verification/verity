"""`Pieces` (row i reads the concatenation of its pieces) and the shared `Const` (one instance per value in the resolution), on the
toy profile of `test_fold_synthetic`: a read split into one piece per KV slot is the same Program as `GatherMany` of the slots, a
shared constant repeated across rows and parts is one instance per value and resolution (a piece of no copies too), and a piece's
unshared constant is refused."""

from __future__ import annotations

import pytest

from verity.ml.circuits.defs import bind
from verity_vllm.observe.fold.fold import FoldError, fold
from verity_vllm.observe.fold.resolver import Compute, Const, Gather, Group, Pieces, Profile
from verity_vllm.observe.fold.views import row_view
from verity.ml.definitions.registry import b1
from tests.observe import test_fold_synthetic as T
from tests.observe.test_fold_synthetic import memory  # noqa: F401  (the real memory model, or the stub while it is a skeleton)


def _attention(v_operand):
    """`test_fold_synthetic.p_attention` with K read as one piece per slot and V as `v_operand(view, slots)`."""
    def match(node, ctx):
        if node.op != "vllm.unified_attention_with_output.default" or len(node.children) != 1:
            return None
        ch = node.children[0]
        q3, kc, vc, out = ch.args[:4]
        k, v, parts = T._kv_flat(kc), T._kv_flat(vc), []
        for i, (req, p) in enumerate(ctx.start.rows):
            slots = tuple(T._slot(ctx, req, j) for j in range(p + 1))
            parts.append(Compute(bind(b1.AttentionV3, T=p + 1, NH=T.C["NH"], KVH=T.C["KVH"], D=T.C["D"], BN=T.BN), 1,
                                 (Gather(q3, (i,)), Pieces((tuple(row_view(k, s) for s in slots),)), v_operand(v, slots)),
                                 (Gather(out, (i,)),), (), node.enter.module or "", attribution=((req, p),)))
        return Group(tuple(parts), (node.id, ch.id), node.enter.module or "")
    return match


def _profile(attention):
    pats = [T.Pat(n, attention if n == "attention" else p.fn) for n, p in ((p.name, p) for p in T.PATTERNS)]
    return Profile("toy", pats, dict(T.C), T.weight_field, False)


def _zeros(v, slots):
    return Pieces(((Const(16, 0, "v0", shared=True, count=len(slots) * T.C["KVH"] * T.C["D"]),),))


def test_pieces_of_slot_rows_are_the_program_gather_many_of_the_slots_is(memory):
    from verity_vllm.observe.fold.resolver import GatherMany
    a = fold(T.toy_log(), T.toy_profile(), strict=True)
    b = fold(T.toy_log(), _profile(_attention(lambda v, slots: GatherMany(v, (slots,)))), strict=True)
    c = fold(T.toy_log(), _profile(_attention(lambda v, slots: Pieces((tuple(row_view(v, s) for s in slots),)))), strict=True)
    assert a.digest == b.digest == c.digest


def test_a_shared_constant_is_one_instance_per_value_in_the_resolution(memory):
    r = fold(T.toy_log(), _profile(_attention(_zeros)), strict=True, with_program=False)
    attn = {m["resolution"] for m in r.instances if m["spec"].startswith("Attention_v3")}
    consts = [m["resolution"] for m in r.instances if m.get("const") and m["const"]["bits"] == "0x0000"]
    assert attn and sorted(consts) == sorted(attn), "one zero per attention resolution, whatever its rows and copies"


def test_a_shared_constant_of_no_copies_is_still_its_instance(memory):
    def padded(v, slots):
        return Pieces(((*_zeros(v, slots).rows[0], Const(16, 0xFFFF, "pad", shared=True, count=0)),))
    r = fold(T.toy_log(), _profile(_attention(padded)), strict=True, with_program=False)
    attn = {m["resolution"] for m in r.instances if m["spec"].startswith("Attention_v3")}
    pads = [m["resolution"] for m in r.instances if m.get("const") and m["const"]["bits"] == "0xffff"]
    assert attn and sorted(pads) == sorted(attn), "a Build rule's [c] * 0 still calls c"


def test_a_pieces_constant_that_is_not_shared_is_refused(memory):
    def unshared(v, slots):
        return Pieces(((Const(16, 0, "v0", count=len(slots) * T.C["KVH"] * T.C["D"]),),))
    with pytest.raises(FoldError, match="not a shared constant"):
        fold(T.toy_log(), _profile(_attention(unshared)), strict=True, with_program=False)
