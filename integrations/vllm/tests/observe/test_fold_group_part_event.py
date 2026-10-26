"""SHARED-CODE (lane ov-moe): an instance is attributed to the first event its Compute claims, else the resolution's root.

Snapshots are keyed by the event that produced a tensor (`vllm_adapter._after_call` / `_after_launch`), and every consumer
(`compare.SnapshotStore.find(view, at=event)`, `replay.expected_words`, `vu_query`) looks a value up under the instance's
`event`.  Before this change every part of a D70 Group was stamped with the ROOT event, so the intermediates of a scope
such as `vllm.moe_forward` (router weights/ids, expert caches — snapshotted only under the nested kernels' event ids)
were invisible to tier A/B/chain although the capture holds them.

Invariant kept: a primary claims the whole subtree (min = the root) and `also`/`before` members claim nothing, so every
pre-existing pattern still stamps `root.id` — checked here on the toy profile of `test_fold_synthetic`."""

from __future__ import annotations

from verity_vllm.observe.fold.fold import fold
from verity_vllm.observe.fold.resolver import Compute, Gather, GatherMany, Group, Profile
from tests.observe import test_fold_synthetic as T
from tests.observe.test_fold_synthetic import memory  # noqa: F401  (the real memory model, or the stub while it is a skeleton)
from verity.ml.circuits.defs import bind
from verity_vllm.program.registry import b1


def _attention_parts_claiming_the_launch(node, ctx):
    """`test_fold_synthetic.p_attention` with each per-row part claiming the nested launch (the op that produced its output)."""
    if node.op != "vllm.unified_attention_with_output.default" or len(node.children) != 1:
        return None
    ch = node.children[0]
    q3, kc, vc, out = ch.args[:4]
    parts = []
    for i, (req, p) in enumerate(ctx.start.rows):
        Tn = p + 1
        slots = tuple(T._slot(ctx, req, j) for j in range(Tn))
        parts.append(Compute(bind(b1.AttentionV3, T=Tn, NH=T.C["NH"], KVH=T.C["KVH"], D=T.C["D"], BN=T.BN), 1,
                             (Gather(q3, (i,)), GatherMany(T._kv_flat(kc), (slots,)), GatherMany(T._kv_flat(vc), (slots,))),
                             (Gather(out, (i,)),), (ch.id,), node.enter.module or "", attribution=((req, p),)))
    return Group(tuple(parts), (node.id, ch.id), node.enter.module or "")


def _profile(attention):
    pats = [T.Pat(n, f) for n, f in (("embedding", T.p_embedding), ("rms_triton", T.p_rms_triton), ("mm", T.p_mm), ("rope", T.p_rope),
                                      ("kv_update", T.p_kv_update), ("attention", attention), ("fused_norm", T.p_fused_norm),
                                      ("silu", T.p_silu), ("index", T.p_index), ("argmax", T.p_argmax))]
    return Profile("toy", pats, dict(T.C), T.weight_field, False)


def _attention_events(result):
    return [(m["event"], m["step"]) for m in result.instances if m["spec"].startswith("Attention_v3")]


def _launch_and_root(log):
    """(root op id, nested launch id) of every `unified_attention_with_output` in the toy log, in order."""
    from verity_vllm.observe.log import split_steps
    from verity_vllm.observe.fold.tree import step_events
    out = []
    for se in (step_events(s) for _pre, s in split_steps(log) if s):
        for r in se.roots:
            if r.op == "vllm.unified_attention_with_output.default":
                out.append((r.id, r.children[0].id))
    return out


def test_group_parts_claiming_nothing_keep_the_root_event(memory):
    """The pre-existing spelling (parts with `consumed=()`, the Group consuming the subtree) is unchanged."""
    r = fold(T.toy_log(), T.toy_profile(), strict=True, with_program=False)
    pairs = _launch_and_root(T.toy_log())
    roots = {root for root, _ in pairs}
    launches = {ch for _, ch in pairs}
    ev = {e for e, _ in _attention_events(r)}
    assert ev and ev <= roots and not (ev & launches)
    # a primary that claims the whole subtree stamps the root as well (min of the subtree ids is the root)
    mm = [m for m in r.instances if m["spec"].startswith("Gemm_v1")]
    assert mm and all(isinstance(m["event"], int) for m in mm)


def test_group_parts_claiming_their_own_op_are_attributed_to_it(memory):
    r = fold(T.toy_log(), _profile(_attention_parts_claiming_the_launch), strict=True, with_program=False)
    pairs = _launch_and_root(T.toy_log())
    launches = {ch for _, ch in pairs}
    roots = {root for root, _ in pairs}
    ev = {e for e, _ in _attention_events(r)}
    assert ev and ev <= launches and not (ev & roots)


def test_attribution_does_not_change_the_program_digest(memory):
    """`event` is instance metadata: the folded Program (its digest) is identical under both spellings."""
    a = fold(T.toy_log(), T.toy_profile(), strict=True)
    b = fold(T.toy_log(), _profile(_attention_parts_claiming_the_launch), strict=True)
    assert a.digest == b.digest
