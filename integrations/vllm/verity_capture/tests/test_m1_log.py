"""G0 acceptance of the M1 raw observation log (torch-free; skipped when data/logs/m1.jsonl.gz is absent).

Reads the whole log with `log.read_log`, splits steps with `log.split_steps`, builds trees with `tree.step_events`
for every step and checks the DESIGN §3 / G0 facts of the frozen workload (SmolLM2-135M, 1 request, 32 prompt
tokens, 16 greedy tokens, eager, batch-invariant, V2 model runner).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from verity_capture.events import (DispatchEnter, DispatchExit, Root, RunHeader, SampleEnd, StepEnd, StepLink, StepStart,
                                   StorageAlloc, StorageFree, TritonLaunch, ViewDesc, views_in)
from verity_capture.log import read_log, split_steps
from verity_capture.tree import op_histogram, step_events

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "data/logs/m1.jsonl.gz"
TOKENS = ROOT / "data/logs/m1/tokens.json"

# the accepted record out/scale/e8/live/V5-N32x16-1req-bex0-inc-sc1 (adapter/selective/ru/<req>/boundary.json)
ACCEPTED = [359, 253, 1123, 1183, 282, 253, 1772, 338, 553, 7527, 288, 11492, 351, 260, 4340, 1357]
REQ = "7cc7c3bb-adf5-58cd-87fc-9f1b8e7c34d6"
FA2_RE = re.compile(r"(_vllm_fa2_C\.|flash_attn|fa2|varlen_fwd)", re.I)

pytestmark = pytest.mark.skipif(not LOG.exists(), reason=f"{LOG} absent (run bench/m1_capture.py on the pod)")


@pytest.fixture(scope="module")
def log_events():
    return list(read_log(LOG))


@pytest.fixture(scope="module")
def steps(log_events):
    groups = [(pre, st) for pre, st in split_steps(log_events) if st]
    return [(pre, step_events(st)) for pre, st in groups]


def _count(hist: dict[str, int], pred) -> int:
    return sum(n for op, n in hist.items() if pred(op))


def test_header_and_step_count(log_events, steps):
    assert isinstance(log_events[0], RunHeader)
    hdr = log_events[0]
    assert hdr.schema == "verity-capture/log/v1"
    assert hdr.workload["requests"][0]["request_id"] == REQ
    assert len(steps) == 16
    s0 = steps[0][1].start
    assert s0.step == 0 and len(s0.rows) == 32 and s0.num_tokens == 32 and s0.phase_hint == "prefill"
    assert s0.input_token_ids == hdr.workload["requests"][0]["prompt_token_ids"]
    assert s0.positions == list(range(32)) and all(r == (REQ, p) for r, p in zip(s0.rows, range(32)))
    assert s0.logits_rows == [(REQ, 31)] and s0.logits_indices == [31]
    assert s0.block_size in (16, 32) and s0.block_tables[REQ] and len(s0.slot_mapping) == 32
    assert s0.num_computed[REQ] == 0 and s0.seq_lens[REQ] == 32
    assert isinstance(s0.input_ids_view, ViewDesc) and s0.input_ids_view.shape[0] >= 32
    for i, (_, se) in enumerate(steps[1:], start=1):
        st = se.start
        assert st.step == i and len(st.rows) == 1 and st.num_tokens == 1 and st.phase_hint == "decode"
        assert st.rows == [(REQ, 31 + i)] and st.positions == [31 + i]
        assert st.num_computed[REQ] == 31 + i and st.seq_lens[REQ] == 32 + i
        assert len(st.slot_mapping) == 1 and st.logits_indices == [0]
        assert se.end is not None and isinstance(se.end, StepEnd) and se.end.forward_s > 0


def test_prefill_roots_and_decode_links(steps):
    roots = [e for e in steps[0][1].other if isinstance(e, Root)]
    prompt = [r for r in roots if r.role == "prompt_token"]
    assert len(prompt) == 32
    s0 = steps[0][1].start
    for i, r in enumerate(prompt):
        assert r.values == [s0.input_token_ids[i]] and r.view.shape == (1,) and r.view.alloc == s0.input_ids_view.alloc
        assert r.view.offset == s0.input_ids_view.offset + i * s0.input_ids_view.strides[0]
    assert steps[0][1].links == []
    prev_sample = steps[0][1].sample
    assert prev_sample is not None
    for i, (_, se) in enumerate(steps[1:], start=1):
        assert len(se.links) == 1
        link = se.links[0]
        assert isinstance(link, StepLink) and link.step == i and link.row == 0 and link.request_id == REQ
        assert link.src_step == i - 1 and link.relation == "eq"
        assert link.value == prev_sample.sampled_token_ids[0] == se.start.input_token_ids[0]
        prev_sample = se.sample
        assert prev_sample is not None and prev_sample.step == i


def test_sampled_tokens_equal_the_accepted_record(steps):
    sampled = [se.sample.sampled_token_ids[0] for _, se in steps]
    assert sampled == ACCEPTED
    if TOKENS.exists():
        doc = json.loads(TOKENS.read_text())
        req = doc["requests"][0]
        assert req["request_id"] == REQ and req["generated_token_ids"] == ACCEPTED
        assert doc["per_step_sampled"][str(0)] == [ACCEPTED[0]]


def test_sample_end_views(steps):
    for _, se in steps:
        s = se.sample
        assert isinstance(s.logits_view, ViewDesc) and s.logits_view.shape[-1] == 49152
        assert isinstance(s.sampled_view, ViewDesc) and s.sampled_view.dtype in ("int64", "int32")
        assert len(s.sampled_token_ids) == len(se.start.logits_rows) == 1


def test_every_triton_launch_has_viewdescs_for_pointer_args(steps):
    n = 0
    for _, se in steps:
        for node in se.all_nodes():
            if not node.is_triton:
                continue
            n += 1
            ev: TritonLaunch = node.enter
            ptr_args = {k: v for k, v in ev.args.items() if k.endswith("_ptr") or k.endswith("_ptrs")}
            assert ptr_args, (ev.fn, list(ev.args))
            views = [k for k, v in ptr_args.items() if isinstance(v, ViewDesc)]
            assert views, (ev.fn, ptr_args)
            for k, v in ptr_args.items():
                assert isinstance(v, ViewDesc) or v is None, (ev.fn, k, v)
            assert ev.src_sha and len(ev.src_sha) == 64 and isinstance(ev.grid, list) and ev.grid
            assert ev.num_warps and ev.num_warps > 0
            assert isinstance(ev.constexprs, dict)
    assert n > 16 * 121


def test_batch_invariant_kernels_carry_pointers_and_block_constexprs(steps):
    seen_mm = seen_rms = False
    for _, se in steps:
        for node in se.all_nodes():
            if not node.is_triton:
                continue
            ev = node.enter
            if ev.fn.endswith(".matmul_kernel_persistent"):
                seen_mm = True
                for k in ("a_ptr", "b_ptr", "c_ptr"):
                    assert isinstance(ev.args[k], ViewDesc), (k, ev.args.get(k))
                assert {"BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"} <= set(ev.constexprs), ev.constexprs
                assert ev.args["a_ptr"].dtype == "bfloat16"
            elif ev.fn.endswith("._rms_norm_kernel"):
                seen_rms = True
                ptrs = [v for k, v in ev.args.items() if k.endswith("_ptr")]
                assert len(ptrs) >= 3 and all(isinstance(p, ViewDesc) for p in ptrs), ev.args
                assert any(k.startswith("BLOCK") for k in ev.constexprs), ev.constexprs
    assert seen_mm and seen_rms


def test_reentered_custom_ops_have_their_children(steps):
    for i, (_, se) in enumerate(steps):
        attn = [n for n in se.all_nodes() if n.op.startswith("vllm.unified_attention_with_output")]
        kvu = [n for n in se.all_nodes() if n.op.startswith("vllm.unified_kv_cache_update")]
        assert len(attn) == 30, (i, len(attn))
        if kvu:
            assert len(kvu) == 30, (i, len(kvu))
        for n in attn:
            ops = [c.op for c in n.walk() if c is not n]
            assert any(FA2_RE.search(op) for op in ops), (i, ops[:20])
        cache_parents = kvu or attn
        for n in cache_parents:
            ops = [c.op for c in n.walk() if c is not n]
            # the pinned wheel registers the cache op in the `_C_cache_ops` namespace (not `_C`)
            assert any(op.endswith(".reshape_and_cache_flash.default") for op in ops), (i, n.op, ops[:20])
        # the children carry the outer call as parent, in the same step and phase
        for n in attn + kvu:
            for c in n.children:
                assert c.enter.parent == n.id and c.enter.step == n.enter.step and c.enter.phase == "forward"


def test_decode_step_op_counts(steps):
    for i, (_, se) in enumerate(steps[1:], start=1):
        hist = op_histogram(se.roots)
        assert _count(hist, lambda op: op.endswith(".matmul_kernel_persistent")) >= 121, (i, hist)
        assert _count(hist, lambda op: op.endswith("._rms_norm_kernel")) >= 1, (i, hist)
        assert hist.get("_C.fused_add_rms_norm.default", 0) >= 60, (i, hist)
        assert hist.get("_C.rotary_embedding.default", 0) >= 30, (i, hist)
        assert hist.get("_C.silu_and_mul.default", 0) >= 30, (i, hist)
        assert hist.get("aten.embedding.default", 0) >= 1, (i, hist)
        assert hist.get("aten.argmax.default", 0) >= 1, (i, hist)
        phases = {n.phase for n in se.all_nodes()}
        assert "forward" in phases and "sample" in phases
        assert all(n.phase == "sample" for n in se.all_nodes() if n.op == "aten.argmax.default")


def test_prefill_step_op_counts(steps):
    hist = op_histogram(steps[0][1].roots)
    assert _count(hist, lambda op: op.endswith(".matmul_kernel_persistent")) >= 121
    assert hist.get("_C.fused_add_rms_norm.default", 0) >= 60 and hist.get("_C.rotary_embedding.default", 0) >= 30


def test_every_view_alloc_has_a_preceding_storage_alloc(log_events):
    known: set[int] = set()
    for ev in log_events:
        if isinstance(ev, StorageAlloc):
            assert ev.alloc not in known or True     # re-allocation after a free gets a NEW id; ids never repeat
            known.add(ev.alloc)
            continue
        refs: list[ViewDesc] = []
        if isinstance(ev, DispatchEnter):
            refs = list(views_in(ev.args)) + list(views_in(ev.kwargs))
        elif isinstance(ev, DispatchExit):
            refs = list(views_in(ev.outs))
        elif isinstance(ev, TritonLaunch):
            refs = list(views_in(ev.args))
        elif isinstance(ev, Root):
            refs = [ev.view]
        elif isinstance(ev, StepStart):
            refs = [v for v in (ev.input_ids_view, ev.positions_view) if v is not None]
        elif isinstance(ev, SampleEnd):
            refs = [v for v in (ev.logits_view, ev.sampled_view) if v is not None]
        for v in refs:
            assert v.alloc in known, (type(ev).__name__, v)


def test_alloc_ids_are_unique(log_events):
    ids = [ev.alloc for ev in log_events if isinstance(ev, StorageAlloc)]
    assert len(ids) == len(set(ids))


def test_every_storage_free_follows_its_alloc(log_events):
    known: set[int] = set()
    freed: set[int] = set()
    for ev in log_events:
        if isinstance(ev, StorageAlloc):
            known.add(ev.alloc)
        elif isinstance(ev, StorageFree):
            assert ev.alloc in known and ev.alloc not in freed, ev
            freed.add(ev.alloc)


def test_weights_and_rotary_cache_are_roots(log_events):
    roots = [ev for ev in log_events if isinstance(ev, Root)]
    weights = [r for r in roots if r.role == "weight"]
    buffers = [r for r in roots if r.role == "buffer"]
    names = {r.name for r in weights}
    assert "model.embed_tokens.weight" in names and "model.layers.0.self_attn.qkv_proj.weight" in names
    assert "model.layers.29.mlp.down_proj.weight" in names and "model.norm.weight" in names
    assert any("rotary_emb.cos_sin_cache" in r.name for r in buffers), [r.name for r in buffers][:10]
    # 30 x (qkv_proj, o_proj, gate_up_proj, down_proj, input_layernorm, post_attention_layernorm) + embed_tokens + norm
    # (lm_head is tied to embed_tokens)
    assert len(weights) == 30 * 6 + 2
    assert len(buffers) >= 1


def test_trees_are_well_formed(steps):
    for _, se in steps:
        for n in se.all_nodes():
            if not n.is_triton:
                assert n.exit is not None, n.enter.op
            for c in n.children:
                assert c.enter.parent == n.id
