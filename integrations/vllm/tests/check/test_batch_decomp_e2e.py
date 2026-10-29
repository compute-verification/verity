"""X-09 / MB-01 end-to-end: the batch decomposition of a batched eager fold, CPU, bit-exact on the
synthetic toy configuration of `tests/observe/test_fold_synthetic.py` (real `b1` definitions, hand-written raw
events), through the real fold -> `persist_program` record -> `batch_decomp.decompose` path (the log is written with
`log.dumps`, as the capture does).

Oracle of the tests: request r's projection of the B-request fold must be canonical-equal (`cb_a.compare.compare_steps`)
to the fold of the SAME request served alone (B = 1) -- exactly what the per-request request-wrapper Program is compared
with at B = 1 today (the single-request fold stands in for `derive_step`'s output).  Positives: B = 2 with unequal LP and
unequal T; a request finishing early (no rows in later steps); B = 1 is the identity.  Negatives: a mis-attributed row
-> not equal; a cross-request operand (r1's attention reading r0's KV row) -> XREQ, FAIL; a missing derived shape ->
MISSING_DERIVED, never assumed.  An independent check of X-09."""

from __future__ import annotations

import functools
import gzip
import itertools
import json

import pytest

from verity_vllm.pipeline.match import persist_program
from verity_vllm.observe.events import (DispatchEnter, DispatchExit, Root, RunHeader, SampleEnd, StepEnd, StepLink, StepStart, StorageAlloc,
                                   TritonLaunch, ViewDesc)
from verity_vllm.check.match import batch_decomp as bd
from verity_vllm.check.match.program_compare import compare_steps, load_oracle
from verity_vllm.observe.log import dumps as log_dumps
from verity_vllm.observe.fold.fold import fold
from verity_vllm.observe.fold.resolver import Compute, Gather, GatherMany, Group, PerRow, Profile, Shared, Witness
from tests.observe import test_fold_synthetic as toy
from tests.observe.test_fold_synthetic import (A_COS, A_DOWN_W, A_EMBED, A_GU_W, A_LN1, A_LN2, A_NORM, A_O_W, A_QKV_W, BF, BN, C, I32D, I64D,
                                                      W_COS, W_DOWN, W_EMBED, W_GU, W_LN1, W_LN2, W_NORM, W_O, W_QKV, Ids, V, _as3, _kv_flat,
                                                      _slot, memory)  # noqa: F401 - `memory` is the fixture
from verity.ml.circuits.defs import bind
from verity.ml.definitions.registry import b1
from verity.ml.definitions.registry import value_inputs as VI

BS = 4
A_IDS, A_POS, A_KC, A_VC = 10, 11, 12, 13
KC = V(A_KC, (4, BS, 1, 8))
VC = V(A_VC, (4, BS, 1, 8))


# ---------------------------------------------------------------------------------------------------------- toy log, B requests
# request spec: {"id": str, "prompt": [tokens], "sampled": [tokens]} — len(sampled) = served samples (prefill + decodes);
# the request occupies steps 0 .. len(sampled)-1 and has no rows afterwards (finished).


def preamble(reqs):
    hdr = RunHeader("verity-capture/log/v1", "2026-09-19T00:00:00Z", "toy-profile", {"vllm": "toy"}, {"repo": "toy", "config": C},
                    {"requests": [{"request_id": r["id"], "prompt_token_ids": list(r["prompt"]), "max_tokens": len(r["sampled"])} for r in reqs],
                     "sampling": {"temperature": 0.0}})
    evs = [hdr]
    weights = [(A_EMBED, W_EMBED, "model.embed_tokens.weight"), (A_LN1, W_LN1, "model.layers.0.input_layernorm.weight"),
               (A_QKV_W, W_QKV, "model.layers.0.self_attn.qkv_proj.weight"), (A_O_W, W_O, "model.layers.0.self_attn.o_proj.weight"),
               (A_LN2, W_LN2, "model.layers.0.post_attention_layernorm.weight"), (A_GU_W, W_GU, "model.layers.0.mlp.gate_up_proj.weight"),
               (A_DOWN_W, W_DOWN, "model.layers.0.mlp.down_proj.weight"), (A_NORM, W_NORM, "model.norm.weight")]
    for a, v, name in weights:
        evs.append(StorageAlloc(a, 0x1000 * a, v.numel * 2, "cuda:0", 0, "param"))
        evs.append(Root(v, "weight", name))
    evs.append(Root(W_EMBED, "weight", "lm_head.weight"))
    evs.append(StorageAlloc(A_COS, 0x1000 * A_COS, W_COS.numel * 2, "cuda:0", 0, "buffer"))
    evs.append(Root(W_COS, "buffer", "model.layers.0.self_attn.rotary_emb.cos_sin_cache"))
    for a, nbytes in ((A_IDS, 4 * 64), (A_POS, 8 * 64), (A_KC, KC.numel * 2), (A_VC, VC.numel * 2)):
        evs.append(StorageAlloc(a, 0x1000 * a, nbytes, "cuda:0", 0, "op"))
    return evs


def _tok(r, p):
    return r["prompt"][p] if p < len(r["prompt"]) else r["sampled"][p - len(r["prompt"])]


def step_events(step, reqs, active, ids, base_alloc):
    """One engine step: every active request contributes its rows (step 0: the whole prompt; later: one decode row)."""
    rows, toks, links, roots, logits_rows, logits_idx, slots, seq_lens, num_computed, block_tables, sampled = [], [], [], [], [], [], [], {}, {}, {}, []
    for ri, r in enumerate(reqs):
        if ri not in active:
            continue
        lp = len(r["prompt"])
        positions = list(range(lp)) if step == 0 else [lp + step - 1]
        bt = [2 * ri, 2 * ri + 1]  # two private blocks per request
        block_tables[r["id"]] = bt
        for p in positions:
            if step > 0:
                links.append(StepLink(step, len(rows), r["id"], step - 1, "eq", r["sampled"][step - 1]))
            rows.append((r["id"], p))
            toks.append(_tok(r, p))
            slots.append(bt[p // BS] * BS + p % BS)
        if step == 0:
            roots.append((len(rows) - lp, lp, r))
        logits_rows.append((r["id"], positions[-1]))
        logits_idx.append(len(rows) - 1)
        seq_lens[r["id"]] = positions[-1] + 1
        num_computed[r["id"]] = positions[0]
        sampled.append(r["sampled"][step])
    M, R = len(rows), len(logits_rows)
    a = itertools.count(base_alloc)
    pos_view = V(A_POS, (M,), dtype=I64D)
    hidden, normed, qkv, attn, o, gu, act, down, gathered, logits, sampled_v = (V(next(a), s, dtype=d) for s, d in (
        ((M, 16), BF), ((M, 16), BF), ((M, 32), BF), ((M, 2, 8), BF), ((M, 16), BF), ((M, 32), BF), ((M, 16), BF), ((M, 16), BF),
        ((R, 16), BF), ((R, 16), BF), ((R,), I64D)))
    ids_rows = ViewDesc(A_IDS, 0, (M,), (1,), I32D, "cuda:0")
    evs = [StepStart(step, rows, toks, [p for _, p in rows], seq_lens, num_computed, block_tables, BS, slots, logits_rows, logits_idx,
                     R, M, M, ids_rows, pos_view, "prefill" if step == 0 else "decode")]
    for lo, lp, r in roots:
        evs.append(Root(ViewDesc(A_IDS, lo, (lp,), (1,), I32D, "cuda:0"), "prompt_token", f"prompt[{r['id']}]", list(r["prompt"])))
    evs += links
    for v in (hidden, normed, qkv, attn, o, gu, act, down, gathered, logits, sampled_v):
        evs.append(StorageAlloc(v.alloc, 0x10000 + v.alloc * 0x100, v.numel * v.itemsize, "cuda:0", 0, "op"))

    def dispatch(op, args, kwargs, outs, module=None, parent=None, phase="forward", children=()):
        i = ids.next()
        evs.append(DispatchEnter(i, parent, step, phase, op, f"{op}(...)", args, kwargs, 7, module))
        for ch in children:
            ch(i)
        evs.append(DispatchExit(i, outs))
        return i

    def triton(fn, args, parent=None, module=None):
        i = ids.next()
        consts = {k: v for k, v in args.items() if isinstance(v, int)}
        evs.append(TritonLaunch(i, parent, step, "forward", fn, "sha", args, consts, [1, 1, 1], 4, 3, 7, module))
        return i

    def mm(a_view, w_alloc, N, K, c_view, module):
        bT = ViewDesc(w_alloc, 0, (K, N), (1, K), BF, "cuda:0")
        dispatch("aten.mm.default", [a_view, bT], {}, c_view, module,
                 children=[lambda p: triton("vllm.model_executor.layers.batch_invariant.matmul_kernel_persistent",
                                            {"a_ptr": a_view, "b_ptr": bT, "c_ptr": c_view, "M": a_view.shape[0], "N": N, "K": K, "BLOCK_K": 16}, p, module)])

    dispatch("aten.embedding.default", [W_EMBED, ids_rows], {}, hidden, "model.embed_tokens")
    triton("vllm.model_executor.layers.batch_invariant._rms_norm_kernel",
           {"input_ptr": hidden, "weight_ptr": W_LN1, "output_ptr": normed, "input_row_stride": 16, "n_cols": 16, "eps": 1e-5, "BLOCK_SIZE": 1024},
           None, "model.layers.0.input_layernorm")
    mm(normed, A_QKV_W, 32, 16, qkv, "model.layers.0.self_attn.qkv_proj")
    q = ViewDesc(qkv.alloc, 0, (M, 16), (32, 1), BF, "cuda:0")
    k = ViewDesc(qkv.alloc, 16, (M, 8), (32, 1), BF, "cuda:0")
    v = ViewDesc(qkv.alloc, 24, (M, 8), (32, 1), BF, "cuda:0")
    dispatch("_C.rotary_embedding.default", [pos_view, q, k, 8, W_COS, True], {}, None, "model.layers.0.self_attn.rotary_emb")
    dispatch("vllm.unified_kv_cache_update.default", [k, v, "model.layers.0.self_attn.attn"], {}, None, "model.layers.0.self_attn.attn",
             children=[lambda p: dispatch("_C.reshape_and_cache_flash.default", [k, v, KC, VC, pos_view, "auto", None, None], {}, None,
                                          "model.layers.0.self_attn.attn", parent=p)])
    q3 = ViewDesc(qkv.alloc, 0, (M, 2, 8), (32, 8, 1), BF, "cuda:0")
    dispatch("vllm.unified_attention_with_output.default", [q3, k, v, attn, "model.layers.0.self_attn.attn"], {}, None, "model.layers.0.self_attn.attn",
             children=[lambda p: dispatch("_vllm_fa2_C.varlen_fwd.default", [q3, KC, VC, attn], {"causal": True}, [attn], "model.layers.0.self_attn.attn", parent=p)])
    attn2 = ViewDesc(attn.alloc, 0, (M, 16), (16, 1), BF, "cuda:0")
    mm(attn2, A_O_W, 16, 16, o, "model.layers.0.self_attn.o_proj")
    dispatch("_C.fused_add_rms_norm.default", [o, hidden, W_LN2, 1e-5], {}, None, "model.layers.0.post_attention_layernorm")
    mm(o, A_GU_W, 32, 16, gu, "model.layers.0.mlp.gate_up_proj")
    dispatch("_C.silu_and_mul.default", [act, gu], {}, None, "model.layers.0.mlp.act_fn")
    mm(act, A_DOWN_W, 16, 16, down, "model.layers.0.mlp.down_proj")
    dispatch("_C.fused_add_rms_norm.default", [down, hidden, W_NORM, 1e-5], {}, None, "model.norm")
    dispatch("aten.index.default", [down, [ViewDesc(A_POS, 0, (R,), (1,), I64D, "cuda:0")]], {}, gathered, "model")
    mm(gathered, A_EMBED, 16, 16, logits, "lm_head")
    dispatch("aten.argmax.default", [logits], {"dim": -1}, sampled_v, None, phase="sample")
    evs.append(SampleEnd(step, logits, sampled, sampled_v))
    evs.append(StepEnd(step))
    return evs


def batched_log(reqs):
    ids = Ids()
    evs = preamble(reqs)
    n_steps = max(len(r["sampled"]) for r in reqs)
    for s in range(n_steps):
        active = {ri for ri, r in enumerate(reqs) if s < len(r["sampled"])}
        evs += step_events(s, reqs, active, ids, 20 + 40 * s)
    return evs


# ------------------------------------------------------------------------------------------------- patterns (toy + a leaking attention)


def p_attention_leak(node, ctx):
    """Non-separable variant: every request's attention also gathers the KV row of the FIRST request's position 0."""
    if node.op != "vllm.unified_attention_with_output.default" or len(node.children) != 1:
        return None
    ch = node.children[0]
    q3, kc, vc, out = ch.args[:4]
    first = sorted(ctx.start.block_tables)[0]
    parts = []
    for i, (req, p) in enumerate(ctx.start.rows):
        T = p + 1
        slots = tuple(_slot(ctx, req, j) for j in range(T))
        if req != first:
            slots = (_slot(ctx, first, 0),) + slots[1:]  # r1 reads r0's KV row instead of its own first row
        parts.append(Compute(bind(b1.AttentionV3, T=T, NH=C["NH"], KVH=C["KVH"], D=C["D"], BN=BN), 1,
                             (Gather(q3, (i,)), GatherMany(_kv_flat(kc), (slots,)), GatherMany(_kv_flat(vc), (slots,))), (Gather(out, (i,)),),
                             (), node.enter.module or "", attribution=((req, p),)))
    return Group(tuple(parts), (node.id, ch.id), node.enter.module or "")


def p_rms_witness(node, ctx, borrow=False):
    """The residual-free norm reading its epsilon word as each row's next `schedule` witness word: a request-owned witness read, as the
    Marlin align reads its keys.  `borrow`: the step's last row reads the first row's request's witness instead of its own."""
    if not node.is_triton or not node.op.endswith("_rms_norm_kernel"):
        return None
    a = node.enter.args
    keys = [req for req, _ in ctx.start.rows]
    if borrow:
        keys[-1] = keys[0]
    return Compute(bind(VI.RMSNormTritonV3, N=C["H"]), a["input_ptr"].shape[0],
                   (PerRow(a["input_ptr"]), Shared(a["weight_ptr"]), Witness("schedule", tuple(keys))), (PerRow(a["output_ptr"]),), (node.id,),
                   node.enter.module or "")


def profile(leak=False, witness=None):
    """The toy profile; `witness` "own" or "borrow" puts the norm's epsilon in the `schedule` witness (`p_rms_witness`)."""
    pats = [p for p in toy.PATTERNS if not (leak and p.name == "attention")]
    if leak:
        pats.append(toy.Pat("attention_leak", p_attention_leak))
    if witness:
        rms = functools.partial(p_rms_witness, borrow=witness == "borrow")
        pats = [toy.Pat(p.name, rms) if p.name == "rms_triton" else p for p in pats]
    return Profile("toy", pats, dict(C), toy.weight_field, False, witnesses=("schedule",) if witness else ())


# ------------------------------------------------------------------------------------------------------------------- helpers

R0 = {"id": "r0", "prompt": [5, 7, 2], "sampled": [9, 3, 1]}      # LP 3, 3 samples (T = 2 decode steps)
R1 = {"id": "r1", "prompt": [4, 6], "sampled": [8, 11]}           # LP 2, 2 samples (T = 1)
R1_SHORT = {"id": "r1", "prompt": [4, 6], "sampled": [8]}         # finishes at the prefill sample


def persist(result, events, d):
    """What the Match stage leaves behind: program.json + instances.jsonl (persist_program) and capture/log.jsonl.gz."""
    d.mkdir(parents=True, exist_ok=True)
    persist_program(result, d, profile_name="toy", log_path="toy")
    (d / "capture").mkdir(exist_ok=True)
    with gzip.open(d / "capture" / "log.jsonl.gz", "wt") as f:
        f.write(log_dumps(events))
    return str(d)


def fold_alone(req, d, leak=False, witness=None):
    """The request served alone = what the derived request-wrapper Program for (LP_r, T_r) describes."""
    evs = batched_log([req])
    r = fold(evs, profile(leak, witness), strict=True)
    p = persist(r, evs, d)
    json.dump({"structural_inputs": {"LP": len(req["prompt"]), "T": len(req["sampled"]) - 1}}, open(f"{p}/construction_manifest.json", "w"))
    return p


def fold_batched(reqs, d, leak=False, witness=None):
    evs = batched_log(reqs)
    r = fold(evs, profile(leak, witness), strict=True)
    assert r.report.unsupported == 0 and not r.errors
    return persist(r, evs, d), r


@pytest.fixture
def derived_is_a_fold(monkeypatch):
    """The toy has no `derive_step`: a single-request FOLD record stands in for the derived request-wrapper dir."""
    monkeypatch.setattr(bd.C, "load_derived", lambda p, label: load_oracle(p, label))


def _req(res, rid):
    return next(e for e in res["requests"] if e["request_id"] == rid)


# --------------------------------------------------------------------------------------------------------------------- tests


def test_record_carries_request_attribution(memory, tmp_path):
    d, r = fold_batched([R0, R1], tmp_path / "b2")
    lines = [json.loads(x) for x in open(f"{d}/instances.jsonl")]
    hdr, inst = lines[0], lines[1:]
    assert [(q["request_id"], q["prompt_len"], q["served_samples"]) for q in hdr["requests"]] == [("r0", 3, 3), ("r1", 2, 2)]
    assert all(m["request"] in ("r0", "r1") for m in inst)
    # step 0 is the batched prefill: 5 token rows, Embedding rows attributed r0,r0,r0,r1,r1 with their positions
    emb0 = [(m["request"], m["position"]) for m in inst if m["step"] == 0 and m["spec"].startswith("Embedding")]
    assert emb0 == [("r0", 0), ("r0", 1), ("r0", 2), ("r1", 0), ("r1", 1)]
    # lm-head rows (logits rows) and TokenSelect rows are attributed through the logits rows
    assert [(m["request"], m["position"]) for m in inst if m["step"] == 0 and m["hint"] == "lm_head"] == [("r0", 2), ("r1", 1)]
    # the decomposition reads them through its `meta` channel (nothing falls through to data flow / row counts)
    res = bd.decompose(d, [])
    assert res["attribution"]["how"] == {"meta": len(inst)} and res["attribution"]["unattributed"] == 0


def test_b2_unequal_lp_and_t_projections_equal_single_request_folds(memory, tmp_path, derived_is_a_fold):
    d, r = fold_batched([R0, R1], tmp_path / "b2")
    a0, a1 = fold_alone(R0, tmp_path / "LP3_T2"), fold_alone(R1, tmp_path / "LP2_T1")
    res = bd.decompose(d, [a0, a1])
    assert res["verdict"] == "PASS", res["reasons"]
    assert res["attribution"]["attributed"] == res["oracle"]["instances"] and res["attribution"]["unattributed"] == 0
    assert [(e["request_id"], e["LP"], e["T"], e["steps"], e["canonical_equal"]) for e in res["requests"]] == [("r0", 3, 2, 3, True), ("r1", 2, 1, 2, True)]
    assert sum(e["n_instances"] for e in res["requests"]) == res["oracle"]["instances"]
    for e in res["requests"]:
        assert e["totals"]["operands_different"] == 0 and e["totals"]["operands_equal"] > 0 and not e["xreq"]["counts"]
        assert e["chunked_steps"] == 0
    # the whole-fold digest is neither single-request digest (the batched Program is not a request Program)
    assert res["oracle"]["digest"] not in {load_oracle(a0, "a0").digest, load_oracle(a1, "a1").digest}


def test_request_finishing_early_has_no_rows_in_later_steps(memory, tmp_path, derived_is_a_fold):
    d, r = fold_batched([R0, R1_SHORT], tmp_path / "b2e")
    assert r.steps[0]["rows"] == 5 and r.steps[1]["rows"] == 1 and r.steps[2]["rows"] == 1  # r1 finished after the prefill sample
    res = bd.decompose(d, [fold_alone(R0, tmp_path / "LP3_T2"), fold_alone(R1_SHORT, tmp_path / "LP2_T0")])
    assert res["verdict"] == "PASS", res["reasons"]
    r1 = _req(res, "r1")
    assert (r1["LP"], r1["T"], r1["steps"], r1["engine_step_range"], r1["canonical_equal"]) == (2, 0, 1, [0, 0], True)
    assert _req(res, "r0")["engine_step_range"] == [0, 2]


def test_single_request_fold_is_its_own_projection(memory, tmp_path, derived_is_a_fold):
    """B = 1: the decomposition is the identity (the whole-fold `cb_a.compare` path and X-09 agree)."""
    d, _ = fold_batched([R0], tmp_path / "b1")
    json.dump({"structural_inputs": {"LP": 3, "T": 2}}, open(f"{d}/construction_manifest.json", "w"))
    res = bd.decompose(d, [d])
    assert res["verdict"] == "PASS" and res["requests"][0]["n_instances"] == res["oracle"]["instances"]
    A = load_oracle(d, "same")
    assert compare_steps(A, load_oracle(d, "same2"))["canonical_sha256"]["equal"]


def test_pre_x09_record_without_meta_is_attributed_by_data_flow(memory, tmp_path, derived_is_a_fold):
    """A fold record written before the `request`/`position` keys existed: the data-flow / row-count channels must give the
    SAME partition and the same verdict."""
    d, _ = fold_batched([R0, R1], tmp_path / "b2")
    p = f"{tmp_path}/b2/instances.jsonl"
    lines = open(p).read().splitlines()
    out = []
    for ln in lines:
        m = json.loads(ln)
        m.pop("request", None)
        m.pop("position", None)
        out.append(json.dumps(m, separators=(",", ":")))
    open(p, "w").write("\n".join(out) + "\n")
    res = bd.decompose(d, [fold_alone(R0, tmp_path / "LP3_T2"), fold_alone(R1, tmp_path / "LP2_T1")])
    assert res["verdict"] == "PASS", res["reasons"]
    assert "meta" not in res["attribution"]["how"] and res["attribution"]["row_rule_disagreements"] == 0
    assert sum(e["n_instances"] for e in res["requests"]) == res["oracle"]["instances"]
    # same partition as the meta channel gives on the unmodified record
    d2, _ = fold_batched([R0, R1], tmp_path / "b2meta")
    res2 = bd.decompose(d2, [f"{tmp_path}/LP3_T2", f"{tmp_path}/LP2_T1"])
    assert [(e["request_id"], e["n_instances"], e["canonical_sha256"]["b"]) for e in res["requests"]] == \
           [(e["request_id"], e["n_instances"], e["canonical_sha256"]["b"]) for e in res2["requests"]]


def test_negative_misattributed_row_is_not_equal(memory, tmp_path, derived_is_a_fold):
    d, _ = fold_batched([R0, R1], tmp_path / "b2")
    # perturb the record: one of r1's decode-step rows is attributed to r0
    p = f"{tmp_path}/b2/instances.jsonl"
    lines = open(p).read().splitlines()
    for i, ln in enumerate(lines[1:], 1):
        m = json.loads(ln)
        if m["request"] == "r1" and m["step"] == 1 and m["spec"].startswith("SiluMul"):
            m["request"] = "r0"
            lines[i] = json.dumps(m, separators=(",", ":"))
            break
    else:
        pytest.fail("no r1 SiluMul row at step 1")
    open(p, "w").write("\n".join(lines) + "\n")
    res = bd.decompose(d, [fold_alone(R0, tmp_path / "LP3_T2"), fold_alone(R1, tmp_path / "LP2_T1")])
    assert res["verdict"] == "FAIL"
    r0, r1 = _req(res, "r0"), _req(res, "r1")
    assert r0["canonical_equal"] is False and r1["canonical_equal"] is False
    assert r0["n_instances"] + r1["n_instances"] == res["oracle"]["instances"]  # still a partition; the error is WHICH request owns the row
    assert r0["xreq"]["counts"].get("node", 0) + sum(r0["xreq"]["counts"].values()) >= 1  # r0 now holds an instance reading r1's data


def test_negative_cross_request_operand_fails_the_row(memory, tmp_path, derived_is_a_fold):
    d, _ = fold_batched([R0, R1], tmp_path / "b2leak", leak=True)
    res = bd.decompose(d, [fold_alone(R0, tmp_path / "LP3_T2"), fold_alone(R1, tmp_path / "LP2_T1")])
    assert res["verdict"] == "FAIL" and res["xreq_total"] >= 1
    assert any("cross-request" in s for s in res["reasons"])
    r1 = _req(res, "r1")
    assert sum(r1["xreq"]["counts"].values()) >= 1 and r1["canonical_equal"] is False
    assert _req(res, "r0")["canonical_equal"] is True  # r0 reads only its own rows: unaffected


def test_prog_retains_one_copy_of_the_operand_references_and_the_verdicts_are_unchanged(memory, tmp_path, derived_is_a_fold):
    """`cb_a.compare.Prog` keeps ONE representation of the operand references
    (`inst[*].args` runs): the decoded IR body (`nodes`) is released after the instance list is built and the fold rows' `operand_refs`
    (a third copy, O(LP^2) over a long prefill) are reduced to parameter-space runs at read time and dropped once the weight-field map
    is derived.  The comparisons read `inst`/`inputs` only, so the verdicts -- positive (B=2, unequal LP and T) and the cross-request
    negative -- are the ones the retaining representation gave (GM-01 > 351 GiB on the 4096/512 fold)."""
    d, r = fold_batched([R0, R1], tmp_path / "b2")
    P = load_oracle(d, "oracle")
    assert P.nodes is None and P.n_nodes == len(P.inst) + len(P.inputs)
    assert P.oracle_rows and len(P.oracle_rows) == len(P.inst) and all(o["operand_refs"] is None for o in P.oracle_rows)
    # the weight-field map still comes from the rows' parameter-space reads (the derived side canonicalises weights by field name)
    assert P.oracle_wmap and not P.oracle_wmap_overlaps
    names = {nm for _, _, nm in P.oracle_wmap}
    assert {"model.layers.0.self_attn.qkv_proj", "model.layers.0.self_attn.o_proj", "model.layers.0.mlp.gate_up_proj", "model.layers.0.mlp.down_proj",
            "model.embed_tokens"} <= names, names
    assert all(t[0] in ("W", "TOK") for i in range(len(P.inst)) for arg in P.canon(i) for t in arg if t[0] != "N"), "unresolved (W?) weight read"
    # verdicts unchanged: positive
    a0, a1 = fold_alone(R0, tmp_path / "LP3_T2"), fold_alone(R1, tmp_path / "LP2_T1")
    A0 = load_oracle(a0, "a0")
    assert A0.nodes is None and all(o["operand_refs"] is None for o in A0.oracle_rows)
    res = bd.decompose(d, [a0, a1])
    assert res["verdict"] == "PASS", res["reasons"]
    assert [(e["request_id"], e["canonical_equal"], e["totals"]["operands_different"]) for e in res["requests"]] == [("r0", True, 0), ("r1", True, 0)]
    assert all(e["totals"]["operands_equal"] > 0 for e in res["requests"])
    same = compare_steps(load_oracle(a0, "x"), load_oracle(a0, "y"))
    assert same["canonical_sha256"]["equal"] and same["totals"]["operands_different"] == 0 and same["totals"]["operands_equal"] > 0
    # verdicts unchanged: the cross-request negative (r1's attention reading r0's KV row) still fails by name
    dl, _ = fold_batched([R0, R1], tmp_path / "b2leak", leak=True)
    neg = bd.decompose(dl, [a0, a1])
    assert neg["verdict"] == "FAIL" and neg["xreq_total"] >= 1 and _req(neg, "r1")["canonical_equal"] is False
    assert _req(neg, "r0")["canonical_equal"] is True


def test_a_witness_parameter_is_its_requests_input(memory, tmp_path, derived_is_a_fold):
    """A profile witness (the Marlin align's `schedule`) is one parameter per request after the peers: the weights still follow the
    request-owned parameters, every instance is attributed, and each request's projection reads its own witness as the single-request
    fold's witness input of that name."""
    d, r = fold_batched([R0, R1], tmp_path / "b2", witness="own")
    assert [(e["name"], e["request"]) for e in r.witness_layout] == [("schedule_0", "r0"), ("schedule_1", "r1")]
    B = load_oracle(d, "oracle")
    assert B.witness_index == {2: 0, 3: 1} and bd.request_inputs(B, [{"request_id": "r0"}, {"request_id": "r1"}])[2:] == \
        ([("r0", 2), ("r1", 3)], {4})
    res = bd.decompose(d, [fold_alone(R0, tmp_path / "LP3_T2", witness="own"), fold_alone(R1, tmp_path / "LP2_T1", witness="own")])
    assert res["verdict"] == "PASS", res["reasons"]
    assert res["attribution"]["unattributed"] == 0 and res["xreq_total"] == 0
    assert [(e["request_id"], e["canonical_equal"]) for e in res["requests"]] == [("r0", True), ("r1", True)]


def test_a_read_of_another_requests_witness_is_cross_request(memory, tmp_path, derived_is_a_fold):
    """r1's last row of each step reads r0's witness: a cross-request read in r1's projection (and r0's own words land elsewhere)."""
    d, _ = fold_batched([R0, R1], tmp_path / "b2", witness="borrow")
    res = bd.decompose(d, [fold_alone(R0, tmp_path / "LP3_T2", witness="own"), fold_alone(R1, tmp_path / "LP2_T1", witness="own")])
    assert res["verdict"] == "FAIL" and _req(res, "r1")["xreq"]["counts"].get("witness-param", 0) >= 1
    assert _req(res, "r1")["canonical_equal"] is False and _req(res, "r0")["canonical_equal"] is False


def test_missing_derived_shape_is_reported_not_assumed(memory, tmp_path, derived_is_a_fold):
    d, _ = fold_batched([R0, R1], tmp_path / "b2")
    res = bd.decompose(d, [fold_alone(R0, tmp_path / "LP3_T2")])
    assert res["verdict"] == "MISSING_DERIVED" and res["missing_derived"] == [[2, 1]]
    assert _req(res, "r0")["canonical_equal"] is True and _req(res, "r1").get("derived") is None
