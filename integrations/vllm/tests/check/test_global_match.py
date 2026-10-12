"""GM-01 (R14 lane gmatch): `cb_a.global_match` on the CPU toy of `test_batch_decomp_e2e` extended with DECLARED ARRIVALS and
output caps -- positives (B=2 unequal LP/T; staggered arrivals with a co-scheduled prefill+decode step; EOS-truncated request
against its bounded component) and the reviewer's five mutations of the global assembly, each of which must FAIL the named
check: omit a request, duplicate one, swap two requests' outputs, alter an arrival association, break a dependency
(a cross-request KV read in the record; a component wired to the wrong request shape in the Program)."""

from __future__ import annotations

import gzip
import itertools
import os
import json

import pytest

from verity_vllm.pipeline.match import persist_program
from verity_vllm.observe.events import DispatchEnter, DispatchExit, Root, RunHeader, SampleEnd, StepEnd, StepLink, StepStart, StorageAlloc, TritonLaunch, ViewDesc
from verity_vllm.check.match import global_match as gm
from verity_vllm.check.match import global_match_fast as GMF
from verity_vllm.pipeline import cli
from verity_vllm.check.match import declaration as GD
from verity_vllm.check.match import sampler_geometry as SG
from tests.check import test_batch_decomp_e2e as e2e
from tests.check.test_batch_decomp_e2e import (A_DOWN_W, A_EMBED, A_GU_W, A_IDS, A_LN1, A_LN2, A_NORM, A_O_W, A_POS, A_QKV_W, BF,
                                                                            BS, I32D, I64D, W_COS, W_DOWN, W_EMBED, W_GU, W_LN1, W_LN2, W_NORM,
                                                                            W_O, W_QKV, Ids, V, profile)
from tests.observe.test_fold_synthetic import A_COS, C, memory  # noqa: F401 - `memory` is the fixture
from verity_vllm.observe.fold.fold import fold
from verity_vllm.observe.log import dumps as log_dumps

KC = V(e2e.A_KC, (8, BS, 1, 8))   # 8 KV blocks: two private blocks per request for up to 4 requests
VC = V(e2e.A_VC, (8, BS, 1, 8))


# ------------------------------------------------------------------------------------------------- toy log with arrivals + caps
# request: {"id", "prompt", "sampled", "arrive" (engine step, default 0), "cap" (max_tokens, default len(sampled))}


def _cap(r):
    return int(r.get("cap", len(r["sampled"])))


def _arr(r):
    return int(r.get("arrive", 0))


def preamble(reqs, wl_sha=None):
    # the recorder's shape (vllm_adapter build_manifest -> RunHeader.profile): the effective engine kwargs and the compilation facts are
    # nested under `profile` -- G1 reads them there [rev-pm M-0261 / F-pm-21]; an eager run attests mode "0" / cudagraph NONE.
    # RunHeader.workload carries the manifest's sha256 and each request's seed as the driver records them [rev-pm M-0570]
    # [M-0213 / ruling 21] the toy engine schedules synchronously: async_scheduling False is ON RECORD (the lag-0 instance of the EOS
    # lag rule needs the flag too -- a lag-0 record without it is indistinguishable from a lag-1 record with a deleted step)
    hdr = RunHeader("verity-capture/log/v1", "2026-09-19T00:00:00Z", "toy-profile",
                    {"vllm": "toy", "engine_kwargs": {"enforce_eager": True, "enable_prefix_caching": False, "async_scheduling": False, "max_num_seqs": len(reqs)},
                     "compilation": {"mode": "0", "cudagraph_mode": "NONE", "enforce_eager": True}},
                    {"repo": "toy", "config": C},
                    {"requests": [{"request_id": r["id"], "prompt_token_ids": list(r["prompt"]), "max_tokens": _cap(r), "arrive_step": _arr(r), "seed": r.get("seed")} for r in reqs],
                     "sampling": {"temperature": 0.0, "variant": "greedy"}, "sha256": wl_sha})
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
    evs.append(StorageAlloc(A_IDS, 0x1000 * A_IDS, 4096, "cuda:0", 0, "buffer"))
    evs.append(StorageAlloc(A_POS, 0x1000 * A_POS, 8192, "cuda:0", 0, "buffer"))
    for a, v in ((e2e.A_KC, KC), (e2e.A_VC, VC)):
        evs.append(StorageAlloc(a, 0x1000 * a, v.numel * 2, "cuda:0", 0, "buffer"))
    return evs


def _tok(r, p):
    return r["prompt"][p] if p < len(r["prompt"]) else r["sampled"][p - len(r["prompt"])]


def step_events(step, reqs, ids, base_alloc):
    """One engine step; request ri is active at local step `step - arrive` in 0 .. len(sampled)-1 (prefill at local 0)."""
    rows, toks, links, roots, logits_rows, logits_idx, slots, seq_lens, num_computed, block_tables, sampled = [], [], [], [], [], [], [], {}, {}, {}, []
    for ri, r in enumerate(reqs):
        local = step - _arr(r)
        if local < 0 or local >= len(r["sampled"]):
            continue
        lp = len(r["prompt"])
        positions = list(range(lp)) if local == 0 else [lp + local - 1]
        bt = [2 * ri, 2 * ri + 1]
        block_tables[r["id"]] = bt
        for p in positions:
            if local > 0:
                links.append(StepLink(step, len(rows), r["id"], step - 1, "eq", r["sampled"][local - 1]))
            rows.append((r["id"], p))
            toks.append(_tok(r, p))
            slots.append(bt[p // BS] * BS + p % BS)
        if local == 0:
            roots.append((len(rows) - lp, lp, r))
        logits_rows.append((r["id"], positions[-1]))
        logits_idx.append(len(rows) - 1)
        seq_lens[r["id"]] = positions[-1] + 1
        num_computed[r["id"]] = positions[0]
        sampled.append(r["sampled"][local])
    if not rows:
        return []
    M, R = len(rows), len(logits_rows)
    a = itertools.count(base_alloc)
    pos_view = V(A_POS, (M,), dtype=I64D)
    hidden, normed, qkv, attn, o, gu, act, down, gathered, logits, sampled_v = (V(next(a), s, dtype=d) for s, d in (
        ((M, 16), BF), ((M, 16), BF), ((M, 32), BF), ((M, 2, 8), BF), ((M, 16), BF), ((M, 32), BF), ((M, 16), BF), ((M, 16), BF),
        ((R, 16), BF), ((R, 16), BF), ((R,), I64D)))
    ids_rows = ViewDesc(A_IDS, 0, (M,), (1,), I32D, "cuda:0")
    phase = "prefill" if all(p < len(next(q for q in reqs if q["id"] == rid)["prompt"]) for rid, p in rows) else \
        ("decode" if all(p >= len(next(q for q in reqs if q["id"] == rid)["prompt"]) for rid, p in rows) else "mixed")
    evs = [StepStart(step, rows, toks, [p for _, p in rows], seq_lens, num_computed, block_tables, BS, slots, logits_rows, logits_idx,
                     R, M, M, ids_rows, pos_view, phase)]
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


def batched_log(reqs, wl_sha=None):
    ids = Ids()
    evs = preamble(reqs, wl_sha)
    n_steps = max(_arr(r) + len(r["sampled"]) for r in reqs)
    for s in range(n_steps):
        evs += step_events(s, reqs, ids, 20 + 40 * s)
    return evs


def tokens_json(reqs):
    return {"requests": [{"request_id": r["id"], "prompt_token_ids": list(r["prompt"]), "max_tokens": _cap(r), "generated_token_ids": list(r["sampled"]),
                          "prompt_len": len(r["prompt"]), "n_generated": len(r["sampled"]), "arrive_step": _arr(r), "arrive_step_realized": _arr(r),
                          "first_step": _arr(r), "finish_step": _arr(r) + len(r["sampled"]) - 1,
                          "finish_reason": "length" if len(r["sampled"]) == _cap(r) else "stop"} for r in reqs]}


def persist(result, events, d, reqs):
    d.mkdir(parents=True, exist_ok=True)
    persist_program(result, d, profile_name="toy", log_path="toy")
    (d / "capture").mkdir(exist_ok=True)
    with gzip.open(d / "capture" / "log.jsonl.gz", "wt") as f:
        f.write(log_dumps(events))
    (d / "capture" / "tokens.json").write_text(json.dumps(tokens_json(reqs)))
    return str(d)


def component(req, d):
    """The request served alone to its CAP (the bounded request-wrapper component): a B=1 fold record stands in for derive_step."""
    full = dict(req, arrive=0, sampled=list(req["sampled"]) + [1] * (_cap(req) - len(req["sampled"])))
    evs = batched_log([full])
    r = fold(evs, profile(), strict=True)
    p = persist(r, evs, d, [full])
    json.dump({"structural_inputs": {"LP": len(req["prompt"]), "T": _cap(req) - 1}}, open(f"{p}/construction_manifest.json", "w"))
    return p


def p_attention_tgrouped(node, ctx):
    """The PRODUCTION emission order (`patterns.AttentionScope.match`): one Compute per DISTINCT T = position+1 over the batch's rows
    in first-occurrence order, rows = the rows sharing that T (a running decode at position p and an arriving prefill's row p land
    in one group, created when the decode row is seen). The toy default (`p_attention`) emits per row in row order and therefore
    never shows M-ARRIVAL-ORDER; this variant reproduces it on CPU with the same fold code path (`fold._compute_emit` per group)."""
    from tests.observe.test_fold_synthetic import _kv_flat, _slot
    from verity_vllm.observe.fold.resolver import Compute, Gather, GatherMany, Group
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import b1
    BN = e2e.BN
    if node.op != "vllm.unified_attention_with_output.default" or len(node.children) != 1:
        return None
    ch = node.children[0]
    q3, kc, vc, out = ch.args[:4]
    groups: dict[int, list[int]] = {}
    for i, (req, p) in enumerate(ctx.start.rows):
        groups.setdefault(p + 1, []).append(i)
    parts = []
    for T, rows in groups.items():
        slots = tuple(tuple(_slot(ctx, ctx.start.rows[i][0], j) for j in range(T)) for i in rows)
        parts.append(Compute(bind(b1.AttentionV3, T=T, NH=C["NH"], KVH=C["KVH"], D=C["D"], BN=BN), len(rows),
                             (Gather(q3, tuple(rows)), GatherMany(_kv_flat(kc), slots), GatherMany(_kv_flat(vc), slots)), (Gather(out, tuple(rows)),),
                             (), node.enter.module or "", attribution=tuple(ctx.start.rows[i] for i in rows)))
    return Group(tuple(parts), (node.id, ch.id), node.enter.module or "")


def profile_tgrouped(leak=False):
    from verity_vllm.observe.fold.resolver import Profile
    from tests.observe import test_fold_synthetic as toy
    pats = [p for p in toy.PATTERNS if p.name != "attention"]
    pats.append(toy.Pat("attention_leak" if leak else "attention_tgrouped", e2e.p_attention_leak if leak else p_attention_tgrouped))
    return Profile("toy", pats, dict(C), toy.weight_field, False)


def batched(reqs, d, leak=False, tgrouped=False, wl_sha=None):
    evs = batched_log(reqs, wl_sha)
    r = fold(evs, profile_tgrouped(leak) if tgrouped else profile(leak), strict=True)
    assert r.report.unsupported == 0 and not r.errors
    return persist(r, evs, d, reqs)


def move_consumer_before_producer(oracle_dir, rid="r1", sampler_to_head=False):
    """Fixture mutation (rev-pm M-0042 condition 2) on the PERSISTED record: one instance of request `rid` is re-emitted BEFORE the
    instance it consumes -- the same computation, the same wiring (every reference is re-indexed), only the emission order inside one
    request step violates the dependency order.  The IR builder cannot construct such a body (a forward reference has no value to
    substitute), so this is what a hand-edited program.json/instances.jsonl looks like; the DAG criterion must FAIL it (G3: the
    consumer's Merkle hash names a forward reference; G4: no canonical address; G8: consumer-before-producer), as `sequence` does."""
    from verity.ml.circuits import codec

    doc = json.load(open(f"{oracle_dir}/program.json"))
    desc = doc["program"]
    body = desc["definitions"][desc["root"]]["body"]
    nodes = body["nodes"]
    n_in = doc["n_input_nodes"]
    fid_of = (lambda nd: desc["callees"][nd["fn"]]) if "callees" in desc else (lambda nd: nd["fn"])
    lines = open(f"{oracle_dir}/instances.jsonl").read().splitlines()
    head, inst = json.loads(lines[0]), [json.loads(x) for x in lines[1:]]
    by_root = {n_in + m["node"]: m for m in inst}

    def expand(j):   # de-alias (aliases point backwards to an already-expanded argument)
        if isinstance(j, dict) and "alias" in j:
            k, a = j["alias"]
            return expand(nodes[k]["args"][a]["refs"])
        if isinstance(j, dict) and "concat" in j:
            return {"concat": [expand(x) for x in j["concat"]]}
        return j

    for nd in nodes:
        for a in nd["args"]:
            a["refs"] = expand(a["refs"])
    body["ret"] = expand(body["ret"])

    def n_refs(j):
        if "concat" in j:
            return [t for x in j["concat"] for t in n_refs(x)]
        if "affine" in j or "strided" in j:
            v = j.get("affine") or j.get("strided")
            return [(v[0], v[1])]
        return [(s, i) for s, i, _ in j["explicit"]]

    pick = None
    if sampler_to_head:
        # [rev-pm M-0171 fixture (3)] the request's FIRST TokenSelect is re-emitted at the head of the request's instances of its engine
        # step: same computations, same wiring, the sampler before everything it consumes
        for c, nd in enumerate(nodes):
            mc = by_root.get(c)
            if mc and mc.get("request") == rid and fid_of(nd).startswith("TokenSelect"):
                first = next(k for k, nd2 in enumerate(nodes) if (by_root.get(k) or {}).get("request") == rid and by_root[k].get("step") == mc.get("step"))
                assert first != c
                pick = (first, c)
                break
    for c, nd in enumerate(nodes):
        if pick:
            break
        mc = by_root.get(c)
        if not mc or mc.get("request") != rid or fid_of(nd).startswith("TokenSelect"):
            continue
        for s, p in {t for a in nd["args"] for t in n_refs(a["refs"])}:
            mp = by_root.get(p)
            if s == "n" and mp and mp.get("request") == rid and mp.get("step") == mc.get("step") and not fid_of(nodes[p]).startswith("TokenSelect"):
                pick = (p, c)
                break
        if pick:
            break
    assert pick is not None, "no producer->consumer pair of the request inside one step"
    p_old, c_old = pick
    order = [k for k in range(len(nodes)) if k != c_old]
    order.insert(order.index(p_old), c_old)            # new position -> old index: the consumer now precedes its producer
    new_of = {old: new for new, old in enumerate(order)}

    def remap(j):
        if "concat" in j:
            return {"concat": [remap(x) for x in j["concat"]]}
        for key in ("affine", "strided"):
            if key in j:
                v = list(j[key])
                if v[0] == "n":
                    v[1] = new_of[v[1]]
                return {key: v}
        return {"explicit": [[s, new_of[i] if s == "n" else i, l] for s, i, l in j["explicit"]]}

    for nd in nodes:
        for a in nd["args"]:
            a["refs"] = remap(a["refs"])
    body["ret"] = remap(body["ret"])
    body["nodes"] = [nodes[old] for old in order]
    doc["digest"] = codec.program_digest(desc)
    json.dump(doc, open(f"{oracle_dir}/program.json", "w"), separators=(",", ":"), sort_keys=True)
    for m in inst:
        m["node"] = new_of[n_in + m["node"]] - n_in
    inst.sort(key=lambda m: m["node"])                  # instances.jsonl line order == node order
    head["digest"] = doc["digest"]
    with open(f"{oracle_dir}/instances.jsonl", "w") as f:
        f.write(json.dumps(head) + "\n")
        for m in inst:
            f.write(json.dumps(m, separators=(",", ":")) + "\n")
    return new_of[c_old] - n_in, new_of[p_old] - n_in


def workload(reqs, path):
    json.dump({"requests": [{"request_id": r["id"], "prompt_token_ids": list(r["prompt"]), "max_tokens": _cap(r), "arrive_step": _arr(r), "seed": r.get("seed")} for r in reqs],
               "sampling": {"temperature": 0.0, "variant": "greedy"}, "sweep": {"arrive_steps": [_arr(r) for r in reqs]}}, open(path, "w"))
    return str(path)


def _sha(path):
    """RunHeader.workload.sha256 as the driver computes it (vllm_adapter.load_workload): the canonical manifest document."""
    return __import__("hashlib").sha256(json.dumps(json.load(open(path)), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


EOS = 11   # the fixture model's EOS id (R1_EOS stops on it); the Program declares it as its stop rule (F-pm-24)


def assemble(reqs, tmp_path, leak=False, tgrouped=False, eos_token_ids=(EOS,), lag=0):
    """workload manifest + one bounded component per distinct shape + the batched record -> (global Program, oracle dir).  The record's
    RunHeader.workload.sha256 is the manifest's (the driver records the manifest it ran) [M-0570].  [R17-1] `lag` = the LAG the mock
    Program DECLARES (default 0: the toy engine schedules synchronously and its preamble attests async_scheduling False); None = a
    pre-R17-1 Program that declares no lag."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    wl = workload(reqs, tmp_path / "workload.json")
    oracle = batched(reqs, tmp_path / "batched", leak, tgrouped, wl_sha=_sha(wl))
    comps = {}
    for r in reqs:
        sh = (len(r["prompt"]), _cap(r) - 1)
        if sh not in comps:
            comps[sh] = component(r, tmp_path / f"comp_LP{sh[0]}_T{sh[1]}")
    return gm.mock_global_program(wl, list(comps.values()), eos_token_ids=list(eos_token_ids) if eos_token_ids else None, lag=lag), oracle


def _check(res, cid):
    return next(k for k in res["checks"] if k["id"] == cid)


R0 = {"id": "r0", "prompt": [5, 7, 2], "sampled": [9, 3, 1]}                 # LP 3, cap 3 (served to cap)
R1 = {"id": "r1", "prompt": [4, 6], "sampled": [8, 11]}                      # LP 2, cap 2
R1_LATE = {"id": "r1", "prompt": [4, 6], "sampled": [8, 11], "arrive": 1}    # arrives at engine step 1: prefill co-scheduled with r0's decode
R2_LATE = {"id": "r2", "prompt": [3, 3, 3, 3], "sampled": [6, 2, 2], "arrive": 2}
R1_EOS = {"id": "r1", "prompt": [4, 6], "sampled": [8, 11], "cap": 4}        # declared cap 4, stopped after 2 (EOS)


# ------------------------------------------------------------------------------------------------------------------- positives


def test_b2_unequal_requests_pass_globally(memory, tmp_path):
    gp, oracle = assemble([R0, R1], tmp_path)
    res = gm.check(gp, oracle)
    assert res["verdict"] == "PASS", res["reasons"]
    assert [k["status"] for k in res["checks"]] == ["PASS"] * 8
    assert res["comparator"] == "cb_a.global_match/v1" and res["rules_version"] == "r14-v1"
    g7 = _check(res, "G7")
    assert g7["population_query"] == g7["population_fold"] == sum(e["n_instances"] for e in res["requests"])
    assert g7["boundary_probes"] > 0 and g7["boundary_probe_failures"] == 0
    assert res["x09"]["verdict"] == "PASS"
    assert not res["eos_truncated_requests"]


def test_fast_path_releases_its_caches_when_the_outermost_check_returns(memory, tmp_path, monkeypatch):
    """[R17 mprof unit 4, rev F-r17-06] MATCH_IMPL=fast memoises per Prog / Projection for the duration of ONE check() -- including its
    nested alternate-criterion run -- and holds nothing afterwards, so a process that checks several records does not grow by one
    record's caches per record.  The document is the checker of record's, field for field."""
    from verity_vllm.check.match import global_match_fast as GMF
    gp, oracle = assemble([R0, R1_LATE], tmp_path)
    ref = gm.check(gp, oracle)
    monkeypatch.setenv(GMF.FLAG, "fast")
    seen = {}
    orig_leave = GMF.leave

    def spy_leave():
        seen.setdefault("depth_before_leave", []).append(GMF._depth)
        seen.setdefault("memo_at_leave", []).append(len(GMF._dag_memo) + len(GMF._chash_memo) + len(GMF._proj_registry))
        orig_leave()
    monkeypatch.setattr(GMF, "leave", spy_leave)
    res = gm.check(gp, oracle)
    assert seen["depth_before_leave"] == [2, 1]              # the alternate run's leave (keeps the memo), then the outermost one
    assert seen["memo_at_leave"][0] > 0                       # populated while the outermost check() is still running
    assert GMF._depth == 0
    assert not GMF._dag_memo and not GMF._chash_memo and not GMF._cmp_memo and not GMF._live_memo and not GMF._proj_registry
    assert not GMF._merkle_ids and not GMF.stats
    # [rev F-r17-09] the released counters survive as the document's telemetry: impl_info after check() is not empty
    assert GMF.last_stats and GMF.last_stats.get("projection_built", 0) >= 1
    if hasattr(gm, "impl_info"):   # mfast's dispatch block (their branch): the CLI's impl block reads the snapshot
        info = gm.impl_info(GMF)
        assert info["stats"] == GMF.last_stats
        # [rev S-r17-17 condition] the DECLARED configuration is on the block: the workers cap as read, the operand form, the switches
        cfg = info["config"]
        assert cfg["match_workers"] >= 1 and cfg["operand_form"] in ("progressions/v1", "runs") and isinstance(cfg["match_compact"], bool)
        assert cfg["operand_form"] == ("progressions/v1" if cfg["match_compact"] else "runs")
        assert cfg["match_workers"] == (int(os.environ["MATCH_WORKERS"]) if os.environ.get("MATCH_WORKERS", "").strip() else min(os.cpu_count() or 1, 8))
        assert cfg["match_compact"] == (os.environ.get("MATCH_COMPACT", "1").strip() != "0")
    def strip(o):   # timings (seconds, mfast's phase marks) and the impl block are the only fields the path may differ in
        if isinstance(o, dict):   # `utc` too (mfast, rev M-0542): two check() calls straddling a second boundary made this a 1-in-7 flake on rel's pod gate
            return {k: strip(v) for k, v in o.items() if k not in ("seconds", "impl", "phases", "utc")}
        return [strip(v) for v in o] if isinstance(o, list) else o
    assert json.dumps(strip(res), sort_keys=True, default=str) == json.dumps(strip(ref), sort_keys=True, default=str)


def test_fast_path_passes_its_operations_and_rebinds_nothing(memory, tmp_path, monkeypatch):
    """MATCH_IMPL=fast hands GM-01 and its X-09 run `global_match_fast.ops()` explicitly: after a fast check() every Verity core and
    checker-of-record function is the object it was before, the fast run's Programs are its own (MemoProg, progressions), and a record
    check() in the same process still loads the record's Programs and writes the same document."""
    import verity.ml.circuits.codec as codec
    import verity.ml.circuits.refs as refs
    from verity_vllm.check.match import batch_decomp as BD
    from verity_vllm.check.match import program_compare as C
    from verity_vllm.check.match import global_match_fast as GMF
    names = [(codec, "_spec_id"), (refs, "runs"), (C, "compare_steps"), (C, "_canonical_hash"), (C, "_canon_step_relative"),
             (C, "load_oracle"), (C, "load_derived"), (BD, "dag_hashes"), (BD, "live_cone"), (BD, "Projection")]
    before = [getattr(m, n) for m, n in names]
    canon = C.Prog.__dict__["canon"]
    gp, oracle = assemble([R0, R1_LATE], tmp_path)
    ref = gm.check(gp, oracle)
    monkeypatch.setenv(GMF.FLAG, "fast")
    monkeypatch.delenv(GMF.COMPACT_ENV, raising=False)
    loaded_fast: dict = {}
    fast = gm.check(gp, oracle, _loaded=loaded_fast)
    assert all(getattr(m, n) is b for (m, n), b in zip(names, before)) and C.Prog.__dict__["canon"] is canon
    assert type(loaded_fast["oracle"]) is GMF.MemoProg and loaded_fast["oracle"].compact
    assert all(type(A) is GMF.MemoProg for A in loaded_fast["components"].values())
    monkeypatch.delenv(GMF.FLAG)
    loaded_rec: dict = {}
    again = gm.check(gp, oracle, _loaded=loaded_rec)
    assert type(loaded_rec["oracle"]) is C.Prog and not loaded_rec["oracle"].compact
    assert all(type(A) is C.Prog for A in loaded_rec["components"].values())

    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if k not in ("seconds", "impl", "phases", "utc")}
        return [strip(v) for v in o] if isinstance(o, list) else o
    assert json.dumps(strip(fast), sort_keys=True, default=str) == json.dumps(strip(ref), sort_keys=True, default=str) \
        == json.dumps(strip(again), sort_keys=True, default=str)


def test_staggered_arrivals_pass_with_mixed_step_and_declared_chronology(memory, tmp_path):
    gp, oracle = assemble([R0, R1_LATE, R2_LATE], tmp_path)
    res = gm.check(gp, oracle)
    assert res["verdict"] == "PASS", res["reasons"]
    by = {e["request_id"]: e for e in res["requests"]}
    assert (by["r0"]["first_step"], by["r1"]["first_step"], by["r2"]["first_step"]) == (0, 1, 2)
    g8 = _check(res, "G8")
    assert any(l["phase"] == "mixed" for l in g8["batch_layout"]) and not g8["permutations"]
    assert _check(res, "G6")["status"] == "PASS" and by["r2"]["output_events"] == 3


def test_eos_truncated_request_is_compared_over_its_executed_prefix_and_labelled(memory, tmp_path):
    gp, oracle = assemble([R0, R1_EOS], tmp_path)
    res = gm.check(gp, oracle)
    assert res["verdict"] == "PASS", res["reasons"]
    r1 = next(e for e in res["requests"] if e["request_id"] == "r1")
    assert r1["eos_truncated"] and r1["unexecuted_steps"] == 2 and r1["finish"] == "eos" and r1["component"]["T"] == 3
    assert res["eos_truncated_requests"] == ["r1"]
    assert _check(res, "G7")["unexercised_declared"] > 0
    # [F-pm-24] the truncation is EOS-VERIFIED: the last served token (11) is one of the Program's declared EOS ids
    assert r1["eos_verified"] is True and r1["eos_token"] == EOS and _check(res, "G6")["stop_rule"]["eos_token_ids"] == [EOS]


def test_early_stop_without_an_eos_token_fails_g6_whatever_the_driver_says(memory, tmp_path):
    """rev-pm ALERT M-0537 (F-pm-24): a request that stopped before its cap on a NON-EOS token -- an engine that dropped or aborted it, a
    stop string, or a prover that skipped the tail -- was accepted as 'eos-truncated' on the strength of finish_reason alone. The EOS is
    the evidence: G6 requires served[-1] in the declared set; finish_reason None and 'stop' both fail; and the achieved-concurrency
    shortfall such a request causes is not waived."""
    # the fixture's r1 stops on token 11 after 2 of 4 events; declare an EOS set that does NOT contain it
    gp, oracle = assemble([R0, R1_EOS], tmp_path, eos_token_ids=(99,))
    for finish in ("stop", None):
        tp = os.path.join(oracle, "capture", "tokens.json")
        tk = json.load(open(tp))
        next(r for r in tk["requests"] if r["request_id"] == "r1")["finish_reason"] = finish
        json.dump(tk, open(tp, "w"))
        res = gm.check(gp, oracle)
        assert res["verdict"] == "FAIL" and "G6" in res["failed_checks"], (finish, res["reasons"])
        g6 = _check(res, "G6")
        assert any("on token 11, not a declared EOS id [99]" in p and "F-pm-24" in p for p in g6["problems"]), g6["problems"]
        r1 = next(e for e in res["requests"] if e["request_id"] == "r1")
        assert r1["eos_verified"] is False
        # the concurrency waiver is gone with it: observed max 2 (both requests live at step 0..1) vs programmed 2 here, so no
        # shortfall to waive in this fixture -- the G6 failure is the verdict; the waiver's condition is pinned on the entry
        assert not any(e.get("eos_verified") is True for e in res["requests"])
    # the same record against a Program that declares NO stop rule (pre-fix artifact): fail closed, named
    gp2, oracle2 = assemble([R0, R1_EOS], tmp_path / "nodecl", eos_token_ids=None)
    res2 = gm.check(gp2, oracle2)
    assert res2["verdict"] == "FAIL" and "G6" in res2["failed_checks"]
    assert any("declares no EOS set" in p and "fail closed" in p for p in _check(res2, "G6")["problems"]), _check(res2, "G6")["problems"]
    # the concurrency waiver (G7) needs EOS-VERIFIED truncation: r1's early stop leaves step 2 with 2 live requests where the Program
    # schedules 3 (r0, r1 to its cap, r2 arriving) -- verified EOS explains the shortfall and PASSES; the same stop on a non-EOS token does not
    gp4, oracle4 = assemble([R0, R1_EOS, R2_LATE], tmp_path / "waiver_ok")
    res4 = gm.check(gp4, oracle4)
    conc = res4["achieved_concurrency"]
    assert res4["verdict"] == "PASS" and (conc["observed_max"], conc["programmed_max"]) == (2, 3), (res4["reasons"], conc)
    gp5, oracle5 = assemble([R0, R1_EOS, R2_LATE], tmp_path / "waiver_no", eos_token_ids=(99,))
    res5 = gm.check(gp5, oracle5)
    assert res5["verdict"] == "FAIL" and "G6" in res5["failed_checks"], res5["reasons"]
    assert any("no EOS-verified truncated request (1 truncated, 0 EOS-verified)" in p for p in _check(res5, "G6")["problems"]), _check(res5, "G6")["problems"]
    # [COORD M-0571 (1)] and the shortfall is named per ENGINE STEP: r1 (unverified) was programmed for steps 2 and 3 as well
    assert any("engine steps [2, 3]: live requests observed [2, 1] != programmed [3, 2]" in p and "['r1'] stopped early without one" in p
               for p in _check(res5, "G6")["problems"]), _check(res5, "G6")["problems"]
    assert res5["achieved_concurrency"]["steps_short_of_programmed"] == 2
    assert res4["achieved_concurrency"]["steps_short_of_programmed"] == 0 and res4["achieved_concurrency"]["verified_eos_steps"] == [1]
    # [rev-pm M-0622 TIP (a)] an honest EOS in the SampleEnd but another token in the driver's tokens.json (output association FAIL):
    # eos_verified must read False beside the G6 FAIL, never True
    gp6, oracle6 = assemble([R0, R1_EOS], tmp_path / "assoc")
    tp6 = os.path.join(oracle6, "capture", "tokens.json")
    tk6 = json.load(open(tp6))
    next(r for r in tk6["requests"] if r["request_id"] == "r1")["generated_token_ids"] = [8, 99]
    json.dump(tk6, open(tp6, "w"))
    res6 = gm.check(gp6, oracle6)
    assert res6["verdict"] == "FAIL" and "G6" in res6["failed_checks"]
    r1 = next(e for e in res6["requests"] if e["request_id"] == "r1")
    assert r1["eos_verified"] is False and "G6 problem" in r1["eos_verified_note"] and res6["eos_verified"] == {"r1": False}
    # and a request that reached its cap is untouched by the rule (no EOS needed, no declaration needed)
    gp3, oracle3 = assemble([R0, R1], tmp_path / "cap", eos_token_ids=None)
    res3 = gm.check(gp3, oracle3)
    assert res3["verdict"] == "PASS", res3["reasons"]
    assert all(e["finish"] == "length" and "eos_verified" not in e for e in res3["requests"])


# ------------------------------------------------------------------------------------------- the reviewer's five mutations


def test_mutation_omit_request_fails(memory, tmp_path):
    gp, oracle = assemble([R0, R1], tmp_path)
    gp["requests"] = [r for r in gp["requests"] if r["request_id"] != "r1"]
    res = gm.check(gp, oracle)
    assert res["verdict"] == "FAIL"
    assert _check(res, "G1")["status"] == "FAIL" and any("only in record ['r1']" in p for p in _check(res, "G1")["problems"])
    assert _check(res, "G2")["status"] == "FAIL"   # r1 executed but undeclared


def test_manifest_field_edited_after_the_build_is_named_per_request(memory, tmp_path):
    # rev-pm M-0302 (a): a manifest whose per-request field was edited after the Build FAILs G1 naming the request (not by digest alone)
    gp, oracle = assemble([R0, R1_LATE], tmp_path)
    wl_path = gp["workload"]["path"]
    for field, value in (("max_tokens", 5), ("arrive_step", 0)):
        wl = json.load(open(wl_path))
        wl["requests"][1][field] = value
        if field == "arrive_step":
            wl["sweep"]["arrive_steps"][1] = value
        json.dump(wl, open(wl_path, "w"))
        res = gm.check(gp, oracle)
        g1 = _check(res, "G1")
        assert res["verdict"] == "FAIL" and g1["status"] == "FAIL", (field, res["reasons"])
        assert any(p.startswith("r1: manifest " + field) for p in g1["problems"]), (field, g1["problems"])
    # a shorter prompt in the manifest: prompt_len named AND the prompt itself (record != manifest)
    wl = json.load(open(wl_path))
    wl["requests"][1]["prompt_token_ids"] = wl["requests"][1]["prompt_token_ids"][:1]
    json.dump(wl, open(wl_path, "w"))
    g1 = _check(gm.check(gp, oracle), "G1")
    assert any(p.startswith("r1: manifest prompt_len") for p in g1["problems"]) and any("record prompt != manifest prompt" in p for p in g1["problems"])


def test_mutation_duplicate_request_fails(memory, tmp_path):
    gp, oracle = assemble([R0, R1], tmp_path)
    gp["requests"].append(dict(gp["requests"][1]))
    res = gm.check(gp, oracle)
    assert res["verdict"] == "FAIL"
    g1 = _check(res, "G1")
    assert g1["status"] == "FAIL" and any("more than once" in p for p in g1["problems"])


def test_mutation_swapped_outputs_fail_the_chronology(memory, tmp_path):
    gp, oracle = assemble([R0, R1], tmp_path)
    tj = json.load(open(f"{oracle}/capture/tokens.json"))
    a, b = tj["requests"]
    a["generated_token_ids"], b["generated_token_ids"] = b["generated_token_ids"], a["generated_token_ids"]
    json.dump(tj, open(f"{oracle}/capture/tokens.json", "w"))
    res = gm.check(gp, oracle)
    assert res["verdict"] == "FAIL" and res["failed_checks"] == ["G6"]
    assert any("output association" in p for p in _check(res, "G6")["problems"])


def test_mutation_altered_arrival_association_fails_the_chronology(memory, tmp_path):
    gp, oracle = assemble([R0, R1_LATE], tmp_path)
    next(r for r in gp["requests"] if r["request_id"] == "r1")["arrive_step"] = 0     # declared 0, executed at 1
    res = gm.check(gp, oracle)
    # G6: the declaration disagrees with the observed arrival; G1 (M-0302 (a)): the Program's declaration also disagrees with the manifest
    assert res["verdict"] == "FAIL" and res["failed_checks"] == ["G1", "G6"]
    assert any("declared arrival step 0 != observed 1" in p for p in _check(res, "G6")["problems"])
    assert any(p.startswith("r1: manifest arrive_step=1 != the Program's declared arrive_step=0") for p in _check(res, "G1")["problems"])
    # the idle rule does not excuse it either way round: declare it later than it ran while r0 was still running
    gp2, oracle2 = assemble([R0, R1_LATE], tmp_path / "b")
    next(r for r in gp2["requests"] if r["request_id"] == "r1")["arrive_step"] = 2
    res2 = gm.check(gp2, oracle2)
    assert res2["verdict"] == "FAIL" and "G6" in res2["failed_checks"]


def test_mutation_broken_dependency_in_the_record_fails_shared_state(memory, tmp_path):
    gp, oracle = assemble([R0, R1], tmp_path, leak=True)          # r1's attention reads r0's KV row
    res = gm.check(gp, oracle)
    assert res["verdict"] == "FAIL"
    g5 = _check(res, "G5")
    assert g5["status"] == "FAIL" and any("cross-request operand reads" in p for p in g5["problems"])
    assert _check(res, "G3")["status"] == "FAIL"


def test_mutation_broken_dependency_in_the_program_fails_subcircuits(memory, tmp_path):
    gp, oracle = assemble([R0, R1], tmp_path)
    r0, r1 = gp["requests"]
    r1["component"] = dict(r0["component"])                       # r1 wired to r0's component (wrong request-specific input shape)
    res = gm.check(gp, oracle)
    assert res["verdict"] == "FAIL"
    assert _check(res, "G3")["status"] == "FAIL" and _check(res, "G4")["status"] == "FAIL"


def test_missing_component_is_a_precise_failure_not_an_assumption(memory, tmp_path):
    gp, oracle = assemble([R0, R1], tmp_path)
    gp["requests"][1]["component"]["dir"] = None
    res = gm.check(gp, oracle)
    assert res["verdict"] == "FAIL" and any("no component Program" in p for p in _check(res, "G3")["problems"])


# ------------------------------------------------------------------------------- R15: request ids r<k> (COORD ruling 2)

U0 = dict(R0, id="0fe3d023-6b51-5e71-92f2-01412ddad63f")   # the manifest's UUIDs: the ids vLLM serves under
U1 = dict(R1_LATE, id="ffe63996-f232-5aef-8946-d832cd542fa3")


def test_record_uuids_are_mapped_to_rk_and_the_program_names_requests_rk(memory, tmp_path):
    gp, oracle = assemble([U0, U1], tmp_path)
    assert [r["request_id"] for r in gp["requests"]] == ["r0", "r1"]
    assert [r["workload_request_id"] for r in gp["requests"]] == [U0["id"], U1["id"]]
    res = gm.check(gp, oracle)
    assert res["verdict"] == "PASS", res["reasons"]
    assert [e["request_id"] for e in res["requests"]] == ["r0", "r1"]
    assert res["request_ids"]["table"] == {U0["id"]: "r0", U1["id"]: "r1"} and not res["request_ids"]["unknown_ids"]
    by = {e["request_id"]: e for e in res["requests"]}
    assert by["r1"]["first_step"] == 1 and by["r1"]["workload_request_id"] == U1["id"]
    assert res["achieved_concurrency"]["observed_max"] == res["achieved_concurrency"]["programmed_max"] == 2
    assert _check(res, "G6")["generated_lengths"] == {"r0": 3, "r1": 2}
    # [GM-01 memory fix] X-09 shares this process's decoded fold and runs on the RAW record ids: the in-place r<k> renaming of the fold
    # rows (canonicalize_record) must not reach it -- when it did, X-09 reported every request 'not in RunHeader.workload' (cp-dev1 B4)
    x = res["x09"]
    assert x["verdict"] == "PASS", x["reasons"]
    assert [e["request_id"] for e in x["requests"]] == ["r0", "r1"] and [e["workload_request_id"] for e in x["requests"]] == [U0["id"], U1["id"]]
    assert all(e["canonical_equal"] is True for e in x["requests"]) and x["attribution"]["unattributed"] == 0
    # the alternate-criterion run reuses the same decoded fold (already canonical) and reaches the same request set
    assert [e["request_id"] for e in res["alternate_verdict"]["requests"]] == ["r0", "r1"]


def test_record_serving_a_request_the_program_did_not_declare_fails_g1(memory, tmp_path):
    gp, oracle = assemble([U0, U1], tmp_path)
    gp["requests"][1]["workload_request_id"] = "00000000-0000-0000-0000-000000000000"   # Program's r1 is a different workload request
    res = gm.check(gp, oracle)
    assert res["verdict"] == "FAIL" and "G1" in res["failed_checks"]
    assert set(res["request_ids"]["unknown_ids"]) == {U1["id"]}   # the record's UUID is not in the Program's table -> undeclared


# ------------------------------------------------- R15 M-ARRIVAL-ORDER (DECISION M-0021): the production emission order on CPU

# r0's decode at engine step 1 is position 3 (T=4); a prefill of LP 5 arriving at step 1 has a position-3 row -> AttentionScope's
# T=4 group is created by r0's row first, so rX's position-3 attention is emitted before its positions 0..2 (the pod fixture:
# LP147 @ e=6 with the decode at 128; LP256 @ e=18 with decodes at 9 and 140).
RX_LATE = {"id": "r1", "prompt": [4, 6, 1, 2, 3], "sampled": [8, 11], "arrive": 1}
RS_LATE = {"id": "r1", "prompt": [4, 6], "sampled": [8, 11], "arrive": 1}   # LP 2 < the decode's position 3: no displacement (pod: LP8 @ e=16)


def test_tgrouped_emission_without_displacement_passes_both_criteria(memory, tmp_path):
    gp, oracle = assemble([R0, RS_LATE], tmp_path, tgrouped=True)
    for crit in ("sequence", "dag"):
        res = gm.check(gp, oracle, within_step=crit)
        assert res["verdict"] == "PASS", (crit, res["reasons"])
        assert not _check(res, "G8")["permutations"]


def test_coscheduled_prefill_tgrouped_emission_fails_the_sequence_criterion_as_on_the_pod(memory, tmp_path):
    gp, oracle = assemble([R0, RX_LATE], tmp_path, tgrouped=True)
    res = gm.check(gp, oracle, within_step="sequence")   # the r14-v1 criterion as the verdict of record (default is dag since M-0177)
    assert res["verdict"] == "FAIL" and res["failed_checks"] == ["G3", "G7", "G8"], (res["failed_checks"], res["reasons"])
    assert res["within_step_criterion"] == "sequence"
    r1 = next(e for e in res["requests"] if e["request_id"] == "r1")
    assert r1["executed_steps_equal"] is False and r1["first_divergence"]["step"] == 0
    perms = _check(res, "G8")["permutations"]
    assert len(perms) == 1 and perms[0]["same_dag_multiset"] and perms[0]["unmatched"] == 0 and perms[0]["moved"] > 0
    assert perms[0]["displaced_query_positions"] == [3] == perms[0]["coscheduled_other_positions"]
    # the engine-step chronology and the dependencies are NOT what fails
    assert _check(res, "G5")["status"] == "PASS" and _check(res, "G6")["status"] == "PASS"
    assert _check(res, "G7")["population_query"] == _check(res, "G7")["population_fold"]


def test_coscheduled_prefill_tgrouped_emission_passes_the_dag_criterion_with_the_layout_recorded(memory, tmp_path):
    gp, oracle = assemble([R0, RX_LATE], tmp_path, tgrouped=True)
    res = gm.check(gp, oracle, within_step="dag")
    assert res["verdict"] == "PASS", res["reasons"]
    assert res["within_step_criterion"] == "dag" and "M-0177" in res["within_step_note"]
    r1 = next(e for e in res["requests"] if e["request_id"] == "r1")
    assert r1["executed_steps_equal"] and r1["criterion"].startswith("dag")
    assert r1["within_step"]["permuted_instances"] > 0 and r1["within_step"]["sequence_also_equal"] is False
    g8 = _check(res, "G8")
    assert g8["status"] == "PASS" and g8["steps_with_emission_out_of_canonical_order"] == 1
    assert len(g8["permutations"]) == 1 and g8["permutations"][0]["same_dag_multiset"] and g8["permutations"][0]["displaced_query_positions"] == [3]
    assert any(l.get("emitted_out_of_canonical_order") for l in g8["batch_layout"])
    g7 = _check(res, "G7")
    assert g7["boundary_probes"] > 0 and g7["boundary_probe_failures"] == 0   # canonical addresses resolve (structural, not emission index)
    r0 = next(e for e in res["requests"] if e["request_id"] == "r0")
    assert r0["within_step"]["permuted_instances"] == 0 and r0["within_step"]["sequence_also_equal"]


def test_dag_criterion_negatives_wrong_arrival_swapped_outputs_wiring(memory, tmp_path):
    # wrong declared arrival (the Program says 0, the record ran it at 1) -> G6, whatever the within-step criterion
    gp, oracle = assemble([R0, RX_LATE], tmp_path / "a", tgrouped=True)
    next(r for r in gp["requests"] if r["request_id"] == "r1")["arrive_step"] = 0
    res = gm.check(gp, oracle, within_step="dag")
    assert res["verdict"] == "FAIL" and "G6" in res["failed_checks"]
    assert any("declared arrival step 0 != observed 1" in p for p in _check(res, "G6")["problems"])
    # swapped request outputs -> G6 (output association)
    gp, oracle = assemble([R0, RX_LATE], tmp_path / "b", tgrouped=True)
    tj = json.load(open(f"{oracle}/capture/tokens.json"))
    a, b = tj["requests"]
    a["generated_token_ids"], b["generated_token_ids"] = b["generated_token_ids"], a["generated_token_ids"]
    json.dump(tj, open(f"{oracle}/capture/tokens.json", "w"))
    res = gm.check(gp, oracle, within_step="dag")
    assert res["verdict"] == "FAIL" and "G6" in res["failed_checks"] and any("output association" in p for p in _check(res, "G6")["problems"])
    # a permutation that also changes the wiring (attention reading another request's KV row) -> G5 + G3, not a layout
    gp, oracle = assemble([R0, R1], tmp_path / "c", leak=True)
    res = gm.check(gp, oracle, within_step="dag")
    assert res["verdict"] == "FAIL" and _check(res, "G5")["status"] == "FAIL" and _check(res, "G3")["status"] == "FAIL"
    # swapped requests (component of the wrong request) -> G3 + G4 under dag as well (unmatched instances: no canonical address)
    gp, oracle = assemble([R0, R1], tmp_path / "d", tgrouped=True)
    r0, r1 = gp["requests"]
    r1["component"] = dict(r0["component"])
    res = gm.check(gp, oracle, within_step="dag")
    assert res["verdict"] == "FAIL" and _check(res, "G3")["status"] == "FAIL" and _check(res, "G4")["status"] == "FAIL"
    assert any("no structurally identical component instance" in p for p in _check(res, "G4")["problems"])


def test_consumer_before_producer_emission_is_not_a_layout_it_fails_under_both_criteria(memory, tmp_path):
    # the same DAG, one instance of r1 emitted before the instance it reads (rev-pm M-0042 condition 2)
    gp, oracle = assemble([R0, R1], tmp_path, tgrouped=True)
    c_new, p_new = move_consumer_before_producer(oracle, "r1")
    assert c_new < p_new
    for crit in ("sequence", "dag"):
        res = gm.check(gp, oracle, within_step=crit)
        assert res["verdict"] == "FAIL", (crit, res["failed_checks"])
        assert "G3" in res["failed_checks"] and "G8" in res["failed_checks"], (crit, res["failed_checks"], res["reasons"])
        g8 = _check(res, "G8")
        assert g8["dependency_order_violations"] >= 1 and any("consumer-before-producer" in p for p in g8["problems"])
        if crit == "dag":
            assert "G4" in res["failed_checks"]   # the consumer's Merkle hash names a forward reference: no structurally identical component instance
            assert any("no structurally identical component instance" in p for p in _check(res, "G4")["problems"])
            fd = next(e for e in res["requests"] if e["request_id"] == "r1")["first_divergence"]
            assert fd["forward_refs_in_fold"] >= 1
    # control: the un-mutated record passes under dag (and sequence: R0/R1 are not co-scheduled prefills)
    gp, oracle = assemble([R0, R1], tmp_path / "ctl", tgrouped=True)
    assert gm.check(gp, oracle, within_step="dag")["verdict"] == "PASS"


def test_mis_emitted_sampler_is_named_as_a_segmentation_disagreement_not_a_resegmented_request(memory, tmp_path):
    # rev-pm M-0171 fixture (3): r1's sampler re-emitted at the head of its engine step.  The request-local steps are segmented at the
    # request's TokenSelects, so the mutation re-segments r1; G8 cross-checks that segmentation against the engine step the fold rows
    # record for every instance and names the disagreement (FAIL under both criteria, as before -- now with the cause stated).
    gp, oracle = assemble([R0, R1_LATE], tmp_path, tgrouped=True)
    c_new, p_new = move_consumer_before_producer(oracle, "r1", sampler_to_head=True)
    assert c_new < p_new
    for crit in ("dag", "sequence"):
        res = gm.check(gp, oracle, within_step=crit)
        assert res["verdict"] == "FAIL" and "G8" in res["failed_checks"], (crit, res["failed_checks"], res["reasons"])
        g8 = _check(res, "G8")
        assert g8["segmentation"]["disagreements"] and any("segmentation disagrees" in p for p in g8["problems"]), g8["problems"]
        assert g8["segmentation"]["disagreements"][0]["request_id"] == "r1"
    # control: the un-mutated staggered record has 0 disagreements and PASSes under dag
    gp, oracle = assemble([R0, R1_LATE], tmp_path / "ctl", tgrouped=True)
    res = gm.check(gp, oracle, within_step="dag")
    assert res["verdict"] == "PASS", res["reasons"]
    assert _check(res, "G8")["segmentation"]["disagreements"] == []


def test_workload_applicability_refuses_a_row_the_components_do_not_model():
    # COORD M-0063 (1): the workload Program declares the execution profile + engine conditions it models and refuses other rows
    from verity_vllm.pipeline import global_program as G
    comps = {(256, 31): {"execution": "bi-eager", "numerics_profile": "b1-eager-v3", "gemm_target": "ada"},
             (11, 1): {"execution": "bi-eager", "numerics_profile": "b1-eager-v3", "gemm_target": "ada"}}
    ok = G.workload_applicability({"sweep": {"row_id": "x__b2__i256__o32__mixed__greedy__bi-eager"}}, comps)
    assert ok["refusal"] is None and ok["execution"] == "bi-eager" and ok["row_execution"] == "bi-eager"
    assert ok["engine"]["enforce_eager"] is True and ok["engine"]["enable_prefix_caching"] is False
    assert ok["components"]["numerics_profile"] == "b1-eager-v3"
    bad = G.workload_applicability({"sweep": {"row_id": "x__b2__i256__o32__mixed__greedy__bi-compiled"}}, comps)
    assert bad["refusal"] and "'bi-compiled' is not modelled" in bad["refusal"]
    assert G.execution_of_workload({"execution": "bi-compiled", "sweep": {"row_id": "x__bi-eager"}}) == "bi-compiled"   # the field wins
    mixed = dict(comps)
    mixed[(11, 1)] = dict(comps[(11, 1)], execution="bi-compiled")
    assert "different execution profiles" in G.workload_applicability({"sweep": {"row_id": "x__bi-eager"}}, mixed)["refusal"]
    assert "no execution profile" in G.workload_applicability({"sweep": {"row_id": "x__bi-eager"}}, {k: {} for k in comps})["refusal"]
    dis = dict(comps)
    dis[(11, 1)] = dict(comps[(11, 1)], gemm_target="hopper")
    assert "disagree on applicability fields ['gemm_target']" in G.workload_applicability({"sweep": {"row_id": "x__bi-eager"}}, dis)["refusal"]


def test_gp01_component_provenance_refuses_a_descriptor_that_is_not_the_derived_program(tmp_path):
    # rev-pm M-0123 (E): a component descriptor edited after the derive composed silently to another workload digest -> now refused, named
    from verity.ml.circuits.codec import program_digest
    from verity_vllm.pipeline import global_program as G
    desc = {"schema": "toy/descriptor", "program_fn": "Toy_v1", "definitions": {"Toy_v1": {"form": "composite", "body": {"nodes": [1, 2]}}},
            "root": "Toy_v1", "annotations": {"free": "text"}}
    d = tmp_path / "build_request_LP3_T2"
    d.mkdir()
    (d / "result.json").write_text(json.dumps({"ok": True, "program": {"digest": program_digest(desc)}}))
    ok = G.component_provenance(str(d), desc)
    assert ok["refusal"] is None and ok["descriptor_digest"] == ok["recorded_digest"] == program_digest(desc)
    tampered = dict(desc, definitions={"Toy_v1": {"form": "composite", "body": {"nodes": [2, 1]}}})
    bad = G.component_provenance(str(d), tampered)
    assert bad["refusal"] and "not the Program derive_step built" in bad["refusal"]
    annotated = dict(desc, annotations={"free": "other text"})          # annotations are outside the digest: not a tamper
    assert G.component_provenance(str(d), annotated)["refusal"] is None
    (d / "result.json").unlink()
    assert "no provenance" in G.component_provenance(str(d), desc)["refusal"]
    (d / "artifact.json").write_text(json.dumps({"program_digest": program_digest(desc)}))   # artifact.json is the other recorded place
    assert G.component_provenance(str(d), desc)["refusal"] is None


def test_from_gprog_recomputes_the_build_digest_from_the_descriptor_and_refuses_an_edited_one(tmp_path):
    # rev-pm M-0112: the artifact's workload_digest is a claim; the checker verifies it against the descriptor GP-01 wrote beside it
    from verity.ml.circuits.codec import canonical_json, program_digest
    desc = {"schema": "toy/descriptor", "definitions": [{"name": "D0"}], "root": {"body": []}, "annotations": {"free": "text"}}
    d = tmp_path / "build_workload"
    d.mkdir()
    with gzip.open(d / "descriptor.json.gz", "wb") as f:
        f.write(canonical_json(desc))
    art = {"schema": "verity/workload-program/v1", "ok": True, "workload_digest": program_digest(desc),
           "descriptor": {"sha256": __import__("hashlib").sha256(canonical_json(desc)).hexdigest(), "bytes": 1}, "requests": [], "inputs": {"derived": {}},
           "address_map": {"requests": [], "convention": "toy"}, "chronology": {"requests": []}}
    p = d / "workload_program.json"
    p.write_text(json.dumps(art))
    gp = gm.from_gprog(str(p))
    assert gp["workload_digest"] == program_digest(desc)
    assert gp["gprog_artifact"]["digest_verification"]["verified"] is True
    # the digest field edited (zeroed) with the descriptor intact -> refused, named
    bad = dict(art, workload_digest="0" * 64)
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="not the Program it names"):
        gm.from_gprog(str(p))
    # the descriptor swapped under an intact digest -> refused (the digest no longer names the descriptor)
    p.write_text(json.dumps(art))
    with gzip.open(d / "descriptor.json.gz", "wb") as f:
        f.write(canonical_json(dict(desc, definitions=[{"name": "D1"}])))
    with pytest.raises(ValueError, match="not the Program it names"):
        gm.from_gprog(str(p))
    # no descriptor beside the artifact (an evidence dir pulled without it): accepted but recorded as UNVERIFIED, never as verified
    (d / "descriptor.json.gz").unlink()
    gp = gm.from_gprog(str(p))
    assert gp["gprog_artifact"]["digest_verification"]["verified"] is False


def test_from_gprog_relocates_component_dirs_beside_a_staged_build_artifact(tmp_path):
    """[lane/vllm-v2-gm-scale] A `vllm-build/v1` records the building pod's absolute component paths (`/workspace/cp/<sweep>/<row>/
    build_request_LP<lp>_T<t>`).  Staged elsewhere (another pod; pod_match_v2.sh's symlinked attempt dir) those paths dangle while the
    same dirs sit beside the artifact -- the v2 acquisition's B64 GM-01 died on exactly this FileNotFoundError.  from_gprog resolves them
    beside `<row>/build_workload/workload_program.json`, records what moved, and leaves an existing path / a path missing at both places
    alone (the loader's error must still name the recorded path)."""
    from verity.ml.circuits.codec import canonical_json, program_digest
    desc = {"schema": "toy/descriptor", "definitions": [{"name": "D0"}], "root": {"body": []}}
    row = tmp_path / "staged" / "row"
    bw = row / "build_workload"
    bw.mkdir(parents=True)
    with gzip.open(bw / "descriptor.json.gz", "wb") as f:
        f.write(canonical_json(desc))
    (row / "build_request_LP1024_T127").mkdir()            # beside the artifact: the component the Build named
    present = tmp_path / "elsewhere" / "build_request_LP7_T3"   # a recorded path that DOES exist here: untouched
    present.mkdir(parents=True)
    derived = {"1024,127": "/workspace/cp/sweep-v2-5a/row/build_request_LP1024_T127",   # dangling -> relocated
               "7,3": str(present),                                                       # exists -> kept
               "5,1": "/workspace/cp/sweep-v2-5a/row/build_request_LP5_T1"}              # missing at both places -> kept as recorded
    art = {"schema": "verity/workload-program/v1", "ok": True, "workload_digest": program_digest(desc), "requests": [],
           "inputs": {"derived": derived}, "address_map": {"requests": [], "convention": "toy"}, "chronology": {"requests": []}}
    p = bw / "workload_program.json"
    p.write_text(json.dumps(art))
    gp = gm.from_gprog(str(p))
    reloc = gp["gprog_artifact"]["relocated_components"]
    assert set(reloc) == {"1024,127"}
    assert reloc["1024,127"] == {"from": derived["1024,127"], "to": str(row / "build_request_LP1024_T127")}
    # the same resolution the requests' component dirs get (from_gprog looks the "lp,t" key up in the resolved map)
    r = GD.relocate_components(derived, str(p))
    assert r["derived"]["1024,127"] == str(row / "build_request_LP1024_T127")
    assert r["derived"]["7,3"] == str(present) and r["derived"]["5,1"] == derived["5,1"]
    assert r["base"] == str(row)
    # pod_match_v2.sh's layout: the attempt dir holds SYMLINKS to the staged build_* dirs; abspath (not realpath) keeps the artifact's
    # own directory as the base, so the symlinked siblings are the ones found
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    for name in ("build_workload", "build_request_LP1024_T127"):
        os.symlink(row / name, attempt / name)
    r2 = GD.relocate_components(derived, str(attempt / "build_workload" / "workload_program.json"))
    assert r2["derived"]["1024,127"] == str(attempt / "build_request_LP1024_T127")
    # nothing to move -> an empty record (a document of a Build read where it was built is unchanged in content)
    assert GD.relocate_components({"7,3": str(present)}, str(p))["relocated"] == {}


def test_impl_module_defaults_to_the_match_of_record_and_names_what_runs(monkeypatch):
    """[lane/vllm-v2-gm-scale] MATCH_IMPL unset -> `fast` (the Match of record, row_pod.sh's export); `record` still selects the audit
    checker explicitly; an unknown name is an error.  The CLI banner names the implementation before the run."""
    monkeypatch.delenv(gm.IMPL_ENV, raising=False)
    assert gm.IMPL_DEFAULT == "fast"
    assert gm.impl_module() is GMF
    assert gm.impl_module("record") is gm and gm.impl_module("fast") is GMF
    monkeypatch.setenv(gm.IMPL_ENV, "record")
    assert gm.impl_module() is gm
    monkeypatch.setenv(gm.IMPL_ENV, "")
    assert gm.impl_module() is GMF                       # empty = unset
    monkeypatch.setenv(gm.IMPL_ENV, "quick")
    with pytest.raises(ValueError, match="not one of"):
        gm.impl_module()
    monkeypatch.delenv(gm.IMPL_ENV, raising=False)
    b = gm.impl_banner(GMF, {"gprog_artifact": {"relocated_components": {"1024,127": {"from": "/workspace/cp/x/build_request_LP1024_T127", "to": "/y"}}}})
    assert b.startswith("# global_match impl=fast") and "unset -> default fast" in b and "workers=" in b and "relocated_components=1" in b
    assert gm.impl_banner(gm).startswith("# global_match impl=record") and "audit checker" in gm.impl_banner(gm)


def _real_workload_descriptor(lag: int | None):
    """A REAL workload Program through verity_vllm.program.workload + the descriptor codec (the toy request Programs of tests/program/
    test_workload_program.py): the `lag` static lands where GP-01's descriptor.json.gz carries it -- inlined in the program_fn spec id and the
    root id, NOT a root `bindings` entry.  lag=None: a pre-R17-1 shaped descriptor (the static stripped from both ids)."""
    from verity.ml.circuits.codec import encode_program
    from verity.ml.circuits.defs import REGISTRY
    from verity.ml.circuits.program import Program
    from verity_vllm.program import workload as W
    from tests.program.test_workload_program import _toy_request, _WP
    reqs = []
    for i, (LP, T) in enumerate([(3, 2), (5, 0)]):
        c = W.request_component(encode_program(Program(_toy_request(LP, T))), REGISTRY, _WP(LP))
        reqs.append(W.WorkloadRequest(f"r{i}", i, LP, T + 1, 0, c, T=T))
    wl = W.WorkloadProgram(reqs, manifest_digest="m" * 8, lag=lag if lag is not None else 0)
    desc = encode_program(wl.program)
    if lag is None:
        desc = dict(desc, program_fn=desc["program_fn"].replace(",lag=0}", "}"), root=desc["root"].replace(",lag=0}]", "}]"))
        desc["definitions"] = {(k.replace(",lag=0}]", "}]") if k.startswith("Root[") else k): v for k, v in desc["definitions"].items()}
    return wl, desc


def test_from_gprog_reads_the_lag_static_where_the_codec_writes_it_and_refuses_an_edited_lag_block(tmp_path):
    """[R17-1] (mechanism row OLMoE B2 16:43Z: the first real Build crashed GM-01 here -- the mock-shaped reader looked for a root
    `bindings.lag` the codec never writes).  The static is inlined in program_fn / the root id; both inside the verified digest."""
    from verity.ml.circuits.codec import canonical_json, program_digest
    for lag in (0, 1):
        wl, desc = _real_workload_descriptor(lag)
        assert desc["program_fn"].endswith(f",lag={lag}}}") and desc["root"].endswith(f",lag={lag}}}]_v1"), (desc["program_fn"][-30:], desc["root"][-30:])
        assert GD.workload_lag_static(desc) == lag
        d = tmp_path / f"lag{lag}" / "build_workload"; d.mkdir(parents=True)
        with gzip.open(d / "descriptor.json.gz", "wb") as f:
            f.write(canonical_json(desc))
        art = {"schema": "verity/workload-program/v1", "ok": True, "workload_digest": program_digest(desc),
               "descriptor": {"sha256": __import__("hashlib").sha256(canonical_json(desc)).hexdigest(), "bytes": 1}, "requests": [], "inputs": {"derived": {}},
               "address_map": {"requests": [], "convention": "toy"}, "chronology": {"requests": []},
               "lag": {"value": lag, "async_scheduling": bool(lag)}, "stop_rule": {"eos_token_ids": [3], "lag": lag, "eos_lag_rows": lag}}
        p = d / "workload_program.json"; p.write_text(json.dumps(art))
        gp = gm.from_gprog(str(p))
        assert gp["gprog_artifact"]["digest_verification"] == {**gp["gprog_artifact"]["digest_verification"], "verified": True, "lag_static": lag}
        # the lag block edited to the other value under the intact Program -> refused by name (the block is a claim, the static is the Program)
        p.write_text(json.dumps(dict(art, lag={"value": 1 - lag, "async_scheduling": not lag})))
        with pytest.raises(ValueError, match="lag declaration is not the Program's"):
            gm.from_gprog(str(p))
        # the block deleted -> refused too (a Program that declares a lag must carry the block GM-01 consumes)
        p.write_text(json.dumps({k: v for k, v in art.items() if k != "lag"}))
        with pytest.raises(ValueError, match="lag declaration is not the Program's"):
            gm.from_gprog(str(p))
    # a pre-R17-1 shaped descriptor: no static anywhere -> None; an artifact CLAIMING a lag over it is refused
    _, old = _real_workload_descriptor(None)
    assert GD.workload_lag_static(old) is None
    # program_fn and root disagreeing on the static -> not one Program
    _, dz = _real_workload_descriptor(1)
    with pytest.raises(ValueError, match="not one Program"):
        GD.workload_lag_static(dict(dz, root=dz["root"].replace(",lag=1}]", ",lag=0}]")))
    # [rev S-r17-03 n2] an explicit root bindings.lag (mock form) disagreeing with the spec-id static -> refused too
    dd = {k: (dict(v, bindings={**v.get("bindings", {}), "lag": 0}) if k == dz["root"] else v) for k, v in dz["definitions"].items()}
    with pytest.raises(ValueError, match="bindings.lag 0 != the spec-id lag static 1"):
        GD.workload_lag_static(dict(dz, definitions=dd))
    dd1 = {k: (dict(v, bindings={**v.get("bindings", {}), "lag": 1}) if k == dz["root"] else v) for k, v in dz["definitions"].items()}
    assert GD.workload_lag_static(dict(dz, definitions=dd1)) == 1


def test_record_engine_conditions_outside_the_declared_applicability_fail_g1(memory, tmp_path):
    from verity_vllm.pipeline import global_program as G
    # [R17-1] this test rewrites headers WITHOUT async_scheduling to exercise G1's other engine facts; a lag-declaring Program would (rightly)
    # also fail G6 on every such header, so the Program here declares no lag (pre-R17-1 mock) and no request stops before its cap
    gp, oracle = assemble([R0, R1], tmp_path, lag=None)
    gp["applicability"] = {"execution": "bi-eager", "row_execution": "bi-eager", "engine": dict(G.MODELLED_ENGINE), "refusal": None}

    def set_header(profile_engine=None, profile_compilation=None, top_engine=None, top_compilation=None):
        """Rewrite the RunHeader's engine facts at the RECORDER'S location (profile.*) and/or the top level; None = absent."""
        p = f"{oracle}/capture/log.jsonl.gz"
        lines = gzip.open(p, "rt").read().splitlines()
        for i, line in enumerate(lines):
            if line.startswith('{"$e":"RunHeader"'):
                d = json.loads(line)
                prof = dict(d.get("profile") or {})
                prof.pop("engine_kwargs", None), prof.pop("compilation", None)
                if profile_engine is not None:
                    prof["engine_kwargs"] = profile_engine
                if profile_compilation is not None:
                    prof["compilation"] = profile_compilation
                d["profile"] = prof
                d.pop("engine_kwargs", None), d.pop("compilation", None)
                if top_engine is not None:
                    d["engine_kwargs"] = top_engine
                if top_compilation is not None:
                    d["compilation"] = top_compilation
                lines[i] = json.dumps(d, separators=(",", ":"))
        with gzip.open(p, "wt") as f:
            f.write("\n".join(lines) + "\n")

    eager_ek = {"enforce_eager": True, "enable_prefix_caching": False, "max_num_seqs": 2}
    eager_comp = {"mode": "0", "cudagraph_mode": "NONE", "enforce_eager": True}
    # the recorder's shape (the toy preamble writes it): facts read from RunHeader.profile.*, PASS
    res = gm.check(gp, oracle)
    assert res["verdict"] == "PASS" and res["execution"]["effective_engine"] == {"enforce_eager": True, "enable_prefix_caching": False, "async_scheduling": False}
    assert "RunHeader.profile.engine_kwargs" in res["execution"]["source"]["enforce_eager"]
    assert res["execution"]["effective_compilation"] == {"mode": "NONE", "cudagraph_mode": "NONE"}
    # [rev-pm M-0261 / F-pm-21] (a) cross-request cache reuse switched on AT THE RECORDER'S LOCATION: outside the modelled engine, FAIL
    set_header(profile_engine=dict(eager_ek, enable_prefix_caching=True), profile_compilation=eager_comp)
    res = gm.check(gp, oracle)
    assert res["verdict"] == "FAIL" and res["failed_checks"] == ["G1"]
    assert any("enable_prefix_caching=True is outside the Program's declared applicability" in p for p in _check(res, "G1")["problems"])
    # (b) a compiled record (enforce_eager false + the compiled record's compilation block) against the eager Program: FAIL, named twice
    set_header(profile_engine=dict(eager_ek, enforce_eager=False, compilation_config={"cudagraph_mode": "PIECEWISE"}),
               profile_compilation={"mode": "3", "cudagraph_mode": "PIECEWISE", "enforce_eager": False, "capture_sizes": [1, 2]})
    res = gm.check(gp, oracle)
    pr = _check(res, "G1")["problems"]
    assert res["verdict"] == "FAIL" and any("enforce_eager=False is outside" in p for p in pr)
    assert any("cudagraph_mode='PIECEWISE': the record attests a compiled execution" in p for p in pr)
    # (c) a header that attests NOTHING about its engine: fail closed (the old checker PASSed with 'not recorded')
    set_header()
    res = gm.check(gp, oracle)
    assert res["verdict"] == "FAIL" and res["failed_checks"] == ["G1"] and res["execution"]["effective_engine"] is None
    assert any("does not attest its engine conditions" in p and "fail closed" in p for p in _check(res, "G1")["problems"])
    # (d) control: facts at the top level only (foreign/older header) are read and compared too
    set_header(top_engine={"enforce_eager": True, "enable_prefix_caching": False, "enable_chunked_prefill": True})
    res = gm.check(gp, oracle)
    assert res["verdict"] == "PASS" and res["execution"]["effective_engine"] == {"enforce_eager": True, "enable_prefix_caching": False, "enable_chunked_prefill": True}
    set_header(top_engine={"enforce_eager": True, "enable_prefix_caching": True})
    res = gm.check(gp, oracle)
    assert res["verdict"] == "FAIL" and any("enable_prefix_caching=True is outside" in p for p in _check(res, "G1")["problems"])
    # (e) the record's own locations disagree: a record that does not attest one execution, named (and the offending value still compared)
    set_header(profile_engine=eager_ek, profile_compilation=eager_comp, top_engine={"enforce_eager": True, "enable_prefix_caching": True})
    res = gm.check(gp, oracle)
    pr = _check(res, "G1")["problems"]
    assert res["verdict"] == "FAIL" and any("own sources disagree on enable_prefix_caching" in p for p in pr)
    assert any("enable_prefix_caching=True is outside" in p for p in pr)
    # profile.compilation.enforce_eager is an engine fact too: it must agree with profile.engine_kwargs.enforce_eager
    set_header(profile_engine=eager_ek, profile_compilation=dict(eager_comp, enforce_eager=False))
    res = gm.check(gp, oracle)
    assert res["verdict"] == "FAIL" and any("own sources disagree on enforce_eager" in p for p in _check(res, "G1")["problems"])
    # [rev-pm M-0464] (g) a knob the Program declares but the record omits EVERYWHERE (prefix caching, vLLM V1 default ON) is not
    # applicability by absence: fail closed per declared key, named
    set_header(profile_engine={"enforce_eager": True, "max_num_seqs": 2}, profile_compilation=eager_comp)
    res = gm.check(gp, oracle)
    assert res["verdict"] == "FAIL" and res["execution"]["effective_engine"] == {"enforce_eager": True}
    assert any("does not attest enable_prefix_caching, which the Program's applicability declares" in p for p in _check(res, "G1")["problems"])
    # (f) a Program without a declaration cannot be checked: fail closed, named
    set_header(profile_engine=eager_ek, profile_compilation=eager_comp)
    gp2 = dict(gp, applicability={"execution": "bi-eager", "row_execution": "bi-eager", "refusal": None})
    res = gm.check(gp2, oracle)
    assert res["verdict"] == "FAIL" and any("declares no applicability.engine" in p for p in _check(res, "G1")["problems"])
    # a Program that refused the row can never Match it
    gp["applicability"]["refusal"] = "row execution 'bi-compiled' is not modelled"
    res = gm.check(gp, oracle)
    assert res["verdict"] == "FAIL" and any("refused this row's applicability" in p for p in _check(res, "G1")["problems"])


def test_compiled_execution_is_modelled_only_from_components_that_declare_its_compilation_facts():
    # compiled M-0153 (4a) / M-0174: derive_step writes applicability.engine = profile.modelled_engine() (enforce_eager False +
    # compilation_mode/backend/cudagraph_mode/...) for a bi-compiled profile; GP-01 models the compiled execution iff every component
    # declares it with AGREEING facts; the workload engine block = enforce_eager False + the normalised facts. Eager rows untouched.
    from verity_vllm.pipeline import global_program as G
    eng = {"enforce_eager": False, "compilation_mode": "VLLM_COMPILE", "backend": "inductor", "cudagraph_mode": "PIECEWISE",
           "attention": "splitting op executed outside the CUDA graphs", "program_derivation": "eager module-forward trace"}
    comps = {(256, 31): {"execution": "bi-compiled", "numerics_profile": "b1-eager-v3", "gemm_target": "ada", "engine": dict(eng), "compilation": "piecewise"},
             (11, 1): {"execution": "bi-compiled", "numerics_profile": "b1-eager-v3", "gemm_target": "ada", "engine": dict(eng), "compilation": "piecewise"}}
    row = {"sweep": {"row_id": "x__b2__i256__o32__mixed__greedy__bi-compiled"}}
    ok = G.workload_applicability(row, comps)
    assert ok["refusal"] is None and ok["execution"] == "bi-compiled"
    assert ok["engine"]["enforce_eager"] is False and ok["engine"]["enable_prefix_caching"] is False
    assert ok["engine"]["compilation"] == {"mode": "VLLM_COMPILE", "backend": "inductor", "cudagraph_mode": "PIECEWISE"}   # normalised keys
    # the label without any compilation facts: refused (the Program cannot state the compiled execution it models)
    bare = {k: {kk: vv for kk, vv in v.items() if kk not in ("engine", "compilation")} for k, v in comps.items()}
    assert "no compilation facts" in G.workload_applicability(row, bare)["refusal"]
    # a compilation dict of facts (no engine block) is accepted too
    facts_only = {k: dict(bare[k], compilation={"mode": 3, "cudagraph_mode": "PIECEWISE", "capture_sizes": [1, 2]}) for k in bare}
    f = G.workload_applicability(row, facts_only)
    # the CompilationMode enum VALUE (3) is spelled as its NAME so the record's own sources are comparable (M-0261)
    assert f["refusal"] is None and f["engine"]["compilation"] == {"mode": "VLLM_COMPILE", "cudagraph_mode": "PIECEWISE", "capture_sizes": [1, 2]}
    # facts that disagree between components: refused; an engine block saying eager under the compiled label: refused
    dis = {k: dict(v, engine=dict(v["engine"])) for k, v in comps.items()}
    dis[(11, 1)]["engine"]["cudagraph_mode"] = "FULL_AND_PIECEWISE"
    ref = G.workload_applicability(row, dis)["refusal"]
    assert "disagree" in ref and ("compilation facts" in ref or "engine" in ref)
    # [M-0174 (a)] the components' agreed engine block travels verbatim into the workload engine block (claim text and all), the
    # facts normalised beside it under `compilation`; GP-01's own compose-level conditions stay
    claimed = {k: dict(v, engine=dict(v["engine"], claim="boundary-level: module outputs correspond under graph replay")) for k, v in comps.items()}
    e = G.workload_applicability(row, claimed)["engine"]
    assert e["claim"].startswith("boundary-level") and e["enforce_eager"] is False and e["compilation"]["cudagraph_mode"] == "PIECEWISE"
    assert e["enable_prefix_caching"] is False
    eag = {k: dict(v, engine=dict(v["engine"], enforce_eager=True)) for k, v in comps.items()}
    assert "enforce_eager true" in G.workload_applicability(row, eag)["refusal"]
    # an eager row served by compiled components is still refused (the label rule), and eager components keep GP-01's eager engine block
    assert "'bi-eager' is not modelled" in G.workload_applicability({"sweep": {"row_id": "x__bi-eager"}}, comps)["refusal"]
    eager = {k: dict(v, execution="bi-eager", engine={"enforce_eager": True}) for k, v in comps.items()}
    e = G.workload_applicability({"sweep": {"row_id": "x__bi-eager"}}, eager)
    assert e["refusal"] is None and e["engine"] == G.MODELLED_ENGINE


def test_compiled_program_requires_the_record_to_attest_the_same_compilation_facts_g1(memory, tmp_path):
    from verity_vllm.pipeline import global_program as G
    gp, oracle = assemble([R0, R1], tmp_path, lag=None)     # [R17-1] headers below omit async_scheduling: G1's compilation facts are the subject
    decl = {"mode": "VLLM_COMPILE", "backend": "inductor", "cudagraph_mode": "PIECEWISE"}
    gp["applicability"] = {"execution": "bi-compiled", "row_execution": "bi-compiled", "refusal": None,
                           "engine": {**G.MODELLED_ENGINE, "enforce_eager": False, "compilation": decl}}

    def set_header(engine_kwargs=None, compilation=None):
        """The facts at the RECORDER'S location, RunHeader.profile.{engine_kwargs,compilation} (M-0261); None = absent."""
        p = f"{oracle}/capture/log.jsonl.gz"
        lines = gzip.open(p, "rt").read().splitlines()
        for i, line in enumerate(lines):
            if line.startswith('{"$e":"RunHeader"'):
                d = json.loads(line)
                prof = dict(d.get("profile") or {})
                prof.pop("engine_kwargs", None), prof.pop("compilation", None)          # replace, do not accumulate
                if engine_kwargs is not None:
                    prof["engine_kwargs"] = engine_kwargs
                if compilation is not None:
                    prof["compilation"] = compilation
                d["profile"] = prof
                lines[i] = json.dumps(d, separators=(",", ":"))
        with gzip.open(p, "wt") as f:
            f.write("\n".join(lines) + "\n")

    # the eager toy record (mode "0" / cudagraph NONE / enforce_eager true) against a compiled Program: outside its applicability, named
    res = gm.check(gp, oracle, record_alternate=False)
    assert res["verdict"] == "FAIL" and res["failed_checks"] == ["G1"]
    pr = _check(res, "G1")["problems"]
    assert any("enforce_eager=True is outside" in p for p in pr) and any("cudagraph_mode='NONE' is outside" in p for p in pr)
    # a header that attests nothing about compilation: a compiled Program cannot Match it (fail closed, named)
    set_header(engine_kwargs={"enforce_eager": False, "enable_prefix_caching": False})
    res = gm.check(gp, oracle, record_alternate=False)
    assert res["verdict"] == "FAIL" and any("carries no compilation facts" in p for p in _check(res, "G1")["problems"])
    # compiled's RunHeader (M-0174): profile.engine_kwargs carries enforce_eager False + compilation_config={cudagraph_mode: PIECEWISE} -> passes
    set_header(engine_kwargs={"enforce_eager": False, "enable_prefix_caching": False, "compilation_config": {"cudagraph_mode": "PIECEWISE"}})
    res = gm.check(gp, oracle, record_alternate=False)
    assert res["verdict"] == "PASS" and res["execution"]["effective_compilation"] == {"cudagraph_mode": "PIECEWISE"}
    # the recorder's profile.compilation block (rev-pm M-0261: mode VLLM_COMPILE, cudagraph PIECEWISE, sizes [1,2]) is read there, the
    # enum value spelling ("3") normalised to the name, compiled's key names (compilation_mode) normalised too
    set_header(engine_kwargs={"enforce_eager": False, "enable_prefix_caching": False},
               compilation={"mode": "3", "cudagraph_mode": "PIECEWISE", "backend": "inductor", "enforce_eager": False, "capture_sizes": [1, 2]})
    res = gm.check(gp, oracle, record_alternate=False)
    assert res["verdict"] == "PASS" and res["execution"]["effective_compilation"]["mode"] == "VLLM_COMPILE"
    assert res["execution"]["effective_compilation"]["capture_sizes"] == [1, 2]
    set_header(engine_kwargs={"enforce_eager": False, "enable_prefix_caching": False},
               compilation={"compilation_mode": "VLLM_COMPILE", "cudagraph_mode": "PIECEWISE", "backend": "inductor"})
    res = gm.check(gp, oracle, record_alternate=False)
    assert res["verdict"] == "PASS" and res["execution"]["effective_compilation"]["mode"] == "VLLM_COMPILE"
    # a different cudagraph mode / an eager header / no cudagraph_mode -> outside the declared applicability, named
    set_header(engine_kwargs={"enforce_eager": False, "enable_prefix_caching": False, "compilation_config": {"cudagraph_mode": "FULL_AND_PIECEWISE"}})
    res = gm.check(gp, oracle, record_alternate=False)
    assert res["verdict"] == "FAIL" and any("cudagraph_mode='FULL_AND_PIECEWISE' is outside" in p for p in _check(res, "G1")["problems"])
    set_header(engine_kwargs={"enforce_eager": True, "enable_prefix_caching": False, "compilation_config": {"cudagraph_mode": "PIECEWISE"}})
    res = gm.check(gp, oracle, record_alternate=False)
    assert res["verdict"] == "FAIL" and any("enforce_eager=True is outside" in p for p in _check(res, "G1")["problems"])
    set_header(engine_kwargs={"enforce_eager": False, "enable_prefix_caching": False}, compilation={"backend": "inductor"})
    res = gm.check(gp, oracle, record_alternate=False)
    assert res["verdict"] == "FAIL" and any("do not attest cudagraph_mode" in p for p in _check(res, "G1")["problems"])
    # two header locations that disagree with each other are a record defect, named
    set_header(engine_kwargs={"enforce_eager": False, "enable_prefix_caching": False, "compilation_config": {"cudagraph_mode": "PIECEWISE"}},
               compilation={"cudagraph_mode": "FULL"})
    res = gm.check(gp, oracle, record_alternate=False)
    assert res["verdict"] == "FAIL" and any("own sources disagree on cudagraph_mode" in p for p in _check(res, "G1")["problems"])


def test_default_criterion_is_dag_and_the_sequence_verdict_is_kept_beside_it(memory, tmp_path):
    # COORD ruling M-0177: dag is the criterion of record; the sequence verdict on the same record stays in the artifact, never merged
    assert gm.DEFAULT_WITHIN_STEP == "dag"
    gp, oracle = assemble([R0, RX_LATE], tmp_path, tgrouped=True)     # the M-ARRIVAL-ORDER shape
    res = gm.check(gp, oracle)
    assert res["within_step_criterion"] == "dag" and res["verdict"] == "PASS"
    alt = res["alternate_verdict"]
    assert alt["criterion"] == "sequence" and alt["verdict"] == "FAIL" and set(alt["failed_checks"]) >= {"G3", "G7"}
    assert any(r["request_id"] == "r1" and r["executed_steps_equal"] is False for r in alt["requests"])
    assert ("G3", "FAIL") in alt["checks"] and "alternate_verdict" not in alt        # the alternate is one level deep
    # the verdict of record is untouched by the alternate: the primary checks all PASS
    assert res["failed_checks"] == [] and all(k["status"] == "PASS" for k in res["checks"])
    # X-09's batch decomposition (the evidence inside, and row_pod's per-request comparator) applies the same criterion of record:
    # the displaced request is canonical-equal by the live-cone DAG multiset, its sequence inequality stays recorded
    x = res["x09"]
    assert x["verdict"] == "PASS" and x["within_step_criterion"] == "dag"
    xr1 = next(e for e in x["requests"] if e["request_id"] == "r1")
    assert xr1["canonical_equal"] is True and xr1["sequence_equal"] is False and xr1["dag_multiset_equal"] is True and "M-0177" in xr1["criterion"]
    # explicit sequence as the verdict of record keeps the dag verdict beside it; X-09 then fails the displaced request as in r14-v1
    res2 = gm.check(gp, oracle, within_step="sequence")
    assert res2["verdict"] == "FAIL" and res2["alternate_verdict"]["criterion"] == "dag" and res2["alternate_verdict"]["verdict"] == "PASS"
    assert res2["x09"]["verdict"] == "FAIL" and next(e for e in res2["x09"]["requests"] if e["request_id"] == "r1")["canonical_equal"] is False
    # and it can be switched off
    assert "alternate_verdict" not in gm.check(gp, oracle, record_alternate=False)


def test_dag_criterion_is_not_vacuous_the_mutations_still_fail(memory, tmp_path):
    gp, oracle = assemble([R0, R1], tmp_path, tgrouped=True)
    gp["requests"] = [r for r in gp["requests"] if r["request_id"] != "r1"]
    res = gm.check(gp, oracle, within_step="dag")
    assert res["verdict"] == "FAIL" and _check(res, "G1")["status"] == "FAIL" and _check(res, "G2")["status"] == "FAIL"
    gp, oracle = assemble([R0, R1], tmp_path / "b", tgrouped=True)
    gp["requests"].append(dict(gp["requests"][1]))
    res = gm.check(gp, oracle, within_step="dag")
    assert res["verdict"] == "FAIL" and _check(res, "G1")["status"] == "FAIL"


def test_gp01_stop_rule_is_read_from_the_pinned_checkpoint_and_the_manifest(tmp_path):
    """[F-pm-24] the Program's declared EOS set = union of the checkpoint's config.json / generation_config.json `eos_token_id` (int or
    list, Llama-3.2-Instruct declares [128001, 128008, 128009]) and the manifest's sampling.stop_token_ids; ignore_eos declares the EMPTY
    set (every early stop fails); no source at all -> None (GM-01 fails closed on early stops)."""
    from verity_vllm.pipeline import global_program as gp
    ck = tmp_path / "ckpt"
    ck.mkdir()
    json.dump({"eos_token_id": 128001}, open(ck / "config.json", "w"))
    json.dump({"eos_token_id": [128001, 128008, 128009]}, open(ck / "generation_config.json", "w"))
    sr = gp.stop_rule(str(ck), None, {"sampling": {"variant": "greedy"}})
    assert sr["eos_token_ids"] == [128001, 128008, 128009]
    assert sr["source"] == {"config.json": [128001], "generation_config.json": [128001, 128008, 128009]}
    sr = gp.stop_rule(str(ck), [7], {"sampling": {"stop_token_ids": [42]}})
    assert sr["eos_token_ids"] == [7, 42, 128001, 128008, 128009] and sr["source"]["manifest.sampling.stop_token_ids"] == [42]
    sr = gp.stop_rule(str(ck), None, {"sampling": {"ignore_eos": True}})
    assert sr["eos_token_ids"] == [] and sr["ignore_eos"] is True
    # [COORD M-0571 (1)] pinned INTO the manifest: the declaration of record; a checkpoint beside it must agree, else refused by name
    sr = gp.stop_rule(str(ck), None, {"sampling": {"eos_token_ids": [128009, 128001, 128008]}})
    assert sr["eos_token_ids"] == [128001, 128008, 128009] and sr["source"]["manifest.sampling.eos_token_ids"] == [128001, 128008, 128009] and not sr.get("refusal")
    sr = gp.stop_rule(None, None, {"sampling": {"eos_token_ids": [2]}})
    assert sr["eos_token_ids"] == [2] and sr["source"] == {"manifest.sampling.eos_token_ids": [2]}
    sr = gp.stop_rule(str(ck), None, {"sampling": {"eos_token_ids": [2]}})
    assert sr["eos_token_ids"] is None and "manifest sampling.eos_token_ids [2] != the pinned checkpoint's eos_token_id set [128001, 128008, 128009]" in sr["refusal"]
    sr = gp.stop_rule(None, None, {"sampling": {}})
    assert sr["eos_token_ids"] is None and "fails closed" in sr["note"]
    (ck / "generation_config.json").write_text("{not json")
    sr = gp.stop_rule(str(ck), None, None)
    assert sr["eos_token_ids"] == [128001] and sr["source"]["generation_config.json"].startswith("unreadable")
    # GM-01 reads exactly this: ignore_eos -> declared empty set -> an early stop is named, not waived
    gpm, oracle = assemble([R0, R1_EOS], tmp_path / "ign", eos_token_ids=None)
    gpm["stop_rule"] = {"eos_token_ids": [], "ignore_eos": True}
    res = gm.check(gpm, oracle)
    assert res["verdict"] == "FAIL" and any("not a declared EOS id []" in p for p in _check(res, "G6")["problems"])
    assert res["eos_verified"] == {"r1": False}


def test_gp01_binds_a_per_request_sampler_schedule_component_only_to_its_declared_arrival(tmp_path):
    """rev-pm M-0560 (1) on DECISION M-0529 (A): under a per-EVENT split schedule S (stochastic top-p) two requests with equal (LP, T) but
    different arrivals have different request Programs; a component that declares the arrival its schedule was computed for must not be
    reused for the other request, and a schedule computed against another manifest's chronology must not compose."""
    from verity_vllm.pipeline import global_program as G
    reqs = [{"request_id": "r0", "index": 0, "prompt_len": 8, "cap": 4, "arrive_step": 0},
            {"request_id": "r1", "index": 1, "prompt_len": 8, "cap": 4, "arrive_step": 2},
            {"request_id": "r2", "index": 2, "prompt_len": 5, "cap": 2, "arrive_step": 1}]
    live = G.declared_live_counts(reqs)
    assert live == [1, 2, 3, 2, 1, 1]
    def comp(name, sampling):
        d = tmp_path / name
        d.mkdir()
        (d / "result.json").write_text(json.dumps({"ok": True, "sampling": sampling}))
        return str(d)
    # greedy / constant-S components bind by shape (nothing to check)
    plain = comp("c_greedy", {"kind": "greedy"})
    assert G.sampler_schedule_binding(reqs, {(5, 1): plain})["refusal"] is None
    # the (8, 3) shape is shared by r0 (arrive 0) and r1 (arrive 2): a component computed for r0 must not bind to r1
    c0 = comp("c_r0", {"kind": "stochastic", "splits": [32, 32, 16, 32], "geometry": {"request": {"request_id": "r0", "arrive_step": 0, "cap": 4},
                                                                                        "declared_live_counts": live}})
    bad = G.sampler_schedule_binding(reqs, {(8, 3): c0, (5, 1): plain})
    assert bad["refusal"] and "binds r1 (arrive_step 2)" in bad["refusal"] and "one component per request" in bad["refusal"]
    # with only r0 of that shape (and the schedule computed from THAT chronology) it binds, and the checked table names the binding
    two = reqs[:1] + reqs[2:]
    c0b = comp("c_r0_two", {"kind": "stochastic", "splits": [32, 32, 16, 32], "geometry": {"request": {"request_id": "r0", "arrive_step": 0, "cap": 4},
                                                                                            "declared_live_counts": G.declared_live_counts(two)}})
    ok = G.sampler_schedule_binding(two, {(8, 3): c0b, (5, 1): plain})
    assert ok["refusal"] is None and ok["checked"]["8,3"]["bound_requests"] == ["r0"] and ok["checked"]["8,3"]["declared_arrive_step"] == 0
    # a schedule of the wrong length, and one computed from another manifest's chronology, are refused by name
    short = comp("c_short", {"kind": "stochastic", "splits": [32, 16], "geometry": {"request": {"request_id": "r0", "arrive_step": 0, "cap": 4}}})
    assert "S-schedule has 2 entries" in G.sampler_schedule_binding(reqs[:1], {(8, 3): short})["refusal"]
    other = comp("c_other", {"kind": "stochastic", "splits": [32, 32, 16, 32], "geometry": {"request": {"request_id": "r0", "arrive_step": 0, "cap": 4},
                                                                                             "declared_live_counts": [2, 2, 1, 1]}})
    assert G.declared_live_counts(reqs[:1]) == [1, 1, 1, 1]
    assert "derived against another workload" in G.sampler_schedule_binding(reqs[:1], {(8, 3): other})["refusal"]


def test_a_single_request_constant_s_component_binds_only_in_a_single_request_workload(tmp_path):
    """[single-request workloads] A component built for a single-request workload takes S = SplitsFor_v1(1, num_SMs) as a constant: it
    binds in a workload of one request, and a workload of several requests (whose S_t follows the live set) refuses it by name."""
    from verity_vllm.pipeline import global_program as G
    d = tmp_path / "c_one"
    d.mkdir()
    (d / "result.json").write_text(json.dumps({"ok": True, "sampling": {"kind": "stochastic", "splits": 32,
                                                                         "geometry": {"sampler_batch": 1, "num_sms": 142, "splits": 32}}}))
    one = [{"request_id": "r0", "index": 0, "prompt_len": 8, "cap": 4, "arrive_step": 0}]
    assert G.sampler_schedule_binding(one, {(8, 3): str(d)})["refusal"] is None
    three = one + [{"request_id": "r1", "index": 1, "prompt_len": 8, "cap": 4, "arrive_step": 2},
                   {"request_id": "r2", "index": 2, "prompt_len": 5, "cap": 2, "arrive_step": 1}]
    bad = G.sampler_schedule_binding(three, {(8, 3): str(d)})
    assert bad["refusal"] and "S = 32 is the single-request constant" in bad["refusal"] and "3 requests" in bad["refusal"]


def test_g1_ties_the_manifest_of_record_to_the_run_header_sha_and_sampling_fields(memory, tmp_path):
    """rev-pm M-0570 (2): GM-01 was blind to the manifest's sampling fields -- seeds swapped or a seed deleted in the manifest (GP-01
    recomposed: digest unchanged, seeds are Inputs) passed G1 by ids/prompts/caps/arrivals. G1 now requires RunHeader.workload.sha256 ==
    the manifest of record and compares each request's seed and the workload sampling policy with the header, by name."""
    R0s, R1s = dict(R0, seed=1901028918), dict(R1, seed=2097394646)
    gp, oracle = assemble([R0s, R1s], tmp_path)
    res = gm.check(gp, oracle)
    assert res["verdict"] == "PASS", res["reasons"]
    g1 = _check(res, "G1")
    assert g1["header_workload_sha256"] == g1["workload_doc_sha256"] != g1["workload_sha256"] and g1["sampling"]["manifest"]["variant"] == "greedy"
    wl_path = gp["workload"]["path"]
    comps = [r["component"]["dir"] for r in gp["requests"]]

    def recomposed(mutate):
        wl = json.load(open(wl_path))
        mutate(wl)
        json.dump(wl, open(wl_path, "w"))
        return gm.mock_global_program(wl_path, comps, eos_token_ids=[EOS])   # GP-01 recomposed on the edited manifest, as the attacker would

    def swap(wl):
        wl["requests"][0]["seed"], wl["requests"][1]["seed"] = wl["requests"][1]["seed"], wl["requests"][0]["seed"]
    res = gm.check(recomposed(swap), oracle)
    assert res["verdict"] == "FAIL" and "G1" in res["failed_checks"]
    pr = _check(res, "G1")["problems"]
    assert any("manifest of record differs from the header" in p for p in pr), pr
    assert any("r0: manifest declares seed=2097394646 but the run header records seed=1901028918" in p for p in pr), pr
    res = gm.check(recomposed(swap), oracle)          # swapped back: sha and seeds agree again
    assert res["verdict"] == "PASS", res["reasons"]

    def delete(wl):
        wl["requests"][1]["seed"] = None
    res = gm.check(recomposed(delete), oracle)
    pr = _check(res, "G1")["problems"]
    assert res["verdict"] == "FAIL" and any("no declared seed for r1 but the run header records seed=2097394646" in p for p in pr), pr

    def restore_and_repolicy(wl):
        wl["requests"][1]["seed"] = 2097394646
        wl["sampling"] = {"variant": "stoch-t0.8-p0.95"}
    res = gm.check(recomposed(restore_and_repolicy), oracle)
    pr = _check(res, "G1")["problems"]
    assert res["verdict"] == "FAIL"
    assert any("sampling variant='stoch-t0.8-p0.95' but the run header records variant='greedy'" in p for p in pr), pr
    assert any("sampling temperature=0.8 but the run header records temperature=0.0" in p for p in pr), pr
    lab = GD.manifest_sampling({"sampling": {"variant": "stoch-t0.8-p0.95-k20"}})
    assert {k: lab[k] for k in ("variant", "temperature", "top_p", "top_k")} == {"variant": "stoch-t0.8-p0.95-k20", "temperature": 0.8, "top_p": 0.95, "top_k": 20}
    assert "conflict" not in lab
    # [rev-pm M-0667 TIP N5] the driver consumes variants[variant] (vllm_adapter.load_workload), not the label: the table is the value of
    # record, and label / table / explicit fields / sweep params must agree -- a laundered header (sha recomputed over the edited doc) with
    # variants[v].top_p edited 0.95 -> 0.9 under the label p0.95 is caught by the manifest's OWN disagreement, fail closed, named
    tab = GD.manifest_sampling({"sampling": {"variant": "stoch-t0.8-p0.95"}, "variants": {"stoch-t0.8-p0.95": {"temperature": 0.8, "top_p": 0.9, "top_k": 0}}})
    assert tab["top_p"] == 0.9 and tab["conflict"] == ["top_p: label=0.95, variants_table=0.9"]
    agree = GD.manifest_sampling({"sampling": {"variant": "stoch-t0.8-p0.95"}, "variants": {"stoch-t0.8-p0.95": {"temperature": 0.8, "top_p": 0.95, "top_k": 0}},
                                  "sweep": {"sampling_params": {"top_p": 0.95}}})
    assert "conflict" not in agree and agree["sources"]["top_p"] == ["label", "sweep.sampling_params", "variants_table"]
    assert "not in the manifest's `variants` table" in GD.manifest_sampling({"sampling": {"variant": "stoch-t0.8-p0.95"}, "variants": {"greedy": {}}})["conflict"][0]

    # [rev-pm BLOCKER M-0717 (i)] a per-request variant pin added AFTER the run: the header has no per-request sampling record -> FAIL named
    def pin_r1_greedy(wl):
        wl["sampling"] = {"variant": "greedy"}
        wl["variants"] = {"greedy": {"temperature": 0.0, "top_p": 1.0, "top_k": 0}}
        wl["requests"][1]["variant"] = "greedy"
    res = gm.check(recomposed(pin_r1_greedy), oracle)
    pr = _check(res, "G1")["problems"]
    assert res["verdict"] == "FAIL" and any("r1: manifest pins variant 'greedy' for the request but the run header records no per-request sampling" in p for p in pr), pr

    def launder_table(wl):
        wl["sampling"] = {"variant": "greedy"}
        wl["variants"] = {"greedy": {"temperature": 0.5, "top_p": 1.0, "top_k": 0}}   # the table the driver would consume says 0.5 under the greedy label
    res = gm.check(recomposed(launder_table), oracle)
    pr = _check(res, "G1")["problems"]
    assert res["verdict"] == "FAIL" and any("not one sampling declaration -- temperature: label=0.0, variants_table=0.5" in p for p in pr), pr
    assert any("sampling temperature=0.5 but the run header records temperature=0.0" in p for p in pr), pr   # and the table value vs the header
    # a header without workload.sha256 cannot tie the manifest of record to the run: fail closed, named
    reqs = [R0, R1]
    d = tmp_path / "nosha"
    d.mkdir()
    wl2 = workload(reqs, d / "workload.json")
    oracle2 = batched(reqs, d / "batched", wl_sha=None)
    res = gm.check(gm.mock_global_program(wl2, comps, eos_token_ids=[EOS], lag=0), oracle2)
    assert res["verdict"] == "FAIL" and any("carries no workload.sha256" in p for p in _check(res, "G1")["problems"])


def test_execution_past_a_declared_eos_fails_g6(memory, tmp_path):
    """LIFTING-SPEC v0.3 s7 [P1] (i) / stoch M-0590: the served prefix n_r must equal the declared activity a_r in BOTH directions.
    r0 serves [9, 3, 1] to its cap; with 3 declared as an EOS id the Program calls r0 finished after event 1 and the two further
    events are execution past EOS -> FAIL named (n_r 3 > a_r 2). With the EOS set {11} (r0 never emits it) the same record passes."""
    gp, oracle = assemble([R0, R1], tmp_path, eos_token_ids=(3, EOS))
    res = gm.check(gp, oracle)
    assert res["verdict"] == "FAIL" and "G6" in res["failed_checks"], res["reasons"]
    pr = _check(res, "G6")["problems"]
    # [R17-1] the Program declares LAG 0 (the toy record attests async_scheduling False): the one row after the EOS is execution past EOS,
    # named with the two lengths (executed_prefix_len 3 > served_output_len 2 + LAG 0)
    assert any("r0: execution past EOS -- output event 1 is the declared EOS id 3 but 1 more row(s) were executed where the Program's declared "
               "scheduler (Program lag 0" in p and "computes LAG 0 after an EOS (executed_prefix_len 3 > served_output_len 2 + LAG 0)" in p for p in pr), pr
    r0 = next(e for e in res["requests"] if e["request_id"] == "r0")
    assert r0["eos_verified"] is False and r0["eos_passed_at_event"] == 1 and res["eos_verified"] == {"r0": False}
    assert r0["answer_events"] == 2 and r0["lag_rows"] == 1 and r0["eos_step"] == 1
    assert r0["served_output_len"] == 2 and r0["executed_prefix_len"] == 3 and r0["cap_exhausted"] is False
    gp2, oracle2 = assemble([R0, R1], tmp_path / "ok", eos_token_ids=(EOS,))
    assert gm.check(gp2, oracle2)["verdict"] == "PASS"


R1_LAG = {"id": "r1", "prompt": [4, 6], "sampled": [8, EOS, 77], "cap": 4}   # EOS at event 1, then ONE placeholder row (input = the EOS, sample 77 discarded)


_NULL = object()   # [R17-1 / revb N3] the key PRESENT with a JSON null: the recorder's read-back found no resolved bool (unattested, never a default)


def _set_async_scheduling(oracle, value):
    """Rewrite RunHeader.profile.engine_kwargs.async_scheduling on the persisted record; value None = the key absent (unattested); _NULL =
    the key present with null (resolved_async_scheduling returned None: the engine exposed no resolved bool)."""
    p = f"{oracle}/capture/log.jsonl.gz"
    lines = gzip.open(p, "rt").read().splitlines()
    for i, line in enumerate(lines):
        if line.startswith('{"$e":"RunHeader"'):
            d = json.loads(line)
            ek = dict((d.get("profile") or {}).get("engine_kwargs") or {})
            ek.pop("async_scheduling", None)
            if value is _NULL:
                ek["async_scheduling"] = None
            elif value is not None:
                ek["async_scheduling"] = value
            d["profile"] = dict(d.get("profile") or {}, engine_kwargs=ek)
            lines[i] = json.dumps(d, separators=(",", ":"))
    with gzip.open(p, "wt") as f:
        f.write("\n".join(lines) + "\n")


def _driver_stop_at(oracle, rid, n_answer, eos_step):
    """The driver's word for an EOS stop under async scheduling (pad's fixture M-0162: generated_token_ids ends in the EOS, finish_step = the
    EOS step, the lag row's sample is NOT reported)."""
    tp = os.path.join(oracle, "capture", "tokens.json")
    tk = json.load(open(tp))
    e = next(r for r in tk["requests"] if r["request_id"] == rid)
    e["generated_token_ids"] = e["generated_token_ids"][:n_answer]
    e["n_generated"], e["finish_step"], e["finish_reason"] = n_answer, eos_step, "stop"
    json.dump(tk, open(tp, "w"))


def _driver_claims(oracle, rid=None, lag=None, **fields):
    """[R17-1] the driver's tokens.json record-schema claims: doc.lag.value (the resolved fact as the driver wrote it) and per request the
    served_output_len / executed_prefix_len / cap_exhausted fields m1_capture writes."""
    tp = os.path.join(oracle, "capture", "tokens.json")
    tk = json.load(open(tp))
    if lag is not None:
        tk["lag"] = {"value": lag, "async_scheduling": bool(lag), "source": "test"}
    if rid is not None:
        next(r for r in tk["requests"] if r["request_id"] == rid).update(fields)
    json.dump(tk, open(tp, "w"))


def _r(res, rid):
    return next(e for e in res["requests"] if e["request_id"] == rid)


R1_EOS_X_EOS = {"id": "r1", "prompt": [4, 6], "sampled": [EOS, 5, EOS], "cap": 5}   # [rev F-r17-14] EOS at event 0, served on, a second EOS


def test_served_past_the_first_eos_fails_g6_by_name_f_r17_14(memory, tmp_path):
    """[rev F-r17-14 (M-0540)] the GM-01 negative for the first-EOS rule, the same record the Commit's executed_prefix_of_record refuses:
    r1 sampled [EOS, 5, EOS] (cap 5) under a LAG-1 Program on a record attesting async_scheduling True, the driver reporting all three as
    generated_token_ids with finish_reason 'stop'.  The Program's rule stops at the FIRST EOS: a_r = 1, so the two rows after it are
    execution past EOS (2 > LAG 1) and the driver's three-token output is not the record's answer -- G6 FAILS by name on both; the second
    EOS launders nothing (served_output_len 1, executed_prefix_len 3).  Continuation_v1 (pad R17-32) stops at the same index."""
    gp, oracle = assemble([R0, R1_EOS_X_EOS], tmp_path, lag=1)
    _set_async_scheduling(oracle, True)
    res = gm.check(gp, oracle)
    assert res["verdict"] == "FAIL" and "G6" in res["failed_checks"], res["reasons"]
    pr = _check(res, "G6")["problems"]
    assert any("r1: execution past EOS -- output event 0 is the declared EOS id 11 but 2 more row(s) were executed where the Program's declared "
               "scheduler (Program lag 1" in p and "computes LAG 1 after an EOS (executed_prefix_len 3 > served_output_len 1 + LAG 1)" in p for p in pr), pr
    assert any("r1: the driver's generated_token_ids has 3 tokens but the request's answer is 1 event(s) (up to its first declared EOS): the rows "
               "after the EOS are not output (a lag row sample reported)" in p for p in pr), pr
    r1 = _r(res, "r1")
    assert r1["served_output_len"] == 1 and r1["executed_prefix_len"] == 3 and r1["cap_exhausted"] is False
    assert r1["answer_events"] == 1 and r1["lag_rows"] == 2 and r1["eos_step"] == 0 and r1["eos_verified"] is False and r1["eos_passed_at_event"] == 0
    assert "eos_lag_rows_accepted" not in r1
    # the Commit's executed_prefix_of_record on the same served list: the same refusal, by name, from the same rule
    from verity_vllm.check import executed_prefix as EP
    e = EP.executed_prefix_of_record(["r1"], {"r1": [EOS, 5, EOS]}, caps={"r1": 5}, eos_of_record=[EOS], lag=1)
    assert not e["ok"] and "served past an EOS at index 0" in e["problems"][0] and "F-r17-14" in e["problems"][0]


def test_g6_consumes_the_programs_declared_lag_and_the_record_must_attest_the_same_scheduler(memory, tmp_path):
    """R17-1 / DP-pad-02 (b) (Daniel, BRIEF s4; supersedes the ruling-21 reader): LAG is the workload Program's DECLARED specialization
    parameter (mock: `lag=`; GP-01: the Workload_v1 `lag` static from TargetProfile.lag()).  vLLM's async scheduling computes ONE more row
    for a request after its EOS sample (input = the EOS, sample discarded by the driver): under LAG 1 that row is active internal
    computation -- its VUs required, its sampled token NOT a served output.  G6 consumes the Program's lag and nothing else; the record's
    attested async_scheduling (the engine's resolved read-back) must AGREE (G1 + G6 by name); nothing is inferred from the trace.  Per request
    the two lengths are kept distinct: served_output_len (a_r) and executed_prefix_len (a_r + LAG on an EOS stop before the cap).  FAILS by
    name: a LAG-0 header on a LAG-1 Program (and the reverse), an unattested record against a declaring Program, a deleted lag row, a second
    lag row, a reported lag sample, a driver finish_step on the lag row, a pre-R17-1 Program (no lag) with an EOS stop, VARIABLE lag."""
    # (1) LAG 1 Program, record attests async_scheduling True, the lag row served: PASS -- the row accepted and labelled, the lengths distinct
    gp, oracle = assemble([R0, R1_LAG], tmp_path / "lag1", lag=1)
    _set_async_scheduling(oracle, True)
    _driver_stop_at(oracle, "r1", n_answer=2, eos_step=1)
    res = gm.check(gp, oracle)
    assert res["verdict"] == "PASS", res["reasons"]
    r1 = _r(res, "r1")
    assert r1["finish"] == "eos" and r1["eos_verified"] is True and r1["eos_token"] == EOS and r1["eos_step"] == 1
    assert r1["answer_events"] == 2 and r1["lag_rows"] == 1 and r1["eos_lag_rows_accepted"] == 1 and r1["output_events"] == 3
    assert r1["served_output_len"] == 2 and r1["executed_prefix_len"] == 3 and r1["cap_exhausted"] is False      # the two lengths, distinct
    assert r1["token_selects"] == 3 and r1["events_fed_forward"] == 2        # the EOS is fed to the lag row; the lag sample is fed nowhere
    assert r1["eos_truncated"] and r1["unexecuted_steps"] == 1 and r1["finish_step"] == 2
    g6 = _check(res, "G6")
    assert g6["stop_rule"]["eos_lag_rows"] == 1 and g6["stop_rule"]["eos_lag_rows_declared"] == 1 and "Program lag 1" in g6["stop_rule"]["eos_lag_source"]
    assert g6["lag"] == {**g6["lag"], "declared": 1, "record_attests": 1, "agree": True, "exercised": True, "eos_stop_requests": ["r1"],
                         "lag_rows_observed": {"r1": 1}, "variable_lag": False}
    assert res["lag"]["declared"] == 1 and res["lag"]["exercised"] is True and res["lag"]["variable_lag"] is False and "alerts" not in res
    assert g6["lengths_by_request"]["r1"] == {"served_output_len": 2, "executed_prefix_len": 3, "cap": 4, "cap_exhausted": False}
    assert g6["lengths_by_request"]["r0"] == {"served_output_len": 3, "executed_prefix_len": 3, "cap": 3, "cap_exhausted": True}
    assert res["eos_truncated_requests"] == ["r1"] and res["eos_verified"] == {"r1": True}
    r0 = _r(res, "r0")   # r0 ran to its cap: the rule does not touch it
    assert r0["finish"] == "length" and r0["lag_rows"] == 0 and "eos_verified" not in r0 and "eos_lag_rows_accepted" not in r0

    # (2) LAG=0 HEADER on the LAG-1 trace under the LAG-1 Program: the record is not an execution of the declared scheduler -- G1 (engine fact)
    # and G6 (the lag it implies) FAIL by name; the value is never re-read from the trace
    _set_async_scheduling(oracle, False)
    res = gm.check(gp, oracle)
    assert res["verdict"] == "FAIL" and {"G1", "G6"} <= set(res["failed_checks"]), res["reasons"]
    assert any("record engine async_scheduling=False is outside the Program's declared applicability (async_scheduling=True)" in p for p in _check(res, "G1")["problems"])
    pr = _check(res, "G6")["problems"]
    assert any("the Program is specialized for LAG 1 (async_scheduling=True" in p and "but the record attests async_scheduling=False (RunHeader.profile.engine_kwargs) -> LAG 0"
               in p and "the header's LAG-0 attestation on a LAG-1 Program FAILS by name" in p for p in pr), pr
    assert _check(res, "G6")["lag"]["agree"] is False and res["lag"]["record_attests"] == 0 and res["lag"]["declared"] == 1
    # the EOS-stopped request itself is not established under the disagreement: named per request, its EOS evidence not accepted
    assert any("r1: EOS stop at engine step 1 on a record whose attested scheduler (async_scheduling=False (RunHeader.profile.engine_kwargs)) is not the "
               "Program's declared LAG 1: the 1 post-EOS row(s) are not accepted as the declared lag (R17-1)" in p for p in pr), pr
    assert _r(res, "r1")["eos_verified"] is False and "eos_lag_rows_accepted" not in _r(res, "r1")
    # ... and the reverse: a LAG-0 Program on a record attesting True (a lag-1 trace) fails by name too, with the lag row named as execution past EOS
    gpr, oracler = assemble([R0, R1_LAG], tmp_path / "lag0_program_on_lag1_record", lag=0)
    _set_async_scheduling(oracler, True)
    _driver_stop_at(oracler, "r1", n_answer=2, eos_step=1)
    resr = gm.check(gpr, oracler)
    assert resr["verdict"] == "FAIL" and {"G1", "G6"} <= set(resr["failed_checks"]), resr["reasons"]
    prr = _check(resr, "G6")["problems"]
    assert any("the Program is specialized for LAG 0 (async_scheduling=False" in p and "-> LAG 1" in p for p in prr), prr
    assert any("r1: execution past EOS" in p and "computes LAG 0 after an EOS (executed_prefix_len 3 > served_output_len 2 + LAG 0)" in p for p in prr), prr

    # (3) unattested (the key absent everywhere) against a declaring Program: refuse by name (G1 + G6), no default
    _set_async_scheduling(oracle, None)
    res = gm.check(gp, oracle)
    assert res["verdict"] == "FAIL" and {"G1", "G6"} <= set(res["failed_checks"])
    pr = _check(res, "G6")["problems"]
    assert any("the Program is specialized for LAG 1" in p and "but the record does not attest async_scheduling (not attested; RunHeader.profile.engine_kwargs / "
               "RunHeader.engine_kwargs / capture/versions.json)" in p and "refuse by name (R17-1)" in p for p in pr), pr
    assert _check(res, "G6")["lag"]["record_attests"] is None and _check(res, "G6")["lag"]["declared"] == 1
    gp0, oracle0 = assemble([R0, R1_EOS], tmp_path / "lag0_unattested", lag=0)
    _set_async_scheduling(oracle0, None)
    res0 = gm.check(gp0, oracle0)
    assert res0["verdict"] == "FAIL" and any("the Program is specialized for LAG 0" in p and "does not attest async_scheduling" in p
                                             for p in _check(res0, "G6")["problems"]), _check(res0, "G6")["problems"]
    # [revb M-0042 N3] the key PRESENT with null (the recorder's read-back found no resolved bool): the RESOLVED boolean is what is compared --
    # null is not False (LAG 0) and not True: refused by name, G1 (declared key unattested) and G6 (not a boolean)
    _set_async_scheduling(oracle0, _NULL)
    res0 = gm.check(gp0, oracle0)
    assert res0["verdict"] == "FAIL" and {"G1", "G6"} <= set(res0["failed_checks"]), res0["reasons"]
    assert any("does not attest async_scheduling, which the Program's applicability declares" in p for p in _check(res0, "G1")["problems"]), _check(res0, "G1")["problems"]
    assert any("the Program is specialized for LAG 0" in p and "does not attest async_scheduling" in p for p in _check(res0, "G6")["problems"]), _check(res0, "G6")["problems"]
    assert res0["lag"]["record_attests"] is None
    _set_async_scheduling(oracle0, False)   # the LAG-0 instance WITH the flag on record passes (the toy preamble's default)
    res0 = gm.check(gp0, oracle0)
    assert res0["verdict"] == "PASS" and _r(res0, "r1")["served_output_len"] == 2 and _r(res0, "r1")["executed_prefix_len"] == 2

    # (4) LAG 1 Program, True attested, but NO lag row served (R1_EOS): a deleted step, named with the lengths
    gpd, oracled = assemble([R0, R1_EOS], tmp_path / "deleted", lag=1)
    _set_async_scheduling(oracled, True)
    resd = gm.check(gpd, oracled)
    assert resd["verdict"] == "FAIL" and "G6" in resd["failed_checks"]
    assert any("r1: the Program's declared scheduler (Program lag 1" in p and "computes LAG 1 post-EOS row(s) after an EOS stop but 0 were executed after "
               "the EOS at engine step 1 (executed_prefix_len 2 < served_output_len 2 + LAG 1): a deleted step" in p
               for p in _check(resd, "G6")["problems"]), _check(resd, "G6")["problems"]
    assert _r(resd, "r1")["eos_verified"] is False

    # (5) two rows after the EOS under LAG 1: execution past EOS (executed_prefix_len 4 > served_output_len 2 + LAG 1)
    R1_LAG2 = dict(R1_LAG, sampled=[8, EOS, 77, 78], cap=6)
    gp2, oracle2 = assemble([R0, R1_LAG2], tmp_path / "lag2", lag=1)
    _set_async_scheduling(oracle2, True)
    _driver_stop_at(oracle2, "r1", n_answer=2, eos_step=1)
    res2 = gm.check(gp2, oracle2)
    assert res2["verdict"] == "FAIL" and any("r1: execution past EOS -- output event 1 is the declared EOS id 11 but 2 more row(s) were executed where "
                                             "the Program's declared scheduler (Program lag 1" in p and
                                             "computes LAG 1 after an EOS (executed_prefix_len 4 > served_output_len 2 + LAG 1)" in p
                                             for p in _check(res2, "G6")["problems"]), _check(res2, "G6")["problems"]

    # (6) the lag sample reported in the driver's generated_token_ids: not a served output, named (output association + the count)
    gp6, oracle6 = assemble([R0, R1_LAG], tmp_path / "leak", lag=1)
    _set_async_scheduling(oracle6, True)
    _driver_stop_at(oracle6, "r1", n_answer=3, eos_step=1)      # generated_token_ids = [8, 11, 77]
    res6 = gm.check(gp6, oracle6)
    assert res6["verdict"] == "FAIL" and any("r1: the driver's generated_token_ids has 3 tokens but the request's answer is 2 event(s)" in p
                                             and "a lag row sample reported" in p for p in _check(res6, "G6")["problems"]), _check(res6, "G6")["problems"]
    assert _r(res6, "r1")["eos_verified"] is False

    # (7) the driver's finish_step naming the lag row's step instead of the EOS step: named
    gp7, oracle7 = assemble([R0, R1_LAG], tmp_path / "finish", lag=1)
    _set_async_scheduling(oracle7, True)
    _driver_stop_at(oracle7, "r1", n_answer=2, eos_step=2)
    res7 = gm.check(gp7, oracle7)
    assert res7["verdict"] == "FAIL" and any("r1: driver finish_step 2 != observed 1 (the EOS step; the request's last served row is the lag row at 2)" in p
                                             for p in _check(res7, "G6")["problems"]), _check(res7, "G6")["problems"]

    # (8) a pre-R17-1 Program (no lag declared): an EOS stop FAILS by name -- the record's attestation is NOT read as the rule (True attested,
    # the lag row present: a ruling-21 reader would have passed it); with every request at its cap the record passes, labelled exercised=false
    from verity_vllm.pipeline import global_program as G
    assert G.eos_lag_rows_declared({"async_scheduling": True}) == 1 and G.eos_lag_rows_declared({"async_scheduling": False}) == 0
    assert G.eos_lag_rows_declared({}) is None and G.eos_lag_rows_declared(None) is None and G.eos_lag_rows_declared({"async_scheduling": "maybe"}) is None
    gp8, oracle8 = assemble([R0, R1_LAG], tmp_path / "undeclared", lag=None)
    assert gp8["lag"] is None and "lag" not in gp8["stop_rule"] and "async_scheduling" not in gp8["applicability"]["engine"]
    _set_async_scheduling(oracle8, True)
    _driver_stop_at(oracle8, "r1", n_answer=2, eos_step=1)
    res8 = gm.check(gp8, oracle8)
    assert res8["verdict"] == "FAIL" and "G6" in res8["failed_checks"], res8["reasons"]
    pr8 = _check(res8, "G6")["problems"]
    assert any("r1: stopped on the declared EOS id 11 at engine step 1 with 1 more row(s) executed after it, but the Program declares no LAG "
               "(Program declares no lag (pre-R17-1 Build); record attests async_scheduling=True (RunHeader.profile.engine_kwargs) (NOT used as the rule))"
               in p and "nothing is inferred from the trace or read from the header as the rule" in p for p in pr8), pr8
    assert _check(res8, "G6")["lag"]["declared"] is None and _check(res8, "G6")["lag"]["exercised"] is True and "pre-R17-1" in _check(res8, "G6")["lag"]["note"]
    gp8b, oracle8b = assemble([R0, R1], tmp_path / "undeclared_caps", lag=None)      # every request to its cap
    res8b = gm.check(gp8b, oracle8b)
    assert res8b["verdict"] == "PASS", res8b["reasons"]
    assert res8b["lag"]["declared"] is None and res8b["lag"]["exercised"] is False and res8b["lag"]["record_attests"] == 0
    assert "not exercised" in res8b["lag"]["note"] and "labelled, not accepted as a lag-N record" in res8b["lag"]["note"]

    # (9) an EOS sampled AT the cap is a length stop (vLLM checks max_tokens first; no placeholder row): the lag rule is not exercised, the
    # declaration still has to be attested (the toy attests False = LAG 0, the Program declares 0)
    gp9, oracle9 = assemble([R0, R1], tmp_path / "eos_at_cap", lag=0)   # R1 = [8, 11] to its cap 2, 11 the declared EOS
    res9 = gm.check(gp9, oracle9)
    assert res9["verdict"] == "PASS", res9["reasons"]
    assert _r(res9, "r1")["finish"] == "length" and _r(res9, "r1")["cap_exhausted"] is True and res9["lag"]["exercised"] is False

    # (10) VARIABLE LAG: two EOS-stopped requests of ONE record, one followed by the lag row and one not, under the LAG-1 Program with True
    # attested -> the deleted-step failure on the second AND the record-level variable-lag finding, surfaced by name in res.alerts (the lane
    # posts ALERT + DECISIONS-PENDING; never folded into a constant-lag model)
    R2_EOS_NOLAG = {"id": "r2", "prompt": [3, 3, 3], "sampled": [6, EOS], "cap": 5}    # EOS at event 1, NO row after it
    gpv, oraclev = assemble([R0, R1_LAG, R2_EOS_NOLAG], tmp_path / "variable", lag=1)
    _set_async_scheduling(oraclev, True)
    _driver_stop_at(oraclev, "r1", n_answer=2, eos_step=1)
    resv = gm.check(gpv, oraclev)
    assert resv["verdict"] == "FAIL" and "G6" in resv["failed_checks"], resv["reasons"]
    prv = _check(resv, "G6")["problems"]
    assert any(p.startswith("VARIABLE LAG: the EOS-stopped requests of this record show different post-EOS row counts {'r1': 1, 'r2': 0} under one "
                            "declared scheduler (Program LAG 1") and "never fold into a constant-lag model (R17-1)" in p for p in prv), prv
    assert any("r2: the Program's declared scheduler" in p and "a deleted step" in p for p in prv), prv
    assert resv["lag"]["variable_lag"] is True and resv["lag"]["lag_rows_observed"] == {"r1": 1, "r2": 0} and resv["lag"]["eos_stop_requests"] == ["r1", "r2"]
    assert resv["alerts"] == [{"kind": "variable_lag", "check": "G6", "lag_rows_observed": {"r1": 1, "r2": 0}, "declared": 1, "record_attests": 1,
                               "action": "ALERT + DECISIONS-PENDING entry (R17-1)"}]
    assert _r(resv, "r1")["executed_prefix_len"] == 3 and _r(resv, "r2")["executed_prefix_len"] == 2 and _r(resv, "r2")["served_output_len"] == 2

    # (12) rev M-0027 (4): a LENGTH stop runs no post-cap row under any LAG -- REQUIRED from the trace: r1 = [8, 5, 77] executed against cap 2
    # (no EOS among them) under the LAG-1 Program is 'execution past cap', named with the two lengths (served 2 = cap, executed 3)
    R1_PASTCAP = {"id": "r1", "prompt": [4, 6], "sampled": [8, 5, 77], "cap": 2}
    gpp, oraclep = assemble([R0, R1_PASTCAP], tmp_path / "pastcap", lag=1)
    _set_async_scheduling(oraclep, True)
    _driver_stop_at(oraclep, "r1", n_answer=2, eos_step=1)
    _driver_claims(oraclep, "r1", finish_reason="length")
    resp = gm.check(gpp, oraclep)
    assert resp["verdict"] == "FAIL" and "G6" in resp["failed_checks"], resp["reasons"]
    prp = _check(resp, "G6")["problems"]
    assert any("r1: execution past cap -- 3 output events executed but the declared max_tokens is 2 (executed_prefix_len 3 > served_output_len 2 = cap; "
               "a length stop computes no post-cap row under any LAG, the Program's LAG 1 applies to EOS stops only)" in p for p in prp), prp
    assert _r(resp, "r1")["served_output_len"] == 2 and _r(resp, "r1")["executed_prefix_len"] == 3 and _r(resp, "r1")["cap_exhausted"] is True
    assert _r(resp, "r1")["finish"] == "length" and resp["lag"]["exercised"] is False      # not an EOS stop: the lag rule is not what failed

    # (11) the record schema (m1_capture's tokens.json): doc.lag.value + per-request served_output_len / executed_prefix_len are the driver's
    # CLAIMS from the same recorded fact; claims that agree pass, a tokens.json lag disagreeing with the header is "not one attested execution",
    # a length claim disagreeing with the rows is named with both lengths
    gpc, oraclec = assemble([R0, R1_LAG], tmp_path / "claims", lag=1)
    _set_async_scheduling(oraclec, True)
    _driver_stop_at(oraclec, "r1", n_answer=2, eos_step=1)
    _driver_claims(oraclec, "r1", lag=1, served_output_len=2, executed_prefix_len=3, cap_exhausted=False)
    _driver_claims(oraclec, "r0", served_output_len=3, executed_prefix_len=3, cap_exhausted=True)
    resc = gm.check(gpc, oraclec)
    assert resc["verdict"] == "PASS", resc["reasons"]
    assert _check(resc, "G6")["lag"]["tokens_json_lag"] == 1
    _driver_claims(oraclec, lag=0)
    resc = gm.check(gpc, oraclec)
    assert resc["verdict"] == "FAIL" and any("the record's own sources disagree on the scheduler lag: tokens.json lag.value 0 vs async_scheduling=True" in p
                                             for p in _check(resc, "G6")["problems"]), _check(resc, "G6")["problems"]
    _driver_claims(oraclec, "r1", lag=1, executed_prefix_len=2)
    resc = gm.check(gpc, oraclec)
    assert resc["verdict"] == "FAIL" and any("r1: the driver's tokens.json claims executed_prefix_len 2 but the record's rows show 3 (served_output_len 2, "
                                             "executed_prefix_len 3, cap 4)" in p for p in _check(resc, "G6")["problems"]), _check(resc, "G6")["problems"]


def test_gp01_carries_manbind_weights_of_record_per_component_and_refuses_stale_incomplete_or_two_checkpoints(tmp_path):
    """COORD M-0614 (1) / tp M-0632 / manbind MAN-09 (M-0638): the weights of record is ONE function (manbind's weights_of_record.py) of the
    pinned shards and the (rank) Program's weights struct; GP-01 carries each component's record into the workload Program -- per rank for
    TP -- and refuses a record that is not the component's (stale digest), incomplete, or a second checkpoint for the same rank."""
    from verity_vllm.pipeline import global_program as G
    def comp(name, rec, digest="d" * 64):
        d = tmp_path / name
        d.mkdir()
        if rec is not None:
            json.dump(rec, open(d / "weights_of_record.json", "w"))
        return {"dir": str(d), "recorded_digest": digest}
    def rec(root, rank=0, world=1, digest="d" * 64, complete=True, unresolved=(), pinned=True, source=None):
        r = {"schema": "verity/weights_of_record/v1", "program_digest": digest, "tp": {"world": world, "rank": rank}, "root_of_record": root,
             "complete": complete, "unresolved": list(unresolved), "n_pinned": 98, "n_fields": 163, "checkpoint": {"dir": "/ck", "all_shards_pinned": pinned}}
        if source is not None:
            r["source"] = source
        return r
    # (+) two shapes, one checkpoint -> carried, per_rank {0: root}
    ok = G.weights_of_record_binding({"8,3": comp("a", rec("A" * 64)), "5,1": comp("b", rec("A" * 64))})
    assert ok["refusal"] is None and ok["complete"] and ok["per_rank"] == {"0": "A" * 64}
    assert ok["components"]["8,3"]["root_of_record"] == "A" * 64 and ok["components"]["5,1"]["tp"] == {"world": 1, "rank": 0}
    # (+) TP2: rank 0 and rank 1 components carry DIFFERENT roots (different shards) -- one per rank, no refusal
    tp = G.weights_of_record_binding({"8,3": comp("r0", rec("A" * 64, rank=0, world=2)), "8,3@1": comp("r1", rec("B" * 64, rank=1, world=2))})
    assert tp["refusal"] is None and tp["per_rank"] == {"0": "A" * 64, "1": "B" * 64}
    # (+) a component without a record is carried as null, named in the note (Commit then fails 'weights unpinned'), never invented
    part = G.weights_of_record_binding({"8,3": comp("c", rec("A" * 64)), "5,1": comp("d", None)})
    assert part["refusal"] is None and not part["complete"] and part["components"]["5,1"] is None and "1 of 2" in part["note"]
    # (-) same rank, two roots -> refused (one workload Program cannot span two checkpoints)
    two = G.weights_of_record_binding({"8,3": comp("e", rec("A" * 64)), "5,1": comp("f", rec("C" * 64))})
    assert "cannot span two checkpoints" in two["refusal"]
    # (-) stale record: program_digest is not the component's recorded digest
    stale = G.weights_of_record_binding({"8,3": comp("g", rec("A" * 64, digest="e" * 64), digest="d" * 64)})
    assert "stale record" in stale["refusal"]
    # (-) incomplete record (a field with no composition rule) -> refused by name, fail closed
    inc = G.weights_of_record_binding({"8,3": comp("h", rec("A" * 64, complete=False, unresolved=["model.layers.3.mlp.down_proj.weight"]))})
    assert "incomplete" in inc["refusal"] and "down_proj" in inc["refusal"]
    # (-) [rev-pm M-0717 (ii)] no program_digest -> refused (the stale check could never fire); complete:true + unresolved -> refused
    nod = G.weights_of_record_binding({"8,3": comp("i", rec("A" * 64, digest=None))})
    assert "carries no program_digest" in nod["refusal"]
    contra = G.weights_of_record_binding({"8,3": comp("j", rec("A" * 64, complete=True, unresolved=["k_scale"]))})
    assert "self-contradicting" in contra["refusal"] and "k_scale" in contra["refusal"]
    # (-) [rev F-r16-06, M-0056] a root that was not composed from PINNED shards is not carried: all_shards_pinned False (a shard differs from
    # its pin) or None (no shard pins recorded) -> refused by name; a record describing LIVE parameters (the Commit side's re-registration
    # digest) is never the weights of record, whatever its root says
    unp = G.weights_of_record_binding({"8,3": comp("k", rec("A" * 64, pinned=False))})
    assert "not composed from pinned shards" in unp["refusal"] and "all_shards_pinned=False" in unp["refusal"]
    nop = G.weights_of_record_binding({"8,3": comp("l", rec("A" * 64, pinned=None))})
    assert "not composed from pinned shards" in nop["refusal"] and "all_shards_pinned=None" in nop["refusal"]
    liv = G.weights_of_record_binding({"8,3": comp("m", rec("A" * 64, source="live_param_digests"))})
    assert "describes live parameters" in liv["refusal"]
    assert ok["components"]["8,3"]["all_shards_pinned"] is True   # the carried entry records the guard's inputs


# ------------------------------------------------------------------------------- [COORD M-0169 / M-0571 (3)] derived S: sampler geometry


class _StubFold:
    def __init__(self, specs):
        self.inst = [{"spec": s} for s in specs]


def _topp(V, S):
    return f"GumbelTopPTokenSelect_v1{{V={V},S={S}}}"


def test_sampler_geometry_check_evaluates_s_per_engine_step_against_the_realized_live_count():
    """M-0169: vLLM picks the top-p split count S = splits_for(sampler batch, num_SMs) on the host per launch; the fold carries it on every
    GumbelTopPTokenSelect of the step. GM-01 evaluates S_t = SplitsFor_v1(|live(t)|, num_SMs) from the RECORD's live count (the requests with
    a logits row at t) and the Program's declared num_SMs -- L40S 142 SMs: 8 live -> 16, 4 live -> 32 (registry/topp_split)."""
    from verity_vllm.program.registry.topp_geometry import splits_for
    assert (splits_for(8, 142), splits_for(4, 142), splits_for(1, 142)) == (16, 32, 32)
    gp = {"applicability": {"components": {"num_sms": 142}}}
    # engine step 0: 8 live requests (S must be 16); step 1: 4 live (S must be 32) -- a verified EOS halved the batch
    steps = {0: {"logits_rows": [(f"r{i}", 9) for i in range(8)]}, 1: {"logits_rows": [(f"r{i}", 10) for i in range(4)]}}
    specs = [_topp(64, 16)] * 8 + ["TokenSelect_v1{V=64}"] + [_topp(64, 32)] * 4
    es = {i: 0 for i in range(8)} | {8: 1} | {i: 1 for i in range(9, 13)}
    ok = SG.sampler_geometry_check(gp, _StubFold(specs), es, steps, {0: 8, 1: 4})
    assert ok["problems"] == [] and ok["info"]["checked_steps"] == 2 and ok["info"]["top_p_token_selects"] == 12
    assert ok["info"]["live_to_S"] == {"live=4:S=32": 1, "live=8:S=16": 1} and ok["info"]["num_sms"] == 142
    assert ok["info"]["declared_by"] == "applicability.components.num_sms" and ok["info"]["sampler_batch_recorded_steps"] == 2
    # (-) the launch at step 1 kept S = 16 for 4 live rows (the geometry of the declared chronology, not of the realized batch): FAIL by name
    bad = SG.sampler_geometry_check(gp, _StubFold([_topp(64, 16)] * 8 + [_topp(64, 16)] * 4), {i: (0 if i < 8 else 1) for i in range(12)}, steps)
    assert len(bad["problems"]) == 1 and "step" in bad["problems"][0] and "'expected_S': 32" in bad["problems"][0] and "'fold_S': [16]" in bad["problems"][0]
    assert bad["info"]["steps_with_mismatch"] == 1 and "M-0571 (3)" in bad["problems"][0]
    # (-) the Program declares no num_SMs at all: the relation cannot be evaluated -> fail closed, named
    nod = SG.sampler_geometry_check({"applicability": {"components": {}}}, _StubFold([_topp(64, 16)]), {0: 0}, steps)
    assert len(nod["problems"]) == 1 and "declares no num_SMs" in nod["problems"][0] and "fail closed" in nod["problems"][0]
    # (+) the Program's own sampler_geometry declaration (GP-01) is the source of record when present
    sg = SG.sampler_geometry_check({"sampler_geometry": {"num_sms": 132}, "applicability": {"components": {"num_sms": 142}}},
                                   _StubFold([_topp(64, 16)] * 8), {i: 0 for i in range(8)}, steps)
    assert sg["problems"] == [] and sg["info"]["declared_by"] == "sampler_geometry" and sg["info"]["num_sms"] == 132
    # (-) the SampleEnd's own logits_view batch disagrees with the chronology's logits rows: named, never averaged away
    sb = SG.sampler_geometry_check(gp, _StubFold([_topp(64, 16)] * 8), {i: 0 for i in range(8)}, steps, {0: 7})
    assert len(sb["problems"]) == 1 and "logits_view batch != the record's logits rows" in sb["problems"][0]
    # (-) a top-p TokenSelect without a readable S / on an engine step the record does not list: named
    nos = SG.sampler_geometry_check(gp, _StubFold(["GumbelTopPTokenSelect_v1{V=64}"]), {0: 0}, steps)
    assert any("no readable split count" in p for p in nos["problems"])
    unk = SG.sampler_geometry_check(gp, _StubFold([_topp(64, 16)]), {}, steps)
    assert any("the record does not list" in p for p in unk["problems"])
    # (0) a greedy fold has nothing to check -- no problem, checked 0, no num_SMs needed
    g = SG.sampler_geometry_check({}, _StubFold(["TokenSelect_v1{V=64}"] * 3), {0: 0, 1: 0, 2: 1}, steps)
    assert g["problems"] == [] and g["info"]["checked_steps"] == 0 and g["info"]["top_p_token_selects"] == 0
    # the fold's spec strings may carry a prefix (batch:) -- S is still read
    assert SG.top_p_split_count("batch:" + _topp(64, 8)) == 8 and SG.top_p_split_count("TokenSelect_v1{V=64}") is None


class _OperandFold:
    """stoch M-0254 / M-0268 (B): every select is `GumbelTopPTokenSelect_v1{V}` (no S static) and the fold carries ONE `Const32[S]` instance
    right before it, read as the 6th operand by exactly one run ("n", kc, 0, 1) -- the shape compare.Prog._bind_splits binds; here UNBOUND
    (no splits_values), the way a Prog-shaped reader without compare's binding sees it."""

    def __init__(self, selects):   # selects: list of (S value | None for a non-Const operand, 6th-operand override or None)
        self.inst, self.k2i, self.splits_values = [], {}, {}
        for S, override in selects:
            kc, ks = len(self.inst) * 2 + 1, len(self.inst) * 2 + 2
            if S is not None:
                self.k2i[kc] = len(self.inst)
                self.inst.append({"k": kc, "spec": f"Const32[0x{S:08x}]_v1", "args": []})
            sixth = override if override is not None else [("n", kc, 0, 1)]
            self.k2i[ks] = len(self.inst)
            self.inst.append({"k": ks, "spec": "GumbelTopPTokenSelect_v1{V=64}", "args": [[("n", 900, 0, 64)], [("n", 901, 0, 1)], [("n", 902, 0, 1)],
                                                                                         [("n", 903, 0, 1)], [("n", 904, 0, 1)], sixth]})


def test_sampler_geometry_check_reads_s_from_the_const32_splits_operand():
    """stoch M-0268 (A): on the operand tree the S static is gone; G6 reads S from the Const32 splits operand -- through compare's binding
    (splits_values, the of-record path) or, unbound, straight from the 6th operand's single run onto a Const32 instance -- and FAILS by name
    when the operand is anything else."""
    gp = {"applicability": {"components": {"num_sms": 142}}}
    steps = {0: {"logits_rows": [(f"r{i}", 9) for i in range(8)]}, 1: {"logits_rows": [(f"r{i}", 10) for i in range(4)]}}
    fold = _OperandFold([(16, None)] * 8 + [(32, None)] * 4)
    sel = [i for i, x in enumerate(fold.inst) if x["spec"].startswith("GumbelTopP")]
    assert len(sel) == 12 and all(SG.top_p_split_count(fold.inst[i]["spec"], fold, fold.inst[i]) == (16 if n < 8 else 32) for n, i in enumerate(sel))
    es = {sel[n]: (0 if n < 8 else 1) for n in range(12)}
    ok = SG.sampler_geometry_check(gp, fold, es, steps, {0: 8, 1: 4})
    assert ok["problems"] == [] and ok["info"]["live_to_S"] == {"live=4:S=32": 1, "live=8:S=16": 1} and ok["info"]["top_p_token_selects"] == 12
    # the bound path wins when compare has bound the operand (splits_values keyed by the select's k), whatever the operand says
    fold.splits_values = {fold.inst[sel[8]]["k"]: 16}
    kept = SG.sampler_geometry_check(gp, fold, es, steps)
    assert len(kept["problems"]) == 1 and "'expected_S': 32" in kept["problems"][0] and "'fold_S': [16]" in kept["problems"][0]
    # (-) S kept at 16 for the 4 live rows of step 1: FAIL by name, exactly as with the static form
    bad = _OperandFold([(16, None)] * 12)
    b = SG.sampler_geometry_check(gp, bad, {sel[n]: (0 if n < 8 else 1) for n in range(12)}, steps)
    assert len(b["problems"]) == 1 and "'expected_S': 32" in b["problems"][0] and "'fold_S': [16]" in b["problems"][0]
    # (-) a 6th operand that is not ONE run onto a Const32 -- a 2-wide run, a non-Const instance, an unknown k -- carries no readable S
    for override in ([("n", 1, 0, 2)], [("n", 900, 0, 1)], [("n", 555, 0, 1)], [("n", 1, 0, 1), ("n", 1, 0, 1)], []):
        f = _OperandFold([(16, override)])
        i = [i for i, x in enumerate(f.inst) if x["spec"].startswith("GumbelTopP")][0]
        r = SG.sampler_geometry_check(gp, f, {i: 0}, {0: {"logits_rows": [("r0", 9)]}})
        assert len(r["problems"]) == 1 and "no readable split count" in r["problems"][0] and "Const32[S] splits operand" in r["problems"][0], override
    # a Const of another width (Const64) in the operand slot is not a split count either
    f = _OperandFold([(16, None)])
    f.inst[0]["spec"] = "Const64[0x0000000000000010]_v1"
    assert SG.top_p_split_count(f.inst[1]["spec"], f, f.inst[1]) is None


def test_read_record_carries_the_sampler_batch_of_every_sample_end(tmp_path):
    """The SampleEnd's logits_view shape[0] is the sampler launch's own batch -- read beside the chronology so the geometry check can
    cross-check it against the step's logits rows."""
    p = tmp_path / "log.jsonl"
    hdr = {"$e": "RunHeader", "profile": {}, "workload": {}}
    ss = {"$e": "StepStart", "step": 0, "rows": [["a", 0], ["b", 0]], "logits_rows": [["a", 0], ["b", 0]], "input_token_ids": [1, 2]}
    se = {"$e": "SampleEnd", "step": 0, "logits_view": {"$v": [619, 0, [2, 49152], [49152, 1], "bfloat16", "cuda:0"]}, "sampled_token_ids": [5, 6]}
    se_old = {"$e": "SampleEnd", "step": 1, "logits_view": None, "sampled_token_ids": [7]}
    ss1 = dict(ss, step=1, rows=[["a", 1]], logits_rows=[["a", 1]])
    p.write_text("\n".join(json.dumps(x) for x in (hdr, ss, se, ss1, se_old)) + "\n")
    rec = gm.read_record(str(p))
    assert rec["sampler_batch"] == {0: 2} and rec["samples"] == {0: [5, 6], 1: [7]}


def test_greedy_rows_report_no_sampler_geometry_to_check(memory, tmp_path):
    """The check is additive: the greedy fixtures pass unchanged and record `sampler_geometry.checked_steps == 0`."""
    gp, oracle = assemble([R0, R1], tmp_path)
    res = gm.check(gp, oracle)
    assert res["verdict"] == "PASS", res["reasons"]
    assert res["sampler_geometry"]["checked_steps"] == 0 and res["sampler_geometry"]["top_p_token_selects"] == 0
    assert _check(res, "G6")["sampler_geometry"]["note"].startswith("no top-p TokenSelect")


def test_gp01_declares_the_sampler_geometry_for_a_stochastic_row_only():
    """GP-01 (M-0169 (2b)): a stochastic row's workload Program declares S_t = SplitsFor_v1(|live(t)|, num_SMs) with the components' agreed
    num_SMs and the declared schedule (every request to its cap); greedy rows declare nothing; a stochastic sampler without num_SMs is a
    named refusal (the relation cannot be evaluated)."""
    from verity_vllm.pipeline import global_program as G
    reqs = [{"request_id": "r0", "prompt_len": 8, "cap": 4, "arrive_step": 0}, {"request_id": "r1", "prompt_len": 5, "cap": 2, "arrive_step": 1}]
    assert G.sampler_geometry({"components": {"sampler": {"kind": "greedy"}, "num_sms": 142}}, reqs) is None
    st = {"kind": "stochastic", "temperature": 0.8, "top_p": 0.95, "top_k": 0, "seeded": True, "splits": "derived"}
    d = G.sampler_geometry({"components": {"sampler": st, "num_sms": 142}}, reqs)
    assert d["relation"] == "SplitsFor_v1" and d["num_sms"] == 142 and "refusal" not in d and d["component_splits"] == "derived"
    # live counts: step 0 -> r0 only (1), steps 1..2 -> r0 + r1 (2), step 3 -> r0 (1); 142 SMs -> S = 32 at every one of them
    assert d["declared_live_counts"] == [1, 2, 2, 1] and d["declared_schedule"] == [32, 32, 32, 32]
    # 8 requests arriving at 0 with cap 2 then 4 of them continuing: 8 live -> 16, 4 live -> 32 (the S flip the B8 rows show)
    reqs8 = [{"request_id": f"r{i}", "prompt_len": 8, "cap": 2 if i < 4 else 4, "arrive_step": 0} for i in range(8)]
    d8 = G.sampler_geometry({"components": {"sampler": st, "num_sms": 142}}, reqs8)
    assert d8["declared_live_counts"] == [8, 8, 4, 4] and d8["declared_schedule"] == [16, 16, 32, 32]
    nod = G.sampler_geometry({"components": {"sampler": st}}, reqs)
    assert "declare no num_SMs" in nod["refusal"] and nod["num_sms"] is None


# ------------------------------------------------------------- R17 stoch: |live(t)| from the SCHEDULED workload state (BRIEF §6 / R17-1)


def _req(rid, first, a_r, cap=None, chunks=1, lag=0, finish="length", eos_verified=None, g6=0):
    e = {"request_id": rid, "first_step": first, "prefill_chunks": chunks, "answer_events": a_r, "lag_rows": lag, "finish": finish,
         "max_tokens": cap if cap is not None else a_r, "g6_problems": g6}
    if eos_verified is not None:
        e["eos_verified"] = eos_verified
    return e


def _present(reqs, extra=None):
    """the record's rows per request: the contiguous steps the scheduled state predicts (or `extra` overrides)"""
    p = {}
    for e in reqs:
        n = (e["prefill_chunks"] - 1) + e["answer_events"] + e["lag_rows"]
        p[e["request_id"]] = list(range(e["first_step"], e["first_step"] + n))
    p.update(extra or {})
    return p


def test_scheduled_live_counts_builds_the_workload_state_from_g6_facts_and_names_every_transition():
    """R17 stoch / BRIEF §6: |live(t)| is derived from the facts G6 established per request -- the declared arrival, the served length (cap or
    verified EOS), the DECLARED lag rows after an EOS stop (R17-1: the post-EOS step is active computation, its row IS in the sampler batch)
    and the prefill chunks -- never from the record's per-step row list. Transitions carry their causes."""
    # r0 arrives 0, cap 6 (length); r1 arrives 1, cap 6, EOS after 2 answers + 1 declared lag row -> sampler rows at steps 1, 2, 3;
    # r2 arrives 2 with a 2-chunk prefill (one discarded chunk sample) and 2 answers -> steps 2, 3, 4
    reqs = [_req("r0", 0, 6), _req("r1", 1, 2, cap=6, lag=1, finish="eos", eos_verified=True), _req("r2", 2, 2, chunks=2)]
    sc = SG.scheduled_live_counts(reqs, _present(reqs))
    assert sc["live"] == {0: 1, 1: 2, 2: 3, 3: 3, 4: 2, 5: 1} and sc["max_live"] == 3 and sc["unestablished"] == {}
    assert sc["requests"]["r1"] == {"established": True, "first_step": 1, "last_step": 3, "prefill_chunks": 1, "answer_events": 2, "lag_rows": 1, "finish": "eos"}
    t = {x["step"]: x for x in sc["transitions"]}
    assert t[0]["arrivals"] == ["r0"] and (t[0]["live_from"], t[0]["live_to"]) == (0, 1)
    assert t[1]["arrivals"] == ["r1"] and t[2]["arrivals"] == ["r2"] and (t[2]["live_from"], t[2]["live_to"]) == (2, 3)
    # step 4: r1 departs AFTER its lag row (EOS at step 2, lag row at 3) -- the departure names the stop kind and the lag rows that ended
    assert t[4]["departures"] == [{"request": "r1", "finish": "eos", "lag_rows": 1, "answer_events": 2, "prefill_chunks": 1}] and (t[4]["live_from"], t[4]["live_to"]) == (3, 2)
    assert t[5]["departures"][0]["request"] == "r2" and t[6]["departures"][0]["request"] == "r0" and t[6]["live_to"] == 0
    # (-) an EOS stop that is NOT verified leaves the state unestablished at every step the request touches (never a fallback to the rows)
    bad = [_req("r0", 0, 6), _req("r1", 1, 2, cap=6, lag=1, finish="eos", eos_verified=False)]
    sb = SG.scheduled_live_counts(bad, _present(bad))
    assert sb["unestablished"] == {1: ["r1"], 2: ["r1"], 3: ["r1"]} and sb["live"] == {s: 1 for s in range(6)} and sb["requests"]["r1"]["established"] is False
    # (-) a request with a G6 problem is unestablished too; (-) rows that do not equal the scheduled steps (a row past the stop) are named
    g6 = SG.scheduled_live_counts([_req("r0", 0, 6, g6=1)], _present([_req("r0", 0, 6)]))
    assert g6["unestablished"] == {s: ["r0"] for s in range(6)}
    past = [_req("r0", 0, 3)]
    sp = SG.scheduled_live_counts(past, {"r0": [0, 1, 2, 3]})
    assert sp["requests"]["r0"]["note"] == "scheduled sampler steps != observed rows" and sp["unestablished"] == {s: ["r0"] for s in range(4)}


def test_sampler_geometry_check_reads_live_from_the_scheduled_state_and_refuses_an_s_that_follows_an_unscheduled_row():
    """R17 stoch: with the scheduled state, the record's logits rows and the SampleEnd batch must EQUAL the scheduled live count and S is
    evaluated at that count. A sampler row the workload does not schedule (an undeclared request, a row past a stop) inflates the observed
    batch and vLLM's S follows it -- the R16 form (observed rows) accepted that; the scheduled form names both disagreements."""
    from verity_vllm.program.registry.topp_geometry import splits_for
    gp = {"applicability": {"components": {"num_sms": 142}}}
    reqs = [_req(f"r{i}", 0, 2) for i in range(4)] + [_req("r4", 0, 1)]      # 5 live at step 0 (S 16), 4 at step 1 (S 32)
    sc = SG.scheduled_live_counts(reqs, _present(reqs))
    assert sc["live"] == {0: 5, 1: 4}
    steps = {0: {"logits_rows": [(f"r{i}", 9) for i in range(5)]}, 1: {"logits_rows": [(f"r{i}", 10) for i in range(4)]}}
    specs = [_topp(64, 16)] * 5 + [_topp(64, 32)] * 4
    es = {i: 0 for i in range(5)} | {i: 1 for i in range(5, 9)}
    ok = SG.sampler_geometry_check(gp, _StubFold(specs), es, steps, {0: 5, 1: 4}, sc)
    assert ok["problems"] == [], ok["problems"]
    assert ok["info"]["live_source"].startswith("scheduled workload state") and ok["info"]["checked_steps"] == 2
    assert ok["info"]["S_transitions"] == [{"step": 0, "live": 5, "S": 16}, {"step": 1, "live": 4, "S": 32}]
    assert ok["info"]["transitions"][1] == {"step": 1, "live_from": 5, "live_to": 4, "arrivals": [], "S_from": 16, "S_to": 32,
                                            "departures": [{"request": "r4", "finish": "length", "lag_rows": 0, "answer_events": 1, "prefill_chunks": 1}]}
    # (-) step 1 carries a FIFTH logits row (a request the workload does not schedule there) and the launch's S = 16 follows that batch:
    # the R16 form (no scheduled state) reads live = 5 and PASSES; the scheduled form names the row AND the S
    steps_x = {0: steps[0], 1: {"logits_rows": [(f"r{i}", 10) for i in range(4)] + [("rX", 3)]}}
    specs_x = [_topp(64, 16)] * 5 + [_topp(64, 16)] * 5
    es_x = {i: 0 for i in range(5)} | {i: 1 for i in range(5, 10)}
    assert splits_for(5, 142) == 16
    r16 = SG.sampler_geometry_check(gp, _StubFold(specs_x), es_x, steps_x, {0: 5, 1: 5})
    assert r16["problems"] == [] and r16["info"]["live_source"].startswith("observed logits rows")
    r17 = SG.sampler_geometry_check(gp, _StubFold(specs_x), es_x, steps_x, {0: 5, 1: 5}, sc)
    assert len(r17["problems"]) == 3, r17["problems"]
    assert any("logits rows != the scheduled live count [{'step': 1, 'scheduled_live': 4, 'logits_rows': 5}]" in p and "never the observed batch" in p for p in r17["problems"])
    assert any("SampleEnd logits_view batch != the scheduled live count" in p for p in r17["problems"])
    assert any("'expected_S': 32" in p and "'fold_S': [16]" in p and "'live': 4" in p for p in r17["problems"])
    # (-) a top-p select on a step the state schedules nobody at; (-) a scheduled sampling step without a select; (-) an unestablished step
    steps_2 = dict(steps) | {2: {"logits_rows": [("rX", 11)]}}
    extra = SG.sampler_geometry_check(gp, _StubFold(specs + [_topp(64, 32)]), es | {9: 2}, steps_2, None, sc)
    assert len(extra["problems"]) == 1 and "scheduled workload state has NO live request" in extra["problems"][0] and "'step': 2" in extra["problems"][0]
    miss = SG.sampler_geometry_check(gp, _StubFold(specs[:5]), {i: 0 for i in range(5)}, steps, None, sc)
    assert len(miss["problems"]) == 1 and "carry no top-p TokenSelect in the fold: [1]" in miss["problems"][0]
    un = dict(sc, unestablished={1: ["r2"]})
    ue = SG.sampler_geometry_check(gp, _StubFold(specs), es, steps, None, un)
    assert len(ue["problems"]) == 1 and "NOT established" in ue["problems"][0] and "fail closed" in ue["problems"][0] and ue["info"]["checked_steps"] == 1
    # (0) a greedy fold still reports the transitions of the scheduled state (the RESULT's achieved-concurrency evidence)
    g = SG.sampler_geometry_check(gp, _StubFold(["TokenSelect_v1{V=64}"] * 9), es, steps, None, sc)
    assert g["problems"] == [] and g["info"]["checked_steps"] == 0 and [t["step"] for t in g["info"]["transitions"]] == [0, 1, 2]


def test_scheduled_state_on_the_r15_b8_record_reproduces_the_kernel_s_flips():
    """The R15 B8 staggered stoch record (157 engine steps, 8 requests arriving 0..44 with caps 2..128, every request served to its cap):
    the scheduled state from (arrival, cap, length stop) alone equals the recorded live count at every step and its S transitions are the
    kernel's flips (0,1,32) -> (32,5,16) -> (101,4,32)."""
    from verity_vllm.program.registry.topp_geometry import splits_for
    p = os.path.join(os.path.dirname(gm.__file__), "..", "..", "..", "tests", "program", "data", "topp_split_fixture", "b8_s_schedule_r15_attempt2.json")
    d = json.load(open(p))
    reqs = [_req(rid, v["arrive"], v["served"], cap=v["cap"]) for rid, v in d["served_vs_cap"].items()]
    present: dict = {}
    for st in d["steps"]:
        for rid in st["requests"]:
            present.setdefault(rid, []).append(st["step"])
    sc = SG.scheduled_live_counts(reqs, present)
    assert sc["unestablished"] == {} and sc["max_live"] == 7 and len(sc["live"]) == d["engine_steps"] == 157
    assert all(sc["live"][st["step"]] == st["live_realized"] == st["live_declared"] for st in d["steps"])
    assert all(splits_for(sc["live"][st["step"]], d["num_sms"]) == st["S_kernel"] for st in d["steps"])
    flips, prev = [], None
    for st in d["steps"]:
        s = splits_for(sc["live"][st["step"]], d["num_sms"])
        if s != prev:
            flips.append([st["step"], sc["live"][st["step"]], s]); prev = s
    assert flips == d["S_flips_(step,live,S)"] == [[0, 1, 32], [32, 5, 16], [101, 4, 32]]
    assert sum(len(t["arrivals"]) for t in sc["transitions"]) == 8 and sum(len(t["departures"]) for t in sc["transitions"]) == 8


def test_g6_puts_the_scheduled_state_and_the_lag_row_on_the_record(memory, tmp_path):
    """e2e on the toy record: the lag-1 request occupies its lag step in the scheduled state (R17-1: active internal computation); the
    transitions name the EOS departure with lag_rows 1; a greedy fold has no S to check but the state is on the record."""
    gp, oracle = assemble([R0, R1_LAG], tmp_path / "sched", lag=1)   # [R17-1] the Program declares LAG 1
    _set_async_scheduling(oracle, True)
    _driver_stop_at(oracle, "r1", n_answer=2, eos_step=1)
    res = gm.check(gp, oracle)
    assert res["verdict"] == "PASS", res["reasons"]
    sl = res["scheduled_live"]
    assert sl["unestablished_steps"] == 0 and sl["max_live"] == 2
    assert sl["requests"]["r1"] == {"established": True, "first_step": 0, "last_step": 2, "prefill_chunks": 1, "answer_events": 2, "lag_rows": 1, "finish": "eos"}
    dep = [d for t in sl["transitions"] for d in t["departures"] if d["request"] == "r1"]
    assert dep == [{"request": "r1", "finish": "eos", "lag_rows": 1, "answer_events": 2, "prefill_chunks": 1}]
    assert [t["step"] for t in res["sampler_geometry"]["transitions"]] == [t["step"] for t in sl["transitions"]]
    # the R16 lag negatives still fail in G6 and the state marks the request unestablished (never the observed rows as a substitute)
    _set_async_scheduling(oracle, False)
    res = gm.check(gp, oracle)
    assert res["verdict"] == "FAIL" and res["scheduled_live"]["requests"]["r1"]["established"] is False and res["scheduled_live"]["unestablished_steps"] == 3


def test_a_record_without_tokens_json_fails_g6_by_name_unless_tokens_optional_is_asked_for_and_labelled(memory, tmp_path):
    """[F-r17-01, rev M-0035] capture/tokens.json is part of the record: absent -> G6 FAILS by name (the output-association legs cannot run);
    tokens_optional=True (CLI --tokens-optional) accepts it LABELLED checks[G6].tokens_json=false -- mechanism/fixture runs only."""
    gp, oracle = assemble([R0, R1], tmp_path, lag=0)
    _set_async_scheduling(oracle, False)
    ok = gm.check(gp, oracle)
    assert ok["verdict"] == "PASS" and _check(ok, "G6")["tokens_json"] is True, ok["reasons"]
    os.remove(os.path.join(oracle, "capture", "tokens.json"))
    res = gm.check(gp, oracle)
    assert res["verdict"] == "FAIL" and res["failed_checks"] == ["G6"], res["reasons"]
    pr = _check(res, "G6")["problems"]
    assert any("the record has no capture/tokens.json: the output-association legs" in p and "--tokens-optional" in p and "(F-r17-01)" in p for p in pr), pr
    assert _check(res, "G6")["tokens_json"] is False
    # the same record accepted when asked for, and the label stays on the check
    acc = gm.check(gp, oracle, tokens_optional=True)
    assert acc["verdict"] == "PASS" and _check(acc, "G6")["tokens_json"] is False, acc["reasons"]
    # the CLI flag reaches check(): the default main() refuses, --tokens-optional accepts
    gpp = tmp_path / "gp.json"; gpp.write_text(json.dumps(gp))
    out1 = tmp_path / "gm1.json"; out2 = tmp_path / "gm2.json"
    cli.main(["global-match", "--oracle", oracle, "--global-program", str(gpp), "--no-x09", "--no-alternate", "--out", str(out1)])
    cli.main(["global-match", "--oracle", oracle, "--global-program", str(gpp), "--no-x09", "--no-alternate", "--tokens-optional", "--out", str(out2)])
    assert json.load(open(out1))["verdict"] == "FAIL" and json.load(open(out2))["verdict"] == "PASS"


def test_fast_path_dag_hash_ids_are_per_process_and_a_foreign_dict_is_refused(memory, tmp_path, monkeypatch):
    """[rev F-r17-07] The compact Merkle payload names producers by small per-process ids; a dag_hashes dict computed in one process
    (a forked worker) must never be matched against one computed in another.  The memo refuses a seeded foreign dict by name."""
    import os
    from verity_vllm.check.match import program_compare as C
    from verity_vllm.check.match import global_match_fast as GMF
    gp, oracle = assemble([R0, R1], tmp_path)
    B = C.load_oracle(oracle, "oracle")
    dag_hashes = GMF.ops().dag_hashes
    try:
        h = dag_hashes(B)
        assert isinstance(h, GMF._Hashes) and h.pid == os.getpid()
        assert dag_hashes(B) is h                         # memo hit, own process: fine
        foreign = GMF._Hashes(h)
        foreign.pid = os.getpid() + 100_003                # a dict from some other process, seeded into this memo
        GMF._dag_memo[(id(B), None)] = (B, None, foreign)
        with pytest.raises(RuntimeError, match="per process"):
            dag_hashes(B)
        GMF._dag_memo[(id(B), None)] = (B, None, dict(h))   # [F-r17-08] a pid-less plain dict is refused too
        with pytest.raises(RuntimeError, match="without an origin"):
            dag_hashes(B)
    finally:
        GMF.reset()


def test_fast_runs_equals_refs_runs_on_every_descriptor_shape():
    """[unit 5 (a)] global_match_fast._runs_fast == verity.ml.circuits.refs.runs on Affine / Concat-of-Affine (incl. zero-count parts and
    fusing neighbours) and falls back to the original on every other shape (Strided, Explicit, nested Concat)."""
    import random
    from verity.ml.circuits.refs import Affine, Concat, Explicit, Strided, runs
    from verity_vllm.check.match import global_match_fast as GMF
    rng = random.Random(7)
    def affine():
        return Affine(rng.choice("np"), rng.randrange(6), rng.randrange(8) * 4, rng.choice([0, 1, 4, 4, 8]))
    cases = [affine() for _ in range(50)]
    cases += [Concat([affine() for _ in range(rng.randrange(1, 12))]) for _ in range(300)]
    # fusing neighbours: same (space, idx), hi == next lo
    cases += [Concat([Affine("n", 3, 0, 4), Affine("n", 3, 4, 4), Affine("n", 3, 8, 0), Affine("n", 4, 12, 4), Affine("n", 4, 16, 4)])]
    cases += [Concat([Affine("n", 3, 0, 4), Strided("n", 5, 0, 2, 4, 1, 8)]), Concat([Concat([Affine("n", 1, 0, 2)]), Affine("n", 1, 2, 2)]),
              Strided("n", 5, 0, 2, 4, 1, 8), Explicit((("n", 1, 0), ("n", 1, 1), ("n", 2, 0))), Concat([Explicit((("p", 0, 3),))])]
    for r in cases:
        assert GMF._runs_fast(r) == runs(r), r


def test_fast_producer_chronology_and_dep_order_scan_equal_the_records_loops(memory, tmp_path):
    """[unit 5 (b)(c)] On a real-shaped fixture (staggered arrivals, a co-scheduled prefill, T-grouped emission = a cross-step
    producer, and the consumer-before-producer negative) the helpers return exactly what the record's loops compute."""
    import collections
    from verity_vllm.check.match import batch_decomp as BD
    from verity_vllm.check.match import program_compare as C
    from verity_vllm.check.match import global_match_fast as GMF
    seen_viol = 0
    for n, (reqs, kw) in enumerate((([R0, R1_LATE, R2_LATE], {}), ([R0, R1_LATE, R2_LATE], {"tgrouped": True}), ([R0, R1], {"leak": True}),
                                    ([R0, R1], {"tgrouped": True, "_swap": True}))):
        swap = kw.pop("_swap", False)
        gp, oracle = assemble(reqs, tmp_path / f"c{n}", **kw)
        if swap:
            move_consumer_before_producer(oracle, "r1")
        B = C.load_oracle(oracle, "oracle")
        meta_of = {o["node"]: o for o in B.oracle_rows}
        engine_step_of = {i: int(meta_of[r["k"]]["step"]) for i, r in enumerate(B.inst) if meta_of.get(r["k"], {}).get("step") is not None}
        weights_nodes = frozenset(BD.weights_input_nodes(B, len(reqs), 0))
        per_req = collections.defaultdict(list)
        for i, r in enumerate(B.inst):
            req = meta_of.get(r["k"], {}).get("request")
            if req is not None:
                per_req[req].append(i)
        assert per_req
        for rid, idxs in per_req.items():
            P = BD.Projection(B, f"oracle[{rid}]", idxs, tok_node=None, weights_node=weights_nodes, seed_node=None)
            # the record's loops
            late = 0
            xstep = collections.Counter()
            wr = False
            for i_local, r in enumerate(P.inst):
                cs = engine_step_of.get(idxs[i_local])
                for a in r["args"]:
                    for space, idx, lo, hi in a:
                        if space != "n":
                            continue
                        if idx in B.inputs:
                            if idx in weights_nodes:
                                wr = True
                            continue
                        j = B.k2i.get(idx)
                        ps = engine_step_of.get(j) if j is not None else None
                        if ps is not None and cs is not None and ps > cs:
                            late += 1
                        if ps is not None and cs is not None and ps < cs:
                            xstep[r["spec"].split("{")[0]] += 1
            dep = []
            for s_, loc in enumerate(P.steps):
                for pos, i in enumerate(loc):
                    for a in P.canon(i, False):
                        for t in a:
                            if t[0] == "N" and P.steps[t[1]][t[2]] >= i:
                                dep.append({"request_id": rid, "local_step": s_, "emission": pos, "consumer": i, "producer": P.steps[t[1]][t[2]]})
            assert GMF.producer_chronology(P, idxs, B, engine_step_of, weights_nodes) == (late, xstep, wr)
            assert GMF.dep_order_violations(P, rid) == dep
            seen_viol += len(dep)
        GMF._chron_memo.clear()
    assert seen_viol >= 1     # the swapped record exercised the violation branch


@pytest.mark.parametrize("compact", ["1", "0"])
def test_fast_dag_hash_interning_induces_exactly_the_records_merkle_partition(memory, tmp_path, monkeypatch, compact):
    """[unit 6] MATCH_IMPL=fast names an instance's Merkle class by a structural id (prefix trie over the operand items, producers by
    their own id) instead of sha256(json).  Spy every dag_hashes call check() makes -- components, projections, live cones, the
    permutation fixture's per-step matching, the consumer-before-producer forward refs -- and require ONE bijection between the
    record's digests and the ids across all of them (so A-vs-P comparisons decide exactly as the record's do), under both operand forms."""
    from verity_vllm.check.match import global_match_fast as GMF
    from verity_vllm.check.match import batch_decomp as BD
    monkeypatch.setenv(GMF.FLAG, "fast")
    monkeypatch.setenv(GMF.COMPACT_ENV, compact)
    monkeypatch.setenv("MATCH_WORKERS", "1")
    rec_of_id: dict[int, str] = {}
    id_of_rec: dict[str, int] = {}
    seen = {"calls": 0, "instances": 0, "forward": 0}
    intern = GMF._dag_hashes_intern

    def spy(P, only=None):
        mine = intern(P, only)
        rec = BD.dag_hashes(P, only)                # the checker of record's function on the same Prog / cone
        assert set(mine) == set(rec)
        for i, hid in mine.items():
            assert rec_of_id.setdefault(hid, rec[i]) == rec[i], ("two digests share an id", i, P.label)
            assert id_of_rec.setdefault(rec[i], hid) == hid, ("one digest got two ids", i, P.label)
        seen["calls"] += 1
        seen["instances"] += len(mine)
        return mine
    monkeypatch.setattr(GMF, "_dag_hashes_intern", spy)
    cases = [([R0, R1], {"tgrouped": True}, None), ([R0, R1_LATE, R2_LATE], {}, None), ([R0, R1], {"leak": True}, None),
             ([R0, R1], {"tgrouped": True}, "swap")]
    for n, (reqs, kw, mut) in enumerate(cases):
        gp, oracle = assemble(reqs, tmp_path / f"c{n}", **kw)
        if mut == "swap":
            move_consumer_before_producer(oracle, "r1")
        for crit in ("dag", "sequence"):
            rec_of_id.clear()   # the id space is per outermost check() (released on leave, unit 4): one bijection per check
            id_of_rec.clear()
            res = gm.check(gp, oracle, within_step=crit)
            if mut == "swap":
                assert res["verdict"] == "FAIL" and "G8" in res["failed_checks"]
            assert len(rec_of_id) == len(id_of_rec) >= 20
    assert seen["calls"] >= 8 and seen["instances"] > 500


@pytest.mark.parametrize("count", [1, 2])
def test_fast_dag_hash_progression_memo_does_not_carry_a_forward_ref_marker_to_a_resolved_consumer(count):
    """[unit 6, F-r17-10 (rev M-0338 / M-0383)] the compact-form memo of the node-space progression walk is keyed on (prefix node,
    base, delta, count) and used to store items computed with the hashes AT WALK TIME.  A consumer emitted BEFORE its producers (the
    G8 attack: a forward reference, named by the record's marker) and a consumer emitted AFTER them, reading the same progression,
    get different digests from the record (marker vs the producers' hashes); the memo handed the second the first's marker node --
    fast A == B where record A != B, so the fast document put the resolved readers in the marker class.  Both shapes rev posted:
    count 2 (M-0338: two producers) and the ordinary count-1 progression (M-0383: one producer, its other readers after it)."""
    import inspect
    from verity_vllm.check.match import batch_decomp as BD
    from verity_vllm.check.match import global_match_fast as GMF
    if "compact" not in inspect.getsource(BD.dag_hashes):
        pytest.skip("the compact (progression-form) operand is mfast's tree; here no Prog is compact and the memo branch is not reached")

    class FakeProg:
        compact = True
        label = "F-r17-10"

        def __init__(self):
            self.steps = [[0, 1, 2, 3, 4]]           # node order: [A consumer (forward refs), P1, P2, B consumer, C consumer]
            leaf = {"spec": "leaf", "form": "f", "out": "T", "args": []}
            cons = {"spec": "gather", "form": "f", "out": "T", "args": [None]}
            self.inst = [dict(cons), dict(leaf, spec="leaf1"), dict(leaf, spec="leaf2"), dict(cons), dict(cons)]

        def canon(self, i, unfold=False):
            if i in (1, 2):
                return []
            return [[("N", (0, 1, 0, 4), (0, 1, 0, 0), count)]]   # producers steps[0][1] (, steps[0][2]); the same progression for A, B, C

    P = FakeProg()
    GMF.enter()
    try:
        fast = GMF._dag_hashes_intern(P)
        rec = BD.dag_hashes(P)
    finally:
        GMF.leave()
    assert set(fast) == set(rec) == {0, 1, 2, 3, 4}
    assert rec[0] != rec[3] and rec[3] == rec[4]                       # the record: marker consumer apart, the two resolved ones together
    assert fast[0] != fast[3] and fast[3] == fast[4]                   # the fast ids induce the same partition
    assert len({rec[i] for i in rec}) == len({fast[i] for i in fast})


# ------------------------------------------------------------------------------------------- COORD R17-28 P1: one X-09 per row


@pytest.mark.parametrize("impl,workers", [("record", "1"), ("fast", "1"), ("fast", "3")])
@pytest.mark.slow
def test_cli_decomp_out_is_the_standalone_batch_decomp_document(memory, tmp_path, monkeypatch, impl, workers):
    """[COORD R17-28, mprof P1] row_pod runs cb_a.batch_decomp in its own process (match.compare) and GM-01 runs the same decompose()
    again inside check() (the x09 block) -- 365 s twice at B8 1024/128.  `--decomp-out` writes the document GM-01's own X-09 run
    produced, in batch_decomp's CLI schema: key for key what a standalone decompose() over the same fold and components returns
    (utc / seconds aside), under both implementations; exit 4 when X-09 names missing derived shapes (row_pod's derive-and-rerun)."""
    from verity_vllm.check.match import batch_decomp as BD
    from verity_vllm.check.match import global_match_fast as GMF
    monkeypatch.setenv(GMF.FLAG, impl)
    monkeypatch.setenv("MATCH_WORKERS", workers)   # mfast's forked X-09 legs return the per-request entries: the document must not depend on W
    gp, oracle = assemble([R0, R1_LATE, R2_LATE], tmp_path)
    gpp = tmp_path / "gp.json"
    gpp.write_text(json.dumps(gp))
    out, dout = tmp_path / "gm.json", tmp_path / "decomp.json"
    rc = cli.main(["global-match", "--global-program", str(gpp), "--oracle", str(oracle), "--out", str(out), "--decomp-out", str(dout), "--within-step", "dag"])
    assert rc == 0
    doc = json.loads(dout.read_text())
    comp_dirs = [g["component"]["dir"] for g in gp["requests"] if (g.get("component") or {}).get("dir")]
    # the standalone stage: its own fold + component decode, the record's operations (the fixtures' components are fold records, as
    # check() reads them)
    ref = BD.decompose(str(oracle), comp_dirs, None, within_step="dag", ops=gm.match_ops(fast=False))
    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if k not in ("utc", "seconds")}
        return [strip(v) for v in o] if isinstance(o, list) else o
    assert sorted(doc) == sorted(ref) and "schema" in doc and "requests" in doc and "attribution" in doc
    assert json.dumps(strip(doc), sort_keys=True, default=str) == json.dumps(strip(ref), sort_keys=True, default=str)
    assert doc["verdict"] == "PASS" and len(doc["requests"]) == 3
    gmdoc = json.loads(out.read_text())
    assert gmdoc["x09"]["verdict"] == doc["verdict"] and gmdoc["x09"]["xreq_total"] == doc["xreq_total"]
    # [rev S-r17-27 n1] the GM-01 document states what its X-09 ran over and under which pipeline (distinguishable by content)
    assert gmdoc["x09"]["derived_available"] == doc["derived_available"] and gmdoc["x09"]["missing_derived"] == doc["missing_derived"] == []
    assert gmdoc["x09"]["pipeline"] == {"shared": True, "decomp_out": str(dout), "derived_args": []} and gmdoc["x09"]["extra_derived"] == []
    # ... and ONLY under the shared pipeline: the record flow's CLI document carries none of these keys (the checker of record's document,
    # byte for byte -- match_diff's drift rule on check() and the CLI alike)
    out_rec = tmp_path / "gm_recflow.json"
    monkeypatch.setenv(GMF.FLAG, impl)
    assert cli.main(["global-match", "--global-program", str(gpp), "--oracle", str(oracle), "--out", str(out_rec), "--within-step", "dag"]) == 0
    recflow = json.loads(out_rec.read_text())
    assert not ({"derived_available", "missing_derived", "extra_derived", "pipeline"} & set(recflow["x09"]))
    assert {k: v for k, v in recflow["x09"].items() if k != "seconds"} == {k: v for k, v in gmdoc["x09"].items()
                                                                          if k not in ("seconds", "derived_available", "missing_derived", "extra_derived", "pipeline")}
    if workers == "1":   # as row_pod runs it: `python -m ...` (the record path runs in __main__, the fast path imports the module again)
        import subprocess, sys as _sys
        dout2 = tmp_path / "decomp_m.json"
        env = dict(os.environ, MATCH_IMPL=impl, MATCH_WORKERS=workers, PYTHONPATH=os.pathsep.join(p for p in _sys.path if p))
        p = subprocess.run([_sys.executable, "-m", "verity_vllm.pipeline.cli", "global-match", "--global-program", str(gpp), "--oracle", str(oracle),
                            "--out", str(tmp_path / "gm_m.json"), "--decomp-out", str(dout2), "--within-step", "dag"], env=env, capture_output=True, text=True)
        assert p.returncode == 0, p.stderr[-800:]
        assert json.dumps(strip(json.loads(dout2.read_text())), sort_keys=True, default=str) == json.dumps(strip(ref), sort_keys=True, default=str)
    # a shape whose component is not derived yet: the document names it and the exit code is batch_decomp's 4
    gp2 = json.loads(json.dumps(gp))
    dropped = gp2["requests"][2]["component"]["dir"]
    for r in gp2["requests"]:
        if (r.get("component") or {}).get("dir") == dropped:
            r["component"]["dir"] = None
    gpp.write_text(json.dumps(gp2))
    monkeypatch.setenv(GMF.FLAG, impl)
    rc = cli.main(["global-match", "--global-program", str(gpp), "--oracle", str(oracle), "--out", str(out), "--decomp-out", str(dout), "--within-step", "dag"])
    doc = json.loads(dout.read_text())
    assert rc == 4 and doc["verdict"] == "MISSING_DERIVED" and doc["missing_derived"]
    # row_pod derives the named shape and reruns with --derived: the same global Program (G1-G8 unchanged), X-09 complete again
    monkeypatch.setenv(GMF.FLAG, impl)
    rc = cli.main(["global-match", "--global-program", str(gpp), "--oracle", str(oracle), "--out", str(out), "--decomp-out", str(dout), "--within-step", "dag",
                  "--derived", dropped])
    doc = json.loads(dout.read_text())
    assert doc["verdict"] == "PASS" and not doc["missing_derived"]
    assert json.dumps(strip(doc), sort_keys=True, default=str) == json.dumps(strip(ref), sort_keys=True, default=str)
    # ... while GM-01's own verdict on a global Program that names no component for r2 is FAIL by name (G3), and the exit code is its 3
    gmdoc = json.loads(out.read_text())
    assert rc == 3 and gmdoc["verdict"] == "FAIL" and "G3" in gmdoc["failed_checks"] and gmdoc["x09"]["verdict"] == "PASS"
    assert gmdoc["x09"]["extra_derived"] == [dropped] and gmdoc["x09"]["pipeline"]["derived_args"] == [dropped]   # n1: the rerun's list
    assert dropped in gmdoc["x09"]["derived_available"].values()


def test_fold_encodes_its_program_once_and_program_json_carries_that_descriptor(memory, tmp_path):
    """[COORD R17-28 F3] `_finish` encodes the Program for the digest; `persist_program` used to encode and digest it AGAIN (36 s +
    a second digest at B8 1024/128).  Now the FoldResult carries the descriptor and program.json's `program` IS it: its digest (the
    codec's canonical bytes) equals the fold's, and a result without a descriptor (older producer) still encodes and cross-checks."""
    from verity.ml.circuits import codec
    from verity_vllm.pipeline.match import persist_program as _persist
    evs = batched_log([R0, R1])
    r = fold(evs, profile(), strict=True)
    assert r.descriptor is not None and r.digest == codec.program_digest(r.descriptor)
    d = tmp_path / "rec"
    d.mkdir()
    calls = {"encode": 0}
    orig = codec.encode_program

    def counting(program, *a, **k):
        calls["encode"] += 1
        return orig(program, *a, **k)
    codec.encode_program = counting
    try:
        _persist(r, d, profile_name="toy", log_path="toy")
        assert calls["encode"] == 0                                   # the fold's own encoding is written, not a second one
        doc = json.loads((d / "program.json").read_text())
        assert doc["digest"] == r.digest == codec.program_digest(doc["program"])
        assert doc["program"] == r.descriptor
        # an older producer's result (no descriptor): encoded here, digest cross-checked as before
        r.descriptor = None
        _persist(r, tmp_path / "rec2", profile_name="toy", log_path="toy") if (tmp_path / "rec2").mkdir() is None else None
        assert calls["encode"] == 1
        assert json.loads((tmp_path / "rec2" / "program.json").read_text())["digest"] == r.digest
    finally:
        codec.encode_program = orig
