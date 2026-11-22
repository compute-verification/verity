"""The two-rank fixture of the collective-received root class.

THE EXACT UNSUPPORTED CASE (reproduced from the TP2 B1 record's `match/per_rank_fold/`):
rank 0, engine step 0, event 88 = `vllm.all_reduce.default` issued by `model.embed_tokens` -- the FIRST collective of the step (k = 0), the
embedding partial [256, 2048] bf16 reduced into a fresh storage (alloc a236).  The rank profile had no pattern for it -> Unsupported;
event 92 (`_rms_norm_kernel`, `model.layers.0.input_layernorm`) then read a236 = "unresolved read of View(a236+0 bfloat16[2048]):
unwritten" -> FoldError cascade.  1088 Unsupported per rank (1056 all_reduce + 32 all_gather), nowhere else; the same on rank 1.

This fixture is that case at toy size: two ranks, one request (LP=2, one decode step), H=16, vocab 16 sharded 8/8, one row-parallel Gemm.
Each rank's log has three collectives per step in issue order -- k=0 all_reduce on `model.embed_tokens` (the EmbeddingShard partial), k=1
all_reduce on `model.layers.0.self_attn.o_proj` (the Gemm partial), k=2 all_gather on `logits_processor` (the lm_head shard) -- and the
kernels that read the RECEIVED storages (RMSNorm x2, argmax).  `test_exact_unsupported_case_without_the_fold_rule` reproduces the failure
(the profile without `TpCollective`); the rest is the D90 fold rule: the received value is the OUTPUT of an `AllReduce2_v1{N}` /
`AllGather2_v1{N}` instance over (own partial, peer partial) in rank order, the peer partial the fold's `peers` ROOT PARAMETER
(collective_received), never a free prescribed input; the rank Program (Build side, built here with the same Definitions the derive
binds) compares instance by instance through `cb_a.compare` with the PEER canonical source.

NEGATIVES (must FAIL BY NAME): wrong peer, swapped rank, off-by-one step, wrong segmentation/count, reordered collectives,
equal-result-but-wrong-producer -- as mutations of the RECORD (the rank's log or the Program's link table), never of the checker.
"""

from __future__ import annotations

import gzip
import itertools
import json
from dataclasses import dataclass

import pytest
from verity.circuits import codec
from verity.circuits.defs import CompositeDefinition, SpecializedDefinition, bind
from verity.circuits.program import Program
from verity.circuits.refs import Coll, array_of, concat, runs
from verity.circuits.types import Array, RecordType, Value

from verity_vllm.definitions.registry import b1, b1_tp2
from verity_vllm.observe.events import (
    DispatchEnter,
    DispatchExit,
    Root,
    RunHeader,
    SampleEnd,
    StepEnd,
    StepLink,
    StepStart,
    StorageAlloc,
    TritonLaunch,
    ViewDesc,
)
from verity_vllm.observe.fold.fold import fold
from verity_vllm.observe.fold.patterns.collectives import TpCollective
from verity_vllm.observe.fold.resolver import (
    Compute,
    Gather,
    PerRow,
    Profile,
    Shared,
    Transparent,
)
from verity_vllm.observe.log import dumps as dump_events

BF16, I32 = Value(16), Value(32)
BF, I32D, I64D = "bfloat16", "int32", "int64"

# ---------------------------------------------------------------------------------------------------------- toy configuration
H, VS, V_FULL, WORLD = 16, 8, 16, 2
C = {"H": H, "NL": 1, "NH": 1, "KVH": 1, "D": 16, "I": 8, "V": VS, "EPS": 1e-5, "MAX_POS": 8, "BIAS": False}
REQ = "r0"
PROMPT = [5, 7]
SAMPLED = [9, 3]
BS = 4
A_EMBED, A_LN1, A_O_W, A_NORM, A_IDS, A_POS = 1, 2, 3, 4, 10, 11


def V(a, shape, strides=None, off=0, dtype=BF):
    shape = tuple(shape)
    if strides is None:
        s, strides = 1, []
        for n in reversed(shape):
            strides.append(s)
            s *= n
        strides = tuple(reversed(strides))
    return ViewDesc(a, off, shape, tuple(strides), dtype, "cuda:0")


W_EMBED, W_LN1, W_O, W_NORM = V(A_EMBED, (VS, H)), V(A_LN1, (H,)), V(A_O_W, (H, H)), V(A_NORM, (H,))


def weight_field(name: str):
    return {"model.embed_tokens.weight": ("embed",), "lm_head.weight": ("embed",), "model.norm.weight": ("norm",),
            "model.layers.0.input_layernorm.weight": ("layers", 0, "ln1"), "model.layers.0.self_attn.o_proj.weight": ("layers", 0, "o_w")}.get(name)


def preamble(rank: int, *, tp_note: bool = True):
    notes = {"tp2": {"rank": rank, "world_size": WORLD}} if tp_note else {}
    hdr = RunHeader("verity-capture/log/v1", "2026-09-20T00:00:00Z", "toy-tp2-rank", {"vllm": "toy"}, {"repo": "toy", "config": C},
                    {"requests": [{"request_id": REQ, "prompt_token_ids": PROMPT, "max_tokens": 2}], "sampling": {"temperature": 0.0}}, notes=notes)
    evs = [hdr]
    for a, v, name in ((A_EMBED, W_EMBED, "model.embed_tokens.weight"), (A_LN1, W_LN1, "model.layers.0.input_layernorm.weight"),
                       (A_O_W, W_O, "model.layers.0.self_attn.o_proj.weight"), (A_NORM, W_NORM, "model.norm.weight")):
        evs.append(StorageAlloc(a, 0x1000 * a, v.numel * 2, "cuda:0", 0, "param"))
        evs.append(Root(v, "weight", name))
    evs.append(Root(W_EMBED, "weight", "lm_head.weight"))   # tied lm_head: the same shard
    for a, nbytes in ((A_IDS, 4 * 4), (A_POS, 8 * 8)):
        evs.append(StorageAlloc(a, 0x1000 * a, nbytes, "cuda:0", 0, "op"))
    return evs


@dataclass
class Ids:
    n: int = 100

    def next(self) -> int:
        self.n += 1
        return self.n


PROMPTS = {"r0": PROMPT, "r1": [6, 2]}          # the B2 form: two requests of the same shape (LP 2, T 1), co-scheduled
SAMPLEDS = {"r0": SAMPLED, "r1": [4, 8]}


def step_events(rank: int, step: int, positions: list[int], ids: Ids, base_alloc: int, sampled: int, *, mut: dict | None = None):
    """One engine step of rank `rank` over `positions` of the single request REQ (the B1 form; see `_step_events` for the plan form)."""
    return _step_events(rank, step, [(REQ, positions)], ids, base_alloc, {REQ: sampled}, mut=mut)


def _step_events(rank: int, step: int, plan: list[tuple[str, list[int]]], ids: Ids, base_alloc: int, sampled_of: dict[str, int], *,
                 mut: dict | None = None, rows_override: list[tuple[str, int]] | None = None):
    """One engine step of rank `rank` over the row plan [(request, positions)] (rows in plan order): EmbeddingShard -> all_reduce(k=0) ->
    RMSNorm -> Gemm(o_proj) -> all_reduce(k=1) -> RMSNorm(norm) -> logits rows -> Gemm(lm_head shard) -> all_gather(k=2) -> argmax.
    `mut` applies one named record mutation; `rows_override` replaces ONLY the StepStart row attribution (the tensors keep their rows):
    the shifted-row-boundary record."""
    mut = mut or {}
    rows_attr = [(q, p) for q, ps in plan for p in ps]
    M, R = len(rows_attr), len(plan)
    p0 = plan[0][1][0]
    prefill = p0 == 0
    a = itertools.count(base_alloc)
    emb, h, normed, g, h2, f, gathered, lg, logits, sampled_v = (V(next(a), s, dtype=d) for s, d in (
        ((M, H), BF), ((M, H), BF), ((M, H), BF), ((M, H), BF), ((M, H), BF), ((M, H), BF), ((R, H), BF), ((R, VS), BF), ((R, V_FULL), BF), ((R,), I64D)))
    pos_view = V(A_POS, (8,), dtype=I64D)
    ids_rows = ViewDesc(A_IDS, 0, (M,), (1,), I32D, "cuda:0")

    def tok(q, p):
        return PROMPTS[q][p] if p < len(PROMPTS[q]) else SAMPLEDS[q][p - len(PROMPTS[q])]

    logits_idx, lo = [], 0
    for q, ps in plan:
        logits_idx.append(lo + len(ps) - 1)
        lo += len(ps)
    evs = [StepStart(step, list(rows_override or rows_attr), [tok(q, p) for q, p in rows_attr], [p for _, p in rows_attr],
                     {q: ps[0] + len(ps) for q, ps in plan}, {q: ps[0] for q, ps in plan}, {q: [i] for i, (q, _) in enumerate(plan)}, BS,
                     [p for _, p in rows_attr], [(q, ps[-1]) for q, ps in plan], logits_idx, R, M, M, ids_rows, pos_view,
                     "prefill" if prefill else "decode")]
    lo = 0
    for q, ps in plan:
        if prefill:
            evs.append(Root(ViewDesc(A_IDS, lo, (len(ps),), (1,), I32D, "cuda:0"), "prompt_token", f"prompt[{q}]", [tok(q, p) for p in ps]))
        else:
            for i, p in enumerate(ps):
                evs.append(StepLink(step, lo + i, q, step - 1, "eq", SAMPLEDS[q][p - len(PROMPTS[q])]))
        lo += len(ps)
    sampled = [sampled_of[q] for q, _ in plan]
    for v in (emb, h, normed, g, h2, f, gathered, lg, logits, sampled_v):
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

    def rms(x, w, out, module):
        triton("vllm.model_executor.layers.batch_invariant._rms_norm_kernel",
               {"input_ptr": x, "weight_ptr": w, "output_ptr": out, "input_row_stride": H, "n_cols": H, "eps": 1e-5, "BLOCK_SIZE": 1024}, None, module)

    def all_reduce(x, out, module):
        dispatch("vllm.all_reduce.default", [x, "tp:0"], {}, out, module)

    START = rank * VS
    dispatch("_C.vocab_parallel_embedding.default", [emb, ids_rows, W_EMBED, START, START + VS, 0, V_FULL, V_FULL], {}, None, "model.embed_tokens")
    if mut.get("drop_collective") == 0 and step == mut.get("at_step", 0):
        h = emb                                                        # wrong count: the embedding partial is never reduced
    elif mut.get("site_of_k0"):
        all_reduce(emb, h, mut["site_of_k0"])                          # wrong site for k=0
    else:
        all_reduce(emb, h, "model.embed_tokens")
    rms(h, W_LN1, normed, "model.layers.0.input_layernorm")
    mm(normed, A_O_W, H, H, g, "model.layers.0.self_attn.o_proj")
    if mut.get("wrong_producer") and step == mut.get("at_step", 0):
        all_reduce(emb, h2, "model.layers.0.self_attn.o_proj")         # equal-shape but WRONG producer: reduces the embedding partial again
    elif mut.get("reorder") and step == mut.get("at_step", 0):
        # reordered collectives: the o_proj reduction is issued AFTER the logits all-gather (k=1 <-> k=2 swapped on this rank)
        pass
    else:
        all_reduce(g, h2, "model.layers.0.self_attn.o_proj")
    if mut.get("reorder") and step == mut.get("at_step", 0):
        rms(g, W_NORM, f, "model.norm")
    else:
        rms(h2, W_NORM, f, "model.norm")
    dispatch("aten.index.default", [f, [ViewDesc(A_POS, 0, (R,), (1,), I64D, "cuda:0")]], {}, gathered, "model")
    mm(gathered, A_EMBED, VS, H, lg, "lm_head")
    dispatch("vllm.all_gather.default", [lg, -1, WORLD, "tp:0"], {}, logits, "logits_processor")
    if mut.get("reorder") and step == mut.get("at_step", 0):
        all_reduce(g, h2, "model.layers.0.self_attn.o_proj")
    dispatch("aten.argmax.default", [logits], {"dim": -1}, sampled_v, None, phase="sample")
    evs.append(SampleEnd(step, logits, list(sampled), sampled_v))
    evs.append(StepEnd(step))
    return evs


def rank_log(rank: int, *, tp_note: bool = True, mut: dict | None = None):
    ids = Ids()
    evs = preamble(rank, tp_note=tp_note)
    evs += step_events(rank, 0, [0, 1], ids, 20, SAMPLED[0], mut=mut)
    evs += step_events(rank, 1, [2], ids, 40, SAMPLED[1], mut=mut)
    return evs


def rank_log_b2(rank: int, *, shift: bool = False):
    """The B2 form: r0 and r1 (LP 2 each) prefilled together in step 0 (rows r0:[0,1] r1:[2,3]), decoded together in step 1.  `shift`
    = the shifted row boundary: step 0's StepStart attributes the rows r0: 3 / r1: 1 instead of 2 / 2 while every tensor keeps its 4 rows."""
    ids = Ids()
    hdr = preamble(rank)
    hdr[0] = RunHeader(hdr[0].schema, hdr[0].created_utc, hdr[0].profile_id, hdr[0].profile, hdr[0].model,
                       {"requests": [{"request_id": q, "prompt_token_ids": PROMPTS[q], "max_tokens": 2} for q in ("r0", "r1")], "sampling": {"temperature": 0.0}},
                       notes=hdr[0].notes)
    evs = hdr
    plan0 = [("r0", [0, 1]), ("r1", [0, 1])]
    evs += _step_events(rank, 0, plan0, ids, 20, {"r0": SAMPLED[0], "r1": SAMPLEDS["r1"][0]},
                        rows_override=([("r0", 0), ("r0", 1), ("r0", 2), ("r1", 0)] if shift else None))
    evs += _step_events(rank, 1, [("r0", [2]), ("r1", [2])], ids, 40, {"r0": SAMPLED[1], "r1": SAMPLEDS["r1"][1]})
    return evs


# ---------------------------------------------------------------------------------------------------------- toy rank patterns


class Pat:
    def __init__(self, name, fn):
        self.name, self.fn = name, fn

    def match(self, node, ctx):
        return self.fn(node, ctx)


def _hint(node, root_name):
    return f"{node.enter.module}/{root_name}"


def p_embedding_shard(node, ctx):
    if node.op != "_C.vocab_parallel_embedding.default":
        return None
    out, idx, w, start, end = node.args[:5]
    roots = ctx.roots_of(w)
    return Compute(bind(b1_tp2.EmbeddingShard, VS=w.shape[0], H=w.shape[1], START=start), idx.shape[0], (PerRow(idx), Shared(w)), (PerRow(out),),
                   (node.id,), _hint(node, roots[0].name))


def p_rms_triton(node, ctx):
    if not node.is_triton or not node.op.endswith("_rms_norm_kernel"):
        return None
    a = node.enter.args
    roots = ctx.roots_of(a["weight_ptr"])
    return Compute(bind(b1.RMSNormTriton, N=H, EPS=C["EPS"]), a["input_ptr"].shape[0], (PerRow(a["input_ptr"]), Shared(a["weight_ptr"])),
                   (PerRow(a["output_ptr"]),), (node.id,), _hint(node, roots[0].name))


def p_mm(node, ctx):
    if node.op != "aten.mm.default" or len(node.children) != 1:
        return None
    a, bT = node.args
    K, N = bT.shape
    w = ViewDesc(bT.alloc, bT.offset, (N, K), (K, 1), bT.dtype, bT.device)
    roots = ctx.roots_of(w)
    return Compute(bind(b1.Gemm, K=K, N=N), a.shape[0], (PerRow(a), Shared(w)), (PerRow(node.outs),), (node.id, node.children[0].id),
                   _hint(node, roots[0].name))


def p_index(node, ctx):
    if node.op != "aten.index.default":
        return None
    idx = tuple(ctx.start.logits_indices)
    return Transparent("move", (node.id,), Gather(node.args[0], idx), PerRow(node.outs), len(idx), "logits_indices")


def p_argmax(node, ctx):
    if node.op != "aten.argmax.default" or node.phase != "sample":
        return None
    logits = node.args[0]
    return Compute(bind(b1.TokenSelect, V=V_FULL), logits.shape[0], (PerRow(logits),), (PerRow(node.outs),), (node.id,), "sampler")


BASE_PATTERNS = [Pat(n, f) for n, f in (("embedding_shard", p_embedding_shard), ("rms_triton", p_rms_triton), ("mm", p_mm),
                                         ("index", p_index), ("argmax", p_argmax))]


def rank_profile(*, with_collectives: bool = True) -> Profile:
    pats = ([TpCollective(world=WORLD)] if with_collectives else []) + BASE_PATTERNS
    return Profile("toy-tp2-rank", pats, dict(C), weight_field, False)


def fold_rank(rank: int, *, with_collectives: bool = True, tp_note: bool = True, mut: dict | None = None, strict: bool = False):
    return fold(rank_log(rank, tp_note=tp_note, mut=mut), rank_profile(with_collectives=with_collectives), strict=strict, with_program=True)


# ---------------------------------------------------------------------------------------------------------- the Build side (rank Program)

PEER_SHAPES = [(2, H), (2, H), (1, VS), (1, H), (1, H), (1, VS)]     # k = 0..5: (rows, N) of the other rank's partial, issue order over steps
SITES = ["model.embed_tokens", "model.layers.0.self_attn.o_proj", "logits_processor"]        # per k within a step (named_modules paths)
SITE_CLASSES = ["VocabParallelEmbedding", "RowParallelLinear", "LogitsProcessor"]             # what the derive's link table names
KINDS = ["all_reduce", "all_reduce", "all_gather"]


def rank_program(rank: int, *, swap_parts: bool = False, peer_of: dict[int, int] | None = None, wrong_op_k: int | None = None,
                 wrong_producer: bool = False) -> SpecializedDefinition:
    """The rank's REQUEST Program as the derive builds it (one Input per user input / weight; per-row AllReduce2 / AllGather2 over
    `array_of([part_rank0, part_rank1])`).  The keyword mutations build a WRONG Program for the named negatives."""
    peer_of = peer_of or {}
    params = [("prompt", Array(2, I32))] + [(f"peers_{k}", Array(r, Array(n, BF16))) for k, (r, n) in enumerate(PEER_SHAPES)]
    params += [("embed", Array(VS, Array(H, BF16))), ("ln1", Array(H, BF16)), ("o_w", Array(H, Array(H, BF16))), ("norm", Array(H, BF16))]
    ret = Array(2, I32)

    def parts(own, peer):
        a, b = (own, peer) if rank == 0 else (peer, own)
        if swap_parts:
            a, b = b, a
        return array_of([a, b])

    def body(B, S, prompt, *rest):
        peers, (embed, ln1, o_w, norm) = list(rest[:len(PEER_SHAPES)]), rest[len(PEER_SHAPES):]
        toks = [prompt[0], prompt[1]]
        outs, k = [], 0

        def peer(kk, i):
            return peers[peer_of.get(kk, kk)][i]

        for s, rows in ((0, [0, 1]), (1, [2])):
            emb = [B.call(bind(b1_tp2.EmbeddingShard, VS=VS, H=H, START=rank * VS), toks[i], embed) for i in rows]
            h = [B.call(bind(b1_tp2.AllReduce2 if wrong_op_k != k else b1_tp2.AllGather2, N=H), parts(e, peer(k, i))) for i, e in enumerate(emb)]
            if wrong_op_k == k:
                h = [Coll(Array(H, BF16), x.refs.slice(0, H)) for x in h]
            k += 1
            n = [B.call(bind(b1.RMSNormTriton, N=H, EPS=C["EPS"]), x, ln1) for x in h]
            g = [B.call(bind(b1.Gemm, K=H, N=H), x, o_w) for x in n]
            own2 = emb if wrong_producer else g
            h2 = [B.call(bind(b1_tp2.AllReduce2, N=H), parts(x, peer(k, i))) for i, x in enumerate(own2)]
            k += 1
            f = [B.call(bind(b1.RMSNormTriton, N=H, EPS=C["EPS"]), x, norm) for x in h2]
            own_logits = B.call(bind(b1.Gemm, K=H, N=VS), f[-1], embed)
            lg = B.call(bind(b1_tp2.AllGather2, N=VS), parts(own_logits, peer(k, 0)))
            k += 1
            t = B.call(bind(b1.TokenSelect, V=V_FULL), lg)
            toks.append(t)
            outs.append(t)
        return Coll(ret, concat([o.refs for o in outs]))

    defn = CompositeDefinition(f"ToyRank{rank}", 1, (), lambda S, _p=tuple(params), _r=ret: (_p, _r), body, register=False)
    return bind(defn)


def write_derived(d, spec: SpecializedDefinition, rank: int, *, links_mut: dict | None = None, spec_histogram: dict | None = None):
    """descriptor.json.gz + derive-report.json + tp_links.json + result.json as derive_step writes them (the fields cb_a.compare and the
    fold's construction read)."""
    from verity_vllm.pipeline.build import _spec_histogram
    prog = Program(spec)
    desc = codec.encode_program(prog)
    with gzip.open(d / "descriptor.json.gz", "wt") as f:
        json.dump(desc, f)
    (d / "result.json").write_text(json.dumps({"spec_histogram": _spec_histogram(prog) if spec_histogram is None else spec_histogram}))
    weights = ["model.embed_tokens.weight", "model.layers.0.input_layernorm.weight", "model.layers.0.self_attn.o_proj.weight", "model.norm.weight"]
    user_inputs = ["prompt"] + [f"peers_{k}" for k in range(len(PEER_SHAPES))]
    (d / "derive-report.json").write_text(json.dumps({"provenance": {"source": {"example_inputs": [[0]] * len(user_inputs)},
                                                                     "weights": [{"field": j, "name": n} for j, n in enumerate(weights)]}}))
    links = []
    for k, (r, n) in enumerate(PEER_SHAPES):
        ln = {"k": k, "kind": KINDS[k % 3], "shape": [r, n], "dtype": "bfloat16", "peer_input": f"peers_{k}", "peer_rank": 1 - rank,
              "peer_collective_k": k, "module_path": f"inner.model.{SITES[k % 3]} ({SITE_CLASSES[k % 3]})"}
        ln.update((links_mut or {}).get(k, {}))
        links.append(ln)
    (d / "tp_links.json").write_text(json.dumps({"schema": "cb-a/tp_links/v1", "world": WORLD, "rank": rank, "user_inputs": user_inputs,
                                                 "prompt_inputs": ["prompt"], "peer_inputs": user_inputs[1:], "links": links}))
    return desc


def persist_fold(result, d, rank: int):
    from verity_vllm.pipeline.match_fold import persist_program
    persist_program(result, d, profile_name="toy-tp2-rank", log_path=f"log.rank{rank}.jsonl.gz")


def compare_rank(tmp_path, rank: int, result, spec: SpecializedDefinition, *, links_mut: dict | None = None, compact: bool = False):
    from verity_vllm.check.match import program_compare as CMP
    fd, dd = tmp_path / f"fold_r{rank}", tmp_path / f"build_r{rank}"
    fd.mkdir(exist_ok=True)
    dd.mkdir(exist_ok=True)
    persist_fold(result, fd, rank)
    write_derived(dd, spec, rank, links_mut=links_mut)
    A, B = CMP.load_oracle(str(fd), "oracle", compact=compact), CMP.load_derived(str(dd), "derived", compact=compact)
    res = CMP.compare_steps(A, B)
    # the record's verdict rule (`pipeline.row_records.match_summary`, B=1): canonical instance sequences equal, same step count, no differing operand
    res["ok"] = bool(res["canonical_sha256"]["equal"]) and res["a"]["steps"] == res["b"]["steps"] and int(res["totals"].get("operands_different", 0)) == 0
    return res, A, B


# ---------------------------------------------------------------------------------------------------------- tests: the exact case


def test_exact_unsupported_case_without_the_fold_rule():
    """The older record at toy size: without the collective rule the FIRST collective (k=0, model.embed_tokens) is Unsupported and the
    RMSNorm that reads the received storage is an unresolved read of an UNWRITTEN view."""
    r = fold_rank(0, with_collectives=False)
    rep = r.report.to_dict()
    assert rep["unsupported"] == 6                      # 3 collectives x 2 steps, nothing else
    uns = rep["unsupported_examples"]
    first = min(uns, key=lambda e: e["id"])
    assert first["op"] == "vllm.all_reduce.default" and first["reason"] == "no pattern"
    assert sorted(u["op"] for u in uns) == ["vllm.all_gather.default"] * 2 + ["vllm.all_reduce.default"] * 4
    unres = [e for e in r.errors if e["class"] == "Unresolved"]
    assert unres, "the received storage must surface as an unresolved read"
    # the very next kernel (the RMSNorm reading the received storage) is the first unresolved read: an UNWRITTEN view
    assert unres[0]["event"] == first["id"] + 1 and unres[0]["op"].endswith("_rms_norm_kernel")
    assert "unwritten" in unres[0]["message"] and unres[0]["step"] == 0
    assert not r.reference_eligible


def test_fold_rule_resolves_both_ranks():
    for rank in (0, 1):
        r = fold_rank(rank)
        assert r.errors == [] and r.report.to_dict()["unsupported"] == 0, (rank, r.errors[:3])
        assert r.tp == {"rank": rank, "world": WORLD}
        assert [n for n, _ in r.params] == ["prompt", "peers", "weights"]          # weights LAST; peers between the request inputs and them
        peers_t = dict(r.params)["peers"]
        assert isinstance(peers_t, RecordType)
        assert [(n, t.n, t.elem.n) for n, t in peers_t.fields] == [("s0_k0", 2, H), ("s0_k1", 2, H), ("s0_k2", 1, VS), ("s1_k0", 1, H), ("s1_k1", 1, H), ("s1_k2", 1, VS)]
        assert [(c["step"], c["k"], c["kind"], c["site"], c["rows"], c["N"]) for c in r.collectives] == [
            (0, 0, "all_reduce", "model.embed_tokens", 2, H), (0, 1, "all_reduce", "model.layers.0.self_attn.o_proj", 2, H),
            (0, 2, "all_gather", "logits_processor", 1, VS), (1, 0, "all_reduce", "model.embed_tokens", 1, H),
            (1, 1, "all_reduce", "model.layers.0.self_attn.o_proj", 1, H), (1, 2, "all_gather", "logits_processor", 1, VS)]
        assert r.peers_layout[0]["request"] == REQ and r.peers_layout[0]["leaves"] == 2 * H + 2 * H + VS + H + H + VS
        assert r.reference_eligible


def test_parts_are_in_rank_order_and_received_is_the_instance_output():
    """rank 0: parts = [own, peer]; rank 1: parts = [peer, own].  The RMSNorm reads the AllReduce2 instance's OUTPUT node, not a root."""
    for rank in (0, 1):
        r = fold_rank(rank)
        ar = [m for m in r.instances if m["spec"].startswith("AllReduce2_v1") and m["step"] == 0 and m["collective"]["k"] == 0]
        assert len(ar) == 2 and all(m["collective"]["rank"] == rank and m["collective"]["peer_field"] == "s0_k0" for m in ar)
        for i, m in enumerate(ar):
            rs = list(runs(r.body.nodes[m["node"]].args[0].refs))
            assert len(rs) == 2, rs
            own, peer = (rs[0], rs[1]) if rank == 0 else (rs[1], rs[0])
            assert own[0] == "n" and r.body.nodes[own[1]].fn.id.startswith("EmbeddingShard_v1")   # produced in THIS rank's log
            assert peer == ("p", 1, i * H, (i + 1) * H)                                            # the `peers` parameter, field s0_k0 row i
        rms = [m for m in r.instances if m["spec"].startswith("RMSNormTriton") and m["step"] == 0 and "input_layernorm" in m["hint"]]
        assert len(rms) == 2
        for i, m in enumerate(rms):
            rs = list(runs(r.body.nodes[m["node"]].args[0].refs))
            assert rs == [("n", ar[i]["node"], 0, H)]                                              # linked to its producer: the collective instance
        # the received storage is never a Root: the fold synthesised no root event and the memory model holds a node write for it
        assert all(e["class"] != "Unresolved" for e in r.errors)


def test_fold_refuses_a_collective_without_the_tp_header_note():
    r = fold_rank(0, tp_note=False)
    assert r.errors and any("RunHeader.notes.tp2" in e["message"] for e in r.errors)


def test_persisted_record_states_the_layout(tmp_path):
    r = fold_rank(0)
    persist_fold(r, tmp_path, 0)
    pj = json.loads((tmp_path / "program.json").read_text())
    assert [n for n, _ in pj["params"]] == ["prompt", "peers", "weights"] and pj["tp"] == {"rank": 0, "world": 2}
    assert [f["name"] for f in pj["peers_layout"][0]["fields"]] == ["s0_k0", "s0_k1", "s0_k2", "s1_k0", "s1_k1", "s1_k2"]
    assert pj["n_input_nodes"] == 3       # prompt, peers (one homogeneous RecordType), weights (one homogeneous RecordType)
    rows = [json.loads(ln) for ln in (tmp_path / "instances.jsonl").read_text().splitlines()[1:]]
    coll = [o for o in rows if o.get("collective")]
    assert len(coll) == 2 + 2 + 1 + 1 + 1 + 1 and coll[0]["collective"]["site"] == "model.embed_tokens"


# ---------------------------------------------------------------------------------------------------------- tests: Build vs fold (compare)


def test_compare_fold_against_rank_program_passes_both_ranks(tmp_path):
    for rank in (0, 1):
        res, A, B = compare_rank(tmp_path, rank, fold_rank(rank), rank_program(rank))
        assert res["ok"], json.dumps(res, default=str)[:2000]
        assert A.peer_fields == {1: [(0, 0), (2 * H, 1), (4 * H, 2), (4 * H + VS, 3), (5 * H + VS, 4), (6 * H + VS, 5)]}
        assert B.peer_fields == {k + 1: [(0, k)] for k in range(6)}
        # the PEER canonical source agrees instance by instance (k = the request's k-th collective, offsets field-relative)
        ar = next(i for i, r in enumerate(A.inst) if r["spec"].startswith("AllReduce2_v1"))
        br = next(i for i, r in enumerate(B.inst) if r["spec"].startswith("AllReduce2_v1"))
        assert A.canon(ar) == B.canon(br)
        assert any(t[0] == "PEER" for t in A.canon(ar)[0])


def _first_fail(res) -> str:
    for s in res.get("steps", []):
        if not s.get("ok", True):
            return json.dumps(s, default=str)
    return json.dumps(res, default=str)


@pytest.mark.parametrize("name, fold_kw, prog_kw", [
    ("swapped_rank", {}, {"swap_parts": True}),                       # parts in the wrong rank order: wiring differs
    ("wrong_peer", {}, {"peer_of": {1: 0}}),                          # step 0's o_proj collective fed the embedding collective's peer (same shape)
    ("off_by_one_step", {}, {"peer_of": {2: 5, 5: 2}}),               # the logits all-gather peers bound to the OTHER step's collective
    ("wrong_op", {}, {"wrong_op_k": 0}),                              # AllGather2 where the log issued an all_reduce
    ("equal_result_wrong_producer", {"mut": {"wrong_producer": True}}, {}),   # the log's k=1 reduces the embedding partial, not the Gemm's
    ("wrong_count", {"mut": {"drop_collective": 0}}, {}),             # the log never reduced the embedding partial
    ("reordered_lanes", {"mut": {"reorder": True}}, {}),              # k=1 issued after k=2 on this rank
])
def test_negatives_fail_the_compare_by_name(tmp_path, name, fold_kw, prog_kw):
    r = fold_rank(0, **fold_kw)
    res, _A, _B = compare_rank(tmp_path, 0, r, rank_program(0, **prog_kw))
    assert not res["ok"], f"{name}: the compare must FAIL: {_first_fail(res)[:600]}"


# ---------------------------------------------------------------------------------------------------------- tests: the cross-rank check


REC_SHA = {0: "0" * 64, 1: "1" * 64}      # the recorder files' digests as the unit tests' cross-rank check loads them (no files here)


def test_swapped_rank_header_fails_the_compare(tmp_path):
    """The log claims rank 1 (RunHeader.notes.tp2) but the record is compared against rank 0's Program: parts order differs -> FAIL."""
    r = fold_rank(1)
    res, _A, _B = compare_rank(tmp_path, 1, r, rank_program(0))
    assert not res["ok"]


# ---------------------------------------------------------------------------------------------------------------------------------
# the B2 form: two co-scheduled requests; the peer field's M_r and the own-part read range derive from the StepStart row
# attribution.  Positive: both ranks fold, every request's projection equals its request Program (batch_decomp X-09), cross-rank PASS.
# Negative: ONE row boundary shifted (r0 3 rows / r1 1 row instead of 2 / 2) FAILS BY NAME -- at the fold's stated segmentation, at
# the cross-rank segmentation check, and at the per-request compare -- not by value.


def fold_rank_b2(rank: int, *, shift: bool = False):
    return fold(rank_log_b2(rank, shift=shift), rank_profile(), strict=False, with_program=True)


def _segments(r):
    """{request: [(field, rows)]} as the fold states them."""
    return {pl["request"]: [(f["name"], f["rows"]) for f in pl["fields"]] for pl in r.peers_layout}


def _decompose_b2(tmp_path, rank: int, r, *, tag: str):
    from verity_vllm.check.match import batch_decomp as BD
    od, dd = tmp_path / f"oracle_{tag}_r{rank}", tmp_path / f"derived_{tag}_r{rank}"
    (od / "capture").mkdir(parents=True)
    dd.mkdir()
    persist_fold(r, od, rank)
    with gzip.open(od / "capture" / "log.jsonl.gz", "wt") as f:
        f.write(dump_events(rank_log_b2(rank, shift=(tag == "shift"))))
    write_derived(dd, rank_program(rank), rank)
    rep = json.loads((dd / "derive-report.json").read_text())
    rep["structural_inputs"] = {"LP": 2, "T": 1}                     # the request Program's shape (LP 2, one served decode step)
    (dd / "derive-report.json").write_text(json.dumps(rep))
    return BD.decompose(str(od), [str(dd)], within_step="dag")


def test_b2_fold_states_per_request_peer_segments_from_the_step_rows():
    for rank in (0, 1):
        r = fold_rank_b2(rank)
        assert r.errors == [], r.errors[:3]
        assert [n for n, _ in r.params] == ["prompt_0", "prompt_1", "peers_0", "peers_1", "weights"]
        assert _segments(r) == {"r0": [("s0_k0", 2), ("s0_k1", 2), ("s0_k2", 1), ("s1_k0", 1), ("s1_k1", 1), ("s1_k2", 1)],
                                "r1": [("s0_k0", 2), ("s0_k1", 2), ("s0_k2", 1), ("s1_k0", 1), ("s1_k1", 1), ("s1_k2", 1)]}
        c0 = r.collectives[0]
        assert c0["rows"] == 4 and [(f["request"], f["rows"], f["offset"]) for f in c0["fields"]] == [("r0", 2, 0), ("r1", 2, 0)]
        # the own-part read of request r1's rows is the SAME rows the StepStart attributes to r1 (rows 2, 3 of the 4-row partial)
        ar = [m for m in r.instances if m["spec"].startswith("AllReduce2_v1") and m["step"] == 0 and m["collective"]["k"] == 0]
        pidx = {pl["request"]: pl["param"] for pl in r.peers_layout}
        assert [m["collective"]["peer_param"] for m in ar] == [pidx["r0"], pidx["r0"], pidx["r1"], pidx["r1"]]
        assert [(m["collective"]["peer_field"], m["collective"]["row_in_request"]) for m in ar] == [("s0_k0", 0), ("s0_k0", 1), ("s0_k0", 0), ("s0_k0", 1)]


def test_b2_every_request_projection_equals_its_program_both_ranks(tmp_path):
    for rank in (0, 1):
        res = _decompose_b2(tmp_path, rank, fold_rank_b2(rank), tag="ok")
        assert res["verdict"] == "PASS", json.dumps({k: res.get(k) for k in ("verdict", "reasons", "missing_derived")}, default=str)
        assert [(e["request_id"], e["LP"], e["T"], e["canonical_equal"]) for e in res["requests"]] == [("r0", 2, 1, True), ("r1", 2, 1, True)]
        assert res["xreq_total"] == 0                                 # no request reads another request's peer rows


