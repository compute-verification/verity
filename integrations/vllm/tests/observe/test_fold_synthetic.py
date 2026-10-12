"""Synthetic end-to-end fold: a toy configuration (H=16, NH=2, KVH=1, D=8, I=16, V=16, NL=1; LP=2, STEPS=1) of the REAL
`b1` definitions, hand-written raw observation events (prefill step with 2 rows, decode step with a StepLink) and a
toy pattern set, folded into `Captured_v1{PROFILE, WORKLOAD}` and compared with `Serve_v4` by `match_program`.

While `memory.py` / `views.py` are skeletons (lane L2 in flight) the test installs a small in-file stub memory model
through monkeypatching (`fold.MemoryModel`, `fold.row_view`); once the real modules import and work, it uses them."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import pytest

from verity_vllm.observe.fold import fold as fold_mod
from verity_vllm.observe.events import (DispatchEnter, DispatchExit, Root, RunHeader, SampleEnd, StepEnd, StepLink, StepStart,
                                   StorageAlloc, TritonLaunch, ViewDesc)
from verity_vllm.observe.fold.fold import FoldError, fold, instance_nodes
from verity_vllm.observe.fold.memory import MemoryStats, Unresolved
from verity_vllm.observe.fold.patterns.norms import eps_input
from verity_vllm.observe.fold.resolver import Compute, Gather, GatherMany, Group, PerRow, Profile, Shared, Supplied, Transparent
from verity.ml.circuits.defs import bind
from verity.ml.circuits.refs import Affine, Explicit, concat
from verity_vllm.program.registry import b1
from verity_vllm.program.registry import value_inputs as VI
from verity.ml.circuits.types import Array, Record, Value

# ---------------------------------------------------------------------------------------------------------
# toy configuration
# ---------------------------------------------------------------------------------------------------------

C = {"H": 16, "NL": 1, "NH": 2, "KVH": 1, "D": 8, "I": 16, "V": 16, "QKV": 32, "GU": 32, "EPS": 1e-5, "MAX_POS": 8, "BIAS": False}
BN = 128
REQ = "r0"
PROMPT = [5, 7]
SAMPLED = [9, 3]  # step 0 (prefill) and step 1 (decode)
BS = 4  # KV block size

BF, I32D, I64D = "bfloat16", "int32", "int64"


def V(a, shape, strides=None, off=0, dtype=BF):
    shape = tuple(shape)
    if strides is None:
        strides = []
        s = 1
        for n in reversed(shape):
            strides.append(s)
            s *= n
        strides = tuple(reversed(strides))
    return ViewDesc(a, off, shape, tuple(strides), dtype, "cuda:0")


# allocation ids
A_EMBED, A_LN1, A_QKV_W, A_O_W, A_LN2, A_GU_W, A_DOWN_W, A_NORM, A_COS = 1, 2, 3, 4, 5, 6, 7, 8, 9
A_IDS, A_POS, A_KC, A_VC = 10, 11, 12, 13

W_EMBED = V(A_EMBED, (16, 16))
W_LN1, W_LN2, W_NORM = V(A_LN1, (16,)), V(A_LN2, (16,)), V(A_NORM, (16,))
W_QKV = V(A_QKV_W, (32, 16))
W_O = V(A_O_W, (16, 16))
W_GU = V(A_GU_W, (32, 16))
W_DOWN = V(A_DOWN_W, (16, 16))
W_COS = V(A_COS, (8, 8))
KC = V(A_KC, (2, BS, 1, 8))
VC = V(A_VC, (2, BS, 1, 8))


def weight_field(name: str):
    import re
    if name in ("model.embed_tokens.weight", "lm_head.weight"):
        return ("embed",)
    if name == "model.norm.weight":
        return ("norm",)
    m = re.match(r"model\.layers\.(\d+)\.(.*)", name)
    if not m:
        return None
    L, rest = int(m.group(1)), m.group(2)
    table = {"input_layernorm.weight": "ln1", "self_attn.qkv_proj.weight": "qkv_w", "self_attn.o_proj.weight": "o_w",
             "post_attention_layernorm.weight": "ln2", "mlp.gate_up_proj.weight": "gate_up_w", "mlp.down_proj.weight": "down_w"}
    if rest == "self_attn.rotary_emb.cos_sin_cache":
        return ("cos_sin",)
    f = table.get(rest)
    return ("layers", L, f) if f else None


def preamble():
    hdr = RunHeader("verity-capture/log/v1", "2026-09-16T00:00:00Z", "toy-profile", {"vllm": "toy"}, {"repo": "toy", "config": C},
                    {"requests": [{"request_id": REQ, "prompt_token_ids": PROMPT, "max_tokens": 2}], "sampling": {"temperature": 0.0}})
    evs = [hdr]
    weights = [(A_EMBED, W_EMBED, "model.embed_tokens.weight"), (A_LN1, W_LN1, "model.layers.0.input_layernorm.weight"),
               (A_QKV_W, W_QKV, "model.layers.0.self_attn.qkv_proj.weight"), (A_O_W, W_O, "model.layers.0.self_attn.o_proj.weight"),
               (A_LN2, W_LN2, "model.layers.0.post_attention_layernorm.weight"), (A_GU_W, W_GU, "model.layers.0.mlp.gate_up_proj.weight"),
               (A_DOWN_W, W_DOWN, "model.layers.0.mlp.down_proj.weight"), (A_NORM, W_NORM, "model.norm.weight")]
    for a, v, name in weights:
        evs.append(StorageAlloc(a, 0x1000 * a, v.numel * 2, "cuda:0", 0, "param"))
        evs.append(Root(v, "weight", name))
    evs.append(Root(W_EMBED, "weight", "lm_head.weight"))  # tied: aliases the embedding allocation
    evs.append(StorageAlloc(A_COS, 0x1000 * A_COS, W_COS.numel * 2, "cuda:0", 0, "buffer"))
    evs.append(Root(W_COS, "buffer", "model.layers.0.self_attn.rotary_emb.cos_sin_cache"))
    for a, nbytes in ((A_IDS, 4 * 4), (A_POS, 4 * 8), (A_KC, KC.numel * 2), (A_VC, VC.numel * 2)):
        evs.append(StorageAlloc(a, 0x1000 * a, nbytes, "cuda:0", 0, "op"))
    return evs


@dataclass
class Ids:
    n: int = 100

    def next(self) -> int:
        self.n += 1
        return self.n


def step_events(step: int, positions: list[int], ids: Ids, base_alloc: int, sampled: int, ids_dtype: str = I32D,
                extra_unknown_op: bool = False, skip_norm0: bool = False):
    """One engine step over `positions` (contiguous, one request): the b1 kernel sequence as raw events."""
    M = len(positions)
    R = 1
    p0 = positions[0]
    a = itertools.count(base_alloc)
    ids_view = V(A_IDS, (4,), dtype=ids_dtype)
    pos_view = V(A_POS, (4,), dtype=I64D)
    hidden, normed, qkv, attn, o, gu, act, down, gathered, logits, sampled_v = (V(next(a), s, dtype=d) for s, d in (
        ((M, 16), BF), ((M, 16), BF), ((M, 32), BF), ((M, 2, 8), BF), ((M, 16), BF), ((M, 32), BF), ((M, 16), BF), ((M, 16), BF),
        ((R, 16), BF), ((R, 16), BF), ((R,), I64D)))
    seq_len = p0 + M
    last = M - 1
    evs = [StepStart(step, [(REQ, p) for p in positions], [PROMPT[p] if p < len(PROMPT) else SAMPLED[p - len(PROMPT)] for p in positions],
                     positions, {REQ: seq_len}, {REQ: p0}, {REQ: [0]}, BS, [p for p in positions], [(REQ, positions[-1])], [last],
                     1, M, M, ViewDesc(A_IDS, 0, (M,), (1,), ids_dtype, "cuda:0"), pos_view, "prefill" if p0 == 0 else "decode")]
    if p0 == 0:
        evs.append(Root(ViewDesc(A_IDS, 0, (M,), (1,), ids_dtype, "cuda:0"), "prompt_token", f"prompt[{REQ}]", list(PROMPT[:M])))
    else:
        for i, p in enumerate(positions):
            evs.append(StepLink(step, i, REQ, step - 1, "eq", SAMPLED[p - len(PROMPT)]))
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

    ids_rows = ViewDesc(A_IDS, 0, (M,), (1,), ids_dtype, "cuda:0")
    dispatch("aten.embedding.default", [W_EMBED, ids_rows], {}, hidden, "model.embed_tokens")
    if extra_unknown_op:
        dispatch("aten.mysterious.default", [hidden], {}, hidden, "model")
    if not skip_norm0:
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
    evs.append(SampleEnd(step, logits, [sampled], sampled_v))
    evs.append(StepEnd(step))
    return evs


def toy_log(ids_dtype: str = I32D, extra_unknown_op: bool = False, skip_norm0: bool = False):
    ids = Ids()
    evs = preamble()
    evs += step_events(0, [0, 1], ids, 20, SAMPLED[0], ids_dtype, extra_unknown_op, skip_norm0)
    evs += step_events(1, [2], ids, 40, SAMPLED[1], ids_dtype)
    return evs


# ---------------------------------------------------------------------------------------------------------
# toy patterns (the fold only sees resolutions; these are minimal stand-ins for L3's profile)
# ---------------------------------------------------------------------------------------------------------


def _slot(ctx, req, p):
    bt = ctx.start.block_tables[req]
    return bt[p // ctx.start.block_size] * ctx.start.block_size + p % ctx.start.block_size


def _as3(v, n_heads, d):
    return ViewDesc(v.alloc, v.offset, (v.shape[0], n_heads, d), (v.strides[0], d, 1), v.dtype, v.device)


class Pat:
    def __init__(self, name, fn):
        self.name, self.fn = name, fn

    def match(self, node, ctx):
        return self.fn(node, ctx)


def p_embedding(node, ctx):
    if node.op != "aten.embedding.default":
        return None
    w, idx = node.args
    return Compute(bind(b1.Embedding, V=C["V"], H=C["H"]), idx.shape[0], (PerRow(idx), Shared(w)), (PerRow(node.outs),), (node.id,), node.enter.module or "")


def p_rms_triton(node, ctx):
    if not node.is_triton or not node.op.endswith("_rms_norm_kernel"):
        return None
    a = node.enter.args
    spec, operands = bind(b1.RMSNormTriton, N=C["H"], EPS=C["EPS"]), (PerRow(a["input_ptr"]), Shared(a["weight_ptr"]))
    if ctx.profile.registered_values:
        spec, operands = bind(VI.RMSNormTritonV3, N=C["H"]), operands + (eps_input(C["EPS"]),)
    return Compute(spec, a["input_ptr"].shape[0], operands, (PerRow(a["output_ptr"]),), (node.id,), node.enter.module or "")


def p_mm(node, ctx):
    if node.op != "aten.mm.default" or len(node.children) != 1 or not node.children[0].op.endswith("matmul_kernel_persistent"):
        return None
    a, bT = node.args
    K, N = bT.shape
    if bT.strides != (1, K):
        return None
    w = ViewDesc(bT.alloc, bT.offset, (N, K), (K, 1), bT.dtype, bT.device)
    return Compute(bind(b1.Gemm, K=K, N=N), a.shape[0], (PerRow(a), Shared(w)), (PerRow(node.outs),), (node.id, node.children[0].id), node.enter.module or "")


def p_rope(node, ctx):
    if node.op != "_C.rotary_embedding.default":
        return None
    _, q, k, d, cs, _neox = node.args
    pos = tuple(ctx.start.positions)
    q3, k3 = _as3(q, C["NH"], d), _as3(k, C["KVH"], d)
    parts = (Compute(bind(b1.RoPE, NHEADS=C["NH"], D=d), q.shape[0], (PerRow(q3), Gather(cs, pos)), (PerRow(q3),), (), node.enter.module or ""),
             Compute(bind(b1.RoPE, NHEADS=C["KVH"], D=d), k.shape[0], (PerRow(k3), Gather(cs, pos)), (PerRow(k3),), (), node.enter.module or ""))
    return Group(parts, (node.id,), node.enter.module or "")


def _kv_flat(kc):
    nb, bs, kvh, d = kc.shape
    return ViewDesc(kc.alloc, kc.offset, (nb * bs, kvh, d), (kvh * d, d, 1), kc.dtype, kc.device)


def p_kv_update(node, ctx):
    if node.op != "vllm.unified_kv_cache_update.default" or len(node.children) != 1:
        return None
    ch = node.children[0]
    k, v, kc, vc = ch.args[:4]
    slots = tuple(ctx.start.slot_mapping)
    parts = (Transparent("move", (), PerRow(_as3(k, C["KVH"], C["D"])), Gather(_kv_flat(kc), slots), k.shape[0], "k"),
             Transparent("move", (), PerRow(_as3(v, C["KVH"], C["D"])), Gather(_kv_flat(vc), slots), v.shape[0], "v"))
    return Group(parts, (node.id, ch.id), node.enter.module or "")


def p_attention(node, ctx):
    if node.op != "vllm.unified_attention_with_output.default" or len(node.children) != 1:
        return None
    ch = node.children[0]
    q3, kc, vc, out = ch.args[:4]
    parts = []
    for i, (req, p) in enumerate(ctx.start.rows):
        T = p + 1
        slots = tuple(_slot(ctx, req, j) for j in range(T))
        spec = bind(b1.AttentionV3, T=T, NH=C["NH"], KVH=C["KVH"], D=C["D"], BN=BN)
        operands = (Gather(q3, (i,)), GatherMany(_kv_flat(kc), (slots,)), GatherMany(_kv_flat(vc), (slots,)))
        if ctx.profile.registered_values:
            spec = bind(VI.AttentionV10, T=T, NH=C["NH"], KVH=C["KVH"], D=C["D"], BN=BN)
            operands += (Supplied("registration", f"scale_log2_d{C['D']}", 32, b1._scale_log2_bits(C["D"])),)
        parts.append(Compute(spec, 1, operands, (Gather(out, (i,)),), (), node.enter.module or "", attribution=((req, p),)))
    return Group(tuple(parts), (node.id, ch.id), node.enter.module or "")


def p_fused_norm(node, ctx):
    if node.op != "_C.fused_add_rms_norm.default":
        return None
    x, res, w, eps = node.args
    spec, operands = bind(b1.RMSNormFusedCuda, N=C["H"], EPS=eps), (PerRow(x), PerRow(res), Shared(w))
    if ctx.profile.registered_values:
        spec, operands = bind(VI.RMSNormFusedCudaV4, N=C["H"]), operands + (eps_input(eps),)
    return Compute(spec, x.shape[0], operands, (PerRow(x), PerRow(res)), (node.id,), node.enter.module or "")


def p_silu(node, ctx):
    if node.op != "_C.silu_and_mul.default":
        return None
    out, inp = node.args
    return Compute(bind(b1.SiluMul, I=C["I"]), inp.shape[0], (PerRow(inp),), (PerRow(out),), (node.id,), node.enter.module or "")


def p_index(node, ctx):
    if node.op != "aten.index.default":
        return None
    src = node.args[0]
    idx = tuple(ctx.start.logits_indices)
    return Transparent("move", (node.id,), Gather(src, idx), PerRow(node.outs), len(idx), "logits_indices")


def p_argmax(node, ctx):
    if node.op != "aten.argmax.default" or node.phase != "sample":
        return None
    logits = node.args[0]
    return Compute(bind(b1.TokenSelect, V=C["V"]), logits.shape[0], (PerRow(logits),), (PerRow(node.outs),), (node.id,), "sampler")


PATTERNS = [Pat(n, f) for n, f in (("embedding", p_embedding), ("rms_triton", p_rms_triton), ("mm", p_mm), ("rope", p_rope),
                                    ("kv_update", p_kv_update), ("attention", p_attention), ("fused_norm", p_fused_norm),
                                    ("silu", p_silu), ("index", p_index), ("argmax", p_argmax))]


def toy_profile(accepted=False, registered_values=False):
    return Profile("toy", PATTERNS, dict(C), weight_field, accepted, registered_values=registered_values)


# ---------------------------------------------------------------------------------------------------------
# stub memory model (only while L2's memory.py is a skeleton)
# ---------------------------------------------------------------------------------------------------------


def stub_row_view(v: ViewDesc, i: int) -> ViewDesc:
    if not v.shape:
        raise ValueError("rank-0 view has no rows")
    if not (0 <= i < v.shape[0]):
        raise ValueError(f"row {i} outside {v!r}")
    return ViewDesc(v.alloc, v.offset + i * v.strides[0], v.shape[1:], v.strides[1:], v.dtype, v.device)


def stub_rows_view(v: ViewDesc, lo: int, hi: int) -> ViewDesc:
    if not (0 <= lo <= hi <= v.shape[0]):
        raise ValueError(f"rows [{lo}, {hi}) outside {v!r}")
    return ViewDesc(v.alloc, v.offset + lo * v.strides[0], (hi - lo,) + v.shape[1:], v.strides, v.dtype, v.device)


def _elements(v: ViewDesc):
    for idx in itertools.product(*(range(n) for n in v.shape)):
        yield v.offset + sum(i * s for i, s in zip(idx, v.strides))


def _compact(refs):
    parts = []
    for r in refs:
        if parts and parts[-1].space == r[0] and parts[-1].idx == r[1] and parts[-1].base + parts[-1].count == r[2]:
            parts[-1] = Affine(r[0], r[1], parts[-1].base, parts[-1].count + 1)
        else:
            parts.append(Affine(r[0], r[1], r[2], 1))
    return concat(parts) if parts else Explicit(())


class StubMemory:
    def __init__(self):
        self.cells: dict[int, dict] = {}
        self.writes = self.reads = 0

    def alloc(self, ev):
        self.cells[ev.alloc] = {}

    def free(self, a):
        self.cells.pop(a)

    def write(self, view, refs):
        if refs.count != view.numel:
            raise ValueError(f"write of {refs.count} refs into {view!r}")
        cells = self.cells[view.alloc]
        for j, e in enumerate(_elements(view)):
            cells[e] = (refs[j], view.bits)
        self.writes += 1

    def read(self, view):
        cells = self.cells.get(view.alloc)
        if cells is None:
            raise Unresolved(view, [(0, view.numel)], "unknown or freed allocation")
        out, missing = [], []
        for j, e in enumerate(_elements(view)):
            c = cells.get(e)
            if c is None or c[1] != view.bits:
                missing.append((j, j + 1))
            else:
                out.append(c[0])
        if missing:
            raise Unresolved(view, missing)
        self.reads += 1
        return _compact(out)

    def read_rows(self, view, indices):
        return concat([self.read(stub_row_view(view, i)) for i in indices])

    def move(self, src, dst):
        self.write(dst, self.read(src))

    def coverage(self, view):
        cells = self.cells.get(view.alloc, {})
        return sum(1 for e in _elements(view) if e in cells) / max(view.numel, 1)

    def stats(self):
        return MemoryStats(len(self.cells), len(self.cells), self.writes, self.reads, 0, sum(len(c) for c in self.cells.values()))


def _real_memory_available() -> bool:
    try:
        from verity_vllm.observe.fold.memory import MemoryModel
        from verity_vllm.observe.fold.views import row_view
        MemoryModel()
        row_view(V(1, (2, 3)), 1)
        return True
    except NotImplementedError:
        return False


def install_stub_memory() -> None:
    """Point fold.py at the in-test memory model / view helpers (only while L2's memory.py is a skeleton)."""
    fold_mod.MemoryModel = StubMemory
    fold_mod.row_view = stub_row_view
    fold_mod.rows_view = stub_rows_view


@pytest.fixture
def memory(monkeypatch):
    if _real_memory_available():
        return "real"
    monkeypatch.setattr(fold_mod, "MemoryModel", StubMemory)
    monkeypatch.setattr(fold_mod, "row_view", stub_row_view)
    monkeypatch.setattr(fold_mod, "rows_view", stub_rows_view)
    return "stub"


# ---------------------------------------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------------------------------------


def test_fold_builds_program_and_is_deterministic(memory):
    r1 = fold(toy_log(), toy_profile(), strict=True)
    r2 = fold(toy_log(), toy_profile(), strict=True)
    assert r1.report.unsupported == 0 and not r1.errors
    assert r1.program is not None and r1.digest and r1.digest == r2.digest
    assert r1.body.ret_type == Array(2, Value(32))
    assert [n for n, _ in r1.params] == ["prompt", "weights"]
    assert r1.params[0][1] == Array(2, Value(32)) and r1.params[1][1] == b1.weights_type(C)
    assert r1.spec.id.startswith("Captured_v1{PROFILE='toy-profile',WORKLOAD='")
    assert r1.sampled == {REQ: [(0, 9), (1, 3)]}
    # per-specialization instance counts: prefill 2 rows + decode 1 row = 3 positions
    by = r1.report.by_spec
    assert by[f"Embedding_v1{{V=16,H=16}}"] == 3
    assert by["RMSNormTriton_v1{N=16,EPS=1e-05}"] == 3
    assert by["Gemm_v1{K=16,N=32}"] == 6          # qkv + gate_up per position
    assert by["Gemm_v1{K=16,N=16}"] == 8          # o + down per position, lm_head per step
    assert by["RoPE_v1{NHEADS=2,D=8}"] == 3 and by["RoPE_v1{NHEADS=1,D=8}"] == 3
    assert by["SiluMul_v1{I=16}"] == 3 and by["TokenSelect_v1{V=16}"] == 2
    assert by["RMSNormFusedCuda_v2{N=16,EPS=1e-05}"] == 3 + 3   # post-attention per position + final norm per row
    assert {k: v for k, v in by.items() if k.startswith("Attention_v3")} == {
        "Attention_v3{T=1,NH=2,KVH=1,D=8,BN=128}": 1, "Attention_v3{T=2,NH=2,KVH=1,D=8,BN=128}": 1, "Attention_v3{T=3,NH=2,KVH=1,D=8,BN=128}": 1}
    assert len(r1.body.nodes) == sum(by.values()) == len(r1.instances)
    assert r1.steps[0]["nodes"] + r1.steps[1]["nodes"] == len(r1.body.nodes)
    assert r1.program.gates > 0 and r1.timings["program_s"] >= 0


def test_registered_values_are_inputs_of_the_program(memory):
    """Under `registered_values` the norms' eps and the attention scale are fields of the registered `values` parameter, read by the
    Definitions that take them as inputs; the Program holds neither as a constant, and each request has a (here empty) run parameter."""
    import numpy as np

    from verity.ml.circuits.constants import REGISTRATION, RUN, inputs, program_data
    from verity_vllm.program.registry import prims as P

    r = fold(toy_log(), toy_profile(registered_values=True), strict=True)
    legacy = fold(toy_log(), toy_profile(), strict=True)
    assert r.report.unsupported == 0 and not r.errors and r.digest != legacy.digest
    assert [n for n, _ in r.params] == ["prompt", "run", "values", "weights"]
    eps, scale = int(P.bits_of(np.float32(C["EPS"]))), b1._scale_log2_bits(C["D"])
    assert r.params[2][1] == Record(("eps", Value(32)), ("scale_log2_d8", Value(32))) and r.params[1][1] == Record()
    assert r.supplied == {"registration": {"param": 2, "name": "values", "fields": [
                              {"name": "eps", "width": 32, "bits": f"0x{eps:08x}", "offset": 0},
                              {"name": "scale_log2_d8", "width": 32, "bits": f"0x{scale:08x}", "offset": 1}]},
                          "run": [{"param": 1, "name": "run", "request": REQ, "fields": []}],
                          "input_nodes": {"prompt": [0], "run": [], "values": [1], "weights": [2]}}
    assert r.program.bindings == (RUN, RUN, REGISTRATION, RUN)
    assert len(inputs(r.program.circuit)[REGISTRATION]) == 2
    by = r.report.by_spec
    assert by["RMSNormTriton_v3{N=16}"] == 3 and by["RMSNormFusedCuda_v4{N=16}"] == 6
    assert sum(v for k, v in by.items() if k.startswith("Attention_v10{")) == 3 and not any(k.startswith("Attention_v3{") for k in by)
    held = {cid for consts in program_data(r.program.root).values() for _g, cid in consts}
    assert {f"Const32[0x{eps:08x}]_v1", f"Const32[0x{scale:08x}]_v1"} & held == set()
    held_legacy = {cid for consts in program_data(legacy.program.root).values() for _g, cid in consts}
    assert {f"Const32[0x{eps:08x}]_v1", f"Const32[0x{scale:08x}]_v1"} <= held_legacy


def test_a_supplied_value_must_agree_with_its_first_use(memory, monkeypatch):
    def p_bad_fused(node, ctx):
        c = p_fused_norm(node, ctx)
        return None if c is None else Compute(c.spec, c.rows, c.operands[:3] + (Supplied("registration", "eps", 32, 0x3F800000),),
                                              c.outputs, c.consumed, c.hint)
    pats = [Pat(p.name, p_bad_fused) if p.name == "fused_norm" else p for p in PATTERNS]
    with pytest.raises(FoldError, match="value 'eps' is 32-bit 0x3f800000 here"):
        fold(toy_log(), Profile("toy", pats, dict(C), weight_field, False, registered_values=True), strict=True)


def test_step_link_and_kv_provenance(memory):
    r = fold(toy_log(), toy_profile(), strict=True)
    nodes = r.body.nodes
    # decode Embedding token operand == the prefill TokenSelect output (StepLink wiring)
    ts0 = instance_nodes(r, spec_prefix="TokenSelect_v1", step=0)[0]["node"]
    emb1 = instance_nodes(r, spec_prefix="Embedding_v1", step=1)[0]["node"]
    assert list(nodes[emb1].args[0].refs) == [("n", ts0, 0)]
    assert list(r.outputs[REQ][0]) == [("n", ts0, 0)]
    # prefill Embedding tokens are the prompt parameter leaves
    embs0 = [m["node"] for m in instance_nodes(r, spec_prefix="Embedding_v1", step=0)]
    assert [list(nodes[k].args[0].refs) for k in embs0] == [[("p", 0, 0)], [("p", 0, 1)]]
    # the return is the two TokenSelect outputs in step order
    ts1 = instance_nodes(r, spec_prefix="TokenSelect_v1", step=1)[0]["node"]
    assert list(r.body.ret) == [("n", ts0, 0), ("n", ts1, 0)]
    # attention K/V operands == RoPE-k outputs / the qkv GEMM's v slice, through the KV-cache move
    rope_k = {m["step"] * 10 + m["row"]: m["node"] for m in instance_nodes(r, spec_prefix="RoPE_v1{NHEADS=1")}
    qkv = {m["step"] * 10 + m["row"]: m["node"] for m in instance_nodes(r, spec_prefix="Gemm_v1{K=16,N=32}") if "qkv" in m["hint"]}
    att = {m["spec"]: m["node"] for m in instance_nodes(r, spec_prefix="Attention_v3")}
    a3 = nodes[att["Attention_v3{T=3,NH=2,KVH=1,D=8,BN=128}"]]
    want_k = [("n", rope_k[0], j) for j in range(8)] + [("n", rope_k[1], j) for j in range(8)] + [("n", rope_k[10], j) for j in range(8)]
    assert list(a3.args[1].refs) == want_k
    want_v = [("n", qkv[0], 24 + j) for j in range(8)] + [("n", qkv[1], 24 + j) for j in range(8)] + [("n", qkv[10], 24 + j) for j in range(8)]
    assert list(a3.args[2].refs) == want_v
    # q operand of the decode attention is the RoPE-q output of the decode row
    rope_q1 = instance_nodes(r, spec_prefix="RoPE_v1{NHEADS=2", step=1)[0]["node"]
    assert list(a3.args[0].refs) == [("n", rope_q1, j) for j in range(16)]
    # weights: the qkv GEMM reads the qkv_w field of the weights record, the lm_head GEMM the (tied) embed field
    wt = b1.weights_type(C)
    wcoll = fold_mod.Coll(wt, Affine("p", 1, 0, wt.leaves))
    assert list(nodes[qkv[0]].args[1].refs) == list(wcoll["layers"][0]["qkv_w"].refs)
    lm = [m["node"] for m in instance_nodes(r, spec_prefix="Gemm_v1{K=16,N=16}") if m["hint"] == "lm_head"]
    assert list(nodes[lm[0]].args[1].refs) == list(wcoll["embed"].refs)
    # instance names carry position / layer hints
    assert nodes[qkv[0]].name == "t=0/layers.0/qkv_proj" and a3.name == "t=2/layers.0/attn"


def test_int64_token_buffers_are_admitted(memory):
    r = fold(toy_log(ids_dtype=I64D), toy_profile(), strict=True)
    assert not r.errors and r.report.unsupported == 0
    ts0 = instance_nodes(r, spec_prefix="TokenSelect_v1", step=0)[0]["node"]
    emb1 = instance_nodes(r, spec_prefix="Embedding_v1", step=1)[0]["node"]
    assert list(r.body.nodes[emb1].args[0].refs) == [("n", ts0, 0)]


def test_unresolved_read_before_write(memory):
    # without the layer-0 norm launch the qkv GEMM reads a buffer nobody wrote
    with pytest.raises(Unresolved):
        fold(toy_log(skip_norm0=True), toy_profile(), strict=True, with_program=False)
    r = fold(toy_log(skip_norm0=True), toy_profile(), strict=False, with_program=False)
    assert r.unresolved >= 1 and r.errors[0]["class"] == "Unresolved" and r.errors[0]["step"] == 0
    assert "Gemm" in r.errors[0]["message"] or r.errors[0]["op"] == "aten.mm.default"


def test_unsupported_dev_records_strict_raises(memory):
    r = fold(toy_log(extra_unknown_op=True), toy_profile(), strict=False, with_program=False)
    assert r.report.unsupported == 1 and r.report.unsupported_ops == {"aten.mysterious.default": 1}
    assert r.steps[0]["unsupported"] == 1
    with pytest.raises(FoldError):
        fold(toy_log(extra_unknown_op=True), toy_profile(), strict=True, with_program=False)
    with pytest.raises(FoldError):  # the accepted profile defaults to strict
        fold(toy_log(extra_unknown_op=True), toy_profile(accepted=True), with_program=False)


def test_structure_matches_serve_v4_up_to_final_norm_rows(memory):
    from verity_vllm.program.frontend.rules.family import match_program
    r = fold(toy_log(), toy_profile(), strict=True, with_program=False)
    fam = bind(b1.ServeV4, C=C, LP=2, STEPS=1, BN=BN)
    rep = match_program(r.body, fam, flat_weights_param=1, allow_extra=True, with_map=False)
    assert rep.get("superset") is True, rep
    extra = rep["extra_nodes"]
    assert len(extra) == 1 and extra[0]["id"].startswith("RMSNormFusedCuda_v2{N=16,EPS=")  # descriptor id (canonical float)
    assert r.body.nodes[extra[0]["index"]].name == "t=0/norm"  # the final norm vLLM applies to every prefill row
    assert rep["weights_binding"]["embed"] == "embed" and rep["weights_binding"]["layers[0].qkv_w"] == "layers[0].qkv_w"
    assert rep["node_bijection"] == len(r.body.nodes) - 1
