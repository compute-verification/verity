"""The values a capture map's attention layers compute outside any hooked module, bound on one rank under the map's names
(`commit.binder` `attention`, `verity_vllm.commit.capture_map`).

The pinned vLLM's DeepSeek-V3.2 sparse MLA layer (`DeepseekV32Attention.forward`) runs its norms, RoPE and cache writes in one
Triton call, `fused_norm_rope`, and its absorbed query projections in torch, then hands every value it computed to
`_sparse_indexer_and_attn(positions, q_c, q_nope, q_pe, index_q_fp8, index_k, index_weights_out, kv_c, k_pe, ql_nope, mqa_q,
output)`, which scores the indexer, attends and writes the attention output into `output`, the `o_proj` input.  None of it is a
module of the committer's acquired classes.  On every attention layer this source wraps

- `_sparse_indexer_and_attn`: before it runs, its arguments `ARGUMENTS` (fused_norm_rope's q_c, kv_c_out and k_pe_out, the W_UK_T
  product, fused_q's three outputs) and the layer's cache rows at the step's slots, which fused_norm_rope has just written: the MLA
  cache entry (`mla_rows`) and, on an indexer layer, the indexer cache row's key bytes and f32 scale (`indexer_rows`).  After it returns, the rows the MQA
  path wrote into `output` (the W_UV product) and, on an indexer layer, `topk_indices_buffer[:M, :TOPK]`, the indices that path
  read (shared by every layer, so the committer flushes it at once);
- the layer's `impl.forward_mqa` (its return: the latent attention of the rows it attends) and `impl.forward_mha` (its `output`
  argument after it returns: the prefill rows' attention, word-equal to FlashAttention's return when the prefill has no chunked
  context; with one, or with fused output quantization, the buffer holds something else and is not bound);
- the module's `fused_norm_rope`, whose slot mapping and caches the next `_sparse_indexer_and_attn` of the same q_c reads.

A step on the sparse route sends every row through `forward_mqa`; a dense-MHA step sends its decode rows (the first
`num_decode_tokens`) through `forward_mqa` inside `forward_impl` and its prefill rows through `forward_mha`, so a mixed step binds
both, each over its own rows.  A cache row is read at the step's slot mapping as given: a padding row's negative slot reads some
other entry, which the copy-equality Program then refuses.  Taps the map marks `target_only` are bound on the target only
(`route_source.on_target`).  `capture-map ids --binder attention --names` prints what a step binds.

Each tap that is a Program Call's port is declared to the binding (`ports`, `TAP_PORTS`), which binds it to the row that port
produces whatever label the map commits it under (`verity_vllm.commit.bind`): the DSA indexer's query and weights (`DsaIndexerQ_v1`
ports 0 / 1) and its cache row's key and scale (`DsaIndexerKRow_v1` ports 0 / 1).  Two of them are Values two Calls of the attention
body exchange, `call_boundaries` identities `<layer>.self_attn/<fx node>` (`CALL_TAPS`): the sparse MLA of a layer (SparseMlaSplit_v1 /
SparseMla_v1 per query row), which `forward_mqa` returns and the W_UV bmm reads (the `attn` tap), and the W_UK_T absorb's product the
sparse MLA reads (`q_nope_absorb`).  The source claims those identities (`claims`, one of `call_boundary_source.CLAIMS`), and each
step's tensor of such a tap is held to the rows and width of the identity it binds (a collector error otherwise).  Nothing more is
committed.
"""
from __future__ import annotations

import inspect
import re
import sys
from typing import Any, Mapping, Sequence

import torch

from verity_vllm.acquire.sources import call_boundary_source as CB
from verity_vllm.commit import capture_map as CM
from verity_vllm.commit.bind import Port
from verity_vllm.engine.hooks import Hooks

BINDER = "attention"
ATTEND, MQA, MHA, NORM_ROPE = "_sparse_indexer_and_attn", "forward_mqa", "forward_mha", "fused_norm_rope"
#: ATTEND's arguments -> the map's tap ids
ARGUMENTS = {"q_c": "q_a_layernorm", "kv_c": "kv_a_layernorm", "k_pe": "k_pe_rope", "ql_nope": "q_nope_absorb", "mqa_q": "q_pe_rope",
             "index_q_fp8": "indexer.q_fp8", "index_weights_out": "indexer.weights"}
MLA_ROWS, INDEX_ROWS, INDEX_SCALE = "mla_kv_cache", "indexer.k_cache", "indexer.k_scale"
ATTN_MQA, ATTN_MHA, V_UP, TOPK = "attn", "attn.mha", "v_absorb", "indexer.topk"
TAPS = frozenset(ARGUMENTS.values()) | {MLA_ROWS, INDEX_ROWS, INDEX_SCALE, ATTN_MQA, ATTN_MHA, V_UP, TOPK}
SCALE_BYTES = 4
_BYTES = {"bfloat16": 2, "float32": 4, "int32": 4, "float8_e4m3fn": 1, "uint8": 1}
#: the Definitions of `dsa_indexer_attn`'s per-row sparse MLA, whose output the `attn` tap is
SPARSE_MLA_SPECS = ("SparseMlaSplit_v1", "SparseMla_v1")
#: the taps that are a Call boundary of the attention body -> the Definitions of the Call whose output they are
CALL_TAPS = {ATTN_MQA: frozenset(SPARSE_MLA_SPECS), ARGUMENTS["ql_nope"]: frozenset({"BatchedGemm_v1"})}
#: the taps that are a Program Call's port (`bind.Port`)
TAP_PORTS = {**{t: Port(fams, "out", call_boundary=True) for t, fams in CALL_TAPS.items()},
             ARGUMENTS["index_q_fp8"]: Port(frozenset({"DsaIndexerQ_v1"}), "0"), ARGUMENTS["index_weights_out"]: Port(frozenset({"DsaIndexerQ_v1"}), "1"),
             INDEX_ROWS: Port(frozenset({"DsaIndexerKRow_v1"}), "0"), INDEX_SCALE: Port(frozenset({"DsaIndexerKRow_v1"}), "1")}
_CALL_SITE = re.compile(r"(?:^|\.)layers\.(\d+)\.self_attn/[^/]+$")


def binds(cmap: Mapping[str, Any]) -> bool:
    return any((t.get("commit") or {}).get("binder") == BINDER for t in cmap["taps"])


def call_tap(r: Mapping[str, Any]) -> tuple[str, int] | None:
    """(tap, layer) of a `call_boundaries` identity a `CALL_TAPS` tap binds (`<...>layers.<i>.self_attn/<fx node>`, member `out`, bf16,
    one of the tap's Definitions), else None."""
    if r.get("family") != CB.FAMILY or r.get("output_member") != "out" or r.get("dtype") != "bf16":
        return None
    fam = str(r.get("spec") or "").split("{", 1)[0]
    tap = next((t for t, fams in CALL_TAPS.items() if fam in fams), None)
    m = _CALL_SITE.search(str(r.get("op_path") or ""))
    return (tap, int(m.group(1))) if tap and m else None


def claims(r: Mapping[str, Any]) -> str | None:
    """`call_boundary_source.CLAIMS`: a Call boundary of the attention body a tap binds is this source's."""
    return BINDER if call_tap(r) is not None else None


if claims not in CB.CLAIMS:
    CB.CLAIMS.append(claims)


def mla_rows(cache: Any, slots: Any, width: int, dtype: str) -> Any:
    """The MLA cache entries at `slots`: a [num_blocks, block_size, width] cache of `dtype` holds slot s at [s // block_size,
    s % block_size] (the latent then the roped key); None for another layout."""
    if cache.dim() != 3 or int(cache.shape[-1]) != width or str(cache.dtype) != f"torch.{dtype}":
        return None
    bs = int(cache.shape[1])
    return cache[slots // bs, slots % bs]


def indexer_rows(cache: Any, slots: Any, head: int) -> tuple[Any, Any] | None:
    """The indexer cache rows at `slots`: their `head` e4m3 key bytes [n, head] and their f32 scales [n, 1].  A [num_blocks,
    block_size, head + SCALE_BYTES] byte cache holds, per block, the block's key rows one after another and then their scales
    (fused_norm_rope's `_fp8_quant_and_cache_write`); None for another layout."""
    width = head + SCALE_BYTES
    if cache.dim() != 3 or int(cache.shape[-1]) != width or cache.element_size() != 1:
        return None
    nb, bs, _w = (int(x) for x in cache.shape)
    if cache.stride(2) != 1 or cache.stride(1) != width:
        return None
    flat = cache.view(torch.uint8).view(nb, bs * width)
    blk, off = (slots // bs)[:, None], (slots % bs)[:, None]
    keys = flat[blk, off * head + torch.arange(head, device=flat.device)]
    scale = flat[blk, bs * head + off * SCALE_BYTES + torch.arange(SCALE_BYTES, device=flat.device)]
    return keys, scale.contiguous().view(torch.float32)


def pitched_2d(t: Any) -> bool:
    """Whether the native collector stages `t` as it lies (`native_collect.cpp` `pitched_2d`): contiguous, or equally spaced
    contiguous rows.  The W_UK_T product `ql_nope`, a transposed bmm output [T, H, L] with strides (L, T * L, 1), is neither."""
    dims = [(int(n), int(s)) for n, s in zip(t.shape, t.stride()) if int(n) != 1]
    k, expect = len(dims) - 1, 1
    while k >= 0 and dims[k][1] == expect:
        expect *= dims[k][0]
        k -= 1
    if k < 0:
        return True
    return all(dims[j][1] == dims[j + 1][1] * dims[j + 1][0] for j in range(k)) and dims[k][1] >= expect


class AttentionSource:
    short_name = "attention"

    def __init__(self, com: Any, routes: Any, identities: Sequence[Mapping[str, Any]] = ()) -> None:
        """`identities`: the rank's required-manifest rows; the call boundaries among them a `CALL_TAPS` tap binds hold its tensors."""
        self.c, self.cmap, self.env, self.on_target = com, routes.cmap, routes.env, routes.on_target
        self.layers: dict[int, Any] = dict(routes.layers)
        free = self.cmap.get("free_symbols", {})
        self.taps = {t["id"]: t for t in self.cmap["taps"] if (t.get("commit") or {}).get("binder") == BINDER}
        unknown = sorted(set(self.taps) - TAPS)
        if unknown:
            raise ValueError(f"capture map {self.cmap.get('id')}: the attention source binds no tap {unknown}")
        #: tap id -> layer -> the committed name, for the taps bound here (the target_only ones on the target only)
        self.names = {tid: {i: CM.at_layer(t["commit"]["name"], i) for i in CM.tap_layers(self.cmap, t, self.env)}
                      for tid, t in self.taps.items() if self.on_target or not t.get("target_only")}
        self.width = {tid: CM.dim(t["shape"][-1], self.env, free) for tid, t in self.taps.items()}
        self.homes = []
        for i, m in sorted(self.layers.items()):
            impl = getattr(m, "impl", None)
            if not callable(getattr(m, ATTEND, None)) or not all(callable(getattr(impl, f, None)) for f in (MQA, MHA)):
                raise ValueError(f"capture map {self.cmap.get('id')}: attention layer {i} has no {ATTEND} or impl.{MQA}/{MHA}")
            home = sys.modules.get(inspect.unwrap(type(m).forward).__module__)
            if not callable(getattr(home, NORM_ROPE, None)):
                raise ValueError(f"capture map {self.cmap.get('id')}: the module of layer {i}'s forward has no {NORM_ROPE}")
            if home not in self.homes:
                self.homes.append(home)
        #: (engine step, tap, layer) -> the call boundaries of that layer and step the tap binds
        self.boundaries: dict[tuple[int, str, int], list[Mapping[str, Any]]] = {}
        for r in identities:
            if (tl := call_tap(r)) is not None:
                self.boundaries.setdefault((int(r.get("engine_step", r.get("step"))), *tl), []).append(r)
        self._copies: list[Any] = []
        self._hooks = Hooks()
        self._pending: tuple[Any, Mapping[str, Any]] | None = None
        self._active: int | None = None
        self._mqa_rows = 0
        self.stats: dict[str, Any] = {}
        self.begin_run()

    # -- install / steps ------------------------------------------------------------------------------------------------------
    def install(self) -> None:
        impls: set[int] = set()
        for layer, m in sorted(self.layers.items()):
            self._hooks.wrap(m, ATTEND, self._attend(layer, m))
            if id(m.impl) not in impls:
                impls.add(id(m.impl))
                self._hooks.wrap(m.impl, MQA, self._forward_mqa)
                self._hooks.wrap(m.impl, MHA, self._forward_mha)
        for home in self.homes:
            self._hooks.wrap(home, NORM_ROPE, self._norm_rope)

    def uninstall(self) -> None:
        self._hooks.uninstall()

    def on_step_begin(self) -> None:
        self._pending, self._active = None, None
        self._copies = []

    def begin_run(self) -> None:
        self.stats = {"layers": len(self.layers), "on_target": self.on_target, "calls": 0, "bound": 0, "unpaired": 0, "layout_refused": 0,
                      "mha_unbound": 0, "by_tap": {}, "copied_by_tap": {},
                      "call_boundaries": sum(len(v) for v in self.boundaries.values()), "call_boundaries_held": 0}

    def _hold_boundary(self, tid: str, layer: int, t: Any, job: Any) -> None:
        """The tensor of `CALL_TAPS` tap `tid` of `layer` at the job's step, held to the step's call boundaries of that layer the tap
        binds: every request's rows packed, `width` bf16 words each, the identities' element ranges whole; otherwise a collector error."""
        step = int(job.step)
        ids = self.boundaries.get((step, tid, layer))
        if not ids:
            return
        rows = sum(int(r["rows"]) for r in ids)
        shape = tuple(int(d) for d in t.shape)
        bad = [r for r in ids if list(r.get("element_range") or ()) != [0, int(r["rows"]) * int(r["width"])] or int(r["width"]) != int(ids[0]["width"])]
        if bad or len({r["op_path"] for r in ids}) != 1 or t.dtype != torch.bfloat16 or not shape or shape[0] != rows \
                or t.numel() != rows * int(ids[0]["width"]):
            self.c.errors.append(f"attention: layer {layer} at engine step {step}: tap {tid} is {shape} {t.dtype}; the manifest's "
                                 f"call boundaries there are {[(r['request_id'], r['op_path'], r['rows'], r['width'], r.get('element_range')) for r in ids][:3]}"[:300])
            return
        self.stats["call_boundaries_held"] += len(ids)

    def ports(self) -> dict[str, Port]:
        """`bind.Binder`'s declared labels: each `TAP_PORTS` tap bound here, under the label the map commits it with."""
        return {n.rsplit("/", 1)[1]: TAP_PORTS[tid] for tid, layers in self.names.items() if tid in TAP_PORTS for n in layers.values()}

    def declaration_classes(self) -> list[dict]:
        return []

    def extra_step_bytes(self, envelope: Mapping[str, Any], model: Any) -> dict[str, int]:
        """Every tap this source binds, as bytes per token (a mixed step binds both routes' taps, each over its own rows)."""
        free = self.cmap.get("free_symbols", {})
        per_token = n = 0
        for tid, layers in self.names.items():
            t = self.taps[tid]
            words = 1
            for d in t["shape"][1:]:
                words *= int(CM.dim(d, self.env, free))
            per_token += words * _BYTES[t["dtype"]] * len(layers)
            n += len(layers)
        return {"per_token_bytes": per_token, "per_seq_bytes": 0, "tensors_per_step": n}

    # -- binding ----------------------------------------------------------------------------------------------------------------
    def _bind(self, tid: str, layer: int, t: Any, job: Any) -> None:
        name = self.names.get(tid, {}).get(layer)
        if name is None or t is None:
            return
        if not pitched_2d(t):
            t = t.contiguous()
            self._copies.append(t)  # alive until the step ends: the committer may stage it asynchronously
            self.stats["copied_by_tap"][tid] = self.stats["copied_by_tap"].get(tid, 0) + 1
        self.c._acquire(name, t, job)
        if tid in CALL_TAPS:
            self._hold_boundary(tid, layer, t, job)
        self.stats["bound"] += 1
        self.stats["by_tap"][tid] = self.stats["by_tap"].get(tid, 0) + 1

    def _flush(self, what: str) -> None:
        f = getattr(self.c, "flush_after_plane", None)
        if f is not None:
            f(what)

    def _cache_rows(self, layer: int, q_c: Any, job: Any) -> None:
        pending, self._pending = self._pending, None
        if pending is None or pending[0] is not q_c:
            self.stats["unpaired"] += 1
            return
        kw = pending[1]
        slots = kw.get("slot_mapping")
        if slots is None:
            self.stats["unpaired"] += 1
            return
        s = slots[: int(q_c.shape[0])].long()
        for tids, cache, rows in (((MLA_ROWS,), kw.get("mla_kv_cache"), lambda c: (mla_rows(c, s, self.width[MLA_ROWS], self.taps[MLA_ROWS]["dtype"]),)),
                                  ((INDEX_ROWS, INDEX_SCALE), kw.get("indexer_k_cache"), lambda c: indexer_rows(c, s, self.width[INDEX_ROWS]))):
            if layer not in self.names.get(tids[0], {}) or cache is None:
                continue
            got = rows(cache)
            if got is None or got[0] is None:
                self.stats["layout_refused"] += 1
                continue
            for tid, t in zip(tids, got):
                self._bind(tid, layer, t, job)

    def _attend(self, layer: int, mod: Any) -> Any:
        me = self

        def make(orig: Any) -> Any:
            sig = inspect.signature(orig)

            def attend(*args: Any, **kwargs: Any) -> Any:
                job = getattr(me.c, "_cur", None)
                if job is None:
                    return orig(*args, **kwargs)
                a = sig.bind(*args, **kwargs).arguments
                me.stats["calls"] += 1
                for arg, tid in ARGUMENTS.items():
                    me._bind(tid, layer, a.get(arg), job)
                me._cache_rows(layer, a["q_c"], job)
                me._active, me._mqa_rows = layer, 0
                try:
                    out = orig(*args, **kwargs)
                finally:
                    me._active = None
                if me._mqa_rows:
                    me._bind(V_UP, layer, a["output"][: me._mqa_rows], job)
                    if layer in me.names.get(TOPK, {}):
                        me._bind(TOPK, layer, mod.topk_indices_buffer[: int(a["q_c"].shape[0]), : int(me.width[TOPK])], job)
                        me._flush(TOPK)
                return out
            return attend
        return make

    def _forward_mqa(self, orig: Any) -> Any:
        me = self

        def forward_mqa(*args: Any, **kwargs: Any) -> Any:
            out = orig(*args, **kwargs)
            job = getattr(me.c, "_cur", None)
            if job is not None and me._active is not None:
                me._mqa_rows = int(out[0].shape[0])
                me._bind(ATTN_MQA, me._active, out[0], job)
            return out
        return forward_mqa

    def _forward_mha(self, orig: Any) -> Any:
        me = self
        sig = inspect.signature(orig)

        def forward_mha(*args: Any, **kwargs: Any) -> Any:
            out = orig(*args, **kwargs)
            job = getattr(me.c, "_cur", None)
            if job is not None and me._active is not None:
                a = sig.bind(*args, **kwargs).arguments
                prefill = getattr(a.get("attn_metadata"), "prefill", None)
                if a.get("output_scale") is None and getattr(prefill, "chunked_context", None) is None:
                    me._bind(ATTN_MHA, me._active, a["output"], job)
                else:
                    me.stats["mha_unbound"] += 1
            return out
        return forward_mha

    def _norm_rope(self, orig: Any) -> Any:
        me = self
        sig = inspect.signature(orig)

        def fused_norm_rope(*args: Any, **kwargs: Any) -> Any:
            out = orig(*args, **kwargs)
            if getattr(me.c, "_cur", None) is not None:
                me._pending = (out, sig.bind(*args, **kwargs).arguments)
            return out
        return fused_norm_rope


__all__ = ["AttentionSource", "BINDER", "ARGUMENTS", "TAPS", "CALL_TAPS", "TAP_PORTS", "binds", "call_tap", "mla_rows", "indexer_rows"]
