"""The values DeepSeek-V4's attention layers compute outside any hooked module, bound on one rank under a capture map's names
(`commit.binder` `dsv4_attention`, `verity_vllm.commit.capture_map`).

The pinned vLLM's `DeepseekV4FlashMLAAttention.forward` (vllm/models/deepseek_v4/attention.py, nvidia/flashmla.py) computes its
projections, norms, cache writes, indexer, compressors, FlashMLA and o-projection in methods and call-time globals, none of them a
module of the committer's acquired classes.  On every attention layer this source wraps

- `forward`, which marks the layer the globals below run for: a call of one at a committer step outside every layer's forward
  fails the step;
- `_run_parallel_input_projections`: its returns 1 and 2, the compressor's and the indexer compressor's kv_score;
- `_fused_qnorm_rope_kv_insert`: its return (the query padded to NHP heads), and the SWA cache rows at the step's SWA slot mapping,
  which it has just written;
- `_sparse_indexer_and_attn`: on entry, what `_prepare_and_attn` computed beside the insert and joined: each compressor's state rows
  at its state slot mapping (`save_partial_states`), the compressed rows its compress kernel wrote this step and, on the scored
  indexer route, the indexer's query and weights (the arguments `index_q`, `index_weights`); after it returns,
  `topk_indices_buffer[:M, :TOPK]` (the short route's fill or the top-k kernels; one buffer for every layer, so flushed at once),
  and the rows FlashMLA wrote into `out` (`o_padded`) over the local heads, split at the decode rows `flash_mla_with_kvcache` attended;

and, in the modules that call them, the globals `fused_q_kv_rmsnorm` (qr and kv), `flash_mla_with_kvcache` (its SWA and extra
indices and lengths over the decode rows), `fused_inv_rope_fp8_quant` (o_fp8 and o_scale) and `fp8_einsum` (its output `z`, the
wo_a einsum, after it returns).

Each wrapped call runs on the step's current stream, after `execute_in_parallel` joined the aux streams the indexer and compressors
ran on, and the committer copies on that stream.  A cache is read in its layout: the fp8_ds_mla SWA and compressed caches
([num_blocks, block_size, 584] bytes) and the indexer cache ([.., 132]) hold, per block, every slot's data bytes and then every slot's
scale bytes, regrouped here per row (`split_rows`; the block stride may be padded); a compressor's state cache
([num_blocks, block_size, width] f32) holds a row per slot (`state_rows`).  A cache in another layout is not bound, and the step
then fails by the name.  A value the native collector cannot stage as it lies (`attention_source.pitched_2d`), such as
`fused_inv_rope_fp8_quant`'s o_fp8 (a transposed view [T, G, D] of a [G, T, D] buffer) and o_scale (token stride 1, its columns
TMA-aligned), is bound as a contiguous copy, counted in `stats["copied_by_tap"]`.  A token's compressed row is written when its state slot and its compressed slot are set and its position
+ 1 is a multiple of the compressor's ratio (the compress kernels' mask, which `c128_boundary` short-cuts); a step on which a
layer's compressor writes no row commits nothing under its compressed-rows name, since the committer commits no empty tensor, and
`empty` records the name per step: the route source refuses the step by that name.  Taps the map marks `target_only` are bound
on the target only (`route_source.on_target`).  `capture-map ids --binder dsv4_attention --names` prints what a step binds.
"""
from __future__ import annotations

import inspect
import sys
from typing import Any, Mapping

import torch

from verity_vllm.acquire.sources.attention_source import pitched_2d
from verity_vllm.commit import capture_map as CM
from verity_vllm.engine.hooks import Hooks

BINDER = "dsv4_attention"
FORWARD, PROJECTIONS, INSERT, ATTEND = "forward", "_run_parallel_input_projections", "_fused_qnorm_rope_kv_insert", "_sparse_indexer_and_attn"
NORM, DECODE, INV_ROPE, EINSUM, O_PROJ = ("fused_q_kv_rmsnorm", "flash_mla_with_kvcache", "fused_inv_rope_fp8_quant", "fp8_einsum",
                                          "deep_gemm_fp8_o_proj")
#: each global -> the method of the attention class whose module calls it (INV_ROPE and EINSUM: the module of O_PROJ, a global there)
CALLER = {NORM: "_split_qkv_and_norm", DECODE: "_forward_decode", INV_ROPE: "_o_proj", EINSUM: "_o_proj"}
#: each global -> the roles it binds
GLOBALS = {NORM: ("attn.qr", "attn.kv"), DECODE: ("attn.swa_indices", "attn.swa_lens", "attn.extra_indices", "attn.extra_lens"),
           INV_ROPE: ("attn.o_fp8", "attn.o_scale"), EINSUM: ("attn.o_einsum",)}
#: flash_mla_with_kvcache's argument for each role it binds
DECODE_ARGUMENTS = {"attn.swa_indices": "indices", "attn.swa_lens": "topk_length", "attn.extra_indices": "extra_indices_in_kvcache",
                    "attn.extra_lens": "extra_topk_length"}
STATE = ("compressor.state", "indexer.compressor.state")
COMPRESSED = ("attn.compressed_rows", "indexer.k_cache")
#: a compressed cache -> the state cache of the compressor that writes it
WRITER = {"attn.compressed_rows": "compressor.state", "indexer.k_cache": "indexer.compressor.state"}
#: a byte cache -> the scale bytes each slot holds after its data bytes
SCALE_BYTES = {"attn.swa_rows": 8, "attn.compressed_rows": 8, "indexer.k_cache": 4}
CACHES = ("attn.swa_rows", *STATE, *COMPRESSED)
TOPK_ROLES = ("indexer.topk_short", "indexer.topk")
ROLES = frozenset({*(r for rs in GLOBALS.values() for r in rs), *CACHES, *TOPK_ROLES, "attn.q", "compressor.kv_score",
                   "indexer.compressor.kv_score", "indexer.q_fp8", "indexer.weights", "attn.out.decode", "attn.out.prefill"})
#: FlashMLA's C128A decode index width is a multiple of this (`_C128A_TOPK_ALIGNMENT`)
C128A_ALIGN = 128


def binds(cmap: Mapping[str, Any]) -> bool:
    return any((t.get("commit") or {}).get("binder") == BINDER for t in cmap["taps"])


def role_of(tap: Mapping[str, Any]) -> str:
    """What a tap is at its layer: its id without the layer set it is split by (`.csa`, `.hca`)."""
    return tap["id"].removesuffix(".csa").removesuffix(".hca")


def split_rows(cache: Any, slots: Any, width: int, scale_bytes: int, block_size: int | None = None) -> Any:
    """The rows at `slots` of a paged byte cache [num_blocks, block_size, width] whose block holds every slot's `width -
    scale_bytes` data bytes, then every slot's `scale_bytes` scale bytes: each row its data bytes, then its scale bytes.  None for
    another layout (or a block size other than `block_size`, when given)."""
    if cache is None or cache.dim() != 3 or int(cache.shape[-1]) != width or cache.element_size() != 1:
        return None
    nb, bs, _w = (int(x) for x in cache.shape)
    if cache.stride(2) != 1 or cache.stride(1) != width or (block_size is not None and int(block_size) != bs):
        return None
    data = width - scale_bytes
    flat = cache.view(torch.uint8).view(nb, bs * width)
    blk, off = (slots // bs)[:, None], (slots % bs)[:, None]
    dev = flat.device
    return torch.cat((flat[blk, off * data + torch.arange(data, device=dev)],
                      flat[blk, bs * data + off * scale_bytes + torch.arange(scale_bytes, device=dev)]), dim=1)


def state_rows(cache: Any, slots: Any, width: int, block_size: int) -> Any:
    """The rows at `slots` of a compressor's state cache [num_blocks, block_size, width] f32 (`save_partial_states` writes slot s at
    [s // block_size, s % block_size]); None for another layout."""
    if (cache is None or cache.dim() != 3 or cache.dtype != torch.float32 or int(cache.shape[-1]) != width
            or int(cache.shape[1]) != int(block_size) or cache.stride(2) != 1):
        return None
    bs = int(cache.shape[1])
    return cache[slots // bs, slots % bs]


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


class Dsv4AttentionSource:
    short_name = "dsv4_attention"

    def __init__(self, com: Any, routes: Any, model: Any) -> None:
        self.c, self.cmap, self.env, self.on_target = com, routes.cmap, routes.env, routes.on_target
        cid = self.cmap.get("id")
        free = self.cmap.get("free_symbols", {})
        taps = [t for t in self.cmap["taps"] if (t.get("commit") or {}).get("binder") == BINDER]
        unknown = sorted({role_of(t) for t in taps} - ROLES)
        if unknown:
            raise ValueError(f"capture map {cid}: the dsv4_attention source binds no {unknown}")
        mods = dict(model.named_modules())

        def module(path: str) -> Any:
            m = mods.get(path)
            if m is None:
                raise ValueError(f"capture map {cid}: the model has no {path}")
            return m

        #: (role, layer) -> tap; role -> layer -> the committed name (the target_only ones on the target only)
        self.taps: dict[tuple[str, int], Mapping[str, Any]] = {}
        self.names: dict[str, dict[int, str]] = {}
        #: (cache role, layer) -> the module holding the cache (its `kv_cache` read at the call, its `prefix` the metadata key)
        self.owners: dict[tuple[str, int], Any] = {}
        self.width: dict[tuple[str, int], int] = {}
        self.layers: dict[int, Any] = {}
        for t in taps:
            role = role_of(t)
            for i in CM.tap_layers(self.cmap, t, self.env) or []:
                self.taps[(role, i)] = t
                if self.on_target or not t.get("target_only"):
                    self.names.setdefault(role, {})[i] = CM.at_layer(t["commit"]["name"], i)
                if role in CACHES:
                    target = CM.at_layer(t["site"]["target"], i)
                    owner = module(target.removesuffix(".kv_cache"))
                    if not target.endswith(".kv_cache") or not isinstance(getattr(owner, "prefix", None), str):
                        raise ValueError(f"capture map {cid}: {t['id']}'s site {target} is no cache module's kv_cache")
                    self.owners[(role, i)] = owner
                    self.width[(role, i)] = int(CM.dim(t["shape"][-1], self.env, free))
                if role == "attn.q":
                    self.layers[i] = module(CM.at_layer(t["module"], i))
        #: (compressed role, layer) -> the writing compressor's ratio
        self.ratio: dict[tuple[str, int], int] = {}
        for (role, i), owner in self.owners.items():
            if role not in COMPRESSED:
                continue
            state = self.owners.get((WRITER[role], i))
            comp = module(state.prefix.removesuffix(".state_cache")) if state is not None else None
            if comp is None or getattr(comp, "k_cache_prefix", None) != owner.prefix:
                raise ValueError(f"capture map {cid}: layer {i}'s {role} has no compressor writing it beside {WRITER[role]}")
            self.ratio[(role, i)] = int(comp.compress_ratio)
        self.heads = int(self.env["NHL"])
        self.topk = int(self.env["TOPK"])
        self.homes: dict[str, list[Any]] = {}
        for i, m in sorted(self.layers.items()):
            missing = [f for f in (FORWARD, PROJECTIONS, INSERT, ATTEND, *set(CALLER.values())) if not callable(getattr(m, f, None))]
            if missing or int(getattr(m, "n_local_heads", self.heads)) != self.heads:
                raise ValueError(f"capture map {cid}: attention layer {i} has no {missing} or not {self.heads} local heads")
            for f, roles in GLOBALS.items():
                if not any(r in self.names for r in roles):
                    continue
                home = sys.modules.get(inspect.unwrap(getattr(type(m), CALLER[f])).__module__)
                if f in (INV_ROPE, EINSUM):
                    o_proj = getattr(home, O_PROJ, None)
                    home = sys.modules.get(getattr(o_proj, "__module__", "")) if o_proj is not None else None
                if not callable(getattr(home, f, None)):
                    raise ValueError(f"capture map {cid}: the module calling layer {i}'s {f} has none")
                if home not in self.homes.setdefault(f, []):
                    self.homes[f].append(home)
        self._hooks = Hooks()
        self._layer: int | None = None
        self._pending: dict[int, Mapping[str, Any]] = {}
        self._decode_rows: dict[int, int] = {}
        self._actives: dict[tuple[int, int, int, int], tuple[tuple[Any, ...], Any]] = {}
        #: engine step -> the names a compressor wrote no row for on it
        self.empty: dict[int, list[str]] = {}
        self.stats: dict[str, Any] = {}
        self.begin_run()

    # -- install / steps ------------------------------------------------------------------------------------------------------
    def install(self) -> None:
        for layer, m in sorted(self.layers.items()):
            self._hooks.wrap(m, FORWARD, self._forward(layer))
            self._hooks.wrap(m, PROJECTIONS, self._projections(layer))
            self._hooks.wrap(m, INSERT, self._insert(layer))
            self._hooks.wrap(m, ATTEND, self._attend(layer, m))
        for f, homes in self.homes.items():
            for home in homes:
                self._hooks.wrap(home, f, self._global(f))

    def uninstall(self) -> None:
        self._hooks.uninstall()

    def on_step_begin(self) -> None:
        self._layer = None
        self._pending, self._decode_rows, self._actives = {}, {}, {}

    def begin_run(self) -> None:
        self.empty = {}
        self.stats = {"layers": len(self.layers), "on_target": self.on_target, "attends": 0, "calls": {f: 0 for f in self.homes},
                      "bound": 0, "by_tap": {}, "copied_by_tap": {}, "unpaired": 0, "layout_refused": 0, "unattributed": 0, "empty": 0}

    def declaration_classes(self) -> list[dict]:
        return []

    def extra_step_bytes(self, envelope: Mapping[str, Any], model: Any) -> dict[str, int]:
        """Every tap this source binds, as bytes per token: a bound, since each leading dimension (M, MD, MP, NC4, NC128) is at most
        the step's tokens; the HCA decode index width NX is FlashMLA's at the envelope's max_model_len."""
        max_model_len = int(envelope.get("max_model_len")
                            or max(1, int(envelope.get("max_prompt") or 1)) + max(1, int(envelope.get("max_new") or 1)))
        env = {**self.env, "NX": _cdiv(_cdiv(max_model_len, int(self.env["HCA_R"])), C128A_ALIGN) * C128A_ALIGN}
        per_token = n = 0
        for role, layers in self.names.items():
            for i in layers:
                t = self.taps[(role, i)]
                words = 1
                for d in t["shape"][1:]:
                    words *= d if isinstance(d, int) else int(CM.evaluate(d.removeprefix("="), env))
                per_token += words * torch.empty((), dtype=getattr(torch, t["dtype"])).element_size()
                n += 1
        return {"per_token_bytes": per_token, "per_seq_bytes": 0, "tensors_per_step": n}

    # -- binding ----------------------------------------------------------------------------------------------------------------
    def _bind(self, role: str, layer: int, t: Any, job: Any) -> bool:
        name = self.names.get(role, {}).get(layer)
        if name is None or t is None:
            return False
        if t.numel() == 0:
            self.empty.setdefault(int(job.step), []).append(name)
            self.stats["empty"] += 1
            return False
        if not pitched_2d(t):
            t = t.contiguous()
            self.stats["copied_by_tap"][role] = self.stats["copied_by_tap"].get(role, 0) + 1
        self.c._acquire(name, t, job)
        self.stats["bound"] += 1
        self.stats["by_tap"][role] = self.stats["by_tap"].get(role, 0) + 1
        return True

    def _bind_rows(self, role: str, layer: int, rows: Any, job: Any) -> None:
        if rows is None:
            self.stats["layout_refused"] += 1
            return
        self._bind(role, layer, rows, job)

    def _fail(self, msg: str) -> None:
        self.c.errors.append(msg[:200])
        self.c.x03_expected = getattr(self.c, "x03_expected", 0) + 1

    def _flush(self, what: str) -> None:
        f = getattr(self.c, "flush_after_plane", None)
        if f is not None:
            f(what)

    def _active(self, state_slots: Any, kv_slots: Any, positions: Any, ratio: int) -> Any:
        """The token indices whose compressed row a compressor of `ratio` writes this step; one host sync per distinct slot mappings
        and ratio a step, the same for every layer that shares them."""
        key = (id(state_slots), id(kv_slots), id(positions), int(ratio))
        hit = self._actives.get(key)
        if hit is not None:
            return hit[1]
        n = min(int(state_slots.shape[0]), int(kv_slots.shape[0]), int(positions.shape[0]))
        sel = (state_slots[:n] >= 0) & (kv_slots[:n] >= 0) & ((positions[:n].long() + 1) % int(ratio) == 0)
        idx = sel.nonzero().flatten()
        self._actives[key] = ((state_slots, kv_slots, positions), idx)
        return idx

    def _caches(self, layer: int, md: Mapping[str, Any], positions: Any, rows: int, job: Any) -> None:
        """The state rows and compressed rows `_prepare_and_attn` wrote for `layer` this step (metadata `md`)."""
        for role in STATE:
            owner = self.owners.get((role, layer))
            if owner is None or layer not in self.names.get(role, {}):
                continue
            smd = md.get(owner.prefix)
            got = None if smd is None else state_rows(owner.kv_cache, smd.slot_mapping[:rows].long(), self.width[(role, layer)],
                                                      smd.block_size)
            self._bind_rows(role, layer, got, job)
        for role in COMPRESSED:
            owner = self.owners.get((role, layer))
            if owner is None or layer not in self.names.get(role, {}):
                continue
            kmd, smd = md.get(owner.prefix), md.get(self.owners[(WRITER[role], layer)].prefix)
            got = None
            if kmd is not None and smd is not None:
                kv_slots = kmd.slot_mapping
                idx = self._active(smd.slot_mapping, kv_slots, positions, self.ratio[(role, layer)])
                got = split_rows(owner.kv_cache, kv_slots.long()[idx], self.width[(role, layer)], SCALE_BYTES[role])
            self._bind_rows(role, layer, got, job)

    def _forward(self, layer: int) -> Any:
        me = self

        def make(orig: Any) -> Any:
            def forward(*args: Any, **kwargs: Any) -> Any:
                prior, me._layer = me._layer, layer
                try:
                    return orig(*args, **kwargs)
                finally:
                    me._layer = prior
            return forward
        return make

    def _projections(self, layer: int) -> Any:
        me = self

        def make(orig: Any) -> Any:
            def projections(*args: Any, **kwargs: Any) -> Any:
                out = orig(*args, **kwargs)
                job = getattr(me.c, "_cur", None)
                if job is not None:
                    me._bind("compressor.kv_score", layer, out[1], job)
                    me._bind("indexer.compressor.kv_score", layer, out[2], job)
                return out
            return projections
        return make

    def _insert(self, layer: int) -> Any:
        me = self

        def make(orig: Any) -> Any:
            sig = inspect.signature(orig)

            def insert(*args: Any, **kwargs: Any) -> Any:
                out = orig(*args, **kwargs)
                job = getattr(me.c, "_cur", None)
                if job is None:
                    return out
                me._bind("attn.q", layer, out, job)
                md = sig.bind(*args, **kwargs).arguments.get("attn_metadata")
                if not isinstance(md, dict):
                    me.stats["unpaired"] += 1
                    return out
                me._pending[layer] = md
                owner = me.owners.get(("attn.swa_rows", layer))
                if owner is not None and layer in me.names.get("attn.swa_rows", {}):
                    smd = md.get(owner.prefix)
                    got = None if smd is None else split_rows(owner.kv_cache, smd.slot_mapping[: int(out.shape[0])].long(),
                                                              me.width[("attn.swa_rows", layer)], SCALE_BYTES["attn.swa_rows"],
                                                              smd.block_size)
                    me._bind_rows("attn.swa_rows", layer, got, job)
                return out
            return insert
        return make

    def _attend(self, layer: int, mod: Any) -> Any:
        me = self

        def make(orig: Any) -> Any:
            sig = inspect.signature(orig)

            def attend(*args: Any, **kwargs: Any) -> Any:
                job = getattr(me.c, "_cur", None)
                if job is None:
                    return orig(*args, **kwargs)
                a = sig.bind(*args, **kwargs).arguments
                me.stats["attends"] += 1
                rows = int(a["hidden_states"].shape[0])
                md = me._pending.pop(layer, None)
                if md is None:
                    me.stats["unpaired"] += 1
                else:
                    me._caches(layer, md, a["positions"], rows, job)
                scored = a.get("index_q") is not None
                if scored:
                    if a.get("index_q_scale") is not None:
                        me.stats["layout_refused"] += 1
                    else:
                        me._bind("indexer.q_fp8", layer, a["index_q"], job)
                    me._bind("indexer.weights", layer, a.get("index_weights"), job)
                me._decode_rows.pop(layer, None)
                out = orig(*args, **kwargs)
                role = TOPK_ROLES[scored]
                if (role, layer) in me.taps:
                    buf = getattr(mod, "topk_indices_buffer", None)
                    if buf is None:
                        me.stats["layout_refused"] += 1
                    elif me._bind(role, layer, buf[:rows, : me.topk], job):
                        me._flush(role)
                decode = me._decode_rows.pop(layer, 0)
                o = a["out"]
                if decode > 0:
                    me._bind("attn.out.decode", layer, o[:decode, : me.heads], job)
                if rows > decode:
                    me._bind("attn.out.prefill", layer, o[decode:rows, : me.heads], job)
                return out
            return attend
        return make

    def _global(self, f: str) -> Any:
        me = self

        def make(orig: Any) -> Any:
            sig = inspect.signature(orig)

            def call(*args: Any, **kwargs: Any) -> Any:
                out = orig(*args, **kwargs)
                job = getattr(me.c, "_cur", None)
                if job is not None:
                    me.on_global(f, sig, args, kwargs, out, job)
                return out
            return call
        return make

    def on_global(self, f: str, sig: inspect.Signature, args: tuple, kwargs: Mapping[str, Any], out: Any, job: Any) -> None:
        """One call of the global `f` at the committer's step `job`, for the layer whose forward is running."""
        layer = self._layer
        if layer is None:
            self.stats["unattributed"] += 1
            self._fail(f"dsv4_attention: {f} called outside every attention layer's forward")
            return
        self.stats["calls"][f] += 1
        if f in (NORM, INV_ROPE):
            for port, role in enumerate(GLOBALS[f]):
                self._bind(role, layer, out[port], job)
        elif f == DECODE:
            a = sig.bind(*args, **kwargs).arguments
            rows = int(a["q"].shape[0])
            self._decode_rows[layer] = rows
            for role, arg in DECODE_ARGUMENTS.items():
                x = a.get(arg)
                if x is not None:
                    x = x[:rows].reshape(rows, -1) if role.endswith("indices") else x[:rows]
                self._bind(role, layer, x, job)
        elif f == EINSUM:
            z = args[3] if len(args) > 3 else None
            if not isinstance(z, torch.Tensor):
                self._fail(f"dsv4_attention: layer {layer}'s {EINSUM} was passed no output tensor as its fourth argument")
                return
            self._bind("attn.o_einsum", layer, z, job)


__all__ = ["Dsv4AttentionSource", "BINDER", "ROLES", "binds", "role_of", "split_rows", "state_rows"]
