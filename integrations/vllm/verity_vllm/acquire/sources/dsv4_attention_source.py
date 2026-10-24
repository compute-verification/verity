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
- `_sparse_indexer_and_attn`: on entry, the step's positions at the first layer, and what `_prepare_and_attn` computed beside
  the insert and joined: each compressor's state rows
  at its state slot mapping (`save_partial_states`), the compressed rows its compress kernel wrote this step and, on the scored
  indexer route, the indexer's query and weights (the arguments `index_q`, `index_weights`); after it returns,
  `topk_indices_buffer[:M, :TOPK]` (the short route's fill or the top-k kernels; one buffer for every layer, so flushed at once),
  and the rows FlashMLA wrote into `out` (`o_padded`) over the local heads, split at the decode rows `flash_mla_with_kvcache` attended;

and, in the modules that call them, the globals `fused_q_kv_rmsnorm` (qr and kv), `flash_mla_with_kvcache` (its SWA and extra
indices and lengths over the decode rows), `flash_mla_sparse_fwd` (the rows of its workspace `kv` the prefill's tokens read,
`gathered_rows`), `fused_inv_rope_fp8_quant` (o_fp8 and o_scale) and `fp8_einsum` (its output `z`, the wo_a einsum, after it
returns).

Each value is committed under its tap's name, its identity in the manifest of record (`query.manifest.bind`).  A cache row is
committed as its parts: an fp8_ds_mla row's NoPE bytes, RoPE dims and scale bytes (`MLA_PARTS`, a tap each), which are the insert's
ports 1, 3 and 2 at the SWA slots and the compress kernel's members `compress.0`, `.1` and `.2` at the compressed slots, and a
compressor's state row [kv | score + ape] as its score half, `Dsv4CompressorScore_v1`'s row (the kv half is kv_score's).  A Call of
the attention's body is committed under a label its Definition's port declares (`PORTS`, `ports_of`), and the source claims those
`call_boundaries` identities (`claims`).  The gathered rows are the Program's `FlashMlaGatherRow_v1` rows of one prefill chunk, so a
layer that runs `flash_mla_sparse_fwd` twice at a step fails the step.

Each wrapped call runs on the step's current stream, after `execute_in_parallel` joined the aux streams the indexer and compressors
ran on, and the committer copies on that stream.  A cache is read in its layout: the fp8_ds_mla SWA and compressed caches
([num_blocks, block_size, 584] bytes) and the indexer cache ([.., 132]) hold, per block, every slot's data bytes and then every slot's
scale bytes, regrouped here per row (`split_rows`; the block stride may be padded); a compressor's state cache
([num_blocks, block_size, width] f32) holds a row per slot (`state_rows`).  A cache in another layout is not bound, and the step
then fails by the name.  A value the native collector cannot stage as it lies (`attention_source.pitched_2d`), such as
`fused_inv_rope_fp8_quant`'s o_fp8 (a transposed view [T, G, D] of a [G, T, D] buffer) and o_scale (token stride 1, its columns
TMA-aligned), is bound as a contiguous copy, counted in `stats["copied_by_tap"]`.

A token's compressed row is written when its state slot and its compressed slot are set and its position + 1 is a multiple of
the compressor's ratio (the compress kernels' mask, which `c128_boundary` short-cuts); a step on which a layer's compressor writes
no row commits nothing under its compressed-rows name, since the committer commits no empty tensor, and `empty` records the name
per step.  Every layer attends at one positions tensor (another fails the step); the first layer's is bound under the map's
positions tap and read to the host (`positions`), and the route source recomputes from it the names the step writes no row for
(the map's `row_counts`, `CM.empty_names`), holds the step to the map's names less those, and fails it when `empty` is another set.
The pinned vLLM gives every real token a state slot and exactly its boundary tokens a compressed slot (v1/worker/block_table.py's
`compute_slot_mapping`, v1/attention/backends/mla/compressor_utils.py's `get_compressed_slot_mapping`), so a compressor writes the
positions' count of rows on a prefill or a decode, whatever compressor state an earlier chunk left, and the route source holds
every other compressed-rows name's leading dimension to that count (`CM.rows_differ`).  Each compressor's ratio is the ratio of its compressed tap's row count, or the source refuses to attach.  Taps the map marks
`target_only` are bound on the target only (`route_source.on_target`).  `capture-map ids --binder dsv4_attention --names` prints
what a step binds.
"""
from __future__ import annotations

__models__ = ("deepseek-ai/DeepSeek-V4-Pro",)

import inspect
import sys
from typing import Any, Mapping

import torch

from verity_vllm.acquire.sources import call_boundary_source as CB
from verity_vllm.acquire.sources.attention_source import pitched_2d
from verity_vllm.commit import capture_map as CM
from verity_vllm.engine.hooks import Hooks
from verity_vllm.query.manifest.bind import Port

BINDER = "dsv4_attention"
FORWARD, PROJECTIONS, INSERT, ATTEND = "forward", "_run_parallel_input_projections", "_fused_qnorm_rope_kv_insert", "_sparse_indexer_and_attn"
NORM, DECODE, SPARSE, INV_ROPE, EINSUM, O_PROJ = ("fused_q_kv_rmsnorm", "flash_mla_with_kvcache", "flash_mla_sparse_fwd",
                                                  "fused_inv_rope_fp8_quant", "fp8_einsum", "deep_gemm_fp8_o_proj")
#: each global -> the method of the attention class whose module calls it (INV_ROPE and EINSUM: the module of O_PROJ, a global there)
CALLER = {NORM: "_split_qkv_and_norm", DECODE: "_forward_decode", SPARSE: "_forward_prefill", INV_ROPE: "_o_proj", EINSUM: "_o_proj"}
#: each global -> the roles it binds
GLOBALS = {NORM: ("attn.qr", "attn.kv"), DECODE: ("attn.swa_indices", "attn.swa_lens", "attn.extra_indices", "attn.extra_lens"),
           SPARSE: ("attn.gather",), INV_ROPE: ("attn.o_fp8", "attn.o_scale"), EINSUM: ("attn.o_einsum",)}
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
#: the fp8_ds_mla caches, whose row is tapped as its parts in byte order (`<cache>.<part>`): e4m3 NoPE, bf16 RoPE, scale bytes
MLA, MLA_PARTS = ("attn.swa_rows", "attn.compressed_rows"), ("nope", "rope", "scale")
TOPK_ROLES = ("indexer.topk_short", "indexer.topk")
POSITIONS = "attn.positions"
ROLES = frozenset({*(r for rs in GLOBALS.values() for r in rs), *(c for c in CACHES if c not in MLA),
                   *(f"{c}.{p}" for c in MLA for p in MLA_PARTS), *TOPK_ROLES, POSITIONS, "attn.q", "compressor.kv_score",
                   "indexer.compressor.kv_score", "indexer.q_fp8", "indexer.weights", "attn.out.decode", "attn.out.prefill"})
#: FlashMLA's C128A decode index width is a multiple of this (`_C128A_TOPK_ALIGNMENT`)
C128A_ALIGN = 128
INSERT_DEF, DECODE_DEF, PREFILL_DEF, INV_ROPE_DEF = ("Dsv4QNormRopeKvInsert_v1", "FlashMlaDecode_v1", "FlashMlaPrefill_v1",
                                                     "Dsv4InvRopeQuantHead_v1")
#: the `call_boundaries` identities this source binds (`claims`): Calls of Definitions only DeepSeek-V4's attention body makes
CLAIMED = frozenset({INSERT_DEF, DECODE_DEF, PREFILL_DEF, INV_ROPE_DEF})


def _call(definitions: tuple[str, ...], port: str) -> Port:
    return Port(frozenset(definitions), port, call_boundary=True)


#: `bind.Binder`'s declared labels: each role -> the port of the Call of the attention's body whose value it commits (`ports_of`)
PORTS = {"attn.q": _call((INSERT_DEF,), "0"), "attn.swa_rows.nope": _call((INSERT_DEF,), "1"),
         "attn.swa_rows.scale": _call((INSERT_DEF,), "2"), "attn.swa_rows.rope": _call((INSERT_DEF,), "3"),
         "compressor.kv_score": _call(("CublasSplitKGemmF32Out_v1", "GemmF32Out_v1"), "out"),
         "attn.out.decode": _call((DECODE_DEF,), "out"), "attn.out.prefill": _call((PREFILL_DEF,), "out"),
         "attn.o_fp8": _call((INV_ROPE_DEF,), "0"), "attn.o_scale": _call((INV_ROPE_DEF,), "1"),
         "attn.o_einsum": _call(("ScaledMmFp8BlockGrouped_v1",), "out")}


def binds(cmap: Mapping[str, Any]) -> bool:
    return any((t.get("commit") or {}).get("binder") == BINDER for t in cmap["taps"])


def role_of(tap: Mapping[str, Any]) -> str:
    """What a tap is at its layer: its id without the layer set it is split by (`.csa`, `.hca`)."""
    return tap["id"].removesuffix(".csa").removesuffix(".hca")


def cache_of(role: str) -> str:
    """The cache a role's tap reads: an fp8_ds_mla row part's cache, else the role itself."""
    cache, _, part = role.rpartition(".")
    return cache if cache in MLA and part in MLA_PARTS else role


def ports_of(cmap: Mapping[str, Any]) -> dict[str, Port]:
    """`bind.Binder`'s declared labels for the taps of `cmap` this source binds: each tap's label (its commit name's last part) -> its
    role's port (`PORTS`)."""
    return {t["commit"]["name"].rsplit("/", 1)[1]: PORTS[role_of(t)] for t in cmap["taps"]
            if (t.get("commit") or {}).get("binder") == BINDER and role_of(t) in PORTS}


def claims(r: Mapping[str, Any]) -> str | None:
    """`call_boundary_source.CLAIMS`: a call boundary of the attention's body is this source's."""
    return BINDER if r.get("family") == CB.FAMILY and str(r.get("spec") or "").split("{", 1)[0] in CLAIMED else None


if claims not in CB.CLAIMS:
    CB.CLAIMS.append(claims)


def gathered_rows(kv: Any, indices: Any, lengths: Any = None) -> Any:
    """The rows of FlashMLA's sparse prefill workspace `kv` ([N, 1, D]) its tokens read, each once, in first-read order: the entries
    of `indices` ([T, 1, K]) that are not -1 and lie below their token's `lengths` (when given), token by token."""
    idx = indices.reshape(indices.shape[0], -1).long()
    valid = idx >= 0
    if lengths is not None:
        valid &= torch.arange(idx.shape[1], device=idx.device)[None] < lengths.reshape(-1, 1)
    read = idx[valid]
    rows, inverse = torch.unique(read, return_inverse=True)
    first = torch.full_like(rows, read.numel()).scatter_reduce_(0, inverse, torch.arange(read.numel(), device=read.device), "amin")
    return kv.reshape(-1, kv.shape[-1])[rows[first.argsort()]]


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
        #: (cache, layer) -> the module holding the cache (its `kv_cache` read at the call, its `prefix` the metadata key)
        self.owners: dict[tuple[str, int], Any] = {}
        self.layers: dict[int, Any] = {}
        for t in taps:
            role = role_of(t)
            for i in CM.tap_layers(self.cmap, t, self.env) or []:
                self.taps[(role, i)] = t
                if self.on_target or not t.get("target_only"):
                    self.names.setdefault(role, {})[i] = CM.at_layer(t["commit"]["name"], i)
                if cache_of(role) in CACHES:
                    target = CM.at_layer(t["site"]["target"], i)
                    owner = module(target.removesuffix(".kv_cache"))
                    if not target.endswith(".kv_cache") or not isinstance(getattr(owner, "prefix", None), str):
                        raise ValueError(f"capture map {cid}: {t['id']}'s site {target} is no cache module's kv_cache")
                    self.owners[(cache_of(role), i)] = owner
                if role == "attn.q":
                    self.layers[i] = module(CM.at_layer(t["module"], i))
        #: (cache, layer) -> its row's width (bytes, or a state row's f32 words), and each of its taps as (role, the row's columns
        #: [lo, hi) it commits, viewed as the tap's dtype): an fp8_ds_mla row's parts in byte order, a state row's score half
        self.width: dict[tuple[str, int], int] = {}
        self.cuts: dict[tuple[str, int], list[tuple[str, int, int, Any]]] = {}
        for cache, i in self.owners:
            roles = [f"{cache}.{p}" for p in MLA_PARTS] if cache in MLA else [cache]
            if any((r, i) not in self.taps for r in roles):
                raise ValueError(f"capture map {cid}: layer {i}'s {cache} has no tap for each of {roles}")
            cuts, lo = [], 0
            for r in roles:
                t = self.taps[(r, i)]
                dtype = getattr(torch, t["dtype"])
                n = int(CM.dim(t["shape"][-1], self.env, free)) * (dtype.itemsize if cache in MLA else 1)
                cuts.append((r, lo, lo + n, dtype))
                lo += n
            if cache in STATE:
                cuts, lo = [(r, a + lo, b + lo, d) for r, a, b, d in cuts], 2 * lo
            self.cuts[(cache, i)], self.width[(cache, i)] = cuts, lo
        rc = self.cmap.get("row_counts")
        if rc and [t["id"] for t in taps if role_of(t) == POSITIONS] != [rc.get("positions")]:
            raise ValueError(f"capture map {cid}: its row counts read {rc.get('positions')!r}, not the {POSITIONS} tap this source binds")
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
            count = CM.row_count_of(self.cmap, self.taps[(self.cuts[(role, i)][0][0], i)])
            if rc and (count is None or self.ratio[(role, i)] != int(CM.evaluate(rc["ratios"][count], self.env))):
                raise ValueError(f"capture map {cid}: layer {i}'s {role} compressor has ratio {self.ratio[(role, i)]}, not the ratio "
                                 f"of its row count {count}")
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
        self._positions_at: tuple[int, tuple[int, ...]] | None = None
        self._gathered: set[int] = set()
        #: engine step -> the names a compressor wrote no row for on it
        self.empty: dict[int, list[str]] = {}
        #: engine step -> the positions bound on it (`POSITIONS`), as read when bound
        self.positions: dict[int, tuple[int, ...]] = {}
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
        self._pending, self._decode_rows, self._actives, self._positions_at, self._gathered = {}, {}, {}, None, set()

    def begin_run(self) -> None:
        self.empty, self.positions = {}, {}
        self.stats = {"layers": len(self.layers), "on_target": self.on_target, "attends": 0, "calls": {f: 0 for f in self.homes},
                      "bound": 0, "by_tap": {}, "copied_by_tap": {}, "unpaired": 0, "layout_refused": 0, "unattributed": 0, "empty": 0}

    def declaration_classes(self) -> list[dict]:
        return []

    def ports(self) -> dict[str, Port]:
        return ports_of(self.cmap)

    def extra_step_bytes(self, envelope: Mapping[str, Any], model: Any) -> dict[str, int]:
        """Every tap this source binds, as bytes per token: a bound, since each leading dimension (M, MD, MP, NC4, NC128) is at most
        the step's tokens; the HCA decode index width NX is FlashMLA's at the envelope's max_model_len.  A prefill chunk's gathered
        rows (GR) are at most its tokens' windows, plus per sequence the SW - 1 rows before them and every compressed row at
        max_model_len."""
        max_model_len = int(envelope.get("max_model_len")
                            or max(1, int(envelope.get("max_prompt") or 1)) + max(1, int(envelope.get("max_new") or 1)))
        env = {**self.env, "NX": _cdiv(_cdiv(max_model_len, int(self.env["HCA_R"])), C128A_ALIGN) * C128A_ALIGN}
        per_token = per_seq = n = 0
        for role, layers in self.names.items():
            for i in layers:
                t = self.taps[(role, i)]
                words = 1
                for d in t["shape"][1:]:
                    words *= d if isinstance(d, int) else int(CM.evaluate(d.removeprefix("="), env))
                row = words * torch.empty((), dtype=getattr(torch, t["dtype"])).element_size()
                per_token += row
                if t["shape"][0] == "GR":
                    per_seq += row * (int(self.env["SW"]) - 1 + _cdiv(max_model_len, self.ratio[("attn.compressed_rows", i)]))
                n += 1
        return {"per_token_bytes": per_token, "per_seq_bytes": per_seq, "tensors_per_step": n}

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

    def _bind_rows(self, cache: str, layer: int, rows: Any, job: Any) -> None:
        """A cache's rows as its taps (`cuts`)."""
        if rows is None:
            self.stats["layout_refused"] += 1
            return
        for role, lo, hi, dtype in self.cuts[(cache, layer)]:
            self._bind(role, layer, rows[:, lo:hi].view(dtype), job)

    def _fail(self, msg: str) -> None:
        self.c.errors.append(msg[:200])
        self.c.x03_expected = getattr(self.c, "x03_expected", 0) + 1

    def _flush(self, what: str) -> None:
        f = getattr(self.c, "flush_after_plane", None)
        if f is not None:
            f(what)

    def _bind_positions(self, layer: int, positions: Any, rows: int, job: Any) -> None:
        """The step's positions, which every layer attends at (one tensor, or the step fails): bound at the positions tap's layer,
        and read to the host there (`positions`), from which the route source recomputes the names the step writes no row for."""
        at = (int(positions.data_ptr()), tuple(int(s) for s in positions.stride()))
        if self._positions_at is None:
            self._positions_at = at
        elif at != self._positions_at:
            self._fail(f"dsv4_attention: layer {layer} attended at other positions than the step's first layer")
        if layer in self.names.get(POSITIONS, {}):
            p = positions[:rows]
            self.positions[int(job.step)] = tuple(int(x) for x in p.tolist())
            self._bind(POSITIONS, layer, p, job)

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
        for cache in STATE:
            owner = self.owners.get((cache, layer))
            if owner is None:
                continue
            smd = md.get(owner.prefix)
            got = None if smd is None else state_rows(owner.kv_cache, smd.slot_mapping[:rows].long(), self.width[(cache, layer)],
                                                      smd.block_size)
            self._bind_rows(cache, layer, got, job)
        for cache in COMPRESSED:
            owner = self.owners.get((cache, layer))
            if owner is None:
                continue
            kmd, smd = md.get(owner.prefix), md.get(self.owners[(WRITER[cache], layer)].prefix)
            got = None
            if kmd is not None and smd is not None:
                kv_slots = kmd.slot_mapping
                idx = self._active(smd.slot_mapping, kv_slots, positions, self.ratio[(cache, layer)])
                got = split_rows(owner.kv_cache, kv_slots.long()[idx], self.width[(cache, layer)], SCALE_BYTES[cache])
            self._bind_rows(cache, layer, got, job)

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
                if owner is not None:
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
                me._bind_positions(layer, a["positions"], rows, job)
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
        elif f == SPARSE:
            if layer in self._gathered:
                self._fail(f"dsv4_attention: layer {layer} ran {SPARSE} on two prefill chunks at one step; the Program gathers one")
                return
            self._gathered.add(layer)
            a = sig.bind(*args, **kwargs).arguments
            self._bind("attn.gather", layer, gathered_rows(a["kv"], a["indices"], a.get("topk_length")), job)
        elif f == EINSUM:
            z = args[3] if len(args) > 3 else None
            if not isinstance(z, torch.Tensor):
                self._fail(f"dsv4_attention: layer {layer}'s {EINSUM} was passed no output tensor as its fourth argument")
                return
            self._bind("attn.o_einsum", layer, z, job)


__all__ = ["Dsv4AttentionSource", "BINDER", "CLAIMED", "MLA_PARTS", "PORTS", "POSITIONS", "ROLES", "binds", "cache_of", "claims",
           "gathered_rows", "ports_of", "role_of", "split_rows", "state_rows"]
