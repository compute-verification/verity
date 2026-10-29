"""Committed words by identity: the store sampled replay and the linkage read, and the weights source.

`CommittedStore` resolves a bound identity (request, engine step, op_path, member, element slice) through the binding map and
the manifest's op_path aliases to the one committed entry it names, and reads its words through the opened reader it wraps.
`model_weights_provider` serves registered weight slices from the engine's model parameters.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from verity_vllm.check.oracle_compare import entry_read_name as _entry_read_name, same_name_invocations as _same_name_invocations
from verity_vllm.records.required_manifest import RequiredManifest


#: the binding map's `encoding.dtype` is the collector's torch dtype name; the manifest's short names are accepted too
_ITEMSIZE = {"bf16": 2, "f16": 2, "u16": 2, "i32": 4, "u32": 4, "f32": 4, "i64": 8, "u64": 8,
             "bfloat16": 2, "float16": 2, "int16": 2, "uint16": 2, "float32": 4, "int32": 4, "uint32": 4, "int64": 8, "uint64": 8, "float64": 8,
             # the LIFTED words of the padded MoE construction (manifest_format.LIFTED_DTYPES): Value<17> -> u32, Value<33> -> u64
             "l17": 4, "l33": 8, "w8": 1, "uint8": 1, "int8": 1, "float8_e4m3fn": 1, "float8_e5m2": 1}
_NPDTYPE = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}


def alias_table(manifest: RequiredManifest | dict[str, Any] | None) -> dict[str, tuple[str, int]]:
    """`manifest.op_path_aliases` as ONE table: Program op_path -> (canonical op_path, invocation).  Shared modules
    (vLLM's one RotaryEmbedding instance called by every layer) are addressed by the required manifest at the FIRST layer's path with
    invocation = layer order; the per-layer Program paths are recorded as aliases ('model.layers.7.self_attn.rotary_emb' ->
    ['model.layers.0.self_attn.rotary_emb', 7]).  The population reconciliation (`query_population._key`) and the committed store
    (`CommittedStore.words`) resolve through THIS table and no other -- the two never disagree on which identity a Program path names."""
    if isinstance(manifest, RequiredManifest):
        return {str(k): (str(v[0]), int(v[1])) for k, v in (manifest.op_path_aliases or {}).items()}
    return {str(k): (str(v[0]), int(v[1])) for k, v in ((manifest or {}).get("op_path_aliases") or {}).items()} if manifest is not None else {}


def resolve_op_path(aliases: dict[str, tuple[str, int]], op_path: str) -> tuple[str, int]:
    """(canonical op_path, invocation) of a Program op_path: its alias entry, else itself at invocation 0 -- EXACTLY (a path with no alias
    names invocation 0 and nothing else; `invocation` is 0 for every non-shared module row)."""
    return aliases.get(op_path, (op_path, 0))


def _invocation(e: dict) -> int:
    return int((e.get("identity") or {}).get("invocation", 0) or 0)


class CommittedStore:
    """Committed words of a bound identity: (request, ENGINE step, op_path, member, request-local element slice) via the binding map.

    C1 requires one UNAMBIGUOUS binding, so every entry of a key is KEPT (not 'first wins'): `words` reads the
    ONE entry of the resolved invocation whose element_range covers the slice; two entries of that invocation covering it = an ambiguous
    binding (two buffers under one required identity, the replay could read either) -> (None, 'ambiguous binding: ...'), the VU fails by
    name.

    The Program op_path is resolved through the manifest's `op_path_aliases` (`aliases`, the SAME table the
    population reconciliation uses) to (canonical path, invocation) and the read is of THAT invocation's entry EXACTLY: a shared module's
    layer-N call reads invocation N of the canonical path; a path with no alias reads invocation 0; an invocation with no entry is
    'no committed identity' BY NAME.  Invocations are never collapsed to the lowest (an earlier store did, so the first evaluator that
    read a shared module's output -- the Attention_v3/v2 row, q = rotary_emb/0 -- missed layers 1..N by name on every record: a wrong or
    missing invocation read fails closed, it never reads a neighbour's words)."""

    def __init__(self, bmap: dict[str, Any], committed: Callable[[int, str, int, int], bytes | None],
                 aliases: dict[str, Any] | None = None):
        self.committed = committed
        # accepts the manifest's raw form (alias -> [canonical, invocation]) or the parsed table (alias -> (canonical, invocation))
        self.aliases: dict[str, tuple[str, int]] = {str(k): (str(v[0]), int(v[1])) for k, v in (aliases or {}).items()}
        self.idx: dict[tuple, list[dict]] = {}
        for e in bmap.get("entries") or []:
            ident = e.get("identity") or {}
            if "step" not in ident:
                continue
            key = (str(ident.get("request_id", "r0")), int(ident["step"]), str(ident.get("op_path")), str(ident.get("output_member")))
            self.idx.setdefault(key, []).append(e)
        # a shared module's invocations share ONE member name in the collector's layouts: the reader is asked for the
        # entry's EXACT meta (`<name>#ord<tensor_ordinal>`, oracle_compare.entry_read_name) whenever more than one invocation is bound under
        # that name at the step -- by the bare name every layer read layer 0's words (the OLMoE mixedstop record: layers 1..15 unequal)
        self.name_inv: dict[tuple[int, str], int] = _same_name_invocations(bmap)

    @classmethod
    def of_record(cls, bmap: dict[str, Any], committed: Callable[[int, str, int, int], bytes | None],
                  manifest: RequiredManifest | dict[str, Any] | None) -> "CommittedStore":
        """The store the replay of record uses: the binding map read through the manifest's alias table (None manifest = no aliases: a
        shared module's layer-N path is then missing by name, never silently the lowest invocation)."""
        return cls(bmap, committed, alias_table(manifest))

    def resolve(self, op_path: str) -> tuple[str, int]:
        return resolve_op_path(self.aliases, op_path)

    def entries(self, rid: str, step: int, op_path: str, member: str) -> tuple[list[dict], str, int, list[int]]:
        """The entries of the RESOLVED (canonical path, invocation) -> (entries, canonical, invocation, invocations bound under the key)."""
        canon, inv = self.resolve(op_path)
        es_all = self.idx.get((rid, int(step), canon, member)) or []
        return [x for x in es_all if _invocation(x) == inv], canon, inv, sorted({_invocation(x) for x in es_all})

    def has(self, rid: str, step: int, op_path: str, member: str) -> bool:
        return bool(self.entries(rid, step, op_path, member)[0])

    def words(self, rid: str, step: int, op_path: str, member: str, elo: int, ehi: int) -> tuple[np.ndarray | None, str]:
        es, canon, inv_min, bound = self.entries(rid, step, op_path, member)
        if not es:
            via = f" [alias -> {canon} invocation {inv_min}]" if canon != op_path else (f" [invocation {inv_min}]" if bound else "")
            have = f"; bound invocations of ({canon}/{member}) at that step: {bound}" if bound else ""
            return None, f"no committed identity for ({rid}, step {step}, {op_path}/{member}){via}{have}"

        def _covers(x: dict) -> bool:
            r = (x.get("identity") or {}).get("element_range") or [0, None]
            return r[1] is not None and int(r[0]) <= elo and ehi <= int(r[1])

        cov = [x for x in es if _covers(x)]
        if len(cov) > 1:
            where = ", ".join(f"{x.get('name')}@{x.get('stream_offset')}" for x in cov[:4])
            return None, (f"ambiguous binding: {len(cov)} entries for ({rid}, step {step}, {op_path}/{member}) [invocation {inv_min}] cover "
                          f"[{elo}, {ehi}) -- {where}; C1 requires one unambiguous binding per required identity")
        e = cov[0] if cov else es[0]
        ident = e.get("identity") or {}
        rng = ident.get("element_range") or [0, None]
        if not cov:
            return None, f"element slice [{elo}, {ehi}) outside the bound identity range {rng} of ({rid}, step {step}, {op_path}/{member})"
        dtype = str((e.get("encoding") or {}).get("dtype") or "")
        isz = _ITEMSIZE.get(dtype)
        if isz is None:
            return None, f"unknown dtype {dtype!r} of ({rid}, step {step}, {op_path}/{member})"
        eo = int(e.get("element_offset") or 0)
        b = self.committed(int(step), _entry_read_name(e, self.name_inv.get((int(step), str(e.get("name"))), 1)), (eo + elo) * isz, (eo + ehi) * isz)
        if b is None or len(b) != (ehi - elo) * isz:
            return None, f"committed bytes not retained for ({rid}, step {step}, {op_path}/{member}) [{elo}, {ehi})"
        return np.frombuffer(bytes(b), dtype=_NPDTYPE[isz]).copy(), dtype


def model_weights_provider(model, prefix: str) -> Callable[[str, int, int], np.ndarray | None]:
    """Registered-input reader over the model's parameters AND buffers: Program field `model.model.layers.0.mlp.down_proj.weight` -> the
    implementation's `layers.0.mlp.down_proj.weight` (wrapper prefix stripped), as bf16 words; cached per field.
    The set read here is the set the committed weights root PINS (`register_weights`: `named_parameters()` +
    `named_buffers()`, one qualname per tensor) -- the rotary table `...rotary_emb.cos_sin_cache` is a BUFFER (vLLM registers it
    non-persistent; one shared RotaryEmbedding instance, the Program names it once at layer 0), so a parameters-only reader left every
    RoPE_v1 row unresolved ('registered weight field ... not available to the replay', 300/300 strata on the smollm2 canary).  A
    parameter shadows a buffer of the same qualname (never the case on a torch Module); a field in neither is still None (unresolved)."""
    params = dict(model.named_buffers())
    params.update(dict(model.named_parameters()))
    cache: dict[str, np.ndarray] = {}

    def read(field: str, lo: int, hi: int) -> np.ndarray | None:
        name = field[len(prefix):] if prefix and field.startswith(prefix) else field
        t = params[name] if name in params else (params[field] if field in params else None)     # never `or` on tensors
        if t is None:
            return None
        if name not in cache:
            import torch
            x = t.detach()
            # the words in the parameter's OWN width: bf16 -> u16; the block-scaled FP8 linear's e4m3 `weight` ([N, K], the
            # checkpoint's stored bytes; the cutlass kernel reads it through a `.T` view, the storage is not re-laid) -> u8; its f32
            # `weight_scale_inv` ([N/G, K/G]) -> u32; an int32 table (DeepSeek-V4's hash-routed `ffn.gate.tid2eid`) -> u32.  The replay
            # checks the width against the Program's declared leaf (weight_bits).
            if x.dtype == torch.bfloat16:
                cache[name] = x.contiguous().view(torch.int16).cpu().numpy().view(np.uint16).reshape(-1)
            elif x.dtype in (torch.float32, torch.int32):
                cache[name] = x.contiguous().view(torch.int32).cpu().numpy().view(np.uint32).reshape(-1)
            elif str(x.dtype) in ("torch.float8_e4m3fn", "torch.float8_e4m3fnuz", "torch.uint8", "torch.int8"):
                cache[name] = x.contiguous().view(torch.int8).cpu().numpy().view(np.uint8).reshape(-1)
            else:
                return None
        arr = cache[name]
        if hi > arr.size:
            return None
        return arr[lo:hi]

    return read
