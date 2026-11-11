"""The int8 attention's K scales: one float32 scale per (layer, KV head), a parameter of the served checkpoint calibrated offline.

Provisional (recipe ``verity-int8-kscale-amax/v0``): which K scale the LUT-softmax attention keeps is the research lane's choice
(a ``benchmarks/ppl`` attention.int8 option, "LUT softmax, K per (layer, KV head)", measured on perplexity).  Until it lands,
for layer l and KV head g:

* amax[l, g] = max |k| over every token and channel of KV head g's K after RoPE (bf16, as the KV cache holds it) in layer l, over
  the calibration windows, served by the same checkpoint with its int8 linears and vLLM's own attention;
* s_k[l, g] = RN32(max(amax[l, g], RN32(1e-30)) / 127) (one float32 division).

The table lives in ``manifests/int8-k-scales.json``, one entry per served checkpoint revision (its scales as ``float.hex`` of the
float32 values, ``scales[l][g]``), so a new choice replaces an entry and nothing else (``tests/engine/int8_kscale_gpu.py`` calibrates
one).  `table` refuses a revision without one: the attention never serves without its K scales.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from verity_vllm import config as C

RECIPE = "verity-int8-kscale-amax/v0"
SCHEMA = "verity-vllm/int8-k-scales/v1"
MANIFEST = C.MANIFESTS / "int8-k-scales.json"
TOP = 127
SCALE_FLOOR = 1e-30
_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def scales(amax: np.ndarray) -> np.ndarray:
    """s_k [L, G] float32 of amax [L, G]: max(amax, RN32(1e-30)) / 127 in float32."""
    a = np.asarray(amax, dtype=np.float32)
    if not np.isfinite(a).all() or (a < 0).any():
        raise ValueError("K amax must be finite and non-negative")
    return (np.maximum(a, np.float32(SCALE_FLOOR)) / np.float32(TOP)).astype(np.float32)


def table(revision: str, path: str | Path | None = None) -> np.ndarray:
    """s_k [L, G] float32 of the checkpoint at `revision`."""
    p = Path(path) if path is not None else MANIFEST
    doc = json.loads(p.read_text()) if p.is_file() else {"entries": []}
    hit = [e for e in doc["entries"] if e["revision"] == revision]
    if len(hit) != 1:
        raise ValueError(f"int8 attention: {len(hit)} K-scale entries for revision {revision} in {p} (int8_k_scales; calibrate one)")
    s = np.array([[float.fromhex(x) for x in row] for row in hit[0]["scales"]], dtype=np.float32)
    if s.ndim != 2 or not (np.isfinite(s).all() and (s > 0).all()):
        raise ValueError(f"{p}: revision {revision}'s scales are not a positive [layers, kv_heads] table")
    return s


def rank_table(s: np.ndarray, kv_heads: int, rank: int = 0, world: int = 1) -> np.ndarray:
    """The columns of s_k [L, G] that tensor-parallel rank `rank` of `world` serves, [L, kv_heads]: vLLM gives each rank G / world
    consecutive KV heads, rank r heads [r * G / world, (r + 1) * G / world).  Refuses any other split."""
    G = s.shape[1]
    if world < 1 or not 0 <= rank < world or G % world or G // world != kv_heads:
        raise ValueError(f"int8 attention: K scales for {G} KV heads, rank {rank} of {world} has {kv_heads}")
    return np.ascontiguousarray(s[:, rank * kv_heads:(rank + 1) * kv_heads])


def entry(revision: str, repo: str, amax: np.ndarray, calibration: dict[str, Any]) -> dict[str, Any]:
    """The table entry of a calibration: its recipe, what it read, amax and s_k as float.hex of float32."""
    hexes = lambda t: [[float(x).hex() for x in row] for row in np.asarray(t, dtype=np.float32)]
    return {"revision": revision, "repo": repo, "recipe": RECIPE, "calibration": calibration, "amax": hexes(amax),
            "scales": hexes(scales(amax))}


def with_entry(doc: dict[str, Any] | None, new: dict[str, Any]) -> dict[str, Any]:
    """`doc` (a table, or None for a new one) with `new` in place of its revision's entry."""
    doc = doc or {"schema": SCHEMA, "entries": []}
    return {**doc, "entries": [e for e in doc["entries"] if e["revision"] != new["revision"]] + [new]}


def layer_index(layer_name: str) -> int:
    """The decoder layer of an attention module's name (``model.layers.<i>.self_attn.attn``)."""
    m = _LAYER.search(layer_name)
    if m is None:
        raise ValueError(f"int8 attention: no layer index in {layer_name!r}")
    return int(m.group(1))

