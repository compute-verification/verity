"""A warm request's cached-prefix inputs as planes of rows, and the attention operands that read each kind of row.

A warm Program (`build --cached C`) reads its cached rows from user inputs, each a stack of planes of C rows (row j of a plane at
leaf j * row_leaves): the dense KV prefix (rule KvPrefix[vllm]) is one input of 2L planes, layer l's key rows then its value rows
(bf16 [L, 2, C, KVH*D]); MLA's (rule MlaPrefix[vllm-dsa]) is three, every MLA layer's latent rows (bf16 [NL, C, L+R]), every
indexer layer's index-K slots (e4m3 [NI, C, D]) and their scales (f32 [NI, C]).  `layout_of` reads them off the derive's
`prefix_cache.json`, and `kv_operands` names, from an attention Definition's statics, the operands that hold history rows of each kind
and their width.  Match (`check.match.prefix_reads`) and the sampled replay (`check.replay.cached_prefix`) read a cached row through
both, the same way on either kind.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache

PREFIX_FILE = "prefix_cache.json"
KEY, VALUE, LATENT, INDEX_K, INDEX_K_SCALE = range(5)
_LAYER = re.compile(r"layers\.(\d+)")
_SPARSE_MLA = ("SparseMla_v", "SparseMlaSplit_v")
_INDEX_LOGITS = ("DsaMqaLogits_v",)


def layer_number(name: str | None) -> int | None:
    m = _LAYER.search(name or "")
    return int(m.group(1)) if m else None


@dataclass(frozen=True)
class Input:
    """One prefix input: `planes` its (layer number, row kind) per plane in input order, each `C` rows of `row_leaves` leaves."""

    index: int
    C: int
    row_leaves: int
    planes: tuple[tuple[int | None, int], ...]

    @property
    def size(self) -> int:
        return len(self.planes) * self.C * self.row_leaves

    def locate(self, lo: int, hi: int) -> tuple[int | None, int, int] | None:
        """(layer, kind, the plane's first leaf) of leaves [lo, hi) inside one plane, None for a slice across planes or past the input."""
        plane = self.C * self.row_leaves
        q = lo // plane
        if hi <= lo or q >= len(self.planes) or hi > (q + 1) * plane:
            return None
        return (*self.planes[q], q * plane)


def inputs_of(lay: dict) -> dict[int, Input]:
    """A `prefix_cache.json` document as its inputs by user-input index.  Raises ValueError on a layer name without a layer number."""
    C = int(lay["C"])
    names = [str(n) for n in lay["layers"]] + [str(n) for n in lay.get("indexer_layers") or ()]
    if (bad := next((n for n in names if layer_number(n) is None), None)) is not None:
        raise ValueError(f"cached-prefix layer {bad!r} names no layer number")
    L = [layer_number(n) for n in lay["layers"]]
    if lay.get("kind") == "mla":
        lat, ik, iks = (int(x) for x in lay["inputs"])
        NI = [layer_number(n) for n in lay["indexer_layers"]]
        return {lat: Input(lat, C, int(lay["row_leaves"]), tuple((l, LATENT) for l in L)),
                ik: Input(ik, C, int(lay["index_d"]), tuple((l, INDEX_K) for l in NI)),
                iks: Input(iks, C, 1, tuple((l, INDEX_K_SCALE) for l in NI))}
    k = int(lay["input"])
    return {k: Input(k, C, int(lay["row_leaves"]), tuple((l, kv) for l in L for kv in (KEY, VALUE)))}


def layout_of(program_dir: str | None) -> dict | None:
    """A warm Program's cached prefix from its derive's `prefix_cache.json`: {"C", "inputs": `inputs_of`}; None for any other Program."""
    p = os.path.join(program_dir, PREFIX_FILE) if program_dir else None
    if not p or not os.path.exists(p):
        return None
    lay = json.load(open(p))
    return {"C": int(lay["C"]), "inputs": inputs_of(lay)}


def statics(spec: str) -> dict[str, int]:
    """The integer statics of a Definition's spec string (`Name_vN{K=V,...}`), top level only (a nested Definition's are its own)."""
    if "{" not in (spec or ""):
        return {}
    body = spec[spec.index("{") + 1: spec.rindex("}")]
    out: dict[str, int] = {}
    depth, start = 0, 0
    for i, ch in enumerate(body + ","):
        if ch in "{[(":
            depth += 1
        elif ch in "}])":
            depth -= 1
        elif ch == "," and depth == 0:
            k, eq, v = body[start:i].partition("=")
            if eq and v.strip().lstrip("-").isdigit():
                out[k.strip()] = int(v)
            start = i + 1
    return out


@lru_cache(maxsize=4096)
def kv_operands(spec: str) -> dict[int, tuple[int, int]]:
    """The operands of an attention Definition that hold history rows, by its statics: operand -> (kind, row leaves).  Dense attention
    (KVH, D; operands q, k, v) its key and value rows; sparse MLA (DR, DV; q_nope, q_pe, latent) its latent rows; the indexer's logits
    (D; q, weights, k, k_scale) its index-K slots and their scales.  {} for any other Definition."""
    name, st = (spec or "").split("{", 1)[0], statics(spec)
    if name.startswith(_SPARSE_MLA) and "DR" in st and "DV" in st:
        return {2: (LATENT, st["DR"] + st["DV"])}
    if name.startswith(_INDEX_LOGITS) and "D" in st:
        return {2: (INDEX_K, st["D"]), 3: (INDEX_K_SCALE, 1)}
    if "KVH" in st and "D" in st:
        return {1: (KEY, st["KVH"] * st["D"]), 2: (VALUE, st["KVH"] * st["D"])}
    return {}


def history_rows(spec: str) -> int | None:
    """The history rows an attention Definition reads (its T, or the logits' N); None without one."""
    st = statics(spec)
    return st.get("N") if (spec or "").startswith(_INDEX_LOGITS) else st.get("T")
