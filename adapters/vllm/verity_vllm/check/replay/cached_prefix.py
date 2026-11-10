"""How the sampled replay reads a warm request's cached prefix: the key / value rows its producers committed.

A warm request's Program (`build --cached C`) reads rows 0..C-1 of every attention layer's key and value from its `prefix` input,
bf16 [L, 2, C, KVH*D] in the layer order its `prefix_cache.json` names.  Row j of a layer was computed by the request the prefix plan
names for j's block (`program.prefix_plan`) and committed under that request's identities.  The replay resolves it through the
producer's own Program: its attention of the same layer at a history length past j reads row j as element j * KVH*D of its key (kv 0)
or value (kv 1) operand, and that operand's run names the committed member.  So the warm request's attention is replayed on the words
its producer committed and its own attention read, never on a value the warm request supplies.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from verity_vllm.program.compact import expanded_file_args
from verity.ml.definitions.numerics.kernel_registry import Declined
from verity_vllm.query.manifest import format as RM

PREFIX_FILE = "prefix_cache.json"
_LAYER = re.compile(r"layers\.(\d+)")
_KV_STATICS = ("T", "KVH", "D")


def layout_of(program_dir: str | None) -> dict[str, Any] | None:
    """A warm Program's prefix layout ({"input", "C", "row_leaves", "layers"}) from its derive's `prefix_cache.json`; None for any other."""
    p = os.path.join(program_dir, PREFIX_FILE) if program_dir else None
    if not p or not os.path.exists(p):
        return None
    lay = json.load(open(p))
    return {"input": int(lay["input"]), "C": int(lay["C"]), "row_leaves": int(lay["row_leaves"]), "layers": [str(x) for x in lay["layers"]]}


def layer_number(name: str | None) -> int | None:
    m = _LAYER.search(name or "")
    return int(m.group(1)) if m else None


class CachedPrefix:
    """One warm request's `prefix` input as its producers' committed identities.  `producers`: the request label of each cached block,
    `indices` / `arrivals`: every request's ProgramIndex and arrival step, the producers' among them."""

    def __init__(self, layout: dict[str, Any], producers: list[str], indices: dict[str, Any], arrivals: dict[str, int]) -> None:
        self.C, self.rl, self.layers = int(layout["C"]), int(layout["row_leaves"]), [layer_number(n) for n in layout["layers"]]
        if not producers or self.C % len(producers):
            raise ValueError(f"{len(producers)} producing requests for {self.C} cached rows: one per whole block")
        self.producers, self.block = list(producers), self.C // len(producers)
        self.indices, self.arrivals = indices, arrivals
        self._attention: dict[str, dict[int | None, list[tuple[int, int]]]] = {}

    def identities(self, lo: int, hi: int) -> list[tuple[str, int, str, str, int, int]]:
        """Leaves [lo, hi) of the prefix input, in order, as committed identities (request, engine step, op_path, member, lo, hi)."""
        plane = self.C * self.rl
        q = lo // plane
        if hi <= lo or q >= 2 * len(self.layers) or hi > (q + 1) * plane:
            raise Declined(f"prefix slice [{lo}, {hi}) is not inside one layer's cached key or value ({len(self.layers)} layers x 2 x {plane})")
        layer, kv, o = self.layers[q // 2], q % 2, lo - q * plane
        out = []
        while o < hi - q * plane:
            j = o // self.rl
            e = min(hi - q * plane, (j + 1) * self.rl)
            out.extend(self._row(layer, kv, j, o - j * self.rl, e - j * self.rl))
            o = e
        return out

    def _row(self, layer: int | None, kv: int, j: int, a: int, b: int) -> list[tuple[str, int, str, str, int, int]]:
        rid = self.producers[j // self.block]
        pi = self.indices.get(rid)
        if pi is None:
            raise Declined(f"cached row {j} was computed by {rid}, whose Program the replay does not hold")
        if rid not in self._attention:
            self._attention[rid] = attention_by_layer(pi.rows)
        att = next((pi.by_i[i] for t, i in self._attention[rid].get(layer, ()) if t > j), None)   # the shortest history past row j
        if att is None:
            raise Declined(f"{rid}'s Program has no attention of layer {layer} past history row {j}: cached row {j} has no producer")
        out, base, want_lo, want_hi = [], 0, j * self.rl + a, j * self.rl + b
        for kind, x, rlo, rhi in ((str(s[0]), int(s[1]), int(s[2]), int(s[3])) for s in expanded_file_args([att["args"][1 + kv]])[0]):
            n = rhi - rlo
            if base < want_hi and want_lo < base + n:
                lo, hi = rlo + max(want_lo, base) - base, rlo + min(want_hi, base + n) - base
                if kind != "n":
                    raise Declined(f"{rid}'s attention of layer {layer} reads cached row {j} from its own input {x}: it did not compute it")
                m = pi.member_at(x, lo, hi)
                if m is None:
                    raise Declined(f"{rid}'s row {x} slice [{lo}, {hi}) (layer {layer} kv {kv} row {j}) is not inside one committed member")
                d, member, mlo, mhi, _dt = m
                w = next(int(mm[1]) for mm in d["members"] if mm[0] == member)
                out.append((rid, int(self.arrivals.get(rid, 0)) + int(d["step"]), str(d["op_path"]), member,
                            int(d["row"]) * w + mlo, int(d["row"]) * w + mhi))
            base += n
        if sum(t[5] - t[4] for t in out) != b - a:
            raise Declined(f"{rid}'s attention of layer {layer} covers {sum(t[5] - t[4] for t in out)} of cached row {j}'s {b - a} leaves")
        return out


def attention_by_layer(rows: list[dict]) -> dict[int | None, list[tuple[int, int]]]:
    """A Program's attention rows (statics T, KVH, D; operands q, k, v) per layer number, as (T, row) by T: its key / value operands hold
    history rows 0..T-1."""
    by_layer: dict[int | None, list[tuple[int, int]]] = {}
    for r in rows:
        st = RM.spec_statics(r["spec"])
        if all(k in st for k in _KV_STATICS) and len(r.get("args") or ()) >= 3:
            by_layer.setdefault(layer_number(RM.split_name(r.get("name") or "")[0]), []).append((int(st["T"]), int(r["i"])))
    return {k: sorted(v) for k, v in by_layer.items()}
