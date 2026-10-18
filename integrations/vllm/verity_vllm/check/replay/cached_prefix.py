"""How the sampled replay reads a warm request's cached prefix: the history rows its producers committed.

A warm request's Program (`build --cached C`) reads rows 0..C-1 of every attention layer's history from its prefix inputs, planes of C
rows in the layer order its `prefix_cache.json` names (`program.prefix_layout`): the dense prefix's key and value rows, or MLA's latent
rows and the DSA indexer's index-K slots and scales.  Row j of a plane was computed by the request the prefix plan names for j's block
(`program.prefix_plan`) and committed under that request's identities.  The replay resolves it through the producer's own Program: its
attention of the same layer at a history length past j reads row j as element j * row leaves of its operand holding rows of that kind
(`prefix_layout.kv_operands`), and that operand's run names the committed member.  So the warm request's attention is replayed on the
words its producer committed and its own attention read, never on a value the warm request supplies.
"""
from __future__ import annotations

from typing import Any

from verity_vllm.program import prefix_layout as PL
from verity_vllm.program.compact import expanded_file_args
from verity_vllm.program.kernels.kernel_registry import Declined
from verity_vllm.query.manifest import format as RM

PREFIX_FILE = PL.PREFIX_FILE
layout_of = PL.layout_of
layer_number = PL.layer_number


class CachedPrefix:
    """One warm request's prefix inputs as its producers' committed identities.  `layout`: `prefix_layout.layout_of`'s; `producers`: the
    request label of each cached block; `indices` / `arrivals`: every request's ProgramIndex and arrival step, the producers' among them."""

    def __init__(self, layout: dict[str, Any], producers: list[str], indices: dict[str, Any], arrivals: dict[str, int]) -> None:
        self.C, self.inputs = int(layout["C"]), dict(layout["inputs"])
        if not producers or self.C % len(producers):
            raise ValueError(f"{len(producers)} producing requests for {self.C} cached rows: one per whole block")
        self.producers, self.block = list(producers), self.C // len(producers)
        self.indices, self.arrivals = indices, arrivals
        self._attention: dict[str, dict[tuple[int | None, int], list[tuple[int, int, int]]]] = {}

    def identities(self, a: int, lo: int, hi: int) -> list[tuple[str, int, str, str, int, int]]:
        """Leaves [lo, hi) of prefix input `a`, in order, as committed identities (request, engine step, op_path, member, lo, hi)."""
        inp = self.inputs.get(a)
        loc = inp.locate(lo, hi) if inp is not None else None
        if loc is None:
            shape = f" ({len(inp.planes)} planes x {inp.C * inp.row_leaves})" if inp is not None else ""
            raise Declined(f"prefix input {a} slice [{lo}, {hi}) is not inside one plane of cached rows{shape}")
        layer, kind, base = loc
        rl, o, out = inp.row_leaves, lo - base, []
        while o < hi - base:
            j = o // rl
            e = min(hi - base, (j + 1) * rl)
            out.extend(self._row(layer, kind, rl, j, o - j * rl, e - j * rl))
            o = e
        return out

    def _row(self, layer: int | None, kind: int, rl: int, j: int, a: int, b: int) -> list[tuple[str, int, str, str, int, int]]:
        rid = self.producers[j // self.block]
        pi = self.indices.get(rid)
        if pi is None:
            raise Declined(f"cached row {j} was computed by {rid}, whose Program the replay does not hold")
        if rid not in self._attention:
            self._attention[rid] = history_by_layer(pi.rows)
        att = next(((pi.by_i[i], ai) for t, i, ai in self._attention[rid].get((layer, kind), ()) if t > j), None)   # the shortest history past j
        if att is None:
            raise Declined(f"{rid}'s Program has no attention of layer {layer} reading row kind {kind} past history row {j}: "
                           f"cached row {j} has no producer")
        out, base, want_lo, want_hi = [], 0, j * rl + a, j * rl + b
        for s_kind, x, rlo, rhi in ((str(s[0]), int(s[1]), int(s[2]), int(s[3])) for s in expanded_file_args([att[0]["args"][att[1]]])[0]):
            n = rhi - rlo
            if base < want_hi and want_lo < base + n:
                lo, hi = rlo + max(want_lo, base) - base, rlo + min(want_hi, base + n) - base
                if s_kind != "n":
                    raise Declined(f"{rid}'s attention of layer {layer} reads cached row {j} from its own input {x}: it did not compute it")
                m = pi.member_at(x, lo, hi)
                if m is None:
                    raise Declined(f"{rid}'s row {x} slice [{lo}, {hi}) (layer {layer} row kind {kind} row {j}) is not inside one committed member")
                d, member, mlo, mhi, _dt = m
                w = next(int(mm[1]) for mm in d["members"] if mm[0] == member)
                out.append((rid, int(self.arrivals.get(rid, 0)) + int(d["step"]), str(d["op_path"]), member,
                            int(d["row"]) * w + mlo, int(d["row"]) * w + mhi))
            base += n
        if sum(t[5] - t[4] for t in out) != b - a:
            raise Declined(f"{rid}'s attention of layer {layer} covers {sum(t[5] - t[4] for t in out)} of cached row {j}'s {b - a} leaves")
        return out


def history_by_layer(rows: list[dict]) -> dict[tuple[int | None, int], list[tuple[int, int, int]]]:
    """A Program's attention rows per (layer number, row kind), as (history rows, row, operand) by history length: that operand holds
    history rows 0..T-1 of that kind (`prefix_layout.kv_operands`, `history_rows`)."""
    by: dict[tuple[int | None, int], list[tuple[int, int, int]]] = {}
    for r in rows:
        ops, T = PL.kv_operands(r["spec"]), PL.history_rows(r["spec"])
        if not ops or T is None:
            continue
        layer = layer_number(RM.split_name(r.get("name") or "")[0])
        for ai, (kind, _rl) in ops.items():
            if len(r.get("args") or ()) > ai:
                by.setdefault((layer, kind), []).append((T, int(r["i"]), ai))
    return {k: sorted(v) for k, v in by.items()}
