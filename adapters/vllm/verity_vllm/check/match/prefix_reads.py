"""How Match reads a prefix-cache hit: a warm request's attention reads of key / value rows another request computed.

A warm request's Program (`build --cached C`, rule KvPrefix[vllm]) reads rows 0..C-1 of every attention layer's key and value from its
`prefix` input, bf16 [L, 2, C, KVH*D]; `prefix_cache.json` beside the derive names the input and its layers in order.  The fold reads
the same rows where the engine left them: the key / value instances of the requests that computed the cached blocks, reached through
the slot mapping and block tables vLLM reported.  Both sides canonicalise such a read as ("PFX", layer, kv, lo, hi): `layer` the layer
number of the attention reading it, `kv` 0 for the key and 1 for the value, and lo / hi leaf offsets into rows 0..C-1 of that layer's
key or value (row j at j * KVH*D).  Adjacent reads merge, since the Program reads its C rows as one run and the fold one run per row.

On the fold side a read of another request's node is a prefix read only when it is linked: it sits in the key or value operand of the
warm request's attention at a row j < C, where C is the first position the engine scheduled for the request (the scheduler's
num_computed_tokens; 0 unless the record's engine served with prefix caching), and its producer is another request's instance of the same layer at position j.  Any other read of another
request's node stays a cross-request read.  GM-01 (`check_hits`) holds the engine's C and each cached block's producer to the Program's
declared `prefix` (`program.prefix_plan`).
"""

from __future__ import annotations

import json
import os
import re

from verity_vllm.check.match.canonical_operands import as_progression
from verity_vllm.program.compact import expand, idx_values, rle_mapped

PREFIX_FILE = "prefix_cache.json"
KV_OPERANDS = (1, 2)   # an attention Call's key and value operands: (q, k, v[, scale]), the fold's and the Program's alike
FOLD_META = frozenset({"request", "position", "hint"})   # the fold rows' fields a link reads: program_loaders.load_oracle must keep them
_LAYER = re.compile(r"layers\.(\d+)")
_KVH, _D = re.compile(r"[{,]KVH=(\d+)"), re.compile(r"[{,]D=(\d+)")


def layer_number(name: str | None) -> int | None:
    m = _LAYER.search(name or "")
    return int(m.group(1)) if m else None


def serves_prefix_caching(header: dict) -> bool:
    """Whether the record attests an engine serving with vLLM's prefix cache (the RunHeader's engine kwargs; G1 holds every location the
    recorder writes them to the Program's applicability)."""
    prof = header.get("profile") if isinstance(header.get("profile"), dict) else {}
    return any(isinstance(ek, dict) and ek.get("enable_prefix_caching") is True for ek in (prof.get("engine_kwargs"), header.get("engine_kwargs")))


def cached_rows(header: dict, steps: dict, rid: str) -> int:
    """The rows the prefix cache served `rid`: under prefix caching the first position the engine scheduled for it (StepStart.rows, the
    scheduler's num_computed_tokens at its first step), else 0."""
    if not serves_prefix_caching(header):
        return 0
    return min((p for st in steps.values() for r, p in st["rows"] if r == rid), default=0)


# ------------------------------------------------------------------------------------------------ the Program's prefix input


def load_layout(P, d: str) -> None:
    """Bind a derived Program's prefix input (its `prefix_cache.json`) so `PrefixInput` canonicalises reads of it."""
    p = os.path.join(d, PREFIX_FILE)
    if not os.path.exists(p):
        return
    lay = json.load(open(p))
    k, C, rl, names = int(lay["input"]), int(lay["C"]), int(lay["row_leaves"]), list(lay["layers"])
    nums = [layer_number(n) for n in names]
    if None in nums:
        raise ValueError(f"{p}: layer {names[nums.index(None)]!r} names no layer number")
    if k not in P.inputs or P.inputs[k][1] != 2 * len(names) * C * rl:
        raise ValueError(f"{p}: user input {k} is not one Input node of {2 * len(names) * C * rl} values (L={len(names)}, C={C}, KVH*D={rl})")
    P.pfx = {k: (C, rl, nums)}


class PrefixInput:
    """`instance_sequence.RootView`'s reads of a warm request's `prefix` Input node (`pfx`: node -> (C, KVH*D, layer numbers))."""

    pfx: dict = {}

    def _pfx_src(self, idx: int, lo: int, hi: int):
        C, rl, layers = self.pfx[idx]
        plane = C * rl
        q = lo // plane
        if q >= 2 * len(layers) or hi > (q + 1) * plane:
            return ("PFX?", idx, lo, hi)
        return ("PFX", layers[q // 2], q % 2, lo - q * plane, hi - q * plane)

    def _pfx_srcP(self, p: tuple, unfold: bool) -> list[tuple]:
        _space, (idx, lo0, hi0), (dI, dL, dH), c = p
        C, rl, layers = self.pfx[idx]
        plane = C * rl
        lo, hi = min(lo0, lo0 + (c - 1) * dL), max(hi0, hi0 + (c - 1) * dH)
        q = lo // plane
        if dI == 0 and q < 2 * len(layers) and hi <= (q + 1) * plane:
            return [("PFX", layers[q // 2], q % 2, (lo0 - q * plane, hi0 - q * plane), (dL, dH), c)]
        return self._per_run(p, unfold)


# ------------------------------------------------------------------------------------------------ the fold's cached rows


def merge_reads(items: list[tuple], compact: bool) -> list[tuple]:
    """Adjacent ("PFX", layer, kv, ..) reads of contiguous leaves as one read (a compact progression of several first split into its runs),
    on both sides alike."""
    if compact:
        items = [r for t in items for r in ([as_progression(x) for x in expand([t])] if t[0] == "PFX" and t[-1] > 1 else [t])]
    out: list[tuple] = []
    for t in items:
        if out and t[0] == "PFX" and out[-1][0] == "PFX" and out[-1][1:3] == t[1:3]:
            q = out[-1]
            if compact and q[-1] == t[-1] == 1 and q[3][1] == t[3][0]:
                out[-1] = q[:3] + ((q[3][0], t[3][1]), (0, 0), 1)
                continue
            if not compact and q[4] == t[3]:
                out[-1] = q[:4] + (t[4],)
                continue
        out.append(t)
    return out


class CachedKv:
    """`batch_decomp.Projection`'s canon for a warm request (`cached_rows` > 0): its attention's reads of other requests' key / value."""

    cached_rows = 0   # the request's rows the prefix cache served (`cached_rows`), set by its caller

    def canon(self, i: int, unfold: bool = False) -> list[list[tuple]]:
        rl = self._kv_row_leaves(i) if self.cached_rows else None
        if rl is None:
            return super().canon(i, unfold)
        out = []
        for ai, a in enumerate(self.inst[i]["args"]):
            if ai in KV_OPERANDS:
                mapped = [t for t, _ in self._kv_reads(i, ai - 1, a, rl, unfold)]
                mapped = merge_reads(mapped, self.compact)
                out.append(rle_mapped(mapped) if self.compact else mapped)
            elif self.compact:
                out.append(rle_mapped([t for p in a for t in self._srcP(p, unfold)]))
            else:
                out.append([self._src(r, unfold) for run in a for r in self._split_peer(run)])
        return out

    def _kv_row_leaves(self, i: int) -> int | None:
        """KVH*D of an attention instance (its spec's statics), None for any other instance."""
        sp = self.inst[i]["spec"]
        kvh, d = _KVH.search(sp), _D.search(sp)
        return int(kvh.group(1)) * int(d.group(1)) if kvh and d else None

    def _foreign(self, idx: int) -> bool:
        return idx not in self.inputs and idx not in self.k2i

    def _meta(self, k: int) -> dict | None:
        fold = self._fold
        m = getattr(fold, "_pfx_meta", None)
        if m is None:
            n_in = fold.n_input_nodes
            m = fold._pfx_meta = {o["node"] + n_in: o for o in fold.oracle_rows}
        return m.get(k)

    def _kv_reads(self, i: int, kv: int, a: list, rl: int, unfold: bool, count: bool = True) -> list[tuple[tuple, str | None]]:
        """Canonical sources of one key / value operand, each with the request whose row it links to (None for the request's own)."""
        layer = layer_number((self._meta(self.inst[i]["k"]) or {}).get("hint"))
        out, o = [], 0
        for p in a:
            if self.compact and not (p[0] == "n" and any(self._foreign(x) for x in idx_values(p))):
                out.extend((t, None) for t in self._srcP(p, unfold))
                (_i, lo0, hi0), (_dI, dL, dH), c = p[1], p[2], p[3]
                o += c * (hi0 - lo0) + (dH - dL) * c * (c - 1) // 2
                continue
            for run in (expand([p]) if self.compact else [p]):
                if run[0] == "n" and self._foreign(run[1]):
                    t, who = self._link(kv, layer, run, o, rl, count)
                else:
                    t, who = self._src(run, unfold), None
                out.append((as_progression(t) if self.compact else t, who))
                o += run[3] - run[2]
        return out

    def _link(self, kv: int, layer: int | None, run: tuple, o: int, rl: int, count: bool) -> tuple[tuple, str | None]:
        """("PFX", ..) and its producer for a linked read of another request's row, else a cross-request read (counted)."""
        _space, idx, lo, hi = run
        j = o // rl
        m = self._meta(idx) or {}
        if (hi > lo and (o + hi - lo - 1) // rl == j and j < self.cached_rows and m.get("position") == j and layer is not None
                and layer_number(m.get("hint")) == layer and m.get("request") is not None):
            return ("PFX", layer, kv, o, o + hi - lo), str(m["request"])
        if count:
            self._xreq("prefix-unlinked", idx, lo, hi)
        return ("XREQ", "node", idx, lo, hi), None

    def prefix_producers(self) -> dict[int, set[str]]:
        """Cached row -> the requests whose rows the warm request's attention read for it, over every instance."""
        rows: dict[int, set[str]] = {}
        for i in range(len(self.inst)):
            rl = self._kv_row_leaves(i)
            for ai, a in enumerate(self.inst[i]["args"] if rl else ()):
                if ai not in KV_OPERANDS:
                    continue
                for t, who in self._kv_reads(i, ai - 1, a, rl, False, count=False):
                    if who is not None:
                        lo, hi = t[3] if self.compact else t[3:5]
                        for j in range(lo // rl, (hi - 1) // rl + 1):
                            rows.setdefault(j, set()).add(who)
        return rows


def check_hits(g: dict, P) -> tuple[list[str], dict | None]:
    """GM-01 (G5) for one request: the engine served the Program's declared prefix rows, and every cached block's rows were read from the
    request the Program declares for it.  (problems, the request's `prefix` entry | None for a cold request on both sides)."""
    decl = g.get("prefix") or {}
    C_decl, C_obs = int(decl.get("rows") or 0), int(getattr(P, "cached_rows", 0) or 0)
    if not C_decl and not C_obs:
        return [], None
    rid, why = g["request_id"], []
    if C_obs != C_decl:
        why.append(f"{rid}: the engine served {C_obs} prompt rows from the prefix cache (its first scheduled position), the Program declares {C_decl}")
    rows = P.prefix_producers() if C_obs else {}
    if C_obs and sorted(rows) != list(range(C_obs)):
        why.append(f"{rid}: {C_obs - len(rows)} of its {C_obs} cached rows are read from no other request's row of the same layer and position")
    want = list(decl.get("producers") or [])
    bs = C_decl // len(want) if want else 0   # the plan caches whole blocks: one declared producer per block
    got = [sorted(set().union(*(rows.get(j, set()) for j in range(b * bs, min((b + 1) * bs, C_obs))))) for b in range(len(want))] if bs else []
    bad = [b for b, (w, h) in enumerate(zip(want, got)) if h != [w]]
    if bs and bad:
        why.append(f"{rid}: cached block {bad[0]} was read from {got[bad[0]]}, the Program declares {want[bad[0]]!r} ({len(bad)} block(s) differ)")
    return why, {"rows": C_obs, "declared_rows": C_decl, "block_size": bs or None, "declared_producers": want, "read_from": got,
                 "linked_rows": len(rows)}
