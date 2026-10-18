"""How Match reads a prefix-cache hit: a warm request's attention reads of cached rows another request computed.

A warm request's Program (`build --cached C`) reads rows 0..C-1 of every attention layer's history from its prefix inputs, planes of
C rows each (`program.prefix_layout`): the dense KV prefix's key and value rows (rule KvPrefix[vllm]), or MLA's latent rows and the
DSA indexer's index-K slots and scales (rule MlaPrefix[vllm-dsa]); `prefix_cache.json` beside the derive names the inputs and their
layers in order.  The fold reads the same rows where the engine left them: the instances of the requests that computed the cached
blocks, reached through the slot mapping and block tables vLLM reported.  Both sides canonicalise such a read as
("PFX", layer, kind, lo, hi): `layer` the layer number of the attention reading it, `kind` the row kind (`prefix_layout.KEY` ..
`INDEX_K_SCALE`), and lo / hi leaf offsets into rows 0..C-1 of that layer's plane of that kind (row j at j * its row leaves).
Adjacent reads merge, since the Program reads its C rows as one run and the fold one run per row (MLA: two, the latent's normed part and
its rope part).

On the fold side a read of another request's node is a prefix read only when it is linked: it sits in an operand holding history rows
(`prefix_layout.kv_operands`) of the warm request's attention at a row j < C, where C is the first position the engine scheduled for
the request (the scheduler's num_computed_tokens; 0 unless the record's engine served with prefix caching), and its producer is another
request's instance of the same layer at position j.  Any other read of another request's node stays a cross-request read.  GM-01
(`check_hits`) holds the engine's C and each cached block's producer to the Program's declared `prefix` (`program.prefix_plan`).  A
tensor-parallel rank links its own fold the same way: MLA's latent and index-K rows are replicated, so every rank reads all of them.
"""

from __future__ import annotations

import os

from verity_vllm.check.match.canonical_operands import as_progression
from verity_vllm.program import prefix_layout as PL
from verity_vllm.program.compact import expand, idx_values, rle_mapped

PREFIX_FILE = PL.PREFIX_FILE
FOLD_META = frozenset({"request", "position", "hint"})   # the fold rows' fields a link reads: program_compare.load_oracle must keep them
layer_number = PL.layer_number


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


def served_block_size(steps: dict, rid: str) -> int | None:
    """The KV block size of the engine step that first scheduled `rid` (StepStart.block_size, its first KV group's spec), None when the
    record carries none: the blocks the prefix cache hashed its prompt in."""
    first = min((s for s, st in steps.items() if any(r == rid for r, _ in st["rows"])), default=None)
    bs = steps[first].get("block_size") if first is not None else None
    return int(bs) if bs else None


# ------------------------------------------------------------------------------------------------ the Program's prefix inputs


def load_layout(P, d: str) -> None:
    """Bind a derived Program's prefix inputs (its `prefix_cache.json`) so `PrefixInput` canonicalises reads of them."""
    p = os.path.join(d, PREFIX_FILE)
    try:
        lay = PL.layout_of(d)
    except ValueError as e:
        raise ValueError(f"{p}: {e}") from None
    if lay is None:
        return
    for k, inp in lay["inputs"].items():
        if k not in P.inputs or P.inputs[k][1] != inp.size:
            raise ValueError(f"{p}: user input {k} is not one Input node of {inp.size} values ({len(inp.planes)} planes x C={inp.C} x "
                             f"{inp.row_leaves})")
    P.pfx = dict(lay["inputs"])


class PrefixInput:
    """`program_compare.Prog`'s reads of a warm request's prefix Input nodes (`pfx`: node -> `prefix_layout.Input`)."""

    pfx: dict = {}

    def _pfx_src(self, idx: int, lo: int, hi: int):
        loc = self.pfx[idx].locate(lo, hi)
        if loc is None:
            return ("PFX?", idx, lo, hi)
        layer, kind, base = loc
        return ("PFX", layer, kind, lo - base, hi - base)

    def _pfx_srcP(self, p: tuple, unfold: bool) -> list[tuple]:
        _space, (idx, lo0, hi0), (dI, dL, dH), c = p
        lo, hi = min(lo0, lo0 + (c - 1) * dL), max(hi0, hi0 + (c - 1) * dH)
        loc = self.pfx[idx].locate(lo, hi) if dI == 0 else None
        if loc is not None:
            layer, kind, base = loc
            return [("PFX", layer, kind, (lo0 - base, hi0 - base), (dL, dH), c)]
        return self._per_run(p, unfold)


# ------------------------------------------------------------------------------------------------ the fold's cached rows


def merge_reads(items: list[tuple], compact: bool) -> list[tuple]:
    """Adjacent ("PFX", layer, kind, ..) reads of contiguous leaves as one read (a compact progression of several first split into its
    runs), on both sides alike."""
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
    """`batch_decomp.Projection`'s canon for a warm request (`cached_rows` > 0): its attention's reads of other requests' history rows."""

    cached_rows = 0   # the request's rows the prefix cache served (`cached_rows`), set by its caller

    def canon(self, i: int, unfold: bool = False) -> list[list[tuple]]:
        ops = PL.kv_operands(self.inst[i]["spec"]) if self.cached_rows else None
        if not ops:
            return super().canon(i, unfold)
        out = []
        for ai, a in enumerate(self.inst[i]["args"]):
            if ai in ops:
                mapped = [t for t, _ in self._kv_reads(i, *ops[ai], a, unfold)]
                mapped = merge_reads(mapped, self.compact)
                out.append(rle_mapped(mapped) if self.compact else mapped)
            elif self.compact:
                out.append(rle_mapped([t for p in a for t in self._srcP(p, unfold)]))
            else:
                out.append([self._src(r, unfold) for run in a for r in self._split_peer(run)])
        return out

    def _foreign(self, idx: int) -> bool:
        return idx not in self.inputs and idx not in self.k2i

    def _meta(self, k: int) -> dict | None:
        fold = self._fold
        m = getattr(fold, "_pfx_meta", None)
        if m is None:
            n_in = fold.n_input_nodes
            m = fold._pfx_meta = {o["node"] + n_in: o for o in fold.oracle_rows}
        return m.get(k)

    def _kv_reads(self, i: int, kind: int, rl: int, a: list, unfold: bool, count: bool = True) -> list[tuple[tuple, str | None]]:
        """Canonical sources of one history operand (rows of `kind`, `rl` leaves each), each with the request whose row it links to (None
        for the request's own)."""
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
                    t, who = self._link(kind, layer, run, o, rl, count)
                else:
                    t, who = self._src(run, unfold), None
                out.append((as_progression(t) if self.compact else t, who))
                o += run[3] - run[2]
        return out

    def _link(self, kind: int, layer: int | None, run: tuple, o: int, rl: int, count: bool) -> tuple[tuple, str | None]:
        """("PFX", ..) and its producer for a linked read of another request's row, else a cross-request read (counted)."""
        _space, idx, lo, hi = run
        j = o // rl
        m = self._meta(idx) or {}
        if (hi > lo and (o + hi - lo - 1) // rl == j and j < self.cached_rows and m.get("position") == j and layer is not None
                and layer_number(m.get("hint")) == layer and m.get("request") is not None):
            return ("PFX", layer, kind, o, o + hi - lo), str(m["request"])
        if count:
            self._xreq("prefix-unlinked", idx, lo, hi)
        return ("XREQ", "node", idx, lo, hi), None

    def prefix_producers(self) -> dict[int, set[str]]:
        """Cached row -> the requests whose rows the warm request's attention read for it, over every instance and row kind."""
        rows: dict[int, set[str]] = {}
        for i in range(len(self.inst)):
            ops = PL.kv_operands(self.inst[i]["spec"])
            for ai, a in enumerate(self.inst[i]["args"] if ops else ()):
                if ai not in ops:
                    continue
                kind, rl = ops[ai]
                for t, who in self._kv_reads(i, kind, rl, a, False, count=False):
                    if who is not None:
                        lo, hi = t[3] if self.compact else t[3:5]
                        for j in range(lo // rl, (hi - 1) // rl + 1):
                            rows.setdefault(j, set()).add(who)
        return rows


def check_hits(g: dict, P) -> tuple[list[str], dict | None]:
    """GM-01 (G5) for one request: the engine served the Program's declared prefix rows, in blocks of the plan's size, and every cached
    block's rows were read from the request the Program declares for it.  (problems, the request's `prefix` entry | None for a cold
    request on both sides)."""
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
    served = getattr(P, "block_size", None)
    if bs and served and served != bs:
        why.append(f"{rid}: the engine hashed its prompt in blocks of {served} tokens (StepStart.block_size), the Program's plan in blocks of {bs}: "
                   f"the row declares its attention backend's block size, engine.args.block_size: {served}")
    got = [sorted(set().union(*(rows.get(j, set()) for j in range(b * bs, min((b + 1) * bs, C_obs))))) for b in range(len(want))] if bs else []
    bad = [b for b, (w, h) in enumerate(zip(want, got)) if h != [w]]
    if bs and bad:
        why.append(f"{rid}: cached block {bad[0]} was read from {got[bad[0]]}, the Program declares {want[bad[0]]!r} ({len(bad)} block(s) differ)")
    return why, {"rows": C_obs, "declared_rows": C_decl, "block_size": bs or None, "served_block_size": served, "declared_producers": want,
                 "read_from": got, "linked_rows": len(rows)}
