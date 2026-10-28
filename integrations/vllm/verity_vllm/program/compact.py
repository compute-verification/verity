"""Compact representation: operand references as RUN PROGRESSIONS instead of one tuple per run.

The checker of record holds every instance operand as its list of MAXIMAL CONTIGUOUS RUNS `(space, idx, lo, hi)` (verity.ml.circuits.refs.runs).
On a real fold 91 % of all runs are the K/V gathers of Attention (mprof, llama B8 256/32: 3.52 M runs, 200 per Attention
instance): the SAME window [lo, hi) read from an ARITHMETIC SEQUENCE OF PRODUCER NODES -- one K-projection node per cached position,
consecutive in a prefill (idx delta 1), one per decode step later (idx delta = the step's instance count).  Every per-instance pass of
the checker (canon, the wiring compare, both hashes) is O(runs) and the runs are the resident Program.

COMPACT FORM.  The maximal-run list is run-length-encoded as ARITHMETIC PROGRESSIONS over the integer fields of the run:

    (space, (idx0, lo0, hi0), (dI, dL, dH), count)   ==   [(space, idx0 + k*dI, lo0 + k*dL, hi0 + k*dH) for k in range(count)]

by the greedy left-to-right rule "the next run joins the open progression iff it has the same space and (the progression has one run --
which fixes the delta -- or it equals the last run plus the delta)".  A single run is `(space, (idx, lo, hi), (0, 0, 0), 1)`.  The rule
is a deterministic function of the maximal-run list and the progression list expands back to it, so

    runs(a) == runs(b)   <=>   compact(a) == compact(b)                       (equality is EXACTLY the record's equality)

-- canonical over RUNS, never over descriptors: two descriptors expanding to the same runs compact alike.  A Strided with contiguous
inner rows is one progression built in O(1) from its descriptor; a gather over T consecutive producer nodes is one progression.

The same shape carries CANONICAL sources: `head + (base, delta, count)` with the record's tuple `head + base` per run, e.g.
`("N", (s, jj, lo, hi), (ds, djj, dl, dh), count)` -- the producers of a gather are one per step in decode (ds = 1) or consecutive in
one step in prefill (djj = 1).  `compare.RootView.canon` maps a progression UNIFORMLY when the mapping provably is (same producer; producers
affine in the node index inside one step or across equal-sized steps; a weights window inside one field) and per run otherwise, then
re-normalises with the same greedy rule (`rle_mapped`) so canonical operands compare exactly as the record's do.  `expand` gives the
record's runs back wherever its BYTES are needed (canonical sha256 serialisation, the <= 12 wiring / xreq examples).
"""
from __future__ import annotations

from typing import Callable, Iterable

Prog = tuple  # head... , base: tuple[int, ...], delta: tuple[int, ...], count: int


def _add(base: tuple, delta: tuple, k: int) -> tuple:
    return tuple(b + k * d for b, d in zip(base, delta))


def _sub(a: tuple, b: tuple) -> tuple:
    return tuple(x - y for x, y in zip(a, b))


class _Acc:
    """Greedy accumulator over raw runs: `out` = progressions of the CLOSED maximal runs, `open` = the last maximal run, still growing (a
    contiguous run appended to it merges by the record's `push` rule).  Closing a run applies the RLE join rule against out[-1].  The
    greedy rule's prefix property (the grouping of runs 1..k depends on runs 1..k only) makes the bulk shortcut exact."""

    __slots__ = ("out", "open")

    def __init__(self) -> None:
        self.out: list[Prog] = []
        self.open: tuple | None = None   # (space, idx, lo, hi)

    def _close(self) -> None:
        space, idx, lo, hi = self.open
        v = (idx, lo, hi)
        out = self.out
        if out:
            sp, base, delta, c = out[-1]
            if sp == space:
                if c == 1:
                    out[-1] = (sp, base, _sub(v, base), 2)
                    return
                if v == _add(base, delta, c):
                    out[-1] = (sp, base, delta, c + 1)
                    return
        out.append((space, v, (0, 0, 0), 1))

    def run(self, space: str, idx: int, lo: int, hi: int) -> None:
        """One maximal-run candidate, exactly `refs.runs.push`: merges into the open run when contiguous, else closes it."""
        if hi <= lo:
            return
        o = self.open
        if o is not None:
            if o[0] == space and o[1] == idx and o[3] == lo:
                self.open = (space, idx, o[2], hi)
                return
            self._close()
        self.open = (space, idx, lo, hi)

    def progression(self, space: str, base: tuple, delta: tuple, count: int) -> None:
        """`count` runs `base + k*delta` (a Strided's rows: delta (0, outer, outer); an aliased progression).  Consecutive runs of one
        progression are never contiguous with each other (a contiguous pair would have been one maximal run), so after the first FIVE
        are pushed one by one, the closed tail of `out` is a progression of this delta whose last run is run 3 (two cases: run 2 joined
        a progression of the right delta, or run 2 started a singleton that run 3 joined); runs 4..count-2 then join in O(1) and run
        count-1 becomes the open run."""
        if count <= 0 or base[2] <= base[1]:
            return
        dI, dL, dH = delta
        if dI == 0 and dL == dH == base[2] - base[1]:        # contiguous rows: the record's push merges them all into ONE run
            self.run(space, base[0], base[1], base[1] + count * dL)
            return
        if count <= 6:
            for k in range(count):
                self.run(space, *_add(base, delta, k))
            return
        for k in range(5):
            self.run(space, *_add(base, delta, k))
        sp, b0, d0, c0 = self.out[-1]
        assert sp == space and d0 == delta and _add(b0, d0, c0 - 1) == _add(base, delta, 3), "bulk invariant"
        self.out[-1] = (sp, b0, d0, c0 + (count - 5))         # runs 4 .. count-2 closed into it
        self.open = (space,) + _add(base, delta, count - 1)     # the last run stays open

    def splice(self, progs: list[Prog]) -> None:
        """A finished greedy list pushed into an EMPTY accumulator, in one list copy: by the prefix property pushing its runs one by one
        leaves the list without its last run closed and that run open (its runs are maximal: no two consecutive ones are contiguous)."""
        sp, base, delta, c = progs[-1]
        self.out = progs[:-1] + ([(sp, base, delta if c > 2 else (0,) * len(base), c - 1)] if c > 1 else [])
        self.open = (sp,) + _add(base, delta, c - 1)

    def finish(self) -> list[Prog]:
        if self.open is not None:
            self._close()
            self.open = None
        return self.out


def _walk(acc: _Acc, j, alias: Callable[[int, int], list[Prog]] | None) -> None:
    """The refs JSON of one operand, in reference order (`refs_from_json` / `_decode_refs` shapes)."""
    if "affine" in j:
        space, idx, base, count = j["affine"]
        acc.run(space, idx, base, base + count)
    elif "strided" in j:
        space, idx, base, inner, outer, inner_stride, count = j["strided"]
        if inner_stride == 1 or inner == 1:
            rows = count // inner
            acc.progression(space, (idx, base, base + inner), (0, outer, outer), rows)
        else:   # scattered rows (a transposed / gathered block): per row one progression of its unit runs -- O(count/inner),
            _scattered_strided(acc, space, idx, base, inner, outer, inner_stride, count)   # not the per-leaf O(count)
    elif "concat" in j:
        for x in j["concat"]:
            _walk(acc, x, alias)
    elif "alias" in j:
        k, a = j["alias"]
        if alias is None:
            raise ValueError("alias reference without an alias resolver")
        src = alias(k, a)
        if src and not acc.out and acc.open is None:   # [GLM-5.3] an operand opening with an alias of a ~2048-progression weight read
            acc.splice(src)
            return
        for space, base, delta, c in src:
            acc.progression(space, base, delta, c)
    else:
        for r in j["explicit"]:
            acc.run(r[0], r[1], r[2], r[2] + 1)


def _scattered_strided(acc: _Acc, space: str, idx: int, base: int, inner: int, outer: int, inner_stride: int, count: int) -> None:
    """A Strided whose rows are not contiguous (`inner != 1` and `inner_stride` not 1): row q's references are the unit runs
    `base + q*outer + r*inner_stride`, one progression of delta `(0, inner_stride, inner_stride)` per row.  Exactly the runs the
    per-leaf loop pushed (`_Acc.progression` == pushing its runs one by one), in O(count / inner) instead of O(count) -- the
    GLM-5.3 MLA-absorb transposed-weight read is ~1e6 leaves per operand, one operand per token per layer."""
    rows, last = divmod(count, inner)
    for q in range(rows):
        b = base + q * outer
        acc.progression(space, (idx, b, b + 1), (0, inner_stride, inner_stride), inner)
    if last:
        b = base + rows * outer
        acc.progression(space, (idx, b, b + 1), (0, inner_stride, inner_stride), last)


def from_json(j, alias: Callable[[int, int], list[Prog]] | None = None) -> list[Prog]:
    """Compact operand from its refs JSON.  `alias(k, a)` returns the compact operand `a` of node `k` (v1 descriptors)."""
    acc = _Acc()
    _walk(acc, j, alias)
    return acc.finish()


def _walk_refs(acc: _Acc, r) -> None:
    """`_walk` over the fold's own Refs OBJECTS (verity.ml.circuits.refs: Affine / Strided / Concat / Explicit -- a fold body has no alias
    references, those are a descriptor's), the same greedy rule, without the `to_json()` dict tree in between.  A Concat's run of
    consecutive Affine parts (the real shape: the alias-spliced K/V gathers are one Affine per cached position, per mprof) is
    folded by ONE local loop that inlines `_Acc.run` + `_Acc._close` -- no method call, no tuple arithmetic per run; the accumulator's
    state is taken into locals at entry and written back at exit, so a non-Affine part in the middle simply hands over and resumes."""
    from verity.ml.circuits.refs import Affine, Concat, Explicit, Strided
    tr = type(r)
    if tr is Affine:
        if r.count > 0:
            acc.run(r.space, r.idx, r.base, r.base + r.count)
    elif tr is Concat:
        parts = r.parts
        if type(parts) is not list:   # a PartLog cut's shared prefix: indexed below, so one list for the walk
            parts = list(parts)
        n = len(parts)
        i = 0
        while i < n:
            p = parts[i]
            if type(p) is not Affine:
                _walk_refs(acc, p)
                i += 1
                continue
            # ---- inlined greedy over the maximal stretch of Affine parts starting at i
            out = acc.out
            out_append = out.append
            o = acc.open
            if o is None:
                osp = None
                oi = olo = ohi = 0
            else:
                osp, oi, olo, ohi = o
            while i < n:
                p = parts[i]
                if type(p) is not Affine:
                    break
                i += 1
                cnt = p.count
                if cnt <= 0:
                    continue
                sp, idx, lo = p.space, p.idx, p.base
                hi = lo + cnt
                if osp is not None:
                    if osp == sp and oi == idx and ohi == lo:      # contiguous: the record's push merges into the open run
                        ohi = hi
                        continue
                    # close the open run (oi, olo, ohi) against out[-1] by the RLE join rule
                    if out:
                        lsp, base, delta, c = out[-1]
                        if lsp == osp:
                            if c == 1:
                                out[-1] = (lsp, base, (oi - base[0], olo - base[1], ohi - base[2]), 2)
                                osp, oi, olo, ohi = sp, idx, lo, hi
                                continue
                            if oi == base[0] + c * delta[0] and olo == base[1] + c * delta[1] and ohi == base[2] + c * delta[2]:
                                out[-1] = (lsp, base, delta, c + 1)
                                osp, oi, olo, ohi = sp, idx, lo, hi
                                continue
                    out_append((osp, (oi, olo, ohi), (0, 0, 0), 1))
                osp, oi, olo, ohi = sp, idx, lo, hi
            acc.open = None if osp is None else (osp, oi, olo, ohi)
    elif tr is Strided:
        if r.inner_stride == 1 or r.inner == 1:
            rows = r.count // r.inner
            acc.progression(r.space, (r.idx, r.base, r.base + r.inner), (0, r.outer_stride, r.outer_stride), rows)
        else:
            _scattered_strided(acc, r.space, r.idx, r.base, r.inner, r.outer_stride, r.inner_stride, r.count)
    elif tr is Explicit:
        for space, idx, leaf in r.items:
            acc.run(space, idx, leaf, leaf + 1)
    else:
        raise TypeError(f"from_refs: not a fold Refs object: {tr.__name__}")


def from_refs(r) -> list[Prog]:
    """Compact operand from the fold's own Refs object: == `from_json(r.to_json())` == `from_runs(runs(r))` for every Refs (the tests
    hold all three equal), without materialising the JSON tree."""
    acc = _Acc()
    _walk_refs(acc, r)
    return acc.finish()


def from_runs(runs: Iterable[tuple]) -> list[Prog]:
    """Compact form of a run list (the reference definition: what `from_json` must equal on the expanded runs)."""
    acc = _Acc()
    for space, idx, lo, hi in runs:
        acc.run(space, idx, lo, hi)
    return acc.finish()


def _expand_into(out: list, head: tuple, base: tuple, delta: tuple, c: int) -> None:
    """Append the `c` runs `head + base + k*delta`; one comprehension when a single field moves (the real shapes: a gather's node index /
    step / in-step position, a Strided's window), the generic sum otherwise."""
    moving = [f for f, d in enumerate(delta) if d]
    if len(moving) == 1:
        f = moving[0]
        d = delta[f]
        pre, b, post = head + base[:f], base[f], base[f + 1:]
        out.extend([pre + (v,) + post for v in range(b, b + c * d, d)])
        return
    if len(moving) == 2 and moving[1] == moving[0] + 1:      # a sliding window (lo, hi) moving together
        f = moving[0]
        d0, d1 = delta[f], delta[f + 1]
        pre, b0, b1, post = head + base[:f], base[f], base[f + 1], base[f + 2:]
        out.extend([pre + (b0 + k * d0, b1 + k * d1) + post for k in range(c)])
        return
    out.extend([head + _add(base, delta, k) for k in range(c)])


def expand(progs: Iterable[tuple]) -> list[tuple]:
    """Back to the record's tuples `head + base_k`, in run order (any head)."""
    out: list[tuple] = []
    for t in progs:
        c = t[-1]
        if c == 1:
            out.append(t[:-3] + t[-3])
        else:
            _expand_into(out, t[:-3], t[-3], t[-2], c)
    return out


def n_runs(progs: Iterable[tuple]) -> int:
    return sum(t[-1] for t in progs)


HASH_RLE_RUNS = 1 << 16


def hash_operand(items: list[tuple]) -> list | dict:
    """One canonical operand as `program_compare._canonical_hash` serialises it: the record's per-run tuples, except an operand in
    progression form (`RootView.canon` without unfolding: `rle_mapped`'s normal form) of more than HASH_RLE_RUNS runs, which is serialised
    as `{"rle": its progressions}`.  The normal form is a function of the run list and expands back to it, so equal hashes still mean
    equal run lists; the tag keeps the two serialisations apart.  [GLM-5.3] a strided W_UK_T / W_UV read is ~1e6 runs in ~2048
    progressions, 156 such reads a decode step: expanding them is ~5e9 tuples a hash.  A record-form RootView expands every operand, so the
    two forms' hashes agree on every operand of at most HASH_RLE_RUNS runs."""
    if items and all(is_prog(t) for t in items) and n_runs(items) > HASH_RLE_RUNS:
        return {"rle": [list(t) for t in items]}
    return [list(t) for t in expand_mixed(items)]


# ---- the STORED form: instances.jsonl `operand_refs` items ------------------------------------------------------
# An operand in the file is a list of items in reference order.  Record form (verity-gen/instances/v1): every item a maximal run
# ["space", idx, lo, hi].  Progressions form (v2): node- and parameter-space references as progressions
# ["N"|"P", idx, lo, hi, dI, dL, dH, count] -- the UPPER-CASE space is the marker, one progression per maximal arithmetic run sequence of
# the compact form above with count >= 2; a single run stays ["n"|"p", idx, lo, hi], and any other space is expanded to its runs.  ("P"
# since the GLM-5.3 scattered weight reads: a transposed-weight operand is ~2048 parameter-space progressions over ~1e6 runs, which the
# first v2 form, node-space progressions only, stored run by run.)  `file_items_from_runs` is lossless:
# `runs_from_file_items(file_items_from_runs(R)) == R` for every run list R (the greedy rule is a function of the run list and expands back).
FILE_PROG = "N"
_FILE_MARKERS = {"n": "N", "p": "P"}   # space -> stored progression marker
_FILE_SPACES = {"N": "n", "P": "p"}    # stored progression marker -> space


def file_items_from_refs_json(j) -> list[list]:
    """The stored items of one operand from its refs JSON (`Refs.to_json()` of the fold's own objects, or the descriptor's): compact
    progressions built in O(1) per Affine / Strided part (no `runs()` expansion), node space stored as progressions, every other space
    (parameters "p", anything else) expanded to its runs as the record form has them."""
    return _file_items(from_json(j))


def file_items_from_refs(r) -> list[list]:
    """The stored items of one operand from the fold's own Refs object (== `file_items_from_refs_json(r.to_json())`, the writer's path)."""
    return _file_items(from_refs(r))


def file_items_from_runs(runs_: Iterable[tuple]) -> list[list]:
    """The same from a run list (converting a v1 file: `from_runs` is the reference definition `from_json` equals)."""
    return _file_items(from_runs(runs_))


def _file_items(progs: list[tuple]) -> list[list]:
    out: list[list] = []
    for space, (idx, lo, hi), (dI, dL, dH), c in progs:
        m = _FILE_MARKERS.get(space)
        if m is not None and c > 1:
            out.append([m, idx, lo, hi, dI, dL, dH, c])
        elif c == 1:
            out.append([space, idx, lo, hi])
        else:
            out.extend([[space, idx + k * dI, lo + k * dL, hi + k * dH] for k in range(c)])
    return out


def runs_from_file_items(items: Iterable) -> list[tuple]:
    """Every stored item as the record's run tuples `(space, idx, lo, hi)`: a v1 file passes through, a v2 "N"/"P" progression expands."""
    out: list[tuple] = []
    for it in items:
        if len(it) == 8 and (sp := _FILE_SPACES.get(it[0])) is not None:
            _, idx, lo, hi, dI, dL, dH, c = it
            if c == 1:
                out.append((sp, idx, lo, hi))
            else:
                _expand_into(out, (sp,), (idx, lo, hi), (dI, dL, dH), c)
        else:
            out.append(tuple(it))
    return out


def is_file_prog(it) -> bool:
    return len(it) == 8 and it[0] in _FILE_SPACES


def p_item_indices(r) -> tuple[int, ...]:
    """The parameter indices a stored parameter-space item reads: a "p" run's one, a "P" progression's `count` (O(1) when dI == 0,
    the weight reads' shape).  Any other item reads none."""
    if r[0] == "p":
        return (r[1],)
    if r[0] != "P":
        return ()
    idx, dI, c = r[1], r[4], r[7]
    return (idx,) if dI == 0 or c <= 1 else tuple(range(idx, idx + c * dI, dI))


def p_item_weight_span(r, w_param: int) -> tuple[int, int] | None:
    """[lo, hi) union bounds of the runs of a stored parameter-space item that read parameter `w_param`; None when it reads none."""
    if r[0] == "p":
        return (r[2], r[3]) if r[1] == w_param else None
    if r[0] != "P":
        return None
    _, idx, lo, hi, dI, dL, dH, c = r
    if dI == 0:
        if idx != w_param:
            return None
        last = c - 1
        return min(lo, lo + last * dL), max(hi, hi + last * dH)
    q, rem = divmod(w_param - idx, dI)
    if rem or not 0 <= q < c:
        return None
    return lo + q * dL, hi + q * dH


def expanded_file_args(args: Iterable) -> list:
    """Each operand of a stored instance row as the record form's per-leaf run lists, bit-identical to what `runs()` wrote (v1):
    a record-form operand passes through as the same object, one with any "N"/"P" progression item expands (`runs_from_file_items`).
    The read-side choke point of the Build's `instances.json.gz` (v2, `derive_step._instance_rows`): every reader sees v1 rows."""
    return [a if all(len(it) != 8 for it in a) else [list(t) for t in runs_from_file_items(a)] for a in args]


def first_runs(progs: Iterable[tuple], n: int) -> list[tuple]:
    """The first `n` runs of a compact list (bounded work: examples)."""
    out = []
    for t in progs:
        head, base, delta, c = t[:-3], t[-3], t[-2], t[-1]
        for k in range(min(c, n - len(out))):
            out.append(head + _add(base, delta, k))
        if len(out) >= n:
            break
    return out


def idx_values(t: tuple) -> Iterable[int]:
    """The node / parameter index of every run of a RAW progression `(space, (idx, lo, hi), (dI, dL, dH), count)`."""
    (idx0, _lo, _hi), (dI, _dL, _dH), c = t[1], t[2], t[3]
    if dI == 0:
        return (idx0,) * c if c > 1 else (idx0,)
    return range(idx0, idx0 + c * dI, dI)


def singleton(head: tuple, base: tuple) -> tuple:
    return head + (base, (0,) * len(base), 1)


def rle_mapped(items: list[tuple]) -> list[tuple]:
    """Re-normalise a list of MAPPED progressions `head + (base, delta, count)` (one canonical source per progression of an operand) to the
    greedy RLE of the mapped RUN list: adjacent progressions with the same head merge where the record's per-run canonical list would
    have been one progression.  Equality of the result <=> equality of the mapped run lists (no contiguity merge here: the record never
    merges canonical sources)."""
    out: list[tuple] = []
    for t in items:
        head, base, delta, c = t[:-3], t[-3], t[-2], t[-1]
        if c <= 0:
            continue
        if c == 1:
            delta = (0,) * len(base)
        if out:
            p = out[-1]
            ph, pb, pd, pc = p[:-3], p[-3], p[-2], p[-1]
            if ph == head:
                if pc == 1:
                    d = _sub(base, pb)
                    if c == 1 or delta == d:
                        out[-1] = head + (pb, d, 1 + c)
                        continue
                    out[-1] = head + (pb, d, 2)
                    rest = c - 1
                    out.append(head + (_add(base, delta, 1), delta if rest > 1 else (0,) * len(base), rest))
                    continue
                if base == _add(pb, pd, pc):
                    if c == 1 or delta == pd:
                        out[-1] = head + (pb, pd, pc + c)
                        continue
                    out[-1] = head + (pb, pd, pc + 1)
                    rest = c - 1
                    out.append(head + (_add(base, delta, 1), delta if rest > 1 else (0,) * len(base), rest))
                    continue
        out.append(head + (base, delta, c))
    return out


def rle_runs(mapped_runs: list[tuple], n_base: int) -> list[tuple]:
    """Reference for `rle_mapped`: greedy RLE (no contiguity merge) of a mapped run list whose last `n_base` fields are the integers."""
    out: list[tuple] = []
    for t in mapped_runs:
        head, v = t[:-n_base], t[-n_base:]
        if out:
            p = out[-1]
            ph, pb, pd, pc = p[:-3], p[-3], p[-2], p[-1]
            if ph == head:
                if pc == 1:
                    out[-1] = head + (pb, _sub(v, pb), 2)
                    continue
                if v == _add(pb, pd, pc):
                    out[-1] = head + (pb, pd, pc + 1)
                    continue
        out.append(head + (v, (0,) * n_base, 1))
    return out


def lean_root(desc: dict, registry) -> dict:
    """The root body of a descriptor as compact instances WITHOUT building its Node / Coll / Refs objects (codec.decode_program builds
    them for every root node, then compare.Prog expanded their runs and dropped the IR).  Every non-root definition is decoded by
    decode_program's own reader, `codec.definition_decoder`, refusals included (they are the Definitions' small bodies: their spec
    ids come from the same SpecializedDefinition objects, `descriptor_id`, so the spec strings are the record's).  Returns
    {"nodes": [(k, fid, form, args, out_str, n)], "n_nodes", "params"}.  Root-body semantics mirrored from codec.decode_program:
    v1 `alias` = the compact operand of an earlier node (flattened concat parts and nesting do not change the run sequence); v0
    descriptors have no aliases."""
    from verity.ml.circuits.codec import SCHEMA_V0, definition_decoder, descriptor_id, type_from_json
    get = definition_decoder(desc, registry)
    schema = desc["schema"]
    if schema == SCHEMA_V0:
        callee_id, arg_type = (lambda f: f), type_from_json
    else:
        callees, types = desc["callees"], [type_from_json(t) for t in desc["types"]]
        callee_id, arg_type = callees.__getitem__, types.__getitem__
    b = desc["definitions"][desc["root"]]["body"]
    args_of: list[list[list[Prog]]] = []   # node k -> compact operands (alias targets)
    out_nodes = []
    spec_cache: dict = {}
    out_cache: dict = {}

    def alias(k: int, a: int) -> list[Prog]:
        if k >= len(args_of):
            raise ValueError(f"alias to node {k} before it is defined")
        return args_of[k][a]

    for k, nd in enumerate(b["nodes"]):
        fkey = nd["fn"]
        fid = spec_cache.get(fkey)
        if fid is None:
            fid = spec_cache[fkey] = descriptor_id(get(callee_id(fkey)))
        okey = nd["out"]
        out = out_cache.get(okey) if not isinstance(okey, dict) else None
        if out is None:
            out = str(arg_type(okey))
            if not isinstance(okey, dict):
                out_cache[okey] = out
        args = [from_json(a["refs"], alias if schema != SCHEMA_V0 else None) for a in nd["args"]]
        args_of.append(args)
        out_nodes.append((k, fid, nd["form"], args, out, nd.get("n", 1)))
    return {"nodes": out_nodes, "n_nodes": len(b["nodes"]), "params": [n for n, _ in b["params"]]}


def is_prog(t: tuple) -> bool:
    """A tuple in progression form `head + (base, delta, count)` (vs a record-form per-run tuple)."""
    return len(t) >= 3 and isinstance(t[-3], tuple) and isinstance(t[-2], tuple) and isinstance(t[-1], int)


def expand_mixed(items: Iterable[tuple]) -> list[tuple]:
    """`expand` of a list that may mix progressions with record-form tuples (left as they are)."""
    out: list[tuple] = []
    for t in items:
        if len(t) >= 3 and isinstance(t[-3], tuple) and isinstance(t[-2], tuple) and isinstance(t[-1], int):
            c = t[-1]
            if c == 1:
                out.append(t[:-3] + t[-3])
            else:
                _expand_into(out, t[:-3], t[-3], t[-2], c)
        else:
            out.append(t)
    return out


def n_producers(steps: list, t: tuple) -> list[int]:
    """The producer instance of every run of a canonical `N` source, record form ("N", s, jj, lo, hi) or progression
    ("N", (s, jj, lo, hi), (ds, djj, dL, dH), c); [] for any other head."""
    if t[0] != "N":
        return []
    if isinstance(t[1], tuple):
        (s, jj, _lo, _hi), (ds, djj, _dl, _dh), c = t[1], t[2], t[3]
        if c == 1:
            return [steps[s][jj]]
        return [steps[s + k * ds][jj + k * djj] for k in range(c)]
    return [steps[t[1]][t[2]]]


def n_max_producer(steps: list, t: tuple) -> int | None:
    """max(n_producers) in O(1) when the producer index is monotone in the run (one step, or one producer per step): the endpoints."""
    if t[0] != "N":
        return None
    if isinstance(t[1], tuple):
        (s, jj, _lo, _hi), (ds, djj, _dl, _dh), c = t[1], t[2], t[3]
        if c == 1:
            return steps[s][jj]
        if ds == 0 or djj == 0:
            return max(steps[s][jj], steps[s + (c - 1) * ds][jj + (c - 1) * djj])
        return max(steps[s + k * ds][jj + k * djj] for k in range(c))
    return steps[t[1]][t[2]]


def n_crosses_step(t: tuple, s: int) -> bool:
    """Does a canonical `N` source (either form) read a producer outside step `s`?"""
    if t[0] != "N":
        return False
    if isinstance(t[1], tuple):
        return t[1][0] != s or (t[3] > 1 and t[2][0] != 0)
    return t[1] != s
