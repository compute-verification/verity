"""MATCH_IMPL=fast -- compute-once / index-once for the Match stage (GM-01 `global_match.check`, X-09 `batch_decomp.decompose`).

[R17 mprof, BRIEF s5 / MATCH-PROFILE.md] The checker of record traverses immutable per-request structure many times per invocation:
`Prog.canon(i)` (every operand source of an instance) >= 6x per instance (G3 loop, compare_steps wiring, _canonical_hash, dag_hashes,
G8 dependency order, live_cone; then everything again in the alternate-criterion run, and again inside X-09), `compare_steps(A, P)` 2x
per request, `dag_hashes(A)` once per REQUEST (not per component) and `dag_hashes(A) + dag_hashes(P)` over the FULL programs once per
permuted step in G8 (<= 32 more), `_canonical_hash(A)` once per request per pass.  Every one of these is a pure function of Prog
objects that are read-only after construction (compare.Prog: `inst`/`steps`/`inputs`/... are built in __init__ and never written
again; global_match renames the fold ROWS' request ids in place, which no canonical form reads).

This module memoises exactly those functions, by object identity of the Progs (objects are pinned in the memo so an id is never
reused while cached), and lets `check()` reuse each request's Projection between the two criteria.  NOTHING is skipped, no check is
weakened, no fingerprint replaces an exact comparison: the memoised function runs once and its (identical) result is reused.  The
one side effect a memoised function has -- `Projection._xreq` counting cross-request reads during `canon` -- is REPLAYED on every
cache hit of `canon`, so `P.xreq` / `P.xreq_examples` evolve exactly as in the checker of record.  (`compare_steps` / `dag_hashes`
hits skip their inner canon calls; after those calls the checker reads `P.xreq` only as a boolean, which the G3 loop has already
established -- see the call sites in global_match.check.)

Enabled by the environment variable MATCH_IMPL=fast (`enabled()`): `global_match.check` then runs over `ops()` (a `program_ops.Ops`,
passed explicitly to GM-01 and to the X-09 run inside it) -- the fold and components loaded as `MemoProg` (canon memoised per instance),
Projections as `_ProjectionFast`, and memoising wrappers that call the checker of record's `compare_steps` / `_canonical_hash` /
`dag_hashes` / `live_cone`.  Nothing is rebound: the record's functions stay what they are.  Without the flag nothing here runs.

Differential gate: mfast's `verity_vllm/harness/match_diff.py` (old vs fast, identical verdicts / per-check outcomes / populations /
mismatch locations) -- every FIX_READY of this module names the run it passed.
"""
from __future__ import annotations

import collections
import functools
import json
import os
from collections.abc import Callable

from verity_vllm.check.match import batch_decomp as BD
from verity_vllm.check.match import program_compare as C
from verity_vllm.check.program_ops import Ops
from verity_vllm.check.replay.fork import die_with_parent

FLAG = "MATCH_IMPL"

# [R17 mfast, COORD R17-7.2] what this module IS at this revision, on every result document (`impl` block) and in the differential
# harness's report header; `identity` meant "check re-exports the checker of record" (harness plumbing proof, 69a67a90e).
IMPL_NAME = "fast"
CANDIDATE_STAGE = "v6-memo-x09share-internhash-specid+legs+compactargs"   # mprof units 1-4b (memo, X-09 Projection share, G7 O(1), digest-id Merkle payload, identity shortcut, _spec_id on the definition, cache release, per-process ids enforced) + mfast legs (MATCH_WORKERS; G8 matching returned as facts) + mfast compact operand progressions (MATCH_COMPACT)
_force: bool | None = None    # set by `check` below so the entry point selects the path, not only the environment

# memo tables (per process). Values pin the Prog objects they were computed from, so ids stay unique while cached.
_cmp_memo: dict[tuple, tuple] = {}         # (id(A), id(B), unfold_a) -> (A, B, result)
_dag_memo: dict[tuple, tuple] = {}         # (id(P), id(only) | None) -> (P, only, hashes)
_live_memo: dict[int, tuple] = {}          # id(P) -> (P, live set)
_chash_memo: dict[tuple, tuple] = {}       # (id(P), unfold) -> (P, sha)
stats: collections.Counter = collections.Counter()


def enabled() -> bool:
    if _force is not None:
        return _force
    return os.environ.get(FLAG, "").strip().lower() == "fast"


def reset() -> None:
    """Drop every memo (tests; a long-lived process checking several records)."""
    _cmp_memo.clear()
    _dag_memo.clear()
    _live_memo.clear()
    _chash_memo.clear()
    _merkle_ids.clear()
    _trie.clear()
    _chron_memo.clear()
    stats.clear()


# [R17 rev F-r17-06] the memo tables pin the Prog / Projection objects they were computed from; without a release a process that
# checks several records in a row (harness in-process mode, tests, a profiler loop) grows by one record's caches per record.  check()
# brackets its body with enter()/leave(): the OUTERMOST call (depth 0 again -- the alternate-criterion run is a nested check() and must
# keep the memo) drops every table.  row_pod = one record per process, so the release changes nothing on a row of record.
_depth = 0


def enter() -> None:
    global _depth
    _depth += 1


config: dict = {}       # [rev S-r17-17 / M-0276 (4)] the declared configuration of the fast path as `ops()` read it (workers cap, operand form,
                        # switches as read from the environment) -- stated on the document's impl block
last_stats: dict = {}   # [rev F-r17-09] the outermost check()'s counters, snapshotted at leave() BEFORE the release: the CLI's impl block
                        # (impl_info, called after check() returned) reports the run's configuration -- MATCH_WORKERS used, MATCH_COMPACT,
                        # memo hit / miss -- instead of an empty table


def leave() -> None:
    """Outermost check() done (or raised): release every cache so the next record starts from nothing (the counters survive in
    `last_stats` for the document)."""
    global _depth, last_stats
    _depth -= 1
    if _depth <= 0:
        _depth = 0
        last_stats = {k: int(v) for k, v in sorted(stats.items())}
        reset()
        reset_projections()


# ------------------------------------------------------------------------------------------------ canon (per Prog, per instance)

class _CanonMemo:
    """Mixin over compare.Prog (`MemoProg`) and batch_decomp.Projection (`_ProjectionFast`): `canon` computed once per (instance,
    unfold) and returned as the SAME list object thereafter (no caller mutates it: compare._canon_step_relative, batch_decomp.dag_hashes
    / live_cone and global_match iterate only).  A Projection's `_xreq` calls made during the first computation are recorded and
    replayed on every hit, so cross-request read counts are unchanged."""

    def canon(self, i: int, unfold: bool = False):
        cache = self.__dict__.get("_fast_canon")
        if cache is None:
            cache = self._fast_canon = {}
        key = (i, unfold)
        hit = cache.get(key)
        if hit is not None:
            stats["canon_hit"] += 1
            out, events = hit
            if events:
                xreq = self._xreq
                for ev in events:
                    xreq(*ev)
            return out
        stats["canon_miss"] += 1
        if hasattr(self, "_xreq"):
            rec: list = []
            orig_xreq = self._xreq

            def _rec_xreq(*ev):   # (kind, idx, lo, hi) on the record | (kind, (idx, lo, hi), (dI, dL, dH), count) compact [mfast]
                rec.append(ev)
                orig_xreq(*ev)

            self._xreq = _rec_xreq          # instance attribute shadows the class method for this one computation
            try:
                out = super().canon(i, unfold)
            finally:
                del self._xreq
            events = tuple(rec)
        else:
            out = super().canon(i, unfold)
            events = ()
        cache[key] = (out, events)
        return out


# ------------------------------------------------------------------------------------------------ compare_steps / hashes

def _compare_steps_memo(A: C.Prog, B: C.Prog, unfold_a: bool = False) -> dict:
    """compare.compare_steps once per (A, B, unfold_a) object pair; the result dict is returned as the same object (callers read it)."""
    key = (id(A), id(B), bool(unfold_a))
    hit = _cmp_memo.get(key)
    if hit is not None and hit[0] is A and hit[1] is B:
        stats["compare_steps_hit"] += 1
        return hit[2]
    stats["compare_steps_miss"] += 1
    res = C.compare_steps(A, B, unfold_a, canon_step_relative=_canon_step_relative_fast, canonical_hash=_canonical_hash_memo)
    _cmp_memo[key] = (A, B, res)
    return res


def _canonical_hash_memo(P: C.Prog, seqs: list[list[int]], unfold: bool) -> str:
    """compare._canonical_hash once per (Prog, unfold) for Progs WITHOUT cross-request bookkeeping (components, folds): `seqs` is
    `compare_steps.seq(P, unfold)`, a pure function of the same Prog.  A Projection (has `_xreq`) is hashed by the original every
    time, so its `P.xreq` counting (read by X-09 right after compare_steps) is untouched -- its canon calls hit the canon memo."""
    if hasattr(P, "_xreq"):
        stats["canonical_hash_projection"] += 1
        return C._canonical_hash(P, seqs, unfold, canon_step_relative=_canon_step_relative_fast)
    key = (id(P), bool(unfold))
    hit = _chash_memo.get(key)
    if hit is not None and hit[0] is P:
        stats["canonical_hash_hit"] += 1
        return hit[1]
    stats["canonical_hash_miss"] += 1
    h = C._canonical_hash(P, seqs, unfold, canon_step_relative=_canon_step_relative_fast)
    _chash_memo[key] = (P, h)
    return h


# [rev F-r17-07] THE ID SPACE IS PER PROCESS.  `_merkle_ids` assigns small ids to payload digests in FIRST-SIGHT order, so the same
# digest gets different ids in two processes that saw different things first: a dag_hashes dict is meaningful ONLY in the process that
# computed it (or in a child forked afterwards, which inherits the table).  Two dicts from two processes must never be compared or
# mixed (a worker's dict seeded into the parent's memo beside a parent-computed dict = two id spaces, F-r17-07).  The dicts therefore
# carry their origin pid and the memo refuses a dict from any process other than this one or its parent -- loudly, never a wrong match.
_merkle_ids: dict[tuple, int] = {}   # structural Merkle key (unit 6) -> small id; one table per process (see above)


class _Hashes(dict):
    """dag_hashes result (instance -> small id) that knows which process's id space it belongs to."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.pid = os.getpid()


def _check_origin(h) -> None:
    """[F-r17-07, hardening F-r17-08] a memo hit must be a _Hashes of THIS process (or of the parent whose id table a forked worker
    inherited); a plain dict (no origin) is refused like a foreign one -- never compared."""
    if not isinstance(h, _Hashes):
        raise RuntimeError(f"global_match_fast: dag_hashes memo holds a {type(h).__name__} without an origin pid -- only a _Hashes computed "
                           f"by this process is accepted (F-r17-08)")
    if h.pid not in (os.getpid(), os.getppid()):
        raise RuntimeError(f"global_match_fast: dag_hashes ids from process {h.pid} used in process {os.getpid()} -- the small-id space is "
                           f"per process (F-r17-07); recompute in this process or map ids back to digests before crossing")


# [R17 mprof unit 6] STRUCTURAL INTERNING replaces sha256(json(payload)) in the fast path's dag_hashes.  The record's Merkle hash is
# sha256 of the JSON text of [spec, form, parts, out] with every node-space run written as ["N", <producer's hash>, lo, hi] and every
# other run as its tuple; two instances compare equal iff those texts are equal.  On this domain (str / int leaves only) text equality
# is structural equality, so an id assigned by a dict keyed on the STRUCTURE -- (spec, form, out, one id per operand), the operand id
# by a prefix trie over its run items, the producer named by ITS id -- induces exactly the same partition (by induction over the node
# order, in which producers precede consumers; a forward / dead producer keeps the record's marker string).  No text, no digest, no
# collision.  For a compact (progression-form) operand the trie walk over a node-space progression of `count` runs is memoised per
# (prefix node, base, delta, count) WITHIN ONE CALL (the producers named by a (step, index) base are this Prog's, under this cone):
# the attention gather at position t+1 is the gather at position t plus one producer, so it costs ONE trie step instead of t -- the
# run count no longer multiplies the work.  The relation is unchanged: every memoised node is the trie node the full walk would reach.
# A non-node progression is one trie step of its own (a different item set, the same partition: the operand is a normal form).  Values stay equality keys inside ONE process (F-r17-07: `_Hashes` carries the pid).
_trie: dict[tuple, int] = {}         # (prefix node, run item) -> node; node 0 = the empty operand; per process, released on leave


def _dag_hashes_intern(P: C.Prog, only: set[int] | None = None) -> dict[int, int]:
    ids = _merkle_ids
    trie = _trie
    pm: dict[tuple, int] = {}   # (prefix node, base, delta, count) -> node, for THIS Prog / cone only
    hashes: dict[int, int] = _Hashes()
    steps = P.steps
    compact = getattr(P, "compact", False)
    canon = P.canon
    for s, idxs in enumerate(steps):
        for i in idxs:
            if only is not None and i not in only:
                continue
            r = P.inst[i]
            parts = []
            for arg in canon(i, False):
                node = 0
                for t in arg:
                    if compact and len(t) >= 3 and isinstance(t[-3], tuple) and isinstance(t[-2], tuple) and isinstance(t[-1], int):
                        head, base, delta, c = t[:-3], t[-3], t[-2], t[-1]
                        if head == ("N",):
                            k0 = (node, base, delta, c)
                            n = pm.get(k0)
                            if n is not None:
                                node = n
                                continue
                            start, n = 0, None
                            if c > 1:
                                n = pm.get((node, base, delta, c - 1))
                            node0 = node
                            if n is not None:
                                node, start = n, c - 1
                            s0, jj0, lo0, hi0 = base
                            ds, djj, dlo, dhi = delta
                            # [F-r17-10, rev M-0338/M-0383] pm is keyed on STRUCTURE, so an entry must be what the full walk reaches for
                            # EVERY later consumer: a producer named by ITS id is stable (hashes only grows), a producer named by the
                            # marker (not hashed yet: forward ref, or dead under this cone) is not -- the record re-reads `hashes` per
                            # consumer, and a resolved consumer after the producer gets the producer's digest where the forward-ref
                            # consumer before it got the marker.  So entries are written only up to the first marker item; a walk that
                            # met a marker recomputes from the last marker-free prefix next time (the prefix reuse above is sound by
                            # the same rule: prefixes stored are marker-free).  PASS records have no markers -- nothing changes there.
                            memo = True
                            for k in range(start, c):
                                ii = steps[s0 + k * ds][jj0 + k * djj]
                                h = hashes.get(ii)
                                if h is None:
                                    memo = False
                                    item = ("N", f"forward-ref-or-dead:{ii}", lo0 + k * dlo, hi0 + k * dhi)
                                else:
                                    item = ("N", h, lo0 + k * dlo, hi0 + k * dhi)
                                nk = trie.get((node, item))
                                if nk is None:
                                    nk = trie[(node, item)] = len(trie) + 1
                                node = nk
                                if memo:
                                    pm[(node0, base, delta, k + 1)] = node
                            continue
                        # an input / parameter progression is ONE trie step: canon's operands are rle_mapped's normal form, so equal run
                        # lists give equal items ([GLM-5.3] a strided W_UV read is ~2048 progressions over ~1e6 runs)
                        nk = trie.get((node, t))
                        if nk is None:
                            nk = trie[(node, t)] = len(trie) + 1
                        node = nk
                        continue
                    if t[0] == "N":
                        ii = steps[t[1]][t[2]]
                        h = hashes.get(ii)
                        item = ("N", h if h is not None else f"forward-ref-or-dead:{ii}") + tuple(t[3:])
                    else:
                        item = tuple(t)
                    nk = trie.get((node, item))
                    if nk is None:
                        nk = trie[(node, item)] = len(trie) + 1
                    node = nk
                parts.append(node)
            key = (r["spec"], r["form"], r["out"], tuple(parts))
            hid = ids.get(key)
            if hid is None:
                hid = ids[key] = len(ids)
            hashes[i] = hid
    return hashes


def _dag_hashes_memo(P: C.Prog, only: set[int] | None = None) -> dict[int, int]:
    """dag_hashes once per (Prog, `only` set object).  X-09 passes the live cone (memoised below, so the same set object per Prog);
    GM-01 passes None.  Returned dict is shared (callers read it)."""
    key = (id(P), None if only is None else id(only))
    hit = _dag_memo.get(key)
    if hit is not None and hit[0] is P and (only is None or hit[1] is only):
        _check_origin(hit[2])   # F-r17-07: a dict computed in another process (seeded from a worker) is refused, not compared
        stats["dag_hashes_hit"] += 1
        return hit[2]
    stats["dag_hashes_miss"] += 1
    h = _dag_hashes_intern(P, only)
    _dag_memo[key] = (P, only, h)
    return h


def _canon_step_relative_fast(P: C.Prog, i: int, seqs: list[list[int]], unfold: bool):
    """compare._canon_step_relative: with unfold=False and `seqs` == P.steps (compare_steps' seq(P, False) is P.steps minus empty
    trailing segments, and _segment never emits an empty one) the position remap is the identity -- ('N', s, jj, lo, hi) already names
    the producer by (step, index within step) -- so the result is canon(i) with each operand as a tuple.  Any other case runs the
    original (unfold_a=True is the padded-vs-flat comparator, not used by GM-01/X-09)."""
    if unfold:
        return C._canon_step_relative(P, i, seqs, unfold)
    ident = P.__dict__.get("_fast_seqs_ident")
    if ident is None or ident[0] is not seqs:
        P._fast_seqs_ident = ident = (seqs, seqs == P.steps)
        stats["seqs_identity_checks"] += 1
    if not ident[1]:
        return C._canon_step_relative(P, i, seqs, unfold)
    return [tuple(arg) for arg in P.canon(i, False)]


def _live_cone_memo(P: C.Prog) -> set[int]:
    hit = _live_memo.get(id(P))
    if hit is not None and hit[0] is P:
        stats["live_cone_hit"] += 1
        return hit[1]
    stats["live_cone_miss"] += 1
    s = BD.live_cone(P)
    _live_memo[id(P)] = (P, s)
    return s


# ------------------------------------------------------------------------------------------------ Projections (X-09 <-> GM-01 <-> alternate)

_proj_registry: dict[tuple, tuple] = {}    # construction key -> (oracle Prog, Projection): every _ProjectionFast built


def _proj_key(oracle: C.Prog, inst_idx, tok_node, weights_node, seed_node, peer_node=None) -> tuple:
    w = frozenset(weights_node) if isinstance(weights_node, (set, frozenset, list, tuple)) else frozenset({weights_node})
    return (id(oracle), tuple(inst_idx), tok_node, w, seed_node, peer_node)   # [R17 D90] the request's peers node is part of the view


class _ProjectionFast(_CanonMemo, BD.Projection):
    """batch_decomp.Projection with canon memoised (`_CanonMemo`) that registers itself by its construction arguments.
    `Projection.__init__` is deterministic in (oracle, inst_idx, tok_node, weights_node, seed_node, peer_node) -- the label only names the
    object -- so two Projections with the same key are the same view of the fold: X-09 (`decompose`, run first inside GM-01 on the same
    decoded fold + components) and GM-01's G3 build one each per request; `projection()` below hands G3 the X-09 one (relabelled) instead
    of rebuilding + recanonicalising it."""

    def __init__(self, oracle, label, inst_idx, tok_node, weights_node, seed_node=None, peer_node=None):  # noqa: D107
        super().__init__(oracle, label, inst_idx, tok_node, weights_node, seed_node, peer_node)
        key = _proj_key(oracle, inst_idx, tok_node, weights_node, seed_node, peer_node)
        _proj_registry[key] = (oracle, self)
        stats["projection_built"] += 1


# ------------------------------------------------------------------------------------------------ unit 5: the remaining O(runs) loops
# On the representative record (llama B8 1024/128: 698,398 instances, 54,310,062 operand runs -- 97.6 % of them the attention K/V
# gathers, one run per cached position) every phase left on the fast path is a Python loop over runs.  The representation that makes
# them O(progressions) is mfast's; these three keep the record's loops but do each run's work once and in C where the original did it
# several times in Python: (a) refs.runs for the shapes the fold's descriptor actually has, (b) the per-request producer-chronology count
# and (c) the dep-order scan as an any() before the per-tuple loop that names violations.  Each returns exactly what the original does.

def _runs_fast(refs):
    """verity.ml.circuits.refs.runs for an Affine or a Concat of Affines (every operand of a fold descriptor: 1.40 M affine + 137 k concat, and
    the alias chains codec._decode_refs splices flat are Concats of Affines) -- one loop, no recursion, no closures; the merge rule is
    the record's `push` (adjacent runs of the same (space, idx) with hi == next lo fuse).  Anything else goes to the original."""
    from verity.ml.circuits.refs import Affine, Concat
    if type(refs) is Affine:
        return [(refs.space, refs.idx, refs.base, refs.base + refs.count)] if refs.count else []
    if type(refs) is Concat:
        parts = refs.parts
        out: list[tuple] = []
        for p in parts:
            if type(p) is not Affine:
                break
            c = p.count
            if not c:
                continue
            s, i, b = p.space, p.idx, p.base
            if out:
                l = out[-1]
                if l[0] == s and l[1] == i and l[3] == b:
                    out[-1] = (s, i, l[2], b + c)
                    continue
            out.append((s, i, b, b + c))
        else:
            return out
    from verity.ml.circuits.refs import runs
    return runs(refs)


_chron_memo: dict[int, tuple] = {}   # id(engine_step_of) -> (engine_step_of, B, step_of_k)


def producer_chronology(P, idxs: list[int], B, engine_step_of: dict, weights_nodes) -> tuple[int, collections.Counter, bool]:
    """The record's per-request producer-chronology loop (global_match._check, G5 material), same three results: `late` = node operand
    runs whose producer instance sits in a LATER engine step than the consumer; `xstep[spec family]` = runs whose producer is in an
    EARLIER step; `weight_reader` = the request reads a weights input.  The record does four dict lookups per run (B.inputs, weights,
    B.k2i, engine_step_of); this resolves node -> producer's engine step ONCE per fold (`step_of_k`, memoised on the engine_step_of
    object; inputs map to None exactly as `B.k2i.get` misses did) and counts per operand with comprehensions."""
    key = id(engine_step_of)
    hit = _chron_memo.get(key)
    if hit is not None and hit[0] is engine_step_of and hit[1] is B:
        step_of_k = hit[2]
    else:
        get = engine_step_of.get
        step_of_k = {k: get(i) for k, i in B.k2i.items()}
        _chron_memo[key] = (engine_step_of, B, step_of_k)
    w_in = frozenset(k for k in weights_nodes if k in B.inputs)
    sk = step_of_k.get
    late = 0
    xstep: collections.Counter = collections.Counter()
    weight_reader = False
    for i_local, r in enumerate(P.inst):
        cs = engine_step_of.get(idxs[i_local])
        fam = None
        for a in r["args"]:
            if not weight_reader and any(t[0] == "n" and t[1] in w_in for t in a):
                weight_reader = True
            if cs is None:
                continue
            ps_list = [sk(t[1]) for t in a if t[0] == "n"]
            if not ps_list:
                continue
            late += sum(1 for ps in ps_list if ps is not None and ps > cs)
            early = sum(1 for ps in ps_list if ps is not None and ps < cs)
            if early:
                if fam is None:
                    fam = r["spec"].split("{")[0]
                xstep[fam] += early
    return late, xstep, weight_reader


def dep_order_violations(P, rid: str) -> list[dict]:
    """The record's G8 dependency-order scan of one request's projection (every canon tuple: producer emitted at or after its consumer
    inside the request = a violation, one record per tuple, in scan order).  Same records, same order; operands with no violation
    (all of them on a valid record) are settled by one any() instead of the per-tuple loop."""
    out: list[dict] = []
    steps = P.steps
    canon = P.canon
    for s_, loc in enumerate(steps):
        for pos, i in enumerate(loc):
            for a in canon(i, False):
                if any(t[0] == "N" and steps[t[1]][t[2]] >= i for t in a):
                    for t in a:
                        if t[0] == "N" and steps[t[1]][t[2]] >= i:
                            out.append({"request_id": rid, "local_step": s_, "emission": pos, "consumer": i, "producer": steps[t[1]][t[2]]})
    return out


# ------------------------------------------------------------------------------------------------ the fast path's Programs and operations

def _spec_id_memo(fn):
    """[R17 mprof unit 3, BRIEF s5 'cache immutable Definition/specialization analysis'] `codec._spec_id(fn)` serialises a
    specialization's static bindings to canonical JSON on EVERY call; `compare.Prog.__init__` calls it once per NODE (162 k for the
    B8 256/32 fold, 3-9 k per component), while the `fn` objects are shared by every node of the same specialization -- 11 % of a
    component's load on the real record.  The id is a pure function of the (immutable after construction) definition + bindings: it
    is kept ON the definition object (no memo table pins the decoded IR, which Prog.__init__ drops for memory), and only after a
    successful computation -- `_spec_id` reads every static's binding, so a partially bound specialization raises instead of caching."""
    try:
        return fn.__dict__["_gmf_spec_id"]
    except (AttributeError, KeyError):
        pass
    from verity.ml.circuits.codec import _spec_id
    sid = _spec_id(fn)
    try:
        fn.__dict__["_gmf_spec_id"] = sid
    except AttributeError:   # a __slots__ definition class: nothing to keep it on; recompute as the checker of record does
        pass
    return sid


class MemoProg(_CanonMemo, C.Prog):
    """The fast path's compare.Prog (what `ops()` loads the fold and the components as): canon computed once per instance
    (`_CanonMemo`), and a run-form decode whose two per-node functions are `_spec_id_memo` and `_runs_fast` (unit 5 (a)) -- each
    returns what the record's `codec._spec_id` / `refs.runs` returns for the same node."""

    def _node_functions(self):
        return _spec_id_memo, _runs_fast


def ops() -> Ops:
    """The fast path's Program operations for one `global_match.check` and the X-09 run inside it: the fold and components as MemoProg,
    Projections that register for reuse, the memoised comparison / hashes / live cone, X-09's per-request legs on forked workers
    (MATCH_WORKERS).  Also states the run's declared configuration (`config`)."""
    # [R17 mfast, BRIEF §5 step 2] every Prog loaded here holds its operands as run progressions (cb_a.compact) unless MATCH_COMPACT=0
    # (the A/B switch for the representation alone); the record's Progs are runs
    compact = os.environ.get(COMPACT_ENV, "1").strip() != "0"
    stats["compact_args"] = int(compact)
    # [rev S-r17-17 condition / M-0276 (4)] the run's DECLARED configuration, stated on the document's impl block beside the counters:
    # the effective worker cap (MATCH_WORKERS or the default), the operand form, and the switches as read -- not inferred from hits
    raw_w = os.environ.get(WORKERS_ENV, "").strip()
    config.clear()
    config.update({"match_workers": workers(1 << 30), "match_workers_env": raw_w or None, "cpu_count": os.cpu_count(),
                   "match_compact": compact, "match_compact_env": os.environ.get(COMPACT_ENV) or None,
                   "operand_form": "progressions/v1" if compact else "runs",
                   "x09_legs": "forked" if workers(1 << 30) > 1 else "in-process", "candidate_stage": CANDIDATE_STAGE})
    return Ops("fast", load_oracle=functools.partial(C.load_oracle, cls=MemoProg, compact=compact),
                  load_derived=functools.partial(C.load_derived, cls=MemoProg, compact=compact), projection=_ProjectionFast,
                  compare_steps=_compare_steps_memo, dag_hashes=_dag_hashes_memo, live_cone=_live_cone_memo, legs=_x09_legs)


def reset_projections() -> None:
    _proj_registry.clear()


# ------------------------------------------------------------------------------------------------ check() hooks

def projection(loaded: dict, rid: str, oracle: C.Prog, label: str, inst_idx: list[int], tok_node, weights_node, seed_node,
               peer_node=None) -> BD.Projection:
    """The request's Projection of the fold for GM-01's G3, built at most once per process per construction key:
      1. the alternate-criterion run reuses the verdict-of-record run's (per `_loaded`);
      2. the first run reuses X-09's when `decompose` built one with the SAME (fold, instance list, prompt node, weights nodes, seed node)
         -- relabelled to G3's label (the label names the object; no canonical form or comparison reads it);
      3. else it is built here (and registered).
    On every reuse the cross-request bookkeeping is reset: the G3 loop (`P.canon(i)` for every instance) then replays exactly the
    `_xreq` events a freshly built Projection records, so `P.xreq` at the G5 read equals the checker of record's."""
    projs = loaded.setdefault("fast_projections", {})
    P = projs.get(rid)
    if P is None:
        key = _proj_key(oracle, inst_idx, tok_node, weights_node, seed_node, peer_node)
        hit = _proj_registry.get(key)
        if hit is not None and hit[0] is oracle:
            P = hit[1]
            P.label = label
            P.digest = f"{oracle.digest}#{label}"
            stats["projection_reused_from_x09"] += 1
        else:
            P = _ProjectionFast(oracle, label, inst_idx, tok_node, weights_node, seed_node, peer_node)
        projs[rid] = P
    else:
        stats["projection_reused_alternate"] += 1
    P.xreq = collections.Counter()
    P.xreq_examples = []
    return P


def g7_owner_blocks(gis: list[int], glob2loc: dict) -> tuple[list[tuple[str, int]], list[int]]:
    """G7 boundary probes: the owner blocks of one engine step in emission order, plus the LAST attributed instance of each block.
    The checker of record finds the instance before a boundary as `max(g for g in gis if g < gb and owner(g) == oa)` -- a rescan of the
    step per boundary.  Blocks are maximal runs of equal owner over the ATTRIBUTED instances of the sorted step (unattributed ones are
    skipped by both forms), so that maximum IS the last attributed instance of the block that ends at the boundary: O(1) per boundary,
    same value."""
    owners: list[tuple[str, int]] = []
    last: list[int] = []
    for gi in gis:
        o = glob2loc.get(gi)
        if o is None:
            continue
        if not owners or owners[-1][0] != o[0]:
            owners.append((o[0], gi))
            last.append(gi)
        else:
            last[-1] = gi
    return owners, last


# ------------------------------------------------------------------------------------------------ per-request legs on forked workers

WORKERS_ENV = "MATCH_WORKERS"
COMPACT_ENV = "MATCH_COMPACT"


def workers(n_items: int) -> int:
    """[R17 mfast, BRIEF §5 step 3] How many forked workers the per-request legs run on: `MATCH_WORKERS` (default min(cpu_count, 8)),
    never more than there are legs; 1 = every leg in the parent process (the memo-only path)."""
    raw = os.environ.get(WORKERS_ENV, "").strip()
    n = int(raw) if raw else min(os.cpu_count() or 1, 8)
    return max(1, min(n, max(1, n_items)))


def _x09_legs(items: list, leg: Callable, weight: Callable) -> list:
    """batch_decomp.decompose's per-request legs (Projection + compare_steps + DAG multiset + xreq of ONE request) on forked workers:
    the fold and every component are loaded by the parent before this call (copy-on-write); results are facts (the request's entry).
    MATCH_WORKERS=1 -> in this process."""
    n = workers(len(items))
    stats["x09_workers"] = n
    return request_legs(items, leg, n, weight=weight)


def seed_dag_hashes(P: C.Prog, hashes: dict) -> None:
    """RETIRED (F-r17-07, rev M-0154): a worker's dag_hashes dict must never seed the parent's memo -- the Merkle ids are per process.
    The legs return G8's per-step matching as facts instead (global_match._perm_match in the leg)."""
    raise RuntimeError("seed_dag_hashes is retired (F-r17-07): legs return matching facts, never hash dicts across processes")


def request_legs(items: list, fn: Callable, n_workers: int, weight: Callable | None = None) -> list:
    """Run `fn(item)` for every item and return the results IN ITEM ORDER.

    n_workers <= 1 (or one item): in this process.  Otherwise: the caller has built every shared object the legs read (fold Prog,
    component Progs, Projections) -- `os.fork()` gives each worker a copy-on-write view of exactly those objects (no pickling of the
    trace, no copy per process; a worker touches its own requests' instances and the shared indices), each worker runs its share of the
    legs sequentially (heaviest legs dealt first, round-robin by `weight`) and pickles the RESULTS (facts: entries, problem strings,
    address maps, hash dicts) to a temp file; the parent waits for every worker, reads the results, and RAISES if any worker failed --
    never a silent fallback to an incomplete set of legs.  gc is frozen around the fork so the children's collector does not dirty the
    parent's pages.  `fn` runs the same code on the same inputs wherever it runs; the differential harness checks the facts are the
    same."""
    if n_workers <= 1 or len(items) <= 1 or not hasattr(os, "fork"):
        return [fn(x) for x in items]
    import gc
    import pickle
    import tempfile
    import traceback
    order = sorted(range(len(items)), key=lambda j: -(weight(items[j]) if weight else 0))
    parts: list[list[int]] = [[] for _ in range(min(n_workers, len(items)))]
    for rank, j in enumerate(order):
        parts[rank % len(parts)].append(j)
    gc.collect()
    gc.freeze()
    children = []
    parent = os.getpid()
    try:
        for idxs in parts:
            if not idxs:
                continue
            fd, path = tempfile.mkstemp(prefix="gm_leg_", suffix=".pkl")
            os.close(fd)
            pid = os.fork()
            if pid == 0:   # worker
                code = 0
                try:
                    die_with_parent(parent)
                    out = {j: fn(items[j]) for j in idxs}
                    with open(path, "wb") as f:
                        pickle.dump({"ok": True, "results": out, "stats": dict(stats)}, f, protocol=pickle.HIGHEST_PROTOCOL)
                except BaseException as e:  # noqa: BLE001 - reported to the parent, which raises
                    code = 1
                    try:
                        with open(path, "wb") as f:
                            pickle.dump({"ok": False, "error": repr(e)[:2000], "trace": traceback.format_exc()[-4000:]}, f)
                    except BaseException:  # noqa: BLE001
                        pass
                finally:
                    os._exit(code)
            children.append((pid, path, idxs))
    finally:
        gc.unfreeze()
    results: list = [None] * len(items)
    errors = []
    for pid, path, idxs in children:
        _, status = os.waitpid(pid, 0)
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
        except Exception as e:  # noqa: BLE001
            payload = {"ok": False, "error": f"worker {pid} (legs {idxs}) left no result, wait status {status}: {e!r}"}
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        if not payload.get("ok"):
            errors.append(payload)
            continue
        for j, r in payload["results"].items():
            results[j] = r
        for k, v in (payload.get("stats") or {}).items():
            if not k.startswith("worker."):   # the child's own counters only (it inherited the parent's, including earlier workers')
                stats[f"worker.{k}"] += v
        stats["legs_on_workers"] += len(payload["results"])
    if errors:
        raise RuntimeError(f"{len(errors)} of {len(children)} per-request workers failed: "
                           + " | ".join(str(e.get("error")) + ("\n" + e["trace"] if e.get("trace") else "") for e in errors)[:6000])
    if any(r is None for r in results):
        raise RuntimeError("per-request workers returned an incomplete set of legs")
    return results


# ------------------------------------------------------------------------------------------------ entry point (MATCH_IMPL dispatch)

def check(gp: dict, oracle_dir: str, log_path: str | None = None, tokens_path: str | None = None, run_x09: bool = True,
          within_step: str | None = None, record_alternate: bool = True, _loaded: dict | None = None,
          tokens_optional: bool = False) -> dict:
    """[R17 mfast] The candidate's global Match as `global_match.impl_module("fast").check(...)` runs it: the checker of record's
    `check` over this module's `ops()` for the duration of the call (whatever the environment says).  Same signature, same result
    schema; nothing else is different by construction -- the differential harness proves it on the artifacts.
    [rel r17-int-2] `tokens_optional` (stoch F-r17-01, CLI --tokens-optional) forwarded unchanged so the CLI's `mod.check(...)` call has one
    signature for both implementations."""
    global _force
    from verity_vllm.check.match import global_match as _record  # deferred: global_match imports this module at load
    prev = _force
    _force = True
    try:
        return _record.check(gp, oracle_dir, log_path, tokens_path, run_x09=run_x09,
                             within_step=within_step if within_step is not None else _record.DEFAULT_WITHIN_STEP,
                             record_alternate=record_alternate, _loaded=_loaded, tokens_optional=tokens_optional)
    finally:
        _force = prev
