"""`Q_word_v1{X,W,R=no-recompute}`: every Call's circuit cut into proof blocks whose committed outputs are at most X bits, or one value
of at most W bits, over values serving commits.

The rule (no recomputes):

  * A gate is interior to exactly one unit or on the boundary of one or more units; each gate is **computed in exactly one unit**,
    and so is each value: a unit never recomputes a value another unit computes.  A value computed twice inside one unit reaches
    no other unit; it is reported as the unit's redundant gates (`redundant_gates`), not a recompute.
  * For every computation gate exactly one unit certifies it: its proof being valid implies the gate is the correct function of the
    committed values the unit reads.  A value read across units is **committed once at serving**, and every reader reuses it.  There
    are no public inputs: the Call's inputs (weights, tokens, earlier Calls' outputs) are committed, private values, and so is every
    value a program or a request brings (a norm's epsilon, the attention scale, a softcap, temperature, top-p): a registered or run
    input, committed under a hiding leaf.  Only an operation's own constant bits and wiring (`WIRING`: bit rearrangement, such as
    BF16 -> FP32 zero-extension) are circuit structure, not counted as gates; a unit reads through wiring to its source.
  * **Width** counts bits (`verity.ml.circuits.cut.fits`): a unit's boundary gates (read by another unit, or returned) total at most X declared
    bits (16), or the unit has exactly ONE boundary value, of at most W bits (32: float interiors such as a norm scale, attention's
    max, sum and inverse sum).  No dtype is read.
  * It is the query of record (`required.QUERY_OF_RECORD`): a Call's inputs and outputs are the bridge's required values.

The cut of one Call (`partition`), on its computing gates (`Graph`, the graph of core's `verity.ml.circuits.cut.CallGraph` in flat arrays:
wiring and constants removed, reads through them re-pointed):

  1. The returned gates, in return order, packed into outputs of at most X bits (a wider value is an output alone).
  2. Backward over the gates (the layout is topological): a gate joins its consumers' unit when they all belong to one; a gate
     several units read is committed as its own unit, and the pass repeats.
  3. A gate no returned value depends on (dead) joins the one unit it reads, else heads a dead unit; any value such a gate reads
     across units is committed too.

`verity.ml.circuits.partition.validate_unit_cut` checks every Call's cut (`cut_verdict`, its verdict in numpy passes over the reads); the
committed interior values are what serving must acquire (a tap
where it does not today).  `unit_rule` summarises one specialization; `check_query` checks a Program and fails by name.

A specialization's unit rule is a function of the Definition, the query and the gate limits alone.  Attention has one specialization
per key length, each cut in time linear in it, so a long context's rules cost hours; `use_rule_cache(dir)` (the manifest CLI's
`--unit-rule-cache`, env `RULE_CACHE_ENV`) makes this process and its forked prefetch read and write the rules in that directory.
"""
from __future__ import annotations

import hashlib
import json
import os
from array import array
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from verity.ml.circuits.codec import canonical_json, definition_entry
from verity.ml.circuits.cut import WIDE_FAN_IN, WIRING, QueryInapplicable, call_scope, cut_word, fits, separable, wide_operands
from verity.ml.circuits.defs import PrimitiveDefinition
from verity.ml.circuits.layout import resolve
from verity.ml.circuits.partition import MAX_LISTED, CutVerdict

from verity.ml.definitions.call_scope import QueryRuleViolation, specialization

QUERY_ID = "Q_word_v1"


#: the rule's name in the query id
RULE = "no-recompute"

_KINDS = ("output", "committed", "dead")

#: Calls with more gates than this are not partitioned explicitly (`check_calls` names them).  A run raises it (`gate_limits`) with a
#: subcommand's `--word-max-gates` (or `VERITY_QWORD_MAX_GATES`): a memory budget (about 0.5 KB per gate and 20 bytes per operand it
#: reads: a MoE expert gather reads every expert's word), not part of `query_id`
MAX_GATES = 24_000_000
MAX_GATES_ENV = "VERITY_QWORD_MAX_GATES"

#: `max_gates` as the cut functions take it: a limit for every Definition, {family or descriptor id: limit, "*": the rest}, a
#: `gate_limits` spec, or None (`MAX_GATES`)
Limits = int | str | Mapping[str, int] | None


def gate_limits(spec: Limits = None) -> dict[str, int]:
    """`spec` as {"*": limit, family or descriptor id: limit}.  A spec string is comma-separated entries, each a limit for every
    Definition (`120000000`) or one family's (`GumbelTopPTokenSelect_v2=120000000`); None is `MAX_GATES` for every one."""
    if spec is None:
        spec = MAX_GATES
    if isinstance(spec, int):
        return {"*": int(spec)}
    if isinstance(spec, Mapping):
        return {"*": MAX_GATES, **{str(k): int(v) for k, v in spec.items()}}
    out = {"*": MAX_GATES}
    for e in str(spec).split(","):
        k, _, v = e.strip().rpartition("=")
        out[k or "*"] = int(v.replace("_", ""))
    return out


def gate_limit(fn: Any, limits: Limits = None) -> int:
    """The gate limit of `fn` under `limits` (`gate_limits`): its descriptor id's, else its family's, else every Definition's."""
    lim = gate_limits(limits)
    return lim.get(fn.id, lim.get(fn.id.split("{")[0], lim["*"]))


def query_id(X: int = 16, W: int = 32, extras: int = 0) -> str:
    return f"{QUERY_ID}{{X={int(X)},W={int(W)}" + (f",EXTRAS={int(extras)}" if extras else "") + f",R={RULE}}}"


def parse(spec: str) -> tuple[int, int, int]:
    """`"16"` / `"16/32"` / `"16/32+2"` -> (X, W, extras); W defaults to max(X, 32)."""
    body, _, e = str(spec).partition("+")
    x, _, w = body.partition("/")
    return int(x), int(w) if w else max(int(x), 32), int(e or 0)


def _view(a: array) -> np.ndarray:
    """`a`'s items as a numpy array over its buffer (no copy)."""
    return np.frombuffer(a, np.dtype(a.typecode)) if len(a) else np.zeros(0, np.dtype(a.typecode))


class Graph:
    """The computing gates of one activation of `fn`: the graph of core's `verity.ml.circuits.cut.CallGraph` (the reference, whose step 3
    this follows; `tests/test_word.py` holds the two equal field by field), built in one pass over the gates into flat arrays.  The
    reference keeps Python objects per operand read (an interned token per input, a tuple per input read, the full key of every computing
    gate), hundreds of bytes each: GLM-5.3's MoE expert FP8 GEMMs read 257 operands per gate, 468 GB for one Definition.  Here a read
    costs a few bytes, and a computing gate's key is matched by its hash, then compared in full, so the recomputed pairs are the
    reference's.

    Numpy arrays: `keep`, `width`, `prim`, `scope`, `external`, the reads `src` -> `dst`, `input_ids` (root gate id of each Call
    parameter leaf read, first-read order) and the input reads `input_read` (input index) -> `input_reader` (gate), which `input_reads`
    lists as the reference's pairs.  Lists: `outputs`, `returned_inputs`, `recomputed`."""

    def __init__(self, fn: Any, wiring: frozenset[str] = WIRING) -> None:
        c, call = call_scope(fn)
        lo, n = call.offset, fn.gates
        tc = "i" if c.size < 1 << 31 else "q"
        width, prim, scope, tok = (array("q", bytes(8 * n)) for _ in range(4))
        idx = array("q", [-1]) * n                                   # computing gate -> its compact index
        external, free, const = bytearray(n), bytearray(n), bytearray(n)
        self.prims: dict[str, int] = {}
        self.scopes: dict[str, int] = {}
        roots = array(tc)                                            # input index -> the root gate it reads
        seen = memoryview(np.zeros(lo, np.dtype(tc)))                # a Call parameter gate's input index + 1 (0: unread)
        far: dict[int, int] = {}
        keep, src, dst = array("q"), array("q"), array("q")
        rin, rgate = array(tc), array(tc)
        structure: dict[tuple, int] = {}                             # a structure key -> the first structure gate with it
        first: dict[int, Any] = {}                                   # hash of a computing key -> the first gate (or gates) with it
        srcs: dict[int, tuple[int, ...]] = {}
        dup: list[tuple[int, int]] = []

        def operands(g: Any, j: int, wide: bool) -> tuple[list[int], bool]:
            """The gate's operands as the reference lists them (a gate of the Call by its local index, an input as -1 - its index;
            a wide gate's inputs not listed), and whether it reads outside the Call."""
            mine, ext = [], wide
            for o in g.operands if not wide else wide_operands(g, call):
                q = o - lo
                if 0 <= q < n:
                    if q >= j:
                        raise QueryInapplicable(f"{fn.id}: gate {j} reads gate {q}, not before it (the layout is not topological)",
                                                {"definition": fn.id, "gate": j, "reads": q, "why": "not topological"})
                    mine.append(q)
                    continue
                ext = True
                if wide:
                    continue
                if 0 <= o < lo:
                    x = seen[o]
                    if not x:
                        roots.append(o)
                        x = seen[o] = len(roots)
                else:
                    x = far.get(o)
                    if x is None:
                        roots.append(o)
                        x = far[o] = len(roots)
                mine.append(-x)
            return mine, ext

        def key(j: int, mine: list[int]) -> tuple:
            # an input's token is its operand (< 0), a gate's the first gate with its value (a structure gate: with its key)
            return (prim[j], *[o if o < 0 else tok[o] for o in mine])

        for j in range(n):
            g = c.gate(lo + j)
            p = g.prim
            width[j] = p.ret.w
            prim[j] = self.prims.setdefault(p.id, len(self.prims))
            scope[j] = self.scopes.setdefault(g.scope.spec.id, len(self.scopes))
            wide = len(g.operands) > WIDE_FAN_IN
            mine, ext = operands(g, j, wide)
            external[j] = ext
            cj = not p.params
            fj = cj or p.id in wiring
            if not fj and not ext and mine and all(o >= 0 and const[o] for o in mine):   # a value of constants alone is a constant
                cj = fj = True
            const[j], free[j] = cj, fj
            if fj:
                tok[j] = structure.setdefault(key(j, mine), j)
                s: set[int] = set()
                for o in mine:
                    s.update(srcs[o] if o >= 0 and free[o] else (o,))
                srcs[j] = tuple(sorted(s))
                continue
            tok[j] = j
            if not wide:                                             # a wide gate's operands were not all listed: it matches nothing
                k = key(j, mine)
                h = hash(k)
                got = first.get(h)
                cands = () if got is None else got if isinstance(got, list) else (got,)
                f = next((f for f in cands if key(f, operands(c.gate(lo + f), f, False)[0]) == k), None)
                if f is not None:
                    dup.append((j, f))
                    tok[j] = tok[f]
                elif got is None:
                    first[h] = j
                else:
                    first[h] = [*cands, j]
            i = idx[j] = len(keep)
            keep.append(j)
            reads = mine if not any(o >= 0 and free[o] for o in mine) else \
                [u for o in mine for u in (srcs[o] if o >= 0 and free[o] else (o,))]
            for u in dict.fromkeys(reads):                           # each distinct source once, in operand order
                if u < 0:
                    rin.append(-1 - u)
                    rgate.append(i)
                else:
                    src.append(idx[u])
                    dst.append(i)
        self.total, self.free = n, sum(free)
        self.n = len(keep)
        self.keep = _view(keep)
        self.width, self.prim, self.scope = _view(width)[self.keep], _view(prim)[self.keep], _view(scope)[self.keep]
        self.external = np.frombuffer(bytes(external), np.uint8)[self.keep].astype(bool) if n else np.zeros(0, bool)
        self.src, self.dst = _view(src), _view(dst)
        self.input_ids = _view(roots)
        self.input_read, self.input_reader = _view(rin), _view(rgate)
        self.recomputed = [(idx[a], idx[b]) for a, b in dup]
        outs: list[int] = []
        returned_inputs: set[int] = set()
        for g in ([0] if not hasattr(fn, "body") else (resolve(call, r) - lo for r in fn.body.ret)):
            if 0 <= g < n:
                for u in (srcs[g] if free[g] else (g,)):
                    if u >= 0:
                        outs.append(idx[u])
                    else:
                        returned_inputs.add(-1 - u)
        self.outputs = outs
        self.returned_inputs = sorted(returned_inputs)
        self.prim_names = {v: k for k, v in self.prims.items()}
        self.scope_names = {v: k for k, v in self.scopes.items()}
        counts = Counter(prim[j] for j in range(n) if free[j])
        self.free_by_prim = Counter({self.prim_names[k]: counts[k] for k in sorted(counts)})

    @property
    def input_reads(self) -> list[tuple[int, int]]:
        """The input reads as the reference's (input index, gate) pairs."""
        return list(zip(self.input_read.tolist(), self.input_reader.tolist()))


def partition(G: Graph, X: int) -> tuple[np.ndarray, list[str], np.ndarray]:
    """(owner per gate, kind per unit, head per gate) of core's `verity.ml.circuits.cut.cut_word` (the reference): "output" (returned values),
    "committed" (a value another unit reads), "dead" (read by no returned value); `head[g] >= 0` marks the gates the cut commits -- the
    returned ones and every value read across units."""
    own, kinds, head = cut_word(G, X)
    return np.asarray(own, np.int64), kinds, np.asarray(head, np.int64)


def cut_verdict(G: Graph, owner: np.ndarray, head: np.ndarray) -> CutVerdict:
    """`verity.ml.circuits.partition.validate_unit_cut` of the cut `owner` committing the gates `head >= 0`, the Call's parameters standing
    for the input unit (`verity.ml.circuits.cut.check_cut`'s arguments), in numpy passes over `G`'s reads: the reference's verdict, detail
    for detail (`tests/test_word.py`), with no Python object per read.  A parameter's read crosses from the input unit into a
    computing unit and reads a committed value, so it fails nothing: it counts in `cross_unit_reads`, and a parameter no unit reads and
    the Call does not return is `committed-unread`."""
    n, k = G.n, len(G.input_ids)
    own = np.asarray(owner, np.int64)
    uncertified = np.flatnonzero(own < 0)
    certified = n - len(uncertified)
    x = own[G.src] != own[G.dst]
    cs, cd = G.src[x], G.dst[x]
    o = np.lexsort((cd, cs))
    cs, cd = cs[o], cd[o]
    committed = np.asarray(head) >= 0
    uncommitted = ~committed[cs]
    outs = np.unique(np.asarray(G.outputs, np.int64))
    boundary = np.zeros(n, bool)
    boundary[cs] = True
    boundary[outs] = True
    read = np.zeros(k, bool)
    for s in range(0, len(G.input_read), _READS_PER_STEP):                # numpy indexes through an int64 copy of its index
        read[G.input_read[s:s + _READS_PER_STEP]] = True
    read[G.returned_inputs] = True
    across = sorted((int(a), int(b)) for a, b in G.recomputed if own[a] != own[b])
    pairs = np.stack([cs[uncommitted][:MAX_LISTED], cd[uncommitted][:MAX_LISTED]], 1).tolist()
    unread = np.concatenate([np.flatnonzero(committed & ~boundary), n + np.flatnonzero(~read)])
    problems = {"input-outside-input-unit": (0, []), "gate-not-certified-once": (len(uncertified), uncertified[:MAX_LISTED].tolist()),
                "gate-recomputed": (len(across), [list(p) for p in across[:MAX_LISTED]]),
                "gate-count": (0, []) if certified + G.free == G.total else (1, [certified]),
                "read-uncommitted": (int(uncommitted.sum()), pairs),
                "output-uncommitted": (int((~committed[outs]).sum()), outs[~committed[outs]][:MAX_LISTED].tolist()),
                "committed-unread": (len(unread), unread[:MAX_LISTED].tolist())}
    detail: dict = {"input_gates": k, "computed_gates": n, "certified": certified, "structure_gates": int(G.free), "total_gates": int(G.total) + k,
                    "committed": int(committed.sum()) + k, "cross_unit_reads": len(cs) + len(G.input_read),
                    "computing_units": len(np.unique(own[own >= 0])), "redundant_gates": len(G.recomputed) - len(across)}
    codes = tuple(c for c, (m, _g) in problems.items() if m)
    for c in codes:
        detail[c] = {"n": problems[c][0], "gates": problems[c][1]}
    return CutVerdict(not codes, codes, detail)


def units(G: Graph, X: int, W: int, extras: int = 0) -> dict[str, Any]:
    """The partition of `G` with, per unit: gates, |out(S)| in bits over its boundary gates (read by another unit, or returned), and
    the partition invariant's check (`cut_verdict`, `verity.ml.circuits.partition.validate_unit_cut` over the gates the cut commits;
    violations by gate id)."""
    owner, kinds, head = partition(G, X)
    U = len(kinds)
    gates = np.bincount(owner, minlength=U)
    cross = owner[G.src] != owner[G.dst]
    boundary = np.zeros(G.n, bool)
    boundary[G.src[cross]] = True
    boundary[G.outputs] = True
    bo = owner[boundary]
    out_bits = np.bincount(bo, weights=G.width[boundary], minlength=U).astype(np.int64)
    out_gates = np.bincount(bo, minlength=U)
    first = np.full(U, -1, np.int64)
    bidx = np.flatnonzero(boundary)
    uu, pos = np.unique(bo, return_index=True)
    first[uu] = bidx[pos]
    # the Call's parameters stand in for the input unit: values committed outside the Call (the Program's inputs under their roots, or
    # another Call's committed outputs; `check_calls` requires each to be a required value)
    cut = cut_verdict(G, owner, head)
    if not cut.ok:                                                 # the Call's gate ids (its canonical layout), not the compacted ones
        def gid(a: int) -> int | str:
            return int(G.keep[a]) if a < G.n else f"input:{int(G.input_ids[a - G.n])}"
        for k in cut.codes:
            if k != "gate-count":
                cut.detail[k]["gates"] = [[gid(a) for a in x] if isinstance(x, list) else gid(x) for x in cut.detail[k]["gates"]]
    ok = fits(out_bits, out_gates, X, W, extras)
    return {"owner": owner, "kinds": kinds, "gates": gates, "out_bits": out_bits, "out_gates": out_gates, "ok": ok, "cut": cut,
            "head_prim": np.where(first >= 0, G.prim[np.maximum(first, 0)], -1),
            "head_scope": np.where(first >= 0, G.scope[np.maximum(first, 0)], -1)}


def _combine(fid: str, gates: int, parts: list[tuple[dict[str, Any], int]], X: int, W: int, extras: int) -> dict[str, Any]:
    agg: dict[tuple, dict[str, Any]] = {}
    interior, ibits = Counter(), Counter()
    viol: dict[str, dict[str, Any]] = {}
    for r, m in parts:
        for c in r["classes"]:
            k = (c["kind"], c["head"], c["scope"], c["gates"], c["out_bits"], c.get("out_gates", 1))
            a = agg.get(k)
            if a is None:
                a = agg[k] = dict(c, count=0)
            a["count"] += c["count"] * m
        for i in r["committed_interior"]:
            interior[(i["scope"], i["head"])] += i["words"] * m
            ibits[(i["scope"], i["head"])] += i.get("bits", 32 * i["words"]) * m
        for v in r["violations"]:
            e = viol.setdefault(v["class"], dict(v, units=0))
            e["units"] += v.get("units", 1) * m
    classes = sorted(agg.values(), key=lambda c: (-c["count"], c["kind"], str(c["scope"])))
    rule = _finish(fid, gates, classes, [{"scope": s, "head": p, "words": w, "bits": ibits[(s, p)]} for (s, p), w in interior.most_common()],
                   X, W, extras)
    rule["violations"] = [dict(v, definition=fid) for v in viol.values()]
    rule["free_gates"] = sum(r.get("free_gates", 0) * m for r, m in parts)
    rule["redundant_gates"] = sum(r.get("redundant_gates", 0) * m for r, m in parts)
    free = Counter()
    for r, m in parts:
        for k, v in (r.get("free_by_prim") or {}).items():
            free[k] += v * m
    rule["free_by_prim"] = dict(free.most_common(8))
    return rule


#: the environment variable the manifest CLI's `--unit-rule-cache` defaults to
RULE_CACHE_ENV = "VERITY_UNIT_RULE_CACHE"
_RULE_CACHE_SCHEMA = "verity-vllm/unit-rule/v1"
#: the directory of unit rules this process shares (`use_rule_cache`; None: none).  An entry is keyed by the Definition's content (its
#: entry and its callees', transitively), X, W, extras, the gate limits and the source of the code that cuts (`verity.ml.circuits`, this
#: package's `word`, `cross_call` and `call_scope`); a rule is written only when JSON gives it back equal, so a read rule is the computed one
_RULE_CACHE: list[str | None] = [None]
_DIGESTS: dict[str, str] = {}
_CODE: list[str] = []


def use_rule_cache(directory: str | None) -> None:
    """Share this process's unit rules (and its forked prefetch's) through `directory`; None: no sharing."""
    _RULE_CACHE[0] = directory or None


def _code_version() -> str:
    if not _CODE:
        import verity.ml.circuits
        from verity.ml.definitions import call_scope, cross_call
        h = hashlib.sha256()
        for p in [*sorted(Path(verity.ml.circuits.__file__).parent.glob("*.py")), Path(__file__), *(Path(m.__file__) for m in (cross_call, call_scope))]:
            h.update(p.name.encode() + b"\0" + p.read_bytes() + b"\0")
        _CODE.append(h.hexdigest())
    return _CODE[0]


def definition_digest(fn: Any) -> str:
    """SHA-256 of `fn`'s descriptor entry and, sorted, its callees' digests: its whole content, whatever program it is in."""
    d = _DIGESTS.get(fn.id)
    if d is None:
        callees = [] if isinstance(fn, PrimitiveDefinition) else sorted({definition_digest(n.fn) for n in fn.body.nodes})
        d = _DIGESTS[fn.id] = hashlib.sha256(canonical_json({"definition": definition_entry(fn), "callees": callees})).hexdigest()
    return d


def _rule_entry(fn: Any, X: int, W: int, extras: int, limits: dict[str, int]) -> tuple[Path, dict[str, Any]] | None:
    root = _RULE_CACHE[0]
    if not root or isinstance(fn, PrimitiveDefinition):
        return None
    key = {"schema": _RULE_CACHE_SCHEMA, "definition": fn.id, "digest": definition_digest(fn), "query": [X, W, extras],
           "limits": [[k, v] for k, v in sorted(limits.items())], "code": _code_version()}
    k = hashlib.sha256(canonical_json(key)).hexdigest()
    return Path(root) / k[:2] / f"{k}.json", key


def _cached_rule(fn: Any, X: int, W: int, extras: int, limits: dict[str, int]) -> dict[str, Any] | None:
    entry = _rule_entry(fn, X, W, extras, limits)
    if entry is None:
        return None
    try:
        doc = json.loads(entry[0].read_text())
    except (OSError, ValueError):
        return None
    return doc.get("rule") if isinstance(doc, dict) and doc.get("key") == entry[1] else None


def _cache_rule(fn: Any, X: int, W: int, extras: int, limits: dict[str, int], rule: dict[str, Any]) -> None:
    entry = _rule_entry(fn, X, W, extras, limits)
    if entry is None:
        return
    path, key = entry
    try:
        text = json.dumps({"key": key, "rule": rule})
    except (TypeError, ValueError):
        return
    if json.loads(text)["rule"] != rule:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def unit_rule(fn: Any, X: int = 16, W: int = 32, extras: int = 0, *, memo: dict | None = None, max_gates: Limits = None) -> dict[str, Any]:
    """One Definition specialization under `Q_word_v1{X,W}`: {"definition", "query", "gates", "units_per_call", "by_kind",
    "committed_interior_words", "max_out_bits", "classes", "violations"}.  A class groups units by kind, the primitive and the
    activation of their first boundary gate, gates per unit and |out(S)|; `count` is units per Call, `nodes` the primitives of one
    unit.  `committed_interior` lists the committed interior values by (activation, primitive): what a tap must acquire.  A
    Definition over its gate limit (`gate_limit` of `max_gates`) is not cut: a `too-large` violation.  One `memo` per `max_gates`;
    under `use_rule_cache` the rule is read from, or written to, the shared directory."""
    memo = {} if memo is None else memo
    key = (fn.id, X, W, extras)
    if key in memo:
        return memo[key]
    max_gates = gate_limits(max_gates)
    hit = _cached_rule(fn, X, W, extras, max_gates)
    if hit is not None:
        memo[key] = hit
        return hit
    limit = gate_limit(fn, max_gates)
    cap = X + extras
    if isinstance(fn, PrimitiveDefinition) and (not fn.params or fn.id in WIRING):
        rule = _finish(fn.id, 1, [], [], X, W, extras)
        rule["free_gates"], rule["free_by_prim"] = 1, {fn.id: 1}
    elif isinstance(fn, PrimitiveDefinition):
        bits = fn.ret.w
        classes = [] if not fn.params else [{"kind": "output", "head": fn.id, "scope": fn.id, "gates": 1, "out_bits": bits, "count": 1,
                                             "nodes": {fn.id: 1}, "ok": bits <= cap or bits <= W + extras}]
        rule = _finish(fn.id, 1, classes, [], X, W, extras)
    elif (sep := separable(fn)) is not None:
        from verity.ml.definitions.cross_call import members_check
        rule = _combine(fn.id, fn.gates, [(unit_rule(c, X, W, extras, memo=memo, max_gates=max_gates), m) for c, m in sep], X, W, extras)
        across = members_check(fn, memo)                            # the members are cut one by one: a value two of them compute
        if across["recomputes"]:
            rule["violations"].append({"definition": fn.id, "class": "cut", "codes": ["gate-recomputed"],
                                       "detail": {"gate-recomputed": {"n": across["recomputed_gates"], "between": "node members",
                                                                      "classes": across["recomputes"][:8]}}})
    elif fn.gates > limit:
        rule = _finish(fn.id, fn.gates, [], [], X, W, extras)
        rule["violations"] = [{"definition": fn.id, "class": "too-large", "gates": fn.gates, "limit": limit}]
    else:
        G = Graph(fn)
        R = units(G, X, W, extras)
        cut = R["cut"]
        kinds = np.array([_KINDS.index(k) for k in R["kinds"]], np.int64)
        cols = np.stack([kinds, R["head_prim"], R["head_scope"], R["gates"], R["out_bits"], R["out_gates"]], 1)
        keys, first, count = np.unique(cols, axis=0, return_index=True, return_counts=True)
        pick = np.zeros(len(kinds), bool)
        pick[first] = True
        sel = pick[R["owner"]]
        pu, pc = np.unique(np.stack([R["owner"][sel], G.prim[sel]], 1), axis=0, return_counts=True)
        nodes: dict[int, Counter] = {}
        for (u, p), k in zip(pu.tolist(), pc.tolist()):
            nodes.setdefault(u, Counter())[G.prim_names[p]] = k
        classes = []
        for (kd, hp, hs, g, ob, og), u, k in zip(keys.tolist(), first.tolist(), count.tolist()):
            classes.append({"kind": _KINDS[kd], "head": G.prim_names.get(hp), "scope": G.scope_names.get(hs), "gates": g, "out_bits": ob,
                            "out_gates": og, "count": k, "ok": bool(R["ok"][u]), "nodes": dict(nodes.get(u, Counter()).most_common(8))})
        classes.sort(key=lambda c: (-c["count"], c["kind"], str(c["scope"])))
        interior, ibits = Counter(), Counter()
        for c in classes:
            if c["kind"] == "committed":
                interior[(c["scope"], c["head"])] += c["out_gates"] * c["count"]
                ibits[(c["scope"], c["head"])] += c["out_bits"] * c["count"]
        rule = _finish(fn.id, G.total, classes, [{"scope": s, "head": p, "words": w, "bits": ibits[(s, p)]} for (s, p), w in interior.most_common()],
                       X, W, extras)
        rule["free_gates"] = G.free
        rule["free_by_prim"] = dict(G.free_by_prim.most_common(8))
        rule["redundant_gates"] = cut.detail["redundant_gates"]
        if not cut.ok:
            rule["violations"].append({"definition": fn.id, "class": "cut", "codes": list(cut.codes), "detail": cut.detail})
    memo[key] = rule
    _cache_rule(fn, X, W, extras, max_gates, rule)
    return rule


def _finish(fid: str, gates: int, classes: list[dict], interior: list[dict], X: int, W: int, extras: int) -> dict[str, Any]:
    viol = [{"definition": fid, "class": f"{c['kind']}:{c['head']}@{c['scope']}", "out_bits": c["out_bits"], "out_gates": c.get("out_gates", 1),
             "limit": [X + extras, W + extras], "units": c["count"]} for c in classes if not c["ok"]]
    by_kind = Counter()
    for c in classes:
        by_kind[c["kind"]] += c["count"]
    return {"definition": fid, "query": query_id(X, W, extras), "gates": gates, "units_per_call": sum(by_kind.values()),
            "by_kind": dict(by_kind), "committed_interior_words": sum(i["words"] for i in interior), "committed_interior": interior,
            "max_out_bits": max((c["out_bits"] for c in classes), default=0), "classes": classes, "violations": viol}


#: where serving commits each committed interior value today, by (activation family, primitive of the value): the FA2 / FA3 hidden
#: stream's fields (`commit/hidden_stream.py` StreamLayout; S after the softcap), else None -- a new tap
_ATTENTION = ("AttnBlock", "AttnBlockSoftcap", "AttnBlockFA3", "DotBf16", "AttentionHead", "AttentionHeadSoftcap")
_STREAM = {"AmpereBF16TcDot16_v2": "S (scores)", "HopperBF16WgmmaDot16_v1": "S (scores)", "MufuTanh_v1": "S (softcapped scores)",
           "MufuEx2Ftz_v1": "P (f32 probabilities) / ROW rescale", "F2fpBf16_v1": "RP (bf16 probabilities)", "F32Max_v1": "ROW row_max",
           "F32AddFtz_v1": "ROW lane sums / FIN sum", "F32FmaFtz_v1": "ROW lane sums", "Fa2InvSum_v1": "FIN inv_sum", "Fa3InvSum_v1": "FIN inv_sum"}
#: attention values the stream carries only under an opt-in tap, or not yet, by primitive: the tap each needs (ROW word 2 is the
#: kernel's visit index, not max * scale)
_STREAM_NEW = {"F32MulFtz_v1": "FA2 / FA3 tap build (MS plane, `max_scaled`)",
               "GuardNegInfZero_v1": "FA2 / FA3 tap build (ROW word 3, opt-in `GUARDED_MAX_TAP=1`, #95)"}
#: the kernel a new tap goes in, by the Definition family holding the value
_TAP_KERNEL = {"AttnBlock": "FA2 / FA3 tap build (flash_fwd_kernel / hopper mainloop)", "AttnBlockSoftcap": "FA2 tap build (softcap instantiation)",
               "AttentionHead": "FA2 / FA3 tap build (epilogue)", "AttentionHeadSoftcap": "FA2 tap build (softcap epilogue)",
               "RMSNormTriton": "vLLM batch-invariant Triton _rms_norm_kernel", "RMSNormFusedCuda": "vLLM CUDA fused_add_rms_norm_kernel<bf16, 0, true>",
               "MoeRouterTopK": "vLLM _moe_C.topk_softmax (topkGatingSoftmax)", "MoeRouterTopKNorm": "vLLM _moe_C.topk_softmax (topkGatingSoftmax)",
               "MoeRouterProbs": "vLLM _moe_C.topk_softmax (topkGatingSoftmax: the softmax)", "NvExpf": "vLLM _moe_C.topk_softmax (topkGatingSoftmax: expf)",
               "EmbeddingShard": "vLLM's vocabulary-parallel embedding (the rank's vocabulary-range mask)"}


def committed_today(scope: str | None, head: str | None) -> str | None:
    """Where serving commits a committed interior value today (a hidden-stream field), or None: the value needs a new tap."""
    fam = (scope or "").split("{")[0].rsplit("_v", 1)[0]
    if fam in _ATTENTION:
        return _STREAM.get(head or "")
    return None


def tap_kernel(scope: str | None, head: str | None = None) -> str:
    """The kernel a new tap for a committed interior value goes in (`head`: the value's primitive)."""
    fam = (scope or "").split("{")[0].rsplit("_v", 1)[0]
    if fam in _ATTENTION and head in _STREAM_NEW:
        return _STREAM_NEW[head]
    return _TAP_KERNEL.get(fam, f"the {fam} kernel")


#: unit rules by gate limits, then (Definition id, X, W, extras), shared by every check in the process: a row's request Programs share
#: most specializations
_RULES: dict = {}


def _depth(fn: Any, limits: dict[str, int], depth: dict[str, int], fns: dict[str, Any]) -> int:
    """`unit_rule`'s recursion under `fn`, by Definition id into `depth`: 0 for a body cut as a whole, one more than its deepest
    member for a separable body, -1 when there is nothing to cut (a primitive, a body over its gate limit)."""
    got = depth.get(fn.id)
    if got is not None:
        return got
    if isinstance(fn, PrimitiveDefinition):
        d = -1
    elif (sep := separable(fn)) is not None:
        d = 1 + max((_depth(c, limits, depth, fns) for c, _m in sep), default=-1)
    else:
        d = 0 if fn.gates <= gate_limit(fn, limits) else -1
    depth[fn.id], fns[fn.id] = d, fn
    return d


#: the prefetch's work, set before its workers fork (a Definition need not pickle)
_PENDING: list[tuple[Any, int, int, int, dict[str, int]]] = []

#: what `prefetch` budgets a rule's cut at: a body cut as a whole up to 4 KiB a gate (GLM-5.3's MoE expert FP8 GEMMs, whose gathers
#: read every expert's word, peak at 3.8-4.5 KB a gate, 12.8 GB for w13's 3.3M gates; most bodies at about 0.5 KB), over the
#: process's 256 MiB; a separable body's rule combines its members' memoed rules, so the 256 MiB alone
RULE_PEAK_PER_GATE = 4 << 10
RULE_PEAK_BASE = 256 << 20


def _rule_of_pending(i: int) -> dict[str, Any] | str:
    """The unit rule of pending Definition `i`, or what it raised (`repr`): left to `check_calls`, which reports or raises it."""
    fn, X, W, extras, limits = _PENDING[i]
    try:
        return unit_rule(fn, X, W, extras, memo=_RULES[tuple(sorted(limits.items()))], max_gates=limits)
    except Exception as e:  # noqa: BLE001
        return repr(e)


def prefetch(specs: Iterable[str], X: int = 16, W: int = 32, extras: int = 0, *, max_gates: Limits = None, jobs: int = 1,
             budget: int | None = None, specialize: Callable[[str, dict], Any] = specialization) -> int:
    """Memo, `jobs` forked processes at a time, the unit rules `check_calls` would compute one after another for Calls of the
    Definition ids `specs` (`spec_histogram` keys; a `batch:` / `scan:` prefix is dropped): every Definition under them that the
    process has not ruled on yet, the bodies cut as a whole first, then each separable body once its members are memoed.  A rule
    is a function of its Definition, so a check reads what it would have computed; a spec that does not resolve, or a rule that
    raises, is left to `check_calls`.  Nothing is done for `jobs` <= 1 or without `fork`.  Returns how many rules were memoed.

    `budget` (bytes): the cuts in flight together are those whose estimated peaks (`RULE_PEAK_PER_GATE`, `RULE_PEAK_BASE`) fit it,
    the largest that fits first; a cut over it runs alone.  None: `jobs` at a time, whatever they take."""
    import multiprocessing
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    from verity.ml.definitions.call_scope import spec_base, spec_statics
    if jobs <= 1 or "fork" not in multiprocessing.get_all_start_methods():
        return 0
    limits = gate_limits(max_gates)
    memo = _RULES.setdefault(tuple(sorted(limits.items())), {})
    depth: dict[str, int] = {}
    fns: dict[str, Any] = {}
    for s in dict.fromkeys(specs):
        try:
            fn = specialize(spec_base(s), spec_statics(s))
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        _depth(fn, limits, depth, fns)
    for i, fn in fns.items():
        if (i, X, W, extras) not in memo and (hit := _cached_rule(fn, X, W, extras, limits)) is not None:
            memo[(i, X, W, extras)] = hit
    done = 0
    for d in range(max(depth.values(), default=-1) + 1):
        todo = sorted((fns[i] for i, k in depth.items() if k == d and (i, X, W, extras) not in memo), key=lambda fn: -fn.gates)
        if not todo:
            continue
        _PENDING[:] = [(fn, X, W, extras, limits) for fn in todo]
        need = [RULE_PEAK_BASE + (RULE_PEAK_PER_GATE * fn.gates if d == 0 else 0) for fn in todo]
        workers, rules = min(jobs, len(todo)), {}
        try:
            with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("fork")) as ex:
                pending, running, used = list(range(len(todo))), {}, 0
                while pending or running:
                    while pending and len(running) < workers:
                        i = next((i for i in pending if budget is None or used + need[i] <= budget), None if running else pending[0])
                        if i is None:
                            break
                        pending.remove(i)
                        running[ex.submit(_rule_of_pending, i)] = i
                        used += need[i]
                    finished, _ = wait(running, return_when=FIRST_COMPLETED)
                    for f in finished:
                        i = running.pop(f)
                        used -= need[i]
                        rules[i] = f.result()
        finally:
            _PENDING.clear()
        for i, fn in enumerate(todo):
            if isinstance(rules[i], dict):
                memo[(fn.id, X, W, extras)] = rules[i]
                done += 1
    return done


def check_calls(groups: Iterable[dict[str, Any]], X: int = 16, W: int = 32, extras: int = 0, *, strict: bool = True,
                specialize: Callable[[str, dict], Any] = specialization,
                acquired: Mapping[str, int | Callable[[dict], int]] | None = None, max_gates: Limits = None) -> dict[str, Any]:
    """`Q_word_v1{X,W}` over a Program's Calls, grouped: each group {"family", "statics", "definition", "calls", "where",
    "inputs_committed", "outputs_committed"}.  Per group: its Definition's unit rule and the committed-boundary conditions (every
    Value the Call reads and every Value it outputs is committed, else a unit would cross Calls).  Violations are aggregated per
    (Definition, class) with the Calls they cover and the first modules; `strict` raises `QueryRuleViolation` naming them.

    `acquired` (Definition family -> committed interior words per Call a tap acquires, e.g. `query.norm_scales.TAP_WORDS`, or a function
    of the Call's statics for a tap whose words depend on them, `query.router_softmax.TAP_WORDS`): those interior values are committed
    boundaries too.  Each group of such a family reports `acquired_interior_words`, and one whose
    partition commits another number of interior words per Call is an `interior-not-acquired` violation.

    `max_gates` (`gate_limits`): the run's gate limits; the query, and so `res["query"]`, is the same under any.

    A Boolean Program's group is its word link's (`pipeline.build_instances`): the word Definition's rule, units and boundaries,
    and its `bits` Definition's size beside, in each group and the totals (`boolean_gates`, `call_scope.bits_gates`)."""
    limits = gate_limits(max_gates)
    memo = _RULES.setdefault(tuple(sorted(limits.items())), {})
    out: list[dict[str, Any]] = []
    agg: dict[tuple[str, str], dict[str, Any]] = {}

    def note(v: dict[str, Any], g: dict[str, Any]) -> None:
        a = agg.setdefault((v["definition"], v["class"]), dict(v, calls=0, where=[]))
        a["calls"] += int(g.get("calls") or 0)
        if g.get("where") is not None and len(a["where"]) < 8 and g["where"] not in a["where"]:
            a["where"].append(g["where"])

    from verity.ml.definitions.call_scope import bits_gates
    for g in groups:
        try:
            rule = unit_rule(specialize(g["family"], g["statics"]), X, W, extras, memo=memo, max_gates=limits)
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            rule = _finish(g["definition"], 0, [], [], X, W, extras)
            rule["violations"] = [{"definition": g["definition"], "class": "unresolved", "why": f"{type(e).__name__}: {str(e)[:200]}"}]
        v = list(rule["violations"])
        bgates = bits_gates(g, specialize, v)
        if not g.get("inputs_committed", True):
            v.append({"definition": g["definition"], "class": "input-not-committed",
                      "why": "the Call reads a Value interior to its module body: a unit would cross Calls"})
        if not g.get("outputs_committed", True):
            v.append({"definition": g["definition"], "class": "output-not-committed",
                      "why": "the Call's output is interior to its module body (not a required value): no unit can end at it"})
        a = acquired.get(g["family"], 0) if acquired is not None else 0
        tap = int(a(g["statics"]) if callable(a) else a)
        if tap and rule["committed_interior_words"] != tap:
            v.append({"definition": g["definition"], "class": "interior-not-acquired",
                      "why": f"the tap acquires {tap} interior word(s) per Call; the partition commits {rule['committed_interior_words']}"})
        for x in v:
            note(x, g)
        k = int(g.get("calls") or 0)
        out.append({"where": g.get("where"), "definition": g["definition"], "calls": k, "units": rule["units_per_call"] * k,
                    "gates": rule["gates"] * k, "committed_interior_words": rule["committed_interior_words"] * k, "rule": rule,
                    "violations": len(v), **({"acquired_interior_words": min(tap, rule["committed_interior_words"]) * k} if acquired is not None else {}),
                    **({"bits": g["bits"], "boolean_gates": bgates * k} if g.get("bits") else {})})
    viol = sorted(agg.values(), key=lambda a: (-a["calls"], a["definition"], a["class"]))
    totals = ("calls", "units", "gates", "committed_interior_words") + (("acquired_interior_words",) if acquired is not None else ()) \
        + (("boolean_gates",) if any("bits" in g for g in out) else ())
    res = {"query": query_id(X, W, extras), "ok": not viol, "violations": viol, "groups": out, **{k: sum(g.get(k, 0) for g in out) for k in totals}}
    if strict and viol:
        raise QueryRuleViolation(f"{query_id(X, W, extras)}: {len(viol)} violation(s): " + "; ".join(_name(x) for x in viol[:12])
                                 + (" ..." if len(viol) > 12 else ""))
    return res


def _name(x: dict[str, Any]) -> str:
    what = f" (|out(S)| = {x['out_bits']} b over {x['out_gates']} gate(s); limit {x['limit'][0]} b, or one gate of {x['limit'][1]} b)" \
        if "out_bits" in x else ""
    where = f" under {x['where'][0]}" + (f" (+{len(x['where']) - 1})" if len(x["where"]) > 1 else "") if x.get("where") else ""
    return f"{x['definition']}: {x['class']}{what}, {x['calls']} Call(s){where}"


#: the reads `_inputs_committed` and `cut_verdict` take per numpy step
_READS_PER_STEP = 1 << 24


def _inputs_committed(P: Any, committed: frozenset, literals: frozenset[int]) -> list[bool] | None:
    """Per Call of a columnar body: is every Value it reads committed or a literal's?  numpy passes over the read columns, since a
    request Program's attention reads every earlier token's keys and values (B1 4096/512: 3.4e8 reads).  None for any other body."""
    cols = getattr(P, "columns", None)
    if cols is None:
        return None
    n = cols.n
    port_index = {name: i for i, name in enumerate(cols.port_names)}
    keys = np.fromiter(((v.call << 16) | port_index[v.port] for v in committed if v.port in port_index), np.int64)
    keys.sort()
    literal = np.zeros(n, bool)
    literal[np.fromiter(literals, np.int64, len(literals))] = True
    off = np.frombuffer(cols.read_off, cols.read_off.typecode)
    call = np.frombuffer(cols.read_call, cols.read_call.typecode)
    port = np.frombuffer(cols.read_port, cols.read_port.typecode)
    short = np.zeros(n, bool)                            # a Call reading some Value that is neither
    k0 = 0
    while k0 < n:
        k1 = min(n, max(k0 + 1, int(np.searchsorted(off, off[k0] + _READS_PER_STEP, "right")) - 1))
        lo, hi = int(off[k0]), int(off[k1])
        c = call[lo:hi]
        key = (c.astype(np.int64) << 16) | port[lo:hi]
        at = np.minimum(np.searchsorted(keys, key), max(len(keys) - 1, 0))
        ok = keys[at] == key if len(keys) else np.zeros(hi - lo, bool)
        made = c >= 0
        ok |= made & literal[np.where(made, c, 0)]
        short[np.searchsorted(off, np.flatnonzero(~ok) + lo, "right") - 1] = True
        k0 = k1
    return (~short).tolist()


def call_groups(P: Any, Q: Any) -> list[dict[str, Any]]:
    """The Calls of `P` grouped for `check_calls` under the population `Q` (`required.Population`: `Q_module_body_v1` + its
    policy): committed = the required values, and a literal's output (written in the descriptor)."""
    committed = Q.required.values
    literals = Q.literals
    inputs_committed = _inputs_committed(P, committed, literals)
    groups: dict[tuple, dict[str, Any]] = {}
    for c in P.calls:
        if c.id in literals:
            continue
        ins = inputs_committed[c.id] if inputs_committed is not None else all(r in committed or r.call in literals for r in c.reads)
        outs = all((c.id, p.name) in committed for p in c.port_table)
        where = Q.program_path(c.id)
        k = (c.definition, where, ins, outs)
        g = groups.get(k)
        if g is None:
            g = groups[k] = {"family": c.family, "statics": c.statics, "definition": c.definition, "where": where, "calls": 0,
                             "inputs_committed": ins, "outputs_committed": outs, **({"bits": c.bits} if c.bits else {})}
        g["calls"] += 1
    return list(groups.values())


def check_query(P: Any, Q: Any, X: int = 16, W: int = 32, extras: int = 0, *, strict: bool = True,
                acquired: Mapping[str, int] | None = None, max_gates: Limits = None) -> dict[str, Any]:
    """Check the Program `P` against `Q_word_v1{X,W}` refining the population `Q` (see `call_groups`; `acquired`, `max_gates`: see
    `check_calls`); `strict` raises `QueryRuleViolation` naming every violating (Definition, class)."""
    return check_calls(call_groups(P, Q), X, W, extras, strict=strict, acquired=acquired, max_gates=max_gates)


__all__ = ["QUERY_ID", "WIRING", "RULE", "MAX_GATES", "MAX_GATES_ENV", "RULE_CACHE_ENV", "committed_today", "tap_kernel", "QueryRuleViolation",
           "query_id", "parse", "gate_limits", "gate_limit", "specialization", "Graph", "partition", "units", "definition_digest", "use_rule_cache", "unit_rule",
           "RULE_PEAK_PER_GATE", "RULE_PEAK_BASE", "prefetch", "check_calls", "call_groups", "check_query"]
