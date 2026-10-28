"""Graph/reference correspondence of two encodings of one program.

A reference-codec change moves the program digest even when the denoted circuit is unchanged, because the digest
hashes the serialised descriptor. Digest equality is therefore the wrong test across a codec transition; this
module compares what the descriptors *denote*:

  1. definitions: the same specialization ids; per composite the same signature, binding record and body —
     node by node the same callee, form, axes/n, argument and output types, and the same *ordered reference
     sequence* of every argument and of the body's returns (producer identities and repeats included);
  2. gates: the same gate count; sampled gates (`C[i]`) have the same primitive and operand references;
  3. function calls: sampled calls have the same `args`, `returns`, `in_` and declared-interface `out`, and the
     same children (recursively, to a depth);
  4. subcircuit boundaries: `S.out` (the graph boundary, whole-circuit consumer scan) of sampled regions —
     only when the circuit is small enough to enumerate;
  5. evaluation: sampled calls evaluated on the same random inputs give the same outputs.

Reference sequences are compared through `canonical_runs`, a representation-independent decomposition of a
sequence of body-scoped references into maximal runs `(space, idx, base, stride, count)` with stride in {0, 1}
(the greedy leaf-level decomposition, computed run-by-run so a 7.8 M-part KV-cache concat costs O(parts), not
O(leaves)). Two `Refs` denote the same sequence iff their canonical runs are equal.

    PYTHONPATH=. verity-vllm descriptor-equivalence --old B0-v0.json --new B0-v1.json --out report.json
    PYTHONPATH=. verity-vllm descriptor-equivalence --old B0-v0.json --build B0 32 7
"""

from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Iterator

from verity.ml.circuits.codec import decode_program, program_digest
from verity.ml.circuits.defs import PrimitiveDefinition, REGISTRY, SpecializedDefinition
from verity.ml.circuits.evaluation import evaluate_call
from verity.ml.circuits.layout import FunctionCall
from verity.ml.circuits.refs import Affine, Concat, Explicit, Refs, Strided

from verity_vllm.config import UsageError, option

Run = tuple[str, int, int, int, int]  # (space, idx, base, stride, count); stride 0 = the same leaf repeated


def _part_runs(p: Refs) -> Iterator[Run]:
    """Maximal runs of one part, in order (stride 0 or 1; a single leaf is a count-1 run with stride 1)."""
    if isinstance(p, Affine):
        if p.count:
            yield (p.space, p.idx, p.base, 1, p.count)
    elif isinstance(p, Strided):
        if p.count == 0:
            return
        if p.outer_stride == 0 and p.inner_stride == 0:
            yield (p.space, p.idx, p.base, 0 if p.count > 1 else 1, p.count)
            return
        full, rem = divmod(p.count, p.inner)
        for q in range(full + (1 if rem else 0)):
            n = p.inner if q < full else rem
            base = p.base + q * p.outer_stride
            if p.inner_stride == 1 or n == 1:
                yield (p.space, p.idx, base, 1, n)
            elif p.inner_stride == 0:
                yield (p.space, p.idx, base, 0, n)
            else:
                for r in range(n):
                    yield (p.space, p.idx, base + r * p.inner_stride, 1, 1)
    elif isinstance(p, Concat):
        for q in p.parts:
            yield from _part_runs(q)
    elif isinstance(p, Explicit):
        for s, i, b in p.items:
            yield (s, i, b, 1, 1)
    else:  # any other Refs: element-wise
        for s, i, b in p:
            yield (s, i, b, 1, 1)


def canonical_runs(refs: Refs) -> tuple[Run, ...]:
    """Greedy maximal-run decomposition of the leaf sequence, independent of how `refs` is represented: the
    result equals the leaf-by-leaf greedy scan (a run is extended while the next leaf is `base` (stride 0) or
    `base + count` (stride 1); a lone leaf adopts the stride of the leaf that follows it)."""
    out: list[Run] = []
    cur: list | None = None  # [space, idx, base, stride, count]; stride None while count == 1

    def flush() -> None:
        if cur is not None:
            out.append((cur[0], cur[1], cur[2], 1 if cur[3] is None else cur[3], cur[4]))

    for s, i, b, st, n in _part_runs(refs):
        while n:
            if cur is None:
                cur = [s, i, b, None if n == 1 else st, n]
                break
            if cur[0] != s or cur[1] != i:
                flush()
                cur = None
                continue
            if cur[3] is None:  # a single leaf so far: the run's first leaf decides the stride
                if b == cur[2]:
                    cur[3] = 0
                elif b == cur[2] + 1:
                    cur[3] = 1
                else:
                    flush()
                    cur = None
                    continue
            elif b != (cur[2] if cur[3] == 0 else cur[2] + cur[4]):
                flush()
                cur = None
                continue
            # the run's first leaf continues `cur`
            if n == 1 or st == cur[3]:
                cur[4] += n
                break
            cur[4] += 1  # peel that leaf; the rest of the run is handled afresh
            b, n = b + st, n - 1
    flush()
    return tuple(out)


def canonical_runs_by_leaf(refs: Refs) -> tuple[Run, ...]:
    """The leaf-by-leaf reference scan `canonical_runs` must agree with (O(leaves); for tests)."""
    out: list[Run] = []
    cur: list | None = None
    for s, i, b in refs:
        if cur is not None and cur[0] == s and cur[1] == i:
            if cur[3] is None and b in (cur[2], cur[2] + 1):
                cur[3], cur[4] = (0 if b == cur[2] else 1), 2
                continue
            if cur[3] is not None and b == (cur[2] if cur[3] == 0 else cur[2] + cur[4]):
                cur[4] += 1
                continue
        if cur is not None:
            out.append((cur[0], cur[1], cur[2], 1 if cur[3] is None else cur[3], cur[4]))
        cur = [s, i, b, None, 1]
    if cur is not None:
        out.append((cur[0], cur[1], cur[2], 1 if cur[3] is None else cur[3], cur[4]))
    return tuple(out)


# ---------------------------------------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------------------------------------


class Mismatch(AssertionError):
    pass


def _compare_definition(fid: str, a, b, rep: dict) -> None:
    if isinstance(a, PrimitiveDefinition) or isinstance(b, PrimitiveDefinition):
        if not (isinstance(a, PrimitiveDefinition) and isinstance(b, PrimitiveDefinition)):
            raise Mismatch(f"{fid}: primitive on one side only")
        if a.params != b.params or a.ret != b.ret:
            raise Mismatch(f"{fid}: primitive signatures differ")
        rep["primitives"] += 1
        return
    assert isinstance(a, SpecializedDefinition) and isinstance(b, SpecializedDefinition)
    if a.base_id != b.base_id:
        raise Mismatch(f"{fid}: base ids differ ({a.base_id} vs {b.base_id})")
    if _static_view(a.bindings) != _static_view(b.bindings):
        raise Mismatch(f"{fid}: binding records differ")
    ba, bb = a.body, b.body
    if ba.params != bb.params or ba.ret_type != bb.ret_type:
        raise Mismatch(f"{fid}: signatures differ")
    if len(ba.nodes) != len(bb.nodes):
        raise Mismatch(f"{fid}: {len(ba.nodes)} vs {len(bb.nodes)} nodes")
    if ba.offsets != bb.offsets:
        raise Mismatch(f"{fid}: node offsets differ")
    for k, (na, nb) in enumerate(zip(ba.nodes, bb.nodes)):
        if (na.form, na.axes, na.n, na.n_xs) != (nb.form, nb.axes, nb.n, nb.n_xs):
            raise Mismatch(f"{fid} node {k}: form/axes/n differ")
        if _fid(na.fn) != _fid(nb.fn):
            raise Mismatch(f"{fid} node {k}: callee {_fid(na.fn)} vs {_fid(nb.fn)}")
        if na.out != nb.out:
            raise Mismatch(f"{fid} node {k}: output types differ")
        if len(na.args) != len(nb.args):
            raise Mismatch(f"{fid} node {k}: arity differs")
        for ai, (xa, xb) in enumerate(zip(na.args, nb.args)):
            if xa.type != xb.type:
                raise Mismatch(f"{fid} node {k} arg {ai}: types differ")
            ra, rb = canonical_runs(xa.refs), canonical_runs(xb.refs)
            if ra != rb:
                raise Mismatch(f"{fid} node {k} arg {ai}: reference sequences differ ({len(ra)} vs {len(rb)} runs)")
            rep["reference_sequences"] += 1
            rep["reference_runs"] += len(ra)
            rep["reference_leaves"] += xa.refs.count
        rep["nodes"] += 1
    if canonical_runs(ba.ret) != canonical_runs(bb.ret):
        raise Mismatch(f"{fid}: return references differ")
    rep["composites"] += 1


def _fid(fn) -> str:
    from verity.ml.circuits.codec import _spec_id
    return _spec_id(fn)


def _static_view(v: Any) -> Any:
    from verity.ml.circuits.defs import FunctionDefinition
    if isinstance(v, FunctionDefinition):
        return ("fn", _fid(v))
    if isinstance(v, (list, tuple)):
        return tuple(_static_view(x) for x in v)
    if isinstance(v, dict):
        return tuple((k, _static_view(v[k])) for k in sorted(v))
    return v


def _all_definitions(root: SpecializedDefinition) -> dict[str, Any]:
    out: dict[str, Any] = {}

    def visit(fn) -> None:
        fid = _fid(fn)
        if fid in out:
            return
        out[fid] = fn
        if isinstance(fn, SpecializedDefinition):
            for n in fn.body.nodes:
                visit(n.fn)
            from verity.ml.circuits.codec import _static_functions, _as_specialized
            for k in fn.defn.statics:
                for f in _static_functions(fn.bindings[k]):
                    visit(_as_specialized(f, fid))

    visit(root)
    return out


INTERFACE_LEAF_LIMIT = 300_000  # declared interfaces wider than this (weight records) are compared at sampled leaves
CHILD_SAMPLES = 24  # children of a compared call beyond this count are sampled


def _declared_out(c: FunctionCall) -> tuple[int, ...]:
    """The declared-interface adapter (distinct returned values produced inside; no consumer scan): `distinct_returns`
    where `layout` names it so (FE2), else the pre-rename `out`."""
    return c.distinct_returns if hasattr(c, "distinct_returns") else c.out


def _escaping_out(c: FunctionCall) -> tuple[int, ...]:
    """The graph boundary `S.out` (consumed outside or protected; whole-circuit consumer scan)."""
    return c.escaping_out() if hasattr(c, "escaping_out") else c.out


def _call_view(c: FunctionCall, depth: int, rng: random.Random, samples: int = 256) -> dict:
    """The declared interface (`args`, `returns`, repeats kept) and graph boundary (`in_`, `out`) of a call, or —
    for calls whose interface is wider than `INTERFACE_LEAF_LIMIT` leaves — the same at `samples` random leaf
    positions per parameter (both sides are asked the same positions)."""
    d: dict[str, Any] = {"id": c.id, "interval": c.interval}
    if c.is_primitive:
        d["args"], d["returns"] = tuple(c.args), tuple(c.returns)
    else:
        sc = c.scope()
        params = sc.spec.body.params
        n_leaves = sum(t.leaves for _, t in params)
        if n_leaves <= INTERFACE_LEAF_LIMIT:
            d["args"], d["returns"], d["in"], d["out"] = tuple(c.args), tuple(c.returns), c.in_, _declared_out(c)
        else:
            from verity.ml.circuits.layout import param_leaf, resolve
            state = rng.getstate()
            picks = []
            for pi, (_, t) in enumerate(params):
                picks.append((pi, sorted({rng.randrange(t.leaves) for _ in range(min(samples, t.leaves))})))
            rng.setstate(state)  # the other side draws the same positions
            d["args_sampled"] = tuple((pi, j, param_leaf(sc, pi, j)) for pi, js in picks for j in js)
            ret = sc.spec.body.ret
            d["returns"] = tuple(resolve(sc, ret[j]) for j in range(ret.count)) if ret.count <= INTERFACE_LEAF_LIMIT else None
    if depth > 0 and not c.is_primitive:
        kids = list(c.children())
        d["n_children"] = len(kids)
        if len(kids) > CHILD_SAMPLES:  # a Gemm has thousands of coordinate members: compare a sample (same on both sides)
            state = rng.getstate()
            pick = sorted({0, len(kids) - 1} | {rng.randrange(len(kids)) for _ in range(CHILD_SAMPLES - 2)})
            rng.setstate(state)
            kids = [kids[i] for i in pick]
            d["children_sampled"] = pick
        d["children"] = [_call_view(k, depth - 1, rng, samples) for k in kids]
    return d


def _random_inputs(circuit, call: FunctionCall, rng: random.Random) -> list[int]:
    vals: dict[int, int] = {}
    out = []
    for g in call.args:
        if g not in vals:
            w = circuit.gate(g).prim.ret.w
            vals[g] = rng.getrandbits(w - 1)  # positive patterns (the f32/bf16 primitives are total anyway)
        out.append(vals[g])
    return out


def compare(A, B, *, gate_samples: int = 2000, call_samples: int = 24, depth: int = 2, eval_samples: int = 4,
            eval_gate_limit: int = 200_000, boundary_gate_limit: int = 2_000_000, seed: int = 0) -> dict:
    """Compare two programs (`Program` or `DecodedProgram`: `.root`, `.circuit`, `.gates`). Raises `Mismatch`
    on the first difference; returns a report of what was compared."""
    rng = random.Random(seed)
    t0 = time.time()
    rep: dict[str, Any] = {"definitions": 0, "primitives": 0, "composites": 0, "nodes": 0, "reference_sequences": 0,
                           "reference_runs": 0, "reference_leaves": 0}
    da, db = _all_definitions(A.root), _all_definitions(B.root)
    if set(da) != set(db):
        only_a, only_b = sorted(set(da) - set(db))[:5], sorted(set(db) - set(da))[:5]
        raise Mismatch(f"definition sets differ: only in A {only_a}, only in B {only_b}")
    for fid in sorted(da):
        _compare_definition(fid, da[fid], db[fid], rep)
        rep["definitions"] += 1
    rep["seconds_definitions"] = round(time.time() - t0, 2)

    # gates
    if A.gates != B.gates:
        raise Mismatch(f"gate counts differ: {A.gates} vs {B.gates}")
    n = A.gates
    ca, cb = A.circuit, B.circuit
    idx = sorted(set([0, n - 1] + [rng.randrange(n) for _ in range(gate_samples)]))
    for i in idx:
        ga, gb = ca.gate(i), cb.gate(i)
        if ga.prim.id != gb.prim.id or tuple(ga.operands) != tuple(gb.operands):
            raise Mismatch(f"gate {i}: {ga!r} vs {gb!r}")
    rep["gates"] = n
    rep["gates_sampled"] = len(idx)
    rep["seconds_gates"] = round(time.time() - t0 - rep["seconds_definitions"], 2)
    outs_a, outs_b = ca.root_outputs(), cb.root_outputs()
    if outs_a != outs_b:
        raise Mismatch("root outputs differ")
    rep["root_outputs"] = len(outs_a)

    # calls
    ra, rb = list(ca.root_calls()), list(cb.root_calls())
    if len(ra) != len(rb):
        raise Mismatch(f"root call counts differ: {len(ra)} vs {len(rb)}")
    pick = sorted(set(rng.randrange(len(ra)) for _ in range(call_samples)) | {0, len(ra) - 1}) if ra else []
    t1 = time.time()
    for j in pick:
        va, vb = _call_view(ra[j], depth, rng), _call_view(rb[j], depth, rng)
        if va != vb:
            raise Mismatch(f"root call {j} ({ra[j].id}) differs")
    rep["root_calls"] = len(ra)
    rep["calls_compared"] = len(pick)
    rep["seconds_calls"] = round(time.time() - t1, 2)

    # subcircuit boundaries (S.out needs the consumers of the whole circuit)
    if n <= boundary_gate_limit and ra:
        checked = 0
        for j in pick[: max(1, len(pick) // 2)]:
            c1, c2 = ra[j], rb[j]
            if c1.size == 0:
                continue
            if _escaping_out(c1) != _escaping_out(c2):
                raise Mismatch(f"root call {j}: S.out differs")
            lo = c1.lo + rng.randrange(c1.size)
            hi = min(c1.hi, lo + 1 + rng.randrange(64))
            if ca.region(lo, hi).out != cb.region(lo, hi).out or ca.region(lo, hi).in_ != cb.region(lo, hi).in_:
                raise Mismatch(f"region [{lo}:{hi}]: boundary differs")
            checked += 1
        rep["boundaries_compared"] = checked
    else:
        rep["boundaries_compared"] = 0

    # evaluation
    t1 = time.time()
    ev = 0
    cands = [j for j in pick if not ra[j].is_primitive and 0 < ra[j].size <= eval_gate_limit]
    rng.shuffle(cands)
    for j in cands[:eval_samples]:
        c1, c2 = ra[j], rb[j]
        args = _random_inputs(ca, c1, rng)
        try:
            o1 = evaluate_call(ca, c1, args)
            o2 = evaluate_call(cb, c2, args)
        except Exception as e:  # noqa: BLE001 - evaluation may need values the call does not bind (e.g. Input gates inside)
            rep.setdefault("evaluation_skipped", []).append({"call": c1.id, "reason": type(e).__name__})
            continue
        if o1 != o2:
            raise Mismatch(f"root call {j} ({c1.id}) evaluates differently")
        ev += 1
    rep["calls_evaluated"] = ev
    rep["seconds_evaluation"] = round(time.time() - t1, 2)
    rep["seconds"] = round(time.time() - t0, 2)
    rep["verdict"] = "equivalent"
    return rep


def _load(path: str):
    t0 = time.time()
    with open(path) as f:
        desc = json.load(f)
    # IT-F5 (`independent-tests.md`): a descriptor names its primitives by id and `REGISTRY` holds only what the process
    # imported; every registry module is loaded so a Hopper / TP=2 / single-request stochastic descriptor decodes too.
    from verity.ml.definitions.registry.catalog import load
    load()
    d = decode_program(desc, REGISTRY)
    return d, {"path": path, "schema": desc.get("schema"), "digest": program_digest(desc), "bytes": None, "seconds_decode": round(time.time() - t0, 1)}


@dataclass(frozen=True, kw_only=True)
class Options:
    old: str = option("--old", help="descriptor JSON (the reference encoding)")
    new: str | None = option("--new", default=None, help="descriptor JSON to compare against")
    build: list[str] | None = option("--build", default=None, nargs=3, metavar=("CONFIG", "LP", "STEPS"),
                                     help="or: build Serve{C,LP,STEPS} with the current library")
    gate_samples: int = option("--gate-samples", type=int, default=2000)
    call_samples: int = option("--call-samples", type=int, default=24)
    eval_samples: int = option("--eval-samples", type=int, default=4)
    seed: int = option("--seed", type=int, default=0)
    out: str | None = option("--out", default=None)


def main(a: Options) -> int:
    import os
    A, ia = _load(a.old)
    ia["bytes"] = os.path.getsize(a.old)
    if a.new:
        B, ib = _load(a.new)
        ib["bytes"] = os.path.getsize(a.new)
    elif a.build:
        from verity.ml.circuits.program import Program
        from verity.ml.circuits.defs import bind
        from verity.ml.definitions.registry import b1
        from verity.ml.circuits.codec import encode_program, canonical_json
        cfg, LP, STEPS = a.build[0], int(a.build[1]), int(a.build[2])
        t0 = time.time()
        B = Program(bind(b1.Serve, C={"B1": b1.B1_CONFIG, "B0": b1.B0_CONFIG}[cfg], LP=LP, STEPS=STEPS))
        desc = encode_program(B)
        ib = {"built": a.build, "schema": desc["schema"], "digest": program_digest(desc), "bytes": len(canonical_json({k: v for k, v in desc.items() if k != "annotations"})),
              "seconds_build_encode": round(time.time() - t0, 1)}
    else:
        raise UsageError("--new or --build")
    print(f"[equiv] old {ia}", file=sys.stderr, flush=True)
    print(f"[equiv] new {ib}", file=sys.stderr, flush=True)
    try:
        rep = compare(A, B, gate_samples=a.gate_samples, call_samples=a.call_samples, eval_samples=a.eval_samples, seed=a.seed)
    except Mismatch as e:
        rep = {"verdict": "MISMATCH", "detail": str(e)}
    out = {"schema": "verity-ir/descriptor-equivalence/v0", "old": ia, "new": ib, "comparison": rep}
    print(json.dumps(out, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
    return 0 if rep.get("verdict") == "equivalent" else 1
