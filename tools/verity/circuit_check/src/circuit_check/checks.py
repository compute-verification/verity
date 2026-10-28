"""The checks, over one Definition, one subcircuit (a template bound), or one served workload's request Program.

    correspondence   the reference (`verity.ml.circuits.evaluation.evaluate`, the circuits' evaluator, a composite's run body by body by
                     `circuit_check.bodies`; a vLLM Definition's is its registered evaluator) on the edge and random vectors
                     of `circuit_check.vectors`: it must be total (a Definition whose
                     reference raises is partial), and every realization of it must equal it bit for bit: every registered
                     kernel (`verity.ml.circuits.evaluation.self_check`), a template's evaluated target where it is another Definition, and
                     the C-Flock Boolean lowering (`verity_flock.ir_lower`: the whole Definition when it is small, a template's
                     proof blocks otherwise) and, under ``Options.struct_max_gates``, the structural lowering's template
                     (``c-flock:struct_lower``, `struct_lowering`, evaluated from the template); the word Definition a
                     Boolean Definition names
                     (`verity.ml.circuits.boolean.trace.word_of`) must equal the Boolean gates through the word view
                     (``word:<word>``): the Boolean Definition is the definition.  A
                     registered table a Definition reads (`targets.TABLE_PROBES`) is read lazily and probed at chosen
                     entries (`vectors.table_instances`), recorded under ``table`` with the words the vectors read
    ports            a Definition's declared return intervals (`verity.ml.circuits.ports`, PROTOCOL.md §12), which the Build takes
                     as facts about a primitive: its reference on edge and random operands, each leaf of a bounded parameter
                     folded into its bound, returns every leaf within its interval; and a primitive's transfer, on drawn
                     operand intervals within its bounds with every operand leaf folded into its interval, returns within the
                     interval the transfer gives; recorded only for a Definition that declares either
    redundant gates  a value computed twice inside one proof block: the gates `verity.ml.definitions.word.Graph` finds recomputed
                     (same primitive, same operands by value) that `word.units` puts in one proof block, and the ANDs of a
                     Boolean proof block on identical operands; counted, with where (primitive @ activation, or the named
                     subcircuit)
    dead gates       gates no output depends on (`verity.ml.circuits.liveness.dead_gates`, transitive), by primitive @ activation
    partition        `Q_word_v1{X,W,R=no-recompute}` (`word.unit_rule` per Definition, `word.check_query` over a workload's
                     Program): every gate certified by exactly one proof block, reads across proof blocks only through
                     committed values, no value computed in two proof blocks, the width rule
                     (`verity.ml.circuits.partition.validate_unit_cut`'s codes and the word's width classes); recorded: the
                     whole Definition's cut under `Q_word` v2 (its codes, and the values computed in two proof blocks it
                     reports, `recomputed_across`).  A Boolean Call is judged by v2: its `recomputed_across` is recorded,
                     in its entry and the summary, and every other code fails as under v1.  `Q_call` v1 {X = 32} is recorded
                     beside it (`q_call`: proof blocks, committed bits, values computed in two proof blocks; or why it refuses,
                     a warning).
                     Both resolve every parameter leaf, so a Definition over ``partition_max_leaves`` is not partitioned (a
                     warning, ``partition/unchecked``)
    lowering pins    a lowering's AND/XOR/NOT counts and its circuit's digest (``type``, `lowering_type`) against its pin, which
                     must hold both (a piece's, a Boolean Definition's whole lowering's,
                     a template proof block's, in the sidecar of the module that defines it or its file in ``pins/``,
                     `sidecars`; a Boolean Definition's lowering has at most its own And gates, besides its builder's
                     commits), a template proof block's circuit digest against its C-Flock module's `PINS`, and the Boolean
                     export's gate lists checked against the proof block's circuit (`verity_flock.boolean_export`); a gated
                     Definition's circuit digest (``gated``: a Program holds it as one gate naming the digest, such as N2's
                     head check) against its ``digests`` pin

Correctness and invariant violations are failures, each with a name (``"<check>/<code> <definition> ..."``); redundant and dead
gates are warnings with counts (they become failures with the re-baseline's common-subexpression pass).  An unpinned lowering,
and a pin without its lowering's ``type``, is a warning.
"""
from __future__ import annotations

import random
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np

from verity.ml.circuits.evaluation import ScalarReference, evaluate, kernels, self_check
from verity.ml.circuits.evaluation.reference import resolve, signature_of, standalone_call
from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean import is_boolean
from verity.ml.circuits.defs import PrimitiveDefinition

from . import bodies as BD
from . import sidecars as SC
from .vectors import edge_count, flat, instance_args, vectors

SCHEMA = "verity-circuit-check/v1"
PINS = SC.SHARED_PINS
_EXAMPLES = 3
_LIMB = (1 << 64) - 1


@dataclass(frozen=True)
class Options:
    """``vectors`` input vectors per correspondence check (the all-edge ones first), fewer when the reference would spend more
    than ``reference_budget`` gate evaluations on them (never fewer than ``min_vectors``); ``X`` / ``W`` / ``extras`` the word
    query; Definitions over ``partition_max_gates`` gates or ``partition_max_leaves`` parameter leaves are not partitioned, over
    ``lower_max_gates`` gates or about ``lower_max_ands`` ANDs (the piece pins' estimate) not lowered whole; ``input_budget``
    bounds vectors x input leaves, ``lane_budget`` vectors x proof blocks when a template's proof blocks are evaluated;
    a composite of at most ``struct_max_gates`` gates is realized by its structural lowering too (`struct_lowering`; 0, the
    default: none)."""

    vectors: int = 64
    min_vectors: int = 4
    seed: int = 20260926
    reference_budget: int = 400_000
    X: int = 16
    W: int = 32
    extras: int = 0
    partition_max_gates: int = 5_000_000
    partition_max_leaves: int = 1 << 22
    lower_max_gates: int = 256
    lower_max_ands: int = 60_000
    input_budget: int = 1 << 22
    lane_budget: int = 1 << 16
    dead_ids: int = 1 << 17
    struct_max_gates: int = 0


@dataclass
class Findings:
    failures: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)

    def fail(self, check: str, code: str, definition: str, what: str = "", **detail: Any) -> None:
        self.failures.append({"check": check, "code": code, "definition": definition,
                              "name": f"{check}/{code} {definition}" + (f" {what}" if what else ""), **detail})

    def warn(self, check: str, code: str, definition: str, count: int, what: str = "", **detail: Any) -> None:
        self.warnings.append({"check": check, "code": code, "definition": definition, "count": int(count),
                              "name": f"{check}/{code} {definition}: {count}" + (f" {what}" if what else ""), **detail})

    def extend(self, other: "Findings") -> None:
        self.failures += other.failures
        self.warnings += other.warnings


#: the seconds each check has taken in this process, by check, each net of the checks timed inside it (`spent`); `cli` clears
#: it before a target and reports it with the target's result (``seconds_by_check``)
SPENT: Counter = Counter()
_INNER: list[float] = []


@contextmanager
def spent(check: str) -> Iterator[None]:
    t0 = time.perf_counter()
    _INNER.append(0.0)
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        inner = _INNER.pop()
        if _INNER:
            _INNER[-1] += dt
        SPENT[check] += dt - inner


def load_pins() -> dict:
    """Every pin: those in `PINS` (``pins/``) and each sidecar's (`sidecars.load_pins`)."""
    return SC.load_pins(PINS)


def _short(x: Any, n: int = 240) -> str:
    s = repr(x)
    return s if len(s) <= n else s[:n] + "..."


def _where(prim: str, scope: str) -> str:
    return f"{prim} @ {scope}" if scope and scope != prim else prim


# --- correspondence ---------------------------------------------------------------------------------------------------------

def _gates(t: Any) -> int:
    return 1 if isinstance(t, (PrimitiveDefinition, ScalarReference)) else int(t.gates)


def _n_vectors(t: Any, opts: Options, table: int | None = None) -> int:
    params, _ = signature_of(t)
    leaves = sum(pt.leaves for j, pt in enumerate(params) if j != table)
    n = max(opts.min_vectors, min(opts.vectors, opts.reference_budget // max(1, _gates(t))))
    return max(1, min(n, opts.input_budget // max(1, leaves)))


def reference_outputs(t: Any, params: Sequence, arrays: Sequence[np.ndarray]) -> tuple[list[tuple[int, ...] | None], list[dict]]:
    """The reference's output leaves per vector (None where it raises) and the vectors it raised on; a composite's
    reference run body by body (`bodies.evaluate`)."""
    n = arrays[0].shape[0] if arrays else 1
    out: list[tuple[int, ...] | None] = []
    raised: list[dict] = []
    for i in range(n):
        args = instance_args(params, arrays, i)
        try:
            out.append(tuple(BD.evaluate(t, *args)))
        except Exception as e:  # noqa: BLE001  (a partial reference: the finding)
            out.append(None)
            raised.append({"instance": i, "args": _short(args), "error": f"{type(e).__name__}: {str(e)[:160]}"})
    return out, raised


def _probed_correspondence(t: Any, params: Sequence, probe: Any, opts: Options, F: Findings) -> tuple[dict, None, list]:
    """The reference on the vectors of a Definition that reads a registered table: its other parameters' edge and random
    vectors and the probes (`vectors.table_instances`), every vector reading the table lazily.  No kernel takes it."""
    from .vectors import table_instances

    n = _n_vectors(t, opts, probe.param)
    instances, read, table = table_instances(params, probe, n, opts.seed)
    ref: list[tuple[int, ...] | None] = []
    raised: list[dict] = []
    for i, args in enumerate(instances):
        try:
            ref.append(tuple(evaluate(t, *args)))
        except Exception as e:  # noqa: BLE001  (a partial reference: the finding)
            ref.append(None)
            raised.append({"instance": i, "args": _short(args), "error": f"{type(e).__name__}: {str(e)[:160]}"})
    table["words_read"] = len(read)
    other = [pt for j, pt in enumerate(params) if j != probe.param]
    rec: dict[str, Any] = {"vectors": len(instances), "edge_vectors": min(n, edge_count(other)), "reference_raised": len(raised),
                           "realizations": [], "table": table}
    if raised:
        F.fail("correspondence", "reference-not-total", t.id, f"raised on {len(raised)} of {len(instances)} vectors",
               examples=raised[:_EXAMPLES])
    for name in sorted(kernels(t)):
        rec["realizations"].append({"via": f"kernel:{name}", "declines": f"a kernel takes arrays, not the registered table {table['name']!r}"})
    return rec, None, ref


def correspondence(t: Any, opts: Options, F: Findings) -> tuple[dict, list[np.ndarray] | None, list]:
    """The reference on the vectors (totality) and every registered kernel against it; ``None`` for the vectors when the
    Definition reads a registered table (probed, `_probed_correspondence`) or has a parameter over the input budget that is none
    (a failure)."""
    from .targets import table_probe

    params, _ret = signature_of(t)
    probe = table_probe(t)
    if probe is not None:
        return _probed_correspondence(t, params, probe, opts, F)
    wide = [(j, pt.leaves) for j, pt in enumerate(params) if pt.leaves > opts.input_budget]
    if wide:
        F.fail("correspondence", "unprobed-table", t.id, f"parameter {wide[0][0]} has {wide[0][1]} leaves, over the input budget "
               f"{opts.input_budget}, and no registered table probes it (`circuit_check.targets.TABLE_PROBES`)")
        return {"vectors": 0, "edge_vectors": 0, "reference_raised": 0, "realizations": []}, None, []
    n = _n_vectors(t, opts)
    arrays = vectors(params, n, opts.seed)
    ref, raised = reference_outputs(t, params, arrays)
    rec: dict[str, Any] = {"vectors": n, "edge_vectors": min(n, edge_count(params)), "reference_raised": len(raised), "realizations": []}
    if raised:
        if isinstance(t, ScalarReference):
            rec["partial_domain"] = f"{t.id} is a ScalarReference with a declared domain; {len(raised)} of {n} vectors are outside it"
        else:
            F.fail("correspondence", "reference-not-total", t.id, f"raised on {len(raised)} of {n} vectors", examples=raised[:_EXAMPLES])
    for name in sorted(kernels(t)):
        try:
            sc = self_check(t, kernel=name, inputs=arrays)
        except ValueError as e:
            rec["realizations"].append({"via": f"kernel:{name}", "declines": str(e)[:200]})
            continue
        rec["realizations"].append({"via": f"kernel:{name}", "compared": sc.n - sc.declined, "declined": sc.declined,
                                    "mismatches": len(sc.mismatches)})
        if sc.mismatches:
            F.fail("correspondence", "kernel-mismatch", t.id, f"[{name}]: {len(sc.mismatches)} of {sc.n} vectors", via=f"kernel:{name}",
                   examples=[{"instance": i, "args": _short(a), "kernel": _short(k), "reference": _short(r)} for i, a, k, r in sc.mismatches[:_EXAMPLES]])
    return rec, arrays, ref


#: the operand intervals a primitive's transfer is checked on (`port_intervals`)
TRANSFER_DRAWS = 8


def _one_width(t: Any) -> int | None:
    ws = t.width_set()
    return next(iter(ws)) if len(ws) == 1 else None


def _bounds(rs: Sequence, full: Callable) -> list:
    return [(lo, hi, p.bound, w) for lo, hi, p, w in rs if p.bound is not None and not full(p.bound, w)]


def _folded(arrays: list, params: Sequence, pt: Any, n: int, draw: Sequence[tuple[int, int] | None] = ()) -> list:
    """`arrays` with each parameter's leaves folded into its drawn interval (`draw[j]`), else each leaf of a bounded parameter
    into its bound: an interval's two ends first, then the vectors' values within it."""
    from verity.ml.circuits import ports as P

    out = list(arrays)
    for j, rs in enumerate(pt.params):
        a = np.array(out[j]).reshape(n, -1)
        iv = draw[j] if j < len(draw) else None
        runs = [(0, a.shape[1], iv, _one_width(params[j]))] if iv is not None else _bounds(rs, P.full)
        for lo, hi, (b0, b1), w in runs:
            for i in range(n):
                for c in range(lo, hi):
                    v = (b0, b1)[i] if i < 2 else b0 + int(a[i, c]) % (b1 - b0 + 1)
                    a[i, c] = v & ((1 << w) - 1)
        out[j] = a.reshape(out[j].shape)
    return out


def port_intervals(t: Any, opts: Options, F: Findings) -> dict | None:
    """The reference against what `t` declares of its return's intervals (`verity.ml.circuits.ports`): its declared intervals,
    on `vectors` with each leaf of a bounded parameter folded into its bound; a primitive's transfer, on `TRANSFER_DRAWS`
    operand intervals (within the bounds), every leaf of an operand folded into its interval. None when it declares neither."""
    from verity.ml.circuits import ports as P

    pt = P.port_types(t)
    outs = [(lo, hi, o.interval, w) for lo, hi, o, w in (pt.ret if pt is not None else ()) if o.interval is not None]
    if not outs and (pt is None or pt.transfer is None):
        return None
    params, ret = signature_of(t)
    n = _n_vectors(t, opts)
    rec: dict[str, Any] = {}
    if outs:
        ref, _raised = reference_outputs(t, params, _folded(vectors(params, n, opts.seed), params, pt, n))
        outside = [{"instance": i, "leaf": leaf, "value": out[leaf], "interval": list(iv)}
                   for i, out in enumerate(ref) if out is not None
                   for lo, hi, iv, w in outs for leaf in range(lo, hi) if not P.holds(iv, w, out[leaf])]
        if outside:
            F.fail("ports", "interval", t.id, f"{len(outside)} returned leaves outside their declared interval on {n} vectors",
                   examples=outside[:_EXAMPLES])
        rec = {"vectors": n, "intervals": [[lo, hi, list(iv)] for lo, hi, iv, _w in outs], "outside": len(outside)}
    if pt.transfer is not None:
        r, w_ret = random.Random(opts.seed), _one_width(ret)
        drawn = outside_t = 0
        for d in range(TRANSFER_DRAWS):
            draw: list[tuple[int, int] | None] = []
            for pty, rs in zip(params, pt.params):
                w, bs = _one_width(pty), _bounds(rs, P.full)
                whole = bs[0][2] if len(bs) == 1 and bs[0][:2] == (0, pty.leaves) else None
                if w is None or (bs and whole is None) or (not bs and r.random() < 0.25):
                    draw.append(None)
                    continue
                lo, hi = whole or ((-(1 << (w - 1)), (1 << (w - 1)) - 1) if r.random() < 0.5 else (0, (1 << w) - 1))
                x = r.randint(lo, hi)
                y = r.randint(lo, hi) if r.random() < 0.5 else min(hi, x + r.randint(0, 255))
                draw.append((min(x, y), max(x, y)))
            iv = pt.transfer(*draw)
            if iv is None:
                continue
            drawn += 1
            if not (isinstance(iv, (tuple, list)) and len(iv) == 2 and iv[0] <= iv[1] and P.reading(tuple(iv), w_ret) == tuple(iv)):
                F.fail("ports", "transfer", t.id, f"its transfer gives {iv!r} on {draw}, no interval of a Value<{w_ret}>")
                continue
            ref, _raised = reference_outputs(t, params, _folded(vectors(params, n, opts.seed + 1 + d), params, pt, n, draw))
            bad = [{"instance": i, "operands": draw, "value": out[0], "interval": list(iv)}
                   for i, out in enumerate(ref) if out is not None and not P.holds(iv, w_ret, out[0])]
            outside_t += len(bad)
            if bad:
                F.fail("ports", "transfer", t.id, f"{len(bad)} of {n} returns outside the interval its transfer gives on {draw}",
                       examples=bad[:_EXAMPLES])
        rec["transfer"] = {"draws": TRANSFER_DRAWS, "vectors": n, "intervals": drawn, "outside": outside_t}
    return rec


def target_equivalence(target: Any, definition: Any, opts: Options, F: Findings) -> dict | None:
    """A template evaluated through another Definition (``Subcircuit.target``) against its own Definition's reference."""
    from verity.ml.circuits.evaluation import evaluate_batch

    if target is definition or getattr(target, "id", None) == getattr(definition, "id", None):
        return None
    pt, rt = signature_of(target)
    pd, rd = signature_of(definition)
    if [p.leaf_widths() for p in pt] != [p.leaf_widths() for p in pd] or rt.leaf_widths() != rd.leaf_widths():
        return {"target": target.id, "compared": 0, "reason": "the target's leaves are not the Definition's"}
    arrays = vectors(pd, _n_vectors(definition, opts), opts.seed + 1)
    want, _raised = reference_outputs(definition, pd, arrays)
    rows = [i for i, w in enumerate(want) if w is not None]
    got = np.asarray(evaluate_batch(target, *[a[rows].reshape(len(rows), -1) for a in arrays])).reshape(len(rows), -1)
    bad = [(i, got[k].tolist(), want[i]) for k, i in enumerate(rows) if tuple(int(v) for v in got[k]) != want[i]]
    if bad:
        F.fail("correspondence", "target-mismatch", definition.id, f"[{target.id}]: {len(bad)} of {len(rows)} vectors", via=target.id,
               examples=[{"instance": i, "target": _short(g), "reference": _short(w)} for i, g, w in bad[:_EXAMPLES]])
    return {"target": target.id, "compared": len(rows), "mismatches": len(bad)}


def _wide_words(t: Any, words: Sequence[Any], ret: Any, params: Sequence[Any], arrays: Sequence[np.ndarray], n: int) -> np.ndarray:
    """`evaluate_words` through the bits (`verity.ml.circuits.evaluation.bits.evaluate_bits`), in Python ints: each argument split
    into the word Definition's leaves (``words``, ``ret``) and each leaf into its bits, LSB first. For an element wider than its 64
    bits (PoUW's 576-bit sums of squares), and for a port of single bits: `Array<n, Value<1>>` is n one-bit words on the word side
    and, as `boolean.bit_type` leaves it, one n-bit element on bits, so the word view (`Value<n>`) is not the word's signature
    (`DsaWaksmanRows_v2`'s settings)."""
    from verity.ml.circuits.evaluation.batch import type_shape
    from verity.ml.circuits.evaluation.bits import evaluate_bits

    ins = []
    for a, wp, p in zip(arrays, words, params):
        widths = wp.leaf_widths()
        rows = np.asarray(a, dtype=object).reshape(n, len(widths))
        ins.append(np.array([[(int(v) >> j) & 1 for v, w in zip(row, widths) for j in range(w)] for row in rows],
                            dtype=np.uint8).reshape(n, *type_shape(p)))
    bits = np.asarray(evaluate_bits(t, *ins)).reshape(n, -1)
    out = np.zeros((n, ret.leaves), dtype=object)
    at = 0
    for e, w in enumerate(ret.leaf_widths()):
        for i in range(n):
            out[i, e] = sum(int(b) << j for j, b in enumerate(bits[i, at:at + w]))
        at += w
    return out


def word_view(t: Any, opts: Options, F: Findings) -> dict | None:
    """The word Definition a Boolean Definition names (`verity.ml.circuits.boolean.trace.word_of`) against the Boolean one, which is
    the definition: on the word Definition's edge and random vectors, the word Definition must equal the Boolean gates through the
    word view (`verity.ml.circuits.evaluation.bits.evaluate_words`, which evaluates the gates)."""
    if isinstance(t, (PrimitiveDefinition, ScalarReference)) or not is_boolean(t):
        return None
    from verity.ml.circuits.evaluation import evaluate_batch
    from verity.ml.circuits.evaluation.bits import evaluate_words
    from verity.ml.circuits.boolean.trace import word_of

    try:
        word = word_of(t)
    except KeyError:
        return None
    pw, rw = signature_of(word)
    arrays = vectors(pw, 16 * opts.vectors, opts.seed + 2)
    n = arrays[0].shape[0] if arrays else 1
    pb, rb = signature_of(t)
    single_bits = [BL.word_type(p) for p in (*pb, rb)] != [*pw, rw] and [BL.bit_type(p) for p in (*pw, rw)] == [*pb, rb]
    if not pw:
        # a batch evaluator sizes its batch by its arguments, so one without any (a constant) is evaluated as one vector
        bits, at = evaluate(t), np.cumsum([0, *rw.leaf_widths()])
        boolean = np.array([[sum(int(b) << j for j, b in enumerate(bits[a:z])) for a, z in pairwise(at)]], dtype=object)
        got = np.asarray([evaluate(word)], dtype=object)
    elif single_bits or max(w for p in (*pb, rb) for w in BL.elements(p)) > 64:
        boolean = _wide_words(t, pw, rw, pb, arrays, n)
        got = np.asarray(evaluate_batch(word, *arrays), dtype=object).reshape(n, -1)
    else:
        boolean = np.asarray(evaluate_words(t, *arrays), dtype=np.uint64).reshape(n, -1)
        got = np.asarray(evaluate_batch(word, *arrays), dtype=np.uint64).reshape(n, -1)
    bad = np.flatnonzero((got != boolean).any(axis=1)).tolist()
    if bad:
        F.fail("correspondence", "word-mismatch", t.id, f"[{word.id}]: {len(bad)} of {n} vectors", via=f"word:{word.id}",
               examples=[{"instance": i, "args": _short(instance_args(pw, arrays, i)), "word": _short(got[i].tolist()),
                          "boolean": _short(boolean[i].tolist())} for i in bad[:_EXAMPLES]])
    return {"via": f"word:{word.id}", "compared": n, "mismatches": len(bad)}


# --- Boolean circuits -------------------------------------------------------------------------------------------------------

def redundant_ands(C: Any, owner: Callable[[int], str] | None = None) -> tuple[int, Counter]:
    """Live ANDs of the `gf2.Circuit` ``C`` whose operands, with earlier duplicates replaced by their first copy, are those of an
    earlier AND (in either order), are equal, or are constant: values computed again.  A commit (``A = form, B = 1``,
    `forms.Circuit.commit`'s row, which keeps a carried form from growing) is a copy, not a product: it counts only when
    its form, so replaced, is a constant or one bit (free as a form) or was committed before.  Returns (count, count by
    ``owner(bit)``)."""
    live = C.live()
    rep: dict[int, int] = {}
    seen: dict[tuple, int] = {}
    copies: dict[tuple, int] = {}
    by: Counter = Counter()

    def canon(f):
        par: dict[int, int] = {}
        for x in f.s:
            y = rep.get(x, x)
            par[y] = par.get(y, 0) ^ 1
        return tuple(sorted(k for k, p in par.items() if p)), f.c

    n = 0
    for v, kind in enumerate(C.kind):
        if kind != "and" or v not in live:
            continue
        a, b = C.data[v]
        ka = canon(a)
        if not b.s and b.c:
            if len(ka[0]) > 1 and ka not in copies:
                copies[ka] = v
                continue
            if ka in copies:
                rep[v] = copies[ka]
            elif len(ka[0]) == 1 and not ka[1]:
                rep[v] = ka[0][0]
            n += 1
            by[owner(v) if owner else "circuit"] += 1
            continue
        kb = canon(b)
        if not ka[0] or not kb[0] or ka[0] == kb[0]:
            n += 1
            by[owner(v) if owner else "circuit"] += 1
            continue
        key = (ka, kb) if ka <= kb else (kb, ka)
        u = seen.get(key)
        if u is None:
            seen[key] = v
            continue
        rep[v] = u
        n += 1
        by[owner(v) if owner else "circuit"] += 1
    return n, by


def _owner_names(root: Any, n: int) -> Callable[[int], str]:
    """The innermost named subcircuit (`boolean_export` trace) that made each committed bit."""
    from verity_flock import boolean_export as BE

    own: list[Any] = [None] * n

    def visit(node):
        for k in node.kids:
            visit(k)
        for v in range(node.start, min(node.end, n)):
            if own[v] is None:
                own[v] = node
    visit(root)
    return lambda v: BE.DISPLAY.get(own[v].fn, own[v].fn) if own[v] is not None and own[v] is not root else "unit"


def lowering_type(C: Any) -> str:
    """A lowering's digest pin (a pin's ``type``): the digest of the gf2 circuit ``C`` as a flat ``program`` circuit type
    (`circuit_type`; ``SHA-512("verity/circuit-type/v1" ‖ 0 ‖ canon(content))``, which Lean's `Flock.CircuitType.digest`
    computes from the same content), so it reads nothing but ``C``."""
    return circuit_type(C).digest()


def circuit_type(C: Any) -> Any:
    """The gf2 circuit ``C`` as a flat ``program`` circuit type (`verity_flock.circuit_types.from_gf2`), validated. Its input
    ports are ``C``'s named inputs in order, ``[1, bits]`` each; its items the ANDs and table reads the outputs and assertions
    reach, in the order ``C`` made them; its output ports ``C``'s named outputs, ``[1, bits]`` each, then its assertion forms
    (each 0 on an accepted input) as one port ``[n, 1]`` when it has any; a table is named by the SHA-512 of its words."""
    from verity_flock import circuit_types as CT
    from verity_flock import tail_pieces as TP
    from verity_flock import type_trace as TT

    ins = [w for w in C.inputs.values() if w]
    outs = [w for w in C.outputs.values() if w]
    asserts = [f for _name, f in C.asserts]
    tables = {name: (TT.table_sha512(name), TP.TABLE_BITS[name], TP.TABLE_VALUE_BITS.get(name, TP.VALUE_BITS))
              for name, _idx, _width in C.lookups}
    t = CT.from_gf2(C, [b for w in ins for b in w], [b for w in outs for b in w] + asserts, [(1, len(w)) for w in ins],
                    [(1, len(w)) for w in outs] + ([(len(asserts), 1)] if asserts else []), "program", tables)
    CT.validate(t, lambda d: None)
    return t


def _shown(v: Any) -> Any:
    return f"{v[:16]}..." if isinstance(v, str) and len(v) > 16 else v


def _counts_against_pin(kind: str, key: str, got: dict, pins: dict, F: Findings, definition: str) -> dict:
    """``got`` (the lowering's counts and its ``type``, `lowering_type`) against its pin: a pin that differs in any field it holds
    is refused by name. A pin without the lowering's ``type`` is a warning (``untyped``), as an unpinned lowering is, since a
    branch written before ``type`` lands with count-only pins; `--update-pins` writes it."""
    pinned = pins.get(kind, {}).get(key)
    rec = {"pin": key, "counts": got, "pinned": pinned}
    if pinned is None:
        F.warn("lowering", "unpinned", definition, 1, f"[{key}]: no pin", pins=kind, pin=key, counts=got)
        rec["status"] = "unpinned"
    elif diff := {k: [pinned[k], v] for k, v in got.items() if k in pinned and pinned[k] != v}:
        F.fail("lowering", "pin-mismatch", definition, f"[{key}]: " + ", ".join(f"{k} {_shown(a)} -> {_shown(b)}" for k, (a, b) in diff.items())
               + " (`circuit-check --update-pins` pins what it measures; review the diff)", pins=kind, pin=key, counts=got, pinned=pinned)
        rec["status"] = "mismatch"
    elif "type" in got and "type" not in pinned:
        F.warn("lowering", "untyped", definition, 1, f"[{key}]: no type (`circuit-check --update-pins` pins it)", pins=kind,
               pin=key, counts=got)
        rec["status"] = "untyped"
    else:
        rec["status"] = "ok"
    return rec


_UNPINNED_ANDS = 2_000


def _pieces_used(t: Any, pins: dict) -> tuple[Counter, Counter, int]:
    """(pieces used, primitives without a piece, estimated ANDs) of ``t``'s gates that are not constant (a gate whose operands
    are all constant is evaluated, as `ir_lower.lower_gates` does), counted body by body (`bodies.nonconstant`); the
    estimate from the piece pins."""
    from verity_flock import ir_lower as IL

    used, missing, est = Counter(), Counter(), 0
    live = BD.nonconstant(t)
    if live is not None:
        for pid, k in live.items():
            if pid in IL.PIECES:
                used[pid] = k
                est += k * pins.get("pieces", {}).get(pid, {}).get("and", _UNPINNED_ANDS)
            else:
                missing[pid] = k
        return used, missing, est
    _prog, call = standalone_call(t)
    const: dict[int, bool] = {}
    for g in call:
        fam = getattr(g.prim, "family", None)
        if fam == "Input":
            continue
        c = fam == "Const" or not g.prim.params or (bool(g.operands) and all(const.get(o, False) for o in g.operands))
        const[g.index] = c
        if c:
            continue
        if g.prim.id in IL.PIECES:
            used[g.prim.id] += 1
            est += pins.get("pieces", {}).get(g.prim.id, {}).get("and", _UNPINNED_ANDS)
        else:
            missing[g.prim.id] += 1
    return used, missing, est


def _piece_status(pid: str, opts: Options, pins: dict, memo: dict) -> None:
    """Record in ``memo["pieces"]`` how many vectors the piece of the primitive ``pid`` gets wrong."""
    from verity.ml.circuits.defs import REGISTRY

    prim = REGISTRY.defs[pid]
    params, _ = signature_of(prim)
    arrays = vectors(params, _n_vectors(prim, opts), opts.seed)
    ref, _raised = reference_outputs(prim, params, arrays)
    whole_lowering(prim, params, arrays, ref, opts, Findings(), pins, memo)


def _topp_word(t: Any, arrays: Sequence[np.ndarray], ref: list, F: Findings) -> dict:
    """The top-p keep word through ``verity_flock.topp_word``: a composition of pieces too large to build as one circuit, every
    piece instance evaluated on the correspondence vectors."""
    from verity_flock import topp_word as TW

    rows = [i for i, r in enumerate(ref) if r is not None]
    if not rows:
        return {"lowered": True, "via": "c-flock:topp_word", "compared": 0}
    fin = flat(arrays)[rows]
    V = fin.shape[1] - 2
    keep, _E = TW.evaluate(fin[:, :V], fin[:, V], fin[:, V + 1])
    got = [int.from_bytes(np.packbits(k, bitorder="little").tobytes(), "little") for k in keep]
    bad = [(i, g, ref[i][0]) for i, g in zip(rows, got) if g != ref[i][0]]
    if bad:
        F.fail("lowering", "lowering-mismatch", t.id, f"[topp_word]: {len(bad)} of {len(rows)} vectors", via="c-flock:topp_word",
               examples=[{"instance": i, "lowering": hex(g), "reference": hex(r)} for i, g, r in bad[:_EXAMPLES]])
    return {"lowered": True, "via": "c-flock:topp_word", "compared": len(rows), "mismatches": len(bad)}


def whole_lowering(t: Any, params: Sequence, arrays: Sequence[np.ndarray], ref: list, opts: Options, F: Findings, pins: dict,
                   memo: dict | None = None) -> dict:
    """The C-Flock lowering of the whole Definition (a primitive's piece, or a small composite piece by piece) against the
    reference on the correspondence vectors; its redundant ANDs; a primitive's AND count and digest against the piece pins.  A composite's
    mismatch names the pieces that already differ from their own primitive (``memo["pieces"]``: piece -> mismatching vectors)."""
    from verity_flock import gf2
    from verity_flock import ir_lower as IL

    bad_pieces = (memo if memo is not None else {}).setdefault("pieces", {})
    used: Counter = Counter()
    if isinstance(t, ScalarReference):
        return {"lowered": False, "reason": "a ScalarReference has no IR body"}
    if isinstance(t, PrimitiveDefinition):
        if not t.params:
            return {"lowered": False, "reason": "a constant: circuit structure, no gates"}
        if t.id.startswith("TopPMaskWordx") and t.id not in IL.PIECES:
            return _topp_word(t, arrays, ref, F)
        if t.id not in IL.PIECES:
            return {"lowered": False, "reason": f"C-Flock has no piece for {t.id}"}
    elif t.gates > opts.lower_max_gates and not is_boolean(t):
        return {"lowered": False, "reason": f"{t.gates} IR gates: lowered as a template's units, not whole (budget {opts.lower_max_gates})"}
    else:
        used, missing, est = _pieces_used(t, pins)
        if missing:
            return {"lowered": False, "reason": "C-Flock has no piece for " + ", ".join(sorted(missing))}
        if est > opts.lower_max_ands:
            return {"lowered": False, "reason": f"about {est} ANDs: over the whole-Definition budget ({opts.lower_max_ands})"}
        for p in used:
            if p not in bad_pieces:
                _piece_status(p, opts, pins, memo if memo is not None else {})
    widths = [w for pt in params for w in pt.leaf_widths()]
    if any(w > 64 for w in widths):
        return {"lowered": False, "reason": "a leaf wider than 64 bits"}
    C = gf2.Circuit()
    ins = [C.input(f"in{j}", w) for j, w in enumerate(widths)]
    try:
        outs = IL.lower(C, t, ins)
    except (IL.NoPiece, NotImplementedError) as e:
        return {"lowered": False, "reason": str(e)[:200]}
    for k, o in enumerate(outs):
        C.output(f"out{k}", o)
    counts = C.counts()
    rows = [i for i, r in enumerate(ref) if r is not None]
    fin = (flat(arrays) if arrays else np.zeros((len(ref), 0), np.uint64))[rows]
    T = len(rows)
    rec: dict[str, Any] = {"lowered": True, "via": "c-flock:ir_lower", "ands": counts["ands"], "compared": T}
    if T:
        got_s, fail = C.evaluate({f"in{j}": IL.slice_words(fin[:, j], w) for j, w in enumerate(widths)}, T)
        # a lane holds 64 bits, so an output leaf wider than that is compared limb by limb, low limb first
        cols = [(k, lo) for k, o in enumerate(outs) for lo in range(0, max(len(o), 1), 64)]
        got = np.stack([IL.unslice_words(got_s[f"out{k}"][lo:lo + 64], T) for k, lo in cols], axis=1)
        want = np.asarray([[ref[i][k] if len(outs[k]) <= 64 else ref[i][k] >> lo & _LIMB for k, lo in cols] for i in rows],
                          dtype=np.uint64)
        bad = np.flatnonzero((got != want).any(axis=1)).tolist()
        rec["mismatches"] = len(bad)
        if isinstance(t, PrimitiveDefinition):
            bad_pieces[t.id] = len(bad)
        if bad:
            causes = sorted(p for p in used if bad_pieces.get(p))
            F.fail("correspondence", "lowering-mismatch", t.id, f"[c-flock]: {len(bad)} of {T} vectors"
                   + (f" (the pieces {', '.join(causes)} differ from their primitives)" if causes else ""), via="c-flock:ir_lower",
                   causes=causes, examples=[{"instance": rows[i], "inputs": _short(fin[i].tolist()), "lowered": _short(got[i].tolist()),
                                             "reference": _short(want[i].tolist())} for i in bad[:_EXAMPLES]])
    red, _by = redundant_ands(C)
    rec["redundant_ands"] = red
    if red:
        F.warn("redundant-gates", "boolean", t.id, red, "ANDs on identical operands in its C-Flock lowering")
    if isinstance(t, PrimitiveDefinition):
        rec["pin"] = _counts_against_pin("pieces", t.id, {"and": counts["ands"], "type": lowering_type(C)}, pins, F, t.id)
    elif is_boolean(t):
        from verity.ml.circuits.evaluation.bits import compile as compile_bits

        own = rec["definition_ands"] = compile_bits(t).counts["and"]
        commits = rec["commits"] = sum(1 for v in C.live() if C.kind[v] == "and" and not C.data[v][1].s)
        if counts["ands"] - commits > own:
            F.fail("lowering", "ands-added", t.id, f"[c-flock]: {counts['ands'] - commits} ANDs (and {commits} commits) for the "
                   f"Definition's {own} And gates", via="c-flock:ir_lower", ands=counts["ands"], commits=commits, definition_ands=own)
        rec["pin"] = _counts_against_pin("definitions", t.id, {"and": counts["ands"], "type": lowering_type(C)}, pins, F, t.id)
    return rec


STRUCT_VIA = "c-flock:struct_lower"


def struct_lowering(t: Any, params: Sequence, arrays: Sequence[np.ndarray], ref: list, F: Findings) -> dict:
    """The whole composite as `verity_flock.struct_lower` lowers it (each specialized Definition once, its calls, batches and
    scans kept as calls: the ``flock-ir-tmpl/v2`` template a structural stager writes), against the reference on the
    correspondence vectors, evaluated from the template (`tmpl.evaluate`, never unrolled): input column ``i`` of the template
    is bit ``i`` of the parameters' leaves concatenated, its one output port the return leaves concatenated. The template
    must be in the order the Lean reader checks (`tmpl.ordered`). Records the unrolled rows (`tmpl.totals`) and the
    template's own size."""
    from verity_flock import struct_lower as SL
    from verity_flock import tmpl as TM

    if isinstance(t, (ScalarReference, PrimitiveDefinition)):
        return {"via": STRUCT_VIA, "lowered": False, "reason": "a structural lowering lowers a composite's body"}
    widths = [w for pt in params for w in pt.leaf_widths()]
    if any(w > 64 for w in widths):
        return {"via": STRUCT_VIA, "lowered": False, "reason": "a leaf wider than 64 bits"}
    try:
        tm, _lw, _top = SL.lower(t, name=t.id)
    except (KeyError, NotImplementedError) as e:
        return {"via": STRUCT_VIA, "lowered": False, "reason": str(e)[:200]}
    tot, order = TM.totals(tm), TM.ordered(tm)
    rec: dict[str, Any] = {"via": STRUCT_VIA, "lowered": True, "unrolled_rows": tot["rows"],
                           "template": {"defs": len(tm.defs), "rows": sum(len(d.rows) for d in tm.defs),
                                        "calls": sum(len(d.calls) for d in tm.defs)}, "ordered": bool(order["ok"])}
    if not order["ok"]:
        F.fail("lowering", "template-unordered", t.id, f"[{STRUCT_VIA}]: {order}", via=STRUCT_VIA)
        return rec
    rows = [i for i, r in enumerate(ref) if r is not None]
    rec["compared"] = len(rows)
    if not rows:
        return rec
    fin = (flat(arrays) if arrays else np.zeros((len(ref), 0), np.uint64))[rows]
    bits = np.concatenate([(fin[:, j:j + 1] >> np.arange(w, dtype=np.uint64)) & np.uint64(1) for j, w in enumerate(widths)]
                          + [np.zeros((len(rows), 0), np.uint64)], axis=1).astype(np.uint8)
    out = TM.unslice_lanes(TM.evaluate(tm, TM.slice_lanes(bits)), len(rows))
    _p, ret = signature_of(t)
    at = np.cumsum([0, *ret.leaf_widths()])
    got = [[int.from_bytes(np.packbits(o[a:b], bitorder="little").tobytes(), "little") for a, b in pairwise(at)] for o in out]
    want = [[int(x) for x in ref[i]] for i in rows]
    bad = [i for i in range(len(rows)) if got[i] != want[i]]
    rec["mismatches"] = len(bad)
    if bad:
        F.fail("correspondence", "lowering-mismatch", t.id, f"[{STRUCT_VIA}]: {len(bad)} of {len(rows)} vectors", via=STRUCT_VIA,
               examples=[{"instance": rows[i], "inputs": _short(fin[i].tolist()), "lowered": _short(got[i]),
                          "reference": _short(want[i])} for i in bad[:_EXAMPLES]])
    return rec


# --- the circuit: redundant gates, dead gates, the partition ------------------------------------------------------------------

def redundant_gates(fn: Any, opts: Options, F: Findings, rule: dict, graph: tuple | None = None, within_units: bool = True) -> dict:
    """Values computed twice (`word.Graph.recomputed`: same primitive, same operands by value), by where: inside one proof block
    of a Call's cut (``within_units``; across proof blocks they are recomputes, a partition failure), or anywhere in a Definition that is
    not a Call itself."""
    n = int(rule.get("redundant_gates") or 0)
    rec: dict[str, Any] = {"count": n, "scope": "unit" if within_units else "definition"}
    if n and graph is not None:
        G, R = graph
        own = R["owner"]
        where = Counter(_where(G.prim_names[int(G.prim[a])], G.scope_names[int(G.scope[a])])
                        for a, b in G.recomputed if own[a] == own[b] or not within_units)
        rec["where"] = dict(where.most_common())
    if n:
        F.warn("redundant-gates", "ir", fn.id, n, "values computed again inside their own " + ("unit" if within_units else "Definition"),
               where=rec.get("where", {}))
    return rec


def dead_gates(fn: Any, opts: Options, F: Findings) -> dict:
    from verity.ml.circuits.liveness import dead_gates as dead

    prog, _call = standalone_call(fn)
    rep = dead(prog, limit=opts.dead_ids, transitive=True)
    c = prog.circuit
    where = Counter()
    for g in rep.ids:
        gate = c.gate(g)
        if getattr(gate.prim, "family", None) != "Input":
            where[_where(gate.prim.id, gate.scope.spec.id)] += 1
    rec = {"count": rep.computed, "unread_inputs": rep.input, "where": dict(where.most_common()), "where_complete": rep.ids_complete,
           "approximate": rep.approximate}
    if rep.computed:
        F.warn("dead-gates", "ir", fn.id, rep.computed, "gates no output depends on", where=rec["where"], unread_inputs=rep.input)
    return rec


def partition(fn: Any, opts: Options, F: Findings, memo: dict, applies: bool = True, q_word: int = 1) -> tuple[dict, dict]:
    """`Q_word_v1` on one Call of ``fn`` (its record and `word.unit_rule`).  Where the invariant applies (``fn`` is a Call of a
    served Program) every violation fails by name; elsewhere the cut of ``fn`` alone is recorded (``isolated``), since its gates
    are partitioned inside the Call that contains it.  With ``q_word=2`` (a Boolean Call: `Q_word` v2) a value computed in two
    proof blocks is no failure: ``recomputed_across`` records how many gates are (the whole cut's v2 view, else
    `word.unit_rule`'s count;
    None when ``fn`` is over the partition budget and unchecked), and every other code fails as under v1."""
    from verity.ml.definitions import word as W

    P = F if applies else Findings()
    v2 = applies and q_word == 2
    across_rule = 0
    rule = W.unit_rule(fn, opts.X, opts.W, opts.extras, memo=memo, max_gates=opts.partition_max_gates)
    rec: dict[str, Any] = {"applies": applies, "query": rule["query"], "units_per_call": rule["units_per_call"], "by_kind": rule["by_kind"],
                           "committed_interior_words": rule["committed_interior_words"], "max_out_bits": rule["max_out_bits"],
                           "free_gates": rule.get("free_gates")}
    for v in rule["violations"]:
        d = v.get("definition", fn.id)
        if v["class"] == "cut":
            for code in v["codes"]:
                det = v["detail"].get(code, {})
                if v2 and code == "gate-recomputed":
                    across_rule += int(det.get("n", 0)) if isinstance(det, dict) else 0
                    continue
                P.fail("partition", code, d, f"({det.get('n', '?')} gate(s))" if isinstance(det, dict) else "",
                       gates=det.get("gates") if isinstance(det, dict) else det)
        elif v["class"] == "too-large":
            rec["checked"] = False
            rec["reason"] = f"{v['gates']} gates: over the partition budget {v['limit']}"
        else:
            P.fail("partition", "width", d, f"{v['class']}: |out(S)| = {v['out_bits']} b over {v['out_gates']} gate(s), limit "
                   f"{v['limit'][0]} b or one gate of {v['limit'][1]} b", unit_class=v["class"], units=v.get("units"))
    rec.setdefault("checked", True)
    if fn.gates <= opts.partition_max_gates:
        # `unit_rule` partitions a body whose nodes read only its parameters node by node, so it never compares two nodes;
        # the whole Definition's own cut does (a value two nodes compute is a recompute across proof blocks)
        G = W.Graph(fn)
        R = W.units(G, opts.X, opts.W, opts.extras)
        cut = R["cut"]
        seen = {f["code"] for f in P.failures if f["check"] == "partition" and f["definition"] == fn.id}
        for code in cut.codes:
            det = cut.detail.get(code, {})
            if code not in seen and not (v2 and code == "gate-recomputed"):
                P.fail("partition", code, fn.id, f"({det.get('n', '?')} gate(s))", gates=det.get("gates"))
        rule = dict(rule, redundant_gates=cut.detail["redundant_gates"] if applies else len(G.recomputed))
        # `Q_word` v2 reports what v1 refuses as gate-recomputed and checks every other clause as v1 does
        rec["whole"] = {"units": len(R["kinds"]), "redundant_gates": cut.detail["redundant_gates"], "codes": list(cut.codes),
                        "q_word_v2": {"codes": [c for c in cut.codes if c != "gate-recomputed"],
                                      "recomputed_across": cut.detail.get("gate-recomputed", {}).get("n", 0)}}
        rec["_graph"] = (G, R)
    if v2:
        rec["q_word"] = 2
        rec["recomputed_across"] = (rec["whole"]["q_word_v2"]["recomputed_across"] if "whole" in rec
                                    else across_rule if rec["checked"] else None)
    elif applies:
        rec["q_word"] = 1
    rec["q_call"] = q_call(fn, opts, P, memo)
    if not applies:
        rec["isolated"] = [f["name"] for f in P.failures]
    return rec, rule


def q_call(fn: Any, opts: Options, F: Findings, memo: dict) -> dict:
    """`Q_call{X = cut.CALL_X}` v1 on one Call of ``fn``: its proof blocks, committed gates and bits, and the values two proof blocks compute
    (reported, as `Q_word` v2 does); or why the query refuses it (``inapplicable``: a gate wider than X, a warning).  A code of
    the cut's own check fails by name (``partition/q-call-<code>``); none is expected, since the cut derives what it checks."""
    from verity.ml.circuits import cut

    X = cut.CALL_X
    if fn.gates > opts.partition_max_gates:
        return {"X": X, "checked": False}
    with spent("q-call"):
        try:
            c = cut.evaluate_call(fn, X, memo)
        except cut.QueryInapplicable as e:
            F.warn("partition", "q-call-inapplicable", fn.id, 1, e.detail.get("why", ""), detail=e.detail)
            return {"X": X, "checked": True, "inapplicable": e.detail}
        G = cut.call_gates(fn, memo)
        com = cut.call_committed(G, c)
        v = cut.check_call_cut(G, c.owner(), X)
    for code in v.codes:
        det = v.detail.get(code, {})
        F.fail("partition", f"q-call-{code}", fn.id, f"({det.get('n', '?')} gate(s))", gates=det.get("gates"))
    return {"X": X, "checked": True, "units": len(c.units), "committed": len(com), "committed_bits": sum(G.width[G.keep[g]] for g in com),
            "recomputed_across": v.detail["recomputed_across"]["n"], "redundant_gates": v.detail["redundant_gates"], "codes": list(v.codes)}


# --- one Definition ---------------------------------------------------------------------------------------------------------

def check_definition(fn: Any, opts: Options = Options(), *, memo: dict | None = None, pins: dict | None = None,
                     lower_whole: bool = True, as_call: bool | None = None) -> dict:
    """Every check on one Definition (a primitive, a bound composite) or `ScalarReference`: its entry, with its failures and
    warnings.  ``memo`` shares the word query's `word.unit_rule` results and the entries across calls;
    ``lower_whole=False`` leaves the Boolean lowering to a template's proof blocks; ``as_call`` says whether the partition
    invariant applies (default: its family is a Call of a served Program, `circuit_check.targets.is_call`).  A Boolean Definition held as a Call is judged by `Q_word` v2
    (`partition`), a word Definition by v1.  A by-calls root (`circuit_check.targets.is_by_calls`) is checked by its calls
    alone (`circuit_check.by_calls`)."""
    from .targets import is_by_calls

    t = resolve(fn)
    if is_by_calls(t):
        from . import by_calls

        return by_calls.check(t)
    memo = {} if memo is None else memo
    pins = load_pins() if pins is None else pins
    done = memo.setdefault("entries", {})
    if as_call is None:
        from .targets import family, is_call

        as_call = is_call(family(t))
    if (t.id, lower_whole, as_call) in done:
        return done[(t.id, lower_whole, as_call)]
    F = Findings()
    kind = "reference" if isinstance(t, ScalarReference) else "primitive" if isinstance(t, PrimitiveDefinition) else "composite"
    entry: dict[str, Any] = {"definition": t.id, "form": kind, "gates": None if kind == "reference" else _gates(t)}
    with spent("correspondence"):
        corr, arrays, ref = correspondence(t, opts, F)
    params, _ = signature_of(t)
    with spent("lowering"):
        if arrays is None:
            low = {"lowered": False, "reason": "a lowering takes every input leaf, and this Definition's correspondence reads a "
                   "registered table lazily" if "table" in corr else "no vectors"}
        elif lower_whole or kind != "composite":
            low = whole_lowering(t, params, arrays, ref, opts, F, pins, memo)
        else:
            low = {"lowered": False, "reason": "lowered as its template's units"}
        corr["realizations"].append({"via": "c-flock:ir_lower", **low})
        if kind == "composite" and arrays is not None and t.gates <= opts.struct_max_gates:
            corr["realizations"].append(struct_lowering(t, params, arrays, ref, F))
    with spent("word-view"):
        wv = word_view(t, opts, F)
    if wv is not None:
        corr["realizations"].append(wv)
    entry["correspondence"] = corr
    with spent("ports"):
        ports = port_intervals(t, opts, F)
    if ports is not None:
        entry["ports"] = ports
    leaves = 0 if kind == "reference" else sum(pt.leaves for pt in params)
    if kind != "reference" and leaves > opts.partition_max_leaves:
        why = (f"its parameters' {leaves} leaves are over the partition's leaf budget {opts.partition_max_leaves}: the word query "
               "and Q_call resolve every leaf")
        F.warn("partition", "unchecked", t.id, 1, why)
        entry["partition"] = {"applies": as_call, "checked": False, "reason": why}
        entry["redundant_gates"] = {"checked": False}
        with spent("dead-gates"):
            entry["dead_gates"] = dead_gates(t, opts, F) if kind == "composite" else {"count": 0}
    elif kind == "composite":
        from verity.ml.circuits import boolean as BL

        with spent("partition"):
            part, rule = partition(t, opts, F, memo.setdefault("word", {}), applies=as_call, q_word=2 if BL.is_boolean(t) else 1)
        graph = part.pop("_graph", None)
        entry["partition"] = part
        with spent("redundant-gates"):
            entry["redundant_gates"] = redundant_gates(t, opts, F, rule, graph, within_units=as_call) if part["checked"] else {"checked": False}
        with spent("dead-gates"):
            entry["dead_gates"] = dead_gates(t, opts, F)
    elif kind == "primitive":
        entry["partition"] = {"applies": as_call, "checked": True, "units_per_call": 1 if t.params else 0,
                              "q_call": q_call(t, opts, F, memo.setdefault("word", {}))}
        entry["redundant_gates"] = {"count": 0}
        entry["dead_gates"] = {"count": 0}
    if getattr(getattr(t, "defn", t), "gated", False):
        with spent("digest"):
            entry["digest"] = circuit_digest_pin(t, pins, F)
    entry["failures"], entry["warnings"] = F.failures, F.warnings
    done[(t.id, lower_whole, as_call)] = entry
    return entry


def circuit_digest_pin(t: Any, pins: dict, F: Findings) -> dict:
    """A gated Definition's circuit digest, core's `program_digest` of its Program, which the gate a Program holds it as
    names (`verity.core.protocols.certifier.head_check.gate`), against its ``digests`` pin: the pin a verifier reads in
    place of the digest is the Definition's."""
    from verity.ml.circuits.codec import program_digest
    from verity.ml.circuits.program import Program

    return _counts_against_pin("digests", t.id, {"sha512": program_digest(Program(t))}, pins, F, t.id)


# --- one subcircuit -----------------------------------------------------------------------------------------------------------

def _pin_key(sub: Any) -> Any:
    p = sub.params
    if "N" in p and "EPS" in p:
        return int(p["N"]), float(p["EPS"])
    ints = [v for k, v in p.items() if isinstance(v, int) and not isinstance(v, bool)]
    return ints[0] if len(ints) == 1 else None


def _traced(build: Callable[[], Any]) -> tuple[Any, Any]:
    from verity_flock import boolean_export as BE

    return BE.trace(build)


def _export_counts(build: Callable[[], Any], name: str) -> dict:
    """The Boolean export of one proof block's circuit (`boolean_export.Registry.type_of`): its circuit types' AND/XOR/NOT counts, each
    distinct sub-circuit once, every type validated and the root evaluated against the circuit on random inputs (raises on a
    disagreement); ``flat_ands`` the circuit's own, fewer by the ANDs it shares across a call (``shared_across_types``)."""
    from verity_flock import boolean_export as BE

    reg = BE.Registry()
    rid = reg.type_of(build, name, "proof-unit")
    node = reg.nodes[rid]
    _, flat, typed = reg.shared[-1]
    return {"and": node["and"], "xor": node["xor"], "not": node["not"], "subcircuit_types": len(reg.types), "flat_ands": flat,
            "shared_across_types": typed - flat}


def template_lowering(sub: Any, st: dict, d: Any, opts: Options, F: Findings, pins: dict) -> dict:
    """A subcircuit's C-Flock proof blocks: one proof block's circuit digest against its module's pin, its counts against
    their pin and the Boolean export, its redundant ANDs, and the laid-out proof blocks against the reference on the edge and
    random vectors."""
    from verity_flock import ir_lower as IL
    from verity_numerical.bench import lowerings

    reg = lowerings.registry("C-Flock")
    tid = sub.template.id
    mod = reg.module(tid)
    if tid == "gemm-coordinate":
        steps = sorted({g.prim.id for g in standalone_call(d)[1] if g.prim.id.endswith(("Dot16_v1", "Dot16_v2", "Dot32_v1"))}) \
            if hasattr(d, "body") else []
        dot = next((p for p in steps if p in IL.TC_STEPS), None)
        if dot is None:
            return {"lowered": False, "reason": f"{sub.id}: its tensor-core step {', '.join(steps) or '(a silicon reference)'} is not one "
                    f"C-Flock lowers ({', '.join(sorted(IL.TC_STEPS))})"}
        name = f"gemm-coordinate/{dot}/tc-step"
        low = IL.lowering(d, name, unit_fn=lambda t: IL.tc_units(t, dot))
    elif mod is None or not callable(getattr(mod, "lowering", None)):
        return {"lowered": False, "reason": reg.supports(sub, "frame-v3/blake3-keyed") or f"{tid}: its C-Flock module has no lowering"}
    else:
        if hasattr(mod, "width") and mod.width(int(sub.params["N"])) is None:
            return {"lowered": False, "reason": f"{sub.id}: the kernel's warps differ at N = {sub.params['N']}, so no single C-Flock unit"}
        low = mod.lowering(sub, **st) if st else mod.lowering(sub)
        name = getattr(mod, "UNIT", low.name)
    rec: dict[str, Any] = {"lowered": True, "via": f"c-flock:{name}", "units_per_instance": low.units.n,
                           "ands": low.layout.counts["ands"], "rows": low.layout.counts["rows"]}
    pin = getattr(mod, "PINS", {}).get(_pin_key(sub)) if tid != "gemm-coordinate" and mod is not None else None
    digest = low.sha256
    if tid == "gumbel-top-p-token-select":
        from verity_flock import ir_sampling as S

        pin, digest = getattr(mod, "PIN", None), S.pin_digest(low)
    rec["digest"] = digest
    if pin is not None:
        rec["digest_pin"] = "ok" if pin == digest else "mismatch"
        if pin != digest:
            F.fail("lowering", "digest-mismatch", sub.id, f"[{name}]: {digest[:16]} != pinned {pin[:16]}", pinned=pin, digest=digest)
    if isinstance(low, IL.Lowering):
        build = low.units.circuit
    else:
        from verity_flock import ir_sampling as S

        def build():
            return S.lane_circuit(int(sub.params["V"]))
    C, root = _traced(build)
    typ = lowering_type(C)
    counts = _export_counts(build, f"{name} unit")
    rec["export"] = counts
    if counts["flat_ands"] != rec["ands"]:
        F.fail("lowering", "export-count", sub.id, f"[{name}]: the export's circuit has {counts['flat_ands']} ANDs, the layout {rec['ands']}")
    rec["pin"] = _counts_against_pin("units", sub.id, {**{k: counts[k] for k in ("and", "xor", "not")}, "type": typ}, pins, F, sub.id)
    red, by = redundant_ands(C, _owner_names(root, len(C.kind)))
    rec["redundant_ands"] = {"count": red, "where": dict(by.most_common())}
    if red:
        F.warn("redundant-gates", "boolean", sub.id, red, f"ANDs on identical operands in the {name} unit", where=dict(by.most_common()))
    if not isinstance(low, IL.Lowering):
        rec["compared"] = 0
        rec["correspondence"] = "the sampling lane's reference is verity/ml/flock/tests/test_ir_sampling.py (lane_reference)"
        return rec
    params, _ = signature_of(d)
    n = max(1, min(opts.vectors, opts.lane_budget // max(1, low.units.n)))
    arrays = vectors(params, n, opts.seed + 2)
    ref, _raised = reference_outputs(d, params, arrays)
    rows = [i for i, r in enumerate(ref) if r is not None]
    fin = flat(arrays)[rows]
    cuts = IL.cut_words(low, fin) if low.units.cut else None
    got, cut_out, ok = low.evaluate(fin, cuts)
    want = np.asarray([ref[i] for i in rows], dtype=np.uint64).reshape(len(rows), -1)
    covered = sorted({p for u in low.units.out_src for k, p in u if k == "ret"})
    bad = (got[:, covered] != want[:, covered]).any(axis=1) if covered else np.zeros(len(rows), bool)
    if cuts is not None:
        bad |= (cut_out != cuts).any(axis=1)
    lanes = len(rows) * low.units.n
    rec.update(compared=len(rows), mismatches=int(bad.sum()), unsatisfied_lanes=lanes - bin(ok).count("1") if lanes else 0)
    if bad.any():
        idx = np.flatnonzero(bad).tolist()
        F.fail("correspondence", "lowering-mismatch", sub.id, f"[{name}]: {len(idx)} of {len(rows)} vectors", via=f"c-flock:{name}",
               examples=[{"instance": rows[i], "lowered": _short(got[i].tolist()), "reference": _short(want[i].tolist())} for i in idx[:_EXAMPLES]])
    if rec["unsatisfied_lanes"]:
        F.fail("lowering", "unsatisfied", sub.id, f"[{name}]: {rec['unsatisfied_lanes']} honest unit lane(s) fail their rows")
    return rec


def check_subcircuit(sub: Any, opts: Options = Options(), *, instance_statics: Sequence[dict] = ({},), memo: dict | None = None,
                     pins: dict | None = None) -> dict:
    """Every check on a subcircuit: its Definition (and its evaluated target where that is another Definition) at each binding of
    its instance statics, and its C-Flock lowering."""
    memo = {} if memo is None else memo
    pins = load_pins() if pins is None else pins
    F = Findings()
    out: dict[str, Any] = {"subcircuit": sub.id, "template": sub.template.id, "params": dict(sub.params), "definitions": [], "lowering": []}
    for st in instance_statics:
        d = sub.definition(**st)
        e = check_definition(d, opts, memo=memo, pins=pins, lower_whole=False, as_call=True)
        out["definitions"].append(e)
        F.failures += e["failures"]
        F.warnings += e["warnings"]
        tgt = sub.target(**st)
        eq = target_equivalence(tgt, d, opts, F)
        if eq is not None:
            out.setdefault("target_equivalence", []).append(eq)
        low = template_lowering(sub, st, d, opts, F, pins)
        out["lowering"].append(dict(instance_statics=st, **low))
    out["failures"], out["warnings"] = F.failures, F.warnings
    return out


# --- one served workload ------------------------------------------------------------------------------------------------------

def check_row(path: str | Path, opts: Options = Options(), *, memo: dict | None = None, pins: dict | None = None) -> dict:
    """A served workload's request Program (a Build's ``instances.json.gz``, or its directory): `word.check_query` over its Calls (the
    cross-Call conditions: every Value a Call reads or returns is committed) and every check on each distinct Definition."""
    from types import SimpleNamespace

    from verity_vllm.correspondence.reader_for_query import Correspondence
    from verity.ml.definitions import word as W
    from verity_vllm.query.module_body import literal_calls
    from verity_vllm.query.program_view import from_instances, iter_instance_rows
    from verity_vllm.query.required import population

    memo = {} if memo is None else memo
    pins = load_pins() if pins is None else pins
    p = Path(path)
    inst = p / "instances.json.gz" if p.is_dir() else p
    P = from_instances(str(inst))
    corr = Correspondence(P)
    pop = population(P, corr)
    Q = SimpleNamespace(required=pop.required, literals=literal_calls(P), program_path=corr.module_of)
    F = Findings()
    r = W.check_query(P, Q, opts.X, opts.W, opts.extras, strict=False)
    for v in r["violations"]:
        F.fail("partition", v["class"], v["definition"], f"({v['calls']} Call(s)" + (f" under {v['where'][0]}" if v.get("where") else "") + ")",
               calls=v["calls"], where=v.get("where"))
    out: dict[str, Any] = {"row": str(p), "query": r["query"], "calls": r["calls"], "units": r["units"], "gates": r["gates"],
                           "committed_interior_words": r["committed_interior_words"], "definitions": []}
    try:
        from verity.ml.definitions import cross_call
    except ImportError:
        out["cross_call"] = {"checked": False, "reason": "verity.ml.definitions.cross_call is not on this tree"}
    else:
        x = cross_call.check_program(lambda: iter_instance_rows(str(inst)), strict=False)
        out["cross_call"] = {"checked": True, **{k: x[k] for k in ("check", "ok", "calls", "recomputed_gates", "unrefined")}}
        for a in x["recomputes"]:
            F.fail("partition", "cross-call-recompute", a["definition"], f"({a['level']}: {a['calls']} Call(s) compute {a['gates']} "
                   f"gate(s) of {a['originals']} earlier Call(s) again, e.g. {a['examples'][0][0]} = {a['examples'][0][1]})",
                   level=a["level"], calls=a["calls"], gates=a["gates"])
    groups = W.call_groups(P, Q)
    calls: Counter = Counter()
    for g in groups:
        calls[g["definition"]] += int(g["calls"])
    seen: set[str] = set()
    for g in groups:
        if g["definition"] in seen:
            continue
        seen.add(g["definition"])
        try:
            fn = W.specialization(g["family"], g["statics"])
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            F.fail("row", "unresolved", g["definition"], f"{type(e).__name__}: {str(e)[:160]}")
            continue
        e = dict(check_definition(fn, opts, memo=memo, pins=pins, as_call=True), calls=calls[g["definition"]])
        out["definitions"].append(e)
        F.failures += e["failures"]
        F.warnings += [dict(w, calls=e["calls"]) for w in e["warnings"]]
    out["failures"], out["warnings"] = F.failures, F.warnings
    return out


# --- reports ------------------------------------------------------------------------------------------------------------------

def summarize(entries: Iterable[dict]) -> dict:
    """Totals over entries: failures by name, new (`ok` is no new failure) and known (`circuit_check.known`, with the cause),
    warnings' counts by check and by Definition (a finding two entries share, such as a template's Definition checked on its own
    too, once), ``recomputed_across``: each Boolean Call (`Q_word` v2) that computes a value in two proof blocks, with its count, and
    ``q_call_refused``: each Definition `Q_call` v1 does not apply to, with why; ``by_calls``: each root checked by its calls
    (`circuit_check.by_calls`), not lowered whole, with the targets its callees rely on."""
    from .known import known
    entries = list(entries)
    fails: dict[str, dict] = {}
    warns: dict[str, dict] = {}
    for e in entries:
        fails.update((f["name"], f) for f in e.get("failures", []))
        warns.update((w["name"], w) for w in e.get("warnings", []))
    fails, warns = list(fails.values()), list(warns.values())
    by_check: Counter = Counter()
    by_def: dict[str, Counter] = {}
    for w in warns:
        k = f"{w['check']}/{w['code']}"
        by_check[k] += w["count"]
        by_def.setdefault(k, Counter())[w["definition"]] += w["count"]
    new = [f["name"] for f in fails if known(f) is None]
    across, refused, by_calls = {}, {}, {}
    for e in entries:
        p = e.get("partition") or {}
        if p.get("q_word") == 2 and p.get("recomputed_across"):
            across[e.get("definition") or e.get("target")] = p["recomputed_across"]
        if "inapplicable" in (p.get("q_call") or {}):
            refused[e.get("definition") or e.get("target")] = p["q_call"]["inapplicable"].get("why")
        if e.get("mode") == "by-calls":
            by_calls[e.get("definition") or e.get("target")] = sorted({c["binding"] for c in (e.get("by_calls") or {}).get("callees", [])
                                                                     if c.get("binding")})
    return {"ok": not new, "failures": len(fails), "failure_names": [f["name"] for f in fails], "new_failures": new,
            "recomputed_across": dict(sorted(across.items())), "q_call_refused": dict(sorted(refused.items())),
            "by_calls": dict(sorted(by_calls.items())),
            "known_failures": {f["name"]: known(f) for f in fails if known(f) is not None},
            "warnings": {k: {"total": v, "definitions": len(by_def[k]), "by_definition": dict(by_def[k].most_common())}
                         for k, v in by_check.most_common()}}


def report(targets: list[dict], opts: Options) -> dict:
    """The report over ``targets``, each by-calls root with its callees' targets' verdicts (`circuit_check.by_calls.joined`)."""
    from .by_calls import joined

    targets = joined(targets)
    return {"schema": SCHEMA, "options": {k: getattr(opts, k) for k in opts.__dataclass_fields__}, "summary": summarize(targets),
            "targets": targets}


__all__ = ["Findings", "Options", "SCHEMA", "check_definition", "check_row", "check_subcircuit", "correspondence", "dead_gates",
           "load_pins", "partition", "redundant_ands", "redundant_gates", "report", "summarize", "target_equivalence", "template_lowering",
           "whole_lowering"]
