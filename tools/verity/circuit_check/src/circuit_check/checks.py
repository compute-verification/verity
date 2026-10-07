"""The checks, over one Definition, one subcircuit (a template bound), or one served row's request Program.

    correspondence   the reference (`verity.ml.circuits.evaluation.evaluate`, the IR evaluator; a vLLM Definition's is its registered
                     evaluator) on the edge and random vectors of `circuit_check.vectors`: it must be total (a Definition whose
                     reference raises is partial), and every realization of it must equal it bit for bit: every registered
                     kernel (`verity.ml.circuits.evaluation.self_check`), a template's evaluated target where it is another Definition, and
                     the C-Flock Boolean lowering (`verity_flock.ir_lower`: the whole Definition when it is small, a template's
                     proof units otherwise); a Boolean Definition with a word counterpart
                     (`verity.ml.circuits.boolean.trace.word_of`) also equals that word Definition through its word view.  A
                     registered table a Definition reads (`targets.TABLE_PROBES`) is read lazily and probed at chosen
                     entries (`vectors.table_instances`), recorded under ``table`` with the words the instances read
    redundant gates  a value computed twice inside one unit: the IR gates `verity_vllm.query.word.Graph` finds recomputed (same
                     primitive, same operands by value) that `word.units` puts in one unit, and the ANDs of a Boolean unit on
                     identical operands; counted, with where (primitive @ activation, or the named subcircuit)
    dead gates       gates no output depends on (`verity.ml.circuits.liveness.dead_gates`, transitive), by primitive @ activation
    partition        `Q_word_v1{X,W,R=no-recompute}` (`word.unit_rule` per Definition, `word.check_query` over a row): every gate
                     certified by exactly one unit, cross-unit reads only through committed values, no value computed in two
                     units, the width rule (`verity.ml.circuits.partition.validate_unit_cut`'s codes and the word's width classes);
                     recorded: the whole Definition's cut under `Q_word` v2 (its codes, and the values computed in two units
                     it reports, `recomputed_across`).  A Boolean Call is judged by v2: its `recomputed_across` is recorded,
                     in its entry and the summary, and every other code fails as under v1.  `Q_call` v1 {X = 32} is recorded
                     beside it (`q_call`: units, committed bits, values computed in two units; or why it refuses, a warning).
                     Both resolve every parameter leaf, so a Definition over ``partition_max_leaves`` is not partitioned (a
                     warning, ``partition/unchecked``)
    lowering pins    a lowering's AND/XOR/NOT counts against its pin (a piece's, a Boolean Definition's whole lowering's,
                     a template unit's, in the sidecar of the module that defines it or its file in ``pins/``, `sidecars`; a Boolean
                     Definition's lowering has at most its own And gates), a template unit's circuit digest against its C-Flock
                     module's `PINS`, and the Boolean export's gate lists checked against the unit circuit
                     (`verity_flock.boolean_export`)

Correctness and invariant violations are failures, each with a name (``"<check>/<code> <definition> ..."``); redundant and dead
gates are warnings with counts (they become failures with the re-baseline's common-subexpression pass).  An unpinned lowering
is a warning.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from verity.ml.circuits.evaluation import ScalarReference, evaluate, kernels, self_check
from verity.ml.circuits.evaluation.reference import resolve, signature_of, standalone_call
from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean import is_boolean
from verity.ml.circuits.defs import PrimitiveDefinition

from . import sidecars as SC
from .vectors import edge_count, flat, instance_args, vectors

SCHEMA = "verity-circuit-check/v1"
PINS = SC.SHARED_PINS
_EXAMPLES = 3
_LIMB = (1 << 64) - 1


@dataclass(frozen=True)
class Options:
    """``vectors`` instances per correspondence check (the all-edge ones first), fewer when the reference would spend more than
    ``reference_budget`` gate evaluations on them (never fewer than ``min_vectors``); ``X`` / ``W`` / ``extras`` the word query;
    Definitions over ``partition_max_gates`` gates or ``partition_max_leaves`` parameter leaves are not partitioned, over
    ``lower_max_gates`` IR gates or about ``lower_max_ands`` ANDs (the piece pins' estimate) not lowered whole;
    ``input_budget`` bounds instances x input leaves, ``lane_budget`` instances x units when a template's units are evaluated."""

    vectors: int = 64
    min_vectors: int = 4
    seed: int = 20260926
    reference_budget: int = 400_000
    X: int = 16
    W: int = 32
    extras: int = 0
    partition_max_gates: int = 4_000_000
    partition_max_leaves: int = 1 << 22
    lower_max_gates: int = 256
    lower_max_ands: int = 60_000
    input_budget: int = 1 << 22
    lane_budget: int = 1 << 16
    dead_ids: int = 1 << 17


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


_SHA512 = ("one SHA-512 compression under the AND budget, traced from `sha512.compress_gadget`: the trace turns the gadget's "
           "23,455 commits (`sha512.commit`, And(x, ONE)) into x's wire, and `ir_lower` cannot commit an Xor, so every form "
           "is carried whole and grows round over round (mean 3,132 bits, 8.2 GB a third of the way in); each of these four "
           "was OOM-killed at 64 GiB (MemoryMax) after 14 minutes on vy-nebius-1 (run r20261006-074613-c4b9)")
#: Boolean Definition families whose whole C-Flock lowering does not fit in memory, and why
UNLOWERED: dict[str, str] = {f: _SHA512 for f in ("Sha512Compress_v1", "LmsChainStep_v1", "LmsLeaf_v1", "LmsNode_v1")}


def _family(t: Any) -> str:
    return getattr(t, "base_id", None) or t.id


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
    """The reference's output leaves per instance (None where it raises) and the instances it raised on."""
    n = arrays[0].shape[0] if arrays else 1
    out: list[tuple[int, ...] | None] = []
    raised: list[dict] = []
    for i in range(n):
        args = instance_args(params, arrays, i)
        try:
            out.append(tuple(evaluate(t, *args)))
        except Exception as e:  # noqa: BLE001  (a partial reference: the finding)
            out.append(None)
            raised.append({"instance": i, "args": _short(args), "error": f"{type(e).__name__}: {str(e)[:160]}"})
    return out, raised


def _probed_correspondence(t: Any, params: Sequence, probe: Any, opts: Options, F: Findings) -> tuple[dict, None, list]:
    """The reference on the vectors of a Definition that reads a registered table: its other parameters' edge and random
    vectors and the probes (`vectors.table_instances`), every instance reading the table lazily.  No kernel takes it."""
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


def _wide_words(t: Any, params: Sequence[Any], ret: Any, arrays: Sequence[np.ndarray], n: int) -> np.ndarray:
    """`evaluate_words` where an element is wider than its 64 bits (PoUW's 576-bit sums of squares): Python ints, each element
    through its bits (`verity.ml.circuits.evaluation.bits.evaluate_bits`)."""
    from verity.ml.circuits.evaluation.batch import type_shape
    from verity.ml.circuits.evaluation.bits import evaluate_bits

    ins = []
    for a, p in zip(arrays, params):
        widths = BL.elements(p)
        rows = np.asarray(a, dtype=object).reshape(n, len(widths))
        ins.append(np.array([[(int(v) >> j) & 1 for v, w in zip(row, widths) for j in range(w)] for row in rows],
                            dtype=np.uint8).reshape(n, *type_shape(p)))
    bits = np.asarray(evaluate_bits(t, *ins)).reshape(n, -1)
    out = np.zeros((n, len(BL.elements(ret))), dtype=object)
    at = 0
    for e, w in enumerate(BL.elements(ret)):
        for i in range(n):
            out[i, e] = sum(int(b) << j for j, b in enumerate(bits[i, at:at + w]))
        at += w
    return out


def word_view(t: Any, opts: Options, F: Findings) -> dict | None:
    """A Boolean Definition against the word Definition it computes (`verity.ml.circuits.boolean.trace.word_of`), through the word view
    (`verity.ml.circuits.evaluation.bits.evaluate_words`), on the word Definition's edge and random vectors."""
    if isinstance(t, (PrimitiveDefinition, ScalarReference)) or not is_boolean(t):
        return None
    from verity.ml.circuits.evaluation import evaluate_batch
    from verity.ml.circuits.evaluation.bits import evaluate_words
    from verity.ml.circuits.boolean.trace import word_of

    try:
        word = word_of(t)
    except KeyError:
        return None
    pw, _ = signature_of(word)
    arrays = vectors(pw, 16 * opts.vectors, opts.seed + 2)
    n = arrays[0].shape[0]
    pb, rb = signature_of(t)
    if max(w for p in (*pb, rb) for w in BL.elements(p)) > 64:
        want = np.asarray(evaluate_batch(word, *arrays), dtype=object).reshape(n, -1)
        got = _wide_words(t, pb, rb, arrays, n)
    else:
        want = np.asarray(evaluate_batch(word, *arrays), dtype=np.uint64).reshape(n, -1)
        got = np.asarray(evaluate_words(t, *arrays), dtype=np.uint64).reshape(n, -1)
    bad = np.flatnonzero((got != want).any(axis=1)).tolist()
    if bad:
        F.fail("correspondence", "word-view-mismatch", t.id, f"[{word.id}]: {len(bad)} of {n} vectors", via=f"word-view:{word.id}",
               examples=[{"instance": i, "args": _short(instance_args(pw, arrays, i)), "boolean": _short(got[i].tolist()),
                          "word": _short(want[i].tolist())} for i in bad[:_EXAMPLES]])
    return {"via": f"word-view:{word.id}", "compared": n, "mismatches": len(bad)}


# --- Boolean circuits -------------------------------------------------------------------------------------------------------

def redundant_ands(C: Any, owner: Callable[[int], str] | None = None) -> tuple[int, Counter]:
    """Live ANDs of the `gf2.Circuit` ``C`` whose operands, with earlier duplicates replaced by their first copy, are those of an
    earlier AND (in either order), are equal, or are constant: values computed again.  Returns (count, count by ``owner(bit)``)."""
    live = C.live()
    rep: dict[int, int] = {}
    seen: dict[tuple, int] = {}
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
        ka, kb = canon(a), canon(b)
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


def _counts_against_pin(kind: str, key: str, got: dict, pins: dict, F: Findings, definition: str) -> dict:
    pinned = pins.get(kind, {}).get(key)
    rec = {"pin": key, "counts": got, "pinned": pinned}
    if pinned is None:
        F.warn("lowering", "unpinned", definition, 1, f"[{key}]: no pinned counts", pins=kind, pin=key, counts=got)
        rec["status"] = "unpinned"
    elif any(pinned.get(k) != v for k, v in got.items() if k in pinned):
        diff = {k: [pinned.get(k), v] for k, v in got.items() if pinned.get(k) != v}
        F.fail("lowering", "pin-mismatch", definition, f"[{key}]: " + ", ".join(f"{k} {a} -> {b}" for k, (a, b) in diff.items()), pins=kind,
               pin=key, counts=got, pinned=pinned)
        rec["status"] = "mismatch"
    else:
        rec["status"] = "ok"
    return rec


_UNPINNED_ANDS = 2_000


def _pieces_used(t: Any, pins: dict) -> tuple[Counter, Counter, int]:
    """(pieces used, primitives without a piece, estimated ANDs) of ``t``'s gates that are not constant (a gate whose operands
    are all constant is evaluated, as `ir_lower.lower_gates` does); the estimate from the piece pins."""
    from verity_flock import ir_lower as IL

    _prog, call = standalone_call(t)
    const: dict[int, bool] = {}
    used, missing, est = Counter(), Counter(), 0
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
    reference on the correspondence vectors; its redundant ANDs; a primitive's AND count against the piece pins.  A composite's
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
    elif is_boolean(t) and (why := UNLOWERED.get(_family(t))):
        return {"lowered": False, "reason": why}
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
    fin = flat(arrays)[rows]
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
        rec["pin"] = _counts_against_pin("pieces", t.id, {"and": counts["ands"]}, pins, F, t.id)
    elif is_boolean(t):
        from verity.ml.circuits.evaluation.bits import compile as compile_bits

        own = rec["definition_ands"] = compile_bits(t).counts["and"]
        if counts["ands"] > own:
            F.fail("lowering", "ands-added", t.id, f"[c-flock]: {counts['ands']} ANDs for the Definition's {own} And gates",
                   via="c-flock:ir_lower", ands=counts["ands"], definition_ands=own)
        rec["pin"] = _counts_against_pin("definitions", t.id, {"and": counts["ands"]}, pins, F, t.id)
    return rec


# --- the IR circuit: redundant gates, dead gates, the partition ---------------------------------------------------------------

def redundant_gates(fn: Any, opts: Options, F: Findings, rule: dict, graph: tuple | None = None, within_units: bool = True) -> dict:
    """Values computed twice (`word.Graph.recomputed`: same primitive, same operands by value), by where: inside one unit of a
    Call's cut (``within_units``; across units they are recomputes, a partition failure), or anywhere in a Definition that is
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
    """`Q_word_v1` on one Call of ``fn`` (its record and the word's unit rule).  Where the invariant applies (``fn`` is a Call of a
    served Program) every violation fails by name; elsewhere the cut of ``fn`` alone is recorded (``isolated``), since its gates
    are partitioned inside the Call that contains it.  With ``q_word=2`` (a Boolean Call: `Q_word` v2) a value computed in two
    units is no failure: ``recomputed_across`` records how many gates are (the whole cut's v2 view, else the unit rule's count;
    None when ``fn`` is over the partition budget and unchecked), and every other code fails as under v1."""
    from verity_vllm.query import word as W

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
        # the whole Definition's own cut does (a value two nodes compute is a recompute across units)
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
    """`Q_call{X = cut.CALL_X}` v1 on one Call of ``fn``: its units, committed gates and bits, and the values two units compute
    (reported, as `Q_word` v2 does); or why the query refuses it (``inapplicable``: a gate wider than X, a warning).  A code of
    the cut's own check fails by name (``partition/q-call-<code>``); none is expected, since the cut derives what it checks."""
    from verity.ml.circuits import cut

    X = cut.CALL_X
    if fn.gates > opts.partition_max_gates:
        return {"X": X, "checked": False}
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
    warnings.  ``memo`` shares the word query's unit rules and the entries across calls; ``lower_whole=False`` leaves the
    Boolean lowering to a template's units; ``as_call`` says whether the partition invariant applies (default: its family is a
    Call of a served Program, `circuit_check.targets.call_families`).  A Boolean Definition held as a Call is judged by `Q_word` v2
    (`partition`), a word Definition by v1."""
    memo = {} if memo is None else memo
    pins = load_pins() if pins is None else pins
    t = resolve(fn)
    done = memo.setdefault("entries", {})
    if as_call is None:
        from .targets import call_families, family

        as_call = family(t) in call_families()
    if (t.id, lower_whole, as_call) in done:
        return done[(t.id, lower_whole, as_call)]
    F = Findings()
    kind = "reference" if isinstance(t, ScalarReference) else "primitive" if isinstance(t, PrimitiveDefinition) else "composite"
    entry: dict[str, Any] = {"definition": t.id, "form": kind, "gates": None if kind == "reference" else _gates(t)}
    corr, arrays, ref = correspondence(t, opts, F)
    params, _ = signature_of(t)
    if arrays is None:
        low = {"lowered": False, "reason": "a lowering takes every input leaf, and this Definition's correspondence reads a "
               "registered table lazily" if "table" in corr else "no vectors"}
    elif lower_whole or kind != "composite":
        low = whole_lowering(t, params, arrays, ref, opts, F, pins, memo)
    else:
        low = {"lowered": False, "reason": "lowered as its template's units"}
    corr["realizations"].append({"via": "c-flock:ir_lower", **low})
    wv = word_view(t, opts, F)
    if wv is not None:
        corr["realizations"].append(wv)
    entry["correspondence"] = corr
    leaves = 0 if kind == "reference" else sum(pt.leaves for pt in params)
    if kind != "reference" and leaves > opts.partition_max_leaves:
        why = (f"its parameters' {leaves} leaves are over the partition's leaf budget {opts.partition_max_leaves}: the word query "
               "and Q_call resolve every leaf")
        F.warn("partition", "unchecked", t.id, 1, why)
        entry["partition"] = {"applies": as_call, "checked": False, "reason": why}
        entry["redundant_gates"] = {"checked": False}
        entry["dead_gates"] = dead_gates(t, opts, F) if kind == "composite" else {"count": 0}
    elif kind == "composite":
        from verity.ml.circuits import boolean as BL

        part, rule = partition(t, opts, F, memo.setdefault("word", {}), applies=as_call, q_word=2 if BL.is_boolean(t) else 1)
        graph = part.pop("_graph", None)
        entry["partition"] = part
        entry["redundant_gates"] = redundant_gates(t, opts, F, rule, graph, within_units=as_call) if part["checked"] else {"checked": False}
        entry["dead_gates"] = dead_gates(t, opts, F)
    elif kind == "primitive":
        entry["partition"] = {"applies": as_call, "checked": True, "units_per_call": 1 if t.params else 0,
                              "q_call": q_call(t, opts, F, memo.setdefault("word", {}))}
        entry["redundant_gates"] = {"count": 0}
        entry["dead_gates"] = {"count": 0}
    entry["failures"], entry["warnings"] = F.failures, F.warnings
    done[(t.id, lower_whole, as_call)] = entry
    return entry


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
    """The Boolean export of one unit circuit (`boolean_export.Registry.type_of`): its circuit types' AND/XOR/NOT counts, each
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
    """A subcircuit's C-Flock proof units: the unit circuit's digest against its module's pin, its counts against their pin
    and the Boolean export, its redundant ANDs, and the laid-out units against the reference on the edge and random vectors."""
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
    counts = _export_counts(build, f"{name} unit")
    rec["export"] = counts
    if counts["flat_ands"] != rec["ands"]:
        F.fail("lowering", "export-count", sub.id, f"[{name}]: the export's circuit has {counts['flat_ands']} ANDs, the layout {rec['ands']}")
    rec["pin"] = _counts_against_pin("units", sub.id, {k: counts[k] for k in ("and", "xor", "not")}, pins, F, sub.id)
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


# --- one served row -----------------------------------------------------------------------------------------------------------

def check_row(path: str | Path, opts: Options = Options(), *, memo: dict | None = None, pins: dict | None = None) -> dict:
    """A served row's request Program (a Build's ``instances.json.gz``, or its directory): `word.check_query` over its Calls (the
    cross-Call conditions: every Value a Call reads or returns is committed) and every check on each distinct Definition."""
    from types import SimpleNamespace

    from verity_vllm.correspondence.reader_for_query import Correspondence
    from verity_vllm.query import word as W
    from verity_vllm.query.module_body import literal_calls
    from verity_vllm.query.program_view import from_instances
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
        from verity_vllm.query import cross_call
    except ImportError:
        out["cross_call"] = {"checked": False, "reason": "verity_vllm.query.cross_call is not on this tree (PR #98)"}
    else:
        x = cross_call.check_program(str(inst), strict=False)
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
    too, once), ``recomputed_across``: each Boolean Call (`Q_word` v2) that computes a value in two units, with its count, and
    ``q_call_refused``: each Definition `Q_call` v1 does not apply to, with why."""
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
    across, refused = {}, {}
    for e in entries:
        p = e.get("partition") or {}
        if p.get("q_word") == 2 and p.get("recomputed_across"):
            across[e.get("definition") or e.get("target")] = p["recomputed_across"]
        if "inapplicable" in (p.get("q_call") or {}):
            refused[e.get("definition") or e.get("target")] = p["q_call"]["inapplicable"].get("why")
    return {"ok": not new, "failures": len(fails), "failure_names": [f["name"] for f in fails], "new_failures": new,
            "recomputed_across": dict(sorted(across.items())), "q_call_refused": dict(sorted(refused.items())),
            "known_failures": {f["name"]: known(f) for f in fails if known(f) is not None},
            "warnings": {k: {"total": v, "definitions": len(by_def[k]), "by_definition": dict(by_def[k].most_common())}
                         for k, v in by_check.most_common()}}


def report(targets: list[dict], opts: Options) -> dict:
    return {"schema": SCHEMA, "options": {k: getattr(opts, k) for k in opts.__dataclass_fields__}, "summary": summarize(targets),
            "targets": targets}


__all__ = ["Findings", "Options", "SCHEMA", "check_definition", "check_row", "check_subcircuit", "correspondence", "dead_gates",
           "load_pins", "partition", "redundant_ands", "redundant_gates", "report", "summarize", "target_equivalence", "template_lowering",
           "whole_lowering"]
