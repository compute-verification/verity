"""The checks, over one Definition, one subcircuit (a template bound), or one served workload's request Program.

    correspondence   the reference (`verity.circuits.evaluation.evaluate`, the circuits' evaluator, a composite's run body by body by
                     `circuit_check.bodies`; a vLLM Definition's is its registered evaluator) on the edge and random vectors
                     of `circuit_check.vectors`: it must be total (a Definition whose
                     reference raises is partial), and every realization of it must equal it bit for bit: every registered
                     kernel (`verity.circuits.evaluation.self_check`, which must be `SelfCheck.ok`: some vector compared, and
                     none declined inside the reference's domain unless the kernel is partial; else ``kernel-<fault>``), a
                     template's evaluated target where it is another Definition, and
                     the Flock Boolean lowering (`verity_flock.ir_lower`: the whole Definition when it is small, a template's
                     proof blocks otherwise) and, under ``Options.struct_max_gates`` or its family's struct budget
                     (`struct_budget`), the structural lowering's template
                     (``c-flock:struct_lower``, `struct_lowering`, evaluated from the template); the word Definition a
                     Boolean Definition names
                     (`verity.circuits.boolean.trace.word_of`) must equal the Boolean gates through the word view
                     (``word:<word>``): the Boolean Definition is the definition.  A
                     registered table a Definition reads (`targets.TABLE_PROBES`) is read lazily and probed at chosen
                     entries (`vectors.table_instances`), recorded under ``table`` with the words the vectors read; an
                     operand over the input budget (`targets.LAZY_OPERANDS`, a weight) takes its edge and random words one
                     vector at a time and is read lazily (`vectors.wide_words`), recorded under ``wide`` with the reads of
                     its words, plus a random vector per index probe (`targets.OPERAND_PROBES`: a shard's or an expert
                     count's first and last index), and one that no vector reads is a warning; on both paths the kernels decline, as they take arrays, and the lowering takes every leaf.
                     Each realization is compared on at least ``Options.random_floor`` random vectors within
                     ``Options.random_budget``: a lowering on random ones added to the vectors (`floored`), a kernel on
                     those its own sampler samples in its domain (`_sampled`); a target no realization compared on a random
                     vector is a warning, ``vectors/edge-only`` when one compared it on all-edge vectors only, else
                     ``vectors/none`` (`vector_class`).  Each correspondence and word view records how many of its vectors
                     have every input leaf finite and how many give every output leaf finite (``finite_inputs``,
                     ``finite_outputs``; `vectors.finite_words`), and fewer finite inputs than `FINITE_FLOOR` and than the
                     vectors' finite share (`vectors.finite_share`) is a warning, ``vectors/few-finite``
    ports            a Definition's declared return intervals (`verity.circuits.ports`, PROTOCOL.md §12), which the Build takes
                     as facts about a primitive: its reference on edge and random operands, each leaf of a bounded parameter
                     folded into its bound, returns every leaf within its interval; and a primitive's transfer, on sampled
                     operand intervals within its bounds with every operand leaf folded into its interval, returns within the
                     interval the transfer gives; recorded only for a Definition that declares either
    redundant gates  a value computed twice inside one proof block: the gates `verity.definitions.word.Graph` finds recomputed
                     (same primitive, same operands by value) that `word.units` puts in one proof block, and the ANDs of a
                     Boolean proof block on identical operands, its commits apart (`redundant_ands`); counted, with where
                     (primitive @ activation, or the named subcircuit)
    dead gates       gates no output depends on (`verity.circuits.liveness.dead_gates`, transitive), by primitive @ activation
    partition        `Q_word_v1{X,W,R=no-recompute}` (`word.unit_rule` per Definition, `word.check_query` over a workload's
                     Program): every gate certified by exactly one proof block, reads across proof blocks only through
                     committed values, no value computed in two proof blocks, the width rule
                     (`verity.circuits.partition.validate_unit_cut`'s codes and the word's width classes); recorded: the
                     whole Definition's cut under `Q_word` v2 (its codes, and the values computed in two proof blocks it
                     reports, `recomputed_across`).  A Boolean Call is judged by v2: its `recomputed_across` is recorded,
                     in its entry and the summary, and every other code fails as under v1.  `Q_call` v1 {X = 32} is recorded
                     beside it (`q_call`: proof blocks, committed bits, values computed in two proof blocks; or why it refuses,
                     a warning).
                     Both resolve every parameter leaf, so a Definition over ``partition_max_leaves`` is not partitioned,
                     nor is one over ``partition_max_gates`` gates not cut member by member, or cut so down to a member
                     over the budget that isn't (`uncut`); such a Call is
                     partitioned by its callees where each is a Call (`by_callees`: each callee's partition checked as a
                     Call's, failing by its name), and otherwise fails by its own name, ``partition/unpartitioned``, with why
                     its callees can't partition it (`callee_gap`: one is no Call, or is a primitive over the budget): every
                     gate of a served Call sits in a Call whose partition is checked.  A Definition that is no Call is
                     partitioned inside the Call that contains it, so over the budget it is a warning, ``partition/unchecked``
    lowering pins    a lowering's AND/XOR/NOT counts and its circuit's digest (``type``, `lowering_type`) against its pin, which
                     must hold both (a piece's, a Boolean Definition's whole lowering's,
                     a template proof block's, in the sidecar of the module that defines it or its file in ``pins/``,
                     `sidecars`; a Boolean Definition's lowering has at most its own And gates, besides its builder's
                     commits), a template proof block's circuit digest against its Flock module's `PINS`, and the Boolean
                     export's gate lists checked against the proof block's circuit (`verity_flock.boolean_export`); a gated
                     Definition's circuit digest (``gated``: a Program holds it as one gate naming the digest, such as N2's
                     head check) against its ``digests`` pin, as is any Definition's that holds one (a served-size binding)
    block            the Flock block a whole Definition needs as one unit, read off it before lowering
                     (`verity_flock.circuit.unit_block`: its rows' sha512x3 and hm96 slots and the mask beside its own slot
                     of ANDs and ports, as `compose` places them), within Flock's proved range of 2^``K_MAX`` (`block_range`):
                     past it, `compose` refuses the unit.  Failed for every Definition checked whole, as PoUW and a served
                     window's templates each stage one, unless `block_held.HELD` says why not (the report's, `summarize`);
                     a Call serving stages as its cut (its module's ``CUTS``, `cut_of`), for each unit of the cut instead

Correctness and invariant violations are failures, each with a name (``"<check>/<code> <definition> ..."``); redundant and dead
gates are warnings with counts (they become failures with the re-baseline's common-subexpression pass).  An unpinned lowering,
and a pin without its lowering's ``type``, is a warning.  A warning never fails (unless ``--fail-on-warnings``): it is a
property left unchecked, and the report prints it under ``unchecked``, apart from what passed.
"""
from __future__ import annotations

import importlib
import random
import sys
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np

from verity.circuits.evaluation import ScalarReference, evaluate, kernels, self_check
from verity.circuits.evaluation.reference import resolve, signature_of, standalone_call
from verity.circuits import boolean as BL
from verity.circuits.boolean import is_boolean
from verity.circuits.defs import PrimitiveDefinition, conformance_label

from . import bodies as BD
from . import sidecars as SC
from .vectors import edge_count, finite_args, finite_rows, finite_share, finite_words, flat, instance_args, vectors

SCHEMA = "verity-circuit-check/v1"
PINS = SC.SHARED_PINS
_EXAMPLES = 3
_LIMB = (1 << 64) - 1


@dataclass(frozen=True)
class Options:
    """``vectors`` input vectors per correspondence check (the all-edge ones first), fewer when the reference would spend more
    than ``reference_budget`` gate evaluations on them (never fewer than ``min_vectors``); ``X`` / ``W`` / ``extras`` the word
    query; the word view draws 16 times the vectors within 16 times the budget (`_word_vectors`); Definitions over
    ``partition_max_gates`` gates or ``partition_max_leaves`` parameter leaves are not partitioned, over
    ``lower_max_gates`` gates or about ``lower_max_ands`` ANDs (the piece pins' estimate) not lowered whole; ``input_budget``
    bounds vectors x input leaves, ``lane_budget`` vectors x proof blocks when a template's proof blocks are evaluated;
    a composite of at most ``struct_max_gates`` gates is realized by its structural lowering too (`struct_lowering`; 0, the
    default: none).  Each realization is compared on at least ``random_floor`` random vectors while the reference's work on those
    the floor adds is at most ``random_budget`` gate evaluations (and a template's lanes on them at most ``lane_budget``):
    added after the all-edge ones for a lowering (`floored`), sampled by a kernel's own sampler for it (`_sampled`)."""

    vectors: int = 64
    min_vectors: int = 4
    seed: int = 20260926
    reference_budget: int = 400_000
    random_floor: int = 2
    random_budget: int = 10_000_000
    X: int = 16
    W: int = 32
    extras: int = 0
    partition_max_gates: int = 5_000_000
    partition_max_leaves: int = 1 << 22
    lower_max_gates: int = 256
    lower_max_ands: int = 1 << 16
    input_budget: int = 1 << 22
    lane_budget: int = 1 << 16
    dead_ids: int = 1 << 17
    struct_max_gates: int = 0

DEFAULTS = Options()


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


def _n_vectors(t: Any, opts: Options, lazy: int | Iterable[int] = ()) -> int:
    """The vectors of ``t``'s correspondence; the parameters ``lazy`` names are read lazily, so their leaves don't count."""
    params, _ = signature_of(t)
    skip = {lazy} if isinstance(lazy, int) else set(lazy)
    leaves = sum(pt.leaves for j, pt in enumerate(params) if j not in skip)
    n = max(opts.min_vectors, min(opts.vectors, opts.reference_budget // max(1, _gates(t))))
    return max(1, min(n, opts.input_budget // max(1, leaves)))


def _lazy_vectors(t: Any, opts: Options, wide: Sequence[int]) -> tuple[int, bool]:
    """``(vectors, capped)`` of a Definition whose ``wide`` operands are read lazily (`_wide_correspondence`): `_n_vectors`',
    which the input budget doesn't bound (a vector reads only what it selects of them), unless the vectors its
    ``opts.min_vectors`` floor adds past what ``opts.reference_budget`` pays for take the reference's work past
    ``opts.random_budget``; then, capped, as many as ``opts.reference_budget`` pays for and at least ``opts.random_floor + 1``, at
    most half of them all-edge and at least ``opts.random_floor`` random, as the word view splits its own past its budget."""
    n, g = _n_vectors(t, opts, wide), max(1, _gates(t))
    paid = opts.reference_budget // g
    if n <= paid or (n - paid) * g <= opts.random_budget:
        return n, False
    return min(n, max(paid, opts.random_floor + 1)), True


def floored(t: Any, n: int, edge: int, opts: Options, lanes: int = 0) -> tuple[int, int]:
    """``(vectors, all-edge vectors)`` for a realization of ``t`` compared on ``n`` vectors of parameters with ``edge`` edge
    words: the all-edge ones first, as many as fit in ``n``, then the random ones, which the floor makes at least
    ``opts.random_floor`` by adding vectors when the reference's work on them is within ``opts.random_budget`` gate evaluations
    and their ``lanes`` (a template's proof blocks a vector) within ``opts.lane_budget``.  Past either the target keeps ``n`` and
    may compare on edge vectors only (`vector_class`).  The added vectors may take the arrays past ``opts.input_budget`` leaves,
    to at most ``1 + opts.random_floor`` times it, as each vector is within it (a parameter over it is read lazily)."""
    e = min(n, edge)
    add = max(0, opts.random_floor - (n - e))
    if add * max(1, _gates(t)) > opts.random_budget or add * lanes > opts.lane_budget:
        add = 0
    return n + add, e


def reference_outputs(t: Any, params: Sequence, arrays: Sequence[np.ndarray]) -> tuple[list[tuple[int, ...] | None], list[dict]]:
    """The reference's output leaves per vector (None where it raises) and the vectors it raised on; a composite's
    reference run body by body (`bodies.evaluate_rows`)."""
    n = arrays[0].shape[0] if arrays else 1
    rows = [instance_args(params, arrays, i) for i in range(n)]
    out: list[tuple[int, ...] | None] = []
    raised: list[dict] = []
    for i, (args, got) in enumerate(zip(rows, BD.evaluate_rows(t, rows))):
        if isinstance(got, Exception):
            out.append(None)
            raised.append({"instance": i, "args": _short(args), "error": f"{type(got).__name__}: {str(got)[:160]}"})
        else:
            out.append(tuple(got))
    return out, raised


#: the vectors with every input leaf finite a comparison is to have, or all of its finite share (`vectors.finite_share`) when
#: that is fewer: short of it is a warning (``vectors/few-finite``), never a failure
FINITE_FLOOR = 64


def _finite_outputs(t: Any, ref: Sequence[tuple[int, ...] | None]) -> int:
    """How many of the reference's outputs ``ref`` (None where it raised) are finite in every leaf (`vectors.finite_words`)."""
    widths = signature_of(t)[1].leaf_widths()
    rows = [r for r in ref if r is not None]
    return int(finite_words(np.asarray(rows, dtype=object).reshape(len(rows), len(widths)), widths).sum()) if rows else 0


def _finite(rec: dict, t: Any, via: str, inputs: int, outputs: int, share: int, F: Findings) -> None:
    """``rec`` with its counts of vectors whose inputs and whose outputs are all finite, and the warning when the inputs are
    fewer than `FINITE_FLOOR` and than the ``share`` sampled finite."""
    rec["finite_inputs"], rec["finite_outputs"] = int(inputs), int(outputs)
    if inputs < min(FINITE_FLOOR, share):
        F.warn("vectors", "few-finite", t.id, 1, f"[{via}]: {inputs} of its {rec.get('vectors', rec.get('compared'))} vectors have "
               f"every input leaf finite, fewer than {FINITE_FLOOR} and than the {share} sampled finite", via=via)


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
    _finite(rec, t, "correspondence", sum(finite_args(params, a) for a in instances), _finite_outputs(t, ref),
            len(finite_share(n, rec["edge_vectors"])), F)
    if raised:
        F.fail("correspondence", "reference-not-total", t.id, f"raised on {len(raised)} of {len(instances)} vectors",
               examples=raised[:_EXAMPLES])
    for name in sorted(kernels(t)):
        rec["realizations"].append({"via": f"kernel:{name}", "declines": f"a kernel takes arrays, not the registered table {table['name']!r}"})
    return rec, None, ref


_WIDE_HOLDERS: dict[str, PrimitiveDefinition] = {}


def _wide_holder(name: str) -> PrimitiveDefinition:
    """A primitive declaring ``name`` a registered table, as `RegisteredTable` requires of what it reads: an operand read lazily
    is declared by no Definition."""
    h = _WIDE_HOLDERS.get(name)
    if h is None:
        from verity.circuits.types import Value

        h = _WIDE_HOLDERS[name] = PrimitiveDefinition("CircuitCheckLazyOperand", 1, (("x", Value(64)),), Value(64), lambda x: x,
                                                        register=False, tables=((name, "registered"),))
    return h


def _unread_operand(t: Any, params: Sequence, wide: Sequence[int]) -> str | None:
    """Why ``t``'s parameters ``wide`` can't be read lazily on `_wide_correspondence`'s vectors, or None."""
    from .targets import lazy_operands

    names = [n for n, _ in resolve(t).signature()[0]] if not isinstance(t, ScalarReference) else []
    listed = set(lazy_operands(t))
    for j in wide:
        if not names or names[j] not in listed:
            return (f"parameter {j} has {params[j].leaves} leaves, over the input budget, and is neither a registered table a probe "
                    "reads (`circuit_check.targets.TABLE_PROBES`) nor an operand read lazily (`circuit_check.targets.LAZY_OPERANDS`)")
        w = params[j].homogeneous_width()
        if w is None or w > 64:
            return f"parameter {j} ({names[j]!r}) is read lazily, so it needs one width of at most 64 bits"
    if isinstance(t, PrimitiveDefinition):
        if getattr(t, "evaluate_lazy", None) is None:
            return f"{t.id} is a primitive without `evaluate_lazy`, so it reads every leaf of a parameter"
        return None
    from verity.circuits.refs import Affine

    prog, _call = standalone_call(t)
    for j in wide:
        refs = prog.input_coll(j).refs
        if not (isinstance(refs, Affine) and refs.count == params[j].leaves):
            return f"parameter {j} ({names[j]!r}) is read lazily, so it needs one contiguous run of Input gates, not {refs!r}"
    return None


def _wide_correspondence(t: Any, params: Sequence, wide: Sequence[int], opts: Options, F: Findings) -> tuple[dict, None, list]:
    """The reference on the vectors of a Definition with operands over the input budget (`targets.LAZY_OPERANDS`; how many,
    `_lazy_vectors`): every parameter takes its edge and random words, the wide ones one vector at a time
    (`vectors.wide_words`), each read lazily as a `RegisteredTable`, then one random vector per index probe
    (`targets.OPERAND_PROBES`) with its index parameters set.  An operand no vector reads is a warning.  No kernel takes them."""
    from verity.circuits.evaluation import RegisteredTable
    from verity.circuits.types import Value

    from .targets import operand_probes
    from .vectors import edges, wide_words

    n, capped = _lazy_vectors(t, opts, wide)
    rest = [j for j in range(len(params)) if j not in wide]
    other = [params[j] for j in rest]
    n_edge = min(n, max([edge_count(other)] + [len(edges(params[j].homogeneous_width())) for j in wide]))
    if capped:
        n_edge = max(0, min(n_edge, n // 2, n - opts.random_floor))
    arrays = vectors(other, n, opts.seed, n_edge if capped else None)
    names = [nm for nm, _ in resolve(t).signature()[0]]
    reads = dict.fromkeys(wide, 0)
    probes = operand_probes(t)
    at = {nm: k for k, nm in enumerate(names[j] for j in rest)}
    scalars = sorted(nm for nm, k in at.items() if isinstance(other[k], Value))
    for p in probes:
        if not set(p) <= set(scalars):
            raise ValueError(f"{t.id}: the operand probe {p} sets a parameter that isn't one of its Value parameters {scalars} "
                             "(`circuit_check.targets.OPERAND_PROBES`)")
    E0 = edge_count(other)
    extra = vectors(other, E0 + len(probes), opts.seed + 1) if probes else []

    def small(i: int) -> list:
        if i < n:
            return instance_args(other, arrays, i)
        a = instance_args(other, extra, E0 + i - n)
        for nm, v in probes[i - n].items():
            a[at[nm]] = v
        return a

    def lazily(j: int, word: Callable[[int], int]) -> Callable[[int], int]:
        def read(k: int) -> int:
            reads[j] += 1
            return word(k)
        return read

    total = n + len(probes)
    ref: list[tuple[int, ...] | None] = []
    raised: list[dict] = []
    finite_in = 0
    for i in range(total):
        args: list[Any] = [None] * len(params)
        row = small(i)
        for j, a in zip(rest, row):
            args[j] = a
        finite = finite_args(other, row)
        for j in wide:
            words = wide_words(params[j], i, n_edge, opts.seed, j)
            finite = finite and words.finite
            args[j] = RegisteredTable(f"operand:{names[j]}", lazily(j, words), params[j].leaves, _wide_holder(f"operand:{names[j]}"))
        finite_in += finite
        try:
            ref.append(tuple(evaluate(t, *args)))
        except Exception as e:  # noqa: BLE001  (a partial reference: the finding)
            ref.append(None)
            raised.append({"instance": i, "args": _short(args), "error": f"{type(e).__name__}: {str(e)[:160]}"})
    rec: dict[str, Any] = {"vectors": total, "edge_vectors": n_edge, "reference_raised": len(raised), "realizations": [],
                           "wide": [{"param": j, "name": names[j], "leaves": params[j].leaves,
                                     "width": params[j].homogeneous_width(), "reads": reads[j]} for j in wide]}
    _finite(rec, t, "correspondence", finite_in, _finite_outputs(t, ref), len(finite_share(n, n_edge)), F)
    if probes:
        rec["probes"] = probes
    if raised:
        F.fail("correspondence", "reference-not-total", t.id, f"raised on {len(raised)} of {total} vectors",
               examples=raised[:_EXAMPLES])
    for j in wide:
        if not reads[j]:
            F.warn("correspondence", "operand-unread", t.id, 1, f"no vector read {names[j]!r}, so its words are checked on "
                   "none (`circuit_check.targets.OPERAND_PROBES` sets an index that reads it)")
    lazy = ", ".join(repr(names[j]) for j in wide)
    for name in sorted(kernels(t)):
        rec["realizations"].append({"via": f"kernel:{name}", "declines": f"a kernel takes arrays, not {lazy} read lazily"})
    return rec, None, ref


def _lowered(t: Any, opts: Options, lower_whole: bool) -> bool:
    """Whether a lowering takes ``t``'s correspondence vectors, so `floored` adds random ones to them: the Flock lowering of it
    whole (a primitive's piece; a composite under ``lower_max_gates`` gates, or Boolean) or its structural lowering.  A lowering
    takes every input; a kernel declines what is outside its domain, so it takes its floor from its own sampler (`_sampled`)."""
    if isinstance(t, ScalarReference):
        return False
    if isinstance(t, PrimitiveDefinition):
        from verity_flock import ir_lower as IL

        return bool(t.params) and (t.id in IL.PIECES or t.id.startswith("TopPMaskWordx"))
    return (lower_whole and (t.gates <= opts.lower_max_gates or is_boolean(t))) or t.gates <= opts.struct_max_gates


def _sampled(t: Any, name: str, rec: dict, opts: Options, F: Findings) -> None:
    """The floor for kernel ``name``, which ``rec`` says compared ``random`` of the shared random vectors without a fault: up to
    ``opts.random_floor`` more, sampled by the kernel's own sampler (`self_check`'s: random inputs in its domain, where the
    shared ones over many leaves almost always hold a word it declines, a NaN), while the reference's work on them is within
    ``opts.random_budget`` gate evaluations; recorded under ``sampled`` and counted in ``random``.  A mismatch fails as on the
    shared vectors; a sampler whose every vector the kernel declines leaves it compared on edge vectors only."""
    want = opts.random_floor - rec["random"]
    if want <= 0 or want * max(1, _gates(t)) > opts.random_budget:
        return
    try:
        sc = self_check(t, kernel=name, n=want, seed=opts.seed + 3)
    except Exception as e:  # noqa: BLE001  (a sampler or kernel that can't sample or take its vectors: no random one, recorded)
        rec["sampled"] = {"vectors": want, "compared": 0, "raised": f"{type(e).__name__}: {str(e)[:200]}"}
        return
    rec["sampled"] = {"vectors": sc.n, "compared": sc.compared, "mismatches": len(sc.mismatches)}
    rec["random"] += sc.compared
    if sc.mismatches:
        F.fail("correspondence", "kernel-mismatch", t.id, f"[{name}]: {len(sc.mismatches)} of the {sc.n} sampled vectors",
               via=f"kernel:{name}", examples=[{"instance": i, "args": _short(a), "kernel": _short(k), "reference": _short(r)}
                                                for i, a, k, r in sc.mismatches[:_EXAMPLES]])
    elif not sc.partial and sc.declined != sc.outside:
        F.fail("correspondence", "kernel-declined-in-domain", t.id, f"[{name}]: declined {sc.declined} of the {sc.n} sampled "
               f"vectors, {sc.outside} outside the reference's domain", via=f"kernel:{name}")


def correspondence(t: Any, opts: Options, F: Findings, *, lower_whole: bool = True) -> tuple[dict, list[np.ndarray] | None, list]:
    """The reference on the vectors (totality) and every registered kernel against it; ``None`` for the vectors when the
    Definition reads a registered table (probed, `_probed_correspondence`) or an operand over the input budget (read lazily,
    `_wide_correspondence`), or has a parameter over the input budget that is neither: a failure, but for a Boolean composite
    with a word view a warning (``correspondence/unchecked``), as its reference is its gates and `word_view` compares them.
    Where a lowering takes the vectors (`_lowered`; ``lower_whole`` as `check_definition`'s) the floor adds random ones
    (`floored`), and each kernel takes its floor from its own sampler (`_sampled`); its record's ``random`` counts the random
    vectors it compared on, at least (of the shared ones, those it compared past the all-edge ones)."""
    from .targets import table_probe

    params, _ret = signature_of(t)
    probe = table_probe(t)
    if probe is not None:
        return _probed_correspondence(t, params, probe, opts, F)
    wide = [j for j, pt in enumerate(params) if pt.leaves > opts.input_budget]
    if wide:
        why = _unread_operand(t, params, wide)
        if why is None:
            return _wide_correspondence(t, params, wide, opts, F)
        word = _word_or_none(t)
        if word is None:
            F.fail("correspondence", "unprobed-table", t.id, why)
            return {"vectors": 0, "edge_vectors": 0, "reference_raised": 0, "realizations": []}, None, []
        names = [n for n, _ in resolve(t).signature()[0]]
        over = [names[j] for j in wide]
        why = (f"operand {', '.join(f'{names[j]!r} ({params[j].leaves} leaves)' for j in wide)} over the input budget "
               f"({opts.input_budget}): a Boolean Definition's reference is its gates, total on every input, and its word view "
               f"compares them with {word.id}")
        F.warn("correspondence", "unchecked", t.id, 1, why, operands=over)
        return {"vectors": 0, "edge_vectors": 0, "reference_raised": 0, "realizations": [], "unchecked": why}, None, []
    n, edge = _n_vectors(t, opts), edge_count(params)
    n, edge = floored(t, n, edge, opts) if _lowered(t, opts, lower_whole) else (n, min(n, edge))
    arrays = vectors(params, n, opts.seed, edge)
    ref, raised = reference_outputs(t, params, arrays)
    rec: dict[str, Any] = {"vectors": n, "edge_vectors": edge, "reference_raised": len(raised), "realizations": []}
    _finite(rec, t, "correspondence", finite_rows(params, arrays, n).sum(), _finite_outputs(t, ref), len(finite_share(n, edge)), F)
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
        rec["realizations"].append({"via": f"kernel:{name}", "compared": sc.compared, "random": max(0, sc.compared - edge),
                                    "declined": sc.declined, "mismatches": len(sc.mismatches), **({"fault": sc.fault} if sc.fault else {})})
        if sc.fault == "mismatch":
            F.fail("correspondence", "kernel-mismatch", t.id, f"[{name}]: {len(sc.mismatches)} of {sc.n} vectors", via=f"kernel:{name}",
                   examples=[{"instance": i, "args": _short(a), "kernel": _short(k), "reference": _short(r)} for i, a, k, r in sc.mismatches[:_EXAMPLES]])
        elif sc.fault:
            F.fail("correspondence", f"kernel-{sc.fault}", t.id, f"[{name}]: declined {sc.declined} of {sc.n} vectors, "
                   f"{sc.outside} outside the reference's domain", via=f"kernel:{name}")
        else:
            _sampled(t, name, rec["realizations"][-1], opts, F)
    return rec, arrays, ref


#: the operand intervals a primitive's transfer is checked on (`port_intervals`)
TRANSFER_DRAWS = 8


def _one_width(t: Any) -> int | None:
    ws = t.width_set()
    return next(iter(ws)) if len(ws) == 1 else None


def _bounds(rs: Sequence, full: Callable) -> list:
    return [(lo, hi, p.bound, w) for lo, hi, p, w in rs if p.bound is not None and not full(p.bound, w)]


def _folded(arrays: list, params: Sequence, pt: Any, n: int, draw: Sequence[tuple[int, int] | None] = ()) -> list:
    """`arrays` with each parameter's leaves folded into its sampled interval (`draw[j]`), else each leaf of a bounded parameter
    into its bound: an interval's two ends first, then the vectors' values within it."""
    from verity.circuits import ports as P

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
    """The reference against what `t` declares of its return's intervals (`verity.circuits.ports`): its declared intervals,
    on `vectors` with each leaf of a bounded parameter folded into its bound; a primitive's transfer, on `TRANSFER_DRAWS`
    operand intervals (within the bounds), every leaf of an operand folded into its interval. None when it declares neither."""
    from verity.circuits import ports as P

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
    """A template evaluated through another Definition (``Subcircuit.target``) against its own Definition's reference, on the
    floored vectors (`floored`); ``random``, the random vectors it compared on."""
    from verity.circuits.evaluation import evaluate_batch

    if target is definition or getattr(target, "id", None) == getattr(definition, "id", None):
        return None
    pt, rt = signature_of(target)
    pd, rd = signature_of(definition)
    if [p.leaf_widths() for p in pt] != [p.leaf_widths() for p in pd] or rt.leaf_widths() != rd.leaf_widths():
        return {"target": target.id, "compared": 0, "reason": "the target's leaves are not the Definition's"}
    n, e = floored(definition, _n_vectors(definition, opts), edge_count(pd), opts)
    arrays = vectors(pd, n, opts.seed + 1, e)
    want, _raised = reference_outputs(definition, pd, arrays)
    rows = [i for i, w in enumerate(want) if w is not None]
    got = np.asarray(evaluate_batch(target, *[a[rows].reshape(len(rows), -1) for a in arrays])).reshape(len(rows), -1)
    bad = [(i, got[k].tolist(), want[i]) for k, i in enumerate(rows) if tuple(int(v) for v in got[k]) != want[i]]
    if bad:
        F.fail("correspondence", "target-mismatch", definition.id, f"[{target.id}]: {len(bad)} of {len(rows)} vectors", via=target.id,
               examples=[{"instance": i, "target": _short(g), "reference": _short(w)} for i, g, w in bad[:_EXAMPLES]])
    return {"target": target.id, "compared": len(rows), "random": sum(i >= e for i in rows), "mismatches": len(bad)}


def _wide_words(t: Any, words: Sequence[Any], ret: Any, params: Sequence[Any], arrays: Sequence[np.ndarray], n: int) -> np.ndarray:
    """`evaluate_words` through the bits (`verity.circuits.evaluation.bits.evaluate_bits`), in Python ints: each argument split
    into the word Definition's leaves (``words``, ``ret``) and each leaf into its bits, LSB first. For an element wider than its 64
    bits (PoUW's 576-bit sums of squares), and for a port of single bits: `Array<n, Value<1>>` is n one-bit words on the word side
    and, as `boolean.bit_type` leaves it, one n-bit element on bits, so the word view (`Value<n>`) is not the word's signature
    (`DsaWaksmanRows_v2`'s settings)."""
    from verity.circuits.evaluation.batch import type_shape
    from verity.circuits.evaluation.bits import evaluate_bits

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


def _word_or_none(t: Any) -> Any:
    """The word Definition a Boolean composite names (`verity.circuits.boolean.trace.word_of`), or None."""
    if isinstance(t, (PrimitiveDefinition, ScalarReference)) or not is_boolean(t):
        return None
    from verity.circuits.boolean.trace import word_of

    try:
        return word_of(t)
    except KeyError:
        return None


#: the word view compares a Boolean Definition with its word Definition on at least this many random vectors, whatever it costs
WORD_RANDOM_FLOOR = 2


def _word_vectors(word: Any, pw: Sequence, opts: Options) -> tuple[int, int]:
    """``(vectors, all-edge vectors)`` of the word view: ``16 * opts.vectors``, the all-edge ones first, while the word
    reference's work on them is within ``16 * opts.reference_budget`` gate evaluations; past it, as many as that pays for and at
    least ``opts.min_vectors``, at most half of them all-edge; and at least `WORD_RANDOM_FLOOR` of them random either way, so a
    Definition too large for the full count is still compared on random vectors."""
    full = 16 * opts.vectors
    n = max(opts.min_vectors, WORD_RANDOM_FLOOR, min(full, 16 * opts.reference_budget // max(1, _gates(resolve(word)))))
    return n, min(edge_count(pw), n if n == full else n // 2, n - WORD_RANDOM_FLOOR)


def word_view(t: Any, opts: Options, F: Findings) -> dict | None:
    """The word Definition a Boolean Definition names (`verity.circuits.boolean.trace.word_of`) against the Boolean one, which is
    the definition: on the word Definition's edge and random vectors (`_word_vectors`), the word Definition must equal the Boolean
    gates through the word view (`verity.circuits.evaluation.bits.evaluate_words`, which evaluates the gates)."""
    word = _word_or_none(t)
    if word is None:
        return None
    from verity.circuits.evaluation import evaluate_batch
    from verity.circuits.evaluation.bits import evaluate_words

    pw, rw = signature_of(word)
    n, edge = _word_vectors(word, pw, opts)
    arrays = vectors(pw, n, opts.seed + 2, edge)
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
    rec = {"via": f"word:{word.id}", "compared": n, "edge_vectors": min(n, edge), "random": n - min(n, edge), "mismatches": len(bad)}
    _finite(rec, t, rec["via"], finite_rows(pw, arrays, n).sum(), finite_words(got, rw.leaf_widths()).sum(),
            len(finite_share(n, rec["edge_vectors"])), F)
    return rec


# --- Boolean circuits -------------------------------------------------------------------------------------------------------

def is_commit(C: Any, v: int) -> bool:
    """Whether bit ``v`` of the `gf2.Circuit` ``C`` is a commit row (``A = form, B = 1``: `forms.Circuit.commit`,
    `sha512.commit`), the copy of a form into one bit that a builder writes so a carried form stops growing: no product."""
    return C.kind[v] == "and" and not C.data[v][1].s and bool(C.data[v][1].c)


def redundant_ands(C: Any, owner: Callable[[int], str] | None = None) -> tuple[int, Counter]:
    """Live ANDs of the `gf2.Circuit` ``C``, commits apart (`is_commit`), whose operands, each bit read as the form it stands
    for, are those of an earlier AND (in either order), are equal, or are constant: values computed again.  A commit never
    counts (the lowering check counts them as ``commits``), but reading one reads its form: a duplicate AND stands for its
    first copy, a commit of a constant or of one bit for that form, and a commit of a form committed before for the first
    commit.  Returns (count, count by ``owner(bit)``)."""
    live = C.live()
    rep: dict[int, tuple[frozenset, int]] = {}
    seen: dict[frozenset, int] = {}
    copies: dict[frozenset, tuple[int, int]] = {}
    by: Counter = Counter()

    def canon(f):
        hit = rep.keys() & f.s
        if not hit:
            return f.s, f.c
        bits, c = set(f.s - hit), f.c
        for x in hit:
            b, k = rep[x]
            bits ^= b
            c ^= k
        return frozenset(bits), c

    n = 0
    for v, kind in enumerate(C.kind):
        if kind != "and" or v not in live:
            continue
        a, b = C.data[v]
        ka = canon(a)
        if is_commit(C, v):
            if len(ka[0]) <= 1:
                rep[v] = ka
            else:
                u, c = copies.setdefault(ka[0], (v, ka[1]))
                if u != v:
                    rep[v] = (frozenset((u,)), c ^ ka[1])
            continue
        kb = canon(b)
        if not ka[0] or not kb[0] or ka[0] == kb[0]:
            n += 1
            by[owner(v) if owner else "circuit"] += 1
            continue
        key = frozenset((ka, kb))
        u = seen.get(key)
        if u is None:
            seen[key] = v
            continue
        rep[v] = (frozenset((u,)), 0)
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
    from verity.circuits.defs import REGISTRY

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
    """The Flock lowering of the whole Definition (a primitive's piece, or a small composite piece by piece) against the
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
            return {"lowered": False, "reason": IL.no_piece([t.id])}
    elif t.gates > opts.lower_max_gates and not is_boolean(t):
        return {"lowered": False, "reason": f"{t.gates} IR gates: lowered as a template's units, not whole (budget {opts.lower_max_gates})"}
    else:
        used, missing, est = _pieces_used(t, pins)
        if missing:
            return {"lowered": False, "reason": IL.no_piece(sorted(missing))}
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
        from verity.circuits.evaluation.bits import compile as compile_bits

        own = rec["definition_ands"] = compile_bits(t).counts["and"]
        commits = rec["commits"] = sum(1 for v in C.live() if is_commit(C, v))
        if counts["ands"] - commits > own:
            F.fail("lowering", "ands-added", t.id, f"[c-flock]: {counts['ands'] - commits} ANDs (and {commits} commits) for the "
                   f"Definition's {own} And gates", via="c-flock:ir_lower", ands=counts["ands"], commits=commits, definition_ands=own)
        rec["pin"] = _counts_against_pin("definitions", t.id, {"and": counts["ands"], "type": lowering_type(C)}, pins, F, t.id)
    return rec


STRUCT_VIA = "c-flock:struct_lower"
#: a family's struct budget is its key under this kind in its module's sidecar pins (``{"struct_max_gates": {family: gates}}``)
STRUCT_PINS = "struct_max_gates"


def struct_budget(t: Any, opts: Options, pins: Any) -> tuple[int, int | None]:
    """``(gates, declared)``: a composite of at most ``gates`` gates is realized by its structural lowering, the larger of
    ``opts.struct_max_gates`` and its family's struct budget, ``declared``, the ``struct_max_gates`` entry of its module's
    sidecar pins (None when there is none).  Read as a pin, so every run checks a family the same way and a target's cached
    result holds the entry by its value, or its absence."""
    from .targets import family

    budgets = pins.get(STRUCT_PINS)
    declared = None if budgets is None else budgets.get(family(t))
    return max(opts.struct_max_gates, int(declared or 0)), None if declared is None else int(declared)


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
    from verity.circuits.liveness import dead_gates as dead

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
    served Program) every violation fails by name, a member over the budget that `word.unit_rule` leaves whole too
    (``partition/unpartitioned``); elsewhere the cut of ``fn`` alone is recorded (``isolated``), since its gates
    are partitioned inside the Call that contains it.  With ``q_word=2`` (a Boolean Call: `Q_word` v2) a value computed in two
    proof blocks is no failure: ``recomputed_across`` records how many gates are (the whole cut's v2 view, else
    `word.unit_rule`'s count;
    None when ``fn`` is over the partition budget and unchecked), and every other code fails as under v1."""
    from verity.definitions import word as W

    P = F if applies else Findings()
    v2 = applies and q_word == 2
    across_rule = 0
    graphs: dict = {}
    rule = W.unit_rule(fn, opts.X, opts.W, opts.extras, memo=memo, max_gates=opts.partition_max_gates, graphs=graphs)
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
            if applies:
                P.fail("partition", "unpartitioned", d, f"{rec['reason']}, and not cut member by member")
            else:
                P.warn("partition", "unchecked", d, 1, rec["reason"])
        else:
            P.fail("partition", "width", d, f"{v['class']}: |out(S)| = {v['out_bits']} b over {v['out_gates']} gate(s), limit "
                   f"{v['limit'][0]} b or one gate of {v['limit'][1]} b", unit_class=v["class"], units=v.get("units"))
    rec.setdefault("checked", True)
    if fn.gates <= opts.partition_max_gates:
        # `unit_rule` partitions a body whose nodes read only its parameters node by node, so it never compares two nodes;
        # the whole Definition's own cut does (a value two nodes compute is a recompute across proof blocks)
        if (got := graphs.get(fn.id)) is not None:
            G, R = got
        else:
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
    from verity.circuits import cut

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


def uncut(fn: Any, opts: Options = DEFAULTS) -> str | None:
    """Why `partition` can't check a Call of ``fn`` itself, or None: its parameters' leaves are over ``partition_max_leaves`` (the
    word query and `Q_call` resolve every leaf), or it is a composite over its gate budget (``partition_max_gates``,
    `word.gate_limit`) whose body `word.unit_rule` doesn't cut member by member (`cut.separable`), or cuts so down to a member
    over its own budget that it doesn't (``too-large``)."""
    from verity.circuits.cut import separable
    from verity.definitions import word as W

    if isinstance(fn, ScalarReference):
        return None
    if (leaves := sum(pt.leaves for pt in signature_of(fn)[0])) > opts.partition_max_leaves:
        return (f"its parameters' {leaves} leaves are over the partition's leaf budget {opts.partition_max_leaves}: the word query "
                "and Q_call resolve every leaf")
    todo, seen = [fn], set()
    while todo:
        u = todo.pop()
        if u.id in seen or isinstance(u, PrimitiveDefinition) or not hasattr(u, "body") \
                or u.gates <= (limit := W.gate_limit(u, opts.partition_max_gates)):
            continue
        seen.add(u.id)
        if (sep := separable(u)) is None:
            return (f"{fn.gates} gates: over the partition budget {limit}" if u is fn else
                    f"its member {u.id}'s {u.gates} gates are over the partition budget {limit}, and not cut member by member")
        todo += [c for c, _m in sep]
    return None


def _callees(fn: Any) -> list[Any]:
    return list({n.fn.id: n.fn for n in fn.body.nodes}.values())


def _decided_call(fn: Any) -> bool:
    from .targets import Undecided, family, is_call

    try:
        return is_call(family(fn))
    except Undecided:
        return False


def callee_gap(fn: Any, opts: Options = DEFAULTS, seen: dict[str, str | None] | None = None) -> str | None:
    """Why `by_callees` can't partition a Call of ``fn``, or None when it can: ``fn`` is a primitive, which has no callees; or a
    callee is no Call (`targets.is_call`; one it can't decide without the catalog is not); or a callee is `uncut` itself and is
    a primitive (over the leaf budget: nothing partitions it), or a composite with such a gap of its own."""
    seen = {} if seen is None else seen
    if fn.id not in seen:
        seen[fn.id] = f"{fn.id} reaches itself"
        seen[fn.id] = _gap(fn, opts, seen)
    return seen[fn.id]


def _gap(fn: Any, opts: Options, seen: dict[str, str | None]) -> str | None:
    if isinstance(fn, PrimitiveDefinition) or not hasattr(fn, "body"):
        return f"{fn.id} is a primitive, with no callees to partition it"
    for c in _callees(fn):
        if not _decided_call(c):
            return f"{c.id} is no Call"
        if (why := uncut(c, opts)) is not None:
            if isinstance(c, PrimitiveDefinition):
                return f"{c.id} is a primitive and {why}"
            if (inner := callee_gap(c, opts, seen)) is not None:
                return f"{c.id}: {inner}"
    return None


def callees_are_calls(fn: Any, opts: Options = DEFAULTS, seen: dict[str, str | None] | None = None) -> bool:
    """Whether `by_callees` partitions a Call of ``fn``: `callee_gap` finds no gap."""
    return callee_gap(fn, opts, seen) is None


def by_callees(fn: Any, opts: Options, F: Findings, memo: dict, why: str) -> dict | None:
    """The partition of a Call of ``fn`` that `partition` can't check itself (``why``, `uncut`), when every Definition its body
    calls is a Call (`targets.is_call`; `targets.CALLEE_CALLS` lists the families that are one for this reason): each callee's
    partition checked as a Call's (`partition`; `q_call` for a primitive; one `uncut` itself, by its callees in turn), whose
    violations fail by the callee's name, so every gate of ``fn`` sits in a Call whose partition is checked; and, where ``fn``'s
    body nodes read only its parameters (`cut.separable`), the values two of its members compute (`cross_call.members_check`:
    each a failure under `Q_word` v1, counted under v2).  Values two callees compute in a body that isn't separable are not
    compared, as between two Calls of a Program.  ``max_out_bits`` is the widest unit output over its callees' records (a
    primitive callee's `word.unit_rule`'s: its gate's output, none for a constant, which is wired).  None, so the partition stays
    unchecked, where `callee_gap` finds a gap (its record names it).  ``memo`` keeps each callee's record and findings across
    targets."""
    from verity.circuits import boolean as BL
    from verity.circuits.cut import separable
    from verity.definitions import word as W

    if callee_gap(fn, opts, memo.setdefault("callee_gap", {})) is not None:
        return None
    done = memo.setdefault("by_callees", {})
    word = memo.setdefault("word", {})
    out: dict[str, Any] = {}
    for c in _callees(fn):
        if c.id not in done:
            G = Findings()
            if isinstance(c, PrimitiveDefinition):
                rule = W.unit_rule(c, opts.X, opts.W, opts.extras, memo=word, max_gates=opts.partition_max_gates)
                rec = {"checked": True, "max_out_bits": rule["max_out_bits"], "q_call": q_call(c, opts, G, word)}
            elif (inner := uncut(c, opts)) is not None:
                rec = by_callees(c, opts, G, memo, inner)
            else:
                rec, _ = partition(c, opts, G, word, applies=True, q_word=2 if BL.is_boolean(c) else 1)
                rec.pop("_graph", None)
            done[c.id] = (rec, G)
        rec, G = done[c.id]
        F.extend(G)
        out[c.id] = rec
    v2 = BL.is_boolean(fn)
    part: dict[str, Any] = {"applies": True, "checked": True, "reason": why, "q_word": 2 if v2 else 1,
                            "max_out_bits": max((r.get("max_out_bits") or 0 for r in out.values()), default=0), "by_callees": out}
    if separable(fn) is not None:
        from verity.definitions.cross_call import members_check

        with spent("partition-members"):
            across = members_check(fn, word)
        if v2:
            part["recomputed_across"] = across["recomputed_gates"]
        elif across["recomputes"]:
            F.fail("partition", "gate-recomputed", fn.id, f"({across['recomputed_gates']} gate(s) between node members)",
                   classes=across["recomputes"][:8])
    return part


# --- one Definition ---------------------------------------------------------------------------------------------------------

def conformance_of(t: Any) -> str | None:
    """``t``'s conformance label (`verity.circuits.defs.CONFORMANCE`), which its entry records as ``conformance``; None for a
    `ScalarReference`, which has none."""
    return None if isinstance(t, ScalarReference) else conformance_label(getattr(t, "defn", t).conformance, t.id)


def check_definition(fn: Any, opts: Options = DEFAULTS, *, memo: dict | None = None, pins: dict | None = None,
                     lower_whole: bool = True, as_call: bool | None = None) -> dict:
    """Every check on one Definition (a primitive, a bound composite) or `ScalarReference`: its entry, with its failures and
    warnings and its conformance label (`conformance_of`).  ``memo`` shares the word query's `word.unit_rule` results and the
    entries across calls; ``lower_whole=False`` leaves the Boolean lowering to a template's proof blocks; ``as_call`` says whether the partition
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
    entry: dict[str, Any] = {"definition": t.id, "form": kind, "conformance": conformance_of(t),
                             "gates": None if kind == "reference" else _gates(t)}
    with spent("correspondence"):
        corr, arrays, ref = correspondence(t, opts, F, lower_whole=lower_whole)
    params, _ = signature_of(t)
    with spent("lowering"):
        if arrays is None:
            low = {"lowered": False, "reason": "a lowering takes every input leaf, and this Definition's correspondence reads a "
                   "registered table lazily" if "table" in corr else "a lowering takes every input leaf, and this Definition's "
                   "correspondence reads an operand over the input budget lazily" if "wide" in corr else "a lowering takes every "
                   "input leaf, and this Definition has an operand over the input budget" if "unchecked" in corr else "no vectors"}
        elif lower_whole or kind != "composite":
            low = whole_lowering(t, params, arrays, ref, opts, F, pins, memo)
        else:
            low = {"lowered": False, "reason": "lowered as its template's units"}
        corr["realizations"].append({"via": "c-flock:ir_lower", **low})
        budget, declared = struct_budget(t, opts, pins)
        if kind == "composite" and arrays is not None and t.gates <= budget:
            corr["realizations"].append(struct_lowering(t, params, arrays, ref, F))
        elif kind == "composite" and declared is not None:
            corr["realizations"].append({"via": STRUCT_VIA, "lowered": False, "reason": (
                f"{t.gates} gates: over its family's struct budget, {declared} (its sidecar's {STRUCT_PINS!r})"
                if t.gates > budget else "no vectors to compare it on")})
        # a lowering compares on every vector the reference took, so its random ones are those past the all-edge ones
        late = sum(r is not None for r in ref[corr["edge_vectors"]:])
        for rec in corr["realizations"]:
            if rec.get("compared") and "random" not in rec:
                rec["random"] = late
    with spent("word-view"):
        wv = word_view(t, opts, F)
    if wv is not None:
        corr["realizations"].append(wv)
    entry["correspondence"] = corr
    with spent("ports"):
        ports = port_intervals(t, opts, F)
    if ports is not None:
        entry["ports"] = ports
    why, part = uncut(t, opts), None
    if as_call and kind != "reference" and why is not None:
        if kind == "composite":
            with spent("partition"):
                part = by_callees(t, opts, F, memo, why)
        if part is None:
            gap = callee_gap(t, opts, memo.setdefault("callee_gap", {}))
            F.fail("partition", "unpartitioned", t.id, f"{why}; {gap}")
            part = {"applies": True, "checked": False, "q_word": 2 if is_boolean(t) else 1, "reason": why, "callee_gap": gap}
    if part is not None:
        entry["partition"] = part
        entry["redundant_gates"] = {"checked": False}
        with spent("dead-gates"):
            entry["dead_gates"] = dead_gates(t, opts, F) if kind == "composite" else {"count": 0}
    elif kind != "reference" and sum(pt.leaves for pt in params) > opts.partition_max_leaves:
        F.warn("partition", "unchecked", t.id, 1, why)
        entry["partition"] = {"applies": as_call, "checked": False, "reason": why}
        entry["redundant_gates"] = {"checked": False}
        with spent("dead-gates"):
            entry["dead_gates"] = dead_gates(t, opts, F) if kind == "composite" else {"count": 0}
    elif kind == "composite":
        from verity.circuits import boolean as BL

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
    if getattr(getattr(t, "defn", t), "gated", False) or t.id in pins.get("digests", {}):
        with spent("digest"):
            entry["digest"] = circuit_digest_pin(t, pins, F)
    if kind != "reference":
        with spent("block"):
            entry["block"] = unit_block(t, F, lower_whole)
    entry["failures"], entry["warnings"] = F.failures, F.warnings
    done[(t.id, lower_whole, as_call)] = entry
    return entry


def unit_block(t: Any, F: Findings, lower_whole: bool = True) -> dict:
    """`block_range` of ``t``, failed past the range where ``t`` is checked whole (``lower_whole``)."""
    return block_range(t, F, applies=lower_whole)


def block_range(t: Any, F: Findings, applies: bool = True) -> dict:
    """The Flock block ``t`` needs as one unit, read off its Definition (`verity_flock.circuit.unit_block`: its rows'
    sha512x3 and hm96 slots and the mask beside its own slot, as `compose` places them), against Flock's proved range of
    2^``K_MAX``; past it a failure where ``t`` is checked whole (``applies``: not as a subcircuit whose units are its
    template's proof blocks).  Which of those failures are held is the report's (`summarize`, `block_held`), so no target's
    verdict reads the list.  A Call serving stages as its cut (`cut_of`) never needs its own block: each unit of the cut is
    held to the range in its place, a failure naming the Call and the unit, and the entry's ``k_log`` and ``bits`` are the
    largest unit's, its ``whole`` the Call's, its ``cut`` each unit's with how many of it."""
    from verity_flock import circuit as FC

    b = FC.unit_block(t)
    if (cut := cut_of(t)) is None:
        if applies and (why := FC.past_block(b)) is not None:
            F.fail("block", "past-range", t.id, why, **b)
        return {**b, "applies": applies}
    units = []
    for u, n in cut:
        ub = FC.unit_block(u)
        units.append({"unit": u.id, "count": n, **{k: ub[k] for k in ("ands", "bits", "k_log")}})
        if applies and (why := FC.past_block(ub)) is not None:
            F.fail("block", "past-range", t.id, f"its cut's unit {u.id}: {why}", **ub)
    top = max(units, key=lambda u: u["bits"])
    return {**b, "applies": applies, "k_log": top["k_log"], "bits": top["bits"],
            "whole": {"k_log": b["k_log"], "bits": b["bits"]}, "cut": units}


def cut_of(t: Any) -> list[tuple[Any, int]] | None:
    """The units serving stages Call ``t`` as, each Definition bound with how many of it, where the module whose function
    builds ``t`` names its family in ``CUTS`` (family -> its cut at a binding's statics, `sparse_mla.CUTS`, or the name of the
    module whose ``cut`` it is, `sampling.CUTS`); else None.  Only that module is read, which ``t``'s check reads already, and
    the module it names, imported by that name."""
    from . import cache as CC

    mod = sys.modules.get(CC._function_module(getattr(t, "defn", t)) or "")
    cut = getattr(mod, "CUTS", {}).get(getattr(t, "base_id", None))
    if isinstance(cut, str):
        cut = importlib.import_module(cut).cut
    return None if cut is None else cut(t)


def _held_blocks(entries: list[dict], catalog: bool = False) -> tuple[dict[str, str], list[dict]]:
    """Of ``entries``, the Definitions `block_held.HELD` holds past the block, each with why, and a ``block/stale-hold``
    failure for each held one checked whole whose block fits (a cut Call's, each unit of its cut); over the whole catalog
    (``catalog``: an unsharded ``--all`` or every shard of one), a ``block/absent-hold`` failure for each held id no entry
    names, which nothing is checked as any longer."""
    from .block_held import HELD

    fix = "delete its entry (tools/verity/circuit_check/src/circuit_check/block_held.py) and lower MAX"
    held, stale, seen = {}, [], set()
    for e in entries:
        d, b = e.get("definition") or e.get("target"), e.get("block") or {}
        seen.add(d)
        if d not in HELD or not b.get("applies"):
            continue
        if b["k_log"] > b["k_max"]:
            held[d] = HELD[d]
            continue
        fits = (f"each unit of its cut fits 2^{b['k_max']} (the largest 2^{b['k_log']} bits)" if "cut" in b else
                f"its block of 2^{b['k_log']} bits fits 2^{b['k_max']}")
        F = Findings()
        F.fail("block", "stale-hold", d, f"{fits}, but block_held.HELD holds it ({HELD[d]}): {fix}", **b)
        stale += F.failures
    for d in sorted(set(HELD) - seen) if catalog else ():
        F = Findings()
        F.fail("block", "absent-hold", d, f"block_held.HELD holds it ({HELD[d]}), but the catalog checks no Definition of that "
               f"id any longer: {fix}")
        stale += F.failures
    return held, stale


def circuit_digest_pin(t: Any, pins: dict, F: Findings) -> dict:
    """A gated Definition's circuit digest, core's `program_digest` of its Program, which the gate a Program holds it as
    names (`verity.protocols.certifier.verifier.head_check.gate`), against its ``digests`` pin: the pin a verifier reads in
    place of the digest is the Definition's.  Any other Definition holding a ``digests`` pin (a binding at served statics
    over `targets.MAX_GATES`, which the pin keeps in ``--all``) is held to it the same way, so every held pin is compared."""
    from verity.circuits.codec import program_digest
    from verity.circuits.program import Program

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


def _traced_owner_names(build: Callable[[], Any], n: int) -> Callable[[int], str]:
    """`_owner_names` of a traced ``build()``, built when a bit's owner is first asked for: a unit circuit the lowering built
    already is built again only to name where a redundant AND is."""
    got: list[Callable[[int], str]] = []

    def owner(v: int) -> str:
        if not got:
            got.append(_owner_names(_traced(build)[1], n))
        return got[0](v)
    return owner


def _export_counts(build: Callable[[], Any], name: str, C: Any = None) -> dict:
    """The Boolean export of one proof block's circuit (`boolean_export.CircuitTypeTable.type_of`): its circuit types' AND/XOR/NOT counts, each
    distinct sub-circuit once, every type validated and the root evaluated against the circuit (``C``, one ``build`` made
    already, else built again) on random inputs (raises on a disagreement); ``flat_ands`` the circuit's own, fewer by the ANDs
    it shares across a call (``shared_across_types``)."""
    from verity_flock import boolean_export as BE

    reg = BE.CircuitTypeTable()
    rid = reg.type_of(build, name, "proof-unit", circuit=C)
    node = reg.nodes[rid]
    _, flat, typed = reg.shared[-1]
    return {"and": node["and"], "xor": node["xor"], "not": node["not"], "subcircuit_types": len(reg.types), "flat_ands": flat,
            "shared_across_types": typed - flat}


def template_lowering(sub: Any, st: dict, d: Any, opts: Options, F: Findings, pins: dict) -> dict:
    """A subcircuit's Flock proof blocks: one proof block's circuit digest against its module's pin, its counts against
    their pin and the Boolean export, its redundant ANDs, and the laid-out proof blocks against the reference on the edge and
    random vectors (`floored`, its lanes a vector the proof blocks); ``random``, the random vectors it compared on."""
    from verity_flock import ir_lower as IL
    from verity_flock import lowerings

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
        build, C = low.units.circuit, low.circuit
    else:
        from verity_flock import ir_sampling as S

        def build():
            return S.lane_circuit(int(sub.params["V"]))
        C = None
    if C is None:
        C, root = _traced(build)
        owner = _owner_names(root, len(C.kind))
    else:
        owner = _traced_owner_names(build, len(C.kind))
    typ = lowering_type(C)
    counts = _export_counts(build, f"{name} unit", C)
    rec["export"] = counts
    if counts["flat_ands"] != rec["ands"]:
        F.fail("lowering", "export-count", sub.id, f"[{name}]: the export's circuit has {counts['flat_ands']} ANDs, the layout {rec['ands']}")
    rec["pin"] = _counts_against_pin("units", sub.id, {**{k: counts[k] for k in ("and", "xor", "not")}, "type": typ}, pins, F, sub.id)
    red, by = redundant_ands(C, owner)
    rec["redundant_ands"] = {"count": red, "where": dict(by.most_common())}
    if red:
        F.warn("redundant-gates", "boolean", sub.id, red, f"ANDs on identical operands in the {name} unit", where=dict(by.most_common()))
    if not isinstance(low, IL.Lowering):
        rec["compared"] = 0
        rec["correspondence"] = ("verity/protocols/flock/prover/tests/test_ir_sampling.py holds the unit to its lane_reference, the step at one "
                                 "vocabulary position of the row")
        return rec
    params, _ = signature_of(d)
    n, e = floored(d, max(1, min(opts.vectors, opts.lane_budget // max(1, low.units.n))), edge_count(params), opts, low.units.n)
    arrays = vectors(params, n, opts.seed + 2, e)
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
    rec.update(compared=len(rows), random=sum(i >= e for i in rows), mismatches=int(bad.sum()),
               unsatisfied_lanes=lanes - bin(ok).count("1") if lanes else 0)
    if bad.any():
        idx = np.flatnonzero(bad).tolist()
        F.fail("correspondence", "lowering-mismatch", sub.id, f"[{name}]: {len(idx)} of {len(rows)} vectors", via=f"c-flock:{name}",
               examples=[{"instance": rows[i], "lowered": _short(got[i].tolist()), "reference": _short(want[i].tolist())} for i in idx[:_EXAMPLES]])
    if rec["unsatisfied_lanes"]:
        F.fail("lowering", "unsatisfied", sub.id, f"[{name}]: {rec['unsatisfied_lanes']} honest unit lane(s) fail their rows")
    return rec


def check_subcircuit(sub: Any, opts: Options = DEFAULTS, *, instance_statics: Sequence[dict] = ({},), memo: dict | None = None,
                     pins: dict | None = None) -> dict:
    """Every check on a subcircuit: its Definition (and its evaluated target where that is another Definition) at each binding of
    its instance statics, and its Flock lowering."""
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

def check_row(path: str | Path, opts: Options = DEFAULTS, *, memo: dict | None = None, pins: dict | None = None) -> dict:
    """A served workload's request Program (a Build's ``instances.json.gz``, or its directory), as the sidecar of the package
    that writes it reads it (`targets.request_program`): `word.check_query` over its Calls (the cross-Call conditions: every
    Value a Call reads or returns is committed) and every check on each distinct Definition."""
    from verity.definitions import word as W

    from .targets import request_program

    memo = {} if memo is None else memo
    pins = load_pins() if pins is None else pins
    p = Path(path)
    P, Q, rows = request_program(str(p))
    F = Findings()
    r = W.check_query(P, Q, opts.X, opts.W, opts.extras, strict=False)
    for v in r["violations"]:
        F.fail("partition", v["class"], v["definition"], f"({v['calls']} Call(s)" + (f" under {v['where'][0]}" if v.get("where") else "") + ")",
               calls=v["calls"], where=v.get("where"))
    out: dict[str, Any] = {"row": str(p), "query": r["query"], "calls": r["calls"], "units": r["units"], "gates": r["gates"],
                           "committed_interior_words": r["committed_interior_words"], "definitions": []}
    try:
        from verity.definitions import cross_call
    except ImportError:
        out["cross_call"] = {"checked": False, "reason": "verity.definitions.cross_call is not on this tree"}
    else:
        x = cross_call.check_program(rows, strict=False)
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

#: what a target's realizations compared it with the reference on (`vector_class`), best first
VECTOR_CLASSES = ("random", "edge-only", "none")
_VECTOR_WARNINGS = {"edge-only": "its realizations compared it with the reference on all-edge vectors only, on no random one",
                    "none": "no realization compared it with the reference, so only the reference's totality was checked"}


def _corrs(r: dict) -> list[dict]:
    """``r``'s correspondences that read an input: a Definition without parameters (a constant) has no vector to sample."""
    return [c for e in (r, *r.get("definitions", ())) if (c := e.get("correspondence"))
            and not (c.get("vectors") and not c.get("edge_vectors") and "table" not in c and "wide" not in c)]


def vector_class(r: dict) -> str | None:
    """A target's result (a Definition's entry, a template's or a row's) by what its realizations compared it with the reference
    on: ``random`` when one compared it on a random vector (a realization's ``random``: those past the all-edge ones),
    ``edge-only`` when they compared it on all-edge vectors only, ``none`` when none did (no realization took the vectors or
    each declined: the reference only ran, which shows it total).  A row's is its worst Definition's.  None for a result with
    no correspondence over an input: a by-calls root (its callees are targets of their own), a constant, a structural or ambient
    check, a check that raised."""
    if r.get("mode") == "by-calls":
        return None
    if "row" in r:
        got = [c for e in r.get("definitions", ()) if (c := vector_class(e)) is not None]
        return max(got, key=VECTOR_CLASSES.index) if got else None
    corrs = _corrs(r)
    units = r["lowering"] if isinstance(r.get("lowering"), list) else []
    if not corrs and not units:
        return None
    recs = [x for c in corrs for x in c.get("realizations", ())]
    recs += [*r.get("target_equivalence", ()), *(x for x in units if x.get("lowered"))]
    if any((x.get("random") or 0) > 0 for x in recs):
        return "random"
    return "edge-only" if any((x.get("compared") or 0) > 0 for x in recs) else "none"


def classified(r: dict) -> dict:
    """A target's result with a warning when it is `vector_class` ``edge-only`` (``vectors/edge-only``) or ``none``
    (``vectors/none``): its semantics were held to no random vector, a property left unchecked, as every warning is."""
    c = vector_class(r)
    if c is None or c == "random":
        return r
    F = Findings()
    F.warn("vectors", c, r.get("definition") or r.get("subcircuit") or r.get("row") or "?", 1, _VECTOR_WARNINGS[c])
    return {**r, "warnings": [*r.get("warnings", ()), *F.warnings]}


def summarize(entries: Iterable[dict], catalog: bool = False) -> dict:
    """Totals over entries: failures by name (`ok` is none: whether one may stay red is drive's policy, never the runner's),
    warnings' counts by check and by Definition (a finding two entries share, such as a template's Definition checked on its own
    too, once), ``unchecked``: how many warnings (each a property no check enforces: it never fails) on how many Definitions,
    ``recomputed_across``: each Boolean Call (`Q_word` v2) that computes a value in two proof blocks, with its count, and
    ``q_call_refused``: each Definition `Q_call` v1 does not apply to, with why; ``by_calls``: each root checked by its calls
    (`circuit_check.by_calls`), not lowered whole, with the targets its callees rely on; ``by_callees``: each Call partitioned by
    its callees (`by_callees`), with them; ``vectors``: the targets of each `vector_class`, by name, and ``lazy``, those whose
    correspondence read a table or an operand lazily (where kernels decline and no lowering runs); ``block_held``: each
    Definition past the Flock block that `block_held.HELD` holds, with why, its ``block/past-range`` failure not counted, and
    over the whole catalog (``catalog``) each held id it no longer has failed (`_held_blocks`)."""
    entries = list(entries)
    fails: dict[str, dict] = {}
    warns: dict[str, dict] = {}
    held, stale = _held_blocks(entries, catalog)
    for e in entries:
        fails.update((f["name"], f) for f in e.get("failures", [])
                     if (f["check"], f["code"]) != ("block", "past-range") or f["definition"] not in held)
        warns.update((w["name"], w) for w in e.get("warnings", []))
    fails.update((f["name"], f) for f in stale)
    fails, warns = list(fails.values()), list(warns.values())
    by_check: Counter = Counter()
    by_def: dict[str, Counter] = {}
    for w in warns:
        k = f"{w['check']}/{w['code']}"
        by_check[k] += w["count"]
        by_def.setdefault(k, Counter())[w["definition"]] += w["count"]
    across, refused, by_calls, by_callee = {}, {}, {}, {}
    for e in entries:
        p = e.get("partition") or {}
        if "by_callees" in p:
            by_callee[e.get("definition") or e.get("target")] = sorted(p["by_callees"])
        if p.get("q_word") == 2 and p.get("recomputed_across"):
            across[e.get("definition") or e.get("target")] = p["recomputed_across"]
        if "inapplicable" in (p.get("q_call") or {}):
            refused[e.get("definition") or e.get("target")] = p["q_call"]["inapplicable"].get("why")
        if e.get("mode") == "by-calls":
            by_calls[e.get("definition") or e.get("target")] = sorted({c["binding"] for c in (e.get("by_calls") or {}).get("callees", [])
                                                                     if c.get("binding")})
    classes: dict[str, list[str]] = {c: [] for c in (*VECTOR_CLASSES, "lazy")}
    for e in entries:
        if (c := vector_class(e)) is not None:
            name = e.get("target") or e.get("definition") or e.get("subcircuit") or e.get("row")
            classes[c].append(name)
            if any("table" in x or "wide" in x for x in _corrs(e)):
                classes["lazy"].append(name)
    return {"ok": not fails, "failures": len(fails), "failure_names": [f["name"] for f in fails],
            "recomputed_across": dict(sorted(across.items())), "q_call_refused": dict(sorted(refused.items())),
            "by_calls": dict(sorted(by_calls.items())), "by_callees": dict(sorted(by_callee.items())),
            "vectors": {c: {"count": len(v), "targets": sorted(v)} for c, v in classes.items()},
            **({"block_held": dict(sorted(held.items()))} if held else {}),
            "unchecked": {"items": len(warns), "definitions": len({w["definition"] for w in warns})},
            "warnings": {k: {"total": v, "definitions": len(by_def[k]), "by_definition": dict(by_def[k].most_common())}
                         for k, v in by_check.most_common()}}


def report(targets: list[dict], opts: Options, catalog: bool = False) -> dict:
    """The report over ``targets``, each by-calls root with its callees' targets' verdicts (`circuit_check.by_calls.joined`);
    ``catalog`` where they are the whole catalog's (`summarize`)."""
    from .by_calls import joined

    targets = joined(targets)
    return {"schema": SCHEMA, "options": {k: getattr(opts, k) for k in opts.__dataclass_fields__},
            "summary": summarize(targets, catalog), "targets": targets}


__all__ = ["Findings", "Options", "SCHEMA", "VECTOR_CLASSES", "block_range", "by_callees", "callee_gap",
           "callees_are_calls", "check_definition", "check_row", "check_subcircuit", "classified", "conformance_of", "correspondence", "cut_of", "dead_gates",
           "floored", "is_commit", "load_pins", "partition", "redundant_ands", "redundant_gates", "report", "summarize",
           "target_equivalence", "template_lowering", "uncut", "unit_block", "vector_class", "whole_lowering"]
