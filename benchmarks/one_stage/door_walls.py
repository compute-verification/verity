"""The door's walls: every proof block of a registered circuit C put through the served verifier's door's static steps at
once (no commit, no prover, no coins), its templates' lowered rows held to their own evaluation, and random small
Programs put through the same steps and rows.

    door_walls.py census --circuit NAME|DIR [--blocks one-per-template|families|shapes|all|U,LO-HI,...] [--lower EXE]
                         [--units one|floor|plan] [--form unit|whole] [--cap N] [--json FILE] [--no-rows]
    door_walls.py fuzz --seed S [--cases N] [--minutes M] [--form unit|whole] [--lower EXE] [--json FILE]
    door_walls.py case NAME [--lower EXE]

``census`` is the door-ready check a lane runs on its own circuit: ``NAME`` a builder (`BUILDERS`) or ``DIR`` a
directory holding ``C.json`` (C's descriptor bytes) and ``query.json`` (its template query, ``Q_template_instances``
over every template its root calls). Each chosen unit is a proof block of its own in one partition (the gaps between
them blocks of their own, never put through the steps); a ``LO-HI`` item is the block [LO, HI) as given.
``one-per-template``: each template's first unit; ``families``: each template family's largest and smallest template
(by gates), first unit; ``shapes``: one unit of each template for each distinct wiring shape of its parameters
(`shape`); ``all``: every unit. Per block the steps are the door's own, in its order (`proofs.verify`): ``partition``
(`partition_object.validate`), ``block`` (`one_circuit.block`), ``restrict``, ``cuts`` (`one_circuit.cuts`), ``lower``
(`one_circuit.lower`: Lean's `Lower.lower` and `Compose.compose`, `flock-lower-check compose`) and ``reader``
(`proofs.tmpl_why`, the templated reader's head check). Then, once per template digest, unless ``--no-rows``, the rows
of the block's template are held to C_S's own evaluation on random values at its inputs (`block_rows`): ``yes``,
``refused honest``, ``accepted wrong (...)``, ``unlowered`` (Lean's lowering refuses it), ``too large (N gates)`` past
`ROWS_GATES`, or ``unchecked`` (a unit whose outputs are no gates of its own). It prints one table of refusal classes
(the text with numbers, names and digests normalised, so that one cause is one class) with the step, the count, the
templates hit, the rows evaluated, an example, a reading of whose wall it is, the sha256 of C's descriptor bytes and
the commit C was built at (`built_at`), and exits 1 when any chosen block is refused by a step or its rows are
refused honest or accept a wrong output, 2 when the tool itself fails. A census of the class of record runs on a node:
``research run --on auto --cwd clone --mem-gb 64 --cpus 8 --timeout 2h -- env CIRCUITS=class_of_record:families bash
benchmarks/one_stage/door_walls.sh``.

``--units`` sets each chosen block's size: ``one``, the chosen unit alone; ``floor``, from the chosen unit, the fewest
consecutive units of its template (1, 2, 4, ..., then all of its range from there) for which every step passes, each size
in a partition of its own; ``plan``, the block holding the chosen unit in the partition the real path registers
(`plan_blocks`: `IF.partition_blocks` at ``--cap`` units a member for a circuit lowered as an `IF.Circuit`, one unit a
block for Pearl+'s credited linear), every chosen block in that one partition. Under ``floor`` and ``plan`` a second
table gives each block's size, its widest row in bits as `one_circuit.lower` hands it to `Compose.compose`, its gates
(`count`), the lowered template's gates and its rows. In the runner, an item ``NAME:BLOCKS:UNITS[:FORM]``.

``--form`` sets what the door lowers: ``unit``, the door's own (`lowerBlock`: the one template the block's root calls,
each unit a copy with rows of its own); ``whole``, the block's own Program (`whole_circuit`: `restrict`'s ported block,
an internal read a reference between its nodes, called once by a root of in(S)'s runs), put through the same steps from
``cuts`` on as a circuit of one unit, so that `Lower.lower` lowers the block's root and `Compose.compose` writes rows of
in(S)'s runs and out(S)'s; its rows are held on `WHOLE_TRACES` evaluations, and a block that reads a registered value
is refused as Lean's `Spot.Restrict.ported` refuses it (class ``ported``). ``case internal_read`` puts the smallest
block with an internal read through both (`internal_case`). ``--piece EXE`` (the Spot difftest) adds, per whole block,
Lean's own `Spot.Restrict.piece`: Lean's `restrict` against Python's, and `Lower.lower` on Lean's `ported` of it.

What decides each block of a census and each block of a fuzzed case, in both forms, is `spot-one-stage-zk restrict`
(`lean_door`: Lean's `admitted`, `admittedBlock` and `lowerBlock`, asked as the accepting path asks it; under
``whole``, on the whole circuit): its verdict is ``lean_door``, Python's steps' is ``python_door`` beside it, the counts
and classes are Lean's, and a block where they disagree is a ``difference``, printed with both texts. It checks no order
of the Program's nodes, which an accept needs too. ``--no-lean-door`` leaves Python's steps deciding.

``fuzz`` draws ``N`` Programs from ``S`` (`draw`: word-typed parameters and fields of bits of widths not a multiple of
16, calls, nested calls, batches, scans, callees whose word parameters reach no word primitive, mixed widths, parts of
outputs read by a later unit, run and registered inputs) with a random partition into blocks, runs the static steps on
each block, and then holds the lowered template (Lean's `Lower.lower` at the door's groups, written by
`flock-lower-check`'s template mode, read by `tmpl.parse`) to the honest values of every unit of the block (C evaluated
by `reference.evaluate_region`). Each finding is minimised (`minimise`) and printed as a seed; the test keeps them
(``SEEDS``). At length on an idle node: ``research run --on auto --cwd clone --mem-gb 16 --cpus 4 --timeout 75m -- env
CIRCUITS=" " FUZZ_MINUTES=30 FUZZ_SEED=S bash benchmarks/one_stage/door_walls.sh``.

``case`` runs one named case through both (``lean_case``: C(a : Value 32, b : Value 1) = Not b called once at the root;
``lean_odd``: C(a : Array 3 (Value 1), b : Value 1) = Not b; their ``_wide`` forms add a wide field after ``b``).

The row check (`row_check`) has two checks. The honest one (C's values at the block's ports satisfy the rows with the
constant 1) is the completeness of `LowerEval` and `BlockEval` (`security/Properties/Flock/Circuit/Chain.lean`) together
at one trace: it fails where the rows refuse what the block computes, which neither premise says. The wrong one (the
rows on C's inputs refuse each wrong output: one flipped value bit, two value words swapped, one padding bit set) tests
`LowerEval`'s conclusion, that any assignment satisfying the rows has at each output bit the evaluation of the
Definition's lift, at those three outputs, with `BlockEval` stood in for by Python's evaluation of C (or of C_S) in
place of `Prog.evalBool` on `Flock.Lift.lift`'s Program: so a finding does not say which of the two fails; `LowerEval`
over every satisfying assignment is tested only at those outputs (exact where no unconstrained column reaches an
output column, else sampled over `LANES` assignments of those columns, and said so); `BlockEval` over every trace at
the traces drawn (one a block in a census or a fuzz case, four in ``case``); and `LowerEval`'s ports are read through the committed layout below, not `Lower.inBit` and `Lower.outBit`.

The row check's assignment is the committed one (`served_door.stage`'s): each port a row of 16-bit words, a parameter's
leaves low bit first in leaf order, padded with zeros to whole words, a cut port ``p#j`` the j-th of its ``k`` equal
slices; the output ports ``o<j>`` one per return field, likewise. A unit of 0 output bits has no row to check."""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import random
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from verity import lean_process as LP

ROOT = Path(__file__).resolve().parents[2]
LANES = 64
WORD = 16
STEPS = ("partition", "block", "restrict", "cuts", "lower", "reader", "ported", "rows")


def _exe(path: str | None) -> Callable[[], Path]:
    if path:
        return lambda: Path(path)
    from verity.protocols.proof.spot.verifier.one_stage import proofs as PF
    return PF.lower_executable


def _lean_exe(path: str | None) -> Callable[[], Path]:
    if path:
        return lambda: Path(path)
    from verity.protocols.proof.spot.verifier.one_stage import proofs as PF
    return PF.v2_executable


# --- builders --------------------------------------------------------------------------------------------------------

def _bytes(program: Any) -> bytes:
    from verity.circuits.codec import descriptor_chunks
    return b"".join(descriptor_chunks(program))


def _catalog() -> None:
    from verity.circuits import program as PG
    from verity.circuits.ops.boolean import versions as V
    V.load()
    PG.register_boolean_bodies(V.boolean_version)


#: by builder, the `IF.Circuit` it lowered and its wired Program and query, whose plan (`plan_blocks`) the census reads
_CIRCUITS: dict[str, tuple[Any, tuple[Any, dict]]] = {}
#: the units of a plan's member at most: the stated cap of `prove_path.py` and of the served window (`served_door.py window`)
CAP = 64


def _held(name: str, circuit: Any) -> tuple[Any, dict]:
    built = circuit.program()
    _CIRCUITS[name] = (circuit, built)
    return built


def plan_blocks(name: str, cap: int = CAP) -> tuple[list[tuple[int, int]], str]:
    """The builder's blocks as the real path groups its units, and the code that makes them: `IF.partition_blocks` for an
    `IF.Circuit` (each template's units cut at its wires' and registered reads' ends, then into members of at most
    ``cap``), one unit a block for Pearl+'s credited linear (`credited_in_c.py`'s stage). `Unchecked` for a builder with no
    plan."""
    if name == "pearl_plus":
        from verity.circuits import partition_object as PO
        from verity.circuits import restrict as R
        c, q = load(name)
        pop = PO.TemplatePartition(R.program(c), q).population
        return [(u, u + 1) for u in range(pop)], "benchmarks/pouw/credited_in_c.py's stage: one unit a block"
    if name not in _CIRCUITS:
        load(name)
    if name not in _CIRCUITS:
        raise Unchecked(f"{name}: no registered partition or plan groups its units")
    from verity.commitments.periods import interface as IF
    circuit, built = _CIRCUITS[name]
    return ([(lo, hi) for lo, hi in IF.partition_blocks(circuit, cap, built)],
            f"verity.commitments.periods.interface.partition_blocks(circuit, cap={cap})")


def one_linear() -> tuple[bytes, dict]:
    """`one_linear.py`'s registered circuit: 288 ``Gemm_v2{K=576, N=8}`` units at Ampere's dot, ``x`` external, ``w`` the
    registered weight's rows."""
    from verity.circuits.codec import descriptor_id
    from verity.circuits.defs import bind
    from verity.circuits.ops import gemm as G
    from verity.commitments.periods import interface as I
    tokens, k, e, n = 4, 576, 576, 8
    g = bind(G.Gemm, K=k, N=n, **G.GEMM_AMPERE)
    gid = descriptor_id(g)
    reg = I.Registered({"weights": e * k}, reads=(I.Read(gid, "w", "weights", 0, (tokens, e // n), 0, (0, n * k)),))
    program, query = _held("one_linear", I.Circuit("one_linear", ((g, tokens * e // n),), external=(f"{gid}/x",),
                                                   registered=reg))
    return _bytes(program), query


def chain() -> tuple[bytes, dict]:
    """`test_one_circuit_chain`'s C: the periods' chain of eight ``Gemm_v2{K=16, N=16}`` units, one wire, theta registered."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "periods_chain", ROOT / "tests/verity/protocols/proof/spot/verifier/test_one_circuit_chain.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    program, query = _held("chain", m.circuit())
    return _bytes(program), query


def _vllm():
    sys.path.insert(0, str(ROOT / "tests/adapters/vllm"))
    from tests.program.predict import test_steps as S
    from verity_vllm.program.predict import window_circuit as WC
    return S, WC


def test_class() -> tuple[bytes, dict]:
    """The served window's class circuit at the vLLM suite's test class: TINY, `WindowClass(2, 4, 2)`, its grain
    (`test_window_circuit`'s ``window`` fixture)."""
    S, WC = _vllm()
    c = WC.class_circuit(S._registered(), WC.WindowClass(2, 4, 2), None, WC.Grain(gemm_words=1024, rows=256, select=64))
    program, query = _held("test_class", c)
    return _bytes(program), query


def class_of_record() -> tuple[bytes, dict]:
    """The served circuit of the class of record (`served_door.py build`): Llama 3.2 1B, class 4,27,2, four requests."""
    S, WC = _vllm()
    cfg = json.loads((ROOT / "tests/adapters/vllm/tests/program/predict/fixtures/llama32-1b.config.json").read_text())
    reg = S.Registered.of(S.Phi(model={"repo": "org/m", "revision": "r", "hf_config": cfg}, target=dict(S.RTXPRO6000)),
                          default_seed=7)
    c = WC.lower(reg, WC.WindowClass(4, 27, 2), None, requests=[WC.Request(0, j, 27, 2) for j in range(4)]).circuit
    program, query = _held("class_of_record", c)
    return _bytes(program), query


def _composite(name: str, params, ret, body) -> Any:
    from verity.circuits.defs import CompositeDefinition, bind
    return bind(CompositeDefinition(name, 1, (), lambda S: (tuple(params), ret), body, register=False))


def _rooted(callee: Any, params, ret, registered: Sequence[str] = ()) -> tuple[bytes, dict]:
    from verity.circuits import partition_object as PO
    from verity.circuits.program import Program
    root = _composite("WallsRoot", params, ret, lambda B, S, *xs: B.call(callee, *xs))
    return _bytes(Program(root, registered=list(registered))), PO.template_instances_query([callee])


def lean_case() -> tuple[bytes, dict]:
    """lean's case: C(a : Value 32, b : Value 1) = Not b, called once at the root."""
    from verity.circuits import boolean as BL
    from verity.circuits.types import Value
    ps = (("a", Value(32)), ("b", Value(1)))
    return _rooted(_composite("WallsLeanC", ps, Value(1), lambda B, S, a, b: B.call(BL.NOT, b)), ps, Value(1))


def lean_case_wide() -> tuple[bytes, dict]:
    """lean's wide case: C(a : Array 64 (Value 32), b : Value 1) = Not b, past the compose refusal of the narrow one."""
    from verity.circuits import boolean as BL
    from verity.circuits.types import Array, Value
    ps = (("a", Array(64, Value(32))), ("b", Value(1)))
    return _rooted(_composite("WallsLeanWide", ps, Value(1), lambda B, S, a, b: B.call(BL.NOT, b)), ps, Value(1))


def pearl_plus() -> tuple[bytes, dict]:
    """Pearl+'s credited linear (`credited_in_c.py`'s C at K,TM,TN,N,M = 256,1,1,64,2, with its reader), laid under
    the composer's limit as its stage lays it. Its reader adds rows of N = 64 BF16 words: a reader of one word a row
    has no row that spans a compression before its last block, which the composer refuses, and no path of record lays
    one."""
    import importlib.util
    from verity.circuits import laying as LY
    spec = importlib.util.spec_from_file_location("credited_in_c", ROOT / "benchmarks/pouw/credited_in_c.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    program = LY.lay(m.linear(256, 2, 64, (1, 1), True).program, m.fits)
    return _bytes(program), LY.query(program)


def _accounting() -> Any:
    import importlib.util
    spec = importlib.util.spec_from_file_location("door_walls_circuits",
                                                  ROOT / "tests/benchmarks/one_stage/door_walls_circuits.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def pous_setup() -> tuple[bytes, dict]:
    """Storage's store blocks: PoUS's setup C at its test's point (`window.Setup(64, 189, 16, 4, 4)`), every unit one
    `PousP2SetupCheck_v2`."""
    program, query = _held("pous_setup", _accounting().pous_setup())
    return _bytes(program), query


def internal_read() -> tuple[bytes, dict]:
    """The smallest block with an internal read (`door_walls_circuits.internal_read`): T(T(a, w1), w2), two units."""
    program, query = _accounting().internal_read()
    return _bytes(program), query


def lean_odd() -> tuple[bytes, dict]:
    """lean's second case: C(a : Array 3 (Value 1), b : Value 1) = Not b, a field of 3 bits followed by another."""
    from verity.circuits import boolean as BL
    from verity.circuits.types import Array, Value
    ps = (("a", Array(3, Value(1))), ("b", Value(1)))
    return _rooted(_composite("WallsLeanOdd", ps, Value(1), lambda B, S, a, b: B.call(BL.NOT, b)), ps, Value(1))


def lean_odd_wide() -> tuple[bytes, dict]:
    """lean's second case past compose: C(a : Array 3 (Value 1), b : Value 1, c : Array 2048 (Value 1)) = Not b."""
    from verity.circuits import boolean as BL
    from verity.circuits.types import Array, Value
    ps = (("a", Array(3, Value(1))), ("b", Value(1)), ("c", Array(2048, Value(1))))
    return _rooted(_composite("WallsLeanOddWide", ps, Value(1), lambda B, S, a, b, c: B.call(BL.NOT, b)), ps, Value(1))


BUILDERS: dict[str, Callable[[], tuple[bytes, dict]]] = {
    "one_linear": one_linear, "chain": chain, "test_class": test_class, "class_of_record": class_of_record,
    "pearl_plus": pearl_plus, "pous_setup": pous_setup,
    "lean_case": lean_case, "lean_case_wide": lean_case_wide, "lean_odd": lean_odd,
    "lean_odd_wide": lean_odd_wide, "internal_read": internal_read}


def load(spec: str) -> tuple[bytes, dict]:
    """C's bytes and its template query from a builder's name or a directory holding ``C.json`` and ``query.json``."""
    _catalog()
    if spec in BUILDERS:
        return BUILDERS[spec]()
    d = Path(spec)
    return (d / "C.json").read_bytes(), json.loads((d / "query.json").read_text())


# --- the door's static steps -----------------------------------------------------------------------------------------

_NORMS = [
    (re.compile(r"[0-9a-f]{16,}"), "<H>"),
    (re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:\[[^\]]*\])?_v\d+(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})?"), "<T>"),
    (re.compile(r"/[\w./-]+"), "<path>"),
    (re.compile(r"-?\d+"), "<n>"),
    (re.compile(r"\[(?:<n>|<T>|'<T>'|[\s,()])*\]"), "[..]"),
    (re.compile(r"\s+"), " "),
]


def normalise(text: str) -> str:
    """One class per cause: digests, Definition ids, paths, numbers and lists of them replaced, at most 220 characters."""
    for pat, rep in _NORMS:
        text = pat.sub(rep, text)
    return text.strip()[:220]


def owner(step: str, text: str) -> str:
    """A reading of whose wall a refusal is, by the step and the code that refused it."""
    if step == "lean":
        m = re.search(r"\(Spot\.OneStageZk\.admitted\), ([a-z-]+):", text)
        return f"lean's door (spot-one-stage-zk restrict{', ' + m.group(1) if m else ''})"
    if step == "ported":
        return "lean's ported (Spot.Restrict.ported)"
    if step == "rows":
        return "lean's lower (Flock.Lower)"
    if step == "lower" and (m := re.search(r"\(Spot\.OneStageZk\.admitted\), ([a-z-]+):", text)):
        return f"proofs' door predicate (Spot.OneStageZk.admitted, {m.group(1)})"
    if step == "lower" and "flock-lower-check" in text:
        if "lift" in text:
            return "lean's lift (Flock.Lift)"
        if "compose" in text and ("compression" in text or "row" in text or "block" in text):
            return "lean's compose (Flock.Compose)"
        return "lean's lower (Flock.Lower)"
    if step in ("lower", "cuts") and "not derived yet" in text:
        return "proofs' door (a named hole, one_circuit.NotDerived)"
    if step in ("lower", "reader"):
        return "proofs' door (one_circuit.lower, proofs.tmpl_why)"
    if step == "cuts" and "query-refused:" in text:
        return "proofs' door (one_circuit.cuts, Partition.checkTemplates)"
    if step == "cuts":
        return "circuits' edges (edges.io_cuts) at proofs' door"
    if step == "restrict":
        return "circuits' restrict (verity.circuits.restrict)"
    if step in ("partition", "block"):
        return "circuits' partition object / the lane's partition"
    return "unread"


@dataclass
class Result:
    """One block through the steps: the step it stopped at (None: through every step), the refusal's text and class,
    the block's template(s) and shape, and the composed file and C_S where reached."""
    block: int
    units: tuple[int, int]
    templates: tuple[str, ...]
    shape: str = ""
    step: str | None = None
    text: str = ""
    seconds: dict = field(default_factory=dict)
    c_s: bytes | None = field(default=None, repr=False)
    circuit: bytes | None = field(default=None, repr=False)
    key: str | None = None
    rows: str = "not reached"
    row_bits: int | None = None
    gates: int | str | None = None
    tried: list = field(default_factory=list)
    cut: dict = field(default_factory=dict, repr=False)
    tmpl_gates: int | str | None = None
    ported: str = ""
    piece: str = ""
    ports: tuple = field(default=((), ()), repr=False)
    lean_door: str = ""

    @property
    def klass(self) -> str:
        """The block's class by what decided it: `spot-one-stage-zk restrict` (``lean: ...``) where it was asked, else
        Python's steps."""
        if self.lean_door:
            return "" if self.lean_door == ADMITTED else f"lean: {normalise(self.lean_door)}"
        return self.python_klass

    @property
    def python_klass(self) -> str:
        """The step's refusal; under the whole form, a block every step takes but Lean's `ported` refuses is that."""
        if self.step is not None:
            return f"{self.step}: {normalise(self.text)}"
        return f"ported: {normalise(self.ported)}" if self.ported not in ("", "yes") else ""

    @property
    def python_door(self) -> str:
        """Python's door: its static steps (and, under the whole form, the mirror of Lean's `ported`)."""
        if self.step is not None:
            return f"refused: {self.text.strip()}"
        return ADMITTED if self.ported in ("", "yes") else f"refused: {self.ported}"

    @property
    def decider(self) -> str:
        return LEAN_DOOR if self.lean_door else "python's steps"

    @property
    def difference(self) -> str:
        """Where Lean's door and Python's disagree: who admits what the other refuses, or both clauses; "" where they
        agree or Lean's door was not asked."""
        if not self.lean_door or self.lean_door.startswith("unasked"):
            return ""
        if self.lean_door.startswith(CRASHED):
            return f"lean's door gave no answer ({self.lean_door[len(CRASHED):]})"
        lean, py = self.lean_door, self.python_door
        if lean == ADMITTED and py == ADMITTED:
            return ""
        if lean == ADMITTED:
            return f"lean admits, python refuses at {self.step or 'ported'} ({clause(py)})"
        if py == ADMITTED:
            return f"python admits, lean refuses ({clause(lean)})"
        a, b = clause(lean), clause(py)
        return "" if a == b else f"both refuse, clauses differ: lean {a}; python {b}"


ADMITTED = "admitted"
#: what decides a block where Lean's door was asked: it checks no order of the Program's nodes, which an accept also
#: needs (the partition's order check, `Partition.evaluate`)
LEAN_DOOR = "spot-one-stage-zk restrict"
#: `spot-one-stage-zk restrict` (`Spot/OneStageZk.lean`'s `main`): 0 C_S written, 1 the door's refusal, 2 a partition
#: or a block it can't read
RESTRICT_EXITS = frozenset({0, 1, 2})
#: Lean's door where it wrote a panic line or ended at a status not in `RESTRICT_EXITS`: no answer, so a difference
CRASHED = "crashed: "


def clause(text: str) -> str:
    """A refusal's clause: the door predicate's clause name where it names one, else its text with the tool's own
    wrappers (its error code, `flock-lower-check compose (n)`) cut, normalised."""
    from verity.protocols.proof.spot.verifier.one_stage import one_circuit as OC
    i = text.find(OC.NOT_ADMITTED)
    if i >= 0:
        return text[i + len(OC.NOT_ADMITTED):].lstrip(", ").split(":")[0]
    t = text.removeprefix("refused: ").strip()
    t = re.sub(r"^(Refused: )?(E_[A-Z_]+: )?(restrict: )?", "", t)
    t = re.sub(r"^flock-lower-check compose \(\d+\): (flock-lower-check compose: )?", "", t)
    return normalise(t.splitlines()[0] if t else t)


def lean_door(exe: Callable[[], Path], c_file: Path, partition: dict, first: int, d: Path) -> str:
    """Lean's door, as the accepting path asks it (`spot-one-stage-zk restrict C partition first out`: `Own.circuitFile`,
    `restrictOn`, `lowerBlock` under `admitted` and `admittedBlock`): ``admitted`` or ``refused: <its text>``."""
    from verity.circuits import partition_object as PO
    pp, out = d / "partition.json", d / "circuit.txt"
    pp.write_bytes(PO.canonical(partition))
    out.unlink(missing_ok=True)
    p = LP.run([str(exe()), "restrict", str(c_file), str(pp), str(first), str(out)], ok_statuses=RESTRICT_EXITS)
    if p.returncode == 0 and out.exists():
        return ADMITTED
    return f"refused: {(p.stderr or p.stdout).strip()[:1500]}"


def _err(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"


def sizes(c_s: bytes, cut: dict) -> tuple[int | None, int | str]:
    """C_S's widest row in bits as `one_circuit.lower` hands it to `Compose.compose` (an input port's bits, an output
    port's 16-bit words), and its root's two-input gates (`count`); None where C_S calls other than one template."""
    from verity.circuits import restrict as R
    from verity.circuits.codec import descriptor_id
    from verity.circuits.count import count
    from verity.circuits.program import input_id
    from verity.protocols.proof.spot.verifier.one_stage.one_circuit import _fields
    D = R.program(c_s)
    called = sorted({descriptor_id(nd.fn) for nd in D.root.body.nodes if input_id(nd.fn.id) is None})
    try:
        gates = count(D)[descriptor_id(D.root)].gates
    except Exception as e:  # noqa: BLE001 - a block the count can't walk still gets its row width
        gates = f"uncounted: {_err(e)[:120]}"
    if len(called) != 1:
        return None, gates
    params, ret = D.defs[called[0]].signature()
    ks = cut.get(called[0], {})
    rows = [sum(t.leaf_widths()) // max(1, ks.get(n, 1)) for n, t in params]
    rows += [16 * -(-sum(f.leaf_widths()) // 16) for f in _fields(ret)]
    return max(rows, default=0), gates


def template_gates(c_s: bytes) -> int | str | None:
    """The two-input gates (`count`) of the one Definition C_S's root calls, the template the door lowers; None where it
    calls other than one, and why where the count can't walk it."""
    from verity.circuits import restrict as R
    from verity.circuits.codec import descriptor_id
    from verity.circuits.count import count
    from verity.circuits.program import input_id
    D = R.program(c_s)
    called = sorted({descriptor_id(nd.fn) for nd in D.root.body.nodes if input_id(nd.fn.id) is None})
    try:
        return count(D)[called[0]].gates if len(called) == 1 else None
    except Exception as e:  # noqa: BLE001 - a template the count can't walk has no gate count, and the census goes on
        return f"uncounted: {_err(e)[:120]}"


FORMS = ("unit", "whole")


def whole_circuit(c: bytes, partition: dict, units: Sequence[tuple[int, int]], D: Any) -> tuple[bytes, dict, str, tuple]:
    """The block's own Program as a circuit of one unit: `restrict`'s ported block (`Block_v1`, its parameters in(S)'s
    runs, `Spot.Restrict.ported`'s form, its internal reads references between its nodes) called once by a root whose
    parameters bind as in(S)'s; its template query; what Lean's `ported` says of it (``yes``, or its refusal of a
    registered read, word for word); and in(S) and out(S) as C's gates."""
    from verity.circuits import partition_object as PO
    from verity.circuits import program as P
    from verity.circuits import restrict as R
    from verity.circuits.program import REGISTRATION
    b = R.restrict(c, partition, [tuple(u) for u in units], D=D)
    fn = b.program.fn
    params, ret = fn.signature()
    names = [n for n, _t in params]
    bound = dict(zip(names, b.program.bindings))
    ported = "yes"
    if reg := next((t for n, t in params if bound[n] == REGISTRATION), None):
        ported = (f"ported: the block reads a registered value (RegisteredInput{reg.elem.w}_v1), and a parameter does not "
                  "carry that it is registered: its rows would be committed afresh")
    root = _composite("Whole", params, ret, lambda B, S, *xs: B.call(fn, *xs))
    saved = list(P._BODY_LOOKUP)

    def lookup(p):
        try:
            return D.boolean[p.id]
        except KeyError:
            raise LookupError(f"C names no Boolean body for {p.id}") from None

    P.register_boolean_bodies(lookup)
    try:
        prog = P.Program(root, registered=[n for n in names if bound[n] == REGISTRATION], supplied=b.program.supplied)
    finally:
        P._BODY_LOOKUP[:] = saved
    return _bytes(prog), PO.template_instances_query([fn]), ported, (b.inputs, b.outputs)


def lean_piece(server: Any, c: bytes, partition: dict, units: Sequence[tuple[int, int]], decoded: Any) -> str:
    """Lean's `Spot.Restrict.piece` on the block (the Spot difftest's ``restrict``): ``lowers`` where Lean's `restrict`
    is Python's and `Flock.Lower.lower` lowers Lean's `ported` of it at the front's port widths, else what differs or
    the lowering's refusal."""
    from verity.protocols.proof.spot.verifier.one_stage import one_circuit as OC
    us = [u for lo, hi in units for u in range(lo, hi)]
    try:
        py = OC.restrict(c, partition, units, decoded).decode()
        kind, body = server.call("restrict", {"descriptor": c.decode(), "units": us, "python": py})
    except Exception as e:  # noqa: BLE001 - a census records the cross-check's crash by its text
        return f"crashed: {_err(e)[:300]}"
    if kind != "output":
        return f"{kind}: {str(body)[:300]}"
    out = json.loads(body)
    if not out["agree"]:
        return f"restrict differs at {out['at']}"
    return "lowers" if out["lowers"] is None else f"refuses: {out['lowers'][:300]}"


class Door:
    """The door's static steps on C and one partition of it, each run once where it is the partition's and not a block's.
    Under ``form="whole"`` each block's steps from ``cuts`` on run on `whole_circuit` of it, a partition of its one unit."""

    def __init__(self, c: bytes, query: dict, blocks: Sequence[tuple[int, int]], exe: Callable[[], Path],
                 form: str = "unit", piece: Any = None, lean: Callable[[], Path] | None = None):
        from verity.circuits import partition_object as PO
        from verity.protocols.proof.spot.verifier.one_stage import one_circuit as OC
        self.c, self.query, self.exe, self.form, self.piece, self.lean = c, query, exe, form, piece, lean
        self._dir: tempfile.TemporaryDirectory | None = None
        self.decoded = OC.Decoded(c)
        self.tp = PO.TemplatePartition(self.decoded.get(), query)
        self.partition = PO.build(json.loads(c), query, [list(blocks)])
        v = PO.validate(self.partition)
        self.partition_why = None if v.ok else f"the partition object is not valid: {list(v.codes)}"
        self._cuts: tuple[Any, str | None] | None = None

    def cuts(self) -> tuple[Any, str | None]:
        from verity.protocols.proof.spot.verifier.one_stage import one_circuit as OC
        if self._cuts is None:
            try:
                self._cuts = (OC.cuts(self.c, self.partition, self.decoded), None)
            except Exception as e:  # noqa: BLE001 - a census records every refusal and crash by its text
                self._cuts = (None, _err(e))
        return self._cuts

    def templates(self, lo: int, hi: int) -> tuple[str, ...]:
        return tuple(sorted({t for _k, t, a, n in self.tp.ranges if a < hi and lo < a + n}))

    def lean_door(self, lo: int) -> str:
        """Lean's door on the block starting at unit ``lo`` of this door's C and partition."""
        if self._dir is None:
            self._dir = tempfile.TemporaryDirectory(prefix="door-walls-lean-")
            Path(self._dir.name, "C.json").write_bytes(self.c)
        d = Path(self._dir.name)
        try:
            return lean_door(self.lean, d / "C.json", self.partition, lo, d)
        except LP.LeanPanic as e:
            return f"{CRASHED}{str(e)[:300]}"
        except Exception as e:  # noqa: BLE001 - a census records the executable's crash by its text
            return f"unasked: {_err(e)[:300]}"

    def run(self, k: int, lo: int, hi: int, shape: str = "") -> Result:
        """Python's steps, then, with ``lean``, Lean's door on the same block (under ``whole``, on its whole circuit)."""
        r = self._steps(k, lo, hi, shape)
        if self.lean is not None and self.form == "unit":
            t = time.perf_counter()
            r.lean_door = self.lean_door(lo)
            r.seconds["lean_door"] = round(time.perf_counter() - t, 3)
        return r

    def _steps(self, k: int, lo: int, hi: int, shape: str = "") -> Result:
        from verity.protocols.proof.spot.verifier.one_stage import one_circuit as OC
        from verity.protocols.proof.spot.verifier.one_stage import proofs as PF
        r = Result(k, (lo, hi), self.templates(lo, hi), shape)

        def stop(step: str, text: str) -> Result:
            r.step, r.text = step, text
            return r

        if self.partition_why:
            return stop("partition", self.partition_why)
        t = time.perf_counter()
        try:
            units = OC.block(self.c, self.partition, k, self.decoded)
        except Exception as e:  # noqa: BLE001
            return stop("block", _err(e))
        r.seconds["block"] = round(time.perf_counter() - t, 3)
        if self.form == "whole":
            return self._whole(r, units)
        t = time.perf_counter()
        try:
            r.c_s = OC.restrict(self.c, self.partition, units, self.decoded)
        except Exception as e:  # noqa: BLE001
            return stop("restrict", _err(e))
        r.seconds["restrict"] = round(time.perf_counter() - t, 3)
        cut, why = self.cuts()
        if why is not None:
            return stop("cuts", why)
        r.cut = cut or {}
        r.row_bits, r.gates = sizes(r.c_s, cut or {})
        r.tmpl_gates = template_gates(r.c_s)
        t = time.perf_counter()
        try:
            r.circuit = OC.lower(r.c_s, self.partition, self.exe, cut=cut,
                                 formed=OC.formed_in(self.decoded, self.partition, units))
        except Exception as e:  # noqa: BLE001
            return stop("lower", _err(e))
        r.seconds["lower"] = round(time.perf_counter() - t, 3)
        with tempfile.TemporaryDirectory(prefix="door-walls-") as d:
            f = Path(d, "circuit.txt")
            f.write_bytes(r.circuit)
            why = PF.tmpl_why(f)
        if why is not None:
            return stop("reader", why)
        return r

    def _whole(self, r: Result, units: list) -> Result:
        t = time.perf_counter()
        try:
            w, q, r.ported, r.ports = whole_circuit(self.c, self.partition, units, self.decoded.get())
        except Exception as e:  # noqa: BLE001
            r.step, r.text = "restrict", _err(e)
            if self.lean is not None:
                r.lean_door = "unasked: the block has no whole circuit"
            return r
        r.seconds["restrict"] = round(time.perf_counter() - t, 3)
        if self.piece is not None:
            r.piece = lean_piece(self.piece, self.c, self.partition, units, self.decoded)
        try:
            inner = Door(w, q, [(0, 1)], self.exe, lean=self.lean).run(0, 0, 1)
        except Exception as e:  # noqa: BLE001 - the whole block's partition refused is the door's partition step
            r.step, r.text = "partition", _err(e)
            if self.lean is not None:
                r.lean_door = "unasked: the whole circuit's partition is refused"
            return r
        r.lean_door = inner.lean_door
        r.step, r.text, r.c_s, r.circuit, r.cut = inner.step, inner.text, inner.c_s, inner.circuit, inner.cut
        r.row_bits, r.gates, r.tmpl_gates = inner.row_bits, inner.gates, inner.tmpl_gates
        r.seconds |= {k: v for k, v in inner.seconds.items() if k != "block"}
        return r


# --- block choices and wiring shapes ---------------------------------------------------------------------------------

def shape(tp: Any, u: int) -> str:
    """Unit ``u``'s parameters by what their leaves read: ``reg`` (a registered input), ``sup`` (a supplied one), ``run``
    (a fresh run input), ``w1``/``wN`` (one / several writer units), ``part`` (a strict part of a writer unit's
    outputs), ``mixed`` (inputs and writers), joined in parameter order."""
    from verity.circuits.program import input_id
    D = tp.program
    body = D.root.body
    offs = list(body.offsets)
    unit_at = {k: (a, n) for k, _t, a, n in tp.ranges}
    k, _m = tp.locate(u)
    node = body.nodes[k]
    params = node.fn.signature()[0]
    gates = tp.inputs(u)
    out, at = [], 0
    for _name, t in params:
        n = t.leaves
        kinds, writers = set(), {}
        for g in gates[at:at + n]:
            kk = bisect.bisect_right(offs, g) - 1
            fn = body.nodes[kk].fn
            inp = input_id(fn.id)
            if inp is not None:
                kinds.add({"registration": "reg", "supplied": "sup"}.get(inp[0], "run"))
                continue
            a, cnt = unit_at.get(kk, (None, 0))
            if a is None:
                kinds.add("node")
                continue
            per = body.nodes[kk].fn.gates
            v = a + (g - offs[kk]) // per
            writers.setdefault(v, set()).add(g)
        at += n
        if writers:
            part = any(set(gs) != set(tp.outputs(v)) for v, gs in writers.items())
            kinds.add(("w1" if len(writers) == 1 else "wN") + ("+part" if part else ""))
        out.append("mixed" if len(kinds) > 1 else (kinds.pop() if kinds else "none"))
    return ",".join(out)


def choose(c: bytes, query: dict, how: str) -> list[tuple[int, int, str]]:
    """The blocks a census puts through the door: ``(lo, hi, shape)``."""
    from verity.circuits import partition_object as PO
    from verity.circuits import restrict as R
    tp = PO.TemplatePartition(R.program(c), query)
    first: dict[str, int] = {}
    for _k, t, a, n in tp.ranges:
        if n and t not in first:
            first[t] = a
    if how == "one-per-template":
        us = sorted(first.values())
    elif how == "families":
        D = tp.program
        size = {t: D.defs[t].gates if hasattr(D.defs[t], "gates") else 0 for t in first}
        fam: dict[str, list[str]] = defaultdict(list)
        for t in first:
            fam[t.split("{")[0]].append(t)
        us = sorted({first[x] for ts in fam.values() for x in (min(ts, key=size.get), max(ts, key=size.get))})
    elif how == "shapes":
        seen, us = set(), []
        for u in range(tp.population):
            t = next(t for _k, t, a, n in tp.ranges if a <= u < a + n)
            s = shape(tp, u)
            if (t, s) not in seen:
                seen.add((t, s))
                us.append(u)
    elif how == "all":
        us = list(range(tp.population))
    else:
        out = []
        for item in how.split(","):
            lo, _, hi = item.partition("-")
            out.append((int(lo), int(hi) if hi else int(lo) + 1))
        return [(lo, hi, shape(tp, lo) if hi == lo + 1 else "block") for lo, hi in sorted(out)]
    return [(u, u + 1, shape(tp, u)) for u in us]


def tiling(chosen: Sequence[tuple[int, int, str]], population: int) -> tuple[list[tuple[int, int]], list[int]]:
    """One level of blocks covering the population: each chosen block, the gaps between them blocks of their own; and
    each chosen block's index in it."""
    blocks, ks, at = [], [], 0
    for lo, hi, _s in chosen:
        if lo > at:
            blocks.append((at, lo))
        ks.append(len(blocks))
        blocks.append((lo, hi))
        at = hi
    if at < population:
        blocks.append((at, population))
    return blocks, ks


#: the gates of C_S past which the census evaluates no rows (the reference evaluator walks every gate of C_S)
ROWS_GATES = 1 << 22


def _words(fn: Any, cut: dict) -> tuple[int, int]:
    """The template's input and output port words at the door's groups."""
    ins_p, outs_p = ports(fn, cut)
    params, ret = fn.signature()
    in_w = [w for _n, t in params for w in t.leaf_widths()]
    out_w = list(ret.leaf_widths())
    return (sum(-(-sum(in_w[j] for j in lv) // WORD) for _p, lv in ins_p),
            sum(-(-sum(out_w[j] for j in lv) // WORD) for _p, lv in outs_p))


#: the evaluations of C_S a whole block's rows are held to: its one unit is one instance an evaluation
WHOLE_TRACES = 4


def block_rows(c_s: bytes, cuts: dict, exe: Callable[[], Path], seed: int = 0, instances: int = 4, traces: int = 1) -> str:
    """The rows of C_S's template held to C_S's own evaluation on random values at its inputs (up to ``instances``
    of its units, on each of ``traces`` evaluations): ``yes``, ``refused honest``, ``accepted wrong (...)``, ``too large
    (N gates)``, ``unlowered`` or ``unchecked``."""
    from verity.circuits import partition_object as PO
    from verity.circuits import restrict as R
    from verity.circuits.codec import descriptor_id
    from verity.circuits.program import input_id
    D = R.program(c_s)
    called = sorted({descriptor_id(nd.fn) for nd in D.root.body.nodes if input_id(nd.fn.id) is None})
    if len(called) != 1:
        return f"unchecked: C_S calls {len(called)} templates"
    fn, cut = D.defs[called[0]], cuts.get(called[0], {})
    gates = _evaluable(c_s).circuit.size
    if gates > ROWS_GATES:
        return f"too large ({gates} gates)"
    rng = random.Random(seed)
    try:
        t, _log = template_of(c_s, called[0], *_words(fn, cut), exe)
        q = PO.template_instances_query([fn])
        pop = PO.TemplatePartition(D, q).population
        xs, ys, vb = [], [], []
        for _ in range(traces):
            x, y, b = honest_rows(c_s, q, range(min(pop, instances)), fn, cut, trace_of(c_s, rng))
            xs, ys, vb = xs + x, ys + y, vb + b
        v = row_check(t, xs, ys, vb, rng)
    except Unlowered:
        return "unlowered"
    except Unchecked as e:
        return f"unchecked: {e}"
    except Exception as e:  # noqa: BLE001 - a census records every crash by its text
        return f"crashed: {_err(e)[:200]}"
    if not v.honest_ok:
        return "refused honest" + (f"; accepted wrong ({', '.join(v.wrong_ok)})" if v.wrong_ok else "")
    if v.wrong_ok:
        return f"accepted wrong ({', '.join(v.wrong_ok)}{'' if v.exact else ', sampled'})"
    return "yes" if v.exact else "yes (sampled)"


UNITS = ("one", "floor", "plan")


def census(c: bytes, query: dict, how: str, exe: Callable[[], Path], log=None, rows: bool = True, units: str = "one",
           plan: Sequence[tuple[int, int]] | None = None, form: str = "unit", piece: Any = None,
           only: str = "", lean: Callable[[], Path] | None = None) -> list[Result]:
    """Each chosen block through the door's static steps and, with ``rows``, its template's rows (`block_rows`, once
    per template digest and block size). ``units``: ``one``, each chosen unit a block; ``plan``, the block of ``plan``
    (one level of the population) holding each chosen unit, all in that one partition; ``floor``, from each chosen unit
    ``u``, the fewest consecutive units ``k`` of its template, doubling from 1 up to the units of its range from ``u``
    (that last tried too), for which every step passes, each ``k`` its own partition (`floor_of`). ``form``: ``unit``,
    the door's (the one template the block's root calls, lowered); ``whole``, the block's own Program (`whole_circuit`),
    its rows held on `WHOLE_TRACES` evaluations, once per digest of its C_S."""
    from verity.circuits import partition_object as PO
    from verity.circuits import restrict as R
    chosen = choose(c, query, how)
    tp = PO.TemplatePartition(R.program(c), query)
    if only:
        at = {u: t for _k, t, a, n in tp.ranges for u in (a,)}
        names = sorted(at)
        chosen = [x for x in chosen if re.search(only, at[names[bisect.bisect_right(names, x[0]) - 1]])]
    seen: dict[tuple, str] = {}

    def rows_of(r: Result) -> None:
        if rows and r.c_s is not None:
            key = (hashlib.sha256(r.c_s).hexdigest(),) if form == "whole" else ("|".join(r.templates), r.units[1] - r.units[0])
            if key not in seen:
                seen[key] = block_rows(r.c_s, r.cut, exe, traces=WHOLE_TRACES if form == "whole" else 1)
            r.rows = seen[key]
        r.c_s = r.circuit = None

    def said(r: Result) -> Result:
        if log:
            log(json.dumps({"block": r.block, "units": list(r.units), "templates": list(r.templates), "shape": r.shape,
                            "decider": r.decider, "lean_door": r.lean_door[:600], "python_door": r.python_door[:600],
                            "python_step": r.step, "difference": r.difference, "rows": r.rows, "row_bits": r.row_bits,
                            "gates": r.gates, "tmpl_gates": r.tmpl_gates, "ported": r.ported, "piece": r.piece,
                            "tried": r.tried,
                            "seconds": r.seconds}))
        return r

    out = []
    if units == "floor":
        return [said(r) for r in floors(c, query, tp, chosen, exe, rows_of, form, piece, lean)]
    if units == "plan":
        if plan is None:
            raise ValueError("--units plan needs the circuit's plan (`plan_blocks`)")
        blocks = sorted(plan)
        starts = [lo for lo, _ in blocks]
        door = Door(c, query, blocks, exe, form, piece, lean)
        done = set()
        for lo, _hi, s in chosen:
            k = bisect.bisect_right(starts, lo) - 1
            if k in done:
                continue
            done.add(k)
            r = door.run(k, *blocks[k], s)
            rows_of(r)
            out.append(said(r))
        return out
    blocks, ks = tiling(chosen, tp.population)
    door = Door(c, query, blocks, exe, form, piece, lean)
    for (lo, hi, s), k in zip(chosen, ks):
        r = door.run(k, lo, hi, s)
        rows_of(r)
        out.append(said(r))
    return out


def floors(c: bytes, query: dict, tp: Any, chosen: Sequence[tuple[int, int, str]], exe: Callable[[], Path],
           rows_of, form: str = "unit", piece: Any = None, lean: Callable[[], Path] | None = None) -> list[Result]:
    """For each chosen unit ``u``, the block [u, u + k) at the fewest ``k`` (1, 2, 4, ..., then the units of u's range
    from u, short of the next chosen unit; under ``whole``, whose block may hold several templates, then `CAP` units
    from u, the population's end short of that) that passes every step, else at the largest tried; its ``tried`` lists
    each ``k`` with its class (``through``) and widest row. Round ``i`` puts every unpassed unit's ``i``-th size through
    as few partitions as hold its blocks apart."""
    starts = sorted(lo for lo, _hi, _s in chosen)
    ks = []
    for lo, _hi, _s in chosen:
        _k, _t, a, n = next(x for x in tp.ranges if x[2] <= lo < x[2] + x[3])
        nxt = starts[bisect.bisect_right(starts, lo)] if bisect.bisect_right(starts, lo) < len(starts) else tp.population
        most, k, mine = (min(tp.population, lo + CAP) if form == "whole" else min(a + n, nxt)) - lo, 1, []
        while k < most:
            mine.append(k)
            k *= 2
        ks.append(mine + [most])
    last: list[Result | None] = [None] * len(chosen)
    tried: list[list] = [[] for _ in chosen]
    for i in range(max(map(len, ks), default=0)):
        live = [j for j in range(len(chosen)) if i < len(ks[j]) and (last[j] is None or last[j].klass)]
        if not live:
            break
        trial = sorted((chosen[j][0], chosen[j][0] + ks[j][i], chosen[j][2], j) for j in live)
        groups: list[list] = []
        for x in trial:
            g = next((g for g in groups if g[-1][1] <= x[0]), None)
            if g is None:
                groups.append([x])
            else:
                g.append(x)
        for g in groups:
            blocks, at = tiling([(lo, hi, s) for lo, hi, s, _j in g], tp.population)
            door = Door(c, query, blocks, exe, form, piece, lean)
            for (lo, hi, s, j), k in zip(g, at):
                r = door.run(k, lo, hi, s)
                r.circuit = None
                tried[j].append([hi - lo, r.klass or "through", r.row_bits])
                last[j] = r
            del door
    for j, r in enumerate(last):
        r.tried = tried[j]
        rows_of(r)
    return last


def table(results: Sequence[Result], c_sha256: str = "", built: str = "") -> list[dict]:
    """The census's table: one row per refusal class by what decided it (``decider``: `spot-one-stage-zk restrict`
    where it was asked, else Python's steps; ``through`` for blocks it admitted), each naming the sha256 of C's
    descriptor bytes and the commit C was built at."""
    rows: dict[str, dict] = {}
    for r in results:
        key = r.klass or "through"
        step = "-" if key == "through" else "lean" if r.decider == LEAN_DOOR else r.step or "ported"
        row = rows.setdefault(key, {"class": key, "decider": r.decider, "step": step, "count": 0, "templates": [],
                                    "shapes": [], "python_classes": {}, "differences": 0,
                                    "example": {"block": r.block, "units": list(r.units), "templates": list(r.templates),
                                                "text": (r.lean_door or r.text)[:400]},
                                    "owner": owner(step, r.text) if step != "-" else "-", "rows": {},
                                    "c_sha256": c_sha256, "built": built})
        row["count"] += 1
        row["rows"][r.rows] = row["rows"].get(r.rows, 0) + 1
        pk = r.python_klass or "through"
        row["python_classes"][pk] = row["python_classes"].get(pk, 0) + 1
        row["differences"] += bool(r.difference)
        for t in r.templates:
            if t not in row["templates"]:
                row["templates"].append(t)
        if r.shape not in row["shapes"]:
            row["shapes"].append(r.shape)
    return sorted(rows.values(), key=lambda x: (x["class"] == "through", STEPS.index(x["step"]) if x["step"] in STEPS else 9,
                                                -x["count"]))


def render(rows: Sequence[dict]) -> str:
    lines = ["| class (by the decider) | decider | step | count | python's door (blocks) | differences | templates hit "
             "| rows evaluated (blocks) | example (block, units, template) | owner (reading) | C sha256 | built at |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ex = r["example"]
        ts = ", ".join(r["templates"][:4]) + (f" (+{len(r['templates']) - 4})" if len(r["templates"]) > 4 else "")
        rv = "; ".join(f"{k} ({n})" for k, n in r["rows"].items())
        pv = "; ".join(f"{k[:80]} ({n})" for k, n in r["python_classes"].items())
        lines.append(f"| {r['class'][:160]} | {r['decider']} | {r['step']} | {r['count']} | {pv} | {r['differences']} "
                     f"| {ts} | {rv} | block {ex['block']}, units "
                     f"{ex['units']}, {ex['templates'][0] if ex['templates'] else '-'} | {r['owner']} | {r['c_sha256']} "
                     f"| {r['built']} |")
    return "\n".join(lines)


def floor_table(results: Sequence[Result]) -> list[dict]:
    """Per block of a ``floor`` or ``plan`` census: its template, the units at which it passes every step (``never, up
    to N``), its widest row in bits and gates at that size, its rows, and the class it stopped at."""
    out = []
    for r in results:
        k = r.units[1] - r.units[0]
        out.append({"template": r.templates[0] if len(r.templates) == 1 else "|".join(r.templates), "first": r.units[0],
                    "passes_at": k if not r.klass else f"never, up to {k}", "row_bits": r.row_bits,
                    "gates": r.gates, "tmpl_gates": r.tmpl_gates, "ported": r.ported, "piece": r.piece, "rows": r.rows,
                    "class": r.klass or "through", "tried": r.tried})
    return out


def render_floor(rows: Sequence[dict]) -> str:
    lines = ["| template | units per block it passes at | widest row (bits) | block's gates | lowered template's gates "
             "| rows | stopped at |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['template']} | {r['passes_at']} | {r['row_bits']} | {r['gates']} | {r['tmpl_gates']} "
                     f"| {r['rows']} | {r['class'][:120]} |")
    return "\n".join(lines)


def built_at(one_circuit: str = "") -> str:
    """The commit of the tree that built C, and the `one-circuit` commit it holds: the merge-base with that ref where the
    clone has it, else ``one_circuit`` (``--one-circuit``)."""
    def git(*a: str) -> str:
        p = subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else ""
    head = git("rev-parse", "HEAD") or "unknown"
    oc = next((m for ref in ("origin/one-circuit", "one-circuit") if (m := git("merge-base", "HEAD", ref))),
              one_circuit)
    return f"{head} (one-circuit {oc})" if oc else head


# --- the row check ---------------------------------------------------------------------------------------------------

class Unchecked(RuntimeError):
    """The block's rows can't be held to its values: its unit's outputs are no gates of its own (a pass-through)."""


class Unlowered(Unchecked):
    """Lean's `Lower.lower` refuses the template: it has no rows to check."""


#: `flock-lower-check DESC KEY INS OUTS IN TMPL OUT` (`FlockLowerCheck.main`): 0 the template written, 1 its refusal, 2
#: its usage
LOWER_EXITS = frozenset({0, 1, 2})


def template_of(c_s: bytes, key: str, n_in: int, n_out: int, exe: Callable[[], Path]) -> tuple[Any, str]:
    """C_S's template ``key`` lowered as the door lowers it (`Lower.lower` at one 16-bit group a port word), written by
    `flock-lower-check`'s template mode and parsed (`tmpl.parse`); and the tool's output. Raises `Unlowered` when Lean
    refuses to lower it."""
    from verity_flock import tmpl as T
    with tempfile.TemporaryDirectory(prefix="door-walls-") as d:
        d = Path(d)
        (d / "c_s.json").write_bytes(c_s)
        (d / "in.txt").write_text("0\n" * (WORD * n_in))
        p = LP.run([str(exe()), str(d / "c_s.json"), key, ",".join([str(WORD)] * n_in), ",".join([str(WORD)] * n_out),
                    str(d / "in.txt"), str(d / "t.tmpl"), str(d / "out.txt")], ok_statuses=LOWER_EXITS)
        if not (d / "t.tmpl").exists():
            raise Unlowered(f"flock-lower-check ({p.returncode}): {(p.stderr or p.stdout)[-1500:]}")
        return T.parse((d / "t.tmpl").read_text()), p.stdout + p.stderr


def _written(t: Any) -> set[int]:
    """The columns some row of the template writes (`tmpl.unroll`'s placement): every other column past the inputs is
    unconstrained (`TmplEval.SatAt` holds only the rows written), whatever its flat row reads as."""
    out: set[int] = set()

    def place(d: int, o: int) -> None:
        df = t.defs[d]
        out.update(o + col for col, _a, _b in df.rows)
        for c in df.calls:
            for j in range(c.count):
                place(c.e, o + c.base + j * c.stride)

    place(t.root, 0)
    return out


@dataclass
class Rows:
    """A template unrolled: its input columns, its constant, each written column's two XOR sums, its unconstrained
    columns, and the output columns."""
    n_in: int
    const: int
    a: list
    b: list
    free: list[int]
    ins: list[int]
    outs: list[int]

    @classmethod
    def of(cls, t: Any) -> "Rows":
        from verity_flock import tmpl as T
        f = T.unroll(t)
        n_in = T._ports(t)[0]
        written = _written(t)
        a, b, free = [], [], []
        for col in range(f.useful):
            x, y = f.row(col)
            a.append(tuple(x))
            b.append(tuple(y))
            if col >= n_in and col != f.const and col not in written:
                free.append(col)
        ins, outs = _parts(t.groups)
        return cls(n_in, f.const, a, b, free, ins, outs)

    def solve(self, ins: np.ndarray, free: np.ndarray | None = None) -> np.ndarray:
        """Every column on 64 lanes, in column order: inputs, the constant all ones, a written column (A·z)(B·z), an
        unconstrained one ``free``'s word (0 when None)."""
        ones = np.uint64(2 ** 64 - 1)
        z = np.zeros(len(self.a), np.uint64)
        z[:len(ins)] = ins
        z[self.const] = ones
        fw = dict(zip(self.free, free)) if free is not None else {}
        for c in range(self.n_in, len(self.a)):
            if c == self.const:
                continue
            x, y = self.a[c], self.b[c]
            if x and y:
                z[c] = np.bitwise_xor.reduce(z[list(x)]) & np.bitwise_xor.reduce(z[list(y)])
            else:
                z[c] = fw.get(c, np.uint64(0))
        return z

    def reach(self) -> set[int]:
        """The unconstrained columns some output column depends on."""
        dep: list[frozenset] = []
        free = set(self.free)
        for c in range(len(self.a)):
            if c in free:
                dep.append(frozenset([c]))
            elif c < self.n_in or c == self.const:
                dep.append(frozenset())
            else:
                s = set()
                for o in (*self.a[c], *self.b[c]):
                    if o < c:
                        s |= dep[o]
                dep.append(frozenset(s))
        return set().union(*[dep[o] for o in self.outs if o < len(dep)]) if self.outs else set()


def _parts(groups: Sequence[str]) -> tuple[list[int], list[int]]:
    """The template header's input and output groups (each its first column and words, in `tmpl.WORD`s, and its parts'
    bits) as the columns of the port bits, in order."""
    from verity_flock import tmpl as T
    g, at, out = list(groups), 0, []
    for _side in range(2):
        cols, n = [], int(g[at])
        at += 1
        for _ in range(n):
            col, nb = int(g[at]), int(g[at + 2])
            base = col * T.WORD
            for size in map(int, g[at + 3:at + 3 + nb]):
                cols += range(base, base + size)
                base += size
            at += 3 + nb
        out.append(cols)
    return out[0], out[1]


def _bits(widths: Sequence[int], values: Sequence[int]) -> list[int]:
    return [(v >> j) & 1 for v, w in zip(values, widths) for j in range(w)]


def _padded(bits: list[int]) -> list[int]:
    return bits + [0] * (-len(bits) % WORD)


def ports(fn: Any, cut: dict[str, int]) -> tuple[list[tuple[str, list[int]]], list[tuple[str, list[int]]]]:
    """The template's input rows and output rows as leaf-index lists over its parameters' and return's leaves, in
    the door's order (`one_circuit.lower`): each parameter, or its ``k`` slices; each return field."""
    from verity.protocols.proof.spot.verifier.one_stage.one_circuit import _fields
    params, ret = fn.signature()
    ins, at = [], 0
    for name, t in params:
        n, k = t.leaves, cut.get(name, 1)
        if k <= 1:
            ins.append((name, list(range(at, at + n))))
        else:
            ins += [(f"{name}#{j}", list(range(at + j * n // k, at + (j + 1) * n // k))) for j in range(k)]
        at += n
    outs, at = [], 0
    for j, f in enumerate(_fields(ret)):
        outs.append((f"o{j}", list(range(at, at + f.leaves))))
        at += f.leaves
    return ins, outs


@dataclass
class Verdict:
    honest_ok: bool
    wrong_ok: list[str]
    exact: bool
    free_reaching: int
    detail: str = ""


#: the unrolled rows past which the row check reads no unconstrained column (`Rows`): the template's own evaluation only
SMALL = 1 << 17


def row_check(t: Any, ins_words: list[list[int]], honest: list[list[int]], value_bits: list[list[bool]],
              rng: random.Random) -> Verdict:
    """Each instance's honest output (``honest``, its padded output bits) held to the template's rows on its input bits,
    then wrong outputs of each instance: one flipped value bit, two value words swapped, one padding bit set
    (``value_bits``: which output bits are values). The rows' outputs are the template's own evaluation
    (`tmpl.evaluate`); on a template of at most `SMALL` rows, an output column with no row takes any value, and where
    an unconstrained column reaches an output the outputs are sampled over `LANES` assignments of those columns."""
    from verity_flock import tmpl as T
    ins_cols, out_cols = _parts(t.groups)
    if len(out_cols) != len(honest[0]):
        return Verdict(False, [], True, 0, f"the template has {len(out_cols)} output port bits, the ports {len(honest[0])}")
    if len(ins_cols) != len(ins_words[0]):
        return Verdict(False, [], True, 0, f"the template has {len(ins_cols)} input port bits, the ports {len(ins_words[0])}")
    small = T.totals(t)["rows"] <= SMALL
    rows = Rows.of(t) if small else None
    free = set(rows.free) if rows else set()
    fixed = [j for j, col in enumerate(out_cols) if col not in free]
    reach = (rows.reach() - {out_cols[j] for j in range(len(out_cols)) if j not in fixed}) if rows else set()
    exact = small and not reach
    ones = np.uint64(2 ** 64 - 1)
    outs = [out_cols[j] for j in fixed]
    n = len(ins_words)
    X = T.slice_lanes(np.asarray(ins_words, np.uint8))
    Y = T.evaluate(t, X, np.asarray(outs, np.int64)) if outs else np.zeros((0, X.shape[1]), np.uint64)
    computed = [tuple(int((int(Y[b, i // 64]) >> (i % 64)) & 1) for b in range(len(outs))) for i in range(n)]

    def reachable(inst: int) -> set[tuple[int, ...]]:
        if not reach:
            return {computed[inst]}
        got = {computed[inst]}
        lane = np.zeros(rows.n_in, np.uint64)
        for col, bit in zip(rows.ins, ins_words[inst]):
            lane[col] = ones if bit else np.uint64(0)
        for _ in range(LANES):
            fw = np.asarray([rng.getrandbits(64) for _ in rows.free], np.uint64)
            o = rows.solve(lane, fw)[outs]
            for lb in range(LANES):
                got.add(tuple(int((int(v) >> lb) & 1) for v in o))
        return got

    def seen(w: Sequence[int]) -> tuple[int, ...]:
        return tuple(w[j] for j in fixed)

    honest_ok, wrong_ok = True, []
    detail = (f"{len(out_cols) - len(fixed)} of {len(out_cols)} output port bits have no row; "
              if len(fixed) < len(out_cols) else "") + ("" if small else "large: no unconstrained-column analysis; ")
    first = ""
    for i in range(n):
        got = reachable(i)
        h = tuple(honest[i])
        if seen(h) not in got:
            honest_ok = False
            g = sorted(got)[0]
            first = first or (f"instance {i}: honest output bits differ from the rows' at "
                              f"{[fixed[j] for j in range(len(fixed)) if seen(h)[j] != g[j]][:12]}; ")
        vals = [j for j, v in enumerate(value_bits[i]) if v]
        pads = [j for j, v in enumerate(value_bits[i]) if not v]
        wrongs = []
        if vals:
            j = rng.choice(vals)
            w = list(h)
            w[j] ^= 1
            wrongs.append(("flipped value bit", tuple(w)))
        words = [h[j:j + WORD] for j in range(0, len(h), WORD)]
        diffw = [(p, q) for p in range(len(words)) for q in range(p + 1, len(words)) if words[p] != words[q]]
        if diffw:
            p, q = rng.choice(diffw)
            ws = list(words)
            ws[p], ws[q] = ws[q], ws[p]
            wrongs.append(("swapped words", tuple(b for wd in ws for b in wd)))
        if pads:
            j = rng.choice(pads)
            w = list(h)
            w[j] ^= 1
            wrongs.append(("padding bit set", tuple(w)))
        for what, w in wrongs:
            if seen(w) in got and what not in wrong_ok:
                wrong_ok.append(what)
    return Verdict(honest_ok, wrong_ok, exact, len(reach), first + detail)


def honest_rows(c: bytes, query: dict, units: Sequence[int], fn: Any, cut: dict, trace: dict[int, int]) \
        -> tuple[list[list[int]], list[list[int]], list[list[bool]]]:
    """The block's instances' input bits and padded output bits (the committed assignment) from C's trace."""
    from verity.circuits import partition_object as PO
    from verity.circuits import restrict as R
    tp = PO.TemplatePartition(R.program(c), query)
    out_w = list(fn.signature()[1].leaf_widths())
    xs, ys, vb = [], [], []
    for u in units:
        a = [trace[g] for g in tp.inputs(u)]
        o = [trace[g] for g in tp.outputs(u)]
        if len(o) != len(out_w):
            raise Unchecked(f"unit {u} has {len(o)} output gates of its own for {len(out_w)} return leaves")
        x, y, b = _assigned(fn, cut, a, o)
        xs.append(x)
        ys.append(y)
        vb.append(b)
    return xs, ys, vb


def _assigned(fn: Any, cut: dict, a: Sequence[int], o: Sequence[int]) -> tuple[list[int], list[int], list[bool]]:
    """One instance's committed assignment: its input bits, its padded output bits and which of those are values, from
    its parameters' leaf values ``a`` and its return's ``o``."""
    params, ret = fn.signature()
    in_w = [w for _n, t in params for w in t.leaf_widths()]
    out_w = list(ret.leaf_widths())
    ins_p, outs_p = ports(fn, cut)
    row: list[int] = []
    for _p, leaves in ins_p:
        row += _padded(_bits([in_w[j] for j in leaves], [a[j] for j in leaves]))
    out: list[int] = []
    val: list[bool] = []
    for _p, leaves in outs_p:
        bits = _bits([out_w[j] for j in leaves], [o[j] for j in leaves])
        pad = _padded(bits)
        out += pad
        val += [True] * len(bits) + [False] * (len(pad) - len(bits))
    return row, out, val


def trace_of(c: bytes, rng: random.Random) -> dict[int, int]:
    """C evaluated gate by gate (`reference.evaluate_region`) on random input gates."""
    from verity.circuits.evaluation.reference import evaluate_region
    from verity.circuits.program import input_id
    D = _evaluable(c)
    C = D.circuit
    t: dict[int, int] = {}
    for g in C.gates():
        if (inp := input_id(g.prim.id)) is not None:
            t[g.index] = rng.getrandbits(inp[1])
    return evaluate_region(C, C[0:C.size], t)


def _evaluable(c: bytes) -> Any:
    """C decoded with the catalog's evaluators (`restrict.program` declares primitives for their structure only)."""
    from verity.circuits.codec import decode_program
    from verity.circuits.defs import REGISTRY
    return decode_program(json.loads(c), REGISTRY)


# --- the fuzzer ------------------------------------------------------------------------------------------------------

def _prims():
    from verity.circuits import boolean as BL
    from verity.circuits.ops import words as W
    from verity.circuits.types import Value
    v = Value
    return [
        (BL.AND, (v(1), v(1)), v(1)), (BL.XOR, (v(1), v(1)), v(1)), (BL.NOT, (v(1),), v(1)),
        (W.PouwXor32, (v(32), v(32)), v(32)), (W.PouwChi32, (v(32), v(32), v(32)), v(32)),
        (W.PouwAdd8, (v(7), v(8)), v(8)), (W.PouwAdd64Lo, (v(32), v(32)), v(32)),
    ]


#: a parameter's type: ``w`` > 0 is ``Value(w)``, ``w`` < 0 is ``Array(-w, Value(1))``, a field of bits
TYPES = (1, 7, 8, 32, -3, -5, -16)
#: a step that packs 16 bit values into one field of 16 bits
PACK = -1_000_000
#: a word-mode template's parameter types (`_word_template`): fields of bits in whole 16-bit words
WORD_TYPES = (-16, -16, -32)
#: the share of the fuzzer's cases drawn in the door's admitted form (`draw(words=True)`)
WORDS = 0.5
#: the share of the fuzzer's cases drawn as a block in which one unit reads another (`draw_inner`)
INNER = 0.3
#: the bits of the trailing field a drawn template takes with probability 0.4, so that its unit is wide enough to compose
WIDE = 2048


def _ty(w: int):
    from verity.circuits.types import Array, Value
    return Value(w) if w > 0 else Array(-w, Value(1))


def _value_leaves(w: int) -> list[int]:
    return [w] if w > 0 else [1] * -w


@dataclass
class Spec:
    """A drawn case, small enough to print and replay: its templates (each a list of params, a body of steps and a
    return), and its root (each node a template call, a batch of n, or a scan of n, and its arguments' sources)."""
    templates: list
    root: list
    root_params: list
    registered: list
    blocks: list

    def doc(self) -> dict:
        return {"templates": self.templates, "root": self.root, "root_params": self.root_params,
                "registered": self.registered, "blocks": self.blocks}


def _template(rng: random.Random, prims: list, words: bool = False) -> dict:
    if words:
        return _word_template(rng)
    params = [rng.choice(TYPES) for _ in range(rng.randint(1, 3))] + ([-WIDE] if rng.random() < 0.4 else [])
    vals = list(enumerate(params))
    steps = []
    for _s in range(rng.randint(0 if rng.random() < 0.15 else 1, 4)):
        ok = [i for i, (p, ts, r) in enumerate(prims) if all(any(w == t.w for _v, w in vals) for t in ts)]
        fields = [(v, w) for v, w in vals if w < 0]
        if fields and (not ok or rng.random() < 0.4):
            v, w = rng.choice(fields)
            steps.append([-1 - rng.randrange(-w), [v]])
            vals.append((len(vals), 1))
            continue
        if not ok:
            break
        pi = rng.choice(ok)
        _p, ts, r = prims[pi]
        args = [rng.choice([v for v, w in vals if w == t.w]) for t in ts]
        steps.append([pi, args])
        vals.append((len(vals), r.w))
    nested = bool(steps) and rng.random() < 0.25
    ret = rng.sample([v for v, _w in vals], k=min(len(vals), rng.randint(1, 2)))
    return {"params": params, "steps": steps, "ret": ret, "nested": nested}


def _word_template(rng: random.Random) -> dict:
    """A template in the form the door admits: fields of bits in whole 16-bit words, bit steps on picked bits, and a
    return of one or two `PACK`ed fields of 16 of its bits (or a parameter passed through)."""
    params = [rng.choice(WORD_TYPES) for _ in range(rng.randint(1, 3))]
    vals = list(enumerate(params))
    steps = []
    for _s in range(rng.randint(1, 6)):
        bits = [v for v, w in vals if w == 1]
        if len(bits) < 2 or rng.random() < 0.4:
            v, w = rng.choice([(v, w) for v, w in vals if w < 0])
            steps.append([-1 - rng.randrange(-w), [v]])
        else:
            pi = rng.randrange(3)
            steps.append([pi, rng.sample(bits, 1 if pi == 2 else 2)])
        vals.append((len(vals), 1))
    bits = [v for v, w in vals if w == 1]
    ret = []
    for _f in range(rng.randint(1, 2)):
        if rng.random() < 0.2:
            ret.append(rng.randrange(len(params)))
            continue
        steps.append([PACK, [rng.choice(bits) for _ in range(16)]])
        vals.append((len(vals), -16))
        ret.append(len(vals) - 1)
    return {"params": params, "steps": steps, "ret": ret, "nested": rng.random() < 0.25}


def draw_inner(rng: random.Random, words: bool = False) -> Spec:
    """A block of 2-3 calls of one template in which each later call reads a whole return field of an earlier one
    wherever a parameter's width is that field's (a field of bits as a run ``["r", node, leaf, n]``), the whole
    population one block: a unit that reads another unit of its block."""
    prims = _prims()
    for _ in range(50):
        T = _template(rng, prims, words)
        if any(w in _ret_widths(T) for w in T["params"]):
            break
    fields, at = [], 0
    for w in _ret_widths(T):
        fields.append((at, w))
        at += len(_value_leaves(w))
    root_params: list[int] = []
    root, avail = [], []
    for ni in range(rng.randint(2, 3)):
        srcs = []
        for w in T["params"]:
            cands = [(nd, a) for nd, a, fw in avail if fw == w]
            if cands:
                nd, a = rng.choice(cands)
                srcs.append([["n", nd, a] if w > 0 else ["r", nd, a, -w]])
            else:
                root_params.append(w)
                srcs.append([["p", len(root_params) - 1]])
        root.append({"t": 0, "form": "call", "n": 1, "args": srcs})
        avail += [(ni, a, w) for a, w in fields]
    return Spec([T], root, root_params, [], [[0, len(root)]])


def draw(rng: random.Random, small: bool = False, words: bool = False) -> Spec:
    """A random case: 1-3 templates, each 1-3 parameters of `TYPES` (and sometimes a field of `WIDE` bits after them), a body of 0-4 steps (a primitive, or step ``-1-j``
    taking bit ``j`` of a field of bits), or a call of a nested callee, returning 1-2 values (a parameter passed through among them, so a word parameter may reach no word
    primitive); a root of 1-4 nodes, each a call, a batch or a scan of a template, each argument a root parameter of
    its width (run or registered), or an output leaf of an earlier node of that width (a part of its outputs)."""
    prims = _prims()
    nt = rng.randint(1, 2 if small else 3)
    templates = []
    for ti in range(nt):
        templates.append(_template(rng, prims, words))
    root_params: list[int] = []
    registered: list[int] = []
    root = []
    avail: list[tuple[int, int, int]] = []
    for ni in range(rng.randint(1, 2 if small else 4)):
        ti = rng.randrange(nt)
        T = templates[ti]
        form = rng.choice(["call", "call", "batch", "scan"]) if not small else rng.choice(["call", "batch"])
        n = rng.randint(2, 3) if form in ("batch", "scan") else 1
        if form == "scan":
            rw = _ret_widths(T)
            if not (len(rw) == 2 and rw[0] == T["params"][0]):
                form = "batch"
        srcs = []
        for pj, w in enumerate(T["params"]):
            k = 1 if form == "call" or (form == "scan" and pj == 0) else n
            src = []
            for _m in range(k):
                cands = [a for a in avail if a[2] == w]
                if cands and rng.random() < 0.6:
                    nd, leaf, _w = rng.choice(cands)
                    src.append(["n", nd, leaf])
                else:
                    root_params.append(w)
                    if rng.random() < 0.25:
                        registered.append(len(root_params) - 1)
                    src.append(["p", len(root_params) - 1])
            srcs.append(src)
        root.append({"t": ti, "form": form, "n": n, "args": srcs})
        rw = _ret_widths(T)
        one = [w for x in rw for w in _value_leaves(x)]
        if form == "call":
            avail += [(ni, j, w) for j, w in enumerate(one)]
        elif form == "batch":
            avail += [(ni, m * len(one) + j, w) for m in range(n) for j, w in enumerate(one)]
        else:
            carry, ys = _value_leaves(rw[0]), [w for x in rw[1:] for w in _value_leaves(x)]
            avail += [(ni, j, w) for j, w in enumerate(carry)]
            avail += [(ni, len(carry) + m * len(ys) + j, w) for m in range(n) for j, w in enumerate(ys)]
    pop = sum(r["n"] if r["form"] == "batch" else 1 for r in root)
    cutsat = sorted(rng.sample(range(1, pop), k=min(pop - 1, rng.randint(0, 2)))) if pop > 1 else []
    edges = [0, *cutsat, pop]
    return Spec(templates, root, root_params, registered, [[a, b] for a, b in zip(edges, edges[1:])])


def _ret_widths(T: dict) -> list[int]:
    ws = list(T["params"])
    prims = _prims()
    for pi, _a in T["steps"]:
        ws.append(-16 if pi == PACK else 1 if pi < 0 else prims[pi][2].w)
    return [ws[v] for v in T["ret"]]


def realise(spec: Spec) -> tuple[bytes, dict, list]:
    """The Program a spec names, its bytes, its template query and its templates."""
    from verity.circuits import partition_object as PO
    from verity.circuits.program import Program
    from verity.circuits.refs import array_of, tuple_of
    from verity.circuits.types import Array, Tuple
    prims = _prims()
    fns = []
    for ti, T in enumerate(spec.templates):
        rw = _ret_widths(T)
        ret = _ty(rw[0]) if len(rw) == 1 else Tuple(*[_ty(w) for w in rw])
        params = [(f"x{j}", _ty(w)) for j, w in enumerate(T["params"])]

        def inner(B, S, *xs, T=T):
            from verity.circuits import boolean as BL
            vals = list(xs)
            for pi, args in T["steps"]:
                if pi == PACK:
                    vals.append(array_of([vals[a] for a in args]))
                    continue
                if pi < 0:
                    vals.append(vals[args[0]][-1 - pi])
                    continue
                fn, operands = prims[pi][0], [vals[a] for a in args]
                vals.append(B.call(fn, BL.operands(*operands)) if fn in (BL.AND, BL.XOR) else B.call(fn, *operands))
            outs = [vals[v] for v in T["ret"]]
            return outs[0] if len(outs) == 1 else tuple_of(*outs)

        if T.get("nested"):
            callee = _composite(f"WallsN{ti}", params, ret, inner)
            fns.append(_composite(f"WallsT{ti}", params, ret, lambda B, S, *xs, cl=callee: B.call(cl, *xs)))
        else:
            fns.append(_composite(f"WallsT{ti}", params, ret, inner))
    rparams = [(f"r{j}", _ty(w)) for j, w in enumerate(spec.root_params)]
    leaves_of: dict[int, list] = {}

    def body(B, S, *ps):
        outs = []
        for ni, nd in enumerate(spec.root):
            fn = fns[nd["t"]]

            def src(s):
                if s[0] == "r":
                    return array_of(leaves_of[s[1]][s[2]:s[2] + s[3]])
                return ps[s[1]] if s[0] == "p" else leaves_of[s[1]][s[2]]

            if nd["form"] == "call":
                o = B.call(fn, *[src(a[0]) for a in nd["args"]])
            elif nd["form"] == "batch":
                o = B.batch(fn, *[array_of([src(x) for x in a]) for a in nd["args"]], axes=[0] * len(nd["args"]), n=nd["n"])
            else:
                rest = [array_of([src(x) for x in a]) for a in nd["args"][1:]]
                o = B.scan(fn, src(nd["args"][0][0]), xs=rest, n=nd["n"])
            leaves_of[ni] = _leaf_colls(o)
            outs.append(o)
        return outs[-1]

    last = spec.root[-1]
    lastT = spec.templates[last["t"]]
    rw = _ret_widths(lastT)
    one = _ty(rw[0]) if len(rw) == 1 else Tuple(*[_ty(w) for w in rw])
    if last["form"] == "batch":
        rt = Array(last["n"], one)
    elif last["form"] == "scan":
        y = _ty(rw[1]) if len(rw) == 2 else Tuple(*[_ty(w) for w in rw[1:]])
        rt = Tuple(_ty(rw[0]), Array(last["n"], y))
    else:
        rt = one
    root = _composite("WallsRoot", rparams, rt, body)
    program = Program(root, registered=[f"r{j}" for j in spec.registered])
    used = sorted({nd["t"] for nd in spec.root})
    return _bytes(program), PO.template_instances_query([fns[t] for t in used]), fns


def _leaf_colls(o: Any) -> list:
    """An output's value leaves, each a Coll of its own, in leaf order."""
    from verity.circuits.types import Value
    out = []

    def walk(x):
        if isinstance(x.type, Value):
            out.append(x)
            return
        n = len(x.type.fields) if hasattr(x.type, "fields") else x.type.n
        for i in range(n):
            walk(x[i])

    walk(o)
    return out


@dataclass
class Finding:
    klass: str
    step: str
    detail: str
    spec: dict
    block: int


def check_case(spec: Spec, exe: Callable[[], Path], seed: int, form: str = "unit", stats: dict | None = None,
               lean: Callable[[], Path] | None = None) -> list[Finding]:
    """Every block of the spec's partition through the door's static steps (one finding a block: ``through``, or the
    step's refusal class), then its lowered template's rows held to the honest and wrong assignments and, where a unit
    reads another unit of the block, to a wrong read (`wrong_read`); under ``whole``, the block's one instance at C's
    values on in(S) and out(S)."""
    from verity.circuits import restrict as R
    from verity.circuits.codec import descriptor_id
    from verity.circuits.program import input_id
    rng = random.Random(seed)
    try:
        c, query, _fns = realise(spec)
    except Exception as e:  # noqa: BLE001 - a spec the builders refuse is no door finding
        return [Finding(f"build: {normalise(_err(e))}", "build", _err(e)[:400], spec.doc(), -1)]
    try:
        door = Door(c, query, [tuple(b) for b in spec.blocks], exe, form, lean=lean)
    except Exception as e:  # noqa: BLE001 - the query or the partition object refusing C is the partition step's wall
        return [Finding(f"partition: {normalise(_err(e))}", "partition", _err(e)[:600], spec.doc(), -1)]
    trace = trace_of(c, rng)
    out = []
    for k, (lo, hi) in enumerate(spec.blocks):
        r = door.run(k, lo, hi)
        out.append(Finding(r.klass, "lean" if r.lean_door else r.step, (r.lean_door or r.text)[:600], spec.doc(), k) if r.klass
                   else Finding("through", "-", "", spec.doc(), k))
        if stats is not None and lean is not None:
            for key, ok in (("lean_admitted", r.lean_door == ADMITTED), ("python_admitted", r.python_door == ADMITTED),
                            ("differences", bool(r.difference))):
                stats[key] = stats.get(key, 0) + ok
        if r.difference:
            out.append(Finding(f"difference: {normalise(r.difference)}", "difference", json.dumps(
                {"c_sha256": hashlib.sha256(c).hexdigest(), "units": list(r.units), "form": form, "seed": seed,
                 "lean_door": r.lean_door[:600], "python_door": r.python_door[:600]}), spec.doc(), k))
        if r.c_s is None:
            continue
        D = R.program(r.c_s)
        called = sorted({descriptor_id(nd.fn) for nd in D.root.body.nodes if input_id(nd.fn.id) is None})
        if len(called) != 1:
            if r.step is None:
                out.append(Finding("rows: C_S calls several templates, unchecked", "rows", str(called), spec.doc(), k))
            continue
        key = called[0]
        fn = D.defs[key]
        cut = r.cut.get(key, {})
        ins_p, outs_p = ports(fn, cut)
        params, ret = fn.signature()
        in_w = [w for _n, t in params for w in t.leaf_widths()]
        out_w = list(ret.leaf_widths())
        n_in = sum(-(-sum(in_w[j] for j in lv) // WORD) for _p, lv in ins_p)
        n_out = sum(-(-sum(out_w[j] for j in lv) // WORD) for _p, lv in outs_p)
        units = [u for _k, tt, a, n in door.tp.ranges if tt == key for u in range(max(a, lo), min(a + n, hi))]
        try:
            t, _log = template_of(r.c_s, key, n_in, n_out, exe)
            if form == "whole":
                x, y, b = _assigned(fn, cut, [trace[g] for g in r.ports[0]], [trace[g] for g in r.ports[1]])
                xs, ys, vb = [x], [y], [b]
            else:
                xs, ys, vb = honest_rows(c, query, units, fn, cut, trace)
            v = row_check(t, xs, ys, vb, rng)
            if stats is not None:
                stats["rows"] = stats.get("rows", 0) + 1
                if r.step is None:
                    stats["through"] = stats.get("through", 0) + 1
        except Unchecked as e:
            if r.step is not None:
                continue
            if isinstance(e, Unlowered):
                out.append(Finding(f"rows: the door accepts what Lean's lowering refuses: {normalise(str(e))}", "rows",
                                   str(e)[:600], spec.doc(), k))
            else:
                out.append(Finding(f"unchecked: {normalise(str(e))}", "unchecked", str(e)[:600], spec.doc(), k))
            continue
        except Exception as e:  # noqa: BLE001
            out.append(Finding(f"rows: crashed: {normalise(_err(e))}", "rows", _err(e)[:600], spec.doc(), k))
            continue
        door_says = ("door accepts" if r.step is None else f"door refuses at {r.step}") + _odd(r.c_s, key)
        if not v.honest_ok:
            out.append(Finding(f"rows: honest output refused ({door_says})", "rows", v.detail, spec.doc(), k))
        for w in v.wrong_ok:
            out.append(Finding(f"rows: wrong output accepted: {w} ({door_says}{'' if v.exact else ', sampled'})", "rows",
                               f"free columns reaching an output: {v.free_reaching}; {v.detail}", spec.doc(), k))
        if v.honest_ok and form == "unit":
            read = wrong_read(c, query, door.tp, units, t, fn, cut, trace)
            if read:
                out.append(Finding(f"rows: a unit's read of another unit of its block is not tied: the reader given a "
                                   f"value its writer did not produce is accepted ({door_says})", "rows", read,
                                   spec.doc(), k))
    return out


def inner_reads(tp: Any, units: Sequence[int]) -> list[tuple[int, int, int]]:
    """Each (writer, reader, leaf) of the block: the reader's parameter leaf ``leaf`` is an output gate of the writer,
    both units of the block."""
    outs = {g: u for u in units for g in tp.outputs(u)}
    return [(outs[g], v, j) for v in units for j, g in enumerate(tp.inputs(v)) if outs.get(g, v) != v]


def wrong_read(c: bytes, query: dict, tp: Any, units: Sequence[int], t: Any, fn: Any, cut: dict,
               trace: dict[int, int]) -> str:
    """Where a unit of the block reads another's output: the reader's input row with that leaf's low bit flipped (a value
    the writer did not produce) and its output the rows' own evaluation there. When the rows accept that pair, its output
    differs from the honest one, and the writer's rows hold, this returns the pair, since nothing in the block circuit's
    rows ties the reader's input row to the writer's output row. Otherwise it returns ""."""
    from verity_flock import tmpl as T
    pairs = inner_reads(tp, units)
    if not pairs:
        return ""
    params, ret = fn.signature()
    in_w = [w for _n, ty in params for w in ty.leaf_widths()]
    ins_p, outs_p = ports(fn, cut)
    _ins, out_cols = _parts(t.groups)
    for u, v, j in pairs:
        a = [trace[g] for g in tp.inputs(v)]
        a[j] ^= 1
        row: list[int] = []
        for _p, leaves in ins_p:
            row += _padded(_bits([in_w[i] for i in leaves], [a[i] for i in leaves]))
        X = T.slice_lanes(np.asarray([row], np.uint8))
        Y = T.evaluate(t, X, np.asarray(out_cols, np.int64))
        y = [int(int(Y[b, 0]) & 1) for b in range(len(out_cols))]
        _x, (honest,), (vb,) = honest_rows(c, query, [v], fn, cut, trace)
        if y == honest:
            continue
        verdict = row_check(t, [row], [y], [vb], random.Random(0))
        if verdict.honest_ok:
            return (f"unit {v} reads unit {u}'s output at its parameter leaf {j}; given that leaf with its low bit "
                    f"flipped, the rows accept the reader's output {y[:32]} against its honest {honest[:32]}")
    return ""


def _odd(c_s: bytes, key: str) -> str:
    """``, odd field`` where the template has a field of bits not a multiple of 16 followed by another field of its side
    (`codec.odd_fields`, lean's second fault), else ``, no odd field``."""
    try:
        from verity.circuits.codec import odd_fields
        odd = any(follows for _side, _n, _bits, follows in odd_fields(json.loads(c_s), key))
    except Exception:  # noqa: BLE001 - a descriptor the scanner can't read is said so
        return ", odd fields unread"
    return ", odd field" if odd else ", no odd field"


def minimise(spec: Spec, klass: str, exe: Callable[[], Path], seed: int, budget: int = 40, form: str = "unit") -> Spec:
    """A smaller spec showing the same class: drop root nodes, then template steps, then merge blocks, while it does."""
    def shows(s: Spec) -> bool:
        try:
            return any(f.klass == klass for f in check_case(s, exe, seed, form))
        except Exception:  # noqa: BLE001
            return False

    cur = spec
    tries = 0
    changed = True
    while changed and tries < budget:
        changed = False
        for cand in _smaller(cur):
            tries += 1
            if tries > budget:
                break
            if shows(cand):
                cur, changed = cand, True
                break
    return cur


def _smaller(s: Spec):
    import copy
    if len(s.blocks) > 1:
        for i in range(len(s.blocks) - 1):
            b = copy.deepcopy(s.blocks)
            b[i:i + 2] = [[b[i][0], b[i + 1][1]]]
            yield Spec(s.templates, s.root, s.root_params, s.registered, b)
    for ti, T in enumerate(s.templates):
        for si in range(len(T["steps"]) - 1, -1, -1):
            used = {a for _p, args in T["steps"][si + 1:] for a in args} | set(T["ret"])
            if len(T["params"]) + si in used:
                continue
            t2 = copy.deepcopy(s.templates)
            del t2[ti]["steps"][si]
            t2[ti]["ret"] = [v if v < len(T["params"]) + si else v - 1 for v in T["ret"]]
            t2[ti]["steps"] = [[p, [a if a < len(T["params"]) + si else a - 1 for a in args]] for p, args in t2[ti]["steps"]]
            yield Spec(t2, s.root, s.root_params, s.registered, s.blocks)
        if T.get("nested"):
            t2 = copy.deepcopy(s.templates)
            t2[ti]["nested"] = False
            yield Spec(t2, s.root, s.root_params, s.registered, s.blocks)
    if s.registered:
        yield Spec(s.templates, s.root, s.root_params, [], s.blocks)
    if len(s.root) > 1:
        for ni in range(len(s.root) - 1, -1, -1):
            if any(a[0] in ("n", "r") and a[1] >= ni for nd in s.root[ni + 1:] for src in nd["args"] for a in src):
                continue
            root = copy.deepcopy(s.root[:ni] + s.root[ni + 1:])
            pop = sum(nd["n"] if nd["form"] == "batch" else 1 for nd in root)
            yield Spec(s.templates, root, s.root_params, s.registered, [[0, pop]])


def fuzz(seed: int, cases: int, minutes: float, exe: Callable[[], Path], log=None, form: str = "unit",
         lean: Callable[[], Path] | None = None) -> dict:
    """``cases`` drawn cases from ``seed`` for at most ``minutes``; ``checked``: the blocks whose rows were held, and of
    those the ones every door step took (``through``)."""
    _catalog()
    rng = random.Random(seed)
    t0 = time.perf_counter()
    found: dict[str, list] = defaultdict(list)
    tried = blocks = 0
    stats: dict = {}
    for i in range(cases):
        if (time.perf_counter() - t0) / 60 > minutes:
            break
        words = rng.random() < WORDS
        s = draw_inner(rng, words) if rng.random() < INNER else draw(rng, small=rng.random() < 0.5, words=words)
        cs = rng.getrandbits(32)
        fs = check_case(s, exe, cs, form, stats, lean)
        tried += 1
        blocks += len(s.blocks)
        for f in fs:
            found[f.klass].append((i, cs, f))
        if log:
            log(json.dumps({"case": i, "seed": cs, "findings": sorted({f.klass for f in fs})}))
    summary = {"seed": seed, "form": form, "decider": LEAN_DOOR if lean is not None else "python's steps", "cases": tried,
               "blocks": blocks, "seconds": round(time.perf_counter() - t0, 1), "checked": stats, "classes": []}
    for k, hits in sorted(found.items(), key=lambda kv: -len(kv[1])):
        i, cs, f = min(hits, key=lambda h: len(json.dumps(h[2].spec)))
        small = f.spec
        if k.startswith("rows:"):
            small = minimise(Spec(**f.spec), k, exe, cs, form=form).doc()
        summary["classes"].append({"class": k, "count": len(hits), "step": f.step, "example": {"case": i, "seed": cs,
                                   "block": f.block, "detail": f.detail[:1600 if f.step == "difference" else 400],
                                   "spec": small}})
    return summary


# --- named cases -----------------------------------------------------------------------------------------------------

def case(name: str, exe: Callable[[], Path]) -> dict:
    """A builder's circuit as one block of all its units: the door's static steps, then the rows of its one template."""
    from verity.circuits import partition_object as PO
    from verity.circuits import restrict as R
    from verity.circuits.codec import descriptor_id
    from verity.circuits.program import input_id
    c, q = load(name)
    pop = PO.TemplatePartition(R.program(c), q).population
    door = Door(c, q, [(0, pop)], exe)
    r = door.run(0, 0, pop)
    rec: dict = {"case": name, "step": r.step, "text": r.text[:600]}
    if r.c_s is None:
        return rec
    D = R.program(r.c_s)
    key, = sorted({descriptor_id(nd.fn) for nd in D.root.body.nodes if input_id(nd.fn.id) is None})
    fn = D.defs[key]
    cut = (door.cuts()[0] or {}).get(key, {})
    ins_p, outs_p = ports(fn, cut)
    params, ret = fn.signature()
    in_w = [w for _n, t in params for w in t.leaf_widths()]
    out_w = list(ret.leaf_widths())
    n_in = sum(-(-sum(in_w[j] for j in lv) // WORD) for _p, lv in ins_p)
    n_out = sum(-(-sum(out_w[j] for j in lv) // WORD) for _p, lv in outs_p)
    try:
        t, _log = template_of(r.c_s, key, n_in, n_out, exe)
    except Unlowered as e:
        return rec | {"template": key, "rows": f"unlowered: {str(e)[:400]}"}
    rng = random.Random(0)
    verdicts = []
    for s in range(4):
        trace = trace_of(c, random.Random(s))
        xs, ys, vb = honest_rows(c, q, list(range(pop)), fn, cut, trace)
        v = row_check(t, xs, ys, vb, rng)
        verdicts.append({"honest_ok": v.honest_ok, "wrong_ok": v.wrong_ok, "exact": v.exact, "detail": v.detail})
    rec |= {"template": key, "in_words": n_in, "out_words": n_out, "rows": verdicts}
    return rec


def internal_case(exe: Callable[[], Path], seed: int = 0) -> dict:
    """`internal_read`'s block [0, 2) under both forms: the door's steps, the lowered template's input bits, its rows on
    C's honest values, and on a forged read (the reader's x the writer's output with bit 0 flipped, the reader's output
    T of that): under ``unit``, whether the reader's copy's rows take it (``forged_accepted``); under ``whole``, whether
    the block's rows take the block output it gives (``forged_accepted``), with its inputs C's."""
    from verity.circuits import partition_object as PO
    from verity.circuits import restrict as R
    from verity.circuits.codec import descriptor_id
    from verity.circuits.program import input_id
    _catalog()
    c, q = internal_read()
    tp = PO.TemplatePartition(R.program(c), q)
    rng = random.Random(seed)
    trace = trace_of(c, rng)
    out: dict = {"c_sha256": hashlib.sha256(c).hexdigest()}
    for form in FORMS:
        r = Door(c, q, [(0, 2)], exe, form).run(0, 0, 2)
        rec: dict = {"step": r.step, "text": r.text[:400], "ported": r.ported, "row_bits": r.row_bits,
                     "gates": r.gates, "tmpl_gates": r.tmpl_gates}
        out[form] = rec
        if r.c_s is None:
            continue
        D = R.program(r.c_s)
        key, = sorted({descriptor_id(nd.fn) for nd in D.root.body.nodes if input_id(nd.fn.id) is None})
        fn, cut = D.defs[key], r.cut.get(key, {})
        rec |= {"template": key, "params": [[n, t.leaves] for n, t in fn.signature()[0]],
                "in_ports": [p for p, _lv in ports(fn, cut)[0]]}
        try:
            t, _log = template_of(r.c_s, key, *_words(fn, cut), exe)
        except Unlowered as e:
            rec["rows"] = f"unlowered: {str(e)[:300]}"
            continue
        if form == "unit":
            xs, ys, vb = honest_rows(c, q, [0, 1], fn, cut, trace)
            a = [trace[g] for g in tp.inputs(1)]
            a[0] ^= 1
            x, y, b = _assigned(fn, cut, a, [1 - v for v in a[:READ_BITS]])
            forged = row_check(t, [xs[0], x], [ys[0], y], [vb[0], b], rng)
            rec["in_bits"] = [len(v) for v in xs]
        else:
            ins, outs = r.ports
            x, y, b = _assigned(fn, cut, [trace[g] for g in ins], [trace[g] for g in outs])
            xs, ys, vb = [x], [y], [b]
            o = [trace[g] for g in outs]
            o[0] ^= 1
            fx, fy, fb = _assigned(fn, cut, [trace[g] for g in ins], o)
            forged = row_check(t, [fx], [fy], [fb], rng)
            rec["in_bits"] = [len(x)]
        honest = row_check(t, xs, ys, vb, rng)
        rec |= {"honest_ok": honest.honest_ok, "wrong_accepted": honest.wrong_ok, "exact": honest.exact,
                "forged_accepted": forged.honest_ok, "forged_exact": forged.exact}
    return out


#: the bits of the internal read: T's x
READ_BITS = 16


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("census")
    a.add_argument("--circuit", required=True)
    a.add_argument("--blocks", default="one-per-template")
    a.add_argument("--lower")
    a.add_argument("--json")
    a.add_argument("--no-rows", action="store_true")
    a.add_argument("--units", choices=UNITS, default="one")
    a.add_argument("--form", choices=FORMS, default="unit")
    a.add_argument("--only", default="", help="the chosen blocks whose first unit's template matches this regex")
    a.add_argument("--piece", help="the Spot difftest executable: under --form whole, Lean's restrict piece on each block")
    a.add_argument("--cap", type=int, default=CAP)
    a.add_argument("--one-circuit", default="", help="the one-circuit commit C's tree holds, where the clone has no such ref")
    b = sub.add_parser("fuzz")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--cases", type=int, default=100)
    b.add_argument("--minutes", type=float, default=30.0)
    b.add_argument("--lower")
    b.add_argument("--json")
    b.add_argument("--form", choices=FORMS, default="unit")
    for x in (a, b):
        x.add_argument("--lean-door", help="spot-one-stage-zk, whose restrict decides (default: built by the tree's `built`)")
        x.add_argument("--no-lean-door", action="store_true", help="Python's steps decide, as before Lean's door was asked")
    k = sub.add_parser("case")
    k.add_argument("name", choices=sorted(BUILDERS))
    k.add_argument("--lower")
    args = ap.parse_args(argv)
    exe = _exe(args.lower)
    say = lambda s: print(s, flush=True)  # noqa: E731
    lean = None if args.cmd == "case" or args.no_lean_door else _lean_exe(args.lean_door)
    if args.cmd == "case":
        _catalog()
        say("CASE " + json.dumps(internal_case(exe) if args.name == "internal_read" else case(args.name, exe), default=str))
        return 0
    if args.cmd == "fuzz":
        s = fuzz(args.seed, args.cases, args.minutes, exe, log=lambda x: say("FUZZ " + x), form=args.form, lean=lean)
        if args.json:
            Path(args.json).write_text(json.dumps(s, indent=1, default=str))
        say("SUMMARY " + json.dumps({k: s[k] for k in ("seed", "form", "decider", "cases", "blocks", "seconds", "checked")}))
        for c in s["classes"]:
            say("CLASS " + json.dumps(c, default=str))
        return 1 if any(c["class"].startswith("rows:") for c in s["classes"]) else 0
    t = time.perf_counter()
    c, q = load(args.circuit)
    say("BUILT " + json.dumps({"circuit": args.circuit, "bytes": len(c), "seconds": round(time.perf_counter() - t, 1)}))
    plan, by = plan_blocks(args.circuit, args.cap) if args.units == "plan" else (None, "")
    if by:
        say("PLAN " + json.dumps({"by": by, "blocks": len(plan)}))
    server = None
    if args.piece:
        from verity_difftest import Server
        server = Server([args.piece])
    results = census(c, q, args.blocks, exe, log=lambda x: say("BLOCK " + x), rows=not args.no_rows, units=args.units,
                     plan=plan, form=args.form, piece=server, only=args.only, lean=lean)
    c_sha = hashlib.sha256(c).hexdigest()
    for r in results:
        if r.difference:
            say("DIFFERENCE " + json.dumps({"difference": r.difference, "c_sha256": c_sha, "units": list(r.units),
                                            "form": args.form, "templates": list(r.templates),
                                            "lean_door": r.lean_door, "python_door": r.python_door}))
    rows = table(results, c_sha, built_at(args.one_circuit))
    sized = floor_table(results) if args.units != "one" else []
    if args.json:
        Path(args.json).write_text(json.dumps({"circuit": args.circuit, "blocks": args.blocks, "units": args.units,
                                               "form": args.form,
                                               "plan": by, "table": rows, "sizes": sized}, indent=1))
    print(render(rows))
    if sized:
        print(render_floor(sized))
    refused = sum(bool(r.klass) or not r.rows.startswith(("yes", "too large", "not reached", "unchecked")) for r in results)
    say("CENSUS " + json.dumps({"circuit": args.circuit, "units": args.units, "form": args.form, "blocks": len(results),
                                "decider": LEAN_DOOR if lean is not None else "python's steps", "refused": refused,
                                "python_refused": sum(r.python_door != ADMITTED for r in results),
                                "differences": sum(bool(r.difference) for r in results),
                                "classes": len([r for r in rows if r["class"] != "through"])}))
    return 1 if refused else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001 - exit 1 is a refused block; a crash of the tool is 2
        import traceback
        traceback.print_exc()
        sys.exit(2)
