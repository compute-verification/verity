"""The filler arm contract (v1) and the baseline arms.

An arm is a filler implementation plus its placement policy. The scorer (`score.py`) owns everything else: the serving
workload, the timing and its serving-alone controls, the gates and the arithmetic. An arm is any object with:

- ``name``: the arm's id (``[a-z0-9-]+``), the record's ``arm``.
- ``lineage``: ``parent`` (the art: id or arm it changes, None for a root), ``approach`` (``pouw/filler-<slug>``, as
  ``research notes claim`` names it), ``mechanism`` (namespaced tags, such as ``place:split``), ``hypothesis`` (one line).
- ``sources``: the repository paths whose bytes are the variant (the filler's source and the arm's own module); their
  digest is the variant id, as `harness/ledger.variant_record` makes it.
- ``filler``: the filler's chunk shape, a `Filler` (warps a block W, chains a warp NC, steps a tile S, chunk_log2), or
  None for no filler. v1's driver runs one filler ABI, dcap.cu's ``dfill_k`` (``DFill``, ``dfill_launch``,
  ``dfill_check``, ``dfill_ref``): an arm with another kernel brings it behind that ABI. ``Filler.noise`` optionally
  declares its operands (`NOISE`); filler-scorer/v3 reads it, with the facts of its own build, to pick the filler's
  certified-fraction entry (`scoring.filler_entry`). ``Filler.fold`` optionally declares its receipt's fold (`FOLDS`),
  which filler-scorer/v5 reads (`scoring.fold_reading`). Neither is part of the variant's run-time config (`config`).
- ``placement(phase, B, L)``: where the filler runs beside serving at a suite point, a `Colocation`. ``phase`` is
  ``"decode"`` (the decode step at batch B, context L) or ``"prefill"`` (the 32 layers of linears at M = B rows, L = 0).
  It is a pure function of the point: the same placement on every call, chosen before the run (desideratum 5's fixed work
  per round), never from the run's own measurements.
- ``choice``: one line saying how the policy was chosen and what choosing costs at serving time (a secondary metric).

The scorer refuses an arm that breaks the contract (`check_arm`), and from filler-scorer/v3 one whose record the ledger
or the store would refuse (`record_problems`), when it loads: before the build and the GPU lease. Arms are named in
`ARMS`, or loaded as ``module:attribute`` from a file under ``benchmarks/pouw/filler/`` (a climber's arm), constructed
with no arguments.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
VERSION = "filler-arm/v1"
PLACES = ("none", "split", "co")
PHASES = ("decode", "prefill")
SMS = 188  # the RTX PRO 6000's SMs
#: a filler's declarable operands: its chain's own outputs, fresh uniform ±1, fresh uniform {0,1} bits on both sides,
#: fresh bits A ∈ {0,1} with B = U uniform on {±1..±64} (pfill-bU), or
#: the dither ⌊x/s⌋ + b (data quantized down plus a fresh uniform bit)
NOISE = ("chained", "pm1", "bits01", "bitsU", "dither")
#: a filler's declarable receipt folds (pfill.cuh's pf_absorb): ram (NC 4), duo (NC 2), and at NC 4 XOR-in (rotations 7
#: and 19, -DPFILL_XIN; superseded), XOR-in (13, 19) (-DPFILL_XIN13) and LOP3+IMAD (-DPFILL_LI); filler-scorer/v5
#: reads the declaration with its build's SASS (`scoring.fold_reading`), undeclared as ram
FOLDS = ("ram", "duo", "xor-in", "xor-in-13", "lop3-imad")
#: the `Filler` fields the driver runs and the variant's run-time config carries
SHAPE = ("W", "NC", "S", "chunk_log2")


class ArmError(ValueError):
    pass


@dataclass(frozen=True)
class Filler:
    W: int = 8
    NC: int = 4
    S: int = 2048
    chunk_log2: int = 19
    noise: str | None = None
    fold: str | None = None


def config(filler: Filler | None) -> dict[str, str]:
    """The variant's run-time config (`ledger.variant_record`): the filler's shape, as the driver runs it."""
    return {k: str(getattr(filler, k)) for k in SHAPE} if filler is not None else {}


@dataclass(frozen=True)
class Colocation:
    place: str = "none"
    fill_sms: int = 0

    def __post_init__(self):
        if self.place not in PLACES:
            raise ArmError(f"placement {self.place!r} is not one of {PLACES}")
        if self.place == "split" and not 0 < self.fill_sms < SMS:
            raise ArmError(f"a split's filler SMs are in (0, {SMS}), not {self.fill_sms}")
        if self.place != "split" and self.fill_sms:
            raise ArmError(f"only a split names filler SMs ({self.place} with {self.fill_sms})")


NONE = Colocation()


def _record_rules():
    """The attempt ledger (`harness/ledger.py`) and the evidence store's canonical form, ids and manifest model: the code
    `score.record` and the store run on the record after the run."""
    for p in (HERE.parent / "harness", REPO / "tools" / "research" / "src"):
        if str(p) not in sys.path:
            sys.path.append(str(p))
    import ledger  # noqa: PLC0415
    from research.store import canon, ids, model  # noqa: PLC0415
    return ledger, canon, ids, model


def record_problems(arm) -> list[str]:
    """What the ledger or the store would refuse in the arm's record (`score.record`), by the code they run after the
    run, so an arm that would lose its record fails when it loads, before a build or a GPU lease (filler-scorer/v3):
    the hypothesis and the choice are one line of at most `ledger.ONE_LINE` characters (`ledger.one_line`); a parent
    named as an ``art:`` id is a whole one, since it becomes the record's parent ref (`ids.is_art`); the mechanism is a
    list of tags; the record's arm fields are canonical JSON within the store's meta limit (`canon`,
    `model.META_MAX_BYTES`); and the sources make a variant (`ledger.source_sha256`: repository files, none twice, none
    named the run-time config's entry)."""
    L, canon, ids, model = _record_rules()
    bad = []
    lin = getattr(arm, "lineage", None)
    lin = lin if isinstance(lin, dict) else {}
    fields = {"parent": lin.get("parent"), "approach": lin.get("approach")}
    for key, what, v in (("hypothesis", "lineage.hypothesis", lin.get("hypothesis")),
                         ("choice", "choice", getattr(arm, "choice", None))):
        try:
            fields[key] = L.one_line(v, what)
        except L.LedgerError as ex:
            bad.append(f"{ex} (the ledger refuses the record after the run): shorten it")
    par = fields["parent"]
    if par is not None and not isinstance(par, str):
        bad.append(f"lineage.parent {par!r} is a string (an art: id, an arm, a branch@sha) or None")
    elif isinstance(par, str) and par.startswith("art:") and not ids.is_art(par):
        bad.append(f"lineage.parent {par!r} becomes the record's parent ref, which the store takes only as a whole art: "
                   "id (art: and 64 hex digits; `research data show` expands an abbreviation)")
    mech = lin.get("mechanism")
    if isinstance(mech, (list, tuple)) and all(isinstance(t, str) and t.strip() for t in mech):
        fields["mechanism"] = sorted(set(mech))
    else:
        bad.append(f"lineage.mechanism {mech!r} is a list of tags (strings)")
    try:
        n = len(canon.dumps(L._clean(fields)))
    except ValueError as ex:
        bad.append(f"the record's arm fields aren't canonical JSON: {ex}")
    else:
        if n > model.META_MAX_BYTES:
            bad.append(f"the record's arm fields are {n} canonical bytes, past the store's meta limit "
                       f"{model.META_MAX_BYTES}")
    srcs = getattr(arm, "sources", None) or ()
    try:
        L.source_sha256([REPO / s for s in srcs], REPO, config(getattr(arm, "filler", None)))
    except (L.LedgerError, ValueError, OSError, TypeError) as ex:
        bad.append(f"sources {srcs!r} don't make a variant: {ex}")
    return bad


def check_arm(arm, *, record: bool = False) -> list[str]:
    """What is wrong with `arm` against the contract (empty when nothing is); with `record`, also what the ledger or the
    store would refuse in its record (`record_problems`, filler-scorer/v3)."""
    bad = []
    name = getattr(arm, "name", None)
    if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name):
        bad.append(f"name {name!r} is not [a-z0-9-]+")
    lin = getattr(arm, "lineage", None)
    if not isinstance(lin, dict) or not lin.get("hypothesis") or not lin.get("mechanism"):
        bad.append("lineage names a hypothesis and mechanism tags")
    srcs = getattr(arm, "sources", None)
    if not srcs or not all((HERE.parents[2] / s).is_file() for s in srcs):
        bad.append(f"sources {srcs!r} are repository files")
    fl = getattr(arm, "filler", None)
    if fl is not None and not isinstance(fl, Filler):
        bad.append("filler is a Filler or None")
    elif fl is not None and fl.noise is not None and fl.noise not in NOISE:
        bad.append(f"filler.noise {fl.noise!r} is one of {NOISE} or None")
    if fl is not None and isinstance(fl, Filler) and fl.fold is not None and fl.fold not in FOLDS:
        bad.append(f"filler.fold {fl.fold!r} is one of {FOLDS} or None")
    if not isinstance(getattr(arm, "choice", None), str):
        bad.append("choice is one line")
    if not callable(getattr(arm, "placement", None)):
        bad.append("placement(phase, B, L) is callable")
    if record:
        bad += record_problems(arm)
    return bad


def placed(arm, phase: str, b: int, l: int) -> Colocation:
    p = arm.placement(phase, b, l)
    if not isinstance(p, Colocation):
        raise ArmError(f"{arm.name}.placement({phase}, {b}, {l}) returned {p!r}, not a Colocation")
    if p.place != "none" and arm.filler is None:
        raise ArmError(f"{arm.name} places a filler at {phase} {b}/{l} but has none")
    return p


@dataclass
class Idle:
    """No filler: serving alone, the scorer's floor and its A/A control."""

    name: str = "idle"
    lineage: dict = field(default_factory=lambda: {
        "parent": None, "approach": "pouw/filler-idle", "mechanism": ["place:none"],
        "hypothesis": "serving alone: coverage is serving's certified share, and the A/A slowdown is the timing noise"})
    sources: tuple = ("benchmarks/pouw/filler/arms.py",)
    filler: Filler | None = None
    choice: str = "nothing to choose"

    def placement(self, phase, b, l):
        return NONE


#: FILL v1's split per (B, L), cuBLASLt re-tuned on the decode partition, chosen on runs 2–5 so decode slows ≤ 5% and
#: re-measured in a holdout run
FILL_V1_SPLIT = {(1, 2048): 108, (32, 1024): 92, (32, 5120): 108, (32, 8192): 124, (32, 24576): 124, (48, 16384): 124,
                 (128, 2048): 92, (132, 5120): 60, (144, 5120): 60, (256, 1024): 60, (256, 2048): 60}


def point_lookup(table: dict, b: int, l: int):
    """FILL's `PointFiller` rule (share.py): the measured point with the smallest B ≥ b and, among those, the longest
    L ≤ l (the shortest when none is); past the largest B, the largest B's."""
    bs = sorted({pb for pb, _ in table})
    pb = next((x for x in bs if x >= b), bs[-1])
    ls = sorted(pl for xb, pl in table if xb == pb)
    pl = max((x for x in ls if x <= l), default=ls[0])
    return table[(pb, pl)]


@dataclass
class FillV1:
    """FILL v1: dcap.cu's register-resident chained int8 mma.sync filler (W 8, NC 4, S 2048), on a green-context split
    sized per (B, L) from FILL's table; no filler during prefill (compute-bound)."""

    name: str = "fill-v1"
    lineage: dict = field(default_factory=lambda: {
        "parent": "cursor/dram-free-filler-ea24@e015e0515", "approach": "pouw/filler-dram-free",
        "mechanism": ["filler:chained-imma-registers", "place:split", "split:per-point-table", "prefill:none"],
        "hypothesis": "FILL v1 as measured: 0.35–0.62 certified ÷ capacity at ≤ 5% decode slowdown"})
    sources: tuple = ("benchmarks/pouw/filler/dcap.cu", "benchmarks/pouw/filler/arms.py")
    filler: Filler | None = Filler(W=8, NC=4, S=2048, chunk_log2=19, noise="chained")
    choice: str = "an offline table of 11 (B, L) splits from FILL's runs 2–5; a lookup and a green split at serving time"
    #: dcap.cu builds pfill.cuh's filler unless -DDFILL_V1; score.py's nvcc reads NVCC_APPEND_FLAGS after the arm loads
    nvcc_flags: str = "-DDFILL_V1"

    def __post_init__(self):
        os.environ["NVCC_APPEND_FLAGS"] = self.nvcc_flags

    def placement(self, phase, b, l):
        if phase == "prefill":
            return NONE
        return Colocation("split", point_lookup(FILL_V1_SPLIT, b, l))


ARMS = {"idle": Idle, "fill-v1": FillV1}


def load(spec: str, *, record: bool = False):
    """An arm by name (`ARMS`) or as ``path/to/file.py:Attr`` under benchmarks/pouw/filler/, refused unless it meets
    `check_arm` (with `record`, its record's limits too)."""
    if spec in ARMS:
        arm = ARMS[spec]()
    elif ":" in spec:
        path, attr = spec.rsplit(":", 1)
        p = (HERE.parents[2] / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
        if HERE not in p.parents and p.parent != HERE:
            raise ArmError(f"an arm's module lives under {HERE}, not {p}")
        sp = importlib.util.spec_from_file_location(f"filler_arm_{p.stem}", p)
        mod = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(mod)
        arm = getattr(mod, attr)()
    else:
        raise ArmError(f"unknown arm {spec!r}: one of {sorted(ARMS)} or path/to/arm.py:Class")
    bad = check_arm(arm, record=record)
    if bad:
        raise ArmError(f"arm {spec}: " + "; ".join(bad))
    return arm
