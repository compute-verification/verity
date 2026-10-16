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
  ``dfill_check``, ``dfill_ref``): an arm with another kernel brings it behind that ABI.
- ``placement(phase, B, L)``: where the filler runs beside serving at a suite point, a `Placement`. ``phase`` is
  ``"decode"`` (the decode step at batch B, context L) or ``"prefill"`` (the 32 layers of linears at M = B rows, L = 0).
  It is a pure function of the point: the same placement on every call, chosen before the run (desideratum 5's fixed work
  per round), never from the run's own measurements.
- ``choice``: one line saying how the policy was chosen and what choosing costs at serving time (a secondary metric).

The scorer refuses an arm that breaks the contract (`check_arm`). Arms are named in `ARMS`, or loaded as
``module:attribute`` from a file under ``benchmarks/pouw/filler/`` (a climber's arm), constructed with no arguments.
"""

from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
VERSION = "filler-arm/v1"
PLACES = ("none", "split", "co")
PHASES = ("decode", "prefill")
SMS = 188  # the RTX PRO 6000's SMs


class ArmError(ValueError):
    pass


@dataclass(frozen=True)
class Filler:
    W: int = 8
    NC: int = 4
    S: int = 2048
    chunk_log2: int = 19


@dataclass(frozen=True)
class Placement:
    place: str = "none"
    fill_sms: int = 0

    def __post_init__(self):
        if self.place not in PLACES:
            raise ArmError(f"placement {self.place!r} is not one of {PLACES}")
        if self.place == "split" and not 0 < self.fill_sms < SMS:
            raise ArmError(f"a split's filler SMs are in (0, {SMS}), not {self.fill_sms}")
        if self.place != "split" and self.fill_sms:
            raise ArmError(f"only a split names filler SMs ({self.place} with {self.fill_sms})")


NONE = Placement()


def check_arm(arm) -> list[str]:
    """What is wrong with `arm` against the contract (empty when nothing is)."""
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
    if not isinstance(getattr(arm, "choice", None), str):
        bad.append("choice is one line")
    if not callable(getattr(arm, "placement", None)):
        bad.append("placement(phase, B, L) is callable")
    return bad


def placed(arm, phase: str, b: int, l: int) -> Placement:
    p = arm.placement(phase, b, l)
    if not isinstance(p, Placement):
        raise ArmError(f"{arm.name}.placement({phase}, {b}, {l}) returned {p!r}, not a Placement")
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
#: re-measured in the holdout r20261016-040257-cec0 (note:20261016T0318Z-draft-dram-free-filler §0)
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
    sources: tuple = ("benchmarks/pouw/decode_capacity/dcap.cu", "benchmarks/pouw/filler/arms.py")
    filler: Filler | None = Filler(W=8, NC=4, S=2048, chunk_log2=19)
    choice: str = "an offline table of 11 (B, L) splits from FILL's runs 2–5; a lookup and a green split at serving time"

    def placement(self, phase, b, l):
        if phase == "prefill":
            return NONE
        return Placement("split", point_lookup(FILL_V1_SPLIT, b, l))


ARMS = {"idle": Idle, "fill-v1": FillV1}


def load(spec: str):
    """An arm by name (`ARMS`) or as ``path/to/file.py:Attr`` under benchmarks/pouw/filler/."""
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
    bad = check_arm(arm)
    if bad:
        raise ArmError(f"arm {spec}: " + "; ".join(bad))
    return arm
