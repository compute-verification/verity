"""The arm interface (v0): what a PoUW prover arm gives the harness, and what the harness gives it. This module is its
contract, and what the harness does with each member, in code, and `check_arm` is what the harness enforces before it runs
an arm.

An arm is any object with the members of `BenchArm` (v0.6; `Arm`, its old name, is a deprecated alias: the vLLM API's
`Arm` is a bench mode at the serving level, not a prover under test); the harness loads it by ``module:Class`` (``--arm``),
constructs it with no arguments, and calls `build` once. Device pointers and CUDA streams are Python ints; everything runs in
the device's primary context, so an arm may use the CUDA runtime, the driver API, or its own shared library.

A call's ``phases`` (v0.6) are named by the step groups the executor schedules, `GROUPS`; the chain times them in three
parts, `PARTS`, the last of which is `MOVABLE`. Its side stream takes `DEFERRED` alone, the served ``deferred`` schedule's
side lane (tile hashing), and the screen stays on the main lane. v0.4's phase names ``before``, ``gemm`` and ``after`` are
still accepted, each as its part, and deprecated; their one ``after`` can't be split, so they get no side stream.

An arm may have ``prefetch(dev, call, shape, flags)`` (optional, v0.6): before a flag set's first gate at a shape, the
harness syncs the device and calls it on each call it is about to gate (`refs.prefetch`), so an arm whose gate recomputes
outputs on the host can start those references, in `refs.pool()` (a process pool of the job's CPUs), while the first gate
runs. It only starts work: the gate still takes what it compares against from the device as it is at the gate, and ignores
a reference made from other inputs. An arm without it is gated as before.

`build` returns the arm's build record (JSON), which names its ``binaries`` (v0.3): the path of every cubin, fatbin or
library its kernels are loaded from, ``[]`` if they all come from libraries mapped into the process (those the SASS gate
finds itself; a cubin loaded with ``cuModuleLoadData`` is never mapped, so it must be listed). `sass_gate.py` passes each
before anything is timed: no .FTZ and no FP32 MUFU outside ptxas's pinned IEEE sequences, so no -ftz=true, --use_fast_math or
approximate div/sqrt build, and only from a pinned toolkit.

The record may also name the arm's reference ``verifier`` (v0.4), which `verify.py` runs after timing, in the same run, on
the transcripts the timed code emitted (panel rule (b), 08:35Z: a measured panel row needs the verifier's acceptance of its own
run's transcript). ``ctx["out_dir"]`` is the run's directory, and ``ctx["transcripts_dir"]`` its ``transcripts/``, where an
arm writes each shape's transcript (``transcripts/<arm>/<shape>/`` by convention). The declaration (`check_verifier`):

- ``transcript``: the file the verifier checks, relative to the run's directory, with ``{arm}`` and ``{shape}``; its sha256
  is what the panel row cites.
- ``commands``: argv lists, run in order from the run's working directory. Placeholders: ``{python}`` (verify.py's
  interpreter), ``{run_dir}``, ``{run_id}``, ``{commit}``, ``{arm}``; with ``per: "shape"`` (the default) they run once per
  timed shape, with ``{shape}``, ``{transcript}`` (its absolute path) and ``{dir}`` (the transcript's directory) too; with
  ``per: "run"`` once for the run (identical commands of several arms run once).
- ``commit`` (optional): the verifier's git commit, when it isn't the run's source commit.
- ``control`` (optional, v0.5): the negative control's transcript, like ``transcript`` (``{arm}`` and ``{shape}``): what the
  arm's no-write control dumped into the same poisoned buffers. The verifier must reject it (panel rule, 11:50Z).

An arm with ``dump(dev, call, shape, dir)`` (v0.5) lets the harness make its transcripts (`bench.poison_dump`, after the
timed items): the harness fills every buffer of ``call.transcript_buffers`` (default ``call.outputs``; name what a setup
step wrote too, such as a root or a seed, and never an operand, which it refuses) with 0xA5, runs the weight side and the
call, and has `dump` write the transcript under ``transcripts/<arm>/<shape>/``; then it poisons them again, launches
nothing, and has `dump` write the no-write control under ``negative/<arm>/<shape>/``. `dump` reads the device and computes
nothing. With a transcript under
``transcripts/{arm}/{shape}/``, `verify.py` judges the control at the same path under ``negative/`` unless the
declaration names another. An arm without `dump` writes its own transcripts, and nothing says the harness poisoned them.

A transcript is accepted only by a line of the commands' output that starts with ``ACCEPT`` and names
``sha256:<the transcript file's sha256>``, with no ``REJECT`` line naming it (and, per shape, every command exiting 0). A
control is rejected only by a line that starts with ``REJECT`` and names its sha256, with no ``ACCEPT`` line naming it.

The record may also name the kernel it runs and what the attempt is (v0.5), which `verify.py` writes into the attempt's
evidence-store record (`ledger.py`, kind ``kernel-attempt/v1``):

- ``variant``: the kernel as `verity.core.protocols.pouw.serving.KernelVariant` names it, from `ledger.variant_record` (family, device,
  schemes, the digest of the sources it was built from, the binary's digest, and ``prices``: the priced path it runs,
  such as ``cast-8.72`` or ``lut256``, `price_twins`' names).
- ``lineage``: ``parent`` (the art: id of the attempt it changes, or that attempt's variant id), ``approach``
  (``<campaign>/<slug>``), ``mechanism`` (namespaced tags such as ``pipeline:tma-ws``), ``hypothesis`` (one line), ``panel``
  (``{"line", "version", "attempt"?, "change"?}``), ``scheme``, ``schedule``, and ``phase_groups`` (each chain phase,
  before, gemm and after, as `verity.core.protocols.pouw.serving.GROUPS`).

γ is never declared here: the ledger looks it up from the Lean-pinned price twins of the panel version's statement on the
variant's priced path, at both FP32 prices. A lineage with ``gamma`` gets no record.
"""

from __future__ import annotations

import importlib
import re
import string
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

VERSION = "v0.6"
#: a call's step groups, in the order its main lane runs them; must equal verity.core.protocols.pouw.serving.GROUPS
GROUPS = ("row_seeds", "a_tree", "forming", "gemm", "cleanup", "tile_hash", "screen")
#: the groups nothing later in the call reads, so they may leave the main lane; must equal verity.core.protocols.pouw.serving.MOVABLE
MOVABLE = frozenset({"tile_hash", "screen"})
#: the chain's parts (`chain.py`), each the groups it runs, in order
PARTS = {"before": ("row_seeds", "a_tree", "forming"), "gemm": ("gemm", "cleanup"), "after": ("tile_hash", "screen")}
#: the groups the chain's side stream takes: the served deferred schedule's side lane, so must equal
#: verity.core.protocols.pouw.serving.DEFERRED.side; moving more of `MOVABLE` would time a schedule nobody serves
DEFERRED = frozenset({"tile_hash"})
VERIFIER_FIELDS = {
    "run": ("python", "run_dir", "run_id", "commit", "arm"),
    "shape": ("python", "run_dir", "run_id", "commit", "arm", "shape", "transcript", "dir"),
}

STREAMS = ("fp8", "fp4")
INPUTS = ("f32", "bf16", "e4m3", "e2m1-nv", "e2m1-mx", "int8")
REQUIRED_ROLES = ("hash", "nohash")


@dataclass(frozen=True)
class Shape:
    """One matmul, D (m x n) = A (m x k) . Bᵀ with B (n x k): A is the activations, B the weight."""

    label: str
    m: int
    k: int
    n: int

    @property
    def mkn(self) -> int:
        return self.m * self.k * self.n


@dataclass(frozen=True)
class Buffer:
    ptr: int
    nbytes: int


@dataclass
class OperandSet:
    """One of a shape's operand sets: its own weight (a distinct buffer), activations and scales, generated on the device from
    `seed`. `host(name)` copies a buffer to the host (bytes) on demand, for an arm's reference."""

    index: int
    weight_id: int
    seed: int
    a: Buffer
    b: Buffer
    a_scale: Buffer | None = None
    b_scale: Buffer | None = None
    host: Callable[[str], bytes] | None = None


@dataclass(frozen=True)
class Work:
    """Per call: tensor-core MACs by precision key (the baselines' keys, e.g. ``{"fp8-e4m3": m*k*n}``), W_ref and the
    credited work as exact strings in `unit` (the scheme's `wref` and `credit_of`), and the bytes absorbed by each hash.
    Per (weight, epoch): the weight side's hashed bytes. `gamma` is the arm's statement (Derived, from its
    certificate)."""

    tensor_macs: dict[str, int]
    w_ref: str
    credited: str
    unit: str
    hashed_bytes: dict[str, int]
    weight_hashed_bytes: dict[str, int] = field(default_factory=dict)
    gamma: dict[str, str] | None = None


@dataclass
class Gate:
    """A gate's verdict at one (shape, flags): `checked` maps each output it compared to ``"all"`` or ``"sampled:<what>"``."""

    ok: bool
    checked: dict[str, str]
    detail: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Call(Protocol):
    """One operand set's prepared prover call. `launch` is the timed call: one complete per-call step, enqueued on `stream`,
    with no host sync, allocation or host-device copy (the harness captures it into a CUDA graph). `weight_side` enqueues the
    per-(weight, epoch) work, under the same rules. `outputs` are the buffers the arm's gate reads.

    The dependent chain (`chain.py`, at decode shapes): before each launch the harness overwrites the operand set's `a` (and
    `a_scale`, for block-scaled inputs) with f of the previous call's y, so `launch` must read A from the operand set every
    time (a copy made in `prepare` would time a chain that isn't one). y is `outputs["y"]` (else `outputs["d"]`): m x n
    row-major, its first m rows, BF16 unless the call sets ``y_format`` ("f32" or "int32"). Optional: ``phases``, a dict of
    step groups (`GROUPS`, ``gemm`` among them), each ``fn(stream)``, which in `GROUPS` order enqueue the same work as
    `launch`; with it the chain splits the call's decode time by `PARTS`, and runs the `DEFERRED` groups on a side stream
    (they may then overlap the next call; the harness keeps call j from reusing its operand set, or the call object, until
    call j - S's are done). Deprecated: v0.4's ``before``, ``gemm`` and ``after``, each run as its part, and no side
    stream."""

    outputs: dict[str, Buffer]

    def launch(self, stream: int) -> None: ...

    def weight_side(self, stream: int) -> None: ...

    def free(self) -> None: ...


@runtime_checkable
class BenchArm(Protocol):
    """A prover under test: its build, gates, timed call, work and verifier. Optional, and so not members here: ``dump``
    (v0.5) and ``prefetch`` (v0.6), the module's docstring."""

    name: str
    stream: str
    precision: str
    inputs: str
    flag_sets: dict[str, frozenset[str]]

    def build(self, ctx: dict) -> dict: ...

    def domain(self, shape: Shape) -> str | None: ...

    def work(self, shape: Shape) -> Work: ...

    def prepare(self, dev: Any, shape: Shape, flags: frozenset[str], operands: OperandSet) -> Call: ...

    def gate(self, dev: Any, call: Call, shape: Shape, flags: frozenset[str]) -> Gate: ...


#: deprecated: BenchArm's v0.4 name
Arm = BenchArm


class ArmError(ValueError):
    pass


def phase_parts(phases: dict | None) -> tuple[dict[str, list], bool] | None:
    """A call's `phases` as the chain's parts, each the list of fns it runs in order, and whether they are named by `GROUPS`
    (else by v0.4's part names); None if the call has none (or, v0.4, lacks a part)."""
    if not phases:
        return None
    keys = set(phases)
    if keys & {"before", "after"}:
        mixed = sorted(keys & set(GROUPS) - {"gemm"})
        if mixed:
            raise ArmError(f"phases mix v0.4's names (before, gemm, after) with step groups {mixed}")
        return ({p: [phases[p]] for p in PARTS}, False) if set(PARTS) <= keys else None
    unknown = sorted(keys - set(GROUPS))
    if unknown or "gemm" not in keys:
        raise ArmError(f"phases {sorted(keys)}: step groups are {', '.join(GROUPS)}, gemm among them"
                       + (f" (unknown {unknown})" if unknown else ""))
    return {p: [phases[g] for g in gs if g in keys] for p, gs in PARTS.items()}, True


def side_split(phases: dict | None) -> tuple[list, list] | None:
    """The side chain's after part: the fns the main lane runs after the gemm part and those the side stream runs
    (`DEFERRED`), each in `GROUPS` order; None unless `phases` are step groups with a deferred one among them."""
    got = phase_parts(phases)
    if got is None or not got[1] or not DEFERRED & set(phases):
        return None
    after = [g for g in PARTS["after"] if g in phases]
    return [phases[g] for g in after if g not in DEFERRED], [phases[g] for g in after if g in DEFERRED]


def check_arm(arm: Any, precisions: set[str]) -> None:
    """Refuses an arm that doesn't meet the interface: its members, stream, inputs, the required flag roles, and a headline
    precision the harness has a baseline for."""
    missing = [m for m in ("name", "stream", "precision", "inputs", "flag_sets", "build", "domain", "work", "prepare", "gate")
               if not hasattr(arm, m)]
    if missing:
        raise ArmError(f"arm lacks {missing}")
    if getattr(arm, "prefetch", None) is not None and not callable(arm.prefetch):
        raise ArmError(f"arm {arm.name}: prefetch must be callable (dev, call, shape, flags), not {arm.prefetch!r}")
    if arm.stream not in STREAMS:
        raise ArmError(f"arm {arm.name}: stream {arm.stream!r} not in {STREAMS}")
    if arm.inputs not in INPUTS:
        raise ArmError(f"arm {arm.name}: inputs {arm.inputs!r} not in {INPUTS}")
    if arm.precision not in precisions:
        raise ArmError(f"arm {arm.name}: no baseline family {arm.precision!r} (have {sorted(precisions)})")
    roles = set(arm.flag_sets)
    if not set(REQUIRED_ROLES) <= roles:
        raise ArmError(f"arm {arm.name}: flag_sets needs roles {REQUIRED_ROLES}, has {sorted(roles)}")


def check_verifier(name: str, build: dict) -> dict | None:
    """The build record's ``verifier``, checked before anything is timed (a bad declaration would otherwise surface after the
    lease); None if the arm names none, and then none of its rows can be measured."""
    v = build.get("verifier")
    if v is None:
        return None
    if not isinstance(v, dict):
        raise ArmError(f"arm {name}: verifier must be a dict, not {type(v).__name__}")
    per = v.get("per", "shape")
    if per not in VERIFIER_FIELDS:
        raise ArmError(f"arm {name}: verifier per {per!r} not in {sorted(VERIFIER_FIELDS)}")
    t = v.get("transcript")
    _check_transcript(name, "transcript", t)
    control = v.get("control")
    if control is not None:
        _check_transcript(name, "control", control)
        if control == t:
            raise ArmError(f"arm {name}: verifier control must be another file than its transcript")
    cmds = v.get("commands")
    if not cmds or not all(isinstance(c, list) and c and all(isinstance(x, str) for x in c) for c in cmds):
        raise ArmError(f"arm {name}: verifier commands must be a non-empty list of non-empty argv lists of strings")
    bad = sorted({f for c in cmds for x in c for f in fields(x)} - set(VERIFIER_FIELDS[per]))
    if bad:
        raise ArmError(f"arm {name}: verifier commands use {bad}, not placeholders of per {per!r} "
                       f"({', '.join(VERIFIER_FIELDS[per])})")
    commit = v.get("commit")
    if commit is not None and not (isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{7,40}", commit)):
        raise ArmError(f"arm {name}: verifier commit must be a git commit (7-40 hex characters), not {commit!r}")
    return {"per": per, "transcript": t, "commands": [list(c) for c in cmds], "commit": commit,
            "name": v.get("name"), "control": control}


def _check_transcript(name: str, what: str, t: Any) -> None:
    if not isinstance(t, str) or not t or t.startswith("/") or ".." in t.split("/"):
        raise ArmError(f"arm {name}: verifier {what} must be a path inside the run's directory, not {t!r}")
    if not {"arm", "shape"} <= fields(t) <= {"arm", "shape"}:
        raise ArmError(f"arm {name}: verifier {what} {t!r} must name {{arm}} and {{shape}}, and nothing else")


def fields(template: str) -> set[str]:
    try:
        return {f for _, f, _, _ in string.Formatter().parse(template) if f is not None}
    except ValueError as ex:
        raise ArmError(f"bad placeholder in {template!r}: {ex}") from None


def check_gate(gate: Gate, call: Call) -> None:
    """A gate must name only outputs the call has, and check at least one of them in full."""
    unknown = set(gate.checked) - set(call.outputs)
    if unknown:
        raise ArmError(f"gate checked unknown outputs {sorted(unknown)}")
    if "all" not in gate.checked.values():
        raise ArmError("gate checks no output in full (one must be 'all': a root, or y's digest)")


def load(spec: str) -> BenchArm:
    """``module:Class`` → an instance."""
    mod, _, cls = spec.partition(":")
    if not cls:
        raise ArmError(f"--arm wants module:Class, not {spec!r}")
    return getattr(importlib.import_module(mod), cls)()
