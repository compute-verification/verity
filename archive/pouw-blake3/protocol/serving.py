"""What a served PoUW executor and its verifier agree on, whatever runs the model: the hashing formats, the Pearl-C family's
scheme names, the schedules a pass may run under, and what one pass publishes.

* `HashFormat` (`HASH_FORMATS`): a format's tree hash, its keyed-BLAKE3 labels, and whether forming waits for A's tree.  An
  executor reads keys and tree hash from here, never from literals, so the format is a switch (`SchemeId.with_hashing`).
* `SchemeId`: the one grammar of Pearl-C names, ``pearl-c-{device}-{forming}[-{hashing}][@{prices}]`` for FP8 and
  ``pearl-c-{nvfp4|mxfp4}-{version}[-{hashing}]`` for Pearl-C4.  Every name in use parses and prints back unchanged.
* `Schedule` (`SCHEDULES`): which step groups of a call leave the critical path, and at what priority; `schedule_refusals`
  says which placements a format forbids (A's tree stays on the main lane unless the format seeds forming per row).
* `PassCommitment`: what one pass on one rank publishes before any draw; `MANIFEST_SCHEMA` names the retained pass the
  verifier reads.
* `KernelVariant`, `StepIO`, `ShapeRule` and `GateRecord`: a kernel as content-addressed data, what each of its call steps
  reads and writes, the shapes it takes, and the device gate that admits it to serving (`variant_refusals`).  Loading a
  kernel directory is the runtime's; naming, gating and selecting one is here, so a bench or a ledger names a variant
  without importing a runtime.

Stdlib and `verity` only (``tests/test_boundaries.py``).
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from verity.ml.crypto.blake3 import blake3


@dataclass(frozen=True)
class HashFormat:
    """``seed_a`` is ``unit-root`` when forming reads seed_A from A's root (so A's tree is on the critical path), ``row``
    when each row seeds its own E_A line (h3, h3s).  ``row_digest`` is what a row's seed is keyed on: None for the row's
    leaf in A's tree (h3), or a row digest's schema (h3s: ``sha512/row-seg/v1``, `rowleaf.sha512_row_seg_digest`)."""

    name: str
    tree_hash: str
    message_label: str | None
    leaf_label: str | None
    seed_a: str
    row_cap: bool = False
    row_digest: str | None = None

    def message_key(self) -> bytes | None:
        return None if self.message_label is None else blake3(self.message_label.encode())

    def leaf_key(self) -> bytes | None:
        return None if self.leaf_label is None else blake3(self.leaf_label.encode())


HASH_FORMATS: dict[str, HashFormat] = {
    "h0": HashFormat("h0", "sha256", None, None, "unit-root"),
    "h1": HashFormat("h1", "blake3", "pearl-c/h1/message", "pearl-c/h1/leaf", "unit-root"),
    "h2": HashFormat("h2", "blake3-s256", "pearl-c/h1/message", "pearl-c/h1/leaf", "unit-root"),
    "h3": HashFormat("h3", "blake3-s256", "pearl-c/h1/message", "pearl-c/h1/leaf", "row", row_cap=True),
    "h3s": HashFormat("h3s", "hm96-sha512", None, None, "row", row_cap=True, row_digest="sha512/row-seg/v1"),
}

#: `pearl_c_device.DEVICES`' records: the panel's pearl-c-sm120 v1 is ``sm120`` (G = 4), its v2 ``sm120-unpromoted``
FP8_DEVICES = ("h100", "sm120", "sm120-unpromoted")
FP8_FORMINGS = ("v0", "v1")
FP4_PRECISIONS = ("nvfp4", "mxfp4")
_FP8 = re.compile(r"pearl-c-(?P<device>[a-z0-9]+(?:-(?!v[0-9])[a-z0-9]+)*)-(?P<forming>v[0-9]+)(?:-(?P<hashing>h[0-9]+s?))?"
                  r"(?:@(?P<prices>[a-z0-9_]+))?")
_FP4 = re.compile(r"pearl-c-(?P<precision>nvfp4|mxfp4)-(?P<forming>v[0-9]+)(?:-(?P<hashing>h[0-9]+))?")


@dataclass(frozen=True)
class SchemeId:
    """A Pearl-C scheme by its parts.  ``hashing`` h0 is left out of the name, and ``prices`` names another device's W1
    prices (``@sm_120``), as `PearlC` names its instances."""

    family: str
    device: str
    forming: str
    precision: str
    hashing: str = "h0"
    prices: str | None = None

    def __post_init__(self) -> None:
        bad = []
        if self.family == "pearl-c":
            if self.device not in FP8_DEVICES:
                bad.append(f"device {self.device!r} (one of {', '.join(FP8_DEVICES)})")
            if self.forming not in FP8_FORMINGS:
                bad.append(f"forming {self.forming!r} (one of {', '.join(FP8_FORMINGS)})")
            if self.precision != "e4m3":
                bad.append(f"precision {self.precision!r} (Pearl-C's is e4m3)")
        elif self.family == "pearl-c4":
            if self.precision not in FP4_PRECISIONS:
                bad.append(f"precision {self.precision!r} (one of {', '.join(FP4_PRECISIONS)})")
            if self.device != "sm120" or self.prices is not None:
                bad.append("Pearl-C4 runs on sm120 at its own prices")
        else:
            bad.append(f"family {self.family!r} (pearl-c or pearl-c4)")
        if self.hashing not in HASH_FORMATS:
            bad.append(f"hashing {self.hashing!r} (one of {', '.join(HASH_FORMATS)})")
        if bad:
            raise ValueError("not a Pearl-C scheme: " + "; ".join(bad))

    @classmethod
    def parse(cls, name: str) -> SchemeId:
        """`name`'s parts; ValueError when it follows neither grammar."""
        if m := _FP4.fullmatch(name):
            return cls("pearl-c4", "sm120", m["forming"], m["precision"], m["hashing"] or "h0")
        if m := _FP8.fullmatch(name):
            return cls("pearl-c", m["device"], m["forming"], "e4m3", m["hashing"] or "h0", m["prices"])
        raise ValueError(f"{name!r} is not a Pearl-C scheme name: pearl-c-{{device}}-{{forming}}[-{{hashing}}][@{{prices}}] "
                         "or pearl-c-{nvfp4|mxfp4}-{version}[-{hashing}]")

    @property
    def name(self) -> str:
        suffix = "" if self.hashing == "h0" else f"-{self.hashing}"
        if self.family == "pearl-c4":
            return f"pearl-c-{self.precision}-{self.forming}{suffix}"
        return f"pearl-c-{self.device}-{self.forming}{suffix}" + (f"@{self.prices}" if self.prices else "")

    def with_hashing(self, hashing: str) -> SchemeId:
        return replace(self, hashing=hashing)


def hash_format(scheme: str | SchemeId | Any) -> HashFormat:
    """The hashing format of a scheme: by name, by `SchemeId`, or from a scheme object's ``hashing``."""
    if isinstance(scheme, str):
        scheme = SchemeId.parse(scheme)
    hashing = scheme.hashing if isinstance(scheme, SchemeId) else getattr(scheme, "hashing", None)
    if hashing not in HASH_FORMATS:
        raise ValueError(f"{scheme!r} has no hashing format (one of {', '.join(HASH_FORMATS)})")
    return HASH_FORMATS[hashing]


#: a call's step groups, in the order the main lane runs them
GROUPS = ("a_tree", "forming", "gemm", "cleanup", "tile_hash", "screen")
#: groups that nothing later in the call reads, so they may always leave the main lane
MOVABLE = frozenset({"tile_hash", "screen"})
MAIN_PRIORITIES = ("caller", "greatest")
SIDE_PRIORITIES = ("least", "caller")


@dataclass(frozen=True)
class Schedule:
    """``main_priority`` ``caller`` leaves the calling runtime's stream as it is; ``greatest`` runs the main lane on a stream
    of the executor's own at the greatest priority, joined to the caller's by events.  ``ring`` is the rotating sets of
    per-call buffers each shape keeps, so a call reuses a set only after the side lane is done with it."""

    name: str
    side: frozenset[str] = field(default_factory=frozenset)
    ring: int = 1
    main_priority: str = "caller"
    side_priority: str = "least"


SERIAL = Schedule("serial")
#: the measured deferred schedule (panel attempt 57, `prefill_fused.py`'s and `decode_chain.py`'s chains): the main lane at the
#: greatest priority, each call's tile hashing beside it at the least.  The screen stayed on the main lane there, as in #564
DEFERRED = Schedule("deferred", frozenset({"tile_hash"}), ring=2, main_priority="greatest")
#: the same placement with the caller's stream left at its own priority (#564's first cut), kept to time against DEFERRED
DEFERRED_CALLER = Schedule("deferred-caller", frozenset({"tile_hash"}), ring=2)
SCHEDULES = {s.name: s for s in (SERIAL, DEFERRED, DEFERRED_CALLER)}


def schedule_refusals(fmt: HashFormat, s: Schedule) -> list[str]:
    """Why `s` can't run under `fmt` (empty: it can)."""
    out = []
    unknown = sorted(set(s.side) - set(GROUPS))
    if unknown:
        out.append(f"{s.name}: unknown step groups {unknown} (known: {', '.join(GROUPS)})")
    for g in sorted(set(s.side) & set(GROUPS) - MOVABLE):
        if g == "a_tree" and fmt.seed_a == "row":
            continue
        why = ("forming reads seed_A from A's root" if g == "a_tree"
               else "a later step of the call reads what it writes")
        out.append(f"{s.name}: {g} can't leave the main lane under {fmt.name}: {why}")
    if s.ring < 1:
        out.append(f"{s.name}: ring {s.ring}, at least 1")
    elif s.side and s.ring < 2:
        out.append(f"{s.name}: a side lane needs ring >= 2, so a call never overwrites buffers the side lane still reads")
    if s.main_priority not in MAIN_PRIORITIES:
        out.append(f"{s.name}: main_priority {s.main_priority!r} (one of {', '.join(MAIN_PRIORITIES)})")
    if s.side_priority not in SIDE_PRIORITIES:
        out.append(f"{s.name}: side_priority {s.side_priority!r} (one of {', '.join(SIDE_PRIORITIES)})")
    return out


MANIFEST_SCHEMA = "verity/pouw/pass-manifest/v1"
#: the manifests a verifier reads: this one, and the MVP's (`verity_vllm.protocol_options.pouw_pearl_c_device`)
MANIFEST_SCHEMAS_READ = (MANIFEST_SCHEMA, "verity-vllm/pouw-pearl-c-run/v1")


@dataclass(frozen=True)
class PassCommitment:
    """``activation_roots`` is the SHA-256 over every call's A root in call order; ``kernel`` the kernel variant's id;
    ``controls`` every poisoning or negative control the pass ran with, empty for an honest pass."""

    scheme: str
    rank: int
    world: int
    pass_index: int
    salt: bytes
    weights_root: bytes
    transcript_root: bytes
    activation_roots: bytes
    units: int
    tiles: int
    records_root: bytes | None = None
    schedule: str = "serial"
    kernel: str | None = None
    controls: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {k: (v.hex() if isinstance(v, bytes) else list(v) if isinstance(v, tuple) else v)
                for k, v in self.__dict__.items()}


@dataclass(frozen=True)
class StepIO:
    """One call step of a kernel's pipeline: its group (`GROUPS`) and the buffers it reads and writes.  A schedule rotates
    the per-call buffers its side lane reads, so a call never overwrites what the side lane still reads."""

    group: str
    reads: frozenset[str] = frozenset()
    writes: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ShapeRule:
    """The shapes a kernel takes: k and n multiples, k's range, and the row granularity m is padded to."""

    k_multiple: int = 1
    n_multiple: int = 1
    m_align: int = 1
    k_min: int = 1
    k_max: int = 1 << 24

    def refusal(self, n: int, k: int) -> str | None:
        if k % self.k_multiple or n % self.n_multiple or not self.k_min <= k <= self.k_max:
            return (f"(n, k) = ({n}, {k}): the kernel takes k a multiple of {self.k_multiple} in [{self.k_min}, {self.k_max}] "
                    f"and n a multiple of {self.n_multiple}")
        return None


@dataclass(frozen=True)
class KernelVariant:
    """A kernel, named by what it was built from: ``id`` is ``{family}-{device}/{source_sha256[:12]}``, so a tuned choice
    that changes a source is a new variant.  ``prices`` names the priced path it implements (the cast and store the
    published γ reads), ``loader`` how the runtime loads it."""

    family: str
    device: str
    schemes: frozenset[str]
    schedules: frozenset[str]
    shapes: ShapeRule
    source_sha256: str
    binary_sha256: str
    loader: str
    prices: str | None = None
    steps: Mapping[str, StepIO] = field(default_factory=dict, compare=False)

    @property
    def id(self) -> str:
        return f"{self.family}-{self.device}/{self.source_sha256[:12]}"


@dataclass(frozen=True)
class GateRecord:
    """The device gate's result for one variant: bit-exact at ``shapes`` against the scheme's reference, poisoned
    transcripts accepted, controls rejected, every declared schedule committing the serial schedule's bytes, and the SASS
    rule held under the pinned ``toolkit``.  ``gates`` is the gate code's version, so a row's validity is derived from it."""

    variant: str
    run: str
    gates: str
    shapes: tuple[tuple[int, int], ...]
    exact: bool
    poisoned_accepted: bool
    controls_rejected: bool
    schedules_equal: bool
    sass_ok: bool
    toolkit: str
    prices: str | None = None

    @property
    def passed(self) -> bool:
        return self.exact and self.poisoned_accepted and self.controls_rejected and self.schedules_equal and self.sass_ok


def variant_refusals(v: KernelVariant, scheme: str, schedule: str, shapes: Sequence[tuple[int, int]],
                     gate: GateRecord | None) -> list[str]:
    """Why `v` can't serve `scheme` under `schedule` at every (n, k) of `shapes` (empty: it can): it must compute both, take
    every shape, and hold a passing gate of its own that covers every shape."""
    out = []
    if scheme not in v.schemes:
        out.append(f"{v.id} computes {', '.join(sorted(v.schemes))}, not {scheme}")
    if schedule not in v.schedules:
        out.append(f"{v.id} runs {', '.join(sorted(v.schedules))}, not {schedule}")
    out.extend(f"{v.id}: {r}" for n, k in shapes if (r := v.shapes.refusal(n, k)))
    if gate is None:
        out.append(f"{v.id} has no device gate")
    else:
        if gate.variant != v.id:
            out.append(f"{v.id}: the gate is {gate.variant}'s")
        if not gate.passed:
            failed = [f for f in ("exact", "poisoned_accepted", "controls_rejected", "schedules_equal", "sass_ok")
                      if not getattr(gate, f)]
            out.append(f"{v.id}: its gate ({gate.run}) failed {', '.join(failed)}")
        missing = sorted(set(map(tuple, shapes)) - set(gate.shapes))
        if missing:
            out.append(f"{v.id}: its gate didn't cover {missing}")
        if v.prices is not None and gate.prices != v.prices:
            out.append(f"{v.id}: gated on priced path {gate.prices}, declares {v.prices}")
    return out


__all__ = ["DEFERRED", "DEFERRED_CALLER", "FP4_PRECISIONS", "FP8_DEVICES", "FP8_FORMINGS", "GROUPS", "HASH_FORMATS", "MAIN_PRIORITIES",
           "MANIFEST_SCHEMA", "MANIFEST_SCHEMAS_READ", "MOVABLE", "SCHEDULES", "SERIAL", "SIDE_PRIORITIES", "GateRecord",
           "HashFormat", "KernelVariant", "PassCommitment", "Schedule", "SchemeId", "ShapeRule", "StepIO", "hash_format",
           "schedule_refusals", "variant_refusals"]
