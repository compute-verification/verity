"""What a served PoUW executor and its verifier agree on, whatever runs the model: the hashing formats, the Pearl-C family's
scheme names, the schedules a pass may run under, and what one pass publishes.

* `HashFormat` (`HASH_FORMATS`): a format's tree hash, whether forming waits for A's tree, and what a row's seed is keyed
  on.  An executor reads the tree hash from here, never from a literal.  The formats are h0 (TurboSHAKE128 digests, SHA-256
  trees) and h3 (SHA-512 throughout, each row seeding its own E_A line); the BLAKE3 formats h1, h2 and BLAKE3 -h3 are in
  ``archive/pouw-blake3/`` (one hash, SHA-512).
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

from verity.ml.commitments import domains


@dataclass(frozen=True)
class HashFormat:
    """``seed_a`` is ``unit-root`` when forming reads seed_A from A's root (so A's tree is on the critical path), ``row``
    when each row seeds its own E_A line (h3).  ``row_digest`` is the schema of the row digest a row's seed is keyed on
    (h3: ``sha512/row-seg/v1``, `rowleaf.sha512_row_seg_digest`), None when rows don't seed."""

    name: str
    tree_hash: str
    seed_a: str
    row_cap: bool = False
    row_digest: str | None = None


HASH_FORMATS: dict[str, HashFormat] = {
    "h0": HashFormat("h0", "sha256", "unit-root"),
    "h3": HashFormat("h3", "hm96-sha512", "row", row_cap=True, row_digest=domains.SHA512_ROW_SEG_DIGEST),
}

#: `pearl_c_device.DEVICES`' records: the panel's pearl-c-sm120 v1 is ``sm120`` (G = 4), its v2 ``sm120-unpromoted``
FP8_DEVICES = ("h100", "sm120", "sm120-unpromoted")
FP8_FORMINGS = ("v0", "v1")
FP4_PRECISIONS = ("nvfp4", "mxfp4")
_FP8 = re.compile(r"pearl-c-(?P<device>[a-z0-9]+(?:-(?!v[0-9])[a-z0-9]+)*)-(?P<forming>v[0-9]+)(?:-(?P<hashing>h[0-9]+))?"
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


#: a call's step groups, in the order the main lane runs them.  ``row_seeds`` is A's per-row keying (the pads, each row's
#: digest and its seed), which forming reads, so it never leaves the main lane; ``a_tree`` is A's hm96 leaves, tree and
#: unit seed, which may leave it where the format seeds forming per row (`schedule_refusals`)
GROUPS = ("row_seeds", "a_tree", "forming", "gemm", "cleanup", "tile_hash", "screen")
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
#: the measured deferred schedule (`prefill_fused.py`'s and `decode_chain.py`'s chains): the main lane at the
#: greatest priority, each call's tile hashing beside it at the least.  The screen stays on the main lane there
DEFERRED = Schedule("deferred", frozenset({"tile_hash"}), ring=2, main_priority="greatest")
#: the same placement with the caller's stream left at its own priority (the first placement), kept to time against DEFERRED
DEFERRED_CALLER = Schedule("deferred-caller", frozenset({"tile_hash"}), ring=2)
#: DEFERRED with A's tree beside the tile hashing: the main lane keeps only the per-row keying (`row_seeds`), about a fifth
#: of a call's hashing.  Runs only where the format seeds forming per row (h3)
DEFERRED_A = Schedule("deferred-a", frozenset({"a_tree", "tile_hash"}), ring=2, main_priority="greatest")
SCHEDULES = {s.name: s for s in (SERIAL, DEFERRED, DEFERRED_CALLER, DEFERRED_A)}


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
               else "forming reads each row's digest and seed" if g == "row_seeds"
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


MANIFEST_SCHEMA = domains.POUW_PASS_MANIFEST
#: the manifests a verifier reads: this one, and the MVP's (`verity_vllm.protocols.pouw.pearl_c`), v1 (salted
#: from a beacon, before epoch coins) only as a replay of a recorded pass
MANIFEST_SCHEMAS_READ = (MANIFEST_SCHEMA, domains.VLLM_PEARL_C_RUN, domains.VLLM_PEARL_C_RUN_V1)


@dataclass(frozen=True)
class PassCommitment:
    """``activation_roots`` is the SHA-512 over every call's A root in call order; ``kernel`` the kernel variant's id;
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


__all__ = ["DEFERRED", "DEFERRED_A", "DEFERRED_CALLER", "FP4_PRECISIONS", "FP8_DEVICES", "FP8_FORMINGS", "GROUPS", "HASH_FORMATS", "MAIN_PRIORITIES",
           "MANIFEST_SCHEMA", "MANIFEST_SCHEMAS_READ", "MOVABLE", "SCHEDULES", "SERIAL", "SIDE_PRIORITIES", "GateRecord",
           "HashFormat", "KernelVariant", "PassCommitment", "Schedule", "SchemeId", "ShapeRule", "StepIO", "hash_format",
           "schedule_refusals", "variant_refusals"]
