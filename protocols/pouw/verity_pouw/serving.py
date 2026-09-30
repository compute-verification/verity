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

Stdlib and `verity` only (the repository's ``tests/test_protocol_boundaries.py``).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any

from verity.commitments.blake3 import blake3


@dataclass(frozen=True)
class HashFormat:
    """``seed_a`` is ``unit-root`` when forming reads seed_A from A's root (so A's tree is on the critical path), ``row``
    when each row seeds its own E_A line (h3)."""

    name: str
    tree_hash: str
    message_label: str | None
    leaf_label: str | None
    seed_a: str
    row_cap: bool = False

    def message_key(self) -> bytes | None:
        return None if self.message_label is None else blake3(self.message_label.encode())

    def leaf_key(self) -> bytes | None:
        return None if self.leaf_label is None else blake3(self.leaf_label.encode())


HASH_FORMATS: dict[str, HashFormat] = {
    "h0": HashFormat("h0", "sha256", None, None, "unit-root"),
    "h1": HashFormat("h1", "blake3", "pearl-c/h1/message", "pearl-c/h1/leaf", "unit-root"),
    "h2": HashFormat("h2", "blake3-s256", "pearl-c/h1/message", "pearl-c/h1/leaf", "unit-root"),
    "h3": HashFormat("h3", "blake3-s256", "pearl-c/h1/message", "pearl-c/h1/leaf", "row", row_cap=True),
}

FP8_DEVICES = ("h100", "sm120")
FP8_FORMINGS = ("v0", "v1", "v2", "v2-hot")
FP4_PRECISIONS = ("nvfp4", "mxfp4")
_FP8 = re.compile(r"pearl-c-(?P<device>[a-z0-9]+)-(?P<forming>v2-hot|v[0-9]+)(?:-(?P<hashing>h[0-9]+))?(?:@(?P<prices>[a-z0-9_]+))?")
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
DEFERRED = Schedule("deferred", frozenset({"tile_hash", "screen"}), ring=2)
SCHEDULES = {s.name: s for s in (SERIAL, DEFERRED)}


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


__all__ = ["DEFERRED", "FP4_PRECISIONS", "FP8_DEVICES", "FP8_FORMINGS", "GROUPS", "HASH_FORMATS", "MAIN_PRIORITIES",
           "MANIFEST_SCHEMA", "MANIFEST_SCHEMAS_READ", "MOVABLE", "SCHEDULES", "SERIAL", "SIDE_PRIORITIES", "HashFormat",
           "PassCommitment", "Schedule", "SchemeId", "hash_format", "schedule_refusals"]
