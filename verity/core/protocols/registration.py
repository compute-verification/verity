"""The registration log in Python: `Core.Registration` (`interface/CoreProtocol/Registration.lean`), the type every
subprotocol and both composites read, entry for entry and refusal for refusal. The composition suite's difftest holds
this module to the Lean (`composition/tests/test_registration_difftest.py`).

A deployment's registration is one append-only log of entries, each naming the head of the log before it (`previous`):
the digest of the log's last entry, or its composite's genesis when the log is empty, so the head names the whole log.
<<<<<<< 94ca7fd0d18
<<<<<<< HEAD
An entry is one of three kinds:
||||||| merged common ancestors
An entry is one of two formats:
=======
An entry is one of three formats:
>>>>>>> 705be2b532
||||||| e10cf2c8559
An entry is one of three formats:
=======
An entry is one of three kinds:
>>>>>>> 7af4d24fca3e

<<<<<<< 94ca7fd0d18
<<<<<<< HEAD
* a site entry, the Site's own (`verity.core.site.Entry`, `verity/registration/site/v1`): the sheets, locations and
  links it adds to the Site, its JSON, digest and check the Site's;
* an information entry (`Information`, `verity/registration/information/v1`): its items, each registered once in the
  log (`item_key`): θ, each registered value by its committed root, and the builder B (its digest, its advice's format
  and its public cap ℓ on one declaration's advice, in bits). Registration registers the site, θ and B, not the circuit
  B builds; a subprotocol's Program and parameters, and the proof system, are that protocol's public verifier
  parameters;
* a declaration (`Declaration`, `verity/registration/declaration/v1`): a day's circuit, the day by its index among the
  monitoring period's days, its program's digest and its block Definitions by descriptor key. The prover declares one
  circuit a day, before the day's first coin, so a second declaration of a day is refused (`duplicate-declaration`), and
  each window's program is its day's (`View.declared`; the sampled-proofs registration record's R2).
||||||| merged common ancestors
* `verity/registration/hardware/v1`: the cards, locations and links it adds to the site (`Site`);
* `verity/registration/information/v1`: its items, each registered once in the log (`item_key`): θ, each registered
  value by its committed root, and the builder B (its digest, its advice's format and its public cap ℓ on one
  declaration's advice, in bits). Registration registers the site, θ and B, not the circuit B builds; a subprotocol's
  Program and parameters, and the proof system, are that protocol's public verifier parameters.
=======
* `verity/registration/hardware/v1`: the cards, locations and links it adds to the site (`Site`);
* `verity/registration/information/v1`: its items, each registered once in the log (`item_key`): θ, each registered
  value by its committed root, and the builder B (its digest, its advice's format and its public cap ℓ on one
  declaration's advice, in bits). Registration registers the site, θ and B, not the circuit B builds; a subprotocol's
  Program and parameters, and the proof system, are that protocol's public verifier parameters.
* `verity/registration/declaration/v1`: a day's circuit (`Declaration`): the day, by its index among the monitoring
  period's days, its program's digest and its block Definitions by descriptor key. The prover declares one circuit a
  day, before the day's first coin, so a second declaration of a day is refused (`duplicate-declaration`), and each
  window's program is its day's (`View.declared`; the sampled-proofs registration record's R2).
>>>>>>> 705be2b532
||||||| e10cf2c8559
* `verity/registration/hardware/v1`: the cards, locations and links it adds to the site (`Site`);
* `verity/registration/information/v1`: its items, each registered once in the log (`item_key`): θ, each registered
  value by its committed root, and the builder B (its digest, its advice's format and its public cap ℓ on one
  declaration's advice, in bits). Registration registers the site, θ and B, not the circuit B builds; a subprotocol's
  Program and parameters, and the proof system, are that protocol's public verifier parameters.
* `verity/registration/declaration/v1`: a day's circuit (`Declaration`): the day, by its index among the monitoring
  period's days, its program's digest and its block Definitions by descriptor key. The prover declares one circuit a
  day, before the day's first coin, so a second declaration of a day is refused (`duplicate-declaration`), and each
  window's program is its day's (`View.declared`; the sampled-proofs registration record's R2).
=======
* a site entry, the Site's own (`verity.core.site.Entry`, `verity/registration/site/v1`): the sheets, locations and
  links it adds to the Site, its JSON, digest and check the Site's;
* an information entry (`Information`, `verity/registration/information/v1`): its items, each registered once in the
  log (`item_key`): θ, each registered value by its committed root, and the builder B (its digest, its advice's format
  and its public cap ℓ on one declaration's advice, in bits). Registration registers the site, θ and B, not the circuit
  B builds; a subprotocol's Program and parameters, and the proof system, are that protocol's public verifier
  parameters;
* a declaration (`Declaration`, `verity/registration/declaration/v1`): a day's circuit: the day, by its index among
  the monitoring period's days, its program's digest and its block Definitions by descriptor key. The prover declares
  one circuit a day, before the day's first coin, so a second declaration of a day is refused (`duplicate-declaration`),
  and each window's program is its day's (`View.declared`; the sampled-proofs registration record's R2).
>>>>>>> 7af4d24fca3e

~~~text
entry digest   SHA-512(format ‖ 0 ‖ canon(entry))                          the entry's format and previous among its keys
genesis        the Site's: SHA-512("verity/registration/genesis/v1" ‖ 0 ‖ canon({"composite": NAME}))
the log        {"composite": "onsite" | "remote", "entries": [ENTRY, ...]}   committed; its digest is its head
~~~

`check` holds each entry to the log before it and raises `Refused` at the first entry that fails, by the first check it
<<<<<<< 94ca7fd0d18
<<<<<<< HEAD
fails (`REFUSALS`): a site entry by the Site's (`verity.core.site.entry_refusal`), an information entry by `previous`
and then `duplicate-information`, a declaration by `previous` and then `duplicate-declaration`. A component reads its
view of an accepted log (`View`): the Site, θ's values (`View.value`), the builder and the days' declared circuits
(`View.declared`), at the head. Decoding takes exactly what the encoders write and raises `Malformed`
otherwise; a log or parameters holding a character outside ASCII without DEL are `Malformed` (`ascii`), in
`Core.canonicalJson`'s words.
||||||| merged common ancestors
fails (`REFUSALS`, in order). A component reads its view of an accepted log (`View`): the site, θ's values
(`View.value`) and the builder, at the head. Decoding takes exactly what the encoders write and raises `Malformed`
otherwise; registration is ASCII-only without DEL, and a log or parameters holding any other character is `Malformed` as
`ascii` before any digest is taken.
=======
fails (`REFUSALS`, in order). A component reads its view of an accepted log (`View`): the site, θ's values
(`View.value`), the builder and the days' declared circuits (`View.declared`), at the head. Decoding takes exactly what the encoders write and raises `Malformed`
otherwise; registration is ASCII-only without DEL, and a log or parameters holding any other character is `Malformed` as
`ascii` before any digest is taken.
>>>>>>> 705be2b532
||||||| e10cf2c8559
fails (`REFUSALS`, in order). A component reads its view of an accepted log (`View`): the site, θ's values
(`View.value`), the builder and the days' declared circuits (`View.declared`), at the head. Decoding takes exactly what the encoders write and raises `Malformed`
otherwise; registration is ASCII-only without DEL, and a log or parameters holding any other character is `Malformed` as
`ascii` before any digest is taken.
=======
fails (`REFUSALS`): a site entry by the Site's (`verity.core.site.entry_refusal`), an information entry by `previous`
and then `duplicate-information`, and a declaration by `previous` and then `duplicate-declaration`. A component reads
its view of an accepted log (`View`): the Site, θ's values (`View.value`), the builder and the days' declared circuits
(`View.declared`), at the head. Decoding takes exactly what the encoders write and raises `Malformed`
otherwise; a log or parameters holding a character outside ASCII without DEL are `Malformed` (`ascii`), in
`Core.canonicalJson`'s words.
>>>>>>> 7af4d24fca3e
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from verity.core import site as S
from verity.core.site import Auditor, Malformed
from verity.core.wire import canonical

INFORMATION = "verity/registration/information/v1"
DECLARATION = "verity/registration/declaration/v1"
PROTOCOL = "verity/registration/log/v1"
<<<<<<< 94ca7fd0d18
<<<<<<< HEAD
REFUSALS = (*S.REFUSALS, "duplicate-information", "duplicate-declaration")
||||||| merged common ancestors
OUTSIDE = "outside"
"""The rest of the world: an end every site has, whose id no location may take."""
COMPOSITES = ("onsite", "remote")
KINDS = ("site", "host", "device", "memory", "certifier", "challenger", "outside")
TIERS = ("hbm", "dram", "disk")
OWNERS = ("prover", "auditor")
PARENTS: dict[str, tuple[str | None, ...]] = {
    "site": (None,), "host": ("site",), "device": ("host",), "memory": ("device", "host"), "certifier": ("site",),
    "challenger": ("site",), "outside": ()}
"""The kinds of parent a location of each kind may have, None being the root."""
LINKS = (("host", "host"), ("host", "outside"), ("certifier", "host"), ("certifier", "outside"), ("challenger", "host"))
"""The pairs of kinds a link may join, in either order."""
AUDITOR_KINDS = ("certifier", "challenger")
REFUSALS = ("previous", "duplicate-id", "unknown-ref", "kind", "card-digest", "figures", "capacity", "owner",
            "uncertified", "challenger-reach", "duplicate-information")
OPERAND_CLASSES = ("uniform-s8", "uniform-s2", "uniform-u2", "uniform-u4", "zero")
"""The operand classes of a tensor-core multiply-add, `DeviceFigures`' `OperandClass`: int8 operands drawn uniformly
over `[-128, 127]`, `[-2, 1]`, `[0, 3]` or `[0, 15]`, or the one value 0."""


class Malformed(ValueError):
    """Bytes or JSON that aren't what the encoders write."""
=======
OUTSIDE = "outside"
"""The rest of the world: an end every site has, whose id no location may take."""
COMPOSITES = ("onsite", "remote")
KINDS = ("site", "host", "device", "memory", "certifier", "challenger", "outside")
TIERS = ("hbm", "dram", "disk")
OWNERS = ("prover", "auditor")
PARENTS: dict[str, tuple[str | None, ...]] = {
    "site": (None,), "host": ("site",), "device": ("host",), "memory": ("device", "host"), "certifier": ("site",),
    "challenger": ("site",), "outside": ()}
"""The kinds of parent a location of each kind may have, None being the root."""
LINKS = (("host", "host"), ("host", "outside"), ("certifier", "host"), ("certifier", "outside"), ("challenger", "host"))
"""The pairs of kinds a link may join, in either order."""
AUDITOR_KINDS = ("certifier", "challenger")
REFUSALS = ("previous", "duplicate-id", "unknown-ref", "kind", "card-digest", "figures", "capacity", "owner",
            "uncertified", "challenger-reach", "duplicate-information", "duplicate-declaration")
OPERAND_CLASSES = ("uniform-s8", "uniform-s2", "uniform-u2", "uniform-u4", "zero")
"""The operand classes of a tensor-core multiply-add, `DeviceFigures`' `OperandClass`: int8 operands drawn uniformly
over `[-128, 127]`, `[-2, 1]`, `[0, 3]` or `[0, 15]`, or the one value 0."""


class Malformed(ValueError):
    """Bytes or JSON that aren't what the encoders write."""
>>>>>>> 705be2b532
||||||| e10cf2c8559
OUTSIDE = "outside"
"""The rest of the world: an end every site has, whose id no location may take."""
COMPOSITES = ("onsite", "remote")
KINDS = ("site", "host", "device", "memory", "certifier", "challenger", "outside")
TIERS = ("hbm", "dram", "disk")
OWNERS = ("prover", "auditor")
PARENTS: dict[str, tuple[str | None, ...]] = {
    "site": (None,), "host": ("site",), "device": ("host",), "memory": ("device", "host"), "certifier": ("site",),
    "challenger": ("site",), "outside": ()}
"""The kinds of parent a location of each kind may have, None being the root."""
LINKS = (("host", "host"), ("host", "outside"), ("certifier", "host"), ("certifier", "outside"), ("challenger", "host"))
"""The pairs of kinds a link may join, in either order."""
AUDITOR_KINDS = ("certifier", "challenger")
REFUSALS = ("previous", "duplicate-id", "unknown-ref", "kind", "card-digest", "figures", "capacity", "owner",
            "uncertified", "challenger-reach", "duplicate-information", "duplicate-declaration")
OPERAND_CLASSES = ("uniform-s8", "uniform-s2", "uniform-u2", "uniform-u4", "zero")
"""The operand classes of a tensor-core multiply-add, `DeviceFigures`' `OperandClass`: int8 operands drawn uniformly
over `[-128, 127]`, `[-2, 1]`, `[0, 3]` or `[0, 15]`, or the one value 0."""


class Malformed(ValueError):
    """Bytes or JSON that aren't what the encoders write."""
=======
REFUSALS = (*S.REFUSALS, "duplicate-information", "duplicate-declaration")
>>>>>>> 7af4d24fca3e


class Refused(Exception):
    """The check refuses entry `entry` by `name`, one of `REFUSALS`."""

    def __init__(self, entry: int, name: str) -> None:
        super().__init__(f"entry {entry}: {name}")
        self.entry, self.name = entry, name


@dataclass(frozen=True)
class Information:
    """An information entry: the items it registers, each canonical JSON with its `kind`: `theta` (`value`, `root`,
    `leaves`, `words`, `domain`, `word_bits` when it is 32, `checkpoint_sha512` for public weights) or `builder`
    (`digest`, `advice`, `cap`), and the head of the log it follows (`previous`, set by `chain`)."""

    items: tuple[Mapping[str, Any], ...] = ()
    previous: str = ""

    def to_json(self) -> dict[str, Any]:
        return {"format": INFORMATION, "items": [dict(i) for i in self.items], "previous": self.previous}

    def digest(self) -> str:
        return S.tagged(INFORMATION, self.to_json())


@dataclass(frozen=True)
class Declaration:
<<<<<<< 94ca7fd0d18
    """A day's declared circuit (`Core.Registration.Declared`): the day, by its index among the monitoring period's days,
    its program's SHA-512 as hex, its block Definitions by descriptor key, and the head of the log it follows
    (`previous`, set by `chain`)."""

    day: int
    program: str
    blocks: tuple[str, ...] = ()
    previous: str = ""

    def to_json(self) -> dict[str, Any]:
        return {"blocks": list(self.blocks), "day": self.day, "format": DECLARATION, "previous": self.previous,
                "program": self.program}

    def declared(self) -> dict[str, Any]:
        """What the view reads of it, `{"blocks", "day", "program"}`."""
        return {"blocks": list(self.blocks), "day": self.day, "program": self.program}

    def digest(self) -> str:
        return S.tagged(DECLARATION, self.to_json())


Entry = S.Entry | Information | Declaration
"""An entry of the registration log: the Site's site entry, an information entry or a declaration."""
||||||| e10cf2c8559
=======
    """A day's declared circuit (`Core.Registration.Declared`): the day, by its index among the monitoring period's
    days, its program's SHA-512 as hex, and its block Definitions by descriptor key; and the head of the log it follows
    (`previous`, set by `chain`)."""

    day: int
    program: str
    blocks: tuple[str, ...] = ()
    previous: str = ""

    def body(self) -> dict[str, Any]:
        return {"blocks": list(self.blocks), "day": self.day, "program": self.program}

    def to_json(self) -> dict[str, Any]:
        return {**self.body(), "format": DECLARATION, "previous": self.previous}

    def digest(self) -> str:
        return S.tagged(DECLARATION, self.to_json())


Entry = S.Entry | Information | Declaration
"""An entry of the registration log: the Site's site entry, an information entry, or a declaration."""
>>>>>>> 7af4d24fca3e


def item_key(item: Mapping[str, Any]) -> str:
    """What an item registers, which the log registers once: each θ value by its name, and one builder."""
    return "theta " + item["value"] if item["kind"] == "theta" else item["kind"]


<<<<<<< 94ca7fd0d18
<<<<<<< HEAD
def site(sheets: Sequence[S.Sheet] = (), locs: Sequence[S.Loc] = (), links: Sequence[S.Link] = ()) -> S.Entry:
    """A site entry adding `sheets`, `locs` and `links`, its `previous` for `chain` to set."""
    return S.Entry("", tuple(sheets), tuple(locs), tuple(links))
||||||| merged common ancestors
@dataclass(frozen=True)
class Entry:
    previous: str
    body: Hardware | Information

    @property
    def format(self) -> str:
        return HARDWARE if isinstance(self.body, Hardware) else INFORMATION

    def to_json(self) -> dict[str, Any]:
        b = self.body
        if isinstance(b, Hardware):
            return {"cards": [k.to_json() for k in b.cards], "format": HARDWARE,
                    "links": [k.to_json() for k in b.links], "locs": [x.to_json() for x in b.locs],
                    "previous": self.previous}
        return {"format": INFORMATION, "items": [dict(i) for i in b.items], "previous": self.previous}

    def digest(self) -> str:
        return tagged(self.format, self.to_json())


def genesis(composite: str) -> str:
    """The head of a composite's empty log."""
    return tagged(GENESIS, {"composite": composite})
=======
@dataclass(frozen=True)
class Declaration:
    """A day's declared circuit (`Core.Registration.Declared`): the day, by its index among the monitoring period's days,
    its program's SHA-512 as hex, and its block Definitions by descriptor key."""

    day: int
    program: str
    blocks: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {"blocks": list(self.blocks), "day": self.day, "program": self.program}


@dataclass(frozen=True)
class Entry:
    previous: str
    body: Hardware | Information | Declaration

    @property
    def format(self) -> str:
        return (HARDWARE if isinstance(self.body, Hardware) else DECLARATION if isinstance(self.body, Declaration)
                else INFORMATION)

    def to_json(self) -> dict[str, Any]:
        b = self.body
        if isinstance(b, Hardware):
            return {"cards": [k.to_json() for k in b.cards], "format": HARDWARE,
                    "links": [k.to_json() for k in b.links], "locs": [x.to_json() for x in b.locs],
                    "previous": self.previous}
        if isinstance(b, Declaration):
            return {**b.to_json(), "format": DECLARATION, "previous": self.previous}
        return {"format": INFORMATION, "items": [dict(i) for i in b.items], "previous": self.previous}

    def digest(self) -> str:
        return tagged(self.format, self.to_json())


def genesis(composite: str) -> str:
    """The head of a composite's empty log."""
    return tagged(GENESIS, {"composite": composite})
>>>>>>> 705be2b532
||||||| e10cf2c8559
@dataclass(frozen=True)
class Declaration:
    """A day's declared circuit (`Core.Registration.Declared`): the day, by its index among the monitoring period's days,
    its program's SHA-512 as hex, and its block Definitions by descriptor key."""

    day: int
    program: str
    blocks: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {"blocks": list(self.blocks), "day": self.day, "program": self.program}


@dataclass(frozen=True)
class Entry:
    previous: str
    body: Hardware | Information | Declaration

    @property
    def format(self) -> str:
        return (HARDWARE if isinstance(self.body, Hardware) else DECLARATION if isinstance(self.body, Declaration)
                else INFORMATION)

    def to_json(self) -> dict[str, Any]:
        b = self.body
        if isinstance(b, Hardware):
            return {"cards": [k.to_json() for k in b.cards], "format": HARDWARE,
                    "links": [k.to_json() for k in b.links], "locs": [x.to_json() for x in b.locs],
                    "previous": self.previous}
        if isinstance(b, Declaration):
            return {**b.to_json(), "format": DECLARATION, "previous": self.previous}
        return {"format": INFORMATION, "items": [dict(i) for i in b.items], "previous": self.previous}

    def digest(self) -> str:
        return tagged(self.format, self.to_json())


def genesis(composite: str) -> str:
    """The head of a composite's empty log."""
    return tagged(GENESIS, {"composite": composite})
=======
def site(sheets: Sequence[S.Sheet] = (), locs: Sequence[S.Loc] = (), links: Sequence[S.Link] = ()) -> S.Entry:
    """A site entry adding `sheets`, `locs` and `links`, its `previous` for `chain` to set."""
    return S.Entry("", tuple(sheets), tuple(locs), tuple(links))
>>>>>>> 7af4d24fca3e


def head(composite: str, log: Sequence[Entry]) -> str:
    """The log's head, which names it."""
    return log[-1].digest() if log else S.genesis(composite)


<<<<<<< 94ca7fd0d18
<<<<<<< HEAD
def chain(composite: str, entries: Sequence[Entry]) -> list[Entry]:
    """The log of `entries` in order, each following the head of the log before it."""
||||||| merged common ancestors
def chain(composite: str, bodies: Sequence[Hardware | Information]) -> list[Entry]:
    """The log of `bodies` in order, each entry following the head of the log before it."""
=======
def chain(composite: str, bodies: Sequence[Hardware | Information | Declaration]) -> list[Entry]:
    """The log of `bodies` in order, each entry following the head of the log before it."""
>>>>>>> 705be2b532
||||||| e10cf2c8559
def chain(composite: str, bodies: Sequence[Hardware | Information | Declaration]) -> list[Entry]:
    """The log of `bodies` in order, each entry following the head of the log before it."""
=======
def chain(composite: str, entries: Sequence[Entry]) -> list[Entry]:
    """The log of `entries` in order, each following the head of the log before it."""
>>>>>>> 7af4d24fca3e
    log: list[Entry] = []
    for e in entries:
        log.append(replace(e, previous=head(composite, log)))
    return log


def committed(composite: str, log: Sequence[Entry]) -> bytes:
    """The log as committed: the canonical JSON the protocol's commitment is."""
    return canonical({"composite": composite, "entries": [e.to_json() for e in log]})


# --- decoding: exactly what the encoders write ------------------------------------------------------------------------

def _exactly(j: Any, keys: Sequence[str], what: str) -> None:
    if not isinstance(j, dict) or sorted(j) != list(keys):
        raise Malformed(f"{what}: the keys must be [{', '.join(keys)}]")


def _str(j: Any, what: str) -> str:
    if not isinstance(j, str):
        raise Malformed(f"{what}: not a string")
    return j


def _nat(j: Any, what: str) -> int:
    if not isinstance(j, int) or isinstance(j, bool) or j < 0:
        raise Malformed(f"{what}: not a natural number")
    return j


def _list(j: Any, what: str) -> list:
    if not isinstance(j, list):
        raise Malformed(f"{what}: not a list")
    return j


def _one_of(j: Any, names: Sequence[str], what: str) -> str:
    if _str(j, what) not in names:
        raise Malformed(f"{what}: {j} is not one of {list(names)}")
    return j


def _hex64(j: Any, what: str) -> str:
    """`Flock.unhex` of 64 bytes: lowercase hex of a SHA-512 digest."""
    s = _str(j, what)
    if len(s) % 2 or not set(s) <= set("0123456789abcdef"):
        raise Malformed(f"{what}: not lowercase hex")
    if len(s) != 128:
        raise Malformed(f"{what}: not a SHA-512 digest (64 bytes)")
    return s


THETA_KEYS = ["domain", "kind", "leaves", "root", "value", "words"]
"""A θ item's keys, besides `word_bits` (present only as 32) and `checkpoint_sha512` (a public checkpoint's pin)."""
BUILDER_KEYS = ["advice", "cap", "digest", "kind"]


def item_of(j: Any) -> dict[str, Any]:
    if not isinstance(j, dict) or "kind" not in j:
        raise Malformed("an item: kind missing")
    kind = _str(j["kind"], "an item's kind")
    if kind == "theta":
        _exactly(j, sorted(THETA_KEYS + [k for k in ("checkpoint_sha512", "word_bits") if k in j]), "a theta item")
        if "word_bits" in j and _nat(j["word_bits"], "word_bits") != 32:
            raise Malformed("a theta item: word_bits is 32 when present (16 is left out)")
        out = {"domain": _hex64(j["domain"], "domain"), "kind": kind, "leaves": _nat(j["leaves"], "leaves"),
               "root": _hex64(j["root"], "root"), "value": _str(j["value"], "value"), "words": _nat(j["words"], "words")}
        if "word_bits" in j:
            out["word_bits"] = 32
        if "checkpoint_sha512" in j:
            out["checkpoint_sha512"] = _hex64(j["checkpoint_sha512"], "checkpoint_sha512")
        return out
    if kind == "builder":
        _exactly(j, BUILDER_KEYS, "a builder item")
        return {"advice": _str(j["advice"], "advice"), "cap": _nat(j["cap"], "cap"), "digest": _hex64(j["digest"], "digest"),
                "kind": kind}
    raise Malformed(f"an item: kind {kind} is not one of [theta, builder]")


def ascii(j: Any) -> None:
    """ASCII without DEL, refused in `Core.canonicalJson`'s words: `j`'s JSON, unescaped, holds no character from 0x7f
    up."""
    if any(ord(c) >= 0x7F for c in json.dumps(j, ensure_ascii=False)):
        raise Malformed("not ASCII without DEL (a byte at or above 0x7f)")


def entry_of(j: Any) -> Entry:
    if not isinstance(j, dict) or "format" not in j:
        raise Malformed("an entry: format missing")
    fmt = _str(j["format"], "an entry's format")
    if fmt == S.SITE:
        return S.Entry.of_json(j)
    if fmt == INFORMATION:
        _exactly(j, ["format", "items", "previous"], "an information entry")
<<<<<<< 94ca7fd0d18
<<<<<<< HEAD
        return Information(tuple(item_of(i) for i in _list(j["items"], "items")), _str(j["previous"], "previous"))
    if fmt == DECLARATION:
        _exactly(j, ["blocks", "day", "format", "previous", "program"], "a declaration")
        previous, day, program = _str(j["previous"], "previous"), _nat(j["day"], "day"), _hex64(j["program"], "program")
        blocks = tuple(_str(b, "a declaration's block") for b in _list(j["blocks"], "blocks"))
        return Declaration(day, program, blocks, previous)
    raise Malformed(f"an entry: format {fmt} is not one of [{S.SITE}, {INFORMATION}, {DECLARATION}]")
||||||| merged common ancestors
        previous = _hex64(j["previous"], "previous")
        return Entry(previous, Information(tuple(item_of(i) for i in _list(j["items"], "items"))))
    raise Malformed(f"an entry: format {fmt} is not one of [{HARDWARE}, {INFORMATION}]")
=======
        previous = _hex64(j["previous"], "previous")
        return Entry(previous, Information(tuple(item_of(i) for i in _list(j["items"], "items"))))
||||||| e10cf2c8559
        previous = _hex64(j["previous"], "previous")
        return Entry(previous, Information(tuple(item_of(i) for i in _list(j["items"], "items"))))
=======
        return Information(tuple(item_of(i) for i in _list(j["items"], "items")), _str(j["previous"], "previous"))
>>>>>>> 7af4d24fca3e
    if fmt == DECLARATION:
        _exactly(j, ["blocks", "day", "format", "previous", "program"], "a declaration")
        previous = _hex64(j["previous"], "previous")
        blocks = tuple(_str(b, "a declaration's block") for b in _list(j["blocks"], "blocks"))
<<<<<<< 94ca7fd0d18
        return Entry(previous, Declaration(_nat(j["day"], "day"), _hex64(j["program"], "program"), blocks))
    raise Malformed(f"an entry: format {fmt} is not one of [{HARDWARE}, {INFORMATION}, {DECLARATION}]")
>>>>>>> 705be2b532
||||||| e10cf2c8559
        return Entry(previous, Declaration(_nat(j["day"], "day"), _hex64(j["program"], "program"), blocks))
    raise Malformed(f"an entry: format {fmt} is not one of [{HARDWARE}, {INFORMATION}, {DECLARATION}]")
=======
        return Declaration(_nat(j["day"], "day"), _hex64(j["program"], "program"), blocks, previous)
    raise Malformed(f"an entry: format {fmt} is not one of [{S.SITE}, {INFORMATION}, {DECLARATION}]")
>>>>>>> 7af4d24fca3e


def log_of(j: Any) -> tuple[str, list[Entry]]:
    """The committed log's composite and entries."""
    ascii(j)
    _exactly(j, ["composite", "entries"], "the log")
    return _one_of(j["composite"], S.COMPOSITES, "the log's composite"), [entry_of(e) for e in _list(j["entries"], "entries")]


# --- the check --------------------------------------------------------------------------------------------------------

def _nodup(xs: Sequence[str]) -> bool:
    return len(set(xs)) == len(xs)


def site_refusal(auditor: Auditor, composite: str, s: S.Site, head: str, e: S.Entry) -> str | None:
    """The first check a site entry `e` fails against the Site `s` and head `head` of the log before it, or None: the
    Site's (`verity.core.site.entry_refusal`)."""
    return S.entry_refusal(composite, s, head, e, auditor)


@dataclass(frozen=True)
class State:
<<<<<<< 94ca7fd0d18
<<<<<<< HEAD
    """The log so far, folded: its Site, its information items in order, its head, its length and its declarations in
||||||| merged common ancestors
    """The log so far, folded: its site, its information items in order, its head and its length."""
=======
    """The log so far, folded: its site, its information items in order, its head, its length and its declarations in
>>>>>>> 705be2b532
||||||| e10cf2c8559
    """The log so far, folded: its site, its information items in order, its head, its length and its declarations in
=======
    """The log so far, folded: its Site, its information items in order, its head, its length and its declarations in
>>>>>>> 7af4d24fca3e
    order."""

    site: S.Site
    items: tuple[Mapping[str, Any], ...]
    head: str
    entries: int
    declared: tuple[Declaration, ...] = ()

    def refusal(self, auditor: Auditor, composite: str, e: Entry) -> str | None:
        if isinstance(e, S.Entry):
            return site_refusal(auditor, composite, self.site, self.head, e)
        if e.previous != self.head:
            return "previous"
<<<<<<< 94ca7fd0d18
<<<<<<< HEAD
        if isinstance(e, Declaration):
            return "duplicate-declaration" if any(d.day == e.day for d in self.declared) else None
        return None if _nodup([item_key(i) for i in self.items + e.items]) else "duplicate-information"
||||||| merged common ancestors
        if isinstance(e.body, Hardware):
            return hardware_refusal(auditor, composite, self.site, e.body)
        return None if _nodup([item_key(i) for i in self.items + e.body.items]) else "duplicate-information"
=======
        if isinstance(e.body, Hardware):
            return hardware_refusal(auditor, composite, self.site, e.body)
        if isinstance(e.body, Declaration):
            return "duplicate-declaration" if any(d.day == e.body.day for d in self.declared) else None
        return None if _nodup([item_key(i) for i in self.items + e.body.items]) else "duplicate-information"
>>>>>>> 705be2b532
||||||| e10cf2c8559
        if isinstance(e.body, Hardware):
            return hardware_refusal(auditor, composite, self.site, e.body)
        if isinstance(e.body, Declaration):
            return "duplicate-declaration" if any(d.day == e.body.day for d in self.declared) else None
        return None if _nodup([item_key(i) for i in self.items + e.body.items]) else "duplicate-information"
=======
        if isinstance(e, Declaration):
            return "duplicate-declaration" if any(d.day == e.day for d in self.declared) else None
        return None if _nodup([item_key(i) for i in self.items + e.items]) else "duplicate-information"
>>>>>>> 7af4d24fca3e

    def add(self, e: Entry) -> State:
<<<<<<< 94ca7fd0d18
<<<<<<< HEAD
        if isinstance(e, S.Entry):
            return replace(self, site=self.site.add(e), head=e.digest(), entries=self.entries + 1)
        if isinstance(e, Declaration):
            return replace(self, declared=self.declared + (e,), head=e.digest(), entries=self.entries + 1)
        return replace(self, items=self.items + e.items, head=e.digest(), entries=self.entries + 1)
||||||| merged common ancestors
        if isinstance(e.body, Hardware):
            return replace(self, site=self.site.add(e.body), head=e.digest(), entries=self.entries + 1)
        return replace(self, items=self.items + e.body.items, head=e.digest(), entries=self.entries + 1)
=======
        if isinstance(e.body, Hardware):
            return replace(self, site=self.site.add(e.body), head=e.digest(), entries=self.entries + 1)
        if isinstance(e.body, Declaration):
            return replace(self, declared=self.declared + (e.body,), head=e.digest(), entries=self.entries + 1)
        return replace(self, items=self.items + e.body.items, head=e.digest(), entries=self.entries + 1)
>>>>>>> 705be2b532
||||||| e10cf2c8559
        if isinstance(e.body, Hardware):
            return replace(self, site=self.site.add(e.body), head=e.digest(), entries=self.entries + 1)
        if isinstance(e.body, Declaration):
            return replace(self, declared=self.declared + (e.body,), head=e.digest(), entries=self.entries + 1)
        return replace(self, items=self.items + e.body.items, head=e.digest(), entries=self.entries + 1)
=======
        if isinstance(e, S.Entry):
            return replace(self, site=self.site.add(e), head=e.digest(), entries=self.entries + 1)
        if isinstance(e, Declaration):
            return replace(self, declared=self.declared + (replace(e, previous=""),), head=e.digest(),
                           entries=self.entries + 1)
        return replace(self, items=self.items + e.items, head=e.digest(), entries=self.entries + 1)
>>>>>>> 7af4d24fca3e

    def view(self) -> View:
        return View(self.site, tuple(dict(i) for i in self.items if i["kind"] == "theta"),
<<<<<<< HEAD
                    next((dict(i) for i in self.items if i["kind"] == "builder"), None), self.head,
                    tuple(d.declared() for d in self.declared))
||||||| merged common ancestors
                    next((dict(i) for i in self.items if i["kind"] == "builder"), None), self.head)
=======
                    next((dict(i) for i in self.items if i["kind"] == "builder"), None), self.head, self.declared)
>>>>>>> 705be2b532


def check(auditor: Auditor, composite: str, log: Sequence[Entry]) -> State:
    """The log folded entry by entry, each held to the log before it; `Refused` at the first entry that fails."""
    s = State(S.Site(), (), S.genesis(composite), 0)
    for e in log:
        if r := s.refusal(auditor, composite, e):
            raise Refused(s.entries, r)
        s = s.add(e)
    return s


@dataclass(frozen=True)
class View:
<<<<<<< 94ca7fd0d18
<<<<<<< HEAD
    """What a component reads of an accepted log, at its head: the Site, θ's registered values in order, the builder,
    and the days' declared circuits in order (`{"blocks", "day", "program"}` each)."""
||||||| merged common ancestors
    """What a component reads of an accepted log, at its head: the site, θ's registered values in order, the builder."""
=======
    """What a component reads of an accepted log, at its head: the site, θ's registered values in order, the builder,
||||||| e10cf2c8559
    """What a component reads of an accepted log, at its head: the site, θ's registered values in order, the builder,
=======
    """What a component reads of an accepted log, at its head: the Site, θ's registered values in order, the builder,
>>>>>>> 7af4d24fca3e
    and the days' declared circuits in order."""
>>>>>>> 705be2b532

    site: S.Site
    theta: tuple[dict[str, Any], ...]
    builder: dict[str, Any] | None
    head: str
<<<<<<< HEAD
    declarations: tuple[dict[str, Any], ...] = ()
||||||| merged common ancestors
=======
    declarations: tuple[Declaration, ...] = ()
>>>>>>> 705be2b532

    def value(self, name: str) -> dict[str, Any] | None:
        """θ's registered value `name`."""
        return next((t for t in self.theta if t["value"] == name), None)

<<<<<<< HEAD
    def declared(self, day: int) -> dict[str, Any] | None:
        """Day `day`'s declared circuit, the one declaration of the day in an accepted log; None when it declares none.
        A window of that day runs its program."""
        return next((d for d in self.declarations if d["day"] == day), None)
||||||| merged common ancestors
=======
    def declared(self, day: int) -> Declaration | None:
        """Day `day`'s declared circuit, the one declaration of the day in an accepted log; None when it declares none.
        A window of that day runs its program."""
        return next((d for d in self.declarations if d.day == day), None)
>>>>>>> 705be2b532

    def to_json(self) -> dict[str, Any]:
<<<<<<< 94ca7fd0d18
<<<<<<< HEAD
        return {"builder": self.builder, "declarations": list(self.declarations), "head": self.head,
||||||| merged common ancestors
        return {"builder": self.builder, "head": self.head, "site": self.site.to_json(), "theta": list(self.theta)}
=======
        return {"builder": self.builder, "declarations": [d.to_json() for d in self.declarations], "head": self.head,
>>>>>>> 705be2b532
||||||| e10cf2c8559
        return {"builder": self.builder, "declarations": [d.to_json() for d in self.declarations], "head": self.head,
=======
        return {"builder": self.builder, "declarations": [d.body() for d in self.declarations], "head": self.head,
>>>>>>> 7af4d24fca3e
                "site": self.site.to_json(), "theta": list(self.theta)}


# --- the core protocol ------------------------------------------------------------------------------------------------

def params_of(j: Any) -> tuple[str, Auditor]:
    """The auditor's parameters: `{"auditor": AUDITOR, "composite"}`, the auditor as `verity.core.site.Auditor.of_json`
    reads it."""
    ascii(j)
    _exactly(j, ["auditor", "composite"], "the params")
    return _one_of(j["composite"], S.COMPOSITES, "the params' composite"), Auditor.of_json(j["auditor"])


def params(composite: str, auditor: Auditor) -> dict[str, Any]:
    return {"auditor": auditor.to_json(), "composite": composite}


def check_log(p: tuple[str, Auditor], c: tuple[str, list[Entry]]) -> tuple:
    """`Core.Registration.checkLog`: `("accept", public, profile)` or `("reject", reason)`."""
    (composite, auditor), (theirs, log) = p, c
    if theirs != composite:
        return "reject", f"composite: the log is {theirs}'s, not {composite}'s"
    try:
        s = check(auditor, composite, log)
    except Refused as r:
        return "reject", str(r)
    return "accept", s.view().to_json(), {"entries": s.entries}
