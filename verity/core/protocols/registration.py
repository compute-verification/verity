"""The registration log in Python: `Core.Registration` (`interface/CoreProtocol/Registration.lean`), the type every
subprotocol and both composites read, entry for entry and refusal for refusal. The composition suite's difftest holds
this module to the Lean (`composition/tests/test_registration_difftest.py`).

A deployment's registration is one append-only log of entries, each naming the head of the log before it (`previous`):
the digest of the log's last entry, or its composite's genesis when the log is empty, so the head names the whole log.
An entry is one of two kinds:

* a site entry, the Site's own (`verity.core.site.Entry`, `verity/registration/site/v1`): the sheets, locations and
  links it adds to the Site, its JSON, digest and check the Site's;
* an information entry (`Information`, `verity/registration/information/v1`): its items, each registered once in the
  log (`item_key`): θ, each registered value by its committed root, and the builder B (its digest, its advice's format
  and its public cap ℓ on one declaration's advice, in bits). Registration registers the site, θ and B, not the circuit
  B builds; a subprotocol's Program and parameters, and the proof system, are that protocol's public verifier
  parameters.

~~~text
entry digest   SHA-512(format ‖ 0 ‖ canon(entry))                          the entry's format and previous among its keys
genesis        the Site's: SHA-512("verity/registration/genesis/v1" ‖ 0 ‖ canon({"composite": NAME}))
the log        {"composite": "onsite" | "remote", "entries": [ENTRY, ...]}   committed; its digest is its head
~~~

`check` holds each entry to the log before it and raises `Refused` at the first entry that fails, by the first check it
fails (`REFUSALS`): a site entry by the Site's (`verity.core.site.entry_refusal`), an information entry by `previous`
and then `duplicate-information`. A component reads its view of an accepted log (`View`): the Site, θ's values
(`View.value`) and the builder, at the head. Decoding takes exactly what the encoders write and raises `Malformed`
otherwise; parameters holding a character outside ASCII without DEL are `Malformed` as `ascii`, as the Lean's canonical
JSON refuses them.
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
PROTOCOL = "verity/registration/log/v1"
REFUSALS = (*S.REFUSALS, "duplicate-information")


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


Entry = S.Entry | Information
"""An entry of the registration log: the Site's site entry, or an information entry."""


def item_key(item: Mapping[str, Any]) -> str:
    """What an item registers, which the log registers once: each θ value by its name, and one builder."""
    return "theta " + item["value"] if item["kind"] == "theta" else item["kind"]


def site(sheets: Sequence[S.Sheet] = (), locs: Sequence[S.Loc] = (), links: Sequence[S.Link] = ()) -> S.Entry:
    """A site entry adding `sheets`, `locs` and `links`, its `previous` for `chain` to set."""
    return S.Entry("", tuple(sheets), tuple(locs), tuple(links))


def head(composite: str, log: Sequence[Entry]) -> str:
    """The log's head, which names it."""
    return log[-1].digest() if log else S.genesis(composite)


def chain(composite: str, entries: Sequence[Entry]) -> list[Entry]:
    """The log of `entries` in order, each following the head of the log before it."""
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


def ascii(j: Any, what: str) -> None:
    """ASCII without DEL, which the Lean's canonical JSON refuses anything outside of: `j`'s JSON, unescaped, holds no
    character from 0x7f up."""
    if any(ord(c) >= 0x7F for c in json.dumps(j, ensure_ascii=False)):
        raise Malformed(f"ascii: {what} holds a character outside ASCII")


def entry_of(j: Any) -> Entry:
    if not isinstance(j, dict) or "format" not in j:
        raise Malformed("an entry: format missing")
    fmt = _str(j["format"], "an entry's format")
    if fmt == S.SITE:
        return S.Entry.of_json(j)
    if fmt == INFORMATION:
        _exactly(j, ["format", "items", "previous"], "an information entry")
        return Information(tuple(item_of(i) for i in _list(j["items"], "items")), _str(j["previous"], "previous"))
    raise Malformed(f"an entry: format {fmt} is not one of [{S.SITE}, {INFORMATION}]")


def log_of(j: Any) -> tuple[str, list[Entry]]:
    """The committed log's composite and entries."""
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
    """The log so far, folded: its Site, its information items in order, its head and its length."""

    site: S.Site
    items: tuple[Mapping[str, Any], ...]
    head: str
    entries: int

    def refusal(self, auditor: Auditor, composite: str, e: Entry) -> str | None:
        if isinstance(e, S.Entry):
            return site_refusal(auditor, composite, self.site, self.head, e)
        if e.previous != self.head:
            return "previous"
        return None if _nodup([item_key(i) for i in self.items + e.items]) else "duplicate-information"

    def add(self, e: Entry) -> State:
        if isinstance(e, S.Entry):
            return replace(self, site=self.site.add(e), head=e.digest(), entries=self.entries + 1)
        return replace(self, items=self.items + e.items, head=e.digest(), entries=self.entries + 1)

    def view(self) -> View:
        return View(self.site, tuple(dict(i) for i in self.items if i["kind"] == "theta"),
                    next((dict(i) for i in self.items if i["kind"] == "builder"), None), self.head)


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
    """What a component reads of an accepted log, at its head: the Site, θ's registered values in order, the builder."""

    site: S.Site
    theta: tuple[dict[str, Any], ...]
    builder: dict[str, Any] | None
    head: str

    def value(self, name: str) -> dict[str, Any] | None:
        """θ's registered value `name`."""
        return next((t for t in self.theta if t["value"] == name), None)

    def to_json(self) -> dict[str, Any]:
        return {"builder": self.builder, "head": self.head, "site": self.site.to_json(), "theta": list(self.theta)}


# --- the core protocol ------------------------------------------------------------------------------------------------

def params_of(j: Any) -> tuple[str, Auditor]:
    """The auditor's parameters: `{"auditor": AUDITOR, "composite"}`, the auditor as `verity.core.site.Auditor.of_json`
    reads it."""
    ascii(j, "the params")
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
