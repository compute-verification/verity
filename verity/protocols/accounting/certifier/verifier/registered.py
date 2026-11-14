"""The network certifier's link and key, read from the registered hardware: the Python mirror of
`NetworkCheck.Registered` (`lean/NetworkCheck/Registered.lean`), which defines it.

`registered(site, head, link, key)` reads a link's name and a certifier's key off the site of an accepted registration
log's view: the link joins a network certifier and a host, and the key is the one that certifier registered. It returns
what the registration states of them (`Registered`): the link's id, its host end, the certifier's id and registered
key, the link's capacity β in bits a second each way, and the head of the log the site was read at, which it carries
and doesn't read. `Refused`, with the Lean's message, for a link that is no link of the site, a link that doesn't join a
network certifier and a host (the certifier's other half, to the outside; a host's bare uplink; a link between two
hosts; a memory challenger's link), a certifier whose registered key is not lower-case hex, a key that is not the
certifier's registered key, and a network certifier's link that states no capacity, with or without a reason no
certifier splits it: there is no line rate to read for it.

The registered key of a network certifier is the string the checks' `key` parameter is: its LMS public key, I ‖ T[1],
in lower-case hex. `key_refusal` is the one function that reads it.

`site_of` reads the site as the view writes it, a plain mapping `{"links", "locs", "sheets"}` with each location and
link as the Site's own JSON has them, and keeps what the reading reads: no sheet and no figure. It imports nothing of the
Site's Python, and nothing here holds the site to an accepted log: its caller does. `Malformed` for a mapping that is
not such a site, where the two sides agree on the refusal and not on its words.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

OUTSIDE = "outside"
KINDS = ("site", "host", "device", "memory", "certifier", "challenger", "auditor-host", "outside")
OWNERS = ("prover", "auditor")
TIERS = ("hbm", "dram", "disk")
LOC_KEYS = ("bw", "bytes", "chip", "drift", "id", "key", "kind", "owner", "parent", "scheme", "shared", "sheet", "tier",
            "time_unit", "timebase")
SHARED_KEYS = ("bw", "tier")
LINK_KEYS = ("beta", "ends", "id", "latency", "unmediated")
SITE_KEYS = ("links", "locs", "sheets")
_LOWER_HEX = re.compile(r"(?:[0-9a-f]{2})+")
_NATURAL = re.compile(r"0|[1-9][0-9]*")


class Refused(ValueError):
    """The reading refused: its message is the Lean's."""


class Malformed(Refused):
    """The site is not one a registration log's view writes."""


@dataclass(frozen=True, slots=True)
class Registered:
    """What the registered hardware states of a network certifier's link."""

    head: str
    link: str
    host: str
    certifier: str
    key: str
    beta: int  # bits a second each way


def _exactly(j: Any, keys: tuple[str, ...], what: str) -> None:
    if not isinstance(j, Mapping) or sorted(j) != list(keys):
        raise Malformed(f"{what}: the keys must be [{', '.join(keys)}]")


def _str(x: Any, what: str) -> str:
    if type(x) is not str:
        raise Malformed(f"{what}: not a string")
    return x


def _nat(x: Any, what: str) -> int:
    if type(x) is not int or x < 0:
        raise Malformed(f"{what}: not a natural number")
    return x


def _list(x: Any, what: str) -> Sequence[Any]:
    if type(x) not in (list, tuple):
        raise Malformed(f"{what}: not a list")
    return x


def _named(x: Any, names: tuple[str, ...], what: str) -> str:
    if _str(x, what) not in names:
        raise Malformed(f"{what}: {x} is not one of [{', '.join(names)}]")
    return x


def _figure(x: Any, what: str) -> str:
    """A figure as the Site writes one, and only so: `n`, or `n/d` in lowest terms with d at least 2, each a natural
    number in ASCII digits with no leading zero."""
    parts = _str(x, what).split("/")
    if len(parts) > 2 or not all(_NATURAL.fullmatch(part) for part in parts):
        raise Malformed(f"{what}: {x} is not a ratio n/d of natural numbers in canonical form")
    if len(parts) == 2 and (int(parts[1]) <= 1 or math.gcd(int(parts[0]), int(parts[1])) != 1):
        raise Malformed(f"{what}: {x} has denominator 0 or 1, or is not in lowest terms")
    return x


def _or_none(x: Any, read: Callable[[Any, str], Any], what: str) -> Any:
    return None if x is None else read(x, what)


def _loc(j: Any) -> dict[str, str]:
    _exactly(j, LOC_KEYS, "a location")
    _named(j["owner"], OWNERS, "a location's owner")
    _str(j["sheet"], "a location.sheet")
    _nat(j["bytes"], "a location.bytes")
    _str(j["scheme"], "a location.scheme")
    _or_none(j["parent"], _str, "a location's parent")
    _or_none(j["bw"], _nat, "a location's bw")
    _or_none(j["chip"], _nat, "a location's chip")
    _or_none(j["drift"], _figure, "a location's drift")
    _or_none(j["time_unit"], _figure, "a location's time unit")
    _or_none(j["timebase"], _str, "a location's time base")
    if j["tier"] is not None:
        _named(j["tier"], TIERS, "a location's tier")
    for shared in _list(j["shared"], "a location.shared"):
        _exactly(shared, SHARED_KEYS, "a shared rate")
        _named(shared["tier"], TIERS, "a shared rate's tier")
        _nat(shared["bw"], "a shared rate.bw")
    return {"id": _str(j["id"], "a location.id"), "kind": _named(j["kind"], KINDS, "a location's kind"),
            "key": _str(j["key"], "a location.key")}


def _link(j: Any) -> dict[str, Any]:
    _exactly(j, LINK_KEYS, "a link")
    ends = _list(j["ends"], "a link.ends")
    if len(ends) != 2:
        raise Malformed("a link: ends: not two ends")
    _nat(j["latency"], "a link.latency")
    _or_none(j["unmediated"], _str, "a link's unmediated")
    return {"id": _str(j["id"], "a link.id"), "ends": (_str(ends[0], "a link's end"), _str(ends[1], "a link's end")),
            "beta": _or_none(j["beta"], _nat, "a link.beta")}


def site_of(site: Any) -> dict[str, tuple[dict[str, Any], ...]]:
    """The site of a registration log's view, as the Site's JSON writes it: its locations (`id`, `kind`, `key`) and its
    links (`id`, `ends`, and `beta`, None where the link states no capacity), each in the site's order. Every other
    field of a location and a link is held to the Site's form and dropped. Its sheets are not read."""
    _exactly(site, SITE_KEYS, "the site")
    _list(site["sheets"], "the site.sheets")
    locs = tuple(_loc(x) for x in _list(site["locs"], "the site.locs"))
    links = tuple(_link(x) for x in _list(site["links"], "the site.links"))
    return {"locs": locs, "links": links}


def _loc_of(site: Mapping[str, Any], i: str) -> Mapping[str, Any] | None:
    return next((x for x in site["locs"] if x["id"] == i), None)


def _kind(site: Mapping[str, Any], i: str) -> str | None:
    if i == OUTSIDE:
        return OUTSIDE
    x = _loc_of(site, i)
    return None if x is None else x["kind"]


def _joins_certifier_and_host(site: Mapping[str, Any], k: Mapping[str, Any]) -> bool:
    return sorted((_kind(site, k["ends"][0]) or "", _kind(site, k["ends"][1]) or "")) == ["certifier", "host"]


def certifier_links(site: Mapping[str, Any]) -> list[str]:
    """The network certifiers' links of the site, by id, in its order: those that join a network certifier and a host."""
    return [k["id"] for k in site["links"] if _joins_certifier_and_host(site, k)]


def _right_links(site: Mapping[str, Any]) -> str:
    return f"the network certifiers' links there are [{', '.join(certifier_links(site))}]"


def _end(site: Mapping[str, Any], i: str) -> str:
    if i == OUTSIDE:
        return "the outside"
    kind = _kind(site, i)
    return f"{i}, which is not registered" if kind is None else f"{kind} {i}"


def lower_hex(s: str) -> bool:
    """`s` is lower-case hex: an even, positive number of the digits 0-9 and a-f."""
    return _LOWER_HEX.fullmatch(s) is not None


def key_refusal(certifier: Mapping[str, Any], key: str) -> str | None:
    """Why `key` is not the key of the network certifier, or None when it is: the certifier's registered key is its LMS
    public key in lower-case hex, and `key` is that string. The one reading of a certifier's registered key."""
    c, registered_key = certifier["id"], certifier["key"]
    if not lower_hex(registered_key):
        return (f"network certifier {c} registers the key {registered_key}, which is not lower-case hex; a network "
                "certifier registers its LMS public key in lower-case hex")
    if registered_key != key:
        return f"the key {key} is not network certifier {c}'s registered key, {registered_key}"
    return None


def registered(site: Mapping[str, Any], head: str, link: str, key: str) -> Registered:
    """The network certifier's link and key, as registered: `link` names a link of `site` (a `site_of`) that joins a
    network certifier and a host, and `key` is that certifier's registered key. `head`, the head of the log the site
    was read at, is carried unread. `Refused` otherwise, naming the network certifiers' links of the site or the
    certifier's registered key, and last for a link that states no capacity."""
    k = next((x for x in site["links"] if x["id"] == link), None)
    if k is None:
        raise Refused(f"link {link}: no link of the registered hardware; {_right_links(site)}")
    a, b = k["ends"]
    if not _joins_certifier_and_host(site, k):
        raise Refused(f"link {link}: it joins {_end(site, a)} and {_end(site, b)}, not a network certifier and a host; "
                      f"{_right_links(site)}")
    c, host = (a, b) if _kind(site, a) == "certifier" else (b, a)
    x = _loc_of(site, c)
    if x is None:
        raise Refused(f"link {link}: network certifier {c} is not registered")
    why = key_refusal(x, key)
    if why is not None:
        raise Refused(f"link {link}: {why}")
    if k["beta"] is None:
        raise Refused(f"link {link}: it states no capacity; a network certifier's link states its line rate")
    return Registered(head=head, link=k["id"], host=host, certifier=c, key=x["key"], beta=k["beta"])
