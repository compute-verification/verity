"""The Site in code (`Core.Site`, `CoreProtocol/Site.lean`, served by `site-difftest`) held to its Python mirror,
`verity.core.site`: an entry's refusal against the log before it (`site_refusal`, `Registration.Entry.refusal`), a whole
log's check and fold (`site_check`, `Registration.check`), and the views of a log's site, unchecked (`site_views`). Both
sides take the digests that are deployed, so every head, digest and refusal is compared.

The inputs are logs that start from each composite's first entry, the lifecycle example's site, followed by its 14 later
entries (`verity/core/site/tests/test_site.py`'s `ENTRIES`, copied here: every refusal and an accepted entry). Each entry
is changed at random a few times (a field of a location, link or sheet, an element dropped, repeated or added, the
head it follows). They also come as malformed JSON: a node replaced or deleted, or a key added.
"""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import strategies as st

import verity_difftest as D
from verity.core.site import (COMPOSITES, KINDS, OUTSIDE, OWNERS, TIERS, Auditor, Entry, Link, Loc, Malformed, Refused,
                              Registration, Sheet, Site, check, genesis, refusal)

PACKAGE = Path(__file__).resolve().parents[1]
EXE = "site-difftest"
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")


def _input(j, keys: tuple[str, ...]) -> dict:
    if not isinstance(j, dict) or set(j) != set(keys):
        raise D.Reject(f"the input is {sorted(keys)}")
    return j


def ref_refusal(j):
    _input(j, ("auditor", "entry", "log"))
    auditor, log, e = Auditor.of_json(j["auditor"]), Registration.of_json(j["log"]), Entry.of_json(j["entry"])
    return {"digest": e.digest(), "head": log.head, "refusal": refusal(log, e, auditor)}


def ref_check(j):
    _input(j, ("auditor", "log"))
    auditor, log = Auditor.of_json(j["auditor"]), Registration.of_json(j["log"])
    try:
        site = check(log, auditor)
    except Refused as r:
        return {"refused": {"entry": r.entry, "name": r.name}}
    return {"head": log.head, "site": site.to_json()}


def _view(s: Site, i: str) -> dict:
    sheet, node = s.sheet(i), s.node(i)
    return {"boundary": [k.id for k in s.boundary(i)], "device": s.device(i), "host": s.host(i),
            "node": None if node is None else node.to_json(), "registered": s.registered(i),
            "sheet": None if sheet is None else sheet.to_json(), "tiers": s.tiers(i)}


def ref_views(j):
    _input(j, ("log",))
    s = Site.of_log(Registration.of_json(j["log"]))
    return {"boundary": [k.id for k in s.boundary()],
            "ids": {i: _view(s, i) for i in [*(x.id for x in s.locs), OUTSIDE, "nowhere"]}, "units": list(s.units)}


# --- the lifecycle example's site and its 14 later entries -----------------------------------------------------------

SHEET_A = Sheet("sim-int8-a", 64, 3200, 1 << 25, 64 << 30)
SHEET_B = Sheet("sim-int8-b", 128, 6400, 1 << 26, 128 << 30)
SHEET_Q = Sheet('sim-"q"\\\n\té', 2, 2, 2, 2 << 30)
"""A sheet whose SKU canonical JSON escapes, which the auditor sometimes has."""
A, B, Q = SHEET_A.digest(), SHEET_B.digest(), SHEET_Q.digest()
AUDITOR = {"onsite": Auditor((SHEET_A, SHEET_B), (("cert0", "dummy-cert0-key"), ("chal0", ""))),
           "remote": Auditor((SHEET_A, SHEET_B))}
NODE = Entry("", (SHEET_A, SHEET_B), (
    Loc("site0", "site"), Loc("node0", "host", "site0"),
    Loc("gpu0", "device", "node0", sheet=A), Loc("hbm0", "memory", "gpu0", tier="hbm", bytes=64 << 30),
    Loc("gpu1", "device", "node0", sheet=B), Loc("hbm1", "memory", "gpu1", tier="hbm", bytes=128 << 30),
    Loc("dram0", "memory", "node0", tier="dram", bytes=512 << 30),
    Loc("nvme0", "memory", "node0", tier="disk", bytes=1 << 40)))
FIRST = {
    "onsite": replace(NODE, locs=(*NODE.locs, Loc("cert0", "certifier", "site0", "auditor", key="dummy-cert0-key"),
                                  Loc("chal0", "challenger", "site0", "auditor")),
                      links=(Link("uplink0", ("node0", "cert0"), 100_000, 100_000),
                             Link("uplink0-out", ("cert0", OUTSIDE), 100_000, 0),
                             Link("chal0-link", ("chal0", "node0"), 100_000, 1_000_000))),
    "remote": replace(NODE, links=(Link("uplink0", ("node0", OUTSIDE), 100_000, 100_000),)),
}
LATER = Entry("", locs=(Loc("gpu2", "device", "node0", sheet=B), Loc("hbm2", "memory", "gpu2", tier="hbm", bytes=128 << 30),
                        Loc("nvme1", "memory", "node0", tier="disk", bytes=1 << 40)))
UPLINK9 = (Link("u9", ("node0", "cert9"), 1, 1), Link("u9-out", ("cert9", OUTSIDE), 1, 0))
ENTRIES = {
    "accepted": (None, LATER),
    "stale": ("previous", LATER),
    "gpu0-again": ("duplicate-id", Entry("", locs=(Loc("gpu0", "device", "node0", sheet=A),))),
    "no-parent": ("unknown-ref", Entry("", locs=(Loc("hbm9", "memory", "gpu9", tier="hbm", bytes=1),))),
    "device-at-site": ("kind", Entry("", locs=(Loc("gpu9", "device", "site0", sheet=A),))),
    "loop-link": ("kind", Entry("", links=(Link("l9", ("node0", "node0"), 1, 1),))),
    "not-the-sheet": ("sheet-digest", Entry("", sheets=(replace(SHEET_A, tc=1 << 20),))),
    "unregistered-sheet": ("sheet-digest", Entry("", locs=(Loc("gpu9", "device", "node0", sheet="00"),))),
    "over-hbm": ("capacity", Entry("", locs=(Loc("gpu9", "device", "node0", sheet=A),
                                             Loc("hbm9", "memory", "gpu9", tier="hbm", bytes=(64 << 30) + 1)))),
    "empty-memory": ("capacity", Entry("", locs=(Loc("dram9", "memory", "node0", tier="dram"),))),
    "foreign-certifier": ("owner", Entry("", locs=(Loc("cert9", "certifier", "site0", "auditor", key="00"),),
                                         links=UPLINK9)),
    "auditor-host": ("owner", Entry("", locs=(Loc("node9", "host", "site0", "auditor"),))),
    "bare-uplink": ("uncertified", Entry("", links=(Link("u9", ("node0", OUTSIDE), 1, 1),))),
    "unreached-host": ("challenger-reach", Entry("", locs=(Loc("node1", "host", "site0"),
                                                           Loc("dram1", "memory", "node1", tier="dram", bytes=1)),
                                                 links=(Link("n01", ("node0", "node1"), 1, 1),))),
}
REMOTE_ACCEPTS = {"bare-uplink", "unreached-host"}
LATERS = [e for _, e in ENTRIES.values()]


def first(composite: str) -> Registration:
    log = Registration(composite)
    return log.append(replace(FIRST[composite], previous=log.head))


def case(composite: str, name: str) -> tuple[Registration, Entry]:
    log = first(composite)
    return log, replace(ENTRIES[name][1], previous=genesis(composite) if name == "stale" else log.head)


GENESIS_ONSITE = ("d6356b427985ba6dab9a8deb46447a8108cdd9ca3fafa720f5b51af748999cc2"
                  "9c5829bbfcef4105fc8407c9eee2943b6e8fb07feff9645a73e0a24fe5a379c0")


# --- strategies ------------------------------------------------------------------------------------------------------

POOL = ["site0", "node0", "node1", "gpu0", "gpu1", "gpu9", "hbm0", "dram0", "cert0", "cert9", "chal0", "uplink0", OUTSIDE,
        "nowhere", 'gpu"\n\té']
"""The ids a change draws from: the site's own, unregistered ones, the outside's, a link's, and one canonical JSON
escapes."""
LOC_FIELDS = {"id": st.sampled_from(POOL), "kind": st.sampled_from(KINDS), "parent": st.sampled_from([None, *POOL]),
              "owner": st.sampled_from(OWNERS), "sheet": st.sampled_from([A, B, Q, "00", ""]),
              "tier": st.sampled_from([None, *TIERS]),
              "bytes": st.sampled_from([0, 1, 64 << 30, (64 << 30) + 1, 1 << 40]),
              "key": st.sampled_from(["", "dummy-cert0-key", "00"])}
LINK_FIELDS = {"id": st.sampled_from([*POOL, "u9"]), "ends": st.tuples(st.sampled_from(POOL), st.sampled_from(POOL)),
               "beta": st.sampled_from([0, 1, 100_000]), "latency": st.sampled_from([0, 1])}
SHEETS = st.sampled_from([SHEET_Q, replace(SHEET_A, tc=1), replace(SHEET_B, sku="sim-int8-a"), SHEET_A, SHEET_B])


def _mostly(draw, usual, *others):
    """`usual` three times in four or more: Hypothesis draws an index near 0 more often than uniformly."""
    return draw(st.sampled_from([usual, usual, usual, *others]))


@st.composite
def changed(draw, e: Entry) -> Entry:
    """`e` with one change: a field of one of its locations, links or sheets drawn anew, an element dropped or repeated,
    or a location, link or sheet added."""
    parts = {"locs": list(e.locs), "links": list(e.links), "sheets": list(e.sheets)}
    what = draw(st.sampled_from(["field", "field", "drop", "repeat", "add"]))
    part = draw(st.sampled_from(sorted(parts)))
    xs = parts[part]
    if what == "add" or not xs:
        fields = {"locs": LOC_FIELDS, "links": LINK_FIELDS}.get(part)
        xs.append(draw(SHEETS) if fields is None else (Loc if part == "locs" else Link)(
            **{f: draw(s) for f, s in fields.items()} | {"id": draw(st.sampled_from(["new0", "new1", *POOL]))}))
    else:
        n = draw(st.integers(0, len(xs) - 1))
        if what == "drop":
            xs.pop(n)
        elif what == "repeat":
            xs.insert(draw(st.integers(0, len(xs))), xs[n])
        elif part == "sheets":
            xs[n] = draw(SHEETS)
        else:
            fields = LOC_FIELDS if part == "locs" else LINK_FIELDS
            f = draw(st.sampled_from(sorted(fields)))
            xs[n] = replace(xs[n], **{f: draw(fields[f])})
    return replace(e, **{k: tuple(v) for k, v in parts.items()})


@st.composite
def bodies(draw, e: Entry) -> Entry:
    for _ in range(draw(st.sampled_from([0, 0, 1, 1, 2, 3]))):
        e = draw(changed(e))
    return e


def _follows(draw, log: Registration, e: Entry) -> Entry:
    """`e` following the log's head, but at times another."""
    return replace(e, previous=_mostly(draw, log.head, log.head, log.head, "00", genesis(log.composite), genesis("remote")))


@st.composite
def logs(draw, composite: str, laters: list[Entry] | None = None) -> Registration:
    """A log of a first entry (mostly its composite's) and `laters`, or up to two of the later entries, each changed a
    few times or not."""
    log = Registration(composite)
    start = FIRST[_mostly(draw, composite, *COMPOSITES)]
    if laters is None:
        laters = draw(st.permutations(LATERS))[:draw(st.integers(0, 2))]
    for body in [start, *laters]:
        log = log.append(_follows(draw, log, draw(bodies(body))))
    return log


@st.composite
def auditors(draw, composite: str) -> Auditor:
    """Mostly the composite's auditor; at times another's, or one of other sheets and keys, the first under an SKU or
    a device being the one the check reads."""
    if _mostly(draw, True, False):
        return AUDITOR[_mostly(draw, composite, *COMPOSITES)]
    sheets = draw(st.lists(SHEETS, max_size=4))
    keys = draw(st.lists(st.sampled_from([("cert0", "dummy-cert0-key"), ("cert0", "00"), ("chal0", ""), ("cert9", "00"),
                                          ('gpu"\n\té', "")]), max_size=4))
    return Auditor(tuple(sheets), tuple(keys))


@st.composite
def refusal_inputs(draw) -> dict:
    composite = draw(st.sampled_from(COMPOSITES))
    laters = draw(st.permutations(LATERS))
    n = draw(st.integers(0, 1))
    log = draw(logs(composite, laters[:n]))
    e = _follows(draw, log, draw(bodies(laters[n])))
    return {"auditor": draw(auditors(composite)).to_json(), "entry": e.to_json(), "log": log.to_json()}


@st.composite
def check_inputs(draw) -> dict:
    composite = draw(st.sampled_from(COMPOSITES))
    return {"auditor": draw(auditors(composite)).to_json(), "log": draw(logs(composite)).to_json()}


@st.composite
def views_inputs(draw) -> dict:
    return {"log": draw(logs(draw(st.sampled_from(COMPOSITES)))).to_json()}


JSON = st.recursive(st.none() | st.booleans() | st.integers(-3, 3) | st.text("ab/1", max_size=3),
                    lambda inner: st.lists(inner, max_size=3) | st.dictionaries(st.text("abk", max_size=2), inner, max_size=3),
                    max_leaves=6)


def _nodes(j, out: list) -> list:
    """Every (container, key) of `j`'s nested objects and lists."""
    for k in (j if isinstance(j, dict) else range(len(j)) if isinstance(j, list) else ()):
        out.append((j, k))
        _nodes(j[k], out)
    return out


@st.composite
def broken(draw, inputs) -> dict:
    """An input of `inputs` with one node replaced by arbitrary JSON or deleted, or with a key added to an object."""
    j = draw(inputs)
    container, k = draw(st.sampled_from(_nodes(j, [])))
    how = draw(st.sampled_from(["replace", "replace", "delete", "extra"]))
    if how == "replace":
        container[k] = draw(JSON)
    elif how == "delete":
        del container[k]
    else:
        (container[k] if isinstance(container[k], dict) else j)["extra"] = draw(JSON)
    return j


def _inputs(valid):
    return st.one_of(*[valid() for _ in range(4)], broken(valid()), JSON)


REJECTS = (D.Reject, Malformed)
REFUSAL = D.Port(EXE, "site_refusal", ref_refusal, _inputs(refusal_inputs), package=PACKAGE, rejects=REJECTS)
CHECK = D.Port(EXE, "site_check", ref_check, _inputs(check_inputs), package=PACKAGE, rejects=REJECTS)
VIEWS = D.Port(EXE, "site_views", ref_views, _inputs(views_inputs), package=PACKAGE, rejects=REJECTS)

server = D.server_fixture(REFUSAL)


@pytest.mark.parametrize("port", [REFUSAL, CHECK, VIEWS], ids=lambda p: p.fn)
def test_the_site_agrees(server, port):
    stats = D.run(port, examples=600, server=server)
    assert stats.outputs > 300 and stats.rejects > 0, stats


@pytest.mark.parametrize("composite", COMPOSITES)
@pytest.mark.parametrize("name", ENTRIES)
def test_the_lifecycle_cases(server, composite: str, name: str):
    """Each of the 14 later entries draws the same refusal from both sides, and the log it ends is refused at it or
    accepted alike."""
    log, e = case(composite, name)
    want = None if composite == "remote" and name in REMOTE_ACCEPTS else ENTRIES[name][0]
    j = {"auditor": AUDITOR[composite].to_json(), "entry": e.to_json(), "log": log.to_json()}
    out = {"digest": e.digest(), "head": log.head, "refusal": want}
    assert ref_refusal(j) == out
    assert server.call("site_refusal", j) == ("output", D.canonical(out))
    j = {"auditor": AUDITOR[composite].to_json(), "log": log.append(e).to_json()}
    assert server.call("site_check", j) == ("output", D.canonical(ref_check(j)))
    assert ("refused" in ref_check(j)) == (want is not None)


def test_the_sample_site(server):
    """The grown onsite site's views and its check agree, and Lean's genesis is the pinned one."""
    log = first("onsite")
    log = log.append(replace(LATER, previous=log.head))
    j = {"log": log.to_json()}
    views = ref_views(j)
    assert views["units"] == ["gpu0", "gpu1", "gpu2"] and views["boundary"] == ["uplink0"]
    assert views["ids"]["gpu0"]["device"] == {"tc": 64, "bw": 3200, "chip": 1 << 25, "mem": 64 << 30}
    assert views["ids"]["node0"]["tiers"] == {"hbm": 320 << 30, "dram": 512 << 30, "disk": 2 << 40}
    assert views["ids"]["hbm0"]["host"] == "node0" and views["ids"]["cert0"]["host"] is None
    assert server.call("site_views", j) == ("output", D.canonical(views))
    j = {"auditor": AUDITOR["onsite"].to_json(), "log": log.to_json()}
    assert ref_check(j) == {"head": log.head, "site": Site.of_log(log).to_json()}
    assert server.call("site_check", j) == ("output", D.canonical(ref_check(j)))
    empty = {"auditor": AUDITOR["onsite"].to_json(), "log": Registration("onsite").to_json()}
    assert server.call("site_check", empty) == ("output", D.canonical({"head": GENESIS_ONSITE,
                                                                        "site": {"links": [], "locs": [], "sheets": []}}))
