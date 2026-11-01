"""The Site in code (`Core.Site`, `CoreProtocol/Site.lean`, served by `site-difftest`) held to its Python mirror,
`verity.core.site`: an entry's refusal against the log before it (`site_refusal`, `Registration.Entry.refusal`), a whole
log's check and fold (`site_check`, `Registration.check`), and the views of a log's site, unchecked (`site_views`), its
devices' models read against an auditor at a tick. Both sides take the digests that are deployed, so every head, digest
and refusal is compared.

The inputs are logs that start from each composite's first entry, the lifecycle example's site, followed by its 15 later
entries (`verity/core/site/tests/test_site.py`'s `ENTRIES`, copied here: every refusal and an accepted entry). Each entry
is changed at random a few times (a field of a location, link or sheet, an element dropped, repeated or added, the
head it follows), and checked against auditors whose sheets and tolerances vary. They also come as malformed JSON: a
node replaced or deleted, or a key added. Every string is ASCII, the domain of canonical JSON (`verity.core.wire`).
"""

from __future__ import annotations

import shutil
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest
from hypothesis import strategies as st

import verity_difftest as D
from verity.core.site import (COMPOSITES, KINDS, OPERAND_CLASSES, OUTSIDE, OWNERS, TICK, TIERS, Auditor, DeviceFigures,
                              Entry, Link, Loc, Malformed, Refused, Registration, Sheet, Site, _rat_json, check, genesis,
                              refusal)

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


def _view(s: Site, auditor: Auditor, tick: Fraction, i: str) -> dict:
    sheet, node = s.sheet(i), s.node(i)
    return {"boundary": [k.id for k in s.boundary(i)], "device": s.device(i, auditor, tick), "host": s.host(i),
            "node": None if node is None else node.to_json(), "registered": s.registered(i),
            "sheet": None if sheet is None else sheet.to_json(), "tiers": s.tiers(i)}


def ref_views(j):
    _input(j, ("auditor", "log", "tick"))
    auditor, log = Auditor.of_json(j["auditor"]), Registration.of_json(j["log"])
    tick, s = _rat_json(j["tick"], "the tick"), Site.of_log(log)
    return {"boundary": [k.id for k in s.boundary()],
            "ids": {i: _view(s, auditor, tick, i) for i in [*(x.id for x in s.locs), OUTSIDE, "nowhere"]},
            "units": list(s.units)}


# --- the lifecycle example's site and its 15 later entries -----------------------------------------------------------

GHZ = 10 ** 9
SHEET_A = Sheet(DeviceFigures("sim-int8-a", GHZ, 64 * GHZ), 3200, 1 << 25, 64 << 30)
SHEET_B = Sheet(DeviceFigures("sim-int8-b", GHZ, 128 * GHZ), 6400, 1 << 26, 128 << 30)
TOLERANCE = Fraction(1, 20)


def figures(sheet: Sheet, clock: Fraction | int | None = None, **rates: Fraction | int) -> Sheet:
    """`sheet` stating another clock, or other peak rates for the classes named (`uniform_s8=` for `uniform-s8`)."""
    f = sheet.figures
    peak = {c: rates.get(c.replace("-", "_"), f.rate(c)) for c in OPERAND_CLASSES}
    return replace(sheet, figures=replace(f, clock=f.clock if clock is None else clock, peak_rate=peak))


SHEET_Q = Sheet(DeviceFigures('sim-"q"\\\n\t', Fraction(1, 3), (Fraction(7, 2), 1, Fraction(2, 3), 5, Fraction(1, 7))),
                2, 2, 2 << 30)
"""A sheet whose SKU canonical JSON escapes and whose figures are fractions, which the auditor sometimes has."""
SHEET_S = figures(SHEET_A, clock=Fraction(21, 20) * GHZ, uniform_s8=Fraction(21, 20) * 64 * GHZ,
                  zero=Fraction(19, 20) * 64 * GHZ)
"""`SHEET_A`'s SKU with figures at the edges of the auditor's 5%: refused at a tolerance of 0, and a device on it models
the higher rate it states."""
A, B, Q, S = SHEET_A.digest(), SHEET_B.digest(), SHEET_Q.digest(), SHEET_S.digest()
AUDITOR = {"onsite": Auditor((SHEET_A, SHEET_B), TOLERANCE, (("cert0", "dummy-cert0-key"), ("chal0", ""))),
           "remote": Auditor((SHEET_A, SHEET_B), TOLERANCE)}
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
    "unknown-sku": ("sheet-digest", Entry("", sheets=(replace(SHEET_A, figures=replace(SHEET_A.figures,
                                                                                       sku="sim-int8-z")),))),
    "unregistered-sheet": ("sheet-digest", Entry("", locs=(Loc("gpu9", "device", "node0", sheet="00"),))),
    "doubled-clock": ("figures", Entry("", sheets=(figures(SHEET_A, clock=2 * GHZ),))),
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
        "nowhere", 'gpu"\n\t\\']
"""The ids a change draws from: the site's own, unregistered ones, the outside's, a link's, and one canonical JSON
escapes."""
LOC_FIELDS = {"id": st.sampled_from(POOL), "kind": st.sampled_from(KINDS), "parent": st.sampled_from([None, *POOL]),
              "owner": st.sampled_from(OWNERS), "sheet": st.sampled_from([A, B, Q, S, "00", ""]),
              "tier": st.sampled_from([None, *TIERS]),
              "bytes": st.sampled_from([0, 1, 64 << 30, (64 << 30) + 1, 1 << 40]),
              "key": st.sampled_from(["", "dummy-cert0-key", "00"])}
LINK_FIELDS = {"id": st.sampled_from([*POOL, "u9"]), "ends": st.tuples(st.sampled_from(POOL), st.sampled_from(POOL)),
               "beta": st.sampled_from([0, 1, 100_000]), "latency": st.sampled_from([0, 1])}
SHEETS = st.sampled_from([SHEET_Q, SHEET_S, figures(SHEET_A, zero=0), figures(SHEET_B, uniform_u4=2 * 128 * GHZ),
                          replace(SHEET_A, bw=1), replace(SHEET_B, figures=replace(SHEET_B.figures, sku="sim-int8-a")),
                          SHEET_A, SHEET_B])
"""The sheets a change or an auditor draws from: an escaped SKU's, figures at the 5% edges, a zero rate, a rate at a
tolerance of 1's edge, another memory figure, another SKU's figures under `SHEET_A`'s SKU, and the auditor's own."""
TOLERANCES = [Fraction(0), TOLERANCE, Fraction(1), Fraction(3, 2)]
"""None, the auditor's, one at which `figures(SHEET_B, uniform_u4=...)` is on the edge, and one past 1, which sets no lower
bound."""
TICKS = [str(TICK), str(TICK / 2), str(Fraction(3, 7) * TICK), "1", "0"]


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
    """Mostly the composite's auditor, mostly at its tolerance; at times another's, or one of other sheets, keys and
    tolerance, the first under an SKU or a device being the one the check reads."""
    if _mostly(draw, True, False):
        auditor = AUDITOR[_mostly(draw, composite, *COMPOSITES)]
        return replace(auditor, tolerance=_mostly(draw, auditor.tolerance, *TOLERANCES))
    sheets = draw(st.lists(SHEETS, max_size=4))
    keys = draw(st.lists(st.sampled_from([("cert0", "dummy-cert0-key"), ("cert0", "00"), ("chal0", ""), ("cert9", "00"),
                                          ('gpu"\n\t\\', "")]), max_size=4))
    return Auditor(tuple(sheets), draw(st.sampled_from(TOLERANCES)), tuple(keys))


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
    composite = draw(st.sampled_from(COMPOSITES))
    return {"auditor": draw(auditors(composite)).to_json(), "log": draw(logs(composite)).to_json(),
            "tick": _mostly(draw, *TICKS)}


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
    """Each of the 15 later entries draws the same refusal from both sides, and the log it ends is refused at it or
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
    j = {"auditor": AUDITOR["onsite"].to_json(), "log": log.to_json(), "tick": str(TICK)}
    views = ref_views(j)
    assert views["units"] == ["gpu0", "gpu1", "gpu2"] and views["boundary"] == ["uplink0"]
    assert views["ids"]["gpu0"]["device"] == {"bw": 3200, "cap": {}, "chip": 1 << 25, "mem": 8 * (64 << 30),
                                              "rate": {"cuda": 0, "oracle": 0, "tc": 64}}
    assert views["ids"]["node0"]["tiers"] == {"hbm": 320 << 30, "dram": 512 << 30, "disk": 2 << 40}
    assert views["ids"]["hbm0"]["host"] == "node0" and views["ids"]["cert0"]["host"] is None
    assert server.call("site_views", j) == ("output", D.canonical(views))
    j = {"auditor": AUDITOR["onsite"].to_json(), "log": log.to_json()}
    assert ref_check(j) == {"head": log.head, "site": Site.of_log(log).to_json()}
    assert server.call("site_check", j) == ("output", D.canonical(ref_check(j)))
    empty = {"auditor": AUDITOR["onsite"].to_json(), "log": Registration("onsite").to_json()}
    assert server.call("site_check", empty) == ("output", D.canonical({"head": GENESIS_ONSITE,
                                                                        "site": {"links": [], "locs": [], "sheets": []}}))


def test_a_device_models_its_bounding_figures(server):
    """A sheet whose figures sit at the edges of the auditor's 5% is accepted at that tolerance and refused at none, and a
    device on it models the higher rate the sheet states, 67.2 MACs a nanosecond rounded up, on both sides."""
    log = first("onsite")
    log = log.append(Entry(log.head, (SHEET_S,), (Loc("gpu9", "device", "node0", sheet=S),)))
    for tolerance, refused in ((TOLERANCE, False), (Fraction(0), True)):
        j = {"auditor": replace(AUDITOR["onsite"], tolerance=tolerance).to_json(), "log": log.to_json()}
        assert ("refused" in ref_check(j)) == refused
        assert server.call("site_check", j) == ("output", D.canonical(ref_check(j)))
    j = {"auditor": AUDITOR["onsite"].to_json(), "log": log.to_json(), "tick": str(TICK)}
    views = ref_views(j)
    assert views["ids"]["gpu9"]["device"]["rate"]["tc"] == 68 and views["ids"]["gpu0"]["device"]["rate"]["tc"] == 64
    assert server.call("site_views", j) == ("output", D.canonical(views))
