"""The Site in code (`Core.Site`, `CoreProtocol/Site.lean`, served by `site-difftest`) held to its Python mirror,
`verity.hardware.site`: an entry's refusal against the log before it (`site_refusal`, `Registration.Entry.refusal`), a whole
log's check and fold (`site_check`, `Registration.check`), and the views of a log's site, unchecked (`site_views`), its
devices' models read against an auditor at the site's unit of time. Both sides take the digests that are deployed, so
every head, digest and refusal is compared.

The inputs are logs that start from each composite's first entry, the lifecycle example's site, followed by up to two of
its later entries (`verity/hardware/site/tests/test_site.py`'s `ENTRIES`, copied here: every refusal an entry gets on its
own and an accepted entry). Each entry is changed at random a few times (a field of a location, link or sheet, an
element dropped, repeated or added, the head it follows), and checked against auditors whose sheets, tolerances, keys
and figures for hosts and memory vary. They also come as malformed JSON: a node replaced or deleted, or a key added.
"""

from __future__ import annotations

import shutil
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest
from hypothesis import strategies as st

import verity_difftest as D
from verity.hardware.site import (COMPOSITES, KINDS, OPERAND_CLASSES, OUTSIDE, OWNERS, SCHEMES, TIERS, Auditor,
                                  DeviceFigures, Entry, HostFigures, Link, Loc, Malformed, MemoryFigures, Refused,
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


def _view(s: Site, auditor: Auditor, i: str) -> dict:
    sheet, node = s.sheet(i), s.node(i)
    return {"boundary": [k.id for k in s.boundary(i)], "device": s.device(i, auditor), "host": s.host(i),
            "node": None if node is None else node.to_json(), "registered": s.registered(i),
            "sheet": None if sheet is None else sheet.to_json(), "tiers": s.tiers(i)}


def ref_views(j):
    _input(j, ("auditor", "log"))
    auditor, log = Auditor.of_json(j["auditor"]), Registration.of_json(j["log"])
    s = Site.of_log(log)
    return {"boundary": [k.id for k in s.boundary()],
            "ids": {i: _view(s, auditor, i) for i in [*(x.id for x in s.locs), OUTSIDE, "nowhere"]},
            "units": list(s.units)}


# --- the lifecycle example's site and its later entries ------------------------------------------------------------

GHZ = 10 ** 9
SHEET_A = Sheet(DeviceFigures("sim-int8-a", GHZ, 64 * GHZ), 3200 * GHZ, 1 << 25, 64 << 30)
SHEET_B = Sheet(DeviceFigures("sim-int8-b", GHZ, 128 * GHZ), 6400 * GHZ, 1 << 26, 128 << 30)
TOLERANCE = Fraction(1, 20)
SCHEME = SCHEMES[0]
NS = Fraction(1, 10 ** 9)
HOSTS = {"node0": HostFigures(1 << 23, (("sim-int8-a", 1), ("sim-int8-b", 1)), (("disk", 50),), (("uplink0", 100_000),)),
         "node1": HostFigures(1 << 23), "node9": HostFigures(1 << 23)}
MEMORY = {"dram0": MemoryFigures("node0", 6554, 512 << 30), "nvme0": MemoryFigures("node0", 28, 1 << 40),
          "nvme1": MemoryFigures("node0", 28, 1 << 40), "dram1": MemoryFigures("node1", 6554, 1)}


def host(i: str) -> Loc:
    return Loc(i, "host", "site0", chip=HOSTS[i].chip, shared=HOSTS[i].shared)


def leaf(i: str, tier: str) -> Loc:
    f = MEMORY[i]
    return Loc(i, "memory", f.host, tier=tier, bytes=f.bytes, bw=f.bw)


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
AUDITOR = {"onsite": Auditor((SHEET_A, SHEET_B), TOLERANCE,
                             (("cert0", "dummy-cert0-key", SCHEME), ("chal0", "", SCHEME),
                              ("vm0", "dummy-vm0-key", SCHEME), ("chal9", "", SCHEME)),
                             tuple(MEMORY.items()), tuple(HOSTS.items())),
           "remote": Auditor((SHEET_A, SHEET_B), TOLERANCE, (), tuple(MEMORY.items()), tuple(HOSTS.items()))}
SITE0 = Loc("site0", "site", time_unit=NS)
NODE = Entry("", (SHEET_A, SHEET_B), (
    SITE0, host("node0"),
    Loc("gpu0", "device", "node0", sheet=A), Loc("hbm0", "memory", "gpu0", tier="hbm", bytes=64 << 30),
    Loc("gpu1", "device", "node0", sheet=B), Loc("hbm1", "memory", "gpu1", tier="hbm", bytes=128 << 30),
    leaf("dram0", "dram"), leaf("nvme0", "disk")))
FIRST = {
    "onsite": replace(NODE, locs=(*NODE.locs,
                                  Loc("cert0", "certifier", "site0", "auditor", key="dummy-cert0-key", scheme=SCHEME),
                                  Loc("chal0", "challenger", "site0", "auditor", scheme=SCHEME)),
                      links=(Link("uplink0", ("node0", "cert0"), 100_000, 100_000),
                             Link("uplink0-out", ("cert0", OUTSIDE), 100_000, 0),
                             Link("chal0-link", ("chal0", "node0"), 100_000, 1_000_000))),
    "remote": replace(NODE, links=(Link("uplink0", ("node0", OUTSIDE), 100_000, 100_000),)),
}
LATER = Entry("", locs=(Loc("gpu2", "device", "node0", sheet=B), Loc("hbm2", "memory", "gpu2", tier="hbm", bytes=128 << 30),
                        leaf("nvme1", "disk")))
UPLINK9 = (Link("u9", ("node0", "cert9"), 1, 1), Link("u9-out", ("cert9", OUTSIDE), 1, 0))
ENTRIES = {
    "accepted": (None, LATER),
    "stale": ("previous", LATER),
    "gpu0-again": ("duplicate-id", Entry("", locs=(Loc("gpu0", "device", "node0", sheet=A),))),
    "no-parent": ("unknown-ref", Entry("", locs=(Loc("hbm9", "memory", "gpu9", tier="hbm", bytes=1),))),
    "device-at-site": ("kind", Entry("", locs=(Loc("gpu9", "device", "site0", sheet=A),))),
    "loop-link": ("kind", Entry("", links=(Link("l9", ("node0", "node0"), 1, 1),))),
    "unrated-link": ("kind", Entry("", links=(Link("u9", ("node0", OUTSIDE), None, 1),))),
    "unknown-sku": ("sheet-digest", Entry("", sheets=(replace(SHEET_A, figures=replace(SHEET_A.figures,
                                                                                       sku="sim-int8-z")),))),
    "unregistered-sheet": ("sheet-digest", Entry("", locs=(Loc("gpu9", "device", "node0", sheet="00"),))),
    "doubled-clock": ("figures", Entry("", sheets=(figures(SHEET_A, clock=2 * GHZ),))),
    "unrated-leaf": ("figures", Entry("", locs=(replace(leaf("nvme1", "disk"), bw=None),))),
    "second-clock": ("figures", Entry("", locs=(Loc("site1", "site", time_unit=2 * NS),))),
    "chipless-host": ("figures", Entry("", locs=(replace(host("node9"), chip=None),))),
    "over-hbm": ("capacity", Entry("", locs=(Loc("gpu9", "device", "node0", sheet=A),
                                             Loc("hbm9", "memory", "gpu9", tier="hbm", bytes=(64 << 30) + 1)))),
    "empty-memory": ("capacity", Entry("", locs=(Loc("hbm9", "memory", "gpu0", tier="hbm"),))),
    "foreign-certifier": ("owner", Entry("", locs=(Loc("cert9", "certifier", "site0", "auditor", key="00"),),
                                         links=UPLINK9)),
    "auditor-owned-host": ("owner", Entry("", locs=(replace(host("node9"), owner="auditor"),))),
    "keyed-host": ("owner", Entry("", locs=(replace(host("node9"), scheme=SCHEME),))),
    "bare-uplink": ("uncertified", Entry("", links=(Link("u9", ("node0", OUTSIDE), 1, 1),))),
    "unreached-host": ("challenger-reach", Entry("", locs=(host("node1"), leaf("dram1", "dram")),
                                                 links=(Link("n01", ("node0", "node1"), 1, 1),))),
}
REMOTE_ACCEPTS = {"bare-uplink", "unreached-host"}
VM0 = Loc("vm0", "auditor-host", "site0", "auditor", key="dummy-vm0-key", scheme=SCHEME)
HOUSED = Entry("", locs=(VM0, Loc("chal9", "challenger", "vm0", "auditor", scheme=SCHEME)))
"""An auditor's host at the site with a second memory challenger in it, which the onsite auditor holds the keys of."""
LATERS = [*(e for _, e in ENTRIES.values()), HOUSED]


def first(composite: str, unit: Fraction = NS) -> Registration:
    log = Registration(composite)
    body = FIRST[composite]
    body = replace(body, locs=(replace(SITE0, time_unit=unit), *body.locs[1:]))
    return log.append(replace(body, previous=log.head))


def grown(unit: Fraction = NS) -> Registration:
    """The onsite site with the later hardware: complete against the auditor's figures, which list the later disk."""
    log = first("onsite", unit)
    return log.append(replace(LATER, previous=log.head))


def case(composite: str, name: str) -> tuple[Registration, Entry]:
    log = first(composite)
    return log, replace(ENTRIES[name][1], previous=genesis(composite) if name == "stale" else log.head)


GENESIS_ONSITE = ("d6356b427985ba6dab9a8deb46447a8108cdd9ca3fafa720f5b51af748999cc2"
                  "9c5829bbfcef4105fc8407c9eee2943b6e8fb07feff9645a73e0a24fe5a379c0")


# --- strategies ------------------------------------------------------------------------------------------------------

POOL = ["site0", "node0", "node1", "gpu0", "gpu1", "gpu9", "hbm0", "dram0", "cert0", "cert9", "chal0", "uplink0", OUTSIDE,
        "nowhere", 'gpu"\n\t\\', "vm0"]
"""The ids a change draws from: the site's own, unregistered ones, the outside's, a link's, and one canonical JSON
escapes."""
LOC_FIELDS = {"id": st.sampled_from(POOL), "kind": st.sampled_from(KINDS), "parent": st.sampled_from([None, *POOL]),
              "owner": st.sampled_from(OWNERS), "sheet": st.sampled_from([A, B, Q, S, "00", ""]),
              "tier": st.sampled_from([None, *TIERS]),
              "bytes": st.sampled_from([0, 1, 64 << 30, (64 << 30) + 1, 1 << 40]),
              "key": st.sampled_from(["", "dummy-cert0-key", "dummy-vm0-key", "00"]),
              "scheme": st.sampled_from(["", SCHEME, "x"]),
              "bw": st.sampled_from([None, 0, 28, 6554]), "chip": st.sampled_from([None, 0, 1 << 23]),
              "shared": st.sampled_from([(), (("disk", 50),), (("disk", 50), ("disk", 60))]),
              "drift": st.sampled_from([None, Fraction(0), Fraction(1, 10 ** 6), Fraction(1)]),
              "time_unit": st.sampled_from([None, Fraction(0), NS, 2 * NS])}
LINK_FIELDS = {"id": st.sampled_from([*POOL, "u9"]), "ends": st.tuples(st.sampled_from(POOL), st.sampled_from(POOL)),
               "beta": st.sampled_from([None, 0, 1, 100_000]), "latency": st.sampled_from([0, 1]),
               "unmediated": st.sampled_from([None, "", "no certifier splits it"])}
SHEETS = st.sampled_from([SHEET_Q, SHEET_S, figures(SHEET_A, zero=0), figures(SHEET_B, uniform_u4=2 * 128 * GHZ),
                          replace(SHEET_A, bw=1), replace(SHEET_B, figures=replace(SHEET_B.figures, sku="sim-int8-a")),
                          SHEET_A, SHEET_B])
"""The sheets a change or an auditor draws from: an escaped SKU's, figures at the 5% edges, a zero rate, a rate at a
tolerance of 1's edge, another memory figure, another SKU's figures under `SHEET_A`'s SKU, and the auditor's own."""
TOLERANCES = [Fraction(0), TOLERANCE, Fraction(1), Fraction(3, 2)]
"""None, the auditor's, one at which `figures(SHEET_B, uniform_u4=...)` is on the edge, and one past 1, which sets no lower
bound."""
KEYS = [("cert0", "dummy-cert0-key", SCHEME), ("cert0", "00", SCHEME), ("cert0", "dummy-cert0-key", "x"),
        ("chal0", "", SCHEME), ("cert9", "00", SCHEME), ('gpu"\n\t\\', "", SCHEME),
        ("vm0", "dummy-vm0-key", SCHEME), ("vm0", "00", SCHEME), ("chal9", "", SCHEME)]
MEMORIES = [*MEMORY.items(), ("nvme1", MemoryFigures("node0", 0, 1 << 40)), ("dram9", MemoryFigures("node9", 1, 1))]
HOST_FIGURES = [*HOSTS.items(), ("node0", HostFigures(1 << 25)), ("node0", HostFigures(1 << 23, ways=(("uplink0", None),))),
                ("node1", HostFigures(1 << 23, (("sim-int8-a", 1),)))]
DRIFTS = [("chal0", Fraction(1, 10 ** 6)), ("cert0", Fraction(1)), ("gpu0", Fraction(0))]


def _mostly(draw, usual, *others):
    """`usual` three times in four or more: Hypothesis samples an index near 0 more often than uniformly."""
    return draw(st.sampled_from([usual, usual, usual, *others]))


@st.composite
def changed(draw, e: Entry) -> Entry:
    """`e` with one change: a field of one of its locations, links or sheets sampled anew, an element dropped or repeated,
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
    keys, memory, hosts, drift = (draw(st.lists(st.sampled_from(xs), max_size=4))
                                  for xs in (KEYS, MEMORIES, HOST_FIGURES, DRIFTS))
    return Auditor(tuple(sheets), draw(st.sampled_from(TOLERANCES)), tuple(keys), tuple(memory), tuple(hosts),
                   tuple(drift))


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
    return {"auditor": draw(auditors(composite)).to_json(), "log": draw(logs(composite)).to_json()}


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
    """Each later entry samples the same refusal from both sides, and the log it ends is refused at it, refused by the
    inventory at its end (the first entry's site lacks the later disk the auditor lists) or accepted alike."""
    log, e = case(composite, name)
    want = None if composite == "remote" and name in REMOTE_ACCEPTS else ENTRIES[name][0]
    j = {"auditor": AUDITOR[composite].to_json(), "entry": e.to_json(), "log": log.to_json()}
    out = {"digest": e.digest(), "head": log.head, "refusal": want}
    assert ref_refusal(j) == out
    assert server.call("site_refusal", j) == ("output", D.canonical(out))
    j = {"auditor": AUDITOR[composite].to_json(), "log": log.append(e).to_json()}
    assert server.call("site_check", j) == ("output", D.canonical(ref_check(j)))
    refused = (want, 1) if want else (None, None) if name == "accepted" else ("inventory", 2)
    assert (ref_check(j).get("refused") or {"name": None, "entry": None}) == dict(zip(("name", "entry"), refused))


def test_the_sample_site(server):
    """The grown onsite site's views and its check agree, and Lean's genesis is the pinned one."""
    log = grown()
    j = {"auditor": AUDITOR["onsite"].to_json(), "log": log.to_json()}
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


def test_an_auditors_host(server):
    """An auditor's host with a challenger in it is taken with the host's key, and refused by `owner` without it and by
    `kind` under a host of the prover's, on both sides; in the views the host and the device in it are in no host of
    the prover's."""
    log = grown()
    housed = replace(HOUSED, previous=log.head)
    keyless = replace(AUDITOR["onsite"], keys=AUDITOR["onsite"].keys[:2])
    under = replace(housed, locs=(replace(VM0, parent="node0"), housed.locs[1]))
    for auditor, e, want in ((AUDITOR["onsite"], housed, None), (keyless, housed, "owner"),
                             (AUDITOR["onsite"], under, "kind")):
        j = {"auditor": auditor.to_json(), "entry": e.to_json(), "log": log.to_json()}
        assert ref_refusal(j)["refusal"] == want
        assert server.call("site_refusal", j) == ("output", D.canonical(ref_refusal(j)))
    log = log.append(housed)
    j = {"auditor": AUDITOR["onsite"].to_json(), "log": log.to_json()}
    assert ref_check(j) == {"head": log.head, "site": Site.of_log(log).to_json()}
    assert server.call("site_check", j) == ("output", D.canonical(ref_check(j)))
    views = ref_views(j)
    assert views["ids"]["vm0"]["host"] is None and views["ids"]["chal9"]["host"] is None
    assert server.call("site_views", j) == ("output", D.canonical(views))


def test_a_device_models_its_bounding_figures(server):
    """A sheet whose figures sit at the edges of the auditor's 5% is accepted at that tolerance and refused at none, and a
    device on it models the higher rate the sheet states, 67.2 MACs a nanosecond rounded up, on both sides."""
    log = grown()
    log = log.append(Entry(log.head, (SHEET_S,), (Loc("gpu9", "device", "node0", sheet=S),)))
    for tolerance, refused in ((TOLERANCE, False), (Fraction(0), True)):
        j = {"auditor": replace(AUDITOR["onsite"], tolerance=tolerance).to_json(), "log": log.to_json()}
        assert ("refused" in ref_check(j)) == refused
        assert server.call("site_check", j) == ("output", D.canonical(ref_check(j)))
    j = {"auditor": AUDITOR["onsite"].to_json(), "log": log.to_json()}
    views = ref_views(j)
    assert views["ids"]["gpu9"]["device"]["rate"]["tc"] == 68 and views["ids"]["gpu0"]["device"]["rate"]["tc"] == 64
    assert server.call("site_views", j) == ("output", D.canonical(views))


def test_a_rational_of_4301_digits(server):
    """A clock, a tolerance and a site's unit of time whose numerator and denominator have 4,301 digits each, one more
    than Python's `int` and `str` convert by default: both sides read them, write the sheet stating one under the same
    digest, accept it at that tolerance, and round a device's rate up over that unit."""
    n = 10 ** 4300 + 1
    over = "1" + "0" * 4299 + "2/1" + "0" * 4299 + "1"
    sheet = figures(SHEET_A, clock=Fraction(n + 1, n))
    log = grown(Fraction(n + 1, n))
    log = log.append(Entry(log.head, (sheet,), (Loc("gpu9", "device", "node0", sheet=sheet.digest()),)))
    j = {"auditor": replace(AUDITOR["onsite"], tolerance=Fraction(n + 1, n)).to_json(), "log": log.to_json()}
    assert j["auditor"]["tolerance"] == over == sheet.to_json()["figures"]["clock"] == log.to_json()["entries"][0][
        "locs"][0]["time_unit"]
    assert ref_check(j) == {"head": log.head, "site": Site.of_log(log).to_json()}
    assert server.call("site_check", j) == ("output", D.canonical(ref_check(j)))
    views = ref_views(j)
    assert views["ids"]["gpu9"]["device"]["rate"]["tc"] == 64 * GHZ + 1 and views["ids"]["gpu9"]["sheet"] == sheet.to_json()
    assert server.call("site_views", j) == ("output", D.canonical(views))
