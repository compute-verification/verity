"""The Site: each check `check_registration` makes, on a later entry of each composite's site's log (the lifecycle
example's cases), the order the checks are made in, the fold and its prefixes, the views, a sheet read from a card file,
and the log's encoding."""

from __future__ import annotations

from dataclasses import replace

import pytest

from verity.core.site import (OUTSIDE, REFUSALS, Auditor, Entry, Link, Loc, Malformed, Refused, Registration, Sheet, Site,
                              check, check_registration, genesis, refusal)

SHEET_A = Sheet("sim-int8-a", 64, 3200, 1 << 25, 64 << 30)
SHEET_B = Sheet("sim-int8-b", 128, 6400, 1 << 26, 128 << 30)
A, B = SHEET_A.digest(), SHEET_B.digest()
AUDITOR = {"onsite": Auditor((SHEET_A, SHEET_B), (("cert0", "dummy-cert0-key"), ("chal0", ""))),
           "remote": Auditor((SHEET_A, SHEET_B))}
"""The auditor's data sheets for the simulated SKUs and its own devices' keys: onsite a certifier and a challenger."""
NODE = Entry("", (SHEET_A, SHEET_B), (
    Loc("site0", "site"), Loc("node0", "host", "site0"),
    Loc("gpu0", "device", "node0", sheet=A), Loc("hbm0", "memory", "gpu0", tier="hbm", bytes=64 << 30),
    Loc("gpu1", "device", "node0", sheet=B), Loc("hbm1", "memory", "gpu1", tier="hbm", bytes=128 << 30),
    Loc("dram0", "memory", "node0", tier="dram", bytes=512 << 30),
    Loc("nvme0", "memory", "node0", tier="disk", bytes=1 << 40)))
"""What both composites' sites register first: one node with two devices, each with its HBM, and the host's DRAM and
disk."""
FIRST = {
    "onsite": replace(NODE, locs=(*NODE.locs, Loc("cert0", "certifier", "site0", "auditor", key="dummy-cert0-key"),
                                  Loc("chal0", "challenger", "site0", "auditor")),
                      links=(Link("uplink0", ("node0", "cert0"), 100_000, 100_000),
                             Link("uplink0-out", ("cert0", OUTSIDE), 100_000, 0),
                             Link("chal0-link", ("chal0", "node0"), 100_000, 1_000_000))),
    "remote": replace(NODE, links=(Link("uplink0", ("node0", OUTSIDE), 100_000, 100_000),)),
}
"""Each composite's first entry: onsite, the uplink split by the network certifier and the memory challenger's link to
the node; remote, a bare uplink."""
LATER = Entry("", locs=(Loc("gpu2", "device", "node0", sheet=B), Loc("hbm2", "memory", "gpu2", tier="hbm", bytes=128 << 30),
                        Loc("nvme1", "memory", "node0", tier="disk", bytes=1 << 40)))
"""The hardware a site adds later: a third device and a disk."""
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
"""A later entry for each check `check_registration` makes, with the refusal it draws, and one it accepts: the lifecycle
example's 14."""
REMOTE_ACCEPTS = {"bare-uplink", "unreached-host"}
"""The entries a remote site accepts: it has no certifier and no memory challenger to hold its links to."""


def first(composite: str) -> Registration:
    log = Registration(composite)
    return check_registration(log, replace(FIRST[composite], previous=log.head), AUDITOR[composite])


def grown(composite: str = "onsite") -> Registration:
    log = first(composite)
    return check_registration(log, replace(LATER, previous=log.head), AUDITOR[composite])


def test_every_check_has_a_case() -> None:
    assert len(ENTRIES) == 14 and {name for name, _ in ENTRIES.values()} - {None} == set(REFUSALS)


@pytest.mark.parametrize("composite", ["onsite", "remote"])
@pytest.mark.parametrize("case", ENTRIES)
def test_check_registration(composite: str, case: str) -> None:
    """Each check, on a later entry of each composite's site's log."""
    log = first(composite)
    name, entry = ENTRIES[case]
    entry = replace(entry, previous=genesis(composite) if case == "stale" else log.head)
    if name is None or (composite == "remote" and case in REMOTE_ACCEPTS):
        assert refusal(log, entry, AUDITOR[composite]) is None
        assert check_registration(log, entry, AUDITOR[composite]).head == entry.digest()
        return
    assert refusal(log, entry, AUDITOR[composite]) == name
    with pytest.raises(Refused) as e:
        check_registration(log, entry, AUDITOR[composite])
    assert (e.value.name, e.value.entry) == (name, 1), e.value


@pytest.mark.parametrize(("entry", "name"), [
    (Entry("00", locs=(Loc("gpu0", "device", "node0", sheet=A),)), "previous"),
    (Entry("", locs=(Loc("gpu9", "device", "nowhere", sheet="00"),)), "unknown-ref"),
    (Entry("", locs=(Loc("gpu9", "device", "site0", sheet="00"),)), "kind"),
    (Entry("", locs=(Loc("dram9", "memory", "node0", "auditor", tier="dram"),)), "capacity"),
])
def test_the_first_check_failed_names_the_refusal(entry: Entry, name: str) -> None:
    """An entry failing several checks is refused by the earliest of them in `REFUSALS`."""
    log = first("onsite")
    entry = entry if entry.previous else replace(entry, previous=log.head)
    assert refusal(log, entry, AUDITOR["onsite"]) == name


def test_the_first_entry_is_checked_too() -> None:
    """A site whose first entry can't stand alone is refused at entry 0: onsite, its uplink must be split."""
    log = Registration("onsite")
    assert refusal(log, replace(FIRST["remote"], previous=log.head), AUDITOR["onsite"]) == "uncertified"
    assert refusal(log, replace(FIRST["onsite"], previous=""), AUDITOR["onsite"]) == "previous"


def test_check_folds_a_valid_log() -> None:
    log = grown()
    assert check(log, AUDITOR["onsite"]) == Site.of_log(log)
    bad = log.prefix(1).append(replace(ENTRIES["gpu0-again"][1], previous=log.prefix(1).head))
    with pytest.raises(Refused) as e:
        check(bad, AUDITOR["onsite"])
    assert (e.value.name, e.value.entry) == ("duplicate-id", 1)


def test_the_site_only_grows_along_the_log() -> None:
    log = grown()
    before, after = Site.of_log(log.prefix(1)), Site.of_log(log)
    assert after.locs[:len(before.locs)] == before.locs and after.links[:len(before.links)] == before.links
    assert Site.of_log(log.prefix(0)) == Site() and Site.of_log(log.entries) == after
    assert log.entries[1].previous == log.prefix(1).head != log.head


def test_views() -> None:
    """What a protocol reads of the grown onsite site: its devices, their sheets and the device model's view of each,
    a host as registered, its memory by tier, its boundary, and what is registered."""
    s = Site.of_log(grown())
    assert s.units == ("gpu0", "gpu1", "gpu2")
    assert s.sheet("gpu0") == SHEET_A and s.sheet("gpu2") == SHEET_B
    assert s.sheet("node0") is None and s.sheet("hbm0") is None and s.sheet("gpu9") is None
    assert s.device("gpu0") == {"tc": 64, "bw": 3200, "chip": 1 << 25, "mem": 64 << 30} and s.device("node0") is None
    node = s.node("node0")
    assert [x.id for x in node.locs] == ["node0", "gpu0", "hbm0", "gpu1", "hbm1", "dram0", "nvme0", "gpu2", "hbm2", "nvme1"]
    assert [k.id for k in node.links] == ["uplink0", "chal0-link"] and node.sheets == (SHEET_A, SHEET_B)
    assert s.node("site0") is None and s.node("gpu0") is None
    assert s.tiers("node0") == {"hbm": 320 << 30, "dram": 512 << 30, "disk": 2 << 40}
    assert s.tiers("gpu1") == {"hbm": 128 << 30, "dram": 0, "disk": 0}
    assert [k.id for k in s.boundary("node0")] == [k.id for k in s.boundary()] == ["uplink0"]
    assert all(s.certified(k) for k in s.boundary())
    assert s.registered("gpu2") and s.registered("chal0-link") and not s.registered(OUTSIDE) and not s.registered("gpu9")
    assert (s.host("hbm0"), s.host("dram0"), s.host("gpu1"), s.host("node0"), s.host("cert0")) == (
        "node0", "node0", "node0", "node0", None)


def test_remote_boundary_is_uncertified() -> None:
    s = Site.of_log(first("remote"))
    assert [k.id for k in s.boundary("node0")] == ["uplink0"] and not s.certified(s.boundary()[0])


def _entry(v: object) -> dict[str, object]:
    return {"value": v, "unit": "-", "mark": "ceiling", "source": "a test's card"}


CARD_FILE = {"sku": "sim-file", "sm": {"count": _entry(2)}, "clock": {"sm": _entry(1530)},
             "rate": {"tensor": _entry(1024)}, "memory": {"bandwidth": _entry(1.23456e12)},
             "store": {"reg": _entry(65536), "reg_extra": _entry(0), "smem": _entry(262144), "l2": _entry(1 << 20),
                       "dram": _entry(16 << 30)}}
"""A card file's table as `verity.hardware.load(sku)` returns it, with the entries `Sheet.of_file` reads."""


def test_a_sheet_from_its_card_file() -> None:
    """At a nanosecond tick: 2 SMs x 1024 MACs a clock x 1.53 GHz is 3133.44 MACs, rounded up; 1.23456e12 B/s is
    9876.48 bits, rounded up; on-chip state is the registers', shared memory's and L2's bits."""
    sheet = Sheet.of_file(CARD_FILE)
    assert sheet == Sheet("sim-file", 3134, 9877, 32 * 2 * 65536 + 8 * (2 * 262144 + (1 << 20)), 16 << 30)
    auditor = Auditor.of_files([CARD_FILE])
    assert auditor.sheet("sim-file") == sheet and auditor.sheet("sim-int8-a") is None
    log = Registration("remote")
    log = check_registration(log, log.entry([sheet], [Loc("site0", "site"), Loc("node0", "host", "site0"),
                                                      Loc("gpu0", "device", "node0", sheet=sheet.digest()),
                                                      Loc("hbm0", "memory", "gpu0", tier="hbm", bytes=16 << 30)]), auditor)
    assert Site.of_log(log).device("gpu0") == {"tc": 3134, "bw": 9877, "chip": sheet.chip, "mem": 16 << 30}
    over = log.entry(locs=[Loc("hbm1", "memory", "gpu0", tier="hbm", bytes=1)])
    assert refusal(log, over, auditor) == "capacity"


def test_the_encoding_round_trips_and_is_pinned() -> None:
    log = grown()
    assert Registration.of_json(log.to_json()) == log and Auditor.of_json(AUDITOR["onsite"].to_json()) == AUDITOR["onsite"]
    assert genesis("onsite") == GENESIS_ONSITE
    assert genesis("remote") != genesis("onsite") and log.head == log.entries[-1].digest()


GENESIS_ONSITE = ("d6356b427985ba6dab9a8deb46447a8108cdd9ca3fafa720f5b51af748999cc2"
                  "9c5829bbfcef4105fc8407c9eee2943b6e8fb07feff9645a73e0a24fe5a379c0")
"""`SHA-512("verity/registration/genesis/v1" ‖ 0 ‖ {"composite":"onsite"})`, the registration log's own."""


@pytest.mark.parametrize(("make", "what"), [
    (lambda: Loc("x", "rack"), "kind"),
    (lambda: Loc("x", "memory", "node0", tier="sram", bytes=1), "tier"),
    (lambda: Loc("x", "host", "site0", "developer"), "owner"),
    (lambda: Loc("x", "memory", "node0", tier="dram", bytes=-1), "bytes"),
    (lambda: Link("l", ("a", "b", "c"), 1, 1), "pair"),
    (lambda: Link("l", ("a", "b"), 1, -1), "latency"),
    (lambda: Registration("hybrid"), "composite"),
    (lambda: Entry.of_json({**LATER.to_json(), "format": "verity/registration/information/v1"}), "format"),
    (lambda: Registration.of_json({"composite": "onsite", "entries": [], "extra": 1}), "keys"),
])
def test_what_no_site_holds_is_malformed(make, what: str) -> None:
    with pytest.raises(Malformed, match=what):
        make()
