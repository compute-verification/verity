"""The Site: each check `check_registration` makes, on a later entry of each composite's site's log (the lifecycle
example's cases, and the figures check it doesn't make yet), the order the checks are made in, a sheet's figures against
the auditor's estimate, the fold and its prefixes, the views, the device model read off the bounding figures, a sheet read
from a card file, and the log's encoding."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest

from verity.core.site import (OPERAND_CLASSES, OUTSIDE, REFUSALS, Auditor, DeviceFigures, Entry, Link, Loc, Malformed,
                              Refused, Registration, Sheet, Site, check, check_registration, genesis, refusal)
from verity.core.wire import canonical

GHZ = 10 ** 9
SHEET_A = Sheet(DeviceFigures("sim-int8-a", GHZ, 64 * GHZ), 3200, 1 << 25, 64 << 30)
SHEET_B = Sheet(DeviceFigures("sim-int8-b", GHZ, 128 * GHZ), 6400, 1 << 26, 128 << 30)
A, B = SHEET_A.digest(), SHEET_B.digest()
TOLERANCE = Fraction(1, 20)
AUDITOR = {"onsite": Auditor((SHEET_A, SHEET_B), TOLERANCE, (("cert0", "dummy-cert0-key"), ("chal0", ""))),
           "remote": Auditor((SHEET_A, SHEET_B), TOLERANCE)}
"""The auditor's data sheets for the simulated SKUs, 1 GHz clocks at 64 and 128 MACs a nanosecond in every operand class,
its 5% tolerance around them, and its own devices' keys: onsite a certifier and a challenger. The node registers the
auditor's own sheets."""


def figures(sheet: Sheet, clock: Fraction | int | None = None, **rates: Fraction | int) -> Sheet:
    """`sheet` stating another clock, or other peak rates for the classes named (`uniform_s8=` for `uniform-s8`)."""
    f = sheet.figures
    peak = {c: rates.get(c.replace("-", "_"), f.rate(c)) for c in OPERAND_CLASSES}
    return replace(sheet, figures=replace(f, clock=f.clock if clock is None else clock, peak_rate=peak))
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
"""A later entry for each check `check_registration` makes, with the refusal it draws, and one it accepts: the lifecycle
example's 14, its `not-the-sheet` now a sheet of an SKU the auditor has no data sheet for, and `doubled-clock`, the
figures check the example doesn't make yet."""
REMOTE_ACCEPTS = {"bare-uplink", "unreached-host"}
"""The entries a remote site accepts: it has no certifier and no memory challenger to hold its links to."""


def first(composite: str) -> Registration:
    log = Registration(composite)
    return check_registration(log, replace(FIRST[composite], previous=log.head), AUDITOR[composite])


def grown(composite: str = "onsite") -> Registration:
    log = first(composite)
    return check_registration(log, replace(LATER, previous=log.head), AUDITOR[composite])


def test_every_check_has_a_case() -> None:
    assert len(ENTRIES) == 15 and {name for name, _ in ENTRIES.values()} - {None} == set(REFUSALS)


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
    (Entry("", sheets=(ENTRIES["unknown-sku"][1].sheets[0], figures(SHEET_A, clock=2 * GHZ))), "sheet-digest"),
    (Entry("", sheets=(figures(SHEET_A, clock=2 * GHZ),), locs=(Loc("dram9", "memory", "node0", tier="dram"),)),
     "figures"),
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
    assert s.device("gpu0", AUDITOR["onsite"]) == {"bw": 3200, "cap": {}, "chip": 1 << 25, "mem": 8 * (64 << 30),
                                                   "rate": {"cuda": 0, "oracle": 0, "tc": 64}}
    assert s.device("node0", AUDITOR["onsite"]) is None and s.device("gpu0", Auditor((SHEET_B,))) is None
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


@pytest.mark.parametrize(("sheet", "accepted"), [
    (figures(SHEET_A, clock=Fraction(95, 100) * GHZ), True),
    (figures(SHEET_A, clock=Fraction(105, 100) * GHZ), True),
    (figures(SHEET_A, clock=Fraction(94, 100) * GHZ), False),
    (figures(SHEET_A, clock=Fraction(106, 100) * GHZ), False),
    (figures(SHEET_A, zero=Fraction(105, 100) * 64 * GHZ, uniform_s2=Fraction(95, 100) * 64 * GHZ), True),
    (figures(SHEET_A, uniform_u4=Fraction(106, 100) * 64 * GHZ), False),
    (figures(SHEET_A, zero=0), False),
    (replace(SHEET_A, bw=3201), False),
    (replace(SHEET_A, mem=(64 << 30) - 1), False),
])
def test_a_sheets_figures_against_the_estimate(sheet: Sheet, accepted: bool) -> None:
    """At the auditor's 5%: the clock and each class's peak rate are accepted within 5% of its estimate's either way,
    and refused past it, less capacity as well as more; a rate of zero is refused; and the memory figures are exactly the
    auditor's."""
    log = first("remote")
    assert sheet.figures.accepted(AUDITOR["remote"].estimate(SHEET_A)) is (accepted or sheet.figures == SHEET_A.figures)
    assert refusal(log, log.entry([sheet]), AUDITOR["remote"]) == (None if accepted else "figures")


def test_a_zero_rate_is_refused_at_any_tolerance() -> None:
    """A tolerance of 1 or more sets no lower bound, so only the rule that every peak rate is positive refuses a zero."""
    loose = Auditor((SHEET_A,), 1)
    assert figures(SHEET_A, clock=0, uniform_s8=1).figures.accepted(loose.estimate(SHEET_A))
    assert not figures(SHEET_A, uniform_s8=0).figures.accepted(loose.estimate(SHEET_A))


@pytest.mark.parametrize(("rates", "tc", "half"), [
    ({"uniform_s8": Fraction(96, 100) * 64 * GHZ}, 64, 32),
    ({"uniform_s8": Fraction(104, 100) * 64 * GHZ}, 67, 34),
    ({"uniform_s8": 61 * GHZ, "zero": Fraction(1001, 1000) * 64 * GHZ}, 65, 33),
])
def test_the_device_model_reads_the_bounding_figures(rates: dict[str, Fraction], tc: int, half: int) -> None:
    """A device's tensor-core MACs a tick are the highest of the larger rates, class by class, of its sheet's and the
    auditor's estimate's, over the tick and rounded up (`half`: at half a nanosecond): stating less than the estimate
    reads the estimate, stating more reads what is stated, and the highest class counts. The rest of the model is the
    sheet's memory figures and the parameters."""
    log = first("remote")
    sheet = figures(SHEET_A, **rates)
    log = check_registration(log, log.entry([sheet], [Loc("gpu9", "device", "node0", sheet=sheet.digest())]),
                             AUDITOR["remote"])
    s = Site.of_log(log)
    assert s.device("gpu9", AUDITOR["remote"], cuda=7, oracle=1, cap={"pcie": 9}) == {
        "bw": 3200, "cap": {"pcie": 9}, "chip": 1 << 25, "mem": 8 * (64 << 30), "rate": {"cuda": 7, "oracle": 1, "tc": tc}}
    assert s.device("gpu9", AUDITOR["remote"], tick=Fraction(1, 2 * GHZ))["rate"]["tc"] == half


def _entry(v: object) -> dict[str, object]:
    return {"value": v, "unit": "-", "mark": "ceiling", "source": "a test's card"}


CARD_FILE = {"sku": "sim-file", "sm": {"count": _entry(2)}, "clock": {"sm": _entry(1530)},
             "rate": {"tensor": _entry(1024)}, "memory": {"bandwidth": _entry(1.23456e12)},
             "store": {"reg": _entry(65536), "reg_extra": _entry(0), "smem": _entry(262144), "l2": _entry(1 << 20),
                       "dram": _entry(16 << 30)}}
"""A card file's table as `verity.hardware.load(sku)` returns it, with the entries `Sheet.of_file` reads."""


def test_a_sheet_from_its_card_file() -> None:
    """The clock is 1.53 GHz, and every class's peak rate 2 SMs x 1024 MACs a clock at it. At a nanosecond tick that is
    3133.44 MACs, rounded up; 1.23456e12 B/s is 9876.48 bits, rounded up; on-chip state is the registers', shared
    memory's and L2's bits."""
    sheet = Sheet.of_file(CARD_FILE)
    assert sheet == Sheet(DeviceFigures("sim-file", 1530 * 10 ** 6, 2 * 1024 * 1530 * 10 ** 6), 9877,
                          32 * 2 * 65536 + 8 * (2 * 262144 + (1 << 20)), 16 << 30)
    auditor = Auditor.of_files([CARD_FILE], TOLERANCE)
    assert auditor.sheet("sim-file") == sheet and auditor.sheet("sim-int8-a") is None
    log = Registration("remote")
    log = check_registration(log, log.entry([sheet], [Loc("site0", "site"), Loc("node0", "host", "site0"),
                                                      Loc("gpu0", "device", "node0", sheet=sheet.digest()),
                                                      Loc("hbm0", "memory", "gpu0", tier="hbm", bytes=16 << 30)]), auditor)
    assert Site.of_log(log).device("gpu0", auditor) == {"bw": 9877, "cap": {}, "chip": sheet.chip, "mem": 8 * (16 << 30),
                                                        "rate": {"cuda": 0, "oracle": 0, "tc": 3134}}
    over = log.entry(locs=[Loc("hbm1", "memory", "gpu0", tier="hbm", bytes=1)])
    assert refusal(log, over, auditor) == "capacity"


def test_the_encoding_round_trips_and_is_pinned() -> None:
    """A rational is the string `n` or `n/d` in lowest terms, so a sheet has one encoding and one digest."""
    log = grown()
    assert Registration.of_json(log.to_json()) == log and Auditor.of_json(AUDITOR["onsite"].to_json()) == AUDITOR["onsite"]
    assert genesis("onsite") == GENESIS_ONSITE
    assert genesis("remote") != genesis("onsite") and log.head == log.entries[-1].digest()
    half = figures(SHEET_A, clock=Fraction(1, 2), zero=Fraction(64 * GHZ, 3))
    assert canonical(half.to_json()) == (
        b'{"bw":3200,"chip":33554432,"figures":{"clock":"1/2","peak_rate":{"uniform-s2":"64000000000",'
        b'"uniform-s8":"64000000000","uniform-u2":"64000000000","uniform-u4":"64000000000","zero":"64000000000/3"},'
        b'"sku":"sim-int8-a"},"mem":68719476736}')
    assert Sheet.of_json(half.to_json()) == half and AUDITOR["onsite"].to_json()["tolerance"] == "1/20"


GENESIS_ONSITE = ("d6356b427985ba6dab9a8deb46447a8108cdd9ca3fafa720f5b51af748999cc2"
                  "9c5829bbfcef4105fc8407c9eee2943b6e8fb07feff9645a73e0a24fe5a379c0")
"""`SHA-512("verity/registration/genesis/v1" ‖ 0 ‖ {"composite":"onsite"})`, the registration log's own."""


def _sheet_json(**f: object) -> Sheet:
    j = SHEET_A.to_json()
    return Sheet.of_json({**j, "figures": {**j["figures"], **f}})


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
    (lambda: DeviceFigures("x", -1, 1), "rational"),
    (lambda: DeviceFigures("x", 1, 0.5), "rational"),
    (lambda: DeviceFigures("x", 1, (1, 2)), "one for each"),
    (lambda: DeviceFigures("x", 1, {"uniform-s8": 1}), "keys"),
    (lambda: Auditor(tolerance=Fraction(-1, 2)), "tolerance"),
    (lambda: _sheet_json(clock="2/4"), "lowest terms"),
    (lambda: _sheet_json(clock="01"), "lowest terms"),
    (lambda: _sheet_json(clock="3/1"), "lowest terms"),
    (lambda: _sheet_json(clock=1), "lowest terms"),
    (lambda: _sheet_json(clock="-1"), "lowest terms"),
    (lambda: _sheet_json(clock="1e9"), "lowest terms"),
    (lambda: Auditor.of_json({**AUDITOR["remote"].to_json(), "tolerance": "0.05"}), "lowest terms"),
])
def test_what_no_site_holds_is_malformed(make, what: str) -> None:
    with pytest.raises(Malformed, match=what):
        make()
