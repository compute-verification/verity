"""The Site: each check `check_registration` makes, on a later entry of each composite's site's log (the lifecycle
example's cases, and the figures check it doesn't make yet), the order the checks are made in, a sheet's figures against
the auditor's estimate, the fold and its prefixes, the views, the device model read off the bounding figures, a sheet read
from a card file, and the log's encoding."""

from __future__ import annotations

import sys
from dataclasses import replace
from fractions import Fraction

import pytest

from verity.hardware.site import (NO_LINE_RATE, OPERAND_CLASSES, OUTSIDE, REFUSALS, SCHEMES, STORAGE_PATH, TIME_UNIT,
                                  Auditor, DeviceFigures, Entry, HostFigures, Link, Loc, Malformed, MemoryFigures,
                                  Refused, Registration, Sheet, Site, check, check_registration, genesis, node_figures,
                                  refusal)
from verity.configuration.wire import canonical

GHZ = 10 ** 9
SHEET_A = Sheet(DeviceFigures("sim-int8-a", GHZ, 64 * GHZ), 3200 * GHZ, 1 << 25, 64 << 30)
SHEET_B = Sheet(DeviceFigures("sim-int8-b", GHZ, 128 * GHZ), 6400 * GHZ, 1 << 26, 128 << 30)
A, B = SHEET_A.digest(), SHEET_B.digest()
TOLERANCE = Fraction(1, 20)
SCHEME = SCHEMES[0]
AUDITOR = {"onsite": Auditor((SHEET_A, SHEET_B), TOLERANCE,
                             (("cert0", "dummy-cert0-key", SCHEME), ("chal0", "", SCHEME))),
           "remote": Auditor((SHEET_A, SHEET_B), TOLERANCE)}
"""The auditor's data sheets for the simulated SKUs, 1 GHz clocks at 64 and 128 MACs a nanosecond in every operand class,
its 5% tolerance around them, and its own devices' keys, each with its scheme: onsite a certifier and a challenger. The
node registers the auditor's own sheets."""


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
    "unrated-link": ("kind", Entry("", links=(Link("u9", ("node0", OUTSIDE), None, 1),))),
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
example's 14, its `not-the-sheet` now a sheet of an SKU the auditor has no data sheet for, `doubled-clock`, the
figures check the example doesn't make yet, and `unrated-link`, a link that states no capacity and no reason."""
REMOTE_ACCEPTS = {"bare-uplink", "unreached-host"}
"""The entries a remote site accepts: it has no certifier and no memory challenger to hold its links to."""


def first(composite: str) -> Registration:
    log = Registration(composite)
    return check_registration(log, replace(FIRST[composite], previous=log.head), AUDITOR[composite])


def grown(composite: str = "onsite") -> Registration:
    log = first(composite)
    return check_registration(log, replace(LATER, previous=log.head), AUDITOR[composite])


def test_every_check_has_a_case() -> None:
    assert len(ENTRIES) == 16 and {name for name, _ in ENTRIES.values()} - {None} == set(REFUSALS)


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
"""A card file's table as `verity.hardware.cards.load(sku)` returns it, with the entries `Sheet.of_file` reads."""


def test_a_sheet_reads_the_memory_a_program_gets() -> None:
    """A card file's second reading of device memory and the caches only a run found are not the sheet's: its `mem`
    is `store.dram`, and its `chip` the registers', shared memory's and L2's."""
    more = {"dram_installed": _entry(24 << 30), "l15": _entry(147456), "l1i": _entry(65536), "l0i": _entry(24576),
            "const_l0": _entry(12288)}
    assert Sheet.of_file({**CARD_FILE, "store": {**CARD_FILE["store"], **more}}) == Sheet.of_file(CARD_FILE)


def test_a_sheet_from_its_card_file() -> None:
    """The clock is 1.53 GHz, and every class's peak rate 2 SMs x 1024 MACs a clock at it. At a nanosecond tick that is
    3133.44 MACs, rounded up; 1.23456e12 B/s is 9.87648e12 bits a second, and 9876.48 bits a tick, rounded up; on-chip
    state is the registers', shared memory's and L2's bits."""
    sheet = Sheet.of_file(CARD_FILE)
    assert sheet == Sheet(DeviceFigures("sim-file", 1530 * 10 ** 6, 2 * 1024 * 1530 * 10 ** 6), 9_876_480_000_000,
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


def _rate(v: object) -> dict[str, object]:
    return {"rate": {**_entry(v), "unit": "bit/s"}}


PORTS = {"eth0": _rate(25 * 10 ** 9), "eth1": _rate("1000000000.5")}
"""A node file's `interface` tables: two interfaces that leave the node, at 25 Gbit/s and at a rate that is not a whole
number of bits a second."""
NODE_FILE = {"node": "sim-file-x2", "device": {"sku": "sim-file", "count": _entry(2)}, "host": {"sku": "sim-host"},
             "memory": {"dram": {"tier": "dram", "bytes": _entry(1 << 39)},
                        "vda": {"tier": "disk", "bytes": _entry(1 << 40)}}, "interface": PORTS}
"""A node file's table as `verity.hardware.nodes.load(kind)` returns it: two devices of the card file's SKU, the
host's DRAM and one disk, and the two interfaces."""
UNRATED = {"eth0": {"driver": "virtio_net", "source": "it reads no rate"}}
"""A node file's `interface` tables where the one interface reads no line rate."""
STORAGE = {"leaves": ["vda"], "source": "the disk is a network disk"}
"""A node file's `storage` table: the path the disk that is not the server's own carries bytes off the node by."""
DISKS = 5_293_292_307_696
"""A rate that bounds a host's disks together, in bits a second."""
DRAM = 4_915_200_000_000
"""A host's DRAM's rate, in bits a second."""
HOST_CARD = {"sku": "sim-host", "store": {"chip": _entry(1 << 20)}}
"""The host card's table `NODE_FILE` names, with the entry `node_entry` reads: its on-chip state, 1 MiB."""
ETH0, ETH1 = (Link("n0/eth0", ("n0", OUTSIDE), 25 * 10 ** 9, 0), Link("n0/eth1", ("n0", OUTSIDE), 1_000_000_001, 0))
"""`NODE_FILE`'s interfaces as links: each at its rate in whole bits a second, rounded up, saying no reason."""
PATH = Link("n0/storage", ("n0", OUTSIDE), DISKS, 0, STORAGE_PATH)
"""The storage path's link: at the rate that bounds the disks together, saying why no certifier splits it."""
QUIET = Link("n0/eth0", ("n0", OUTSIDE), None, 0, NO_LINE_RATE)
"""An interface that reads no line rate as a link: no capacity, and why no certifier splits it."""


def _ways(host: str = "n0", storage: int | None = None, eth0: int | None = 25 * 10 ** 9) -> tuple[tuple, ...]:
    """`NODE_FILE`'s ways out under `host` as `verity.hardware.nodes.ways_out` gives them: each interface with its
    rate in whole bits a second, rounded up, which a certifier may split, then the storage path at `storage`, which
    none may, where the file states one."""
    ports = ((f"{host}/eth0", eth0, True), (f"{host}/eth1", 1_000_000_001, True))
    return ports if storage is None else (*ports, (f"{host}/storage", storage, False))


def _leaves(host: str = "n0", memory: object = None) -> tuple:
    """`NODE_FILE`'s leaves under `host`, or those of these `memory` tables, as `verity.hardware.nodes.leaves` gives
    them: each with its id, tier, bytes and rate, a DRAM leaf at `DRAM` and a disk at `DISKS`, and each tier's rate."""
    rates = {"dram": DRAM, "disk": DISKS}
    tables = NODE_FILE["memory"] if memory is None else memory
    each = tuple((f"{host}/{name}", leaf.get("tier"), leaf.get("bytes", {}).get("value"), rates.get(leaf.get("tier"), 1))
                 for name, leaf in tables.items()) if isinstance(tables, dict) else ()
    return each, rates


def _entry_of(log: Registration, node: dict, host: str = "n0", site: str = "site", **more: object) -> Entry:
    """`log`'s entry of node file `node` as host `host`, given the host card and its leaves."""
    return log.node_entry(node, CARD_FILE, host, site, host_card=HOST_CARD,
                          leaves=_leaves(host, node.get("memory")), **more)


def _reached(entry: Entry, site: str = "site") -> Entry:
    """`entry` with a memory challenger under `site` linked to host `n0`."""
    return replace(entry, locs=(*entry.locs, Loc("chal0", "challenger", site, "auditor")),
                   links=(*entry.links, Link("chal0-link", ("chal0", "n0"), 100_000, 1_000_000)))


def test_a_nodes_entry_from_its_files() -> None:
    """The entry holds the card's sheet, the site at a nanosecond, the host under the id its caller gives with its
    on-chip state in bits and its tiers' shared rates, each device with one `hbm` leaf at the sheet's device memory,
    whole, the host's own leaves in the file's order at their rates, and its ways out as links. A second node on the
    same log adds neither the sheet nor the site again. Remote accepts both; onsite refuses the
    entry by `uncertified`, and the same entry without its links out by `challenger-reach` until a memory challenger
    links to the host."""
    sheet, auditor = Sheet.of_file(CARD_FILE), Auditor.of_files([CARD_FILE], TOLERANCE, {"chal0": ("", SCHEME)})
    log = Registration("remote")
    first = _entry_of(log, NODE_FILE, "n0", "site0", ways=_ways())
    hbm = {"tier": "hbm", "bytes": 16 << 30}
    assert first == log.entry([sheet], [
        Loc("site0", "site", time_unit=Fraction(1, 10 ** 9)),
        Loc("n0", "host", "site0", chip=8 << 20, shared=(("dram", DRAM), ("disk", DISKS))),
        Loc("n0/gpu0", "device", "n0", sheet=sheet.digest()), Loc("n0/gpu0/hbm", "memory", "n0/gpu0", **hbm),
        Loc("n0/gpu1", "device", "n0", sheet=sheet.digest()), Loc("n0/gpu1/hbm", "memory", "n0/gpu1", **hbm),
        Loc("n0/dram", "memory", "n0", tier="dram", bytes=1 << 39, bw=DRAM),
        Loc("n0/vda", "memory", "n0", tier="disk", bytes=1 << 40, bw=DISKS)],
        [ETH0, ETH1])
    assert TIME_UNIT == Fraction(1, 10 ** 9)
    log = check_registration(log, first, auditor)
    second = _entry_of(log, NODE_FILE, "n1", "site0", ways=_ways("n1"))
    assert second.sheets == () and [x.id for x in second.locs[:2]] == ["n1", "n1/gpu0"]
    site = check(log.append(second), auditor)
    assert site.units == ("n0/gpu0", "n0/gpu1", "n1/gpu0", "n1/gpu1") and site.sheet("n1/gpu1") == sheet
    assert site.tiers("n1") == {"hbm": 2 * (16 << 30), "dram": 1 << 39, "disk": 1 << 40}
    assert refusal(log, first, auditor) == "previous"
    assert refusal(log, _entry_of(log, NODE_FILE, "n0", "site0", ways=_ways()), auditor) == "duplicate-id"
    onsite = Registration("onsite")
    bare = _entry_of(onsite, NODE_FILE, "n0", ways=_ways())
    assert bare.locs[0] == Loc("site", "site", time_unit=TIME_UNIT) and refusal(onsite, bare, auditor) == "uncertified"
    assert refusal(onsite, replace(bare, links=()), auditor) == "challenger-reach"
    assert refusal(onsite, _reached(replace(bare, links=())), auditor) is None


def test_a_nodes_interfaces_are_its_own_boundary_links() -> None:
    """Each interface that states a rate is a link from the host to the outside at its rate, at latency 0, saying no
    reason: a boundary link with no certifier at an end, which remote accepts and onsite refuses by `uncertified`,
    with a challenger reaching the host or without one. An interface whose table states no rate is a link that states
    no capacity and says why no certifier splits it, which onsite takes: it names only the rated interface. A site
    with that link is not mediated."""
    node, auditor = NODE_FILE, Auditor.of_files([CARD_FILE], TOLERANCE, {"chal0": ("", SCHEME)})
    entry = _entry_of(Registration("remote"), node, "n0", ways=_ways())
    assert entry.links == (ETH0, ETH1) and (entry.links[0].unmediated, entry.links[1].unmediated) == (None, None)
    quiet = {**node, "interface": {**UNRATED, "eth1": PORTS["eth1"]}}
    unrated = _entry_of(Registration("remote"), quiet, "n0", ways=_ways(eth0=None))
    assert unrated.links == (QUIET, ETH1) and unrated.locs == entry.locs
    site = check(Registration("remote").append(entry), auditor)
    assert site.boundary("n0") == list(entry.links) and not any(site.certified(k) for k in site.links)
    assert site.mediated and not check(Registration("remote").append(unrated), auditor).mediated
    onsite = Registration("onsite")
    for file, ways, bare in ((node, _ways(), "['n0/eth0', 'n0/eth1']"), (quiet, _ways(eth0=None), "['n0/eth1']")):
        e = _entry_of(onsite, file, "n0", ways=ways)
        for at in (e, _reached(e)):
            with pytest.raises(Refused) as refused:
                check_registration(onsite, at, auditor)
            assert refused.value.name == "uncertified" and bare in refused.value.detail, refused.value


def test_a_stand_in_certifier_is_registered_beside_the_nodes_own_links() -> None:
    """With a certifier's id and key the entry also holds the certifier under the site and a certified way out, host
    to certifier and certifier to outside, each at the interfaces' rates summed. The interfaces' own links stay, so
    onsite still refuses the entry by `uncertified`, naming them and not the certified link."""
    node = NODE_FILE
    auditor = Auditor.of_files([CARD_FILE], TOLERANCE, {"cert0": ("dummy-cert0-key", SCHEME), "chal0": ("", SCHEME)})
    plain = _entry_of(Registration("remote"), node, "n0", "site0", ways=_ways())
    entry = _entry_of(Registration("remote"), node, "n0", "site0", certifier=("cert0", "dummy-cert0-key", SCHEME),
                                              ways=_ways())
    assert entry.locs == (*plain.locs, Loc("cert0", "certifier", "site0", "auditor", key="dummy-cert0-key",
                                           scheme=SCHEME))
    assert entry.links == (*plain.links, Link("n0/cert0", ("n0", "cert0"), 26_000_000_001, 0),
                           Link("cert0/outside", ("cert0", OUTSIDE), 26_000_000_001, 0))
    site = check(Registration("remote").append(entry), auditor)
    assert [(k.id, site.certified(k)) for k in site.boundary("n0")] == [
        ("n0/eth0", False), ("n0/eth1", False), ("n0/cert0", True)]
    assert site.mediated
    onsite = Registration("onsite")
    stood = _entry_of(onsite, node, "n0", "site0", certifier=("cert0", "dummy-cert0-key", SCHEME), ways=_ways())
    reached = _reached(stood, "site0")
    with pytest.raises(Refused) as refused:
        check_registration(onsite, reached, auditor)
    assert refused.value.name == "uncertified" and "n0/eth1" in refused.value.detail
    assert "cert0" not in refused.value.detail
    certified = replace(reached, links=tuple(k for k in reached.links if k not in plain.links))
    assert refusal(onsite, certified, auditor) is None


def test_a_nodes_storage_path_is_a_boundary_link_no_certifier_is_linked_to() -> None:
    """A file that states a storage path registers it as a link from the host to the outside, after the interfaces',
    at the path's rate, saying why no certifier splits it. A certifier's two links stay at the interfaces' rates
    summed: the path's rate is not in the sum, and no certifier is at an end of its link. So onsite refuses the entry
    by `uncertified` with the interfaces' bare links, naming them and not the path's, and accepts it without them,
    the path's link and all; the site is then not mediated. Where the file's one interface has no rate, the entry's
    links are that interface's, with no capacity, and the path's, and a certifier named on the node is refused."""
    node = {**NODE_FILE, "storage": STORAGE}
    auditor = Auditor.of_files([CARD_FILE], TOLERANCE, {"cert0": ("dummy-cert0-key", SCHEME), "chal0": ("", SCHEME)})
    plain = _entry_of(Registration("remote"), node, "n0", ways=_ways(storage=DISKS))
    assert plain.links == (ETH0, ETH1, PATH)
    assert plain.locs == _entry_of(Registration("remote"), NODE_FILE, "n0", ways=_ways()).locs
    entry = _entry_of(Registration("remote"), node, "n0", certifier=("cert0", "dummy-cert0-key", SCHEME),
                                              ways=_ways(storage=DISKS))
    assert entry.links == (*plain.links, Link("n0/cert0", ("n0", "cert0"), 26_000_000_001, 0),
                           Link("cert0/outside", ("cert0", OUTSIDE), 26_000_000_001, 0))
    site = check(Registration("remote").append(entry), auditor)
    assert [(k.id, site.certified(k)) for k in site.boundary("n0")] == [
        ("n0/eth0", False), ("n0/eth1", False), ("n0/storage", False), ("n0/cert0", True)]
    assert site.far("cert0") == ["n0", OUTSIDE] and not site.mediated
    onsite = Registration("onsite")
    reached = _reached(_entry_of(onsite, node, "n0", certifier=("cert0", "dummy-cert0-key", SCHEME),
                                         ways=_ways(storage=DISKS)))
    with pytest.raises(Refused) as refused:
        check_registration(onsite, reached, auditor)
    assert refused.value.name == "uncertified" and "['n0/eth0', 'n0/eth1']" in refused.value.detail
    stored = replace(reached, links=tuple(k for k in reached.links if k not in plain.links[:2]))
    assert refusal(onsite, stored, auditor) is None
    assert not check(onsite.append(stored), auditor).mediated
    one = {**node, "interface": UNRATED}
    alone = (("n0/eth0", None, True), ("n0/storage", DISKS, False))
    lone = _entry_of(Registration("remote"), one, "n0", ways=alone)
    assert lone.links == (QUIET, PATH)
    assert refusal(onsite, _reached(_entry_of(onsite, one, "n0", ways=alone)), auditor) is None
    with pytest.raises(Malformed, match=r"no interface the node file states has a rate \(\['n0/eth0'\]\).*takes no certifier"):
        _entry_of(Registration("remote"), one, "n0", certifier=("cert0", "dummy-cert0-key", SCHEME), ways=alone)


def test_an_interface_that_reads_no_rate_is_registered_without_a_capacity() -> None:
    """A node whose one way out is an interface that reads no line rate registers that link alone: it states no
    capacity and says why no certifier splits it, so remote and onsite (with a challenger) accept it and the site is
    not mediated."""
    one = {**NODE_FILE, "interface": UNRATED}
    auditor = Auditor.of_files([CARD_FILE], TOLERANCE, {"chal0": ("", SCHEME)})
    entry = _entry_of(Registration("remote"), one, "n0", ways=[("n0/eth0", None, True)])
    assert entry.links == (QUIET,) and not check(Registration("remote").append(entry), auditor).mediated
    onsite = Registration("onsite")
    assert refusal(onsite, _reached(_entry_of(onsite, one, "n0", ways=[("n0/eth0", None, True)])),
                   auditor) is None


@pytest.mark.parametrize("composite", ["onsite", "remote"])
def test_a_link_states_no_capacity_only_with_its_reason(composite: str) -> None:
    """`kind` refuses a link that states no capacity and says no reason, and a link at a capacity of 0 whatever it
    says, and takes a link with no capacity that says why no certifier splits it. Onsite, `uncertified` refuses a bare
    boundary link that says no reason and takes it with one; remote takes both. A site with a link that says a reason
    is not mediated."""
    log, auditor = first(composite), AUDITOR[composite]
    unrated, bare = Link("u9", ("node0", OUTSIDE), None, 1), Link("u9", ("node0", OUTSIDE), 1, 1)
    assert refusal(log, log.entry(links=[unrated]), auditor) == "kind"
    assert refusal(log, log.entry(links=[replace(bare, beta=0, unmediated="a stand-in")]), auditor) == "kind"
    assert refusal(log, log.entry(links=[bare]), auditor) == ("uncertified" if composite == "onsite" else None)
    for k in (replace(unrated, unmediated="a stand-in"), replace(bare, unmediated="a stand-in")):
        assert refusal(log, entry := log.entry(links=[k]), auditor) is None
        assert Site.of_log(log).mediated and not check(log.append(entry), auditor).mediated


def test_a_links_encoding_is_pinned() -> None:
    """A link's JSON writes `beta` and `unmediated` as null where it has none, as the Lean's `Link.toJson` does."""
    unrated = Link("eth0", ("node0", OUTSIDE), None, 0, "a stand-in")
    bare = Link("eth0", ("node0", OUTSIDE), 1_000_000_000, 0, "a stand-in")
    assert canonical(unrated.to_json()) == (
        b'{"beta":null,"ends":["node0","outside"],"id":"eth0","latency":0,"unmediated":"a stand-in"}')
    assert canonical(bare.to_json()) == (
        b'{"beta":1000000000,"ends":["node0","outside"],"id":"eth0","latency":0,"unmediated":"a stand-in"}')
    for k in (unrated, bare, replace(bare, unmediated=None)):
        assert Link.of_json(k.to_json()) == k


def _node_entry(certifier: tuple[str, str, str] | None = None, ways: object = None, leaves: object = None,
                host_card: object = None, **tables: object) -> Entry:
    """The entry of `NODE_FILE` with these tables in place of its own, and its ways out, its leaves and its host card
    unless others are given."""
    node = {**NODE_FILE, **tables}
    return Registration("remote").node_entry(
        node, CARD_FILE, "n0", certifier=certifier, ways=_ways() if ways is None else ways,
        leaves=_leaves(memory=node.get("memory")) if leaves is None else leaves,
        host_card=HOST_CARD if host_card is None else host_card)



def test_the_encoding_round_trips_and_is_pinned() -> None:
    """A rational is the string `n` or `n/d` in lowest terms, so a sheet has one encoding and one digest."""
    log = grown()
    assert Registration.of_json(log.to_json()) == log and Auditor.of_json(AUDITOR["onsite"].to_json()) == AUDITOR["onsite"]
    assert Site.of_json(Site.of_log(log.entries).to_json()) == Site.of_log(log.entries)
    assert genesis("onsite") == GENESIS_ONSITE
    assert genesis("remote") != genesis("onsite") and log.head == log.entries[-1].digest()
    half = figures(SHEET_A, clock=Fraction(1, 2), zero=Fraction(64 * GHZ, 3))
    assert canonical(half.to_json()) == (
        b'{"bw":3200000000000,"chip":33554432,"figures":{"clock":"1/2","peak_rate":{"uniform-s2":"64000000000",'
        b'"uniform-s8":"64000000000","uniform-u2":"64000000000","uniform-u4":"64000000000","zero":"64000000000/3"},'
        b'"sku":"sim-int8-a"},"mem":68719476736}')
    assert Sheet.of_json(half.to_json()) == half and AUDITOR["onsite"].to_json()["tolerance"] == "1/20"


LEAF = Loc("dram0", "memory", "node0", tier="dram", bytes=1 << 31, bw=6554)
ROOT = Loc("site0", "site", time_unit=Fraction(1, 10 ** 9))
AUDITOR_JSON = {
    "drift": [{"drift": "3/5000", "id": "cert0"}],
    "hosts": [{"cards": [{"count": 8, "sku": "k"}], "chip": 8002732032, "id": "node0",
               "shared": [{"bw": 50, "tier": "disk"}],
               "ways": [{"beta": 400000000000, "id": "node0/eth0"}, {"beta": None, "id": "node0/eth1"}]}],
    "keys": [{"id": "cert0", "key": "k0", "scheme": "lms-sha512/n64-w1-h20/v0"}],
    "memory": [{"bw": 6554, "bytes": 2147483648, "host": "node0", "id": "dram0"}], "sheets": [], "tolerance": "0"}
"""`SiteV2Guards.lean`'s auditor: a certifier's key, scheme and drift, a host's figures and a leaf's."""


def test_a_locations_and_the_auditors_encodings_are_the_leans() -> None:
    """A location writes `bw`, `chip`, `drift` and `time_unit` as null where it has none, `scheme` as a string and
    `shared` as a list of a rate and a tier, each; the auditor writes its keys with their schemes, its leaves' and
    hosts' figures and its devices' drifts. Both read back what they write, as `SiteV2Guards.lean` pins them."""
    assert canonical(LEAF.to_json()) == (
        b'{"bw":6554,"bytes":2147483648,"chip":null,"drift":null,"id":"dram0","key":"","kind":"memory",'
        b'"owner":"prover","parent":"node0","scheme":"","shared":[],"sheet":"","tier":"dram","time_unit":null}')
    assert canonical(ROOT.to_json()) == (
        b'{"bw":null,"bytes":0,"chip":null,"drift":null,"id":"site0","key":"","kind":"site","owner":"prover",'
        b'"parent":null,"scheme":"","shared":[],"sheet":"","tier":null,"time_unit":"1/1000000000"}')
    cert = Loc("cert0", "certifier", "site0", "auditor", key="k0", scheme=SCHEMES[1], drift=Fraction(1, 10))
    host = Loc("node0", "host", "site0", chip=8002732032, shared=(("disk", 50),))
    for x in (LEAF, ROOT, cert, host):
        assert Loc.of_json(x.to_json()) == x
    a = Auditor.of_json(AUDITOR_JSON)
    assert a.memory == (("dram0", MemoryFigures("node0", 6554, 2147483648)),)
    assert a.hosts == (("node0", HostFigures(8002732032, (("k", 8),), (("disk", 50),),
                                             (("node0/eth0", 400000000000), ("node0/eth1", None)))),)
    assert (a.key("cert0"), a.scheme("cert0"), a.scheme("x"), a.drift) == (
        "k0", "lms-sha512/n64-w1-h20/v0", None, (("cert0", Fraction(3, 5000)),))
    assert a.to_json() == AUDITOR_JSON and Auditor.of_json(a.to_json()) == a
    assert SCHEMES == ("lms-sha512/n64-w1-h20/v0", "hss-lms-sha512/n64-w1-h20/n64-w1-h20/v0")


def test_a_nodes_figures_are_the_ones_its_entry_states() -> None:
    """`node_figures` gives the auditor's figures for a node from the files `node_entry` reads: each leaf of the
    host's at its rate and bytes, and the host's on-chip state in bits, its cards, its tiers' shared rates and its ways
    out at their rates. The entry states the same on the host and its leaves, and a certifier's key and scheme are
    the ones its caller gives."""
    node = {**NODE_FILE, "storage": STORAGE}
    ways = _ways(storage=DISKS)
    memory, figures = node_figures(node, HOST_CARD, "n0", ways=ways, leaves=_leaves())
    assert memory == (("n0/dram", MemoryFigures("n0", DRAM, 1 << 39)), ("n0/vda", MemoryFigures("n0", DISKS, 1 << 40)))
    assert figures == HostFigures(8 << 20, (("sim-file", 2),), (("dram", DRAM), ("disk", DISKS)),
                                  (("n0/eth0", 25 * 10 ** 9), ("n0/eth1", 1_000_000_001), ("n0/storage", DISKS)))
    entry = _entry_of(Registration("remote"), node, ways=ways)
    host = next(x for x in entry.locs if x.id == "n0")
    assert (host.chip, host.shared) == (figures.chip, figures.shared)
    assert [(x.id, MemoryFigures(x.parent, x.bw, x.bytes)) for x in entry.locs if x.parent == "n0" and
            x.kind == "memory"] == list(memory)
    assert all(x.bw is None and x.chip is None and x.shared == () for x in entry.locs if x.id != "n0" and
               x.parent != "n0" or x.kind == "device")
    assert [(k.id, k.beta) for k in entry.links] == list(figures.ways)


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
    (lambda: Link("l", ("a", "b"), -1, 1), "beta"),
    (lambda: Loc("x", "host", "site0", chip=-1), "chip"),
    (lambda: Loc("x", "memory", "node0", tier="dram", bytes=1, bw=1.5), "bw"),
    (lambda: Loc("x", "host", "site0", shared=(("sram", 1),)), "tier"),
    (lambda: Loc("x", "host", "site0", shared=("dram",)), "a list of pairs"),
    (lambda: Loc("x", "site", time_unit=-1), "time unit"),
    (lambda: Loc("x", "certifier", "site0", scheme=7), "scheme"),
    (lambda: HostFigures(1, ways=(("w", -1),)), "beta"),
    (lambda: HostFigures(1, cards=(("k",),)), "a list of pairs"),
    (lambda: Auditor(keys=(("cert0", "k0"),)), "each is a device's id, its key and the key's scheme"),
    (lambda: Auditor(memory=(("dram0", 6554),)), "MemoryFigures"),
    (lambda: Auditor.of_json({**AUDITOR_JSON, "keys": [{"id": "cert0", "key": "k0"}]}), "keys"),
    (lambda: Loc.of_json({k: v for k, v in LEAF.to_json().items() if k != "scheme"}), "keys"),
    (lambda: Link("l", ("a", "b"), 1, 1, 7), "unmediated"),
    (lambda: Link.of_json({"beta": 1, "ends": ["a", "b"], "id": "l", "latency": 0}), "keys"),
    (lambda: Registration("hybrid"), "composite"),
    (lambda: Entry.of_json({**LATER.to_json(), "format": "verity/registration/information/v1"}), "format"),
    (lambda: Registration.of_json({"composite": "onsite", "entries": [], "extra": 1}), "keys"),
    (lambda: Site.of_json({"links": [], "locs": [], "sheets": [], "extra": 1}), "keys"),
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
    (lambda: _node_entry(device={"sku": "sim-file"}), "count"),
    (lambda: _node_entry(device={"sku": "sim-file", "count": 2}), "not an entry"),
    (lambda: _node_entry(device={"sku": "sim-file", "count": _entry(1.5)}), "whole number"),
    (lambda: _node_entry(device={"sku": "sim-int8-a", "count": _entry(1)}), "the card given"),
    (lambda: _node_entry(memory={"dram": {"tier": "dram"}}), "bytes"),
    (lambda: _node_entry(memory={**NODE_FILE["memory"], "x": {"tier": "sram", "bytes": _entry(1)}}), "tier"),
    (lambda: _node_entry(memory={"vda": NODE_FILE["memory"]["vda"]}), "no dram leaf"),
    (lambda: _node_entry(memory={"dram": NODE_FILE["memory"]["dram"]}), "no disk leaf"),
    (lambda: _node_entry(memory={}), "no dram and no disk leaf.*lsmem -b --summary=only; lsblk -b -d"),
    (lambda: _node_entry(interface={}, ways=()), "states no interface, so its entry would not say how.*ip route"),
    (lambda: _node_entry(interface={}, storage=STORAGE, ways=[("n0/storage", DISKS, False)]), "states no interface"),
    (lambda: _node_entry(("cert0", "k", SCHEME), interface=UNRATED, storage=STORAGE,
                         ways=[("n0/eth0", None, True), ("n0/storage", DISKS, False)]), "takes no certifier"),
    (lambda: _node_entry(ways=()), r"the ways out given are \[\].*give `verity.hardware.nodes.ways_out\(node, host\)`"),
    (lambda: _node_entry(ways=_ways()[:1]), "the ways out given are"),
    (lambda: _node_entry(ways=_ways(eth0=None)),
     r"the ways out given are \[\('n0/eth0', True, False\).*the node file states \[\('n0/eth0', True, True\)"),
    (lambda: _node_entry(interface={**PORTS, **UNRATED}), r"states \[\('n0/eth0', True, False\), \('n0/eth1', True, True\)\]"),
    (lambda: _node_entry(storage=STORAGE, ways=[*_ways(), ("n0/storage", None, False)]),
     r"the ways out given are .*\('n0/storage', False, False\)\], and the node file states .*\('n0/storage', False, True\)"),
    (lambda: _node_entry(interface={"eth0": 7}, ways=_ways()[:1]), "a node file's table"),
    (lambda: _node_entry(ways=_ways()[::-1]), "the ways out given are"),
    (lambda: _node_entry(ways=_ways(storage=DISKS)), "the ways out given are"),
    (lambda: _node_entry(storage=STORAGE), "the ways out given are"),
    (lambda: _node_entry(storage=STORAGE, ways=[*_ways(), ("n0/storage", DISKS, True)]), "the ways out given are"),
    (lambda: _node_entry(ways=_ways("n1")), "its id is n0/NAME"),
    (lambda: _node_entry(ways=_ways(eth0=0)), "above 0"),
    (lambda: _node_entry(ways=_ways(eth0=-1)), "above 0"),
    (lambda: _node_entry(ways=_ways(eth0=2.5)), "a whole number of bits a second"),
    (lambda: _node_entry(ways=_ways(eth0=True)), "a whole number of bits a second"),
    (lambda: _node_entry(ways=[("n0/eth0", 25, 1), _ways()[1]]), "its split a bool"),
    (lambda: _node_entry(ways=[("n0/eth0", 25), _ways()[1]]), "a way out is its id, its bits a second and whether"),
    (lambda: _node_entry(ways=7), "`ways` is 7; give the node file's ways out"),
    (lambda: _node_entry(interface=7), "a node file's table"),
    (lambda: _node_entry(leaves=(_leaves()[0][:1], _leaves()[1])), "the leaves given are"),
    (lambda: _node_entry(leaves=(_leaves()[0][::-1], _leaves()[1])), "the leaves given are"),
    (lambda: _node_entry(leaves=_leaves("n1")), "its id is n0/NAME"),
    (lambda: _node_entry(leaves=((("n0/dram", "dram", 1 << 39, 0), _leaves()[0][1]), _leaves()[1])), "above 0"),
    (lambda: _node_entry(leaves=(_leaves()[0], {"dram": DRAM, "disk": 0})), "shared rates"),
    (lambda: _node_entry(leaves=(_leaves()[0], {"dram": DRAM, "sram": 1})), "shared rates"),
    (lambda: _node_entry(leaves=7), "`leaves` is 7"),
    (lambda: _node_entry(host_card={**HOST_CARD, "sku": "sim-other"}), "the host card given is 'sim-other'"),
    (lambda: _node_entry(host_card={"sku": "sim-host"}), "no store.chip"),
    (lambda: _node_entry("cert0"), "id, its key and the key's scheme"),
    (lambda: _node_entry(("cert0", "k")), "id, its key and the key's scheme"),
    (lambda: _node_entry(("cert0", 7, SCHEME)), "not a string"),
])
def test_what_no_site_holds_is_malformed(make, what: str) -> None:
    with pytest.raises(Malformed, match=what):
        make()


LONG = "1" + "0" * 4299 + "1"
"""A natural number of 4,301 decimal digits, coprime to 3: one more than Python's `int` and `str` convert at the
interpreter's default limit on integer strings."""


@pytest.fixture
def lowest_limit():
    """The test runs at the lowest limit on integer strings the interpreter takes, whatever its environment set: the
    Site reads and writes a rational under every setting of it."""
    before = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(640)
    yield
    sys.set_int_max_str_digits(before)


def test_a_rational_of_any_length_is_read_and_written(lowest_limit) -> None:
    """A numerator and a denominator of 4,301 digits, which Python's own `int` refuses, are read as the numbers they
    write and written back as they came, so a sheet stating them has its digest."""
    n = 10 ** 4300 + 1
    with pytest.raises(ValueError, match="integer string conversion"):
        int(LONG)
    for clock, q in ((LONG, Fraction(n)), (f"1/{LONG}", Fraction(1, n)), (f"{LONG}/3", Fraction(n, 3)),
                     (f"{LONG[:-1]}2/{LONG}", Fraction(n + 1, n))):
        sheet = _sheet_json(clock=clock)
        assert sheet.figures.clock == q and sheet.to_json()["figures"]["clock"] == clock
        assert Sheet.of_json(sheet.to_json()) == sheet and len(sheet.digest()) == 128
    auditor = Auditor.of_json({**AUDITOR["remote"].to_json(), "tolerance": f"1/{LONG}"})
    assert auditor.tolerance == Fraction(1, n) and Auditor.of_json(auditor.to_json()) == auditor


def test_a_rational_in_another_spelling_is_malformed_at_any_length(lowest_limit) -> None:
    """Lean reads one spelling of a rational and refuses the rest, however long: a fraction not in lowest terms, a
    leading zero, a denominator of 1 or 0, zero over a denominator, a sign, a space, a line end, an underscore, a second
    slash, a missing side, and a digit that is not ASCII's."""
    for clock in (f"{LONG[:-1]}2/4", f"{LONG}/{LONG}", f"3/{'3' * 4301}", "0" + LONG, "1/0" + LONG, LONG + "/1",
                  LONG + "/0", "0/" + LONG, "-" + LONG, "+1/2", " 1/2", "1/2\n", "1_0/3", "1/2/3", "0/5", "1/0", "", "/2",
                  "1/", "\u0663"):
        with pytest.raises(Malformed, match="lowest terms"):
            _sheet_json(clock=clock)
