"""The Site's types: a zero rate against any tolerance, a sheet's memory read from a card file, the encodings of a
link, a location and the auditor, what no Site type holds, a rational of any length, and the read rate of a host's
leaves. The site log's tests are registration's (`test_site_log.py`)."""

from __future__ import annotations

import sys
from dataclasses import replace
from fractions import Fraction

import pytest

from verity.channel import canonical
from verity.device import (OPERAND_CLASSES, OUTSIDE, SCHEMES, Auditor, DeviceFigures, HostFigures, Link, Loc, Malformed,
                           MemoryFigures, Sheet, Site)

GHZ = 10 ** 9
SHEET_A = Sheet(DeviceFigures("sim-int8-a", GHZ, 64 * GHZ), 3200 * GHZ, 1 << 25, 64 << 30)
SHEET_B = Sheet(DeviceFigures("sim-int8-b", GHZ, 128 * GHZ), 6400 * GHZ, 1 << 26, 128 << 30)
TOLERANCE = Fraction(1, 20)
SCHEME = SCHEMES[0]
NS = Fraction(1, 10 ** 9)
HOSTS = {"node0": HostFigures(1 << 23, (("sim-int8-a", 1), ("sim-int8-b", 1)), (("disk", 50),), (("uplink0", 100_000),)),
         "node1": HostFigures(1 << 23), "node9": HostFigures(1 << 23)}
"""The auditor's figures for the hosts a log below registers: `node0` with its on-chip state, a device of each SKU,
its disks' shared rate and its uplink; the others with their on-chip state alone."""
MEMORY = {"dram0": MemoryFigures("node0", 6554, 512 << 30), "nvme0": MemoryFigures("node0", 28, 1 << 40),
          "nvme1": MemoryFigures("node0", 28, 1 << 40), "dram1": MemoryFigures("node1", 6554, 1)}
"""The auditor's figures for the memory leaves directly under a host: its DRAM and two disks under `node0`, the
second registered later, and `node1`'s DRAM."""


AUDITOR = {"onsite": Auditor((SHEET_A, SHEET_B), TOLERANCE,
                             (("cert0", "dummy-cert0-key", SCHEME), ("chal0", "", SCHEME)),
                             tuple(MEMORY.items()), tuple(HOSTS.items())),
           "remote": Auditor((SHEET_A, SHEET_B), TOLERANCE, (), tuple(MEMORY.items()), tuple(HOSTS.items()))}
"""The auditor's data sheets for the simulated SKUs, 1 GHz clocks at 64 and 128 MACs a nanosecond in every operand class,
its 5% tolerance around them, its own devices' keys, each with its scheme (onsite a network attester and a challenger), and
its figures for the hosts and their leaves. The node registers the auditor's own sheets."""


def figures(sheet: Sheet, clock: Fraction | int | None = None, **rates: Fraction | int) -> Sheet:
    """`sheet` stating another clock, or other peak rates for the classes named (`uniform_s8=` for `uniform-s8`)."""
    f = sheet.figures
    peak = {c: rates.get(c.replace("-", "_"), f.rate(c)) for c in OPERAND_CLASSES}
    return replace(sheet, figures=replace(f, clock=f.clock if clock is None else clock, peak_rate=peak))


def test_a_zero_rate_is_refused_at_any_tolerance() -> None:
    """A tolerance of 1 or more sets no lower bound, so only the rule that every peak rate is positive refuses a zero."""
    loose = Auditor((SHEET_A,), 1)
    assert figures(SHEET_A, clock=0, uniform_s8=1).figures.accepted(loose.estimate(SHEET_A))
    assert not figures(SHEET_A, uniform_s8=0).figures.accepted(loose.estimate(SHEET_A))


def _entry(v: object) -> dict[str, object]:
    return {"value": v, "unit": "-", "mark": "ceiling", "source": "a test's card"}


CARD_FILE = {"sku": "sim-file", "kind": "gpu", "sm": {"count": _entry(2)},
             "clock": {"default": "sm", "sm": _entry(1530)},
             "rate": {"tensor": _entry(1024)}, "memory": {"bandwidth": _entry(1.23456e12)},
             "store": {"reg": _entry(65536), "reg_extra": _entry(0), "smem": _entry(262144), "l2": _entry(1 << 20),
                       "dram": _entry(12 << 30), "dram_installed": _entry(16 << 30)}}
"""A card file's table as `load_card(sku)` returns it, with the entries `Sheet.of_file` reads and the memory a program
gets (`store.dram`), which it does not."""


def test_a_sheet_reads_the_memory_on_the_board() -> None:
    """Of a card file's two readings of device memory a sheet's `mem` is the larger, `store.dram_installed`: the
    memory a program gets (`store.dram`) does not move it, and a card file that states no memory on the board gives
    no sheet. The caches only a run found are not the sheet's: its `chip` is the registers', shared memory's and
    L2's."""
    store = CARD_FILE["store"]
    assert Sheet.of_file(CARD_FILE).mem == 16 << 30
    assert Sheet.of_file({**CARD_FILE, "store": {**store, "dram": _entry(16 << 30)}}) == Sheet.of_file(CARD_FILE)
    more = {"l15": _entry(147456), "l1i": _entry(65536), "l0i": _entry(24576), "const_l0": _entry(12288)}
    assert Sheet.of_file({**CARD_FILE, "store": {**store, **more}}) == Sheet.of_file(CARD_FILE)
    with pytest.raises(Malformed, match="dram_installed"):
        Sheet.of_file({**CARD_FILE, "store": {k: e for k, e in store.items() if k != "dram_installed"}})


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


LEAF = Loc("dram0", "memory", "node0", tier="dram", bytes=1 << 31, bw=6554)
ROOT = Loc("site0", "site", time_unit=Fraction(1, 10 ** 9))
AUDITOR_JSON = {
    "drift": [{"drift": "3/5000", "id": "vm0"}],
    "hosts": [{"cards": [{"count": 8, "sku": "k"}], "chip": 8002732032, "id": "node0",
               "shared": [{"bw": 50, "tier": "disk"}],
               "ways": [{"beta": 400000000000, "id": "node0/eth0"}, {"beta": None, "id": "node0/eth1"}]}],
    "keys": [{"id": "cert0", "key": "k0", "scheme": "lms-sha512/n64-w1-h20/v0"}],
    "memory": [{"bw": 6554, "bytes": 2147483648, "host": "node0", "id": "dram0"}], "sheets": [],
    "timebases": [{"id": "vm0", "sku": "tb0"}], "tolerance": "0"}
"""`SiteV2Guards.lean`'s auditor: a network attester's key and scheme, a host's figures and a leaf's, and the drift and the
time base of a host of its own."""


def test_a_locations_and_the_auditors_encodings_are_the_leans() -> None:
    """A location writes `bw`, `chip`, `drift`, `time_unit` and `timebase` as null where it has none, `scheme` as a
    string and `shared` as a list of a rate and a tier, each; the auditor writes its keys with their schemes, its
    leaves' and hosts' figures, and its own hosts' drifts and time bases. Both read back what they write, as
    `SiteV2Guards.lean` pins them."""
    assert canonical(LEAF.to_json()) == (
        b'{"bw":6554,"bytes":2147483648,"chip":null,"drift":null,"id":"dram0","key":"","kind":"memory",'
        b'"owner":"prover","parent":"node0","scheme":"","shared":[],"sheet":"","tier":"dram","time_unit":null,'
        b'"timebase":null}')
    assert canonical(ROOT.to_json()) == (
        b'{"bw":null,"bytes":0,"chip":null,"drift":null,"id":"site0","key":"","kind":"site","owner":"prover",'
        b'"parent":null,"scheme":"","shared":[],"sheet":"","tier":null,"time_unit":"1/1000000000",'
        b'"timebase":null}')
    cert = Loc("cert0", "certifier", "site0", "auditor", key="k0", scheme=SCHEMES[1])
    host = Loc("node0", "host", "site0", chip=8002732032, shared=(("disk", 50),))
    for x in (LEAF, ROOT, cert, host):
        assert Loc.of_json(x.to_json()) == x
    a = Auditor.of_json(AUDITOR_JSON)
    assert a.memory == (("dram0", MemoryFigures("node0", 6554, 2147483648)),)
    assert a.hosts == (("node0", HostFigures(8002732032, (("k", 8),), (("disk", 50),),
                                             (("node0/eth0", 400000000000), ("node0/eth1", None)))),)
    assert (a.key("cert0"), a.scheme("cert0"), a.scheme("x"), a.drift) == (
        "k0", "lms-sha512/n64-w1-h20/v0", None, (("vm0", Fraction(3, 5000)),))
    assert a.timebases == (("vm0", "tb0"),)
    assert a.to_json() == AUDITOR_JSON and Auditor.of_json(a.to_json()) == a
    assert SCHEMES == ("lms-sha512/n64-w1-h20/v0", "hss-lms-sha512/n64-w1-h20/n64-w1-h20/v0")


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
G_HOST = Loc("node0", "host", "site0", chip=8002732032)
G_ROOT = Loc("site0", "site", time_unit=NS)


def g_dram(bw: int | None, size: int = 1 << 31) -> Loc:
    return Loc("dram0", "memory", "node0", tier="dram", bytes=size, bw=bw)


def g_disk(bw: int | None, i: int = 0, parent: str = "node0") -> Loc:
    return Loc(f"disk{i}", "memory", parent, tier="disk", bytes=1 << 40, bw=bw)


def g_disks(shared: tuple = ()) -> Site:
    return Site(locs=(G_ROOT, replace(G_HOST, shared=shared), g_dram(6554), *(g_disk(28, i) for i in range(3))))


@pytest.mark.parametrize(("shared", "xs", "rate"), [
    ((), ["disk0", "disk1", "disk2"], 84 + 6554),
    ((("disk", 50),), ["disk0", "disk1", "disk2"], 50 + 6554),
    ((("disk", 50),), ["disk0", "disk1"], 50 + 6554),
    ((("disk", 50),), ["disk0"], 28 + 6554),
    ((("disk", 100),), ["disk0", "disk1", "disk2"], 84 + 6554),
    ((("disk", 50),), ["disk0", "disk1", "disk2", "dram0"], 50 + 6554),
    ((("disk", 50),), ["dram0"], 6554),
    ((("disk", 50), ("dram", 6000)), ["disk0"], 28 + 6000),
])
def test_the_read_rate_of_a_hosts_leaves(shared: tuple, xs: list[str], rate: int) -> None:
    """`readRate`, as the guards state it: three disks at 28 bits a second each read 84 together, or the host's
    shared rate where that is less; a disk leaf's bytes can be returned from memory, so the host's DRAM tier adds its
    rate, once, whether or not its leaf is among them."""
    assert g_disks(shared).read_rate(xs) == rate
