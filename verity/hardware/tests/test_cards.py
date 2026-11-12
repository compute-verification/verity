"""The card files: every number sourced and marked, one schema across the cards of a kind, and the numbers derived from
others."""

from __future__ import annotations

import copy
import math
import re
from fractions import Fraction

import pytest

from verity.errors import InvalidArtifact
from verity.hardware import cards
from verity.hardware.cards import KINDS, MARKS, load, skus


def entries(table: dict, at: str = "") -> dict[str, dict]:
    """Every number entry of a card, by its dotted key."""
    out = {}
    for key, item in table.items():
        if isinstance(item, dict) and "value" in item:
            out[at + key] = item
        elif isinstance(item, dict):
            out |= entries(item, f"{at}{key}.")
    return out


def value(card: dict, key: str) -> Fraction:
    return Fraction(str(entries(card)[key]["value"]))


def test_the_cards_we_run() -> None:
    assert skus("gpu") == ["b200-sxm-180gb", "h200-sxm-141gb", "rtx-pro-6000-server"]
    assert skus("host") == ["xeon-6776p-2s", "xeon-platinum-8468-2s"]
    assert skus("server") == ["dgx-h200"]
    assert skus() == sorted(skus("gpu") + skus("host") + skus("server"))


#: The fewest entries a card of each kind holds, so a card whose sections didn't parse as entries fails.
FEWEST = {"gpu": 20, "host": 20, "server": 4}


@pytest.mark.parametrize("sku", skus())
def test_every_number_is_sourced_and_marked(sku: str) -> None:
    card = load(sku)
    numbers = entries(card)
    assert len(numbers) >= FEWEST[card["kind"]]
    assert all(e["mark"] in MARKS and e["source"].strip() for e in numbers.values())


@pytest.mark.parametrize("sku", skus())
def test_sources_cite_the_store_not_runs_or_hosts(sku: str) -> None:
    """Evidence is cited by its `art:` id, and a node by its kind, never by a run id or a host name."""
    text = (cards.CARDS / f"{sku}.toml").read_text(encoding="utf-8")
    assert not re.findall(r"\br\d{8}-\d{6}-[0-9a-f]{4}\b|\bvy-[a-z0-9-]+", text), \
        "cite the run's record (`research data show RUN`: outputs.run_record) as art:<12 hex>, and name the node by its kind"


def test_a_measurement_cites_the_run_that_measured_it() -> None:
    """A measured number nobody can check: an entry marked `measurement` whose source names a probe program, a census or
    a document and no run's record, so a reader of the card cannot reach the output that holds its figure."""
    measured = {f"{sku}: {key}": entry["source"] for sku in skus() for key, entry in entries(load(sku)).items()
                if entry["mark"] == "measurement"}
    assert measured, "no card marks a number `measurement`, so this test checks nothing"
    uncited = sorted(at for at, source in measured.items() if not re.search(r"\bart:[0-9a-f]{12}", source))
    assert not uncited, (
        f"marked `measurement` with no run cited: {uncited}; add the record of the run that measured each to its "
        "source as art:<12 hex> (`research data show RUN`: outputs.run_record), or mark it `placeholder` until a run "
        "measures it")


#: The entries a card of a kind holds only where a run measured them: a figure of one node's VM, or of one card's
#: caches, that no document gives for another's, so a card without the measurement has no entry rather than an
#: invented one.
MEASURED_ONLY = {"gpu": {"store.l15", "store.l1i", "store.l0i", "store.const_l0"}, "host": {"disk.read"}}


@pytest.mark.parametrize("kind", KINDS)
def test_one_schema_across_the_cards_of_a_kind(kind: str) -> None:
    """A reader can take any number from any card of a kind by the same key, the entries of `MEASURED_ONLY` apart: a
    card that has one marks it a measurement, and some card has each."""
    apart = MEASURED_ONLY.get(kind, set())
    keys = {sku: sorted(set(entries(load(sku))) - apart) for sku in skus(kind)}
    assert len({tuple(k) for k in keys.values()}) == 1, keys
    for key in apart:
        marks = {card[key]["mark"] for card in map(entries, map(load, skus(kind))) if key in card}
        assert marks == {"measurement"}, (key, marks)


@pytest.mark.parametrize("sku", skus("gpu"))
def test_derived_gpu_numbers(sku: str) -> None:
    """Device memory's bandwidth is its clock's transfers over the bus, which bound its read."""
    card = load(sku)
    assert value(card, "memory.bandwidth") == (value(card, "clock.memory") * 10 ** 6 * value(card, "memory.transfers")
                                               * value(card, "memory.bus") / 8)
    assert 0 < value(card, "memory.read") <= value(card, "memory.bandwidth")


#: The uniform registers a warp that each card's `store.reg_extra` counts: the 80 a run found hold values on the RTX PRO
#: 6000, and 63 on the cards whose uniform registers no run has swept.
UNIFORM_REGISTERS = {"b200-sxm-180gb": 63, "h200-sxm-141gb": 63, "rtx-pro-6000-server": 80}


@pytest.mark.parametrize("sku", skus("gpu"))
def test_the_extra_registers_cover_the_resident_warps(sku: str) -> None:
    """`sm.warps` is the most warps resident on an SM, and `store.reg_extra` holds at least their uniform registers
    (`UNIFORM_REGISTERS` a warp) and their predicates (7 bits a thread at 32 threads a warp, and 7 bits a warp), in
    32-bit words. Its source states the count, and a count that is a run's cites the run's record."""
    card = load(sku)
    warps, uniform = value(card, "sm.warps"), UNIFORM_REGISTERS[sku]
    assert value(card, "store.reg_extra") >= uniform * warps + 7 * warps + math.ceil(7 * warps / 32)
    source = card["store"]["reg_extra"]["source"]
    assert f"({uniform} a warp at {warps} warps" in source
    assert (re.search(r"\bart:[0-9a-f]{12}\b", source) is not None) == (uniform != 63)


def test_the_registers_a_run_found_on_the_rtx_pro_6000() -> None:
    """The RTX PRO 6000's `store.reg_extra` is its 80 uniform registers a warp, its predicates and its uniform
    predicates and nothing more, pinned with the on-chip bytes its stores then give a sheet (the registers, shared
    memory and L2) so that a change to either is deliberate: a Lean table and every sheet of the card move with it."""
    card = load("rtx-pro-6000-server")
    assert value(card, "store.reg_extra") == 80 * 48 + 336 + 11 == 4187
    sms = value(card, "sm.count")
    words = value(card, "store.reg") + value(card, "store.reg_extra")
    assert 4 * sms * words + sms * value(card, "store.smem") + value(card, "store.l2") == 211_290_960


@pytest.mark.parametrize("sku", skus("gpu"))
def test_shared_memory_alone_is_part_of_its_pool(sku: str) -> None:
    """`store.smem_carveout` is the most of `store.smem`, the pool of L1 and shared memory, that a kernel can make
    shared memory, so it is no larger than the pool."""
    card = load(sku)
    assert 0 < value(card, "store.smem_carveout") <= value(card, "store.smem")


@pytest.mark.parametrize("sku", skus("gpu"))
def test_the_memory_on_the_board_bounds_the_memory_a_program_gets(sku: str) -> None:
    """`store.dram_installed` is the second reading of device memory, the memory on the board: a ceiling in whole GiB,
    at or above `store.dram`, the memory a program gets."""
    card = load(sku)
    installed = value(card, "store.dram_installed")
    assert card["store"]["dram_installed"]["mark"] == "ceiling" and card["store"]["dram_installed"]["unit"] == "B"
    assert 0 < value(card, "store.dram") <= installed and installed % 2 ** 30 == 0


def test_the_caches_a_run_found_are_one_cards() -> None:
    """The caches an SM serves code and constants from are on the one card a run swept, each in bytes an SM, and the
    two counted once a warp scheduler are whole multiples of `rate.issue`'s schedulers."""
    found = MEASURED_ONLY["gpu"]
    have = {sku: {key for key in entries(load(sku)) if key in found} for sku in skus("gpu")}
    assert have == {"b200-sxm-180gb": set(), "h200-sxm-141gb": set(), "rtx-pro-6000-server": found}
    card = load("rtx-pro-6000-server")
    assert {entries(card)[key]["unit"] for key in found} == {"B/SM"}
    assert [value(card, f"store.{k}") / value(card, "rate.issue") for k in ("l0i", "const_l0")] == [6144, 3072]


@pytest.mark.parametrize("sku", skus("gpu"))
def test_an_add_in_a_loop_costs_no_less_than_the_add_alone(sku: str) -> None:
    """`price.fp32_loop` is an FP32 add's price in a timed loop, so it is at least the add's price at the card's
    ceilings: a SM-clock's tensor units over the adds it issues, 32 lanes a warp instruction."""
    card = load(sku)
    assert value(card, "price.fp32_loop") >= value(card, "rate.tensor") / (32 * value(card, "rate.issue"))


@pytest.mark.parametrize("sku", skus("host"))
def test_derived_host_numbers(sku: str) -> None:
    """A host's ops are its cores' AMX MACs at their clocks, two ops a MAC; its DRAM's bandwidth is its channels'
    transfers, which bound its reads and are its write ceiling; the second reading of the three, at the fastest memory
    the processor takes, is a ceiling at or above the first; and its on-chip bytes are its caches and allowances."""
    card = load(sku)
    sockets, cores, priority = (value(card, f"cpu.{k}") for k in ("sockets", "cores", "priority_cores"))
    hz = priority * value(card, "clock.priority_turbo") + (cores - priority) * value(card, "clock.turbo")
    assert value(card, "rate.ops") == 2 * value(card, "rate.amx") * sockets * hz * 10 ** 9
    channels = sockets * value(card, "memory.channels") * 10 ** 6 * value(card, "memory.bus") / 8
    bandwidth = value(card, "memory.bandwidth")
    assert bandwidth == channels * value(card, "memory.speed")
    assert value(card, "memory.read") <= bandwidth == value(card, "memory.write")
    most = value(card, "memory.bandwidth_max")
    assert most == channels * value(card, "memory.speed_max") == value(card, "memory.write_max")
    assert value(card, "memory.speed") <= value(card, "memory.speed_max") and bandwidth <= most
    assert {card["memory"][k]["mark"] for k in ("speed_max", "bandwidth_max", "write_max")} == {"ceiling"}
    per_socket = value(card, "store.l3") + value(card, "store.uncore")
    per_core = sum(value(card, f"store.{k}") for k in ("l2", "l1d", "l1i", "core"))
    assert value(card, "store.chip") == sockets * per_socket + sockets * cores * per_core


@pytest.mark.parametrize("sku", skus("host"))
def test_a_vm_gives_a_gpu_no_more_cores_than_the_server_has(sku: str) -> None:
    """`cpu.vm_cores`, the cores a GPU that a node's VM gives, is a measured share of the server's cores, so it is a
    measurement and no larger than them."""
    card = load(sku)
    assert card["cpu"]["vm_cores"]["mark"] == "measurement"
    assert 0 < value(card, "cpu.vm_cores") <= value(card, "cpu.sockets") * value(card, "cpu.cores")


@pytest.mark.parametrize("sku", skus("server"))
def test_derived_server_numbers(sku: str) -> None:
    """A server's line rate is its network ports' together, every class of them, each in bits a second."""
    link = load(sku)["link"]
    assert {e["unit"] for e in link.values()} == {"bit/s"}
    assert value(link, "rate") == sum(value(link, k) for k in link if k != "rate")


def test_the_quoted_numbers() -> None:
    """The clocks, host peaks, bandwidths and measured reads already quoted outside this package, pinned so that a
    change to one is deliberate: among them remote_bandwidth/timing.py's HOST_ figures, which the RTX node's host card
    states beside the disks' read that PoUB's verifier rounds up, and its GPU's measured read, which the RTX PRO 6000's
    card states."""
    server, h200, b200 = load("rtx-pro-6000-server"), load("h200-sxm-141gb"), load("b200-sxm-180gb")
    clocks = ("clock.sm", "clock.tensor", "clock.memory")
    assert [value(server, k) for k in clocks] == [2430, 2430, 12481]
    assert [value(h200, k) for k in clocks] == [1980, 1830, 3201]
    assert [value(b200, k) for k in clocks] == [1965, 1965, 3996]
    assert value(server, "memory.bandwidth") == 1597568 * 10 ** 6
    assert value(h200, "memory.bandwidth") == 4916736 * 10 ** 6
    assert [float(value(card, "memory.read")) for card in (server, h200)] == [1.539e12, 4.26e12]
    granite, sapphire = load("xeon-6776p-2s"), load("xeon-platinum-8468-2s")
    assert round(float(value(granite, "rate.ops")), -11) == 1.0453e15
    assert round(float(value(sapphire, "rate.ops")), -11) == 7.471e14
    timing = ("memory.read", "memory.bandwidth", "memory.write", "store.chip")
    assert [float(value(granite, k)) for k in timing] == [606.8e9, 819.2e9, 819.2e9, 1_000_341_504]
    assert float(value(granite, "disk.read")) == 4.311e9


def test_the_second_readings() -> None:
    """The second reading of each device's memory and of each host's DRAM, pinned so that a change to one is
    deliberate: the memory on a GPU's board beside the memory a program gets, and a host's channels at the fastest
    memory its processor takes beside the figures PoUB's timing mirrors, which stay where they were; and the caches a
    run found on the RTX PRO 6000."""
    gib = 2 ** 30
    boards = {"rtx-pro-6000-server": (96, 96), "h200-sxm-141gb": (141, 144), "b200-sxm-180gb": (180, 192)}
    for sku, (dram, installed) in boards.items():
        card = load(sku)
        assert [value(card, "store.dram"), value(card, "store.dram_installed")] == [dram * gib, installed * gib], sku
    second = ("memory.speed_max", "memory.bandwidth_max", "memory.write_max")
    assert [float(value(load("xeon-6776p-2s"), k)) for k in second] == [8800, 1.1264e12, 1.1264e12]
    assert [float(value(load("xeon-platinum-8468-2s"), k)) for k in second] == [4800, 6.144e11, 6.144e11]
    caches = ("store.l15", "store.l1i", "store.l0i", "store.const_l0")
    assert [value(load("rtx-pro-6000-server"), k) for k in caches] == [147_456, 65_536, 24_576, 12_288]


def test_the_rates_readers_quote() -> None:
    """`rate`, the one calculator of the peaks the benchmarks quote: the tensor peak in s8 MACs and in another format's
    MACs, at the tensor's own clock (the H200's 1830 MHz, under its 1980 MHz SM clock), at the SM clock on a card
    without a `clock.tensor`, and at a clock a reader names; every other resource at the SM clock, and device memory's
    words a second."""
    server, h200 = load("rtx-pro-6000-server"), load("h200-sxm-141gb")
    assert cards.rate(server) == 188 * 1024 * 2430 * 10 ** 6
    assert cards.rate(h200) == 132 * 4096 * 1830 * 10 ** 6
    assert cards.rate(h200, fmt="bf16") == 132 * 4096 * 1830 * 10 ** 6 / 2
    assert cards.rate(h200, "issue") == 132 * 4 * 1980 * 10 ** 6
    untimed = copy.deepcopy(h200)
    del untimed["clock"]["tensor"]
    assert cards.rate(untimed) == 132 * 4096 * 1980 * 10 ** 6
    assert cards.rate(untimed, "issue") == 132 * 4 * 1980 * 10 ** 6
    assert cards.rate(server, "issue", mhz=2092) == 4 * 188 * 2092 * 10 ** 6
    assert cards.rate(server, fmt="nvf4", mhz=2092.5) == 188 * 1024 * Fraction("2092.5") * 10 ** 6 * 2
    assert cards.rate(h200, "dram") == 4916736 * 10 ** 6 / 4
    for call, refusal in ((lambda: cards.rate(load("xeon-6776p-2s")), "is a host card"),
                          (lambda: cards.rate(h200, "dram", mhz=1500), "no SM clock"),
                          (lambda: cards.rate(h200, "tensr"), "no rate.tensr"),
                          (lambda: cards.rate(h200, "issue", fmt="s8"), "weighs tensor MACs only"),
                          (lambda: cards.rate(h200, fmt="fp8"), "weighs tensor MACs only")):
        with pytest.raises((KeyError, ValueError), match=refusal):
            call()


def test_an_unknown_sku_names_the_cards() -> None:
    every = ["b200-sxm-180gb", "dgx-h200", "h200-sxm-141gb", "rtx-pro-6000-server", "xeon-6776p-2s",
             "xeon-platinum-8468-2s"]
    with pytest.raises(KeyError, match=re.escape(f"the cards are {every}")):
        load("rtx-pro-6000-workstation")


@pytest.mark.parametrize(("body", "refusal"), [
    ('sku = "x"\nkind = "gpu"\n[sm]\ncount = 3\n', "sm: count is a bare number"),
    ('sku = "x"\nkind = "gpu"\n[sm]\ncount = { value = 3, unit = "SM", mark = "guess", source = "s" }\n',
     "mark 'guess'"),
    ('sku = "x"\nkind = "gpu"\n[sm]\ncount = { value = 3, unit = "SM", mark = "ceiling", source = " " }\n',
     "give it a source"),
    ('sku = "x"\nkind = "gpu"\n[sm]\ncount = { value = 3, unit = "SM", mark = "ceiling" }\n', "an entry has exactly"),
    ('sku = "x"\nkind = "gpu"\n[sm]\ncount = { value = "3", unit = "SM", mark = "ceiling", source = "s" }\n',
     "is not a number"),
    ('sku = "y"\nkind = "gpu"\n', "set it to the file's name"),
    ('sku = "x"\n', "kind = None; set it to one of"),
    ('sku = "x"\nkind = "cpu"\n', "kind = 'cpu'"),
])
def test_an_unsourced_or_unmarked_number_is_refused(tmp_path, monkeypatch, body: str, refusal: str) -> None:
    (tmp_path / "x.toml").write_text(body, encoding="utf-8")
    monkeypatch.setattr(cards, "CARDS", tmp_path)
    with pytest.raises(InvalidArtifact, match=refusal):
        load("x")
