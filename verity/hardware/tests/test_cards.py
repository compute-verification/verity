"""The card files: every number sourced and marked, one schema across the cards of a kind, and the numbers derived from
others."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

import verity.hardware as hardware
from verity.errors import InvalidArtifact
from verity.hardware import KINDS, MARKS, load, skus


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
    assert skus() == sorted(skus("gpu") + skus("host"))


@pytest.mark.parametrize("sku", skus())
def test_every_number_is_sourced_and_marked(sku: str) -> None:
    card = load(sku)
    numbers = entries(card)
    assert len(numbers) >= 20
    assert all(e["mark"] in MARKS and e["source"].strip() for e in numbers.values())


@pytest.mark.parametrize("sku", skus())
def test_sources_cite_the_store_not_runs_or_hosts(sku: str) -> None:
    """Evidence is cited by its `art:` id, and a node by its kind, never by a run id or a host name."""
    text = (hardware.CARDS / f"{sku}.toml").read_text(encoding="utf-8")
    assert not re.findall(r"\br\d{8}-\d{6}-[0-9a-f]{4}\b|\bvy-[a-z0-9-]+", text), \
        "cite the run's record (`research data show RUN`: outputs.run_record) as art:<12 hex>, and name the node by its kind"


@pytest.mark.parametrize("kind", KINDS)
def test_one_schema_across_the_cards_of_a_kind(kind: str) -> None:
    """A reader can take any number from any card of a kind by the same key."""
    keys = {sku: sorted(entries(load(sku))) for sku in skus(kind)}
    assert len({tuple(k) for k in keys.values()}) == 1, keys


@pytest.mark.parametrize("sku", skus("gpu"))
def test_derived_gpu_numbers(sku: str) -> None:
    """Device memory's bandwidth is its clock's transfers over the bus."""
    card = load(sku)
    assert value(card, "memory.bandwidth") == (value(card, "clock.memory") * 10 ** 6 * value(card, "memory.transfers")
                                               * value(card, "memory.bus") / 8)


@pytest.mark.parametrize("sku", skus("host"))
def test_derived_host_numbers(sku: str) -> None:
    """A host's ops are its cores' AMX MACs at their clocks, two ops a MAC; its DRAM's bandwidth is its channels'
    transfers, which bound its reads and are its write ceiling; and its on-chip bytes are its caches and allowances."""
    card = load(sku)
    sockets, cores, priority = (value(card, f"cpu.{k}") for k in ("sockets", "cores", "priority_cores"))
    hz = priority * value(card, "clock.priority_turbo") + (cores - priority) * value(card, "clock.turbo")
    assert value(card, "rate.ops") == 2 * value(card, "rate.amx") * sockets * hz * 10 ** 9
    bandwidth = value(card, "memory.bandwidth")
    assert bandwidth == (sockets * value(card, "memory.channels") * value(card, "memory.speed") * 10 ** 6
                         * value(card, "memory.bus") / 8)
    assert value(card, "memory.read") <= bandwidth == value(card, "memory.write")
    per_socket = value(card, "store.l3") + value(card, "store.uncore")
    per_core = sum(value(card, f"store.{k}") for k in ("l2", "l1d", "l1i", "core"))
    assert value(card, "store.chip") == sockets * per_socket + sockets * cores * per_core


def test_the_quoted_numbers() -> None:
    """The firmware ceilings, host peaks and bandwidths already quoted outside this package, pinned so that a change to
    one is deliberate: among them remote_bandwidth/timing.py's HOST_ figures, which the RTX node's host card states."""
    server, h200, b200 = load("rtx-pro-6000-server"), load("h200-sxm-141gb"), load("b200-sxm-180gb")
    clocks = ("clock.sm", "clock.tensor", "clock.memory")
    assert [value(server, k) for k in clocks] == [2430, 2430, 12481]
    assert [value(h200, k) for k in clocks] == [1980, 1830, 3201]
    assert [value(b200, k) for k in clocks] == [1965, 1965, 3996]
    assert value(server, "memory.bandwidth") == 1597568 * 10 ** 6
    assert value(h200, "memory.bandwidth") == 4916736 * 10 ** 6
    granite, sapphire = load("xeon-6776p-2s"), load("xeon-platinum-8468-2s")
    assert round(float(value(granite, "rate.ops")), -11) == 1.0453e15
    assert round(float(value(sapphire, "rate.ops")), -11) == 7.471e14
    timing = ("memory.read", "memory.bandwidth", "memory.write", "store.chip")
    assert [float(value(granite, k)) for k in timing] == [606.8e9, 819.2e9, 819.2e9, 1_000_341_504]


def test_the_rates_readers_quote() -> None:
    """`rate`, the one calculator of the peaks the benchmarks quote: the tensor peak in s8 MACs and in another format's
    MACs, at the tensor's ceiling clock (the H200's 1830 MHz, under its 1980 MHz SM clock) and at a clock a reader
    names, every other resource at the SM clock, and device memory's words a second."""
    server, h200 = load("rtx-pro-6000-server"), load("h200-sxm-141gb")
    assert hardware.rate(server) == 188 * 1024 * 2430 * 10 ** 6
    assert hardware.rate(h200, fmt="bf16") == 132 * 4096 * 1830 * 10 ** 6 / 2
    assert hardware.rate(h200, "issue") == 132 * 4 * 1980 * 10 ** 6
    assert hardware.rate(server, "issue", mhz=2092) == 4 * 188 * 2092 * 10 ** 6
    assert hardware.rate(server, fmt="nvf4", mhz=2092.5) == 188 * 1024 * Fraction("2092.5") * 10 ** 6 * 2
    assert hardware.rate(h200, "dram") == 4916736 * 10 ** 6 / 4
    for call, refusal in ((lambda: hardware.rate(load("xeon-6776p-2s")), "is a host card"),
                          (lambda: hardware.rate(h200, "dram", mhz=1500), "no SM clock"),
                          (lambda: hardware.rate(h200, "tensr"), "no rate.tensr"),
                          (lambda: hardware.rate(h200, "issue", fmt="s8"), "weighs tensor MACs only"),
                          (lambda: hardware.rate(h200, fmt="fp8"), "weighs tensor MACs only")):
        with pytest.raises((KeyError, ValueError), match=refusal):
            call()


def test_an_unknown_sku_names_the_cards() -> None:
    cards = ["b200-sxm-180gb", "h200-sxm-141gb", "rtx-pro-6000-server", "xeon-6776p-2s", "xeon-platinum-8468-2s"]
    with pytest.raises(KeyError, match=re.escape(f"the cards are {cards}")):
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
    monkeypatch.setattr(hardware, "CARDS", tmp_path)
    with pytest.raises(InvalidArtifact, match=refusal):
        load("x")
