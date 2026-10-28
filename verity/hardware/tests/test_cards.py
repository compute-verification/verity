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
    assert [value(server, k) for k in ("clock.sm", "clock.memory", "power.limit")] == [2430, 12481, 600]
    assert [value(h200, k) for k in ("clock.sm", "clock.memory", "power.limit")] == [1980, 3201, 700]
    assert [value(b200, k) for k in ("clock.sm", "clock.memory", "power.limit")] == [1965, 3996, 1000]
    assert value(server, "memory.bandwidth") == 1597568 * 10 ** 6
    assert value(h200, "memory.bandwidth") == 4916736 * 10 ** 6
    granite, sapphire = load("xeon-6776p-2s"), load("xeon-platinum-8468-2s")
    assert round(float(value(granite, "rate.ops")), -11) == 1.0453e15
    assert round(float(value(sapphire, "rate.ops")), -11) == 7.471e14
    timing = ("memory.read", "memory.bandwidth", "memory.write", "store.chip")
    assert [float(value(granite, k)) for k in timing] == [606.8e9, 819.2e9, 819.2e9, 1_000_341_504]


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
