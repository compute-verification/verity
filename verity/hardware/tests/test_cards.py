"""The card files: every number sourced and marked, one schema across cards, and the numbers derived from others."""

from __future__ import annotations

from fractions import Fraction

import pytest

import verity.hardware as hardware
from verity.errors import InvalidArtifact
from verity.hardware import MARKS, load, skus


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
    assert skus() == ["h200-sxm-141gb", "rtx-pro-6000-server"]


@pytest.mark.parametrize("sku", skus())
def test_every_number_is_sourced_and_marked(sku: str) -> None:
    card = load(sku)
    numbers = entries(card)
    assert len(numbers) >= 40
    assert all(e["mark"] in MARKS and e["source"].strip() for e in numbers.values())


def test_one_schema_across_cards() -> None:
    """A reader can take any number from any card by the same key."""
    keys = {sku: sorted(entries(load(sku))) for sku in skus()}
    assert len({tuple(k) for k in keys.values()}) == 1, keys


@pytest.mark.parametrize("sku", skus())
def test_derived_numbers(sku: str) -> None:
    """Device memory's bandwidth is its clock's transfers over the bus, and the host's ops are its cores' AMX MACs at
    their clocks, two ops a MAC."""
    card = load(sku)
    assert value(card, "memory.bandwidth") == (value(card, "clock.memory") * 10 ** 6 * value(card, "memory.transfers")
                                               * value(card, "memory.bus") / 8)
    cores, priority = value(card, "host.cores"), value(card, "host.priority_cores")
    hz = priority * value(card, "host.priority_turbo") + (cores - priority) * value(card, "host.turbo")
    assert value(card, "host.ops") == 2 * value(card, "host.amx") * value(card, "host.sockets") * hz * 10 ** 9


def test_the_quoted_numbers() -> None:
    """The firmware ceilings, host peaks and bandwidths already quoted outside this package, pinned so that a change to
    one is deliberate."""
    server, h200 = load("rtx-pro-6000-server"), load("h200-sxm-141gb")
    assert [value(server, k) for k in ("clock.sm", "clock.memory", "power.limit")] == [2430, 12481, 600]
    assert [value(h200, k) for k in ("clock.sm", "clock.memory", "power.limit")] == [1980, 3201, 700]
    assert round(float(value(server, "host.ops")), -11) == 1.0453e15
    assert round(float(value(h200, "host.ops")), -11) == 7.471e14
    assert value(server, "memory.bandwidth") == 1597568 * 10 ** 6
    assert value(h200, "memory.bandwidth") == 4916736 * 10 ** 6


def test_an_unknown_sku_names_the_cards() -> None:
    with pytest.raises(KeyError, match=r"the cards are \['h200-sxm-141gb', 'rtx-pro-6000-server'\]"):
        load("rtx-pro-6000-workstation")


@pytest.mark.parametrize(("body", "refusal"), [
    ('sku = "x"\n[sm]\ncount = 3\n', "sm: count is a bare number"),
    ('sku = "x"\n[sm]\ncount = { value = 3, unit = "SM", mark = "guess", source = "s" }\n', "mark 'guess'"),
    ('sku = "x"\n[sm]\ncount = { value = 3, unit = "SM", mark = "ceiling", source = " " }\n', "give it a source"),
    ('sku = "x"\n[sm]\ncount = { value = 3, unit = "SM", mark = "ceiling" }\n', "an entry has exactly"),
    ('sku = "x"\n[sm]\ncount = { value = "3", unit = "SM", mark = "ceiling", source = "s" }\n', "is not a number"),
    ('sku = "y"\n', "set it to the file's name"),
])
def test_an_unsourced_or_unmarked_number_is_refused(tmp_path, monkeypatch, body: str, refusal: str) -> None:
    (tmp_path / "x.toml").write_text(body, encoding="utf-8")
    monkeypatch.setattr(hardware, "CARDS", tmp_path)
    with pytest.raises(InvalidArtifact, match=refusal):
        load("x")
