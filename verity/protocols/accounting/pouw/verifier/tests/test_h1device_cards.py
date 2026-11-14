"""`circuit.h1device`'s records state each device's SMs, clocks and dense s8 MACs as literals, and the hardware cards
(`verity.device.load_card`) state the same numbers: each literal equals its card's entry."""
from __future__ import annotations

import pytest

from verity.device import load_card
from verity.protocols.accounting.pouw.prover.circuit import h1device as D

#: a record, by its architecture, and the SKU whose card states its numbers
SKU = {"sm_120": "rtx-pro-6000-server", "sm_90": "h200-sxm-141gb"}

#: a record's field and the card entry it equals
ENTRIES = {
    "sms": ("sm", "count"),
    "tensor_mhz": ("clock", "tensor"),
    "fmax_mhz": ("clock", "sm"),
    "s8_macs": ("rate", "tensor"),
}


@pytest.mark.parametrize("arch", sorted(SKU))
def test_a_records_numbers_are_its_cards(arch) -> None:
    """The card is the source: where one differs, the record's literal is the one to change."""
    record, card = D.DEVICES[arch], load_card(SKU[arch])
    assert card["arch"] == f"{arch}a"
    for field, (table, key) in ENTRIES.items():
        entry = card[table][key]
        assert getattr(record, field) == entry["value"], (
            f"h1device's {arch} record has {field} = {getattr(record, field)}, and {SKU[arch]}.toml's {table}.{key} "
            f"is {entry['value']} {entry['unit']}: set the record's {field} to the card's")
