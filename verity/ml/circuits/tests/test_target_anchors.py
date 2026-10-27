"""Each target's ``anchor_device`` is a SKU id, and one with a card file (``verity.hardware``) is that card's, of the
instruction's architecture: the card is the one source of the device's numbers, and a device name never stands in for it."""
from __future__ import annotations

import re

from verity.hardware import load, skus
from verity.ml.circuits.silicon.instructions import INSTRUCTIONS
from verity.ml.circuits.targets.target import FP8_SM120_MMA, TARGETS


def test_every_anchor_is_a_sku_id() -> None:
    for t in TARGETS.values():
        assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", t.anchor_device), (
            f"{t.name}: anchor_device {t.anchor_device!r} is not a SKU id; name the card file's SKU "
            f"(one of {skus()}), else the census line's, as in 'a100-sxm4-80gb'")


def test_a_carded_anchor_is_its_cards_sku_on_the_instructions_arch() -> None:
    carded = [t for t in TARGETS.values() if t.anchor_device in skus()]
    assert FP8_SM120_MMA in carded
    for t in carded:
        arch = load(t.anchor_device)["arch"]
        assert re.fullmatch(rf"{INSTRUCTIONS[t.instruction].sm}a?", arch), (t.name, arch, t.instruction)
