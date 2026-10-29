"""The SM count a served pin's Build reads from its device's pin is the card's: each row class `config.TARGET_CARDS` lists takes
its SM count from its card (`verity.hardware`), and the device pin of every served pin at that class's compute capability records
the same count."""
from __future__ import annotations

import json

from verity import hardware
from verity.ml.definitions.registry import bf16_gemm_check as BC
from verity_vllm.config import TARGET_CARDS, TARGETS
from verity_vllm.program.predict import compiled


def test_every_served_pin_s_device_pin_records_its_row_class_s_card_sm_count():
    """A row class `config.TARGET_CARDS` lists takes its SM count from its card, and the pin of the device a served pin names, at
    that class's compute capability, records the same count: every listed class has such a pin."""
    cards = {TARGETS[t]: hardware.load(sku)["sm"]["count"]["value"] for t, sku in TARGET_CARDS.items()}
    held = set()
    for path in (p for p in compiled.PINS.iterdir() if p.name.endswith(".json")):
        pin = BC.device_pin(json.loads(path.read_text())["device"])
        cc = tuple(pin["compute_capability"])
        if cc in cards:
            assert pin["sms"] == cards[cc], f"{path.name}: {pin['device']}'s pin records {pin['sms']} SMs, its card {cards[cc]}"
            held.add(cc)
    assert held == set(cards), f"no served pin on the classes of compute capability {sorted(set(cards) - held)}"
