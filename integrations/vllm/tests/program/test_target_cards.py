"""The SM counts vLLM declares for a row class are its card's (`verity.hardware`): each class in `config.TARGET_NUM_SMS` has the SM
count its card states, and the device pin of every served pin at that class's compute capability records the same count. The
counts stay literals in config: the pipeline's top-level process, which declares a row's Build target, has no `verity` on its path."""
from __future__ import annotations

import json

from verity import hardware
from verity.ml.definitions.registry import bf16_gemm_check as BC
from verity_vllm.config import TARGET_NUM_SMS, TARGETS
from verity_vllm.program.predict import compiled

#: each `config.TARGET_NUM_SMS` class's card: the H200 NVL has the SXM's SM count, and every RTX PRO 6000 edition the Server Edition's
CARDS = {"rtxpro6000": "rtx-pro-6000-server", "h200": "h200-sxm-141gb"}


def test_each_row_class_s_sm_count_is_its_card_s():
    """`config.TARGET_NUM_SMS` lists exactly `CARDS`' classes, each at the SM count its card states."""
    assert set(TARGET_NUM_SMS) == set(CARDS)
    for t, sku in CARDS.items():
        assert TARGET_NUM_SMS[t] == hardware.load(sku)["sm"]["count"]["value"], f"{t}: config says {TARGET_NUM_SMS[t]}, card {sku}"


def test_every_served_pin_s_device_pin_records_its_row_class_s_card_sm_count():
    """The pin of the device a served pin names, at a listed class's compute capability, records the class's card SM count: every
    listed class has such a pin."""
    cards = {TARGETS[t]: hardware.load(sku)["sm"]["count"]["value"] for t, sku in CARDS.items()}
    held = set()
    for path in (p for p in compiled.PINS.iterdir() if p.name.endswith(".json")):
        pin = BC.device_pin(json.loads(path.read_text())["device"])
        cc = tuple(pin["compute_capability"])
        if cc in cards:
            assert pin["sms"] == cards[cc], f"{path.name}: {pin['device']}'s pin records {pin['sms']} SMs, its card {cards[cc]}"
            held.add(cc)
    assert held == set(cards), f"no served pin on the classes of compute capability {sorted(set(cards) - held)}"
