"""The device pins (`verity_vllm/pins/`): the cc 9.0 device's probed cuBLAS routes are the eager Build's default routes off the
GLM-5.3 capture's shapes, a route is never read for an M the probe did not run, a pin is found by its device's reported name, and
the SM count a served pin's Build reads from its device's pin is the card's."""
from __future__ import annotations

import json

import pytest

from verity import hardware
from verity.ml.definitions.registry import bf16_gemm_check as BC
from verity_vllm.config import TARGET_CARDS, TARGETS
from verity_vllm.program.predict import compiled


def test_the_probed_routes_are_the_default_routes_at_their_m_only():
    assert BC.default_cublas_route("linear", 256, 2048, 5120) == ("non_split", {})
    assert BC.default_cublas_route("linear", 1, 4096, 2048) == ("splitk", {"SPLIT_K": 3})
    assert BC.default_cublas_route("linear", 5, 2048, 5120) is None
    assert BC.default_cublas_route("router", 1, 2048, 128) is None


def test_the_capture_pins_come_first():
    assert BC.default_cublas_route("linear", 1, 6144, 160) == ("splitk", {"SPLIT_K": 4})


def test_a_pin_is_found_by_the_name_its_device_reports():
    pin = BC.device_pin(BC.capability_pin((9, 0))["device"])
    assert tuple(pin["compute_capability"]) == (9, 0) and pin["sms"] > 0
    with pytest.raises(KeyError):
        BC.device_pin("no such device")
    with pytest.raises(KeyError):
        BC.capability_pin((1, 0))


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
