"""A row's target is its card file (`config.target_card`): every target resolves through the card reader, a token with no card file
is refused by name, the device pin of every served pin at a carded class's compute capability records its card's SM count, and the
adapter's library holds no table of its own keyed by card names."""
from __future__ import annotations

import ast
import json

import pytest

from verity_vllm.config import ROOT, UsageError, target_card, target_cc, target_num_sms, target_tokens
from verity_vllm.definitions.registry import bf16_gemm_check as BC
from verity_vllm.program.predict import compiled

#: the classes with served pins
SERVED = ("rtxpro6000", "h200")


def _card_names() -> set[str]:
    cards = [target_card(t) for t in target_tokens()]
    return {*target_tokens(), *(c["short"] for c in cards), *(c["sku"] for c in cards)}


def test_every_target_is_a_card_file_s_and_an_uncarded_one_is_refused_by_name():
    assert {"rtxpro6000", "h200", "h100", "b200"} <= set(target_tokens())
    assert (target_cc("rtxpro6000"), target_num_sms("rtxpro6000")) == ((12, 0), 188)
    assert (target_cc("h200"), target_num_sms("h200")) == ((9, 0), 132)
    with pytest.raises(UsageError, match=r"row target 'l40s': no GPU card file names it"):
        target_card("l40s")


def test_every_served_pin_s_device_pin_records_its_row_class_s_card_sm_count():
    """The pin of the device a served pin names, at a served class's compute capability, records the class's card SM count: every
    served class has such a pin."""
    cards = {target_cc(t): target_num_sms(t) for t in SERVED}
    held = set()
    for path in (p for p in compiled.PINS.iterdir() if p.name.endswith(".json")):
        pin = BC.device_pin(json.loads(path.read_text())["device"])
        cc = tuple(pin["compute_capability"])
        if cc in cards:
            assert pin["sms"] == cards[cc], f"{path.name}: {pin['device']}'s pin records {pin['sms']} SMs, its card {cards[cc]}"
            held.add(cc)
    assert held == set(cards), f"no served pin on the classes of compute capability {sorted(set(cards) - held)}"


def test_the_adapter_holds_no_table_keyed_by_card_names():
    """No dict in `verity_vllm` is keyed by a card's name (its target token, `short` or SKU), and no tuple, list or set lists two:
    a fact of a card is read from its card file (`config.target_card`)."""
    names, found = _card_names(), []
    for path in sorted((ROOT / "verity_vllm").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(), str(path))):
            keys = node.keys if isinstance(node, ast.Dict) else node.elts if isinstance(node, (ast.Tuple, ast.List, ast.Set)) else []
            hit = {k.value for k in keys if isinstance(k, ast.Constant) and k.value in names}
            if hit and (isinstance(node, ast.Dict) or len(hit) > 1):
                found.append(f"{path.relative_to(ROOT)}:{node.lineno} {sorted(hit)}")
    assert not found, f"tables keyed by card names, which config.target_card reads from the card files: {found}"
