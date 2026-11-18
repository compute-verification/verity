from __future__ import annotations

import re

import pytest

from conftest import load

H200, H100, UNKNOWN = "NVIDIA H200", "NVIDIA H100 80GB HBM3", "NVIDIA A100-SXM4-80GB"
TAG = "s8-coord-k-2048"


@pytest.mark.parametrize("model, card, member", [
    ("synthetic", H200, "synthetic-s8-coord-k-2048-h200"), ("synthetic", H100, "synthetic-s8-coord-k-2048-h100"),
    ("synthetic", "none", "synthetic-s8-coord-k-2048-cpu"), ("llama-3.2-1b", H200, "llama-3.2-1b-h200-s8-coord-k-2048"),
    ("llama-3.2-1b", H100, "llama-3.2-1b-h100-s8-coord-k-2048"), ("llama-3.2-1b", "none", "llama-3.2-1b-s8-coord-k-2048")])
def test_a_member_names_the_card_records_short(monkeypatch, model, card, member) -> None:
    r = load(monkeypatch, "record.py")
    assert r.member(model, TAG, r.machine(card)) == member


def test_a_card_no_record_names_is_refused_by_name(monkeypatch) -> None:
    r = load(monkeypatch, "record.py")
    with pytest.raises(SystemExit, match=re.escape(repr(UNKNOWN))):
        r.machine(UNKNOWN)
