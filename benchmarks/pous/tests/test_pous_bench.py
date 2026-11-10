"""The POUS benchmark's dataset and harness (`benchmarks/pous`): the weight sets are the frozen benchmark's, every
implementation is registered, and a certificate gate refuses what the scheme's certificate doesn't cover."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "benchmarks" / "pous"


def _bench():
    spec = importlib.util.spec_from_file_location("pous_bench", BENCH / "bench.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_weight_sets_are_the_frozen_benchmarks():
    doc = json.loads((BENCH / "weight_sets.json").read_text())
    assert doc["schema"] == "pous-weight-sets/v1"
    sets = {ws["id"]: ws for ws in doc["weight_sets"]}
    blocks = {i: -(-ws["bytes"] // 65_536) for i, ws in sets.items()}
    assert blocks == {"qwen2.5-0.5b/linears+head": 15_075, "qwen2.5-7b/layer0+head": 23_744, "llama3.1-8b/layer0+head": 22_688,
                      "llama3.1-8b/layers0-8+head": 75_936}
    for ws in sets.values():
        assert ws["bytes"] == sum(o * k * 2 for _, o, k in ws["tensors"])
        assert all(len(s) == 64 for s in ws["checkpoint"]["safetensors_sha256"].values())
    hardware = {h["id"] for h in json.loads((ROOT / "benchmarks" / "census" / "hardware.json").read_text())}
    assert set(doc["hardware"]) <= hardware


def test_harness_registry_and_certificate_gate():
    pytest.importorskip("verity.core.protocols.pous")
    from verity.core.protocols.pous import schemes

    bench = _bench()
    assert set(bench.IMPLS) == {"reference", "vllm-gpu", "band-gpu", "p2-v1-gpu", "p2-v3-gpu", "p2-v3-mc-gpu"}
    assert set(bench.VllmGpu.MODELS) <= {ws["id"] for ws in json.loads((BENCH / "weight_sets.json").read_text())["weight_sets"]}
    one = schemes.get()
    gate, gaps = bench.certificate_gate(one, 111, 536_800)
    assert gate["passed"] and gate["effective_rounds"] == 2 and gaps == ()
    gate, gaps = bench.certificate_gate(one, 111, 3_000_000)
    assert not gate["passed"] and "covers 5" in gate["refusals"][0]
    _, gaps = bench.certificate_gate(schemes.get(segments=30), 111, 536_800)
    assert gaps == ()                                       # BandMultiMeetsFamily covers every segment count
    _, gaps = bench.certificate_gate(schemes.get("dense-chain/v1", segments=30), 111, 536_800)
    assert gaps == ("W has 30 segments; Pous.SecurityProofs.ChainDenseMeets64 is about 1",)
    assert bench.conformance_gate("band-chain/d12/v1")["passed"]
