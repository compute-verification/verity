"""The serving profiles (`verity_bench.profiles`, `benchmarks/production.toml`): each profile's version pinned with its
digest, the file refusing what it doesn't declare, its datasets and bounds equal to serve_tax's for the same load, and
each model's load read on one card. No engine."""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from verity_bench import profiles as P
from verity_bench.vllm.presets import VLLM_PIN
from verity_config import ConfigError

SERVE_TAX = Path(__file__).resolve().parents[3] / "benchmarks" / "serve_tax" / "serve_tax.py"
PINS = {
    "coding-agent": (2, "fa839372a8da13a51ea65c76d5a74b0a0a8f39cadc1eabb2b8d96364930e61a6"),
    "chat": (1, "553bc90ac63b177fcd651c58798f764d7e30f26f57dee79688a92b613e3c2a40"),
    "fixed-size": (1, "93036f84deab9449aa7d8e4f1343c9f23a1918bed489dc5efbcb036d51e1cbc4"),
}


@pytest.fixture(scope="module")
def S():
    if "serve_tax" not in sys.modules:
        spec = importlib.util.spec_from_file_location("serve_tax", SERVE_TAX)
        sys.modules["serve_tax"] = module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return sys.modules["serve_tax"]


@pytest.fixture(scope="module")
def doc():
    return P.load()


def test_each_profiles_version_rises_with_any_change(doc):
    for name, (version, pinned) in PINS.items():
        profile, now = doc.profiles[name], P.digest(doc, name)
        if now != pinned and profile.version == version:
            pytest.fail(f"{name} changed but its version is still {version}: set version = {version + 1} on "
                        f"[profiles.{name}] in benchmarks/production.toml and pin it here as ({version + 1}, {now!r})")
        assert (profile.version, now) == (version, pinned), (
            f"{name} is now version {profile.version} with digest {now!r}: pin it here as ({profile.version}, {now!r})")


def test_objects_are_frozen(doc):
    with pytest.raises(FrozenInstanceError):
        doc.profiles["chat"].prompts = 1
    with pytest.raises(TypeError):
        doc.profiles["coding-agent"].users["glm53"]["h200"] = 8


@pytest.mark.parametrize(("table", "line"), [
    ("[client]", "retries = 2"),
    ("[profiles.chat]", "temperature = 0.0"),
    ("[profiles.coding-agent.cuts.131072]", "pause_s = 1.0"),
    ("[models.glm53]", "seed = 1"),
])
def test_an_unknown_key_is_refused(tmp_path, table, line):
    text = P.PATH.read_text(encoding="utf-8").replace(f"{table}\n", f"{table}\n{line}\n", 1)
    (tmp_path / "production.toml").write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match=f"unknown key\\(s\\): {line.split()[0]}"):
        P.load(tmp_path / "production.toml")


@pytest.mark.parametrize(("old", "new", "refusal"), [
    ("glm53.h200 = 4\n", "glm53.h200 = 0\n", "a positive count"),
    ("llama32-1b.h100 = \"not found yet\"\n", "llama32-1b.h100 = \"unknown\"\n", "a positive count"),
    ("llama32-1b.h100 = \"not found yet\"\n", "", "names cards \\['h200'\\], not its cards \\['h100', 'h200'\\]"),
    ("glm53.h200 = 4\n", "glm53.h200 = 4\nglm53.h100 = 4\n", "names cards"),
    ("prompts = 200\n", "", "exactly"),
    ("within = [\"token\"]\n", "within = [\"e2e\"]\n", "within names"),
    ("[profiles.coding-agent.cuts.131072]", "[profiles.coding-agent.cuts.65536]", "no cut at llama-3.2-1b"),
    ("factor = 2\n", "factor = 1\n", "factor of at least 2"),
    ("gain = 0.05\n", "gain = 0.0\n", "gain a fraction between 0 and 1"),
    ("most = 64\n", "most = 0\n", "start of at least 1 and at most most"),
])
def test_a_profile_off_its_shape_is_refused(tmp_path, old, new, refusal):
    (tmp_path / "production.toml").write_text(P.PATH.read_text(encoding="utf-8").replace(old, new, 1), encoding="utf-8")
    with pytest.raises(ConfigError, match=refusal):
        P.load(tmp_path / "production.toml")


def test_a_sessions_profile_has_a_ladder_and_the_others_none(tmp_path, doc):
    assert P.rungs(doc, "coding-agent") == (1, 2, 4, 8, 16, 32, 64)
    assert (doc.profiles["chat"].ladder, doc.profiles["fixed-size"].ladder) == (None, None)
    text = P.PATH.read_text(encoding="utf-8")
    start = text.index("[profiles.coding-agent.ladder]")
    (tmp_path / "production.toml").write_text(text[:start] + text[text.index("[profiles.coding-agent.cuts", start):],
                                              encoding="utf-8")
    with pytest.raises(ConfigError, match="a timed_trace profile has exactly .*'ladder'"):
        P.load(tmp_path / "production.toml")


def test_the_file_is_at_the_trees_vllm(doc):
    assert doc.vllm == VLLM_PIN


def test_datasets_bounds_and_shapes_are_serve_taxs(doc, S):
    def pins(d):
        return (d.repository, d.revision, d.file, d.bytes, d.sha512)

    agent = doc.profiles["coding-agent"]
    assert pins(agent.datasets["agentx"]) == tuple(vars(S.AGENTX).values())
    assert pins(agent.datasets["agentx-256k"]) == tuple(vars(S.AGENTX_256K).values())
    assert pins(doc.profiles["chat"].dataset) == tuple(vars(S.LOADS["chat"].file).values())
    assert {name: (cut.dataset, cut.turns) for name, cut in agent.cuts.items()} == {
        str(S.LOADS[load].context): ({S.AGENTX: "agentx", S.AGENTX_256K: "agentx-256k"}[S.LOADS[load].file],
                                     S.LOADS[load].prompts) for load in (S.AGENTIC_256K, S.AGENTIC)}
    assert agent.block == S.BLOCK
    assert doc.profiles["chat"].prompts == S.LOADS["chat"].prompts
    assert doc.profiles["fixed-size"].prompts == S.LOADS["long"].prompts
    assert {(doc.bounds.first_token_s, doc.bounds.token_s)} == {model.slo for model in S.MODELS.values()}
    for name, model in doc.models.items():
        assert (model.repository, model.revision, model.served) == (
            S.MODELS[name].repo, S.MODELS[name].revision, S.MODELS[name].served)


def test_a_load_is_a_models_on_one_card_and_not_found_yet_is_refused(doc):
    assert P.load_point(doc, "coding-agent", "glm53", "h200") == 4
    with pytest.raises(ConfigError, match="no load on card h100; its cards are h200"):
        P.load_point(doc, "coding-agent", "glm53", "h100")
    with pytest.raises(ConfigError, match="chat on glm53.h200: the load is not found yet"):
        P.load_point(doc, "chat", "glm53", "h200")


def test_each_models_cards_are_card_files_shorts(doc):
    from verity.device import load_card, skus

    cards = Path(__file__).resolve().parents[3] / "calibration" / "hardware" / "cards"
    shorts = {load_card(sku, directory=cards).get("short") for sku in skus("gpu", directory=cards)}
    for name, model in doc.models.items():
        assert set(model.cards) <= shorts, f"models.{name}: cards {model.cards} not all of the card files' {sorted(shorts)}"
