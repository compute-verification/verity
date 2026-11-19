"""The card as a spec's parameter: a spec lists the cards it runs on (`cards`, by their `short`), a run's card is the one
named (``SPEC@CARD``), else the spec's one, else the driver's, and a card the spec doesn't list is refused by name."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from verity_config import ConfigError

from verity_bench import jobs, kit, served_table
from verity_bench.vllm.presets import SPEC_ROOT, load_spec, resolve_spec, split_card

ROOT = Path(__file__).resolve().parents[3]
POUW = "llama32-1b-1gpu-pouw-protocol-served"
H100, H200, RTX = "NVIDIA H100 80GB HBM3", "NVIDIA H200", "NVIDIA RTX PRO 6000 Blackwell Server Edition"
VLLM_SPECS = sorted(p.stem for p in SPEC_ROOT.glob("*.toml") if "vllm-spec/v0" in p.read_text().splitlines()[0])


def _on(monkeypatch, card: str) -> None:
    monkeypatch.setattr(kit, "host_stamp", lambda: {"card": card})


def _two_cards(tmp_path, top: str = "", tables: str = "") -> Path:
    base = (SPEC_ROOT / "llama31-8b-fp8-1gpu.toml").read_text()
    path = tmp_path / "spec.toml"
    path.write_text(base.replace('cards = ["rtx-pro-6000"]', 'cards = ["rtx-pro-6000", "h200"]' + top) + tables)
    return path


@pytest.mark.parametrize("name", VLLM_SPECS)
def test_every_vllm_spec_lists_its_cards_by_short_and_resolves_on_each(name):
    spec = load_spec(name)
    shorts = {c["short"] for c in kit._gpu_cards(kit.cards_directory()) if c.get("short")}
    assert spec.cards and set(spec.cards) <= shorts, name
    assert "gpu_id" not in tomllib.loads((SPEC_ROOT / f"{name}.toml").read_text())["preset"], name
    identities = {resolve_spec(f"{name}@{card}").config_sha256 for card in spec.cards}
    assert [resolve_spec(f"{name}@{card}").preset.gpu_id for card in spec.cards] == list(spec.cards)
    assert len(identities) == len(spec.cards), name


def test_a_runs_card_is_the_one_named_else_the_specs_one_else_the_drivers(monkeypatch):
    _on(monkeypatch, "none")
    assert resolve_spec("llama31-8b-fp8-1gpu").preset.gpu_id == "rtx-pro-6000"
    assert resolve_spec(f"{POUW}@h100").preset.gpu_id == "h100"
    assert resolve_spec(SPEC_ROOT / f"{POUW}@h100.toml").preset.gpu_id == "h100"
    with pytest.raises(ConfigError, match=rf"runs on h200 or h100, and no card is here: name one, `{POUW}@CARD`"):
        resolve_spec(POUW)
    _on(monkeypatch, H100)
    assert resolve_spec(POUW).preset.gpu_id == resolve_spec(SPEC_ROOT / f"{POUW}.toml").preset.gpu_id == "h100"
    assert resolve_spec(POUW) == resolve_spec(f"{POUW}@h100")


def test_a_card_the_spec_doesnt_list_is_refused_by_name_named_or_found(monkeypatch):
    with pytest.raises(ConfigError, match=rf"spec '{POUW}' runs on h200, h100, not the rtx-pro-6000"):
        resolve_spec(f"{POUW}@rtx-pro-6000")
    _on(monkeypatch, RTX)
    with pytest.raises(ConfigError, match=rf"spec '{POUW}' runs on h200, h100, not the rtx-pro-6000"):
        resolve_spec(POUW)
    with pytest.raises(ConfigError, match="runs on rtx-pro-6000, not the h200"):
        resolve_spec("llama31-8b-fp8-1gpu@h200")


def test_card_settings_take_the_specs_place_on_their_card_alone_and_name_only_its_cards(tmp_path):
    path = _two_cards(tmp_path, '\nsetup = "benchmarks/harness/run_point.sh"',
                      '\n[card_settings.h200]\nsetup = "benchmarks/harness/setup/pearl_c_sm90.sh"\n'
                      'env = { ARM_KNOB = "sm90" }\n[card_settings.h200.admission]\nmax_minutes = 45\n')
    rtx, h200 = resolve_spec(f"{path.with_suffix('')}@rtx-pro-6000.toml"), resolve_spec(f"{path.with_suffix('')}@h200.toml")
    assert (dict(rtx.env), rtx.setup, rtx.admission) == ({}, "benchmarks/harness/run_point.sh", load_spec(path).admission)
    assert (dict(h200.env), h200.setup, h200.admission.max_minutes) == (
        {"ARM_KNOB": "sm90"}, "benchmarks/harness/setup/pearl_c_sm90.sh", 45)
    assert rtx.config_sha256 != h200.config_sha256
    path.write_text(path.read_text().replace("[card_settings.h200]", "[card_settings.h100]")
                    .replace("[card_settings.h200.admission]", "[card_settings.h100.admission]"))
    with pytest.raises(ConfigError, match=r"card_settings \['h100'\] name no card of the spec's cards"):
        load_spec(path)
    path.write_text(path.read_text().replace('"rtx-pro-6000", "h200"', '"rtx-pro-6000", "a100"'))
    with pytest.raises(ConfigError, match=r"cards \['a100'\] are no card's `short`"):
        load_spec(path)


def test_a_refused_card_is_a_card_the_spec_doesnt_list_and_its_refusal_says_why(monkeypatch, tmp_path):
    whole = "llama31-8b-1gpu-pearl-c-whole-served"
    with pytest.raises(ConfigError, match=rf"^spec '{whole}' refuses the rtx-pro-6000: its whole step served Pearl-C on "
                                          r"sm_120 \(`pouw:pearl-c-sm120-v1-h3`\), which is no declared scheme: declaring "
                                          r"one is compute-accounting's$"):
        resolve_spec(f"{whole}@rtx-pro-6000")
    _on(monkeypatch, RTX)
    assert resolve_spec(whole).preset.gpu_id == "h200"
    assert [served_table._on_card(whole, card) for card in ("rtx-pro-6000", "h200")] == [f"{whole}@rtx-pro-6000", whole]
    path = _two_cards(tmp_path, "", '\n[refused]\nh100 = "not measured there"\n')
    with pytest.raises(ConfigError, match=r"refuses the h100: not measured there$"):
        resolve_spec(f"{path.with_suffix('')}@h100.toml")
    for card in ("h200", "a100"):
        path.write_text(path.read_text().replace("\nh100 = ", f"\n{card} = ").replace("\nh200 = ", f"\n{card} = "))
        with pytest.raises(ConfigError, match=rf"refused \['{card}'\]: each names a card's `short` the spec doesn't list"):
            load_spec(path)


def test_a_spec_argument_names_its_card_after_an_at_by_id_or_recorded_path():
    assert split_card(POUW) == (POUW, None)
    assert split_card(f"{POUW}@h100") == (POUW, "h100")
    assert split_card(f"benchmarks/harness/specs/{POUW}@h100.toml") == (f"benchmarks/harness/specs/{POUW}.toml", "h100")
    assert jobs._relative_spec(f"{POUW}@h100", ROOT) == served_table.spec_path(f"{POUW}@h100")
