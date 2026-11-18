"""The card as a spec's parameter: a spec lists the cards it runs on (`cards`, by their `short`), a run's card is the one
named (``SPEC@CARD``), else the spec's one, else the driver's, and a card the spec doesn't list is refused by name; a
former per-card spec id names a stored run's spec and no launch."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from verity_config import ConfigError

from verity_bench import jobs, kit, served_table
from verity_bench.cli import main
from verity_bench.vllm import presets
from verity_bench.vllm.presets import FORMER_IDS, SPEC_ROOT, load_spec, resolve_spec, split_card

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


def test_a_cards_arms_replace_the_specs_and_its_arm_settings_add_to_the_specs_on_that_card_alone(tmp_path):
    path = _two_cards(tmp_path, "", '\n[arm_settings."stock-fp8@compiled"]\ncontrol = "none"\n'
                      '\n[card_settings.h200]\narms = ["stock-fp8", "stock-fp8@compiled", "stock@compiled"]\n'
                      '[card_settings.h200.arm_settings."stock@compiled"]\ncontrol = "none"\n')
    path.write_text(path.read_text().replace('arms = ["stock", "stock-fp8"]', 'arms = ["stock", "stock-fp8", "stock-fp8@compiled"]'))
    stem = path.with_suffix("")
    rtx, h200 = resolve_spec(f"{stem}@rtx-pro-6000.toml"), resolve_spec(f"{stem}@h200.toml")
    assert (rtx.arms, [*rtx.arm_settings]) == (("stock", "stock-fp8", "stock-fp8@compiled"), ["stock-fp8@compiled"])
    assert (h200.arms, [*h200.arm_settings]) == (("stock-fp8", "stock-fp8@compiled", "stock@compiled"),
                                                 ["stock-fp8@compiled", "stock@compiled"])
    assert h200.arm_settings["stock@compiled"].control == "none"
    path.write_text(path.read_text().replace('"stock-fp8", "stock-fp8@compiled", "stock@compiled"]',
                                             '"stock-fp8", "stock-fp8@compiled"]'))
    with pytest.raises(ConfigError, match=r"card_settings.h200.arm_settings \['stock@compiled'\] name no arm"):
        load_spec(path)

def test_a_spec_argument_names_its_card_after_an_at_by_id_or_recorded_path():
    assert split_card(POUW) == (POUW, None)
    assert split_card(f"{POUW}@h100") == (POUW, "h100")
    assert split_card(f"benchmarks/harness/specs/{POUW}@h100.toml") == (f"benchmarks/harness/specs/{POUW}.toml", "h100")
    assert jobs._relative_spec(f"{POUW}@h100", ROOT) == served_table.spec_path(f"{POUW}@h100")


def test_each_former_id_is_its_spec_on_a_card_that_spec_lists_and_no_file_holds_it():
    former = tomllib.loads(FORMER_IDS.read_text())["former"]
    assert former and presets.former_specs() == {old: (e["spec"], e["card"]) for old, e in former.items()}
    for old, (spec, card) in presets.former_specs().items():
        assert not (SPEC_ROOT / f"{old}.toml").exists(), old
        assert card in load_spec(spec).cards, old
        assert presets.current_spec(old) == (spec if load_spec(spec).cards == (card,) else f"{spec}@{card}")
    assert presets.current_spec(POUW) == POUW


@pytest.mark.parametrize("old", sorted(presets.former_specs()))
def test_no_launch_path_takes_a_former_id(old, monkeypatch, capsys):
    """A former id resolves for a stored run's record alone: loading, resolving and planning it, and the harness's
    `plan`, `run` and `vllm` commands, each refuse it, naming the spec and card it is now."""

    spec, card = presets.former_specs()[old]
    now = rf"is '{spec}' on the {card}, `{spec}@{card}`"
    for launch in (load_spec, resolve_spec, lambda name: jobs.make_plan(name, root=ROOT, commit="a" * 40,
                                                                        lookup=lambda identity, commit: None)):
        with pytest.raises(ConfigError, match=now):
            launch(old)
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("RESEARCH_SOURCE_SHA", "a" * 40)
    monkeypatch.setattr(jobs, "dispatch", lambda *a, **k: pytest.fail(f"launched {a}"))
    for argv in (["plan", old, "--local-only"], ["run", old, "--local-only"], ["vllm", "--preset", old, "--out", "out"]):
        with pytest.raises((SystemExit, ConfigError)) as refused:
            main(argv)
        said = capsys.readouterr().err + str(refused.value)
        assert f"`{spec}@{card}`" in said, argv


def test_a_run_recorded_under_a_former_id_is_read_as_its_spec_on_its_card():
    old = next(iter(presets.former_specs()))
    spec, card = presets.former_specs()[old]
    assert served_table.spec_name(served_table.spec_path(old)) == f"{spec}@{card}"
    assert served_table.spec_path(old) in served_table.spec_paths(f"{spec}@{card}")
    assert served_table.spec_path(spec) in served_table.spec_paths(f"{spec}@{card}")
    on_others = [o for o, now in presets.former_specs().items() if now[0] == spec and now[1] != card]
    assert on_others and not set(map(served_table.spec_path, on_others)) & set(served_table.spec_paths(f"{spec}@{card}"))
    assert set(map(served_table.spec_path, on_others)) <= set(served_table.spec_paths(spec))
    assert served_table.passing(served_table.spec_paths(spec)).count("?") == len(served_table.spec_paths(spec)) + 1
