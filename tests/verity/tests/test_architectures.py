"""The architecture files: each GPU card's `arch` names one and each one is named by a card; `load_architecture`'s
refusals."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from verity.device import load_architecture, load_card, skus
from verity.errors import InvalidArtifact

HARDWARE = Path(__file__).resolve().parents[3] / "calibration/hardware"
CARD_FILES, ARCH_FILES = HARDWARE / "cards", HARDWARE / "architectures"
NAMES = sorted(p.stem for p in ARCH_FILES.glob("*.toml"))


def test_each_gpu_cards_arch_names_an_architecture_and_each_architecture_a_card() -> None:
    named = {sku: load_card(sku, directory=CARD_FILES)["arch"] for sku in skus("gpu", directory=CARD_FILES)}
    missing = sorted(sku for sku, arch in named.items() if arch not in NAMES)
    assert not missing, f"{missing}: add calibration/hardware/architectures/<arch>.toml for each card's `arch`"
    unread = sorted(set(NAMES) - set(named.values()))
    assert not unread, f"{unread}: no GPU card's `arch` names these; delete them or add the card"


@pytest.mark.parametrize("name", NAMES)
def test_each_architecture_loads_with_its_gates_sourced(name: str) -> None:
    arch = load_architecture(name, directory=ARCH_FILES)
    assert arch["virtual"] == name.replace("sm_", "compute_")
    for gate in arch["toolkit"]:
        assert re.fullmatch(r"\d+\.\d+", gate["version"]), gate
        assert re.fullmatch(r"\d+(\.\d+)+", gate["driver_min"]), gate


def _release(text: str) -> tuple[int, ...]:
    return tuple(int(p) for p in text.split("."))


@pytest.mark.parametrize("name", NAMES)
def test_each_architectures_gates_rise_by_version_and_never_lower_the_driver(name: str) -> None:
    gates = load_architecture(name, directory=ARCH_FILES)["toolkit"]
    versions = [_release(g["version"]) for g in gates]
    assert versions == sorted(set(versions)), f"{name}: gates at {[g['version'] for g in gates]}; one a version, oldest first"
    drivers = [_release(g["driver_min"]) for g in gates]
    assert drivers == sorted(drivers), (f"{name}: driver_min {[g['driver_min'] for g in gates]} by version; a newer toolkit "
                                        "never needs an older driver: read its row again")


def _write(tmp_path: Path, text: str, name: str = "sm_90a") -> Path:
    (tmp_path / f"{name}.toml").write_text(text, encoding="utf-8")
    return tmp_path


def test_load_architecture_refuses_what_is_not_an_architecture_file(tmp_path: Path) -> None:
    good = (ARCH_FILES / "sm_90a.toml").read_text(encoding="utf-8")
    with pytest.raises(KeyError, match="no architecture 'sm_89'"):
        load_architecture("sm_89", directory=ARCH_FILES)
    with pytest.raises(InvalidArtifact, match="set it to the file's name"):
        load_architecture("sm_90a", directory=_write(tmp_path, good.replace('name = "sm_90a"', 'name = "sm_100a"')))
    with pytest.raises(InvalidArtifact, match="an architecture has exactly"):
        load_architecture("sm_90a", directory=_write(tmp_path, 'sms = "132"\n' + good))
    with pytest.raises(InvalidArtifact, match="each with exactly"):
        load_architecture("sm_90a", directory=_write(tmp_path, good.replace('driver_min = "525.60.13"\n', "")))
    with pytest.raises(InvalidArtifact, match=r"\['toolkit\[0\]\.driver_min'\]; write each as a string"):
        load_architecture("sm_90a", directory=_write(tmp_path, good.replace('"525.60.13"', "525.6")))
    with pytest.raises(InvalidArtifact, match="give the major.minor whose digits 'sm_90a' names"):
        load_architecture("sm_90a", directory=_write(tmp_path, good.replace('capability = "9.0"', 'capability = "9.1"')))

