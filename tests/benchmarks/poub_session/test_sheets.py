"""The session's node figures are its data sheets', value for value the shared fixture the tests read, and a verifier
started without them is refused by name."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks" / "poub_session"))

import sheets  # noqa: E402
from verity.device import load_card, load_node  # noqa: E402
from verity.protocols.resource_accounting.memory.poub.verifier import timing  # noqa: E402
from verity.protocols.resource_accounting.memory.poub.verifier.poub_session import verifier_process as VP  # noqa: E402

CARD_FILES = ROOT / "calibration/hardware/cards"
NODE_FILES = ROOT / "calibration/hardware/nodes"
FIXTURE = ROOT / "tests/verity/protocols/resource_accounting/memory/poub/figures.json"
RTX = "rtx-pro-6000-server-x8"


def test_the_sheets_figures_are_the_shared_fixtures():
    assert sheets.figures(RTX).to_json() == json.loads(FIXTURE.read_text())
    assert timing.Figures.of_json(json.loads(FIXTURE.read_text())) == sheets.figures(RTX)


def test_the_command_writes_the_figures_the_verifier_reads(capsys):
    assert sheets.main([RTX]) == 0
    assert timing.Figures.of_json(json.loads(capsys.readouterr().out)) == sheets.figures(RTX)


def test_a_node_kind_without_a_node_file_is_refused_by_name(capsys):
    assert sheets.main(["cpu"]) == 3
    assert "no node file cpu.toml" in capsys.readouterr().err


def test_a_card_the_node_does_not_name_is_refused():
    node = load_node(RTX, directory=NODE_FILES, cards_directory=CARD_FILES)
    with pytest.raises(ValueError, match="hand in the device card the node names"):
        timing.Figures.of_sheets(node, load_card("h200-sxm-141gb", directory=CARD_FILES),
                                 load_card(node["host"]["sku"], directory=CARD_FILES), **sheets.UNSHEETED[RTX])


@pytest.mark.parametrize("change, why", [
    (lambda j: j.pop("gpus"), "a node's figures are an object of exactly"),
    (lambda j: j.update(gpus=8.0), "gpus is a positive integer"),
    (lambda j: j.update(gpu_read_bytes_per_s=0), "gpu_read_bytes_per_s is a positive number"),
    (lambda j: j.update(gpu_squarings_per_s=[1]), "gpu_squarings_per_s maps each codec width"),
])
def test_figures_not_of_to_jsons_form_are_refused(change, why):
    j = json.loads(FIXTURE.read_text())
    change(j)
    with pytest.raises(ValueError, match=why):
        timing.Figures.of_json(j)


def test_a_verifier_started_without_figures_is_refused_by_name(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["verifier_process.py", "--out", str(tmp_path)])
    with pytest.raises(SystemExit) as e:
        VP.main()
    assert e.value.code == 2 and "--figures FILE" in capsys.readouterr().err.splitlines()[-1]
