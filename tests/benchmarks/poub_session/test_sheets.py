"""The session's node figures are its data sheets', value for value the shared fixture the tests read, and a verifier
started without them is refused by name."""

import dataclasses
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks" / "poub_session"))

import sheets  # noqa: E402
from verity.device import load_card, load_node, number  # noqa: E402
from verity.protocols.accounting.memory.poub.verifier import codec, ledger, timing  # noqa: E402
from verity.protocols.accounting.memory.poub.verifier.poub_session import verifier_process as VP  # noqa: E402

CARD_FILES = ROOT / "calibration/hardware/cards"
NODE_FILES = ROOT / "calibration/hardware/nodes"
FIXTURE = ROOT / "tests/verity/protocols/accounting/memory/poub/figures.json"
RTX = "rtx-pro-6000-server-x8"


def test_the_sheets_figures_are_the_shared_fixtures():
    assert sheets.figures(RTX).to_json() == json.loads(FIXTURE.read_text())
    assert timing.Figures.of_json(json.loads(FIXTURE.read_text())) == sheets.figures(RTX)


def test_the_command_writes_the_figures_the_verifier_reads(capsys):
    assert sheets.main([RTX]) == 0
    assert timing.Figures.of_json(json.loads(capsys.readouterr().out)) == sheets.figures(RTX)
    assert sheets.main(["--sha512", RTX]) == 0
    assert capsys.readouterr().out == sheets.figures(RTX).sha512() + "\n"


def test_the_handed_squaring_rate_is_the_fastest_variant_at_the_cards_highest_clock_reading():
    """One GPU's share of the card's maximum squaring rate at 512 bits, in squarings a second, is
    `s·REF·f/REF_MHZ + H/gpus` with s the fastest of `gamma.GPU`'s variants and f the highest clock reading its card
    file states, so at least that at every reading."""
    clock, gpus = load_card("rtx-pro-6000-server", directory=CARD_FILES)["clock"], 8
    readings = [float(number(clock[r])) for r in sheets.CLOCK_READINGS]
    s = max(sheets.gamma.GPU.values())
    at = [s * sheets.gamma.REF * f / sheets.gamma.REF_MHZ + sheets.HOST_SQUARINGS_PER_S / gpus for f in readings]
    handed = sheets.figures(RTX).gpu_squarings_per_s["p512"]
    assert readings == [2430, 2617] and handed == at[readings.index(max(readings))]
    assert all(handed >= x for x in at)


def test_an_h200_nodes_rebuild_share_is_refused_by_name():
    """No squaring rate is measured on an H200, and the RTX PRO 6000's is not its bound: the session's ledger refuses
    its rebuild share at each width, naming the card, the width and the fix."""
    node = ledger.Node.stated(dataclasses.replace(sheets.figures("h200-sxm-141gb-x1"), gpu_chip_bytes=1))
    for c in (codec.DEFAULT, codec.Codec(codec.P512, 2, False)):
        with pytest.raises(ValueError, match=f"no maximum squaring rate is established for h200-sxm-141gb at "
                                             f"{c.field.n} bits: no squaring rate is measured on it.*FORMS"):
            node.rebuild(c)


def test_an_h200_nodes_borrowed_sha512_rate_link_burst_and_on_chip_storage_are_refused_by_name():
    """Each bounds what the node can do from above, so the RTX PRO 6000 node's is not the H200 node's: its SHA-512
    rate is refused where it is read, its link burst where its links are charged, and its GPUs' on-chip storage, which
    its own card file doesn't state, wherever S_chip is read, the session's ledger first."""
    fig = sheets.figures("h200-sxm-141gb-x1")
    chip = "no on-chip storage is established for h200-sxm-141gb: its card file states no store.chip.*h200-sxm-141gb.toml"
    with pytest.raises(ValueError, match=chip):
        _ = fig.chip_bytes
    with pytest.raises(ValueError, match=chip):
        ledger.Node.stated(fig)
    with pytest.raises(ValueError, match="no SHA-512 rate is established for h200-sxm-141gb: .*UNSHEETED"):
        _ = fig.sha512_per_s
    with pytest.raises(ValueError, match="no link burst is established for the node kind h200-sxm-141gb-x1: .*UNSHEETED"):
        ledger.Node.stated(dataclasses.replace(fig, gpu_chip_bytes=1), link_bytes_per_s=1e9)
    assert ledger.Node.stated(dataclasses.replace(fig, gpu_chip_bytes=1)).link_burst_bytes == 0


def test_a_node_kind_without_a_node_file_is_refused_by_name(capsys):
    assert sheets.main(["cpu"]) == 3
    assert "no node file cpu.toml" in capsys.readouterr().err


def test_a_card_the_node_does_not_name_is_refused():
    node = load_node(RTX, directory=NODE_FILES, cards_directory=CARD_FILES)
    with pytest.raises(ValueError, match="hand in the device card the node names"):
        timing.Figures.of_sheets(node, load_card("h200-sxm-141gb", directory=CARD_FILES),
                                 load_card(node["host"]["sku"], directory=CARD_FILES), **sheets.UNSHEETED[RTX],
                                 gpu_squarings_per_s={})


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


def test_a_figures_file_not_the_one_its_pinned_digest_names_is_refused_and_a_record_names_the_figures_it_read(
        tmp_path, monkeypatch, capsys):
    """`--figures-sha512` holds the file to the digest the verifier's configuration pins: a lenient file (half the
    card's maximum squaring rate, half S_chip, half the host's DRAM) is refused before the process listens.  The pinned file is
    read, and the record the process writes, here a refused registration's, carries the figures whole and their
    digest, the ingress rate beside them, and says whether they were pinned: unpinned, as where both parties run on
    the developer's machine, it says they weren't."""
    from verity.protocols.accounting.memory.poub.verifier.session import verifier as V

    fixture = timing.Figures.of_json(json.loads(FIXTURE.read_text()))
    lenient = tmp_path / "lenient.json"
    lenient.write_text(json.dumps({**fixture.to_json(), "gpu_chip_bytes": fixture.gpu_chip_bytes // 2,
                                   "host_dram_bytes": fixture.host_dram_bytes // 2,
                                   "gpu_squarings_per_s": {w: r if type(r) is str else r / 2
                                                           for w, r in fixture.gpu_squarings_per_s.items()}}))
    out = tmp_path / "out"
    pinned = ["verifier_process.py", "--out", str(out), "--check-workers", "0", "--ingress-bytes-per-s", "4.4e9",
              "--figures-sha512", fixture.sha512()]
    monkeypatch.setattr(sys, "argv", [*pinned, "--figures", str(lenient)])
    with pytest.raises(SystemExit) as e:
        VP.main()
    assert e.value.code == 2 and f"not {fixture.sha512()}, the one --figures-sha512 pins" in capsys.readouterr().err
    assert not (out / "verifier.json").exists()

    class Server:
        def accept(self):
            return None, ("prover", 1)

    class Refusing:
        def __init__(self, conn, **kw):
            assert kw["figures"] == fixture

        def register(self):
            raise ValueError("refused here")

        def registry_json(self):
            return {}

        def stop(self, check=True):
            pass

    monkeypatch.setattr(V.wire, "listen", lambda host, port: Server())
    monkeypatch.setattr(V.wire, "Conn", lambda sock: None)
    monkeypatch.setattr(V, "Verifier", Refusing)
    monkeypatch.setattr(sys, "setswitchinterval", lambda s: None)
    monkeypatch.setattr(sys, "argv", [*pinned, "--figures", str(FIXTURE)])
    assert VP.main() == 3
    record = json.loads((out / "verifier.json").read_text())
    assert record["refused"] == "refused here"
    assert (record["figures"], record["figures_sha512"]) == (json.loads(FIXTURE.read_text()), fixture.sha512())
    assert (record["figures_pinned"], record["ingress_bytes_per_s"]) == (True, 4.4e9)
    monkeypatch.setattr(sys, "argv", [*pinned[:-2], "--figures", str(FIXTURE)])
    assert VP.main() == 3
    unpinned = json.loads((out / "verifier.json").read_text())
    assert (unpinned["figures_sha512"], unpinned["figures_pinned"]) == (fixture.sha512(), False)
    assert timing.Figures.of_json(json.loads(lenient.read_text())).sha512() != fixture.sha512()
