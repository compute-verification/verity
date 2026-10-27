"""The bundle prover's inputs (`prove_bundle.py --bundle B --window DIR`) on the fetched fixture `fixtures/bundle-window`
(a W8A8 MLP's window, its tampered twin and the bundle of four drawn units): the honest bundle is held to its window with
every check passing, and the command refuses, before staging or proving anything, a bundle whose payload was edited and
the honest bundle against the tampered window. Proving it end to end needs C-Flock and Lean built: `prove_bundle.sh`."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FX = HERE / "fixtures" / "bundle-window"
sys.path.insert(0, str(HERE.parent))
import prove_bundle as PB  # noqa: E402


def _main(tmp_path: Path, bundle: Path, window: Path) -> tuple[int, dict]:
    out = tmp_path / "out"
    rc = PB.main(["--bundle", str(bundle), "--window", str(window), "--bin", "absent", "--lean", "absent",
                  "--firewall", "absent", "--out", str(out)])
    return rc, json.loads((out / "summary.json").read_text())


def test_the_fixtures_bundle_is_held_to_its_window() -> None:
    doc, payload = PB.RB.read(FX / "bundle.json")
    held, bad = PB.hold(doc, payload, FX / "window")
    assert bad == {} and held and all(held.values())
    assert set(held) == {"bundle_check", "run", "registration", "partition"} | {f"public:m{m['k']}" for m in doc["members"]}
    assert PB.held_draw(doc, FX / "window") == {}


def test_an_edited_payload_is_refused_before_anything_is_proved(tmp_path) -> None:
    d = tmp_path / "bundle"
    d.mkdir()
    shutil.copyfile(FX / "bundle.json", d / "bundle.json")
    payload = bytearray((FX / "bundle.bin").read_bytes())
    payload[7] ^= 1
    (d / "bundle.bin").write_bytes(bytes(payload))
    rc, summary = _main(tmp_path, d / "bundle.json", FX / "window")
    assert rc == 1 and summary["held"]["bundle_check"] is False
    assert summary["bundle_findings"] == {"payload": "its bytes are not the ones the bundle names"}
    assert "members" not in summary


def test_the_honest_bundle_is_refused_against_the_tampered_window(tmp_path) -> None:
    rc, summary = _main(tmp_path, FX / "bundle.json", FX / "tampered")
    assert rc == 1 and not all(summary["held"].values()) and "members" not in summary
    assert summary["held"]["bundle_check"] is False
