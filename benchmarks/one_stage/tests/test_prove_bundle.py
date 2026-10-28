"""The bundle prover's inputs (`prove_bundle.py --bundle B --window DIR`) on the H200 interfaces' drawn-unit bundle (four drawn units
of a W8A8 MLP's window) and the part of its window that the hold reads (`fixtures/bundle-window-public`: its
`window.json`'s run, registration and partition digests and the drawn members' public bodies, beside the H200 interfaces' copy of
its registration): the honest bundle is held to its window with every check passing, and the command refuses, before
staging or proving anything, a bundle whose payload was edited and a window whose drawn member's public body isn't the
one the bundle names. Proving it end to end needs Flock, Lean and the window's private files: `prove_bundle.sh`."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PUBLIC = HERE / "fixtures" / "bundle-window-public"
E2E = HERE.parents[2] / "examples" / "h200" / "interfaces" / "tests" / "fixtures"
sys.path.insert(0, str(HERE.parent))
import prove_bundle as PB  # noqa: E402


@pytest.fixture
def window(tmp_path) -> Path:
    w = tmp_path / "window"
    shutil.copytree(PUBLIC, w)
    shutil.copyfile(E2E / "drawn-unit-registration.json", w / "registration.json")
    return w


def _main(tmp_path: Path, bundle: Path, window: Path) -> tuple[int, dict]:
    out = tmp_path / "out"
    rc = PB.main(["--bundle", str(bundle), "--window", str(window), "--bin", "absent", "--lean", "absent",
                  "--firewall", "absent", "--out", str(out)])
    return rc, json.loads((out / "summary.json").read_text())


def test_the_bundle_is_held_to_its_window(window) -> None:
    doc, payload = PB.RB.read(E2E / "drawn-unit-bundle.json")
    held, bad = PB.hold(doc, payload, window)
    assert bad == {} and held and all(held.values())
    assert set(held) == {"bundle_check", "run", "registration", "partition"} | {f"public:m{m['k']}" for m in doc["members"]}
    assert PB.held_draw(doc, window) == {}


def test_an_edited_payload_is_refused_before_anything_is_proved(tmp_path, window) -> None:
    d = tmp_path / "bundle"
    d.mkdir()
    shutil.copyfile(E2E / "drawn-unit-bundle.json", d / "bundle.json")
    payload = bytearray((E2E / "drawn-unit-bundle.bin").read_bytes())
    payload[7] ^= 1
    (d / "bundle.bin").write_bytes(bytes(payload))
    rc, summary = _main(tmp_path, d / "bundle.json", window)
    assert rc == 1 and summary["held"]["bundle_check"] is False
    assert summary["bundle_findings"] == {"payload": "its bytes are not the ones the bundle names"}
    assert "members" not in summary


def test_a_member_whose_public_body_is_not_the_bundles_is_refused(tmp_path, window) -> None:
    body = bytearray((window / "m25-public.bin").read_bytes())
    body[-1] ^= 1
    (window / "m25-public.bin").write_bytes(bytes(body))
    rc, summary = _main(tmp_path, E2E / "drawn-unit-bundle.json", window)
    assert rc == 1 and "members" not in summary
    assert {k for k, v in summary["held"].items() if not v} == {"public:m25"}
