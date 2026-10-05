"""Every circuit set pinned in ``archive/gkr/verifier/pins.txt`` is what the generator on its line writes on this tree.

The verifier refuses statement files that are not a pinned set of the claimed relation (``verity-gkr-verify verify
--relation``); this test ties each pin to the generating code, so a change that moves a circuit fails here until its line
is updated.  The failure message prints the line the tree now generates.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import pytest

GKR = Path(__file__).resolve().parents[1]
REPO = GKR.parents[1]
PINS = GKR / "verifier" / "pins.txt"
FILES = ("circuit.txt", "epilogue.txt", "chain.txt")


class Pin(NamedTuple):
    relation: str
    steps: int
    digests: tuple[str, str, str]
    generator: tuple[str, ...]


def pins() -> list[Pin]:
    out = []
    for line in PINS.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            relation, steps, circuit, epilogue, chain, *generator = line.split()
            out.append(Pin(relation, int(steps), (circuit, epilogue, chain), tuple(generator)))
    return out


def test_pins_name_a_generator_and_each_circuit_set_once():
    rows = pins()
    assert {p.relation for p in rows} >= {"bf16-ampere", "bf16-hopper"}
    assert len({(p.steps, p.digests) for p in rows}) == len(rows)
    assert all(p.generator[:2] == ("python", "-m") for p in rows), "a generator is `python -m <module> ...`, run from archive/gkr"


@pytest.mark.parametrize("pin", pins(), ids=lambda p: f"{p.relation}-{p.digests[0][:8]}")
def test_generator_writes_the_pinned_circuits(pin: Pin, tmp_path: Path):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, (str(REPO), os.environ.get("PYTHONPATH"))))}
    r = subprocess.run([sys.executable, *pin.generator[1:], "--out", str(tmp_path)], cwd=GKR, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    got = tuple(hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in FILES)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    steps = manifest["steps"]
    assert manifest["variant"] == "v2"
    assert (steps, got) == (pin.steps, pin.digests), (
        f"{' '.join(pin.generator)} no longer writes the pinned {pin.relation} circuits; if the statement change is intended, "
        f"this tree's line is:\n{pin.relation} {steps} {' '.join(got)} {' '.join(pin.generator)}")
