"""a5's header check of a member reads the member's registered file as Lean's members check does (`Member.reads`,
`Member.held`): a port the slice holds no root of, read at a registered value the file holds there, is held by that
read; read at any other value it is refused by name (`members.unregistered_why`, `OneStageZk.UNREGISTERED_WHY`), the
refusal `test_header_held_at_difftest.py` holds to the Lean's."""
from __future__ import annotations

import sys
from pathlib import Path

from verity.protocols.proof.spot.verifier.one_stage import members as MB

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks" / "one_stage"))
import a5 as A5  # noqa: E402

SLICE = {"population": 2, "scheme": "s", "roots": [{"domain": "dx", "port": "x", "root": "rx"}]}


def _header(ports: dict[str, str]) -> dict:
    return {"frame_v3": {"domain_ids": {p: f"d{p}" for p in ports}, "roots": dict(ports)}, "instances": 2,
            "scheme": "s", "units": {"indices": [0, 1]}}


def _file(value: str) -> dict:
    return {"t/c0": {"format": "verity/registered-reads/v2",
                     "ports": {"w": {"domain": "d" * 128, "leaves": 2, "root": "a" * 128, "value": value}}}}


def _why(registered: dict | None) -> str | None:
    reads, held = A5.member_reads(registered, "t/c0")
    return MB.population_why([[0, 2]], [("t", [[0, 2]])], _header({"x": "rx", "w": "rw"}), SLICE, reads, held)


def test_a_registered_read_holds_its_port_and_any_other_read_is_refused_by_name():
    assert _why(_file("h/w")) is None
    for v in ("committed/w", "input/n3", "t/o0"):
        assert _why(_file(v)) == MB.unregistered_why("w", v)
    assert _why(None) == MB.unwired_why("w")
    assert _why({"t/c1": _file("h/w")["t/c0"]}) == MB.unwired_why("w")
