"""The Lean soundness package's RoPE data (`FlockSoundness/Rope/`: the gates, the pinned rows, the checkpoints and the
chunk checks) is the unit's Boolean export and its pinned rows: regenerated here, byte for byte, for the netlist
`rope_head.PINS` pins. `FlockSoundness.Rope.pinned_eq` checks the rest in the kernel."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("verity_vllm.program.registry.prims")
from verity_flock import lean_rows  # noqa: E402
from verity_flock.templates import rope_head  # noqa: E402

ROPE = Path(__file__).resolve().parents[1] / "verifier" / "lean" / "soundness" / "FlockSoundness" / "Rope"


def test_rope_data_is_the_export_and_the_pinned_rows():
    data = lean_rows.rope()
    assert data["sha256"] == rope_head.PINS[64]
    for name in lean_rows.FILES:
        assert (ROPE / name).read_text() == data[name], f"{name} is stale: run python -m verity_flock.lean_rows --out {ROPE}"
