"""The Lean soundness package's RoPE data (`Security/Proofs/Flock/Soundness/Rope/`: the gates, the pinned rows,
the checkpoints and the chunk checks): the committed gate circuit computes the unit's circuit (on random inputs), and
the rest is regenerated here from it and the pinned rows, byte for byte, for the netlist `rope_head.PINS` pins.
`FlockSoundness.Rope.pinned_eq` checks the rest in the kernel."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("verity_vllm.program.registry.prims")
from verity_flock import lean_rows  # noqa: E402
from verity_flock.templates import rope_head  # noqa: E402

ROPE = Path(__file__).resolve().parents[4] / "Security" / "Proofs" / "Flock" / "Soundness" / "Rope"


def test_the_encoders_refuse_what_the_lean_decoders_cannot_read():
    assert lean_rows._unpack(lean_rows._pack([3, 4, 5, 9, 2, 2])) == [3, 4, 5, 9, 2, 2]
    for cols in (list(range(0, 1026, 2)), list(range(512)), list(range(8191, 8191 + 511)), [8192]):
        with pytest.raises(AssertionError):
            lean_rows._pack(cols)
    assert lean_rows._ungate(lean_rows._gate_code(1, (1 << 17) - 3, -2)) == (1, (1 << 17) - 3, -2)
    with pytest.raises(AssertionError):
        lean_rows._gate_code(1, (1 << 17) - 2, 0)


def test_rope_data_is_the_unit_and_the_pinned_rows():
    data = lean_rows.rope()
    assert data["sha256"] == rope_head.PINS[64]
    for name in lean_rows.FILES:
        assert (ROPE / name).read_text() == data[name], f"{name} is stale: run python -m verity_flock.lean_rows --out {ROPE}"
