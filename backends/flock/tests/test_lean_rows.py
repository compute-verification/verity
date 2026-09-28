"""The Lean soundness package's verified lowerings (`FlockSoundness/<ns>/`: the gates, the pinned rows, the checkpoints and
the kernel's checks) are each template's Boolean export and pinned rows: regenerated here, byte for byte, for the netlist
the template pins (`lean_rows.unit` refuses any other). `<ns>.sound` checks the rest in the kernel."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("verity_vllm.program.registry.prims")
from verity_flock import lean_rows  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "verifier" / "lean" / "soundness" / "FlockSoundness"


def test_the_encoders_refuse_what_the_lean_decoders_cannot_read():
    assert lean_rows._unpack(lean_rows._pack([3, 4, 5, 9, 2, 2])) == [3, 4, 5, 9, 2, 2]
    for cols in (list(range(0, 1026, 2)), list(range(8192)), list(range(8191, 8191 + 8191)), [8192]):
        with pytest.raises(AssertionError):
            lean_rows._pack(cols)
    assert lean_rows._ungate(lean_rows._gate_code(1, (1 << 17) - 3, -2)) == (1, (1 << 17) - 3, -2)
    with pytest.raises(AssertionError):
        lean_rows._gate_code(1, (1 << 17) - 2, 0)


@pytest.mark.parametrize("template", sorted(lean_rows.TEMPLATES))
def test_lean_data_is_the_export_and_the_pinned_rows(template):
    data = lean_rows.unit(template)
    d = ROOT / lean_rows.TEMPLATES[template][0]
    for name in lean_rows.FILES:
        assert (d / name).read_text() == data[name], f"{d / name} is stale: run python -m verity_flock.lean_rows --out {ROOT}"
