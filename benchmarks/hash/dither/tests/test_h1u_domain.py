"""h1u refuses a K that 512 doesn't divide (``if (K % 512) return 2``), the refusal sm_90's H1 device record
(`h1device.SM90`) states: it serves K >= `h1zy.LEAST_K` (384) with 512 | K, so its least K is 512."""
from __future__ import annotations

from pathlib import Path

from verity.protocols.pouw.prover.circuit import h1device as D
from verity.protocols.pouw.prover.circuit import h1zy as Z

H1U = Path(__file__).resolve().parents[1] / "h1u.cu"


def test_sm90s_domain_is_h1us_refusal() -> None:
    assert "if (K % 512) return 2;" in H1U.read_text()
    assert D.SM90.domain == (Z.LEAST_K, 512) and D.SM90.refuse(512) is None and D.SM90.refuse(1024 + 32) is not None
