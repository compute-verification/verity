"""`resolve_log.quarantined`: gate G4 asks for differential tests of every spec whose body lives in a quarantine, core's or the
experimental lanes' (`verity_quarantine`); a spec from the registry proper does not count."""

from __future__ import annotations

import pytest

from verity_vllm.observe.fold.resolve_log import quarantined


@pytest.mark.parametrize("module, expected", [
    ("verity_vllm.program.registry.quarantine.ln.layer_norm_aten", True),
    ("verity_quarantine.collective.allreduce", True),
    ("verity_vllm.program.registry.b1", False),
    ("verity_vllm.program.registry.act", False),
    (None, False),
])
def test_quarantined(module, expected):
    assert quarantined(module) is expected
