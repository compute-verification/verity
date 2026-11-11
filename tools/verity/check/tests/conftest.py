"""Every test here starts from an environment without credentials, a recording run's variables, or what `check` sets for its
steps: inside a recorded check its own tests inherit all three (`check.env()`).  And with a step cache of its own
(`CHECK_CACHE`): a fake step's pass stored under a real key (`check.STEP_KEYS`, which key the commit's tree) would be
reused by a later test, and by this machine's next real check."""
from __future__ import annotations

import os

import pytest

CLEARED = ("AWS_", "R2_", "RESEARCH_", "CLOUD_AGENT_")      # the last: or, R2_ cleared, the store reads a VM's shell that lost its secrets
SET_BY_CHECK = ("CHECK_STORE_CUSTODY", "VERITY_STORE_REQUIRED", "VERITY_REGRESSION")
#: the packages' `proved_in` as their policies name them, for a test that resolves this tree's plans (`check.audit_provers`):
#: reading every package's lean-audit.json would make each one an input of this suite, which most Lean changes rewrite
PROVERS = {"security": ["security/Proofs"], "security/Properties": ["security/Proofs"]}


@pytest.fixture(autouse=True)
def _no_inherited_run(monkeypatch, tmp_path_factory):
    for k in list(os.environ):
        if k.startswith(CLEARED) or k in SET_BY_CHECK:
            monkeypatch.delenv(k)
    monkeypatch.setenv("CHECK_CACHE", str(tmp_path_factory.mktemp("check-cache")))


@pytest.fixture
def provers(monkeypatch):
    """``provers(check)``: ``check``, a test module's copy of check.py, reads `PROVERS` as its Lean packages' provers
    (`check._lean_audit().provers`); returns them."""
    def stub(check) -> dict[str, list[str]]:
        la = check._lean_audit()

        class Audit:
            provers = staticmethod(lambda p: PROVERS.get(p, []))

            def __getattr__(self, name):
                return getattr(la, name)

        monkeypatch.setattr(check, "_lean_audit", Audit)
        return PROVERS
    return stub
