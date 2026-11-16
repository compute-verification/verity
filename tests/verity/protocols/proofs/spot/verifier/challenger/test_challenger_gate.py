"""Infra's gate as its clients call it (`challenger_gate.gate_call`): a refusal quotes no session's secret, whichever of the
gate's reason, ssh's stderr or the answer it quotes."""

from __future__ import annotations

import json
import subprocess

import pytest

from verity.protocols.proofs.spot.verifier.challenger import challenger as Ch, challenger_gate as G

SECRET = "5e" * 32


def _run(rc: int, stdout: str, stderr: str = ""):
    return lambda argv, **_: subprocess.CompletedProcess(argv, rc, stdout, stderr)


@pytest.mark.parametrize("rc, stdout, stderr", [
    (255, "", f"ssh: the gate closed the session {SECRET} mid-open"),
    (3, json.dumps({"refused": f"open failed after its secret {SECRET} was made"}), ""),
    (255, json.dumps({"session": "s1", "port": 7811, "secret": SECRET}), ""),
], ids=["stderr", "refused", "answer"])
def test_a_refusal_withholds_a_secret_wherever_the_gate_put_it(rc, stdout, stderr):
    with pytest.raises(Ch.CoinError, match="infra's gate refused open: ") as e:
        G.gate_call("research@gate", ["open", "--session", "s1"], run=_run(rc, stdout, stderr))
    assert SECRET[:16] not in str(e.value) and G.HEX_WITHHELD in str(e.value)
