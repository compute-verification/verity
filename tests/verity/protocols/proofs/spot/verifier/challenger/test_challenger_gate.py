"""Infra's gate as its clients call it (`challenger_gate.gate_call`): a refusal quotes no session's secret, whichever of the
gate's reason, ssh's stderr or the answer it quotes; a gate is ``USER@HOST`` over ssh or a command on this machine."""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import sys

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


def test_a_user_at_host_gate_goes_over_ssh_and_a_command_gate_takes_the_verb_as_its_words():
    args = ["reserve", "--families", '[["w", 4]]']
    assert G.gate_argv("research@gate", args) == [*G.SSH, "research@gate", shlex.join(args)]
    local = [sys.executable, "-I", "local_gate.py", "/gate"]
    assert G.gate_argv(local, args) == [*local, *args]
    pearl = ["ssh", "-i", "/k", "research@gate"]
    assert G.gate_argv(pearl, args) == [*pearl, shlex.join(args)]
    assert G.command_of(f"command:{shlex.join(local)}") == local
    assert G.command_of("gate:research@gate") is None and G.command_of("command:") is None


def _gate(tmp_path, answer: dict) -> list[str]:
    script = tmp_path / "gate.py"
    script.write_text(f"import json, sys\nassert sys.argv[1:] == ['pubkey']\nprint(json.dumps({answer!r}))\n")
    return [sys.executable, "-I", str(script)]


def test_a_command_gate_s_key_is_its_own_pubkey_and_an_ssh_gate_s_is_only_its_record(tmp_path):
    from verity.protocols.proofs.spot.verifier.challenger import challenger_session as CS

    public = bytes(range(Ch.PUBLIC_BYTES))
    doc = {"W": 4, "H": 5, "public": public.hex(), "public_sha512": hashlib.sha512(public).hexdigest()}
    assert CS.GateClient(_gate(tmp_path, doc), None, "local", None, []).key == Ch.ChallengerKey(4, 5, public)
    with pytest.raises(Ch.CoinError, match="SHA-512 is not the one it gives"):
        CS.GateClient(_gate(tmp_path, {**doc, "public_sha512": "0" * 128}), None, "local", None, [])
    with pytest.raises(Ch.CoinError, match="held to its deployment's key record"):
        CS.GateClient("research@gate", None, "vy-dc-h200", None, [])
