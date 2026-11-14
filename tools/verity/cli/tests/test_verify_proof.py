"""`verity verify proof ARGS`: the Lean verifier's executable, found below the checkout by `VERIFIER` and `VERIFY_PROOF`,
run with ARGS in the caller's directory and its exit status passed through, against a stand-in executable in a stand-in
checkout, so no Lean build is needed."""
from __future__ import annotations

from pathlib import Path

import pytest

import verity_cli


def stand_in_verifier(root: Path, rc: int) -> Path:
    """An executable where `lake build` writes the verifier's below ``root``, which writes its working directory and
    arguments to the file it returns, and exits with ``rc``."""
    called, exe = root / "called", root / verity_cli.VERIFIER / ".lake" / "build" / "bin" / verity_cli.VERIFY_PROOF
    exe.parent.mkdir(parents=True)
    exe.write_text(f'#!/bin/sh\n{{ pwd -P; for a in "$@"; do echo "$a"; done; }} > "{called}"\nexit {rc}\n')
    exe.chmod(0o755)
    return called


def test_proof_runs_the_verifier_where_the_caller_is_and_passes_its_status(tmp_path, monkeypatch):
    root, here = tmp_path / "checkout", tmp_path / "proofs"
    here.mkdir()
    called = stand_in_verifier(root, 3)
    monkeypatch.setattr(verity_cli, "ROOT", root)
    monkeypatch.chdir(here)
    assert verity_cli.main(["verify", "proof", "verify", "--zk", "--circuit", "c.bin", "--session", "a b"]) == 3
    assert called.read_text().splitlines() == [str(here.resolve()), "verify", "--zk", "--circuit", "c.bin", "--session", "a b"]


def test_proof_names_the_build_when_the_verifier_is_not_built(tmp_path, monkeypatch):
    monkeypatch.setattr(verity_cli, "ROOT", tmp_path)
    with pytest.raises(SystemExit) as e:
        verity_cli.main(["verify", "proof", "verify"])
    exe = f"{verity_cli.VERIFIER}/.lake/build/bin/{verity_cli.VERIFY_PROOF}"
    assert str(e.value.code).startswith(f"verity verify proof: {exe} is not built: ")
    assert f"`lake build {verity_cli.VERIFY_PROOF}`" in str(e.value.code)


def test_the_executable_is_the_one_the_verifiers_lakefile_builds():
    lakefile = (verity_cli.ROOT / verity_cli.VERIFIER / "lakefile.toml").read_text()
    assert f'[[lean_exe]]\nname = "{verity_cli.VERIFY_PROOF}"\n' in lakefile


def test_verify_names_a_command_it_does_not_have_then_its_usage():
    with pytest.raises(SystemExit) as e:
        verity_cli.main(["verify", "remote"])
    assert str(e.value.code) == "verity verify: no command 'remote'\nverity verify proof [ARGS]"
    with pytest.raises(SystemExit) as e:
        verity_cli.main(["verify"])
    assert str(e.value.code) == "verity verify proof [ARGS]"
