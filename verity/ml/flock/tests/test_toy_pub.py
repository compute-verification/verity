"""`flock-witness toy-pub`, `Flock.Test.ToyPub.check` compiled: the toy `verify --zk` session's files
(`Flock.Test.ToyPubFiles`) parse, load and draw to what that module states, by the verifier's own functions. The
Security analysis's toy-file premise is this check's conclusion; nothing cheaper runs the parse, which the kernel can't."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

LEAN = Path(__file__).resolve().parents[3] / "core" / "service" / "flock"
BIN = LEAN / ".lake" / "build" / "bin" / "flock-witness"
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")


def test_toy_pub_check():
    subprocess.run(["lake", "build", "flock-witness"], cwd=LEAN, check=True, capture_output=True)
    p = subprocess.run([str(BIN), "toy-pub"], capture_output=True, text=True)
    assert p.returncode == 0 and p.stdout.strip() == "TOYPUB true", p.stdout + p.stderr
