"""A Lean verifier's answer is read only where it wrote no panic, and the door starts it with the runtime's panic switch."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from verity.protocols.proofs.spot.verifier.one_stage import lean_child as LC, proofs as PF, verifier as V
from verity.protocols.proofs.spot.verifier.one_stage.settings import ABORT_ON_PANIC

PANIC_LINE = "PANIC at Flock.verify Flock.Verify:1:2: index out of bounds"


def _fake(tmp_path: Path, body: str) -> Path:
    exe = tmp_path / "fake-lean"
    exe.write_text(f"#!{sys.executable}\nimport json, os, sys\n{body}")
    exe.chmod(0o755)
    return exe


def test_an_accept_beside_a_panic_is_refused_by_name(tmp_path):
    exe = _fake(tmp_path, f"print({PANIC_LINE!r}, file=sys.stderr)\nprint('VERDICT ' + json.dumps({{'accepted': True}}))\n")
    with pytest.raises(LC.LeanPanic) as e:
        LC.run([str(exe)])
    assert str(e.value) == f"{LC.LEAN_PANIC}: {PANIC_LINE}"


def test_a_served_answer_beside_a_panic_is_refused_by_name(tmp_path, monkeypatch):
    exe = _fake(tmp_path, f"""for line in sys.stdin:
    print({PANIC_LINE!r}, file=sys.stderr, flush=True)
    print(json.dumps({{"output": {{"accepted": True}}}}), flush=True)
""")
    monkeypatch.setattr(V, "executable", lambda *_: exe)
    server = V._Server("fake-lean", tmp_path)
    try:
        with pytest.raises(V.LeanError, match=f"^{LC.LEAN_PANIC}: {PANIC_LINE}$"):
            server.call("verify", {})
    finally:
        server.close()


def test_the_child_starts_with_the_panic_switch(tmp_path, monkeypatch):
    monkeypatch.setenv(ABORT_ON_PANIC, "0")
    exe = _fake(tmp_path, f"print(os.environ.get({ABORT_ON_PANIC!r}))\n")
    assert LC.run([str(exe)]).stdout == "1\n"
    assert LC.run([str(exe)], env={"PATH": "/usr/bin"}).stdout == "1\n"
    assert LC.run([str(exe)], env=PF.verifier_env()).stdout == "1\n"
    assert PF.verifier_env({})[ABORT_ON_PANIC] == "1"


def test_a_clean_answer_is_read(tmp_path):
    exe = _fake(tmp_path, "print('warning: slow', file=sys.stderr)\nprint('VERDICT {}')\nsys.exit(3)\n")
    p = LC.run([str(exe)])
    assert (p.returncode, p.stdout) == (3, "VERDICT {}\n")
