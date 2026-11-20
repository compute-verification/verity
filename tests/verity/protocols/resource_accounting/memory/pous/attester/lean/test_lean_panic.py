"""A Lean verdict is read only where the verifier wrote no panic, and it starts with the runtime's panic switch: the
memory attester's verdict and the wipe's, through the difftest server both launch under. A stand-in that answers accept
beside a panic line is refused by name; the verifiers' Lean is not built here."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import verdict as V
import verity_difftest as D
import wipe_verdict as WV

PANIC_LINE = "PANIC at MemoryChallenger.Judge MemoryChallenger.Judge:1:2: index out of bounds"
ACCEPT = {"accepted": True, "late": 0, "reasons": [], "wrong": 0}


def _fake(tmp_path: Path, body: str) -> D.Server:
    exe = tmp_path / "fake-lean"
    exe.write_text(f"#!{sys.executable}\nimport json, os, sys\n{body}")
    exe.chmod(0o755)
    return D.Server([str(exe)])


def _answer(output: dict, *, panic: bool, exit_code: int = 0) -> str:
    return f"""for line in sys.stdin:
    {'print(%r, file=sys.stderr, flush=True)' % PANIC_LINE if panic else 'pass'}
    print({json.dumps({"output": output}, sort_keys=True, separators=(",", ":"))!r}, flush=True)
    sys.exit({exit_code})
"""


def test_the_memory_attesters_accept_beside_a_panic_is_refused_by_name(tmp_path, monkeypatch):
    server = _fake(tmp_path, _answer(ACCEPT, panic=True))
    monkeypatch.setattr(V, "server", lambda: server)
    monkeypatch.setattr(V, "record", lambda *a, **k: {})
    with pytest.raises(D.ProtocolError) as e:
        V.verdict(None, None, [], lambda i: b"", trusted=set())
    assert str(e.value) == f"{D.LEAN_PANIC}: {PANIC_LINE}"


def test_the_wipes_accept_beside_a_panic_is_refused_by_name(tmp_path, monkeypatch):
    server = _fake(tmp_path, _answer({"accepted": True, "reasons": []}, panic=True))
    monkeypatch.setattr(WV, "server", lambda: server)
    monkeypatch.setattr(WV, "record", lambda *a, **k: {})
    with pytest.raises(D.ProtocolError) as e:
        WV.verdict(None, None, trusted=set())
    assert str(e.value) == f"{D.LEAN_PANIC}: {PANIC_LINE}"


def test_an_accept_from_a_process_that_exits_non_zero_is_refused(tmp_path):
    server = _fake(tmp_path, _answer(ACCEPT, panic=False, exit_code=3))
    try:
        assert server.call("verdict", {})[0] == "output"
        with pytest.raises(D.ProtocolError, match="ended on"):
            server.call("verdict", {})
    finally:
        server.close()


def test_the_server_starts_with_the_panic_switch(tmp_path, monkeypatch):
    monkeypatch.setenv(D.ABORT_ON_PANIC, "0")
    server = _fake(tmp_path, """for line in sys.stdin:
    print(json.dumps({"output": {"switch": os.environ.get("LEAN_ABORT_ON_PANIC")}}, sort_keys=True, separators=(",", ":")), flush=True)
""")
    try:
        assert D.ABORT_ON_PANIC == "LEAN_ABORT_ON_PANIC"
        assert json.loads(server.call("verdict", {})[1]) == {"switch": "1"}
    finally:
        server.close()


def test_a_warning_on_the_error_stream_is_no_panic(tmp_path):
    server = _fake(tmp_path, _answer(ACCEPT, panic=False).replace("    pass\n", "    print('warning: slow', file=sys.stderr, flush=True)\n", 1))
    try:
        assert json.loads(server.call("verdict", {})[1]) == ACCEPT
    finally:
        server.close()
