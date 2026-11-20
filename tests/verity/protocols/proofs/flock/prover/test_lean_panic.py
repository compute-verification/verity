"""Flock's distribution reads a Lean executable's answer only where it wrote no panic, and starts it with the runtime's
panic switch."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from verity_flock import lean_child as LC, rec_live as RL

POD = Path(__file__).resolve().parents[6] / "verity" / "protocols" / "proofs" / "flock" / "prover" / "pod"
sys.path[:0] = [str(POD), str(POD.parent / "agreement")]
import agree as AG  # noqa: E402
import lean_verdicts as LV  # noqa: E402

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


def test_the_child_starts_with_the_panic_switch(tmp_path, monkeypatch):
    monkeypatch.setenv(LC.ABORT_ON_PANIC, "0")
    exe = _fake(tmp_path, f"print(os.environ.get({LC.ABORT_ON_PANIC!r}))\n")
    assert LC.run([str(exe)]).stdout == "1\n"
    assert LC.run([str(exe)], env={"PATH": "/usr/bin"}).stdout == "1\n"


def test_the_live_zk_servers_answer_beside_a_panic_is_refused_by_name(tmp_path):
    exe = _fake(tmp_path, f"""assert sys.argv[1:] == ["live"] and os.environ.get({LC.ABORT_ON_PANIC!r}) == "1"
for line in sys.stdin:
    print({PANIC_LINE!r}, file=sys.stderr, flush=True)
    print(json.dumps({{"ok": True}}), flush=True)
""")
    with pytest.raises(ValueError, match=f"^{LC.LEAN_PANIC}: {PANIC_LINE}$"):
        RL.LeanZkServer(exe, {})


def test_a_composition_accepted_beside_a_panic_is_neither_accepted_nor_refused(tmp_path):
    exe = _fake(tmp_path, f"""assert os.environ.get({LC.ABORT_ON_PANIC!r}) == "1"
print({PANIC_LINE!r}, file=sys.stderr)
print("REC-COMPOSE " + json.dumps({{"accepted": True}}))
""")
    accepted, line = LV.compose(str(exe), tmp_path, [])
    assert (accepted, line) == (None, f"{LC.LEAN_PANIC}: {PANIC_LINE}")


def test_the_agreements_lean_verdict_beside_a_panic_is_not_read(tmp_path, monkeypatch):
    lean = _fake(tmp_path, f"""assert os.environ.get({LC.ABORT_ON_PANIC!r}) == "1"
print({PANIC_LINE!r}, file=sys.stderr)
for s in sys.argv[1:]:
    print("VERDICT " + json.dumps({{"session": s, "accepted": True, "why": ""}}))
""")
    session = str(tmp_path / "s")
    monkeypatch.setattr(AG, "groups", lambda a: {("pub.bin", None): [session]})
    monkeypatch.setattr(AG, "lean_cmd", lambda a, public, registered=None: [str(lean)])
    a = SimpleNamespace(sessions=[session], upstream=AG.LIVE, zk=True, jobs=1, out=tmp_path)
    (tmp_path / "s").mkdir()
    (tmp_path / "s" / "case.json").write_text("{}")
    got, _up = AG.verdicts(a)
    assert got == {}
    assert f"{LC.LEAN_PANIC}: {PANIC_LINE}" in (tmp_path / "lean.err").read_text()
