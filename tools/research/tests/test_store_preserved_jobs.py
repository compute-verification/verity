"""`research data preserved --jobs N`: targets checked at once, each one's lines printed as it finishes, the report and `--json`
in the given order, and an error while checking one target reported on that target's line."""
from __future__ import annotations

import json
import sys
import threading

import pytest

from research.store import preserved as P
from research.store.cli import main as data_main
from research.store.local import LocalStore, manifest_key
from research.store.remote import FsRemote

from test_store import _attempt
from waits import Gate


@pytest.fixture
def st(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(tmp_path / "store.toml"))
    (tmp_path / "store.toml").write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n')
    return LocalStore()


def _art(st: LocalStore, tmp_path, name: str, push: bool = True) -> str:
    p = tmp_path / f"{name}.txt"
    p.write_text(name)
    a = st.put_artifact("run-files/v1", {"name": name}, file=p)
    if push:
        st.push(a)
    return a


class _Stdout:
    """sys.stdout that calls `on_line(line)` for each complete line written."""

    def __init__(self, on_line):
        self.lines: list[str] = []
        self._buf, self._on_line = "", on_line

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self.lines.append(line)
            self._on_line(line)
        return len(s)

    def flush(self) -> None:
        pass


def _wait(gate: Gate) -> None:
    with open(gate.path, "rb") as f:
        f.read(1)


def _hold_heads(monkeypatch, steps: dict) -> None:
    """HEAD of a key in `steps` first runs `steps[key]()` when a worker thread asks (on the main thread it would wait forever: it
    answers at once, and the order the test asserts then fails)."""
    head = FsRemote.head

    def held(self, k):
        if k in steps and threading.current_thread() is not threading.main_thread():
            steps[k]()
        return head(self, k)

    monkeypatch.setattr(FsRemote, "head", held)


def test_a_fast_targets_line_is_printed_while_a_slow_one_is_still_being_checked(st, tmp_path, monkeypatch):
    slow, fast = _art(st, tmp_path, "slow"), _art(st, tmp_path, "fast")
    slow_go, slow_waiting, state = Gate(tmp_path / "slow-go"), Gate(tmp_path / "slow-waiting"), {}

    def slow_head() -> None:
        state["slow"] = "waiting"
        slow_waiting.release()
        _wait(slow_go)
        state["slow"] = "answered"

    _hold_heads(monkeypatch, {manifest_key(slow[4:]): slow_head, manifest_key(fast[4:]): lambda: _wait(slow_waiting)})
    seen = {}

    def on_line(line: str) -> None:
        if line.startswith(fast):
            seen["slow when fast printed"] = state.get("slow")
            slow_go.release()

    out = _Stdout(on_line)
    monkeypatch.setattr(sys, "stdout", out)
    try:
        rc = data_main(["preserved", slow, fast, "--jobs", "2"])
    finally:
        slow_go.close()
        slow_waiting.close()
    assert rc == 0
    assert seen == {"slow when fast printed": "waiting"}, "fast's line came out while slow's manifest HEAD was still waiting"
    assert [ln.split()[0] for ln in out.lines if ln.startswith("art:")] == [fast, slow], "completion order"
    assert out.lines[-1].startswith("PRESERVED on fs:")


def test_the_report_and_json_keep_the_given_order_and_the_exit_code_whatever_finishes_first(st, tmp_path, monkeypatch, capsys):
    slow, fast = _art(st, tmp_path, "slow"), _art(st, tmp_path, "fast")
    local_only = _art(st, tmp_path, "local", push=False)
    run = "r20261007-040000-0001"
    st.put_attempt(_attempt(run_id=run, outputs={"result": fast}))
    st.push_attempt(run)
    targets = [slow, run, local_only, fast]
    want = P.check(st, targets)

    gate, finished, head = Gate(tmp_path / "gate"), [], FsRemote.head
    _hold_heads(monkeypatch, {manifest_key(slow[4:]): lambda: _wait(gate)})

    def on_target(t: dict) -> None:
        finished.append(t["target"])
        if len(finished) == 3:
            gate.release()

    try:
        rep = P.check(st, targets, jobs=4, on_target=on_target)
    finally:
        gate.close()
    assert finished[-1] == slow and sorted(finished) == sorted(targets)
    assert [t["target"] for t in rep["targets"]] == targets
    assert rep["not_preserved"] == want["not_preserved"] and [x["id"] for x in rep["not_preserved"]] == [local_only]
    assert list(rep["artifacts"]) == list(want["artifacts"]) == [slow, fast, local_only], "fast, named by the run, is checked once"
    assert rep["preserved"] is want["preserved"] is False
    monkeypatch.setattr(FsRemote, "head", head)

    assert data_main(["preserved", *targets, "--json", "--jobs", "4"]) == 1
    doc = json.loads(capsys.readouterr().out)
    assert [t["target"] for t in doc["targets"]] == targets and [x["id"] for x in doc["not_preserved"]] == [local_only]
    assert data_main(["preserved", slow, run, fast, "--jobs", "3"]) == 0
    assert data_main(["preserved", slow, local_only, "--jobs", "1"]) == 1
    text = capsys.readouterr().out
    assert text.count("NOT PRESERVED: 1 item(s)") == 1 and text.count("PRESERVED on fs:") == 1


def test_an_error_while_checking_one_target_is_that_targets_line_and_the_others_still_finish(st, tmp_path, monkeypatch, capsys):
    ok, broken = _art(st, tmp_path, "ok"), _art(st, tmp_path, "broken")
    head, bad_key = FsRemote.head, manifest_key(broken[4:])

    def flaky(self, k):
        if k == bad_key:
            raise ConnectionResetError("peer reset")
        return head(self, k)

    monkeypatch.setattr(FsRemote, "head", flaky)
    assert data_main(["preserved", broken, ok, "--jobs", "2"]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert f"{broken}  NOT preserved: ConnectionResetError: peer reset" in lines
    assert any(ln.startswith(ok) and "PRESERVED (" in ln for ln in lines)
    assert lines[-1].startswith("NOT PRESERVED: 1 item(s)")
