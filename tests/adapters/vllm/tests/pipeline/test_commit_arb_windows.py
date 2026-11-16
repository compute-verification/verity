"""`commit_arb.Windows`, the committer's windows over a stepped engine, on a stand-in committer: a window opens at the first
step under the next seed coin and closes after its steps or when the engine idles, and its line of ``windows.jsonl`` passes only
when the window's Commit and its read-back did."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import verity_vllm.pipeline.commit_arb as CA

ROOT = bytes(range(64))


class Committer:
    """What `Windows` asks of the committer: its step count, and reset, install, finalize and uninstall."""

    def __init__(self, root: bytes = ROOT, errors: tuple[str, ...] = ()) -> None:
        self._step_index, self.root, self.errors = 0, root, list(errors)
        self.calls: list[str] = []

    def reset(self) -> None:
        self._step_index = 0
        self.calls.append("reset")

    def install(self) -> None:
        self.calls.append("install")

    def finalize(self) -> Any:
        self.calls.append("finalize")
        return SimpleNamespace(run_root=self.root, steps=[None] * self._step_index, extra={"errors": self.errors, "errors_n": len(self.errors)})

    def uninstall(self) -> None:
        self.calls.append("uninstall")


class Record:
    """The served record's read-back of a finalized window."""

    def __init__(self, passed: bool = True) -> None:
        self.passed = passed
        self.absorbed: list[int] = []

    def absorb(self, com: Committer, window: int) -> dict[str, Any]:
        self.absorbed.append(window)
        return {"passed": self.passed, "steps": com._step_index}


class Coins:
    def __init__(self) -> None:
        self.flipped = 0

    def record(self) -> dict[str, int]:
        self.flipped += 1
        return {"coin": self.flipped}


def windows(tmp_path: Path, com: Committer, record: Record | None = None, steps: int = 2) -> CA.Windows:
    return CA.Windows(com, record or Record(), tmp_path, steps, Coins())


def lines(tmp_path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (tmp_path / "windows.jsonl").read_text().splitlines()]


def test_a_window_opens_at_the_first_step_and_closes_after_its_steps(tmp_path: Path) -> None:
    com, record = Committer(), Record()
    w = windows(tmp_path, com, record)
    w.before_step()
    assert (w.window, w.is_open, w.seed_coin) == (0, True, {"coin": 1}) and com.calls == ["reset", "install"]
    com._step_index = 1
    w.before_step()
    assert (w.window, w.seed_coin) == (0, {"coin": 1}) and com.calls == ["reset", "install"], "a window holds for its steps"
    com._step_index = 2
    w.before_step()
    assert com.calls == ["reset", "install", "finalize", "uninstall", "reset", "install"] and record.absorbed == [0]
    assert (w.window, w.is_open, w.seed_coin) == (1, True, {"coin": 2}), "the next window opens under the next coin"
    commit = {"passed": True, "run_root": ROOT.hex(), "steps": 2, "errors": [], "errors_n": 0, "identity_check": {}}
    closed = {"window": 0, "commit": commit, "seed_coin": {"coin": 1}, "readback": {"passed": True, "steps": 2}, "passed": True}
    assert w.records == [closed] and lines(tmp_path) == [closed]
    assert (tmp_path / "windows.jsonl").read_text() == json.dumps(closed) + "\n"


def test_a_window_no_step_ran_in_closes_empty_and_passes(tmp_path: Path) -> None:
    com, record = Committer(), Record()
    w = windows(tmp_path, com, record)
    w.before_step()
    w.close()
    assert com.calls == ["reset", "install", "uninstall"] and record.absorbed == [], "nothing to finalize or to read back"
    empty = {"window": 0, "commit": {"passed": True, "steps": 0, "empty": True}, "seed_coin": {"coin": 1}, "readback": None, "passed": True}
    assert w.is_open is False and w.records == [empty] and lines(tmp_path) == [empty]
    w.before_step()
    assert (w.window, w.seed_coin) == (1, {"coin": 2}), "an idle engine's next step opens the next window"


@pytest.mark.parametrize(("com", "record", "commit_passed"), [
    (Committer(root=b""), Record(), False),
    (Committer(errors=("a step's rows differ",)), Record(), False),
    (Committer(), Record(passed=False), True),
])
def test_a_window_fails_when_its_commit_or_its_read_back_did(tmp_path: Path, com: Committer, record: Record, commit_passed: bool) -> None:
    w = windows(tmp_path, com, record)
    w.before_step()
    com._step_index = 1
    w.close()
    (rec,) = w.records
    assert rec["commit"]["passed"] is commit_passed and rec["readback"]["passed"] is record.passed and rec["passed"] is False
    assert lines(tmp_path) == [rec]


def test_closing_no_window_does_nothing(tmp_path: Path) -> None:
    com = Committer()
    w = windows(tmp_path, com)
    w.close()
    assert com.calls == [] and w.records == [] and not (tmp_path / "windows.jsonl").exists()
    w.before_step()
    com._step_index = 1
    w.close()
    w.close()
    assert com.calls == ["reset", "install", "finalize", "uninstall"] and len(w.records) == 1, "a closed window closes once"
