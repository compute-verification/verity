"""`research.vqueue_prebuild`: which ready Lean changes the loop builds ahead of their candidates, on which machine, once per
head and `main`'s Lean inputs, and how a detached launch's end is read, on a fake launcher, reader and clock."""
from __future__ import annotations

import json
import os
import subprocess
import sys

from research import vqueue_prebuild as VP
from research.landing import LAUNCH_LOST_S
from research.vqueue import Ready

MAIN, KEY = "m" * 40, "k" * 128


def _ready(n: int, lean: bool = True, conflict: bool = False, head: str | None = None) -> Ready:
    return Ready(n, head or f"{n}" * 40, frozenset({f"security/Proofs/M{n}.lean"}), lean, n, conflict)


class Clock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def _prebuilds(machines=("vy-a", "vy-b"), ends=None, clock=None):
    """`Prebuilds` with a launcher that records each launch's argv and a reader that answers from ``ends`` ({dir: end})."""
    launched, ends = [], ends if ends is not None else {}

    def launch(repo, args):
        launched.append(args)
        return {"dir": f"/launches/{len(launched)}", "pid": 1}

    return VP.Prebuilds(machines, "/clone", now=clock or Clock(), launch=launch, read=lambda l: ends.get(l["dir"])), launched


def _logged(out: list[dict]) -> list[dict]:
    """``out`` as the log holds it: each line through JSON, as `Past.read` reads it back."""
    return [json.loads(json.dumps({"action": "prebuild", **r})) for r in out]


def test_each_ready_lean_change_in_no_candidate_is_built_in_rank_order_one_a_free_machine():
    p, launched = _prebuilds()
    ready = [_ready(1, lean=False), _ready(2, conflict=True), _ready(3), _ready(4), _ready(5), _ready(6)]
    out = p.tick([], ready, held={3}, main=MAIN, key=KEY)
    assert [(r["number"], r["machine"]) for r in out] == [(4, "vy-a"), (5, "vy-b")], \
        "no Lean change, a conflict and a candidate's member are passed over; one a machine"
    assert launched[0] == [sys.executable, str(VP.SCRIPT), MAIN, "4" * 40, "--base", MAIN, "--on", "vy-a", "--mem-gb", "auto",
                           "--repo", "/clone", "--as", VP.LANE]
    assert all(r["lean_key"] == KEY and r["launch"]["at"] == 1000.0 and not r.get("done") for r in out)
    assert p.tick(_logged(out), ready, held={3}, main=MAIN, key=KEY) == [], "both machines run one while they launch"


def test_a_machine_is_free_again_once_its_run_ran_its_time_or_launched_none():
    clock = Clock()
    ends = {}
    p, launched = _prebuilds(ends=ends, clock=clock)
    ready = [_ready(1), _ready(2), _ready(3), _ready(4)]
    log = _logged(p.tick([], ready, set(), MAIN, KEY))
    ends["/launches/1"] = ("0", "lean-prebuild: launched as r20260101-000000-abcd: `research status ...`")
    ends["/launches/2"] = ("0", f"lean-prebuild: 222222222222's change from mmmmmmmmmmmm {VP.NOTHING}: nothing to build")
    out = p.tick(log, ready, set(), MAIN, KEY)
    assert [(r.get("state"), r.get("run"), r.get("machine"), r.get("number")) for r in out] == [
        ("launched", "r20260101-000000-abcd", "vy-a", 1), ("nothing", None, "vy-b", 2), (None, None, "vy-b", 3)], \
        "a launch that built nothing frees its machine at once"
    log += _logged(out)
    assert p.tick(log, ready, set(), MAIN, KEY) == [], "settled once; vy-a's run is counted as running"
    clock.t += VP.RUNS_S
    ends["/launches/3"] = ("1", "lean-prebuild: run r2 was not launched (`research run` exited 1)")
    out = p.tick(log, ready, set(), MAIN, KEY)
    assert [(r.get("state"), r.get("machine"), r.get("number")) for r in out] == [("failed", "vy-b", 3), (None, "vy-a", 4)]
    assert len(launched) == 4


def test_a_head_is_built_once_per_mains_lean_key_and_a_never_built_one_goes_first():
    p, launched = _prebuilds(machines=("vy-a",))
    ends = {}
    p.read = lambda l: ends.get(l["dir"])
    ready = [_ready(1), _ready(2)]
    log = _logged(p.tick([], ready, set(), MAIN, KEY))
    ends["/launches/1"] = ("0", f"... {VP.NOTHING} ...")
    log += _logged(p.tick(log, ready, set(), MAIN, KEY))
    ends["/launches/2"] = ("0", f"... {VP.NOTHING} ...")
    log += _logged(p.tick(log, ready, set(), MAIN, KEY))
    assert [r["number"] for r in log if not r.get("done")] == [1, 2]
    assert p.tick(log, ready, set(), MAIN, KEY) == [], "each head is built against this key"
    moved = p.tick(log, [_ready(1), _ready(2), _ready(9, head="9" * 40)], set(), "n" * 40, "j" * 128)
    assert [(r["number"], r["base"]) for r in moved] == [(9, "n" * 40)], "a head never built goes before one built on a key before"
    assert p.tick(log, ready, set(), MAIN, None) == [], "nothing is due while `main`'s Lean key is unknown"


def test_a_launch_whose_process_is_gone_or_that_printed_nothing_for_too_long_is_lost():
    clock = Clock()
    p, _ = _prebuilds(machines=("vy-a",), clock=clock)
    log = _logged(p.tick([], [_ready(1)], set(), MAIN, KEY))
    assert p.tick(log, [_ready(1)], set(), MAIN, KEY) == []
    clock.t += LAUNCH_LOST_S
    out = p.tick(log, [_ready(1), _ready(2)], set(), MAIN, KEY)
    assert [(r.get("state"), r["number"]) for r in out] == [("lost", 1), (None, 2)]


def test_a_launch_that_raises_is_logged_failed_and_not_tried_again_at_that_key():
    def broken(repo, args):
        raise OSError("no fork")

    p = VP.Prebuilds(["vy-a"], "/clone", now=Clock(), launch=broken, read=lambda l: None)
    out = p.tick([], [_ready(1)], set(), MAIN, KEY)
    assert [(r["state"], r["done"]) for r in out] == [("failed", True)] and "no fork" in out[0]["why"]
    p.launch = lambda repo, args: {"dir": "/d", "pid": 1}
    assert [r["number"] for r in p.tick(_logged(out), [_ready(1), _ready(2)], set(), MAIN, KEY)] == [2]


def test_no_machine_builds_nothing():
    p, launched = _prebuilds(machines=("",))
    assert p.tick([], [_ready(1)], set(), MAIN, KEY) == [] and launched == []


def test_a_detached_launch_is_read_from_its_exit_and_output(tmp_path):
    d = tmp_path / "launch"
    d.mkdir()
    assert VP.ended({"dir": str(d), "pid": os.getpid()}) is None, "running, no exit written"
    gone = subprocess.Popen([sys.executable, "-c", ""])
    gone.wait()
    assert VP.ended({"dir": str(d), "pid": gone.pid}) == ("lost", "")
    (d / "log").write_text("lean-prebuild: launched as r20260101-000000-abcd: `research status r20260101-000000-abcd` says\n")
    (d / "rc").write_text("0\n")
    rc, tail = VP.ended({"dir": str(d), "pid": gone.pid})
    assert VP.outcome(rc, tail) == ("launched", "r20260101-000000-abcd")
    assert VP.outcome("2", "lean-prebuild: abc doesn't merge onto def") == ("failed", None)
    assert VP.outcome("lost", "") == ("lost", None)


def test_the_script_is_the_loops_own_prebuild_with_the_flags_it_reads():
    """The argv `argv` builds is one this tree's `lean_prebuild.py` parses: its `--repo` and `--mem-gb auto`."""
    text = VP.SCRIPT.read_text()
    assert VP.SCRIPT.is_file() and '"--repo"' in text and 'if s == "auto"' in text
    assert "launched as {rid}:" in text and "NOTHING = \"" + VP.NOTHING in text
