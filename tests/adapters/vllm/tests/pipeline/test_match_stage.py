"""One stage of the Match driver run as a command (`verity_vllm.pipeline.match_stage.run_stage`), with the process, `ps` and the
clock stood in, so that no test starts a process or waits: the record of a stage that exits 0 and of one that does not, the
log's first line, what the command is started with, the resident size `ps` gives and its peak, the kill above the memory
limit and past the wall limit, and the stage's marks on the row's timeline."""
from __future__ import annotations

import subprocess
import sys
import types

import pytest

from verity_vllm.pipeline import cli
from verity_vllm.pipeline import match_stage as MS

CMD = ["python", "-m", "a module", "--out", "x"]
SHOWN = "python -m 'a module' --out x"
PS = (["ps", "-o", "rss=", "-p", "4242"], {"capture_output": True, "text": True, "timeout": 5})


def _run(monkeypatch, tmp_path, *, stage="fold", running=0, rc=0, ps=(), clock=(10.0, 10.5), timeline=True, max_rss_gb=None,
         timeout_s=None, cwd=None):
    """`run_stage` on a process that is running at its first ``running`` readings and then has exited ``rc``, with ``ps``'s
    answers in turn (its text, or the error it raises) and the clock's readings in turn.  Gives the record and what was seen:
    how the command was started, what the log held then, what `ps` was asked, each wait, the process's kill and the marks."""
    seen = types.SimpleNamespace(started=[], log_then=[], asked=[], waited=[], process=[], marks=[], log=tmp_path / "logs/of/stage.log")
    answers, readings = list(ps), list(clock)

    class Process:
        pid, returncode = 4242, None

        def poll(self):
            nonlocal running
            if running:
                running -= 1
                return None
            self.returncode = rc
            return rc

        def kill(self):
            seen.process.append("kill")

        def wait(self):
            seen.process.append("wait")
            self.returncode = -9

    def popen(cmd, **kw):
        seen.log_then.append(seen.log.read_text())
        kw["stdout"].write("the stage's output\n")
        seen.started.append((cmd, {**kw, "stdout": kw["stdout"].name}))
        return Process()

    def ps_run(cmd, **kw):
        seen.asked.append((cmd, kw))
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return types.SimpleNamespace(stdout=answer)

    class Timeline:
        @classmethod
        def from_env(cls, name, hot):
            seen.marks.append(("from_env", name, hot))
            if not timeline:
                raise OSError("no directory for the telemetry file")
            return cls()

        def start(self, name):
            seen.marks.append(("start", name))

        def end(self, name, **kw):
            seen.marks.append(("end", name, kw))

    monkeypatch.setattr(MS, "subprocess", types.SimpleNamespace(Popen=popen, run=ps_run, STDOUT=subprocess.STDOUT,
                                                                TimeoutExpired=subprocess.TimeoutExpired))
    monkeypatch.setattr(MS, "time", types.SimpleNamespace(time=lambda: readings.pop(0), sleep=seen.waited.append))
    monkeypatch.setattr(MS, "Timeline", Timeline)
    monkeypatch.setattr(MS, "settings", types.SimpleNamespace(environ=lambda: {"HOME": "/h", "PYTHONUNBUFFERED": "0"}))
    rec = MS.run_stage(stage, CMD, seen.log, max_rss_gb=max_rss_gb, timeout_s=timeout_s, cwd=cwd)
    assert not answers and not readings, "every answer of ps and every reading of the clock is taken"
    return rec, seen


def _record(seen, **change):
    return {"stage": "fold", "cmd": SHOWN, "log": str(seen.log), "status": "ok", "returncode": 0, "reason": None, "seconds": 0.5,
            **change}


def test_a_stage_that_exits_0_is_ok_with_its_command_its_log_and_its_seconds(monkeypatch, tmp_path):
    rec, seen = _run(monkeypatch, tmp_path, running=2, clock=(10.0, 10.123456))
    assert rec == _record(seen, seconds=0.123)
    assert seen.log_then == [f"$ {SHOWN}\n"] and seen.log.read_text() == f"$ {SHOWN}\nthe stage's output\n"
    assert seen.started == [(CMD, {"stdout": str(seen.log), "stderr": subprocess.STDOUT, "cwd": None,
                                   "env": {"HOME": "/h", "PYTHONUNBUFFERED": "1"}})]
    assert seen.waited == [0.25, 0.25] and seen.asked == [] and seen.process == []
    assert seen.marks == [("from_env", "run_config", False), ("start", "match.fold"), ("end", "match.fold", {"rc": 0, "status": "ok"})]


@pytest.mark.parametrize("stage, mark", [("capture", "work.match_capture"), ("control", "work.match_control"), ("resolve", "match.resolve")])
def test_the_capture_and_the_control_are_marked_as_work_and_every_other_stage_as_match(monkeypatch, tmp_path, stage, mark):
    rec, seen = _run(monkeypatch, tmp_path, stage=stage)
    assert rec == _record(seen, stage=stage) and [m[1] for m in seen.marks] == ["run_config", mark, mark]


def test_a_stage_that_exits_with_another_code_is_failed_and_names_its_log(monkeypatch, tmp_path):
    rec, seen = _run(monkeypatch, tmp_path, rc=3, cwd=tmp_path)
    assert rec == _record(seen, status="failed", returncode=3, reason=f"exit 3; see {seen.log}")
    assert seen.started[0][1]["cwd"] == str(tmp_path) and seen.waited == []
    assert seen.marks[-1] == ("end", "match.fold", {"rc": 3, "status": "failed"})


def test_a_stage_runs_unmarked_when_its_timeline_cannot_be_made(monkeypatch, tmp_path):
    rec, seen = _run(monkeypatch, tmp_path, timeline=False)
    assert rec == _record(seen) and seen.marks == [("from_env", "run_config", False)]


def test_under_a_memory_limit_the_record_holds_the_peak_of_what_ps_gives_in_gb(monkeypatch, tmp_path):
    rec, seen = _run(monkeypatch, tmp_path, running=4, max_rss_gb=2.0, ps=["524288\n", "", " 1234567\n", "1048576"])
    assert rec == _record(seen, peak_rss_gb=1.177)
    assert seen.asked == [PS] * 4 and seen.waited == [0.25] * 4 and seen.process == []


@pytest.mark.parametrize("answer", [OSError("no ps"), "no number", subprocess.TimeoutExpired("ps", 5)])
def test_a_reading_ps_does_not_give_is_passed_over(monkeypatch, tmp_path, answer):
    rec, seen = _run(monkeypatch, tmp_path, running=1, max_rss_gb=1.5, ps=[answer])
    assert rec == _record(seen, peak_rss_gb=0.0) and seen.asked == [PS] and seen.process == []


def test_a_stage_above_the_memory_limit_is_killed_and_one_at_the_limit_is_not(monkeypatch, tmp_path):
    rec, seen = _run(monkeypatch, tmp_path, running=9, max_rss_gb=1.5, ps=["1572864", "1677722"])
    assert rec == _record(seen, status="killed_memory", returncode=-9, reason="RSS 1.60 GB > 1.5 GB", peak_rss_gb=1.6)
    assert seen.process == ["kill", "wait"] and seen.waited == [0.25]
    assert seen.marks[-1] == ("end", "match.fold", {"rc": -9, "status": "killed_memory"})


def test_a_stage_past_the_wall_limit_is_killed_and_one_at_the_limit_is_not(monkeypatch, tmp_path):
    rec, seen = _run(monkeypatch, tmp_path, running=9, timeout_s=1.5, clock=(10.0, 11.5, 11.75, 12.0))
    assert rec == _record(seen, status="timeout", returncode=-9, reason="wall > 1.5 s", seconds=2.0)
    assert seen.process == ["kill", "wait"] and seen.waited == [0.25] and seen.asked == []
    assert seen.marks[-1] == ("end", "match.fold", {"rc": -9, "status": "timeout"})


@pytest.mark.parametrize("limit, text", [(180.0, "180"), (180, "180"), (1.5, "1.5"), (0.25, "0.25"), (1e6, "1000000"), (2.5e-7, "2.5e-07")])
def test_the_wall_limit_is_written_as_it_was_given_a_whole_number_as_an_integer(monkeypatch, tmp_path, limit, text):
    rec, _seen = _run(monkeypatch, tmp_path, running=9, timeout_s=limit, clock=(10.0, 11.0 + 2 * limit, 12.0 + 2 * limit))
    assert rec["status"] == "timeout" and rec["reason"] == f"wall > {text} s"


@pytest.mark.parametrize("given", ["180", repr(sys.float_info.max), "inf"])
def test_the_laptop_limit_and_the_largest_the_parser_admits_read_as_they_did(given):
    """180 s is `--laptop`'s limit; `--stage-timeout` admits any float, the largest finite one and `inf` among them."""
    limit = cli.parse("match", ["--stage-timeout", given], {}).stage_timeout
    assert limit == float(given) and MS._limit(limit) == f"{limit:.0f}"


def test_with_both_limits_the_memory_is_read_before_the_wall_in_each_pass(monkeypatch, tmp_path):
    rec, seen = _run(monkeypatch, tmp_path, running=9, max_rss_gb=1.0, timeout_s=1.0, ps=["2097152"], clock=(10.0, 20.0))
    assert rec == _record(seen, status="killed_memory", returncode=-9, reason="RSS 2.00 GB > 1.0 GB", peak_rss_gb=2.0, seconds=10.0)
    rec, seen = _run(monkeypatch, tmp_path, running=9, max_rss_gb=1.0, timeout_s=1.0, ps=["524288"], clock=(10.0, 12.0, 12.5))
    assert rec == _record(seen, status="timeout", returncode=-9, reason="wall > 1 s", peak_rss_gb=0.5, seconds=2.5)
    assert seen.process == ["kill", "wait"] and seen.waited == []
