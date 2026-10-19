"""`research cpu-hold`: a queued CPU run on node 2 gets whole cores of its share of its own, claimed under a lock its run holds, and
waits in a line while none are free. Node 2's queued runs all had all of 48-95, 34 at once at load 1000 (19 Oct). The topology
is files under tmp_path, laid out as node 2's (adjacent SMT pairs); waits run on an injected sleep."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from research import cores, remote
from waits import wait_exit, wait_runner
from test_run_launched import _request

PY = sys.executable


@pytest.fixture
def node(tmp_path, monkeypatch):
    root = tmp_path / "cpu"
    for c in range(192):
        t = root / f"cpu{c}" / "topology"
        t.mkdir(parents=True)
        (t / "thread_siblings_list").write_text(f"{c & ~1}-{c | 1}\n")
    d = tmp_path / "claims"
    d.mkdir()
    monkeypatch.setenv("RESEARCH_CPU_CLAIMS_DIR", str(d))
    monkeypatch.setenv("RESEARCH_CPU_ROOT", str(root))
    return d, root


SHARE = set(range(48, 96))


def test_claims_are_whole_cores_no_two_runs_share_and_an_ended_runs_cores_come_back(node):
    d, root = node
    a, fa = cores.take(4, SHARE, "r1", d, root)
    b, fb = cores.take(3, SHARE, "r2", d, root)                     # three CPUs are two whole cores
    assert (cores.fmt(a), cores.fmt(b)) == ("48-51", "52-55")
    assert cores.carve({48, 50, 51}, 2, root) == {50, 51} and cores.carve({48, 50, 51}, 3, root) is None
    assert cores.claimed(d) == {"r1": a, "r2": b}
    assert cores.take(4, SHARE, "r1", d, root) is None, "a run claims once"
    assert cores.take(41, SHARE, "r3", d, root) is None                # 40 CPUs are left
    os.close(fa)                                                        # r1's run ended
    assert cores.claimed(d) == {"r2": b} and not (d / "r1.cpus").exists()
    c, fc = cores.take(4, SHARE, "r3", d, root)
    assert cores.fmt(c) == "48-51"
    os.close(fb), os.close(fc)


def test_a_claim_lives_as_long_as_anything_the_run_started_holds_its_descriptor(node):
    """The runner holds the claim through custody: a child that inherited the descriptor keeps it after this process lets go."""
    d, root = node
    got, fd = cores.take(2, SHARE, "r1", d, root)
    os.set_inheritable(fd, True)
    child = subprocess.Popen(["cat"], stdin=subprocess.PIPE, pass_fds=(fd,))
    os.close(fd)
    assert cores.claimed(d) == {"r1": {48, 49}}
    child.stdin.close()
    child.wait()
    assert cores.claimed(d) == {}


def test_a_run_waits_in_line_while_its_cores_are_taken_and_a_later_one_never_passes_it(node):
    d, root = node
    _, fa = cores.take(44, SHARE, "big", d, root)                       # 4 CPUs left: 92-95
    held, logs, polls = [], [], []

    class Behind(Exception):
        pass

    def behind(s):
        assert [p.name.split(".")[-1] for p in cores.live(d, "wait.*")] == ["r8", "small"], "it joined the line behind r8"
        raise Behind

    def sleep(s):
        polls.append(s)
        if len(polls) == 1:
            # a 2-CPU run arrives while the 8-CPU one waits: 2 CPUs are free, but it waits behind the first waiter
            with pytest.raises(Behind):
                cores.hold(2, SHARE, "small", d=d, root=root, sleep=behind, clock_ns=lambda: 2, log=lambda s: None)
            assert set(cores.claimed(d)) == {"big"}
        if len(polls) == 2:
            os.close(fa)

    got, fd = cores.hold(8, SHARE, "r8", d=d, root=root, poll_s=15, sleep=sleep, clock_ns=lambda: 1, log=logs.append,
                         on_hold=lambda: held.append(1))
    assert cores.fmt(got) == "48-55" and polls == [15, 15] and held == [1]
    assert "waiting for 8 CPUs in whole cores of 48-95: 1 claim(s) hold 48-91, 0 waiter(s) ahead" in logs[0]
    assert "after 2 poll(s)" in logs[-1]
    assert cores.live(d, "wait.*") == [], "its ticket goes once it holds cores"
    os.close(fd)


def test_the_oldest_live_waiter_goes_first_and_a_dead_ones_ticket_is_dropped(node):
    d, root = node
    ahead = cores.locked(d / "wait.00000000000000000001.other", "4\n")
    order = []

    def sleep(s):
        order.append("polled")
        os.close(ahead)                                                  # the waiter ahead gave up

    got, fd = cores.hold(2, SHARE, "r2", d=d, root=root, sleep=sleep, clock_ns=lambda: 2, log=lambda s: None)
    assert order == ["polled"] and cores.fmt(got) == "48-49" and cores.live(d, "wait.*") == []
    os.close(fd)
    with pytest.raises(ValueError, match="don't fit in 48-95"):
        cores.hold(50, SHARE, "huge", d=d, root=root, sleep=lambda s: pytest.fail("never waits"), log=lambda s: None)


def test_cpu_hold_puts_the_claimed_cores_in_the_wrapper_alone_and_leaves_the_claim_to_it(node):
    d, _ = node
    ran = []
    wrap = ["systemd-run", "-p", "AllowedCPUs=48-95", "--", "sh", "-c", "check", "cpuset-check", "48-95", "taskset", "-c", "48-95"]
    rc = cores.main(["--cpus", "4", "--from", "48-95", "--wrap", str(len(wrap)), "--who", "r1", "--", *wrap,
                     "research", "run", "--", "taskset", "-c", "48-95", "bench.sh"],
                    sleep=lambda s: pytest.fail("cores are free"), execvp=lambda f, a: ran.append(a))
    [argv] = ran
    assert rc == 127 and argv[:len(wrap)] == ["systemd-run", "-p", "AllowedCPUs=48-51", "--", "sh", "-c", "check", "cpuset-check",
                                              "48-51", "taskset", "-c", "48-51"]
    assert argv[len(wrap):] == ["research", "run", "--", "taskset", "-c", "48-95", "bench.sh"], "the workload's own pin is its own"
    [fd] = [fd for fd, path in open_files() if path == str(d / "r1.cpus")]
    assert os.get_inheritable(fd) and (d / "r1.cpus").read_text() == "48-51\n"
    os.close(fd)
    assert cores.main(["--cpus", "64", "--from", "48-95", "--wrap", "0", "--who", "r2", "--", "true"],
                      execvp=lambda f, a: pytest.fail("never runs")) == 2


def open_files() -> list[tuple[int, str]]:
    out = []
    for f in os.listdir("/proc/self/fd"):
        try:
            out.append((int(f), os.readlink(f"/proc/self/fd/{f}")))
        except OSError:
            continue
    return out


def _queued(tmp_path, run_id, record):
    """A launch request whose queue wrapper records the arguments it was given up to `--`, then runs the rest."""
    code = "import os, sys; i = sys.argv.index('--'); open(sys.argv[1], 'w').write(repr(sys.argv[2:i])); os.execvp(sys.argv[i + 1], sys.argv[i + 1:])"
    req, runs_root = _request(tmp_path, run_id, [PY, "-c", "print('ran')"])
    doc = json.loads(req.read_text())
    doc[remote.QUEUE_WRAP] = [PY, "-c", code, str(record), "AllowedCPUs=48-95", "48-95", "--"]
    doc["queue"] = {"cores": {"cpus": "48-95", "n": 4}}
    req.write_text(json.dumps(doc))
    return req, runs_root


def test_a_queued_run_launches_on_cores_of_its_own_and_one_that_must_wait_counts_as_launched(tmp_path, node):
    d, root = node
    req, runs_root = _queued(tmp_path, "q1", tmp_path / "q1.args")
    out = remote.launch_request(req)
    assert out["started"] and out["first_record"] == "job.json"
    wait_runner(runs_root / "q1")
    assert (tmp_path / "q1.args").read_text() == repr(["AllowedCPUs=48-51", "48-51"])
    assert "research cpu-hold: run q1 on CPUs 48-51" in (runs_root / "q1" / "launcher.log").read_text()

    _, fd = cores.take(48, SHARE, "all", d, root)
    req, runs_root = _queued(tmp_path, "q2", tmp_path / "q2.args")
    out = remote.launch_request(req)
    try:
        assert out["started"] and out["first_record"] == "cpu-hold" and not (runs_root / "q2" / "job.json").exists()
        assert "research cpu-hold: waiting for 4 CPUs" in (runs_root / "q2" / "launcher.log").read_text()
    finally:
        os.killpg(out["runner_pid"], signal.SIGTERM)
        wait_exit(out["runner_pid"])
        os.close(fd)
