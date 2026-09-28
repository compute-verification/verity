"""The node-2 adapter: `gpu-lease`'s owner lines, wait files and `preempted.log` as the model's allocations and queue, read
live from the lock files (never through a lock) or from the node's sampler log."""
from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from cluster import agent, model, nebius2, shadow
from cluster.ledger import epoch

DESC = Path(__file__).parents[1] / "descriptions" / "nebius.toml"
FIX = Path(__file__).parent / "fixtures"
GPU_LEASE = FIX / "gpu-lease-0d172cf3"
AGENT_GPU_LEASE = FIX / "gpu-lease-49238797"

FILL = ("pid=448349 run=- who=fill:bc-2aa33ad8 uuid=GPU-af0b since=2026-09-30T08:15:12Z until=2026-09-30T08:40:12Z mem=193G "
        "preempt=1 scope=gpu-lease-448349 cmd=nice -n 19 taskset -c 96-127 bash /workspace/pouw/fill/running/x.sh a=b")
TIMED = ("pid=2074007 run=r20260930-174917-2585 who=bc-7442ca43 uuid=GPU-2b59 since=2026-09-30T17:50:29Z "
         "until=2026-09-30T18:10:29Z mem=1544G timed=1 scope=gpu-lease-2074007 cmd=timeout 1140 bash run.sh")
SESSION = "pid=13484 run=r20260930-060930-5d4e since=2026-09-30T06:09:49Z"


@pytest.fixture(scope="module")
def c() -> model.Cluster:
    return model.load(DESC)


def test_the_fixture_is_the_gpu_lease_live_on_node_2():
    assert hashlib.sha256(GPU_LEASE.read_bytes()).hexdigest().startswith("0d172cf31856e5da")


def test_owner_lines_become_pinned_cpu_unmanaged_allocations_by_the_cutover_mapping(c):
    obs = nebius2.Observation(epoch("2026-09-30T08:20:00Z"), {3: FILL, 5: SESSION})
    fill, sess = sorted(nebius2.allocations(c, obs), key=lambda a: a.job.priority)
    assert (fill.id, fill.gpus, fill.cpus, fill.memory_gib) == (f"gl-448349@{epoch('2026-09-30T08:15:12Z')}", (3,), frozenset(), 193)
    assert (fill.job.priority, fill.job.kind, fill.job.preempt, fill.job.max_s) == ("fill", "batch", "requeue", 1500)
    assert fill.job.submitter == "fill:bc-2aa33ad8" and fill.job.project == "pous" and fill.job.gpu_ids == (3,)
    assert nebius2.fields(FILL)["cmd"].endswith("x.sh a=b")
    assert (sess.job.priority, sess.job.kind, sess.job.preempt, sess.job.max_s, sess.job.expected_s) == \
        ("session", "session", "never", None, 1800)
    assert not sess.job.problems(c) and not fill.job.problems(c)


def test_a_timed_lease_or_one_holder_of_every_gpu_is_a_quiet_window(c):
    one = nebius2.Observation(0, {0: TIMED})
    [w] = nebius2.allocations(c, one)
    assert w.job.quiet and w.job.priority == "timed" and w.job.preempt == "never"
    assert nebius2.timed_now(c, one)
    whole = nebius2.Observation(0, {i: SESSION for i in range(8)})
    [a] = nebius2.allocations(c, whole)
    assert a.gpus == tuple(range(8)) and a.job.quiet and nebius2.timed_now(c, whole)
    assert not nebius2.timed_now(c, nebius2.Observation(0, {i: SESSION for i in range(7)}))


def test_a_waiter_is_a_queued_job_and_a_whole_node_waiter_a_timed_window(c):
    obs = nebius2.Observation(0, {2: FILL}, (("wait.1.147369", "pid=147369 n=8 run=r-1 who=research since=2026-09-30T06:44:04Z"),
                                            ("wait.2.99", "pid=99 n=2 on=4,5 run=- who=bc-x since=2026-09-30T06:45:00Z"),
                                            ("wait.3.448349", "pid=448349 n=1 run=- who=w since=2026-09-30T06:45:00Z")))
    (w, h1), (p, h2) = nebius2.queue(c, obs)
    assert (w.id, w.gpus, w.quiet, w.submitted, h1) == ("gl-147369", 8, True, epoch("2026-09-30T06:44:04Z"), "wait.1.147369")
    assert (p.id, p.gpus, p.gpu_ids, p.quiet, p.priority) == ("gl-99", 2, (4, 5), False, "session")


def test_preempted_log_lines_parse_to_the_requester_and_the_stopped_lease():
    line = ("2026-09-30T07:46:27Z SIGTERM GPU 4 for pid 384594 (n=8, who=bc-0de2d624): " + FILL)
    p = nebius2.preemption(line)
    assert (p["t"], p["signal"], p["gpu"], p["by"], p["requester"]) == (epoch("2026-09-30T07:46:27Z"), "SIGTERM", 4, 384594,
                                                                        "n=8, who=bc-0de2d624")
    assert p["victim"]["pid"] == "448349" and p["victim"]["preempt"] == "1"
    assert nebius2.preemption("garbage") is None


def test_a_lock_is_read_from_its_holders_fdinfo_and_a_stale_owner_line_is_not_a_lease(c, tmp_path):
    """A fake /proc: pid 7 holds 0.lock (its fdinfo lists the lock), pid 8 has 1.lock open without a lock (a take that
    failed), and 2.owner names pid 9, which is gone (the line a new holder hasn't overwritten yet)."""
    gl, proc = tmp_path / "gl", tmp_path / "proc"
    gl.mkdir()
    for i, pid in ((0, 7), (1, 8), (2, 9)):
        (gl / f"{i}.lock").touch()
        (gl / f"{i}.owner").write_text(f"pid={pid} run=- who=w{pid} since=2026-09-30T08:00:00Z\n")
    for pid, locked in ((7, True), (8, False)):
        (proc / str(pid) / "fd").mkdir(parents=True)
        (proc / str(pid) / "fdinfo").mkdir()
        (proc / str(pid) / "fd" / "10").symlink_to(gl / f"{pid - 7}.lock")
        (proc / str(pid) / "fdinfo" / "10").write_text("pos:\t0\nflags:\t0102001\n" +
                                                         ("lock:\t1: FLOCK  ADVISORY  WRITE 0 fe:20:1 0 EOF\n" if locked else ""))
    obs = nebius2.read(0, gl, proc)
    assert list(obs.leases) == [0] and obs.waits == () and obs.held == ()


def _lock_id(path: Path) -> str:
    st = os.stat(path)
    return f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}"


def test_a_lock_held_under_a_dead_owners_line_is_held_not_free(c, tmp_path):
    """2.owner names pid 9, which is gone, but /proc/locks still lists 2.lock (a child kept the descriptor): the GPU is busy
    and nobody's to move, never free. A blocked request (`->`) on 1.lock holds nothing."""
    gl, proc = tmp_path / "gl", tmp_path / "proc"
    gl.mkdir()
    proc.mkdir()
    for i in (1, 2):
        (gl / f"{i}.lock").touch()
        (gl / f"{i}.owner").write_text("pid=9 run=- who=w9 since=2026-09-30T08:00:00Z\n")
    (proc / "locks").write_text(f"1: POSIX  ADVISORY  WRITE 5 {_lock_id(gl / '1.lock')} 0 EOF\n"
                                f"2: FLOCK  ADVISORY  WRITE 4242 {_lock_id(gl / '2.lock')} 0 EOF\n"
                                f"2: -> FLOCK  ADVISORY  WRITE 4243 {_lock_id(gl / '1.lock')} 0 EOF\n")
    obs = nebius2.read(0, gl, proc)
    assert obs.leases == {} and obs.held == (2,)
    sh = shadow.Shadow(c, tmp_path / "out")
    sh.step(obs)
    assert [a.gpus for a in sh.unowned] == [(2,)]


@pytest.mark.skipif(not Path("/proc/locks").exists(), reason="needs /proc/locks")
def test_locked_reads_the_kernels_proc_locks(tmp_path):
    p, q = tmp_path / "0.lock", tmp_path / "1.lock"
    p.touch()
    q.touch()
    fd = os.open(p, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert nebius2.locked({p, q}) == {p}
    finally:
        os.close(fd)
    assert nebius2.locked({p, q}) == set()


def test_a_sampler_line_is_an_observation():
    rec = {"t": "2026-09-30T08:20:00Z", "gpus": [{"i": 0, "lease": FILL}, {"i": 1, "lease": None}],
           "wait_files": ["pid=5 n=1 run=- who=a since=2026-09-30T08:19:00Z"]}
    obs = nebius2.from_sampler(rec)
    assert obs.leases == {0: FILL} and obs.waits == (("wait.5", rec["wait_files"][0]),) and obs.waits_known
    assert not nebius2.from_sampler({"t": rec["t"], "gpus": []}).waits_known


needs_flock = pytest.mark.skipif(shutil.which("flock") is None or not Path("/proc/self/fdinfo").exists(),
                                 reason="gpu-lease needs bash and flock(1); read() needs /proc/<pid>/fdinfo")


@needs_flock
def test_a_lease_holding_its_gpu_with_wait_in_its_argv_is_an_allocation_and_only_a_queued_request_is_a_waiter(c, tmp_path):
    """13f402b2's waiter-versus-holder case against the adapter and the gpu-lease live on node 2: a `gpu-lease 1 --wait`
    holding its GPU keeps `--wait` in its argv but is an allocation, never a waiter; a queued request is a waiter; a wait
    file nobody holds (its waiter died) is neither. The adapter reads locks from the holders' fdinfo, so it never holds one."""
    gl = tmp_path / "gl"
    env = {k: v for k, v in os.environ.items() if k != "RESEARCH_RUN_DIR"}
    env.update(GPU_LEASE_DIR=str(gl), GPU_LEASE_ALLOWED_FILE=str(tmp_path / "no-direct-gpus"), GPU_LEASE_WINDOWS=str(tmp_path / "no-windows"), GPU_LEASE_COUNT="1", RESEARCH_RUN_ID="r-test", GPU_LEASE_WHO="bc-holder",
               GPU_LEASE_UUID_TABLE="GPU-a0", GPU_LEASE_USAGE_LOG=str(tmp_path / "usage.jsonl"),
               GPU_LEASE_UTIL_LOG=str(tmp_path / "util-*.jsonl"))
    pause = [sys.executable, "-c", "import signal; print('ready', flush=True); signal.pause()"]
    holder = subprocess.Popen(["bash", str(GPU_LEASE), "1", "--wait", "--mem-gb", "0", "--", *pause], env=env,
                              stdout=subprocess.PIPE, text=True, start_new_session=True)
    waiter = None
    try:
        assert holder.stdout.readline().strip() == "ready"
        (gl / "wait.1.99999").write_text("pid=99999 n=1 run=- who=gone since=2026-09-30T00:00:00Z\n")
        obs = nebius2.read(0, gl)
        [a] = nebius2.allocations(c, obs)
        assert a.id.startswith(f"gl-{holder.pid}@") and a.gpus == (0,) and a.job.submitter == "bc-holder"
        assert nebius2.queue(c, obs) == []
        assert (gl / "wait.1.99999").exists(), "the adapter reads; it removes nothing (gpu-lease's next reader does)"
        waiter = subprocess.Popen(["bash", str(GPU_LEASE), "1", "--wait", "--mem-gb", "0", "--", "true"],
                                  env={**env, "GPU_LEASE_POLL_S": "3600", "GPU_LEASE_WHO": "bc-waiter"},
                                  stderr=subprocess.PIPE, text=True, start_new_session=True)
        assert waiter.stderr.readline().startswith("gpu-lease: waiting for 1 GPU(s), queued as wait.")
        obs = nebius2.read(0, gl)
        assert [a.id for a in nebius2.allocations(c, obs)] == [a.id]
        [(j, handle)] = nebius2.queue(c, obs)
        assert (j.id, j.submitter, j.gpus) == (f"gl-{waiter.pid}", "bc-waiter", 1) and handle.startswith("wait.")
    finally:
        for p in (waiter, holder):
            if p is not None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)    # the waiter's bash alone would leave its `sleep 3600` orphaned
                except ProcessLookupError:
                    pass
                p.wait()


@needs_flock
def test_live_mode_grants_through_gpu_lease_agent_mode_and_observes_the_lease_it_granted(c, tmp_path):
    """The agent-mode protocol end to end, against the gpu-lease held for deploy: the request's wait file is the planner's
    job (max= as max_s), the agent's grant is the lease gpu-lease takes, and its next reading is that allocation."""
    assert hashlib.sha256(AGENT_GPU_LEASE.read_bytes()).hexdigest().startswith("49238797233d96ec")
    # the wait file's since= is gpu-lease's real clock, so the node must still admit then, whatever nebius.toml's stops_at says
    c = dataclasses.replace(c, nodes={n: dataclasses.replace(nd, stops_at=None) for n, nd in c.nodes.items()})
    gl = tmp_path / "gl"
    gl.mkdir()
    lock = os.open(gl / "agent.lock", os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = {k: v for k, v in os.environ.items() if k != "RESEARCH_RUN_DIR"}
    env.update(GPU_LEASE_DIR=str(gl), GPU_LEASE_ALLOWED_FILE=str(tmp_path / "no-direct-gpus"), GPU_LEASE_WINDOWS=str(tmp_path / "no-windows"), GPU_LEASE_COUNT="8", GPU_LEASE_AGENT_POLL_S="0.05", RESEARCH_RUN_ID="r-test",
               GPU_LEASE_WHO="bc-asker", GPU_LEASE_USAGE_LOG=str(tmp_path / "usage.jsonl"),
               GPU_LEASE_UTIL_LOG=str(tmp_path / "util-*.jsonl"))
    pause = [sys.executable, "-c", "import signal; print('ready', flush=True); signal.pause()"]
    p = subprocess.Popen(["bash", str(AGENT_GPU_LEASE), "2", "--wait", "--max-min", "5", "--mem-gb", "0", "--", *pause], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert "from the node's agent, queued as wait." in p.stderr.readline()
        obs = nebius2.read(0, gl)
        [(j, _)] = nebius2.queue(c, obs)
        assert (j.id, j.gpus, j.max_s, j.priority) == (f"gl-{p.pid}", 2, 300, "session")
        t = j.submitted + 1
        sh = shadow.Shadow(c, tmp_path / "out")
        sh.step(nebius2.read(t, gl))
        agent._carry_out(c, sh, obs, agent.Executor(gl), t)
        granted = tuple(int(x) for x in (gl / f"grant.{p.pid}").read_text().split(","))
        assert len(granted) == 2
        assert p.stdout.readline().strip() == "ready"
        obs = nebius2.read(t + 1, gl)
        [a] = nebius2.allocations(c, obs)
        assert a.job.id == j.id and a.gpus == granted and a.job.max_s == 300 and nebius2.queue(c, obs) == []
        assert not (gl / f"grant.{p.pid}").exists()
        sh.step(obs)
        assert sh.state.allocations[a.id].gpus == granted
    finally:
        p.terminate()
        p.wait()
        os.close(lock)


def test_a_waiter_pinned_to_more_gpus_than_it_asks_for_is_granted_only_from_them(c, tmp_path):
    """5:55 PM PDT on 30 Sep: a `gpu-lease 1 --on 2,3,4,5,6,7` request read without its `on=` was granted GPU 0, which gpu-lease
    can't take, and the node sat idle behind it. `on=` naming more GPUs than the request is its pool."""
    from cluster.plan import Start, plan
    line = "pid=355694 n=1 on=2,3,4,5,6,7 run=r1 who=w since=2026-10-01T00:51:49Z max=20 preempt=1"
    [(j, _)] = nebius2.queue(c, nebius2.Observation(0, {}, (("wait.1.355694", line),)))
    assert (j.gpus, j.gpu_ids, j.gpu_pool) == (1, (), (2, 3, 4, 5, 6, 7)) and not j.problems(c)
    [s] = [a for a in plan(c, "vy-nebius-2", [], [j], epoch("2026-10-01T00:52:00Z")) if isinstance(a, Start)]
    assert s.alloc.gpus[0] in (2, 3, 4, 5, 6, 7)
    exact = "pid=7 n=2 on=3,5 run=r2 who=w since=2026-10-01T00:51:49Z"
    [(k, _)] = nebius2.queue(c, nebius2.Observation(0, {}, (("wait.1.7", exact),)))
    assert (k.gpu_ids, k.gpu_pool) == ((3, 5), ())
