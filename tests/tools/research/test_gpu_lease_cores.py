"""gpu-lease's CPU quota check (`--cores`), run as the script on a made-up machine: a lease directory whose locks the test holds, a
cpu.max with its cpu.stat, a /proc (the leases' processes, and the CPUs a request may run on) and a `taskset` that records what
it was given. No GPU, and nothing waits on a clock."""
from __future__ import annotations

import fcntl
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from research.pods import registry

GPU_LEASE = Path(registry.__file__).resolve().parent / "sh" / "gpu_lease.sh"
SINCE = "since=2026-01-01T00:00:00Z"

pytestmark = pytest.mark.skipif(shutil.which("flock") is None or shutil.which("bash") is None, reason="gpu-lease needs bash and flock(1)")


class Node:
    """A machine as gpu-lease reads it. Its quota is 81.6 of 96 CPUs unless `quota` says otherwise (None: no cpu.max at all), and
    it has no cpu.stat until `throttled` writes one. Its `taskset` writes its first two arguments to `pinned` and runs the rest
    unpinned: the made-up CPUs aren't this machine's. A request with `--cores` calls it once to try its list, reading in the
    made-up /proc the CPUs a request may run on (`self/status`: all 96 until `allows` says otherwise), and once more to start
    CMD when it is leased."""

    def __init__(self, tmp_path: Path, quota: str | None = "8160000 100000"):
        self.gl, self.proc, self.bin = tmp_path / "gl", tmp_path / "proc", tmp_path / "bin"
        for d in (self.gl, self.proc, self.bin):
            d.mkdir()
        (self.proc / "stat").write_text("cpu  1 2 3\n" + "".join(f"cpu{c} 1 2 3\n" for c in range(96)) + "intr 0\n")
        (self.proc / "self").mkdir()
        self.allows("0-95")
        self.cpu_max, self.pinned = tmp_path / "cpu.max", tmp_path / "taskset.args"
        if quota is not None:
            self.cpu_max.write_text(quota + "\n")
        self.program("taskset", f'echo "$1 $2" >> "{self.pinned}"\nshift 2\nexec "$@"')
        env = {k: v for k, v in os.environ.items() if k != "RESEARCH_RUN_DIR"}
        self.path = env["PATH"]                                              # this machine's programs, its taskset among them
        self.env = {**env, "PATH": f"{self.bin}:{self.path}", "GPU_LEASE_DIR": str(self.gl), "GPU_LEASE_COUNT": "4",
                    "GPU_LEASE_CPU_MAX": str(self.cpu_max), "GPU_LEASE_PROC": str(self.proc), "GPU_LEASE_WHO": "asker",
                    "RESEARCH_RUN_ID": "r-new", "GPU_LEASE_USAGE_LOG": str(tmp_path / "usage.jsonl"),
                    "GPU_LEASE_UTIL_LOG": str(tmp_path / "util-*.jsonl"), "GPU_LEASE_POLL_S": "0"}

    def program(self, name: str, script: str) -> None:
        (self.bin / name).write_text(f"#!/bin/sh\n{script}\n")
        (self.bin / name).chmod(0o755)

    def allows(self, cpus: str) -> None:
        (self.proc / "self" / "status").write_text(f"Name:\tcat\nCpus_allowed:\tff\nCpus_allowed_list:\t{cpus}\n")

    def throttled(self, nr: int, usec: int) -> None:
        (self.cpu_max.parent / "cpu.stat").write_text(
            f"usage_usec 9\nuser_usec 5\nsystem_usec 4\nnr_periods 70\nnr_throttled {nr}\nthrottled_usec {usec}\nnr_bursts 0\n")

    def lease(self, gpu: int, line: str) -> int:
        """A live lease of `gpu` with this owner line: its lock is held until the descriptor returned is closed."""
        (self.gl / f"{gpu}.owner").write_text(line + "\n")
        fd = os.open(self.gl / f"{gpu}.lock", os.O_RDWR | os.O_CREAT, 0o664)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd

    def process(self, pid: int, parent: int, allowed: str, threads: dict[int, str] | None = None) -> None:
        d = self.proc / str(pid)
        d.mkdir()
        (d / "stat").write_text(f"{pid} (a job (1)) S {parent} {pid} {pid} 0 -1\n")
        (d / "status").write_text(f"Name:\ta job (1)\nCpus_allowed:\tff\nCpus_allowed_list:\t{allowed}\n")
        for tid, cpus in (threads or {}).items():
            (d / "task" / str(tid)).mkdir(parents=True)
            (d / "task" / str(tid) / "status").write_text(f"Cpus_allowed_list:\t{cpus}\n")

    def ask(self, *args: str, cmd: tuple[str, ...] = ("true",)) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", str(GPU_LEASE), "1", *args, "--mem-gb", "0", "--", *cmd], env=self.env, capture_output=True,
                              text=True)


def test_a_request_whose_cores_pass_the_quota_with_the_live_leases_is_refused_before_it_takes_a_gpu(tmp_path):
    node = Node(tmp_path)
    held = [node.lease(0, f"pid=11 run=r-a who=one uuid=- {SINCE} cores=0-39 cmd=serve --port 1"),
            node.lease(1, f"pid=12 run=r-b who=two uuid=- {SINCE} until=2026-01-01T01:00:00Z cores=40-71,80-83 timed=1 cmd=bench")]
    r = node.ask("--cores", "72-79,84-85")                                   # 40 + 36 + 10 = 86 CPUs against 81.6
    assert r.returncode == 75 and r.stdout == ""
    assert r.stderr.splitlines() == [
        f"gpu-lease: refused, holding nothing: 86 CPUs would be pinned on this machine, over its CPU quota of 81.6 (cpu.max 8160000 "
        f"100000 in {node.cpu_max})",
        "  this request: --cores 72-79,84-85 (10)",
        "  GPU 0, pid 11, run r-a, who one: cores=0-39 (40)",
        "  GPU 1, pid 12, run r-b, who two: cores=40-71,80-83 (36)",
        "  the count covers leases only: a process that holds no lease is not in it",
        "  start when a lease ends, ask for fewer CPUs, or have the holder of a lease that declared none pass --cores"]
    assert not (node.gl / "2.owner").exists()                                # no GPU taken
    assert node.pinned.read_text() == "-c 72-79,84-85\n"                     # its list tried, and CMD never started
    status = subprocess.run(["bash", str(GPU_LEASE), "status"], env=node.env, capture_output=True, text=True).stdout.splitlines()
    assert status[2:] == ["2 free", "3 free"]
    assert len(held) == 2


def test_a_request_that_fits_is_leased_pinned_and_recorded_so_the_next_request_counts_it(tmp_path):
    node = Node(tmp_path)
    held = [node.lease(0, f"pid=11 run=r-a who=one {SINCE} cores=0-39 cmd=serve"),
            node.lease(1, f"pid=12 run=r-b who=two {SINCE} cores=40-69 cmd=bench")]
    inside = 'cat "$GPU_LEASE_DIR/$CUDA_VISIBLE_DEVICES.owner"; bash "$0" 1 --cores 80-81 --mem-gb 0 -- true; echo "next=$?"'
    r = node.ask("--cores", "70-79", "--max-min", "5", cmd=("bash", "-c", inside, str(GPU_LEASE)))    # 40 + 30 + 10 = 80 CPUs
    assert r.returncode == 0
    owner, nxt = r.stdout.splitlines()
    assert owner.startswith("pid=") and " run=r-new who=asker " in owner and " cores=70-79 cmd=bash -c" in owner
    assert nxt == "next=75"                                                  # with this lease's 10, two more are 82 CPUs
    assert "82 CPUs would be pinned" in r.stderr and "  GPU 2, pid " in r.stderr and ", run r-new, who asker: cores=70-79 (10)" in r.stderr
    assert "CMD pinned to CPUs 70-79" in r.stderr
    assert node.pinned.read_text() == "-c 70-79\n-c 70-79\n-c 80-81\n"       # its list tried, CMD started, the next one's tried
    assert len(held) == 2


@pytest.mark.parametrize("quota", [None, "max 100000"])
def test_a_machine_without_a_quota_refuses_nothing(tmp_path, quota):
    node = Node(tmp_path, quota=quota)
    held = node.lease(0, f"pid=11 run=r-a who=one {SINCE} cores=0-95 cmd=serve")
    r = node.ask("--cores", "0-95")
    assert r.returncode == 0 and "would be pinned" not in r.stderr and node.pinned.read_text() == "-c 0-95\n" * 2
    r = node.ask()                                                           # and warns nobody: there is no count to be left out of
    assert r.returncode == 0 and "warning" not in r.stderr
    assert held


def test_a_lease_that_declared_none_counts_as_the_cpus_its_processes_may_run_on_and_the_refusal_names_it(tmp_path):
    node = Node(tmp_path)
    held = node.lease(0, f"pid=11 run=r-a who=one {SINCE} cmd=serve --flag cores=0-3")   # its command line declares nothing
    node.process(11, 1, "48-55")                                             # the lease itself
    node.process(21, 11, "56-71")                                            # its child, pinned elsewhere
    node.process(31, 21, "56-59", threads={31: "56-59", 32: "72-95"})        # a grandchild with one thread allowed wider
    node.process(41, 1, "0-95")                                              # not the lease's
    r = node.ask("--cores", "0-33")                                          # its 48-95 are 48 CPUs, and 34 more are 82
    assert r.returncode == 75 and "82 CPUs would be pinned" in r.stderr
    assert "  GPU 0, pid 11, run r-a, who one: none declared, its processes are allowed on 48-95 (48)" in r.stderr.splitlines()
    assert node.ask("--cores", "0-32").returncode == 0                       # 81 CPUs fit
    assert held


def test_a_shared_cpu_counts_once_an_ended_lease_counts_nothing_and_a_lease_with_no_process_left_counts_every_cpu(tmp_path):
    node = Node(tmp_path)
    held = [node.lease(gpu, f"pid=11 run=r-a who=one {SINCE} cores=0-39 cmd=serve") for gpu in (0, 1)]      # one lease of two GPUs
    held.append(node.lease(2, f"released=2026-01-01T00:30:00Z pid=12 run=r-b who=two {SINCE} cores=0-95 cmd=done"))
    r = node.ask("--cores", "30-81")                                         # 0-81 are 82 CPUs, though 40 and 52 are named
    assert r.returncode == 75 and "82 CPUs would be pinned" in r.stderr and "; a CPU two of them name counts once" in r.stderr
    assert "  GPU 0,1, pid 11, run r-a, who one: cores=0-39 (40)" in r.stderr.splitlines() and "pid 12" not in r.stderr
    assert node.ask("--cores", "30-80").returncode == 0                      # 0-80 are 81 CPUs
    held.append(node.lease(3, f"pid=13 run=r-c who=three {SINCE} cmd=left behind"))     # /proc has no pid 13
    r = node.ask("--cores", "0-3")
    assert r.returncode == 75 and "96 CPUs would be pinned" in r.stderr
    assert ("  GPU 3, pid 13, run r-c, who three: none declared and no process of it left to read, so every CPU, 0-95 (96)"
            in r.stderr.splitlines())


def test_a_pinned_process_that_holds_no_lease_is_not_counted_and_the_refusal_and_the_usage_say_so(tmp_path):
    node = Node(tmp_path)
    held = node.lease(0, f"pid=11 run=r-a who=one {SINCE} cores=0-39 cmd=serve")
    node.process(11, 1, "0-39")
    node.process(41, 1, "40-81")                                             # pinned, running, and no lease's
    assert node.ask("--cores", "40-80").returncode == 0                      # 81 CPUs by the leases, 82 with pid 41's
    r = node.ask("--cores", "40-81")
    assert r.returncode == 75 and "pid 41" not in r.stderr
    assert "  the count covers leases only: a process that holds no lease is not in it" in r.stderr.splitlines()
    usage = subprocess.run(["bash", str(GPU_LEASE), "--help"], env=node.env, capture_output=True, text=True)
    assert usage.returncode == 2 and "The count covers leases only" in usage.stderr
    assert "a job can still widen its own affinity" in usage.stderr
    assert "the leases of this lease directory: a machine with a second directory is not counted across them" in usage.stderr
    assert held


def test_a_request_without_cores_beside_a_declared_lease_starts_with_one_warning_naming_that_lease(tmp_path):
    node = Node(tmp_path)
    held = [node.lease(gpu, f"pid=11 run=r-a who=one {SINCE} cores=0-79 timed=1 cmd=bench") for gpu in (0, 1)]
    held.append(node.lease(2, f"released=2026-01-01T00:30:00Z pid=12 run=r-b who=two {SINCE} cores=0-95 cmd=done"))
    r = node.ask(cmd=("echo", "ran"))
    assert r.returncode == 0 and r.stdout == "ran\n" and not node.pinned.exists()
    warned = [ln for ln in r.stderr.splitlines() if "warning" in ln]
    assert warned == ["gpu-lease: warning: no --cores here, beside a lease that declared its own (GPU 0,1, pid 11, run r-a, who one: "
                      "cores=0-79): this run's CPUs are in no count against the machine's CPU quota until the next request that "
                      "declares counts them; pass --cores"]
    assert "warning" not in node.ask("--cores", "80").stderr                 # a request that declares is counted, not warned
    for fd in held[:2]:
        os.close(fd)
    held = node.lease(0, f"pid=13 run=r-c who=three {SINCE} cmd=serve cores=0-3")       # declared nothing: no one to warn beside
    r = node.ask()
    assert r.returncode == 0 and "warning" not in r.stderr
    assert held


def test_a_lease_writes_the_machines_throttle_counters_at_its_start_and_at_its_release(tmp_path):
    node = Node(tmp_path)
    node.throttled(7, 1500)
    owner = node.gl / "3.owner"
    inside = f'cat "{owner}"; printf "nr_throttled 9\\nthrottled_usec 4000\\n" > "{node.cpu_max.parent / "cpu.stat"}"'
    r = node.ask("--on", "3", "--cores", "0-9", cmd=("sh", "-c", inside))
    assert r.returncode == 0
    assert " nr_throttled=7 throttled_usec=1500 cores=0-9 cmd=sh -c" in r.stdout and "end_" not in r.stdout
    line = owner.read_text()
    assert line.startswith("released=") and " pid=" in line.split(" cmd=")[0]
    assert " nr_throttled=7 throttled_usec=1500 cores=0-9 end_nr_throttled=9 end_throttled_usec=4000 cmd=sh -c" in line
    assert ("gpu-lease: CPU throttling on this machine while the lease was held: nr_throttled +2, throttled_usec +2500 "
            f"({node.cpu_max.parent / 'cpu.stat'})") in r.stderr.splitlines()
    r = node.ask("--on", "2", cmd=("true",))                                 # a lease that declares nothing writes them too
    assert " nr_throttled=9 throttled_usec=4000 end_nr_throttled=9 end_throttled_usec=4000 cmd=true" in (node.gl / "2.owner").read_text()
    assert r.returncode == 0 and node.pinned.read_text() == "-c 0-9\n" * 2   # and is not pinned: both are the first request's
    assert "nr_throttled +0, throttled_usec +0" in r.stderr


def test_a_machine_without_throttle_counters_writes_none(tmp_path):
    node = Node(tmp_path)
    r = node.ask("--on", "3", "--cores", "0-9")
    line = (node.gl / "3.owner").read_text()
    assert r.returncode == 0 and "throttled" not in line and "CPU throttling" not in r.stderr and " cores=0-9 cmd=true" in line


def test_with_wait_a_request_over_the_quota_waits_unqueued_and_starts_when_a_lease_ends(tmp_path):
    node = Node(tmp_path)
    held = node.lease(0, f"pid=11 run=r-a who=one {SINCE} cores=0-79 cmd=serve")
    p = subprocess.Popen(["bash", str(GPU_LEASE), "1", "--wait", "--cores", "80-89", "--mem-gb", "0", "--", "echo", "ran"],
                         env=node.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    said = [p.stderr.readline().rstrip("\n") for _ in range(6)]
    assert said[0].startswith("gpu-lease: 90 CPUs would be pinned on this machine, over its CPU quota of 81.6 ")
    assert said[5] == "gpu-lease: waiting, unqueued, until they fit" and not list(node.gl.glob("wait.*"))
    os.close(held)                                                           # the lease ends
    out, err = p.communicate()
    assert p.returncode == 0 and out == "ran\n" and "would be pinned" not in err
    assert node.pinned.read_text() == "-c 80-89\n" * 2                       # tried once, however long it waited


def test_a_check_that_fails_leases_nothing_and_cores_must_be_a_cpu_list(tmp_path):
    node = Node(tmp_path)
    for bad in ("", "a", "1-", "1,,2", "0-3 4", "5-3"):
        assert node.ask("--cores", bad).returncode == 2
    node.program("python3", "exit 3")
    r = node.ask("--cores", "0-3")
    assert r.returncode == 2 and "the CPU quota check of --cores 0-3 failed (exit 3), so nothing is leased" in r.stderr
    assert node.pinned.read_text() == "-c 0-3\n" and not list(node.gl.glob("*.owner"))


def test_cores_the_machine_cannot_pin_to_are_refused_before_any_gpu_is_taken(tmp_path):
    node = Node(tmp_path)
    node.program("taskset", "echo \"taskset: failed to set pid 7's affinity: Invalid argument\" >&2\nexit 1")
    r = node.ask("--cores", "200-201", cmd=("echo", "ran"))
    assert r.returncode == 2 and r.stdout == ""
    assert r.stderr.splitlines() == ["gpu-lease: --cores 200-201 can't be pinned on this machine, so nothing is leased: taskset: "
                                     "failed to set pid 7's affinity: Invalid argument"]
    assert not list(node.gl.glob("*.owner"))                                 # no GPU was taken for a CMD that could not start


def test_a_list_with_a_cpu_the_request_may_not_run_on_is_refused_naming_those_cpus_before_any_gpu_is_taken(tmp_path):
    node = Node(tmp_path, quota=None)                                        # quota or none
    node.allows("0-47,64-79")
    r = node.ask("--cores", "40-49,60,62-66", cmd=("echo", "ran"))           # taskset would pin CMD to 40-47 and 64-66 alone
    assert r.returncode == 2 and r.stdout == ""
    assert r.stderr.splitlines() == ["gpu-lease: --cores 40-49,60,62-66 names CPUs this request can't run on here (48-49,60,62-63), "
                                     "so nothing is leased: every CPU of the list must be one of this machine's that it may use"]
    assert not list(node.gl.glob("*.owner")) and node.pinned.read_text() == "-c 40-49,60,62-66\n"
    r = node.ask("--cores", "40-47,64-66", cmd=("echo", "ran"))              # every CPU of it inside
    assert r.returncode == 0 and r.stdout == "ran\n"
    (node.proc / "self" / "status").write_text("Name:\tcat\n")               # a /proc that doesn't say what the trial was pinned to
    r = node.ask("--cores", "0-3", cmd=("echo", "ran"))
    assert r.returncode == 2 and r.stdout == "" and "doesn't say which CPUs taskset pinned to" in r.stderr


@pytest.mark.skipif(shutil.which("taskset") is None, reason="needs taskset(1)")
def test_cmd_runs_on_the_cores_it_declared_and_a_cpu_no_machine_has_is_refused(tmp_path):
    node = Node(tmp_path, quota=None)
    cpu = min(os.sched_getaffinity(0))
    env = {**node.env, "PATH": node.path, "GPU_LEASE_PROC": "/proc"}         # this machine's taskset, and what it pinned to
    r = subprocess.run(["bash", str(GPU_LEASE), "1", "--cores", str(cpu), "--mem-gb", "0", "--", sys.executable, "-c",
                        "import os; print(sorted(os.sched_getaffinity(0)))"], env=env, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout == f"[{cpu}]\n"
    owners = {p.name: p.read_text() for p in node.gl.glob("*.owner")}
    r = subprocess.run(["bash", str(GPU_LEASE), "1", "--cores", "99999", "--mem-gb", "0", "--", "echo", "ran"], env=env,
                       capture_output=True, text=True)
    assert r.returncode == 2 and r.stdout == "" and {p.name: p.read_text() for p in node.gl.glob("*.owner")} == owners
    assert "gpu-lease: --cores 99999 can't be pinned on this machine, so nothing is leased: " in r.stderr
