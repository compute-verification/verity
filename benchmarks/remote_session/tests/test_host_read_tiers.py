"""`host_read_tiers.sh` with no card: its dry run prints the lease's line and each size's command and touches nothing;
under the tree's own gpu-lease, faking eight cards, the lease's line is taken as written and each size runs in turn in
that lease; a lease that is short drops the last sizes, and one asked for near END is asked shorter by the wait it
allows; a failing size is the run's exit and the next still runs; a card whose clocks the driver held down during a
size is named on that size's line and over the lease, and each card's clock and temperature are kept between the
sizes unless SAMPLE_MS is 0; cards
in use, a lease of fewer cards, a failed build and a start outside the hours start nothing; and `launch` sends nothing
outside the hours. The clock is NOW; host_read.sh, the Python, nvidia-smi and nvcc are stubs."""
from __future__ import annotations

import fcntl
import os
import shutil
import subprocess
from pathlib import Path

import pytest

RS = Path(__file__).resolve().parents[1]
GPU_LEASE = RS.parents[1] / "tools/research/src/research/pods/sh/gpu_lease.sh"
NOW = 2_000_000_000
EIGHT = "0,1,2,3,4,5,6,7"
#: what the job reads that the tests leave at its default, whatever the suite's own environment holds
KNOBS = {"OUT", "DRY", "SIZES", "GIB", "GPUS", "CHUNK_BLOCKS", "LEASE_MIN", "GRANT_S", "MEM_GB", "CLEAR_MIB", "SIZE_MIN_S",
         "TAIL_S", "SAMPLE_MS", "SAMPLE_CPUS",
         "BUILD_S", "ARCH", "ARGS", "BASE_REV", "EXPECT_COMMIT", "SLOTS", "RESEARCH_SOURCE_SHA", "GPU_LEASE_TIMED",
         "CUDA_VISIBLE_DEVICES", "GPU_LEASE_UUID", "PYTORCH_CUDA_ALLOC_CONF"}

BUILD = """echo "$BUILD_ONLY $ARCH $CUDA_HOME ${BASE_REV:-}" >> "$STUBS/build.calls"
: > "$BUILD_DIR/session.so"; : > "$BUILD_DIR/kv.so"
sha256sum "$BUILD_DIR"/*.so > "$OUT/libs.sha256"
exit "${BUILD_RC:-0}"
"""
PYTHON = """#!/bin/bash
echo "$CUDA_VISIBLE_DEVICES | $*" >> "$STUBS/driver.calls"
echo "${PYTORCH_CUDA_ALLOC_CONF:-}" > "$STUBS/driver.alloc"
while [ $# -gt 0 ]; do [ "$1" != --host-gib ] || size=$2; shift; done
echo "    1.0s store: a host tier of $size GiB"
echo "    2.0s chain: pass 1.00 ms"
[ "$size" != "${FAIL_SIZE:-}" ] || exit 6
"""
SMI = """#!/bin/bash
case "$*" in
  *--query-compute-apps*) [ -z "${BUSY:-}" ] || echo "GPU-0, 77, 5000 MiB, another" ;;
  *clocks_event_reasons*) echo "Field is not a valid field to query."; exit 1 ;;
  *clocks_throttle_reasons*) for i in 0 1 2 3 4 5 6 7; do echo "$i, GPU-$i, card, 3, 40, 80, 2000, 0x000000000000000$((i == 5 ? 4 : 0))"; done ;;
  *) for i in 0 1 2 3 4 5 6 7; do echo "$i, 3"; done ;;
esac
"""
#: a driver that counts how long it has held each card's clocks down: card 1's thermal count grows by 153.4 s, and
#: card 6's power-cap count by 2 s, from each reading of the cards to the next. The sampler's call is not a reading:
#: it says what it was called with.
COUNTING = """#!/bin/bash
case "$*" in
  *--query-compute-apps*) ;;
  *-lms*) echo "$*" > "$STUBS/sampling" ;;
  *clocks_event_reasons_counters*)
    n=$(cat "$STUBS/smi.n" 2> /dev/null || echo 0); echo $((n + 1)) > "$STUBS/smi.n"
    for i in 0 1 2 3 4 5 6 7; do
      echo "$i, GPU-$i, card, 3, 40, 80, 2000, $((i == 6 ? 7000000 + n * 2000000 : 5)), $((i == 1 ? n * 153400000 : 0)), 0, 0, 0x0000000000000000"
    done ;;
  *) for i in 0 1 2 3 4 5 6 7; do echo "$i, 3"; done ;;
esac
"""
GRANT = """#!/bin/bash
[ "$1" != status ] || exit 0
echo "$*" >> "$STUBS/lease.calls"
while [ "$1" != -- ]; do shift; done
shift
export CUDA_VISIBLE_DEVICES=$GRANTED GPU_LEASE_UUID=$(sed 's/[0-9]/GPU-&/g' <<< "$GRANTED")
exec "$@"
"""
TREES = f"""#!/bin/bash
[ "$1" = status ] || echo "$*" >> "$STUBS/lease.calls"
exec bash {GPU_LEASE} "$@"
"""


def start(tmp_path: Path, stage: str = "job", lease: str = GRANT, left: int = 4500, python: str = PYTHON,
          smi: str = SMI, **env: str) -> subprocess.Popen:
    """`host_read_tiers.sh STAGE` started in a tree of stubs, `left` seconds before END."""
    tree, stubs = tmp_path / "tree", tmp_path / "stubs"
    (tree / "benchmarks/remote_session").mkdir(parents=True)
    (stubs / "cuda/bin").mkdir(parents=True)
    shutil.copy(RS / "host_read_tiers.sh", tree / "benchmarks/remote_session")
    (tree / "benchmarks/remote_session/host_read.sh").write_text(BUILD)
    for name, text in {"python": python, "nvidia-smi": smi, "gpu-lease": lease, "cuda/bin/nvcc": "#!/bin/bash\n"}.items():
        (stubs / name).write_text(text)
        (stubs / name).chmod(0o755)
    env = {**{k: v for k, v in os.environ.items() if k not in KNOBS},
           "RESEARCH_RUN_DIR": str(tmp_path / "run"), "RESEARCH_RUN_ID": "r-tiers", "STUBS": str(stubs), "NOW": str(NOW),
           "END": f"@{NOW + left}", "NOT_BEFORE": f"@{NOW}", "NAMES": "ahead2 feed8", "GRANTED": EIGHT,
           "PY": str(stubs / "python"), "SMI": str(stubs / "nvidia-smi"), "GPU_LEASE": str(stubs / "gpu-lease"),
           "CUDA_HOME": str(stubs / "cuda"), "DRIVER_CPUS": str(min(os.sched_getaffinity(0))), "CLEAR_WAIT_S": "0",
           "GPU_LEASE_DIR": str(tmp_path / "gl"), "GPU_LEASE_COUNT": "8", "GPU_LEASE_WHO": "tiers-test",
           "GPU_LEASE_UUID_TABLE": ",".join(f"GPU-{i}" for i in range(8)), "GPU_LEASE_WINDOWS": str(tmp_path / "no-windows"),
           "GPU_LEASE_ALLOWED_FILE": str(tmp_path / "no-direct-gpus"), "GPU_LEASE_USAGE_LOG": str(tmp_path / "usage.jsonl"),
           "GPU_LEASE_UTIL_LOG": str(tmp_path / "util-*.jsonl"), **env}
    return subprocess.Popen(["bash", "benchmarks/remote_session/host_read_tiers.sh", stage], cwd=tree, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def run(tmp_path: Path, *args, **kwargs) -> tuple:
    """`start`, to its end: the process with its output, each stub's calls, out/."""
    job = start(tmp_path, *args, **kwargs)
    p = subprocess.CompletedProcess(job.args, job.returncode, *job.communicate())
    p.returncode = job.returncode
    stubs = tmp_path / "stubs"
    calls = {k: (stubs / f"{k}.calls").read_text().splitlines() if (stubs / f"{k}.calls").exists() else []
             for k in ("build", "lease", "driver")}
    return p, calls, tmp_path / "run/out"


def sizes(calls: dict) -> list[str]:
    """The tier's size in each of the driver's calls, in order."""
    return [c.split("--host-gib ")[1].split()[0] for c in calls["driver"]]


def test_a_dry_run_prints_the_lease_and_each_size_and_touches_nothing(tmp_path):
    p, calls, out = run(tmp_path, DRY="1")
    assert p.returncode == 0, p.stdout + p.stderr
    tiers = tmp_path / "tree/benchmarks/remote_session/host_read_tiers.sh"
    assert f"gpu-lease 8 --wait --timed --max-min 30 --mem-gb 300 -- bash {tiers} lease" in p.stdout
    asked = [line for line in p.stdout.splitlines() if "--host-gib" in line]
    assert [line.split("--host-gib ")[1].split()[0] for line in asked] == ["8", "12", "16"]
    assert all("timeout -k 20 580 taskset -c" in line and line.endswith(
        "--gib 700 --host-gib " + size + " --gpus 8 --chunk-blocks 65536 --budget-s 490 --names ahead2 feed8")
        for line, size in zip(asked, ["8", "12", "16"]))
    assert calls == {"build": [], "lease": [], "driver": []} and not out.exists()


@pytest.mark.skipif(shutil.which("flock") is None, reason="gpu-lease needs flock(1)")
def test_under_the_trees_gpu_lease_each_size_runs_in_turn_on_the_eight_cards(tmp_path):
    p, calls, out = run(tmp_path, lease=TREES, BASE_REV="abc")
    assert p.returncode == 0, p.stdout + p.stderr
    tiers = tmp_path / "tree/benchmarks/remote_session/host_read_tiers.sh"
    assert calls["lease"] == [f"8 --wait --timed --max-min 30 --mem-gb 300 -- bash {tiers} lease"]
    assert calls["build"] == [f"1 120a {tmp_path / 'stubs/cuda'} abc"]
    assert sizes(calls) == ["8", "12", "16"] and all(c.startswith(EIGHT + " | ") for c in calls["driver"])
    assert all(f"--out {out}/host-{size} " in c for c, size in zip(calls["driver"], ["8", "12", "16"]))
    # the driver allocates as it does under host_read.sh, which gates the rows it times
    assert (tmp_path / "stubs/driver.alloc").read_text().strip() == "expandable_segments:True"
    assert 'PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}' in (RS / "host_read.sh").read_text()
    account = (out / "tiers.txt").read_text().splitlines()
    assert account[-1] == p.stdout.splitlines()[-1] and "VERDICT host-read-tiers ok (exit 0): sizes run: 8 12 16 GiB" in account[-1]
    assert sum("chain: pass 1.00 ms" in line for line in account) == 3
    # the cards' record falls back to the field name this nvidia-smi knows, and names the card a throttle reason is on
    assert "clocks_throttle_reasons.active" in (out / "cards.start.txt").read_text()
    assert any("8 GiB: exit 0" in line and line.endswith("cards: 5") for line in account)
    assert (out / "leased").exists() and (out / "cards.end.txt").exists() and not (out / "cards.not-clear.txt").exists()
    # this nvidia-smi counts no held-down clocks: the account says so once, and no size's line names a card for it
    assert sum("does not count how long it holds a card's clocks down" in line for line in account) == 1
    assert not any("clocks held down" in line for line in account)


def test_a_card_the_driver_held_down_is_named_on_its_size_and_over_the_lease(tmp_path):
    """The sampler is a stub that writes its arguments to a FIFO this test holds open, so reading them waits for the
    sampler to have started, whenever that is."""
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    os.mkfifo(stubs / "sampling")
    sampling = os.open(stubs / "sampling", os.O_RDWR)
    try:
        p = start(tmp_path, smi=COUNTING)
        sampler = os.read(sampling, 4096).decode().split()
        stdout, stderr = p.communicate()
    finally:
        os.close(sampling)
    assert p.returncode == 0, stdout + stderr
    assert sampler[:6] == ["-i", ",".join(f"GPU-{i}" for i in range(8)), "-lms", "2000", "--format=csv,nounits",
                           "--query-gpu=timestamp,index,temperature.gpu,clocks.sm,power.draw,utilization.gpu,"
                           "clocks_event_reasons.active"] and len(sampler) == 6
    out = tmp_path / "run/out"
    account = (out / "tiers.txt").read_text().splitlines()
    during = "; clocks held down during it: card 1 sw_thermal_slowdown 153.4 s; card 6 sw_power_cap 2.0 s"
    assert [line.endswith(during) for line in account if " GiB: exit 0 after " in line] == [True] * 3
    over = "clocks held down over the lease: card 1 sw_thermal_slowdown 613.6 s; card 6 sw_power_cap 8.0 s"
    assert account[-2].endswith(over) and "VERDICT host-read-tiers ok (exit 0)" in account[-1]
    assert "clocks_event_reasons_counters.sw_thermal_slowdown" in (out / "cards.start.txt").read_text().splitlines()[0]
    assert not any("does not count" in line for line in account)
    assert any("every 2000 ms, in out/cards.during.csv: one nvidia-smi on" in line for line in account)


def test_no_sampler_runs_with_sample_ms_0_and_a_driver_whose_counts_do_not_move_names_no_card(tmp_path):
    still = COUNTING.replace("n * 153400000", "9").replace("n * 2000000", "0")
    p, calls, out = run(tmp_path, smi=still, SAMPLE_MS="0")
    assert p.returncode == 0 and sizes(calls) == ["8", "12", "16"], p.stdout + p.stderr
    account = (out / "tiers.txt").read_text().splitlines()
    assert account[-2].endswith("clocks held down over the lease: on no card")
    assert not any("clocks held down during it" in line or "cards.during.csv" in line for line in account)
    assert not (out / "cards.during.csv").exists()


def test_a_short_lease_drops_the_last_sizes(tmp_path):
    p, calls, out = run(tmp_path, left=700, GRANT_S="0")
    assert p.returncode == 10, p.stdout + p.stderr
    assert calls["lease"][0].startswith("8 --wait --timed --max-min 11 ") and sizes(calls) == ["8", "12"]
    assert "partial (exit 10): sizes run: 8 12 GiB in chunks of 65536 blocks; not run: 16 GiB" in p.stdout.splitlines()[-1]


@pytest.mark.parametrize("left, minutes, wait, ran", [
    (4500, 30, 2700, ["8", "12", "16"]),   # far from END: LEASE_MIN
    (2000, 28, 320, ["8", "12", "16"]),    # GRANT_S short of END, in whole minutes
    (1000, 11, 340, ["8", "12"]),
    (500, 5, 200, ["8"]),                  # one size needs five minutes: the wait is what they leave
    (300, 5, 0, ["8"]),
])
def test_a_lease_asked_for_near_end_is_shorter_by_the_wait_it_allows(tmp_path, left, minutes, wait, ran):
    p, calls, out = run(tmp_path, left=left, BUILD_S="0")
    assert p.returncode == (0 if len(ran) == 3 else 10), p.stdout + p.stderr
    assert calls["lease"][0].startswith(f"8 --wait --timed --max-min {minutes} ") and sizes(calls) == ran
    assert f"for at most {minutes} min (granted within {wait} s, it asks for nothing past END)" in p.stdout
    assert left - 60 * minutes == wait


def test_no_lease_is_asked_for_with_less_than_one_size_before_end(tmp_path):
    p, calls, out = run(tmp_path, left=299, BUILD_S="0")
    assert p.returncode == 9 and "before the lease: 4 min to END" in p.stdout.splitlines()[-1], p.stdout + p.stderr
    assert len(calls["build"]) == 1 and calls["lease"] == [] and calls["driver"] == []


def test_a_failing_size_is_the_exit_and_the_next_still_runs(tmp_path):
    p, calls, out = run(tmp_path, FAIL_SIZE="12")
    assert p.returncode == 6 and sizes(calls) == ["8", "12", "16"], p.stdout + p.stderr
    assert "failed (exit 6)" in p.stdout.splitlines()[-1]


@pytest.mark.parametrize("env, code, leases, said", [
    ({"BUSY": "1"}, 11, 1, "sizes run: none GiB"),
    ({"GRANTED": "0,1"}, 7, 1, "the lease holds cards '0,1', not 8 distinct ones; nothing started"),
    ({"BUILD_RC": "5"}, 5, 0, "host_read.sh's build or its gate on the kernels"),
    ({"NOW": str(NOW - 1)}, 9, 0, "before the build: before NOT_BEFORE"),
    ({"NOW": str(NOW + 4400)}, 9, 0, "before the build: 1 min to END"),
    ({"EXPECT_COMMIT": "another"}, 8, 0, "not EXPECT_COMMIT (another); nothing started"),
])
def test_what_starts_nothing(tmp_path, env, code, leases, said):
    p, calls, out = run(tmp_path, **env)
    assert p.returncode == code and said in p.stdout.splitlines()[-1], p.stdout + p.stderr
    assert len(calls["lease"]) == leases and calls["driver"] == []
    assert (out / "verdict.txt").read_text().strip() in p.stdout.splitlines()[-1]


@pytest.mark.skipif(shutil.which("flock") is None, reason="gpu-lease needs flock(1)")
def test_a_signal_to_the_job_ends_the_size_under_way_and_its_lease(tmp_path):
    """The driver is a stub that says it started and then waits for a line: only the signal, passed from the job
    through gpu-lease and the lease's stage, ends it before the job ends. The lines written once the job has ended
    let a driver the signal did not reach go on, and the test then fails on what it reads."""
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    os.mkfifo(stubs / "started")
    os.mkfifo(stubs / "release")
    waiting = PYTHON.replace('echo "    1.0s store', 'echo $$ > "$STUBS/started"; read -r _ < "$STUBS/release"; echo "    1.0s store')
    release = os.open(stubs / "release", os.O_RDWR)
    try:
        p = start(tmp_path, lease=TREES, python=waiting)
        driver = int((stubs / "started").read_text())
        p.terminate()
        p.wait()
        os.write(release, b"\n\n\n")
        out, err = p.communicate()
    finally:
        os.close(release)
    assert p.returncode == 143, out + err
    assert len((stubs / "driver.calls").read_text().splitlines()) == 1
    assert "failed (exit 143): a signal ended the lease during 8 GiB" in out.splitlines()[-1]
    with pytest.raises(ProcessLookupError):
        os.kill(driver, 0)
    for i in range(8):
        with open(tmp_path / f"gl/{i}.lock") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)   # every card's lock is free: the lease is gone


@pytest.mark.parametrize("now, said", [(NOW - 301, "more than 5 min before NOT_BEFORE"), (NOW + 4000, "8 min from")])
def test_launch_sends_nothing_outside_the_hours(tmp_path, now, said):
    p, calls, out = run(tmp_path, stage="launch", NOW=str(now), MACHINE="a-node", BY="a-lane")
    assert p.returncode == 2 and p.stderr.startswith("REFUSED: ") and said in p.stderr, p.stdout + p.stderr
    assert p.stdout == "" and calls == {"build": [], "lease": [], "driver": []} and not out.exists()
