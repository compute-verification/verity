"""stage_prove.sh's start check on a GPU node, run against stubs of nvidia-smi, gpu-lease and lake: it starts nothing (exit
9, before any build) when a card of CARDS is leased, when one holds a compute process, or when KILL_AT is too near; it reads
the leases among CARDS alone (GPU_LEASE_ALLOWED_FILE, which every lease of the job is then taken among), so another card's
process or lease holds nothing back; and once every card is free it goes on to its first build."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from conftest import HERE

SP = HERE / "stage_prove.sh"
FAR = "4102444800"  # 2100-01-01T00:00Z
#: nvidia-smi: card i's UUID is GPU-i; the compute processes are $APPS
NVIDIA_SMI = r'''#!/usr/bin/env bash
case "$*" in
  *--query-compute-apps*) printf '%b' "${APPS:-}" ;;
  "-i "*) echo "GPU-$2" ;;
  *) exit 2 ;;
esac
'''
#: gpu-lease status: each card of the allowed file free unless $HELD names it; the file's list logged
GPU_LEASE = r'''#!/usr/bin/env bash
[ "$1" = status ] || exit 2
cat "$GPU_LEASE_ALLOWED_FILE" >> "$CALLS"
for i in $(tr ',' ' ' < "$GPU_LEASE_ALLOWED_FILE"); do
  case ",${HELD:-}," in *",$i,"*) echo "$i held 3m pid=1 who=other" ;; *) echo "$i free" ;; esac
done
'''
#: lake: fails, unless $LAKE_OK
LAKE = '#!/usr/bin/env bash\necho "lake $*" >> "$CALLS"\n[ -n "${LAKE_OK:-}" ]\n'


@pytest.fixture
def job(tmp_path):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    for name, text in (("nvidia-smi", NVIDIA_SMI), ("gpu-lease", GPU_LEASE), ("lake", LAKE)):
        (bin_ / name).write_text(text)
        (bin_ / name).chmod(0o755)
    (tmp_path / "units").write_text("census:EmbeddingRead_v1{V=256,H=4}@full\t4.8\t1.5\t197\t30\n")
    (tmp_path / "verity/protocols/flock/verifier").mkdir(parents=True)
    run_dir, calls = tmp_path / "run", tmp_path / "calls"

    def run(**extra: str) -> tuple[int, str, list[str]]:
        env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "HOME": str(tmp_path), "RESEARCH_RUN_DIR": str(run_dir),
               "CALLS": str(calls), "UNITS": str(tmp_path / "units"), "DEV": "gpu", "CARDS": "4,5", "SM": "90",
               "KILL_AT": FAR, **extra}
        p = subprocess.run(["bash", str(SP)], cwd=tmp_path, capture_output=True, text=True, env=env)
        steps = (run_dir / "out/steps.txt").read_text() if (run_dir / "out/steps.txt").exists() else ""
        return p.returncode, steps, calls.read_text().splitlines() if calls.exists() else []

    return run


def test_starts_nothing_on_a_leased_card(job):
    rc, steps, calls = job(HELD="5")
    assert rc == 9 and "REFUSED: card 5's lease is not free: 5 held" in steps, steps
    assert "flock-read-tmpl" not in steps and not any(c.startswith("lake") for c in calls)


def test_starts_nothing_on_a_card_running_a_compute_process(job):
    rc, steps, _ = job(APPS="GPU-4, 77, python\n")
    assert rc == 9 and "REFUSED: card 4 holds a compute process: GPU-4, 77, python" in steps, steps
    assert "flock-read-tmpl" not in steps


def test_starts_nothing_this_near_kill_at(job):
    rc, steps, calls = job(KILL_AT="0")
    assert rc == 9 and "to KILL_AT, under MIN_LEFT_S=1800: nothing started" in steps, steps
    assert calls == []


def test_reads_the_leases_among_its_cards_alone_and_goes_on_when_they_are_free(job):
    """A process and a lease on card 6 hold nothing back; the leases read are the allowed file's, cards 4 and 5; the job
    then builds (here, a lake that fails: exit 4 at its first build)."""
    rc, steps, calls = job(APPS="GPU-6, 77, python\n", HELD="6")
    assert "start check passed: cards 4,5 free" in steps, steps
    assert calls[0] == "4,5" and rc == 4 and "flock-read-tmpl BUILD FAILED" in steps


def test_stages_nowhere_in_ram(job):
    """FLOCK_WORK holds flock-tmpl-unit, which a noexec /dev/shm can't run: a WORK_ROOT there is refused before staging."""
    rc, steps, _ = job(LAKE_OK="1", WORK_ROOT="/dev/shm/stage-prove-test")
    assert rc == 4 and "REFUSED: WORK_ROOT /dev/shm/stage-prove-test is in RAM" in steps, steps
    assert "window.py stage" not in steps and not Path("/dev/shm/stage-prove-test").exists()


def test_refuses_a_device_it_does_not_know(job):
    rc, steps, calls = job(DEV="tpu")
    assert rc == 2 and "DEV is gpu or cpu, not tpu" in steps and calls == []


def test_the_cards_and_their_lease_status_are_recorded(job, tmp_path: Path):
    job(HELD="6")
    out = tmp_path / "run/out"
    assert (out / "lease-at-start.txt").read_text().split("\n")[:2] == ["4 free", "5 free"]
    assert (out / "cards-at-start.txt").exists()
