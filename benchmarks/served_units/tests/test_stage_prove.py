"""stage_prove.sh's start check on a GPU node and its split into two jobs, run against stubs of nvidia-smi, gpu-lease, lake,
findmnt and window.py. It starts nothing (exit 9, before any build) when a card of CARDS is leased, when one holds a compute
process, or when KILL_AT is too near; it reads the leases among CARDS alone (GPU_LEASE_ALLOWED_FILE, which every lease of
the job is then taken among), so another card's process or lease holds nothing back; once every card is free it goes on to
its first build. STEPS=stage reads no lease, hides every card and keeps FLOCK_WORK with .staged naming its commit;
STEPS=prove builds nothing, refuses a FLOCK_WORK another commit staged before reading a lease, and proves one this commit
staged on its cards alone."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from conftest import HERE

SP = HERE / "stage_prove.sh"
FAR = "4102444800"  # 2100-01-01T00:00Z
TREE = "c0ffee"
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
#: lake: its call and the cards it sees logged; fails, unless $LAKE_OK
LAKE = '#!/usr/bin/env bash\necho "lake $* cvd=[${CUDA_VISIBLE_DEVICES-unset}]" >> "$CALLS"\n[ -n "${LAKE_OK:-}" ]\n'
#: findmnt -no OPTIONS|TARGET -T PATH: a mount that allows exec
FINDMNT = '#!/usr/bin/env bash\ncase "$*" in *OPTIONS*) echo rw,relatime ;; *) echo / ;; esac\n'
#: window.py stage|prove UNITS: its call, groups, allowed cards and visible cards logged; `stage` stages one statement
WINDOW = r'''import os, sys
from pathlib import Path
a = os.environ.get("GPU_LEASE_ALLOWED_FILE")
with open(os.environ["CALLS"], "a") as f:
    f.write(f"window {sys.argv[1]} gpus={os.environ.get('WIN_GPUS', '')} allowed={Path(a).read_text().strip() if a else ''} "
            f"cvd=[{os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}]\n")
if sys.argv[1] == "stage":
    d = Path(os.environ["FLOCK_WORK"]) / "data" / "s"
    d.mkdir(parents=True, exist_ok=True)
    (d / ".done").touch()
'''


@pytest.fixture
def job(tmp_path, monkeypatch):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    monkeypatch.setenv("PATH", str(bin_), prepend=os.pathsep)
    monkeypatch.delenv("GPU_LEASE_ALLOWED_FILE", raising=False)
    for name, text in (("nvidia-smi", NVIDIA_SMI), ("gpu-lease", GPU_LEASE), ("lake", LAKE), ("findmnt", FINDMNT)):
        (bin_ / name).write_text(text)
        (bin_ / name).chmod(0o755)
    (tmp_path / "units").write_text("census:EmbeddingRead_v1{V=256,H=4}@full\t4.8\t1.5\t197\t30\n")
    (tmp_path / "verity/protocols/proofs/flock/verifier").mkdir(parents=True)
    (tmp_path / "benchmarks/served_units").mkdir(parents=True)
    (tmp_path / "benchmarks/served_units/window.py").write_text(WINDOW)
    run_dir, calls = tmp_path / "run", tmp_path / "calls"

    def run(**extra: str) -> tuple[int, str, list[str]]:
        env = {**os.environ, "HOME": str(tmp_path), "RESEARCH_RUN_DIR": str(run_dir),
               "CALLS": str(calls), "UNITS": str(tmp_path / "units"), "DEV": "gpu", "CARDS": "4,5", "SM": "90",
               "KILL_AT": FAR, "RESEARCH_SOURCE_SHA": TREE, **extra}
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
    rc, steps, _ = job(LAKE_OK="1", STEPS="stage", FLOCK_WORK="/dev/shm/stage-prove-test")
    assert rc == 4 and "REFUSED: FLOCK_WORK /dev/shm/stage-prove-test is in RAM" in steps, steps


def test_refuses_a_device_or_steps_it_does_not_know(job):
    rc, steps, calls = job(DEV="tpu")
    assert rc == 2 and "DEV is gpu or cpu, not tpu" in steps and calls == []
    rc, steps, calls = job(STEPS="build")
    assert rc == 2 and 'STEPS is "stage prove", stage or prove, not build' in steps and calls == []


def test_the_cards_and_their_lease_status_are_recorded(job, tmp_path: Path):
    job(HELD="6")
    out = tmp_path / "run/out"
    assert (out / "lease-at-start.txt").read_text().split("\n")[:2] == ["4 free", "5 free"]
    assert (out / "cards-at-start.txt").exists()


def test_stage_alone_reads_no_lease_hides_every_card_and_keeps_its_flock_work(job, tmp_path: Path):
    """On a card that is leased and busy, STEPS=stage still builds and stages, every card hidden, and keeps FLOCK_WORK with
    .staged naming the commit, for a STEPS=prove of the same commit."""
    fw = tmp_path / "fw"
    rc, steps, calls = job(STEPS="stage", LAKE_OK="1", FLOCK_WORK=str(fw), HELD="4,5", APPS="GPU-4, 77, python\n")
    assert rc == 0 and "no card: STEPS=stage" in steps and "start check" not in steps, steps
    assert calls == ["lake build flock-read-tmpl cvd=[]", "window stage gpus= allowed= cvd=[]"]
    assert (fw / ".staged").read_text().split()[0] == TREE and "staged 1" in (fw / ".staged").read_text()
    assert f"STEPS=stage: FLOCK_WORK {fw} kept" in steps


def test_stage_alone_marks_nothing_staged_when_a_statement_is_missing(job, tmp_path: Path):
    (tmp_path / "units").write_text("census:A@full\t1\t1\t1\t1\ncensus:B@full\t1\t1\t1\t1\n")
    fw = tmp_path / "fw"
    rc, steps, _ = job(STEPS="stage", LAKE_OK="1", FLOCK_WORK=str(fw))
    assert not (fw / ".staged").exists() and "none, so no STEPS=prove takes it" in steps, steps


def test_prove_alone_refuses_a_flock_work_another_commit_staged(job, tmp_path: Path):
    fw = tmp_path / "fw"
    fw.mkdir()
    (fw / ".staged").write_text("deadbeef 2026-11-13T17:00:00Z units-sha256 0 staged 1\n")
    rc, steps, calls = job(STEPS="prove", FLOCK_WORK=str(fw))
    assert rc == 9 and f"REFUSED: FLOCK_WORK {fw} was not staged by this commit" in steps, steps
    assert calls == [] and fw.exists()
    rc, steps, calls = job(STEPS="prove", FLOCK_WORK=str(tmp_path / "none"))
    assert rc == 9 and "its .staged reads 'nothing'" in steps and calls == []


def test_prove_alone_proves_this_commits_staging_on_its_cards_alone(job, tmp_path: Path):
    """No build; the start check on cards 4 and 5; window.py prove with one group a card and the leases among 4 and 5;
    FLOCK_WORK removed at the end."""
    fw = tmp_path / "fw"
    fw.mkdir()
    (fw / ".staged").write_text(f"{TREE} 2026-11-13T17:00:00Z units-sha256 0 staged 1\n")
    rc, steps, calls = job(STEPS="prove", FLOCK_WORK=str(fw), HELD="6")
    assert rc == 0 and "staged by this commit" in steps and "start check passed: cards 4,5 free" in steps, steps
    assert calls == ["4,5", "window prove gpus=2 allowed=4,5 cvd=[unset]"]
    assert "flock-read-tmpl" not in steps and not fw.exists()
