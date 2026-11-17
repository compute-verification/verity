"""job.sh refuses, before it builds anything, a machine that lacks a tool or the cores and memory it runs on."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
NEEDS = {"cargo": "no cargo", "lean": "no lake, nor curl", "cores": "cores, and the job runs on 32",
         "memory": "GiB of memory, and the job asks"}
GNU_TIME = os.access("/usr/bin/time", os.X_OK) and subprocess.run(["/usr/bin/time", "-v", "true"],
                                                                     capture_output=True).returncode == 0


def _stand_in(bin_: Path, name: str, body: str) -> None:
    (bin_ / name).write_text(f"#!/bin/sh\n{body}\n")
    (bin_ / name).chmod(0o755)


def _job(tmp_path: Path, lacks: set[str], **env: str) -> tuple[subprocess.CompletedProcess, list[str]]:
    """job.sh on a PATH of stand-ins only, every need met but those in `lacks` (one must be, or the job builds), and the
    lines it refused with."""
    assert lacks
    bin_ = tmp_path / "bin"
    bin_.mkdir(parents=True)
    for tool in ("mkdir", "true"):
        (bin_ / tool).symlink_to(next(Path(d) / tool for d in ("/usr/bin", "/bin") if (Path(d) / tool).exists()))
    (bin_ / "python3").symlink_to(sys.executable)
    _stand_in(bin_, "nproc", f"echo {16 if 'cores' in lacks else 32}")
    pages = (64 if "memory" in lacks else 256) << 18
    _stand_in(bin_, "getconf", f"case $1 in _PHYS_PAGES) echo {pages} ;; *) echo 4096 ;; esac")
    if "cargo" not in lacks:
        _stand_in(bin_, "cargo", "exit 0")
    if "lean" not in lacks:
        _stand_in(bin_, "lake", "exit 0")
    (tmp_path / "home").mkdir()
    done = subprocess.run(["/bin/bash", "benchmarks/recursion_e2e/job.sh", f"OUT={tmp_path / 'out'}",
                           f"FLOCK_WORK={tmp_path / 'work'}", *(f"{k}={v}" for k, v in env.items())], cwd=ROOT,
                          env={"PATH": str(bin_), "HOME": str(tmp_path / "home")}, capture_output=True, text=True)
    return done, [line for line in done.stderr.splitlines() if line.startswith("job.sh: ")]


@pytest.mark.parametrize("lacks", sorted(NEEDS))
def test_the_job_refuses_each_need_on_its_own_line_and_builds_nothing(tmp_path: Path, lacks: str) -> None:
    done, refused = _job(tmp_path, {lacks})
    expected = [NEEDS[lacks]] + ([] if GNU_TIME else ["no GNU time at /usr/bin/time"])
    assert done.returncode == 2 and len(refused) == len(expected), done.stderr
    assert all(any(n in line for line in refused) for n in expected), done.stderr
    assert all(": " in line.removeprefix("job.sh: ") for line in refused)
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["settings.json"]
    assert not (tmp_path / "work" / "cache").exists()


def test_the_memory_it_asks_is_128_gb_only_with_an_oprove_behind_the_zk_server(tmp_path: Path) -> None:
    _, refused = _job(tmp_path / "zk", {"memory", "cargo"})
    assert any("the job asks 128" in line for line in refused)
    _, refused = _job(tmp_path / "no-oprove", {"memory", "cargo"}, STEPS="build,inner", MODES="zk_server")
    assert not any("GiB of memory" in line for line in refused) and any("no cargo" in line for line in refused)
