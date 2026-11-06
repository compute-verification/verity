"""`tools/verity/lean/sandbox.sh`: an RW_DIR reached through a symlink stays writable. `mount --bind` and
`/proc/self/mountinfo` name a directory by its resolved path (a slot tree `~/.cache/verity-check/lean-records/build-N` is a
link into the node's cache root), so the mounts the script keeps writable are matched by that path."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

SANDBOX = Path(__file__).resolve().parents[1] / "sandbox.sh"
UNSHARE = ["unshare", "--user", "--map-root-user", "--mount", "--net", "--pid", "--fork", "--kill-child"]

FAKE = {
    # the mount points mountinfo lists, one per line of $MOUNTS, as `awk '{print $5}' /proc/self/mountinfo` prints them
    "awk": '#!/bin/sh\ncat "$MOUNTS"\n',
    "mount": '#!/bin/sh\necho "$*" >> "$LOG"\nexit 0\n',
}


def fake_run(tmp: Path, args: list[str], mounts: list[str]) -> list[str]:
    """``sandbox.sh ARGS -- true`` with `mount` logged and mountinfo's mount points ``mounts``: the `mount` calls it made."""
    bin_ = tmp / "bin"
    bin_.mkdir()
    for name, body in FAKE.items():
        (bin_ / name).write_text(body)
        (bin_ / name).chmod(0o755)
    (tmp / "mounts").write_text("".join(m + "\n" for m in mounts))
    env = {"PATH": f"{bin_}:/usr/bin:/bin", "MOUNTS": str(tmp / "mounts"), "LOG": str(tmp / "log")}
    r = subprocess.run(["bash", str(SANDBOX), *args, "--", "true"], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return (tmp / "log").read_text().splitlines()


def test_a_symlinked_rw_dir_is_bound_and_kept_by_its_resolved_path(tmp_path):
    real, link = tmp_path / "cache-root" / "build-0", tmp_path / "build-0"
    real.mkdir(parents=True)
    link.symlink_to(real)
    other = tmp_path / "elsewhere"
    log = fake_run(tmp_path, [str(link / ".lake")], ["/", str(real / ".lake"), str(other)])
    assert (real / ".lake").is_dir()
    assert f"--bind {real}/.lake {real}/.lake" in log
    read_only = [line.rsplit(" ", 1)[1] for line in log if line.startswith("-o remount,bind,ro ")]
    assert read_only == ["/", str(other)]


@pytest.mark.skipif(not Path("/dev/shm").is_dir(), reason="no /dev/shm here")
def test_a_symlinked_ro_dir_is_seen_by_its_resolved_path(tmp_path):
    """An RO_DIR under /dev/shm keeps /dev/shm the host's (a tmpfs would hide it), however the caller spells it."""
    link = tmp_path / "artifacts"
    link.symlink_to("/dev/shm/lean-artifacts-that-need-not-exist")
    log = fake_run(tmp_path, ["--ro", str(link), str(tmp_path / "rw")], ["/"])
    tmpfs = [line.rsplit(" ", 1)[1] for line in log if line.startswith("-t tmpfs ")]
    assert "/dev/shm" not in tmpfs
    assert "/var/tmp" in tmpfs or not Path("/var/tmp").is_dir()


@pytest.mark.skipif(subprocess.run([*UNSHARE, "--", "true"], capture_output=True).returncode != 0,
                    reason="no unprivileged user namespaces here (`unshare`), so no sandbox")
def test_inside_the_sandbox_a_symlinked_rw_dir_is_writable_and_the_rest_is_not(tmp_path):
    real, link, other = tmp_path / "root" / "build-1", tmp_path / "build-1", tmp_path / "other"
    real.mkdir(parents=True)
    other.mkdir()
    link.symlink_to(real)
    r = subprocess.run([*UNSHARE, "--", "bash", str(SANDBOX), str(link / ".lake"), "--", "bash", "-c",
                        'touch "$1/ok" && ! touch "$2/no" 2>/dev/null', "_", str(link / ".lake"), str(other)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (real / ".lake" / "ok").exists()
    assert not (other / "no").exists()
