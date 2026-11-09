"""`tools/verity/lean/sandbox.sh`: an RW_DIR reached through a symlink stays writable. `mount --bind` and
`/proc/self/mountinfo` name a directory by its resolved path (a slot tree `~/.cache/verity-check/lean-records/build-N` is a
link into the node's cache root), so the mounts the script keeps writable are matched by that path. A compiler's
temporary files go to a writable private TMPDIR: a node's TMPDIR (`/workspace/tmp`) is read-only inside."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

SANDBOX = Path(__file__).resolve().parents[1] / "sandbox.sh"
CONTROL = Path(__file__).resolve().parent / "fixtures" / "control"
UNSHARE = ["unshare", "--user", "--map-root-user", "--mount", "--net", "--pid", "--fork", "--kill-child"]

FAKE = {
    # the mount points mountinfo lists, one per line of $MOUNTS, as `awk '{print $5}' /proc/self/mountinfo` prints them
    "awk": '#!/bin/sh\ncat "$MOUNTS"\n',
    "mount": '#!/bin/sh\necho "$*" >> "$LOG"\nexit 0\n',
}

needs_unshare = pytest.mark.skipif(subprocess.run([*UNSHARE, "--", "true"], capture_output=True).returncode != 0,
                                   reason="no unprivileged user namespaces here (`unshare`), so no sandbox")


def fake_run(tmp: Path, args: list[str], mounts: list[str], tmpdir: str | None = None) -> tuple[list[str], str]:
    """``sandbox.sh ARGS -- CMD`` with `mount` logged, mountinfo's mount points ``mounts`` and TMPDIR ``tmpdir``: the
    `mount` calls it made, and the TMPDIR CMD got (TEMP and TMP the same)."""
    bin_ = tmp / "bin"
    bin_.mkdir()
    for name, body in FAKE.items():
        (bin_ / name).write_text(body)
        (bin_ / name).chmod(0o755)
    (tmp / "mounts").write_text("".join(m + "\n" for m in mounts))
    env = {"PATH": f"{bin_}:/usr/bin:/bin", "MOUNTS": str(tmp / "mounts"), "LOG": str(tmp / "log")}
    if tmpdir is not None:
        env["TMPDIR"] = tmpdir
    r = subprocess.run(["bash", str(SANDBOX), *args, "--", "sh", "-c", 'printf "%s\\n" "$TMPDIR" "$TEMP" "$TMP"'],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    got = r.stdout.splitlines()
    assert len(got) == 3 and len(set(got)) == 1, got
    return (tmp / "log").read_text().splitlines(), got[0]


def tmpfs(log: list[str]) -> list[str]:
    return [line.rsplit(" ", 1)[1] for line in log if line.startswith("-t tmpfs ")]


def test_a_symlinked_rw_dir_is_bound_and_kept_by_its_resolved_path(tmp_path):
    real, link = tmp_path / "cache-root" / "build-0", tmp_path / "build-0"
    real.mkdir(parents=True)
    link.symlink_to(real)
    other = tmp_path / "elsewhere"
    log, _ = fake_run(tmp_path, [str(link / ".lake")], ["/", str(real / ".lake"), str(other)])
    assert (real / ".lake").is_dir()
    assert f"--bind {real}/.lake {real}/.lake" in log
    read_only = [line.rsplit(" ", 1)[1] for line in log if line.startswith("-o remount,bind,ro ")]
    assert read_only == ["/", str(other)]


@pytest.mark.skipif(not Path("/dev/shm").is_dir(), reason="no /dev/shm here")
def test_a_symlinked_ro_dir_is_seen_by_its_resolved_path(tmp_path):
    """An RO_DIR under /dev/shm keeps /dev/shm the host's (a tmpfs would hide it), however the caller spells it."""
    link = tmp_path / "artifacts"
    link.symlink_to("/dev/shm/lean-artifacts-that-need-not-exist")
    log, _ = fake_run(tmp_path, ["--ro", str(link), str(tmp_path / "rw")], ["/"])
    assert "/dev/shm" not in tmpfs(log)
    assert "/var/tmp" in tmpfs(log) or not Path("/var/tmp").is_dir()


def test_a_tmpdir_under_an_rw_dir_is_kept_as_the_caller_spells_it(tmp_path):
    real, link = tmp_path / "cache-root" / "build-0", tmp_path / "build-0"
    real.mkdir(parents=True)
    link.symlink_to(real)
    _, got = fake_run(tmp_path, [str(link / ".lake")], ["/"], str(link / ".lake" / "tmp"))
    assert got == str(link / ".lake" / "tmp")


@pytest.mark.skipif(not Path("/dev/shm").is_dir(), reason="no /dev/shm here")
def test_a_tmpdir_outside_the_rw_dirs_is_the_private_tmp(tmp_path):
    """The RW_DIR is under /dev/shm, so /tmp gets its tmpfs wherever pytest's tmp_path is."""
    rw = Path(tempfile.mkdtemp(dir="/dev/shm", prefix="sandbox-test-"))
    try:
        log, got = fake_run(tmp_path, [str(rw / ".lake")], ["/"], str(tmp_path / "jobtmp"))
    finally:
        shutil.rmtree(rw)
    assert got == "/tmp" and "/tmp" in tmpfs(log)


@pytest.mark.parametrize("tmpdir", [None, "/workspace/tmp"])
def test_where_tmp_stays_the_hosts_tmpdir_is_a_tmpfs_in_the_first_rw_dir(tmp_path, tmpdir):
    """An RO_DIR under /tmp keeps /tmp the host's, which is read-only inside."""
    real, link = tmp_path / "cache-root" / "build-0", tmp_path / "build-0"
    real.mkdir(parents=True)
    link.symlink_to(real)
    log, got = fake_run(tmp_path, ["--ro", "/tmp/lean-artifacts-that-need-not-exist", str(link / ".lake"),
                                   str(tmp_path / "other" / ".lake")], ["/"], tmpdir)
    assert got == f"{real}/.lake/.sandbox-tmp"
    assert tmpfs(log)[-1] == got and "/tmp" not in tmpfs(log)


@needs_unshare
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


@needs_unshare
def test_inside_the_sandbox_an_overlaid_dir_takes_writes_in_its_upper_layer_and_outside_is_never_written(tmp_path):
    """`--overlay DIR TOP`, as `check.sampled_recompile` runs it: inside, DIR is writable and removing a file hides it; the
    writes land in TOP/upper, and the processes outside, which read DIR meanwhile, see its files as they were."""
    lower, top = tmp_path / "pkg" / ".lake", tmp_path / "work" / "sampled" / "0"
    (lower / "build").mkdir(parents=True)
    (lower / "build" / "M.olean").write_bytes(b"old")
    r = subprocess.run([*UNSHARE, "--", "bash", str(SANDBOX), "--overlay", str(lower), str(top), "--", "bash", "-c",
                        'rm "$1/build/M.olean" && ! test -e "$1/build/M.olean" && printf new > "$1/build/N.olean"',
                        "_", str(lower)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert [p.name for p in (lower / "build").iterdir()] == ["M.olean"]
    assert (lower / "build" / "M.olean").read_bytes() == b"old"
    assert (top / "upper" / "build" / "N.olean").read_bytes() == b"new"
    assert not (top / "upper" / "build" / "M.olean").is_file(), "a removed file is a whiteout in the upper layer"


def test_an_overlay_path_mount_would_misparse_is_refused(tmp_path):
    r = subprocess.run(["bash", str(SANDBOX), "--overlay", str(tmp_path / "a,b"), str(tmp_path / "t"), "--", "true"],
                       capture_output=True, text=True)
    assert r.returncode == 2 and "overlay" in r.stderr, r.stderr


@needs_unshare
@pytest.mark.parametrize("ro", [[], ["--ro", "/tmp/lean-artifacts-that-need-not-exist"]], ids=["as-called", "host-tmp"])
def test_inside_the_sandbox_a_read_only_tmpdir_is_replaced_by_a_private_writable_one(tmp_path, ro):
    rw, job = tmp_path / "rw" / ".lake", tmp_path / "jobtmp"
    job.mkdir()
    r = subprocess.run([*UNSHARE, "--", "bash", str(SANDBOX), *ro, str(rw), "--", "bash", "-c",
                        '! touch "$1/no" 2>/dev/null && f=$(mktemp) && echo x > "$f" && echo "$f"', "_", str(job)],
                       env={**os.environ, "TMPDIR": str(job)}, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert not any(job.iterdir())
    assert not Path(r.stdout.strip()).exists()


LAKEFILES = {
    "b": 'name = "b"\ndefaultTargets = ["B"]\n\n[[lean_lib]]\nname = "B"\n',
    "a": 'name = "a"\ndefaultTargets = ["A"]\n\n[[require]]\nname = "b"\npath = "../b"\n\n[[lean_lib]]\nname = "A"\n',
}
needs_lake = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")


def slot_tree(tmp: Path) -> tuple[Path, Path, list[str]]:
    """Package `a` requiring `b` by path, in a tree reached through a link as a slot tree is: (the tree, the link, the
    sandbox prefix with both packages' `.lake` writable, spelled through the link)."""
    real, link = tmp / "root" / "build-0", tmp / "build-0"
    for p in ("a", "b"):
        (real / p).mkdir(parents=True)
        (real / p / "lakefile.toml").write_text(LAKEFILES[p])
        shutil.copy(CONTROL / "lean-toolchain", real / p / "lean-toolchain")
    (real / "b" / "B.lean").write_text("def B.x : Nat := 1\n")
    (real / "a" / "A.lean").write_text("import B\ndef A.y : Nat := B.x + 1\n")
    link.symlink_to(real)
    return real, link, [*UNSHARE, "--", "bash", str(SANDBOX), str(link / "a" / ".lake"), str(link / "b" / ".lake")]


@needs_unshare
@needs_lake
def test_a_sandboxed_build_in_a_slot_tree_replaces_a_path_dependencys_outputs_and_links_with_a_read_only_tmpdir(tmp_path):
    """The two things a sandboxed build on a node does that a probe doesn't: Lake removes a path dependency's outputs
    before rebuilding it (in the dependency's own `.lake`, reached through the slot tree's link), and `leanc` compiles and
    links a C file in one step, as `facts` is built, which writes a temporary object under TMPDIR."""
    real, link, sandbox = slot_tree(tmp_path)
    job = tmp_path / "jobtmp"
    job.mkdir()
    (real / "a" / "x.c").write_text("int main(void) { return 0; }\n")
    env = {**os.environ, "LAKE_ARTIFACT_CACHE": "false"}
    subprocess.run(["lake", "build"], cwd=link / "a", env=env, check=True, capture_output=True)
    olean = real / "b" / ".lake" / "build" / "lib" / "lean" / "B.olean"
    before = olean.read_bytes()
    (real / "b" / "B.lean").write_text("def B.x : Nat := 2\n")
    for argv in (["lake", "build"], ["lake", "env", "leanc", "x.c", "-o", ".lake/x"]):
        r = subprocess.run([*sandbox, "--", *argv], cwd=link / "a", env={**env, "TMPDIR": str(job)},
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
    assert olean.read_bytes() != before
    assert (real / "a" / ".lake" / "x").exists() and not any(job.iterdir())


@needs_unshare
@needs_lake
def test_a_sandboxed_build_restores_from_a_read_only_artifact_cache_and_leaves_it_as_it_was(tmp_path):
    """As `check.py`'s build steps run it (`--ro` the cache, `LAKE_CACHE_DIR` set, `LAKE_ARTIFACT_CACHE` unset): Lake
    copies each artifact out of the cache (a hard link across the sandbox's mounts fails) and writes nothing into it, so
    the cache needs no write access."""
    real, link, sandbox = slot_tree(tmp_path)
    cache = tmp_path / "cache"
    env = {k: v for k, v in os.environ.items() if k != "LAKE_ARTIFACT_CACHE"} | {"LAKE_CACHE_DIR": str(cache)}
    subprocess.run(["lake", "build"], cwd=link / "a", env={**env, "LAKE_ARTIFACT_CACHE": "true"}, check=True,
                   capture_output=True)

    def contents() -> dict[str, bytes]:
        return {str(f.relative_to(cache)): f.read_bytes() for f in sorted(cache.rglob("*")) if f.is_file()}

    kept = contents()
    assert any(name.endswith(".olean") for name in kept)
    for p in ("a", "b"):
        shutil.rmtree(real / p / ".lake" / "build")
    r = subprocess.run([*sandbox, "--ro", str(cache), "--", "lake", "build", "-v"], cwd=link / "a", env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "restored artifact from cache" in r.stdout + r.stderr
    assert (real / "b" / ".lake" / "build" / "lib" / "lean" / "B.olean").stat().st_nlink == 1
    assert contents() == kept
