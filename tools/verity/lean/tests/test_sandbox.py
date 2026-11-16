"""`tools/verity/lean/sandbox.sh`: an RW_DIR reached through a symlink stays writable. `mount --bind` and
`/proc/self/mountinfo` name a directory by its resolved path (a slot tree `~/.cache/verity-check/lean-records/build-N` is a
link into the node's cache root), so the mounts the script keeps writable are matched by that path. A compiler's
temporary files go to a writable private TMPDIR: a node's TMPDIR (`/workspace/tmp`) is read-only inside. The command
can't change a mount, however it asks."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

SANDBOX = Path(__file__).resolve().parents[1] / "sandbox.sh"
CONTROL = Path(__file__).resolve().parent / "fixtures" / "control"
UNSHARE = ["unshare", "--user", "--map-root-user", "--mount", "--net", "--ipc", "--pid", "--fork", "--kill-child"]

FAKE = {
    # the mount points mountinfo lists, one per line of $MOUNTS, as `awk '{print $5}' /proc/self/mountinfo` prints them;
    # the mounts are never made, so the check that each stays read-only finds none writable
    "awk": '#!/bin/sh\ncase "$*" in *SANDBOX_KEPT*) exit 0;; esac\ncat "$MOUNTS"\n',
    "mount": '#!/bin/sh\necho "$*" >> "$LOG"\nexit 0\n',
    "setpriv": '#!/bin/sh\nwhile [ "$1" != -- ]; do shift; done\nshift\nexec "$@"\n',
}

#: what the sandbox hides under a private read-only tmpfs here, where no TMPDIR is set: /run, and /var/run unless it is
#: a link to it
HIDDEN = [d for d in ("/run", "/var/run") if Path(d).is_dir() and not Path(d).is_symlink()]

needs_unshare = pytest.mark.skipif(subprocess.run([*UNSHARE, "--", "true"], capture_output=True).returncode != 0,
                                   reason="no unprivileged user namespaces here (`unshare`), so no sandbox")


def fake_run(tmp: Path, args: list[str], mounts: list[str], tmpdir: str | None = None) -> tuple[list[str], str]:
    """``sandbox.sh ARGS -- CMD`` with `mount` logged, mountinfo's mount points ``mounts`` and TMPDIR ``tmpdir``: the
    `mount` calls it made, and the TMPDIR CMD got (TEMP and TMP the same). It runs in a user namespace of its own, whose
    count of user namespaces the script sets to 0."""
    bin_ = tmp / "bin"
    bin_.mkdir()
    for name, body in FAKE.items():
        (bin_ / name).write_text(body)
        (bin_ / name).chmod(0o755)
    (tmp / "mounts").write_text("".join(m + "\n" for m in mounts))
    env = {"PATH": f"{bin_}:/usr/bin:/bin", "MOUNTS": str(tmp / "mounts"), "LOG": str(tmp / "log")}
    if tmpdir is not None:
        env["TMPDIR"] = tmpdir
    r = subprocess.run(["unshare", "--user", "--map-root-user", "--", "bash", str(SANDBOX), *args, "--", "sh", "-c",
                        'printf "%s\\n" "$TMPDIR" "$TEMP" "$TMP"'], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    got = r.stdout.splitlines()
    assert len(got) == 3 and len(set(got)) == 1, got
    return (tmp / "log").read_text().splitlines(), got[0]


def tmpfs(log: list[str]) -> list[str]:
    return [line.rsplit(" ", 1)[1] for line in log if line.startswith("-t tmpfs ")]


@needs_unshare
def test_a_symlinked_rw_dir_is_bound_and_kept_by_its_resolved_path(tmp_path):
    real, link = tmp_path / "cache-root" / "build-0", tmp_path / "build-0"
    real.mkdir(parents=True)
    link.symlink_to(real)
    other = tmp_path / "elsewhere"
    log, _ = fake_run(tmp_path, [str(link / ".lake")], ["/", str(real / ".lake"), str(other)])
    assert (real / ".lake").is_dir()
    assert f"--bind {real}/.lake {real}/.lake" in log
    read_only = [line.rsplit(" ", 1)[1] for line in log if line.startswith("-o remount,bind,ro ")]
    assert read_only == ["/", str(other), *HIDDEN]


@needs_unshare
@pytest.mark.skipif(not Path("/dev/shm").is_dir(), reason="no /dev/shm here")
def test_a_symlinked_ro_dir_is_seen_by_its_resolved_path(tmp_path):
    """An RO_DIR under /dev/shm keeps /dev/shm the host's (a tmpfs would hide it), however the caller spells it."""
    link = tmp_path / "artifacts"
    link.symlink_to("/dev/shm/lean-artifacts-that-need-not-exist")
    log, _ = fake_run(tmp_path, ["--ro", str(link), str(tmp_path / "rw")], ["/"])
    assert "/dev/shm" not in tmpfs(log)
    assert "/var/tmp" in tmpfs(log) or not Path("/var/tmp").is_dir()


@needs_unshare
def test_a_tmpdir_under_an_rw_dir_is_kept_as_the_caller_spells_it(tmp_path):
    real, link = tmp_path / "cache-root" / "build-0", tmp_path / "build-0"
    real.mkdir(parents=True)
    link.symlink_to(real)
    _, got = fake_run(tmp_path, [str(link / ".lake")], ["/"], str(link / ".lake" / "tmp"))
    assert got == str(link / ".lake" / "tmp")


@needs_unshare
@pytest.mark.skipif(not Path("/dev/shm").is_dir(), reason="no /dev/shm here")
def test_a_tmpdir_outside_the_rw_dirs_is_the_private_tmp(tmp_path):
    """The RW_DIR is under /dev/shm, so /tmp gets its tmpfs wherever pytest's tmp_path is."""
    rw = Path(tempfile.mkdtemp(dir="/dev/shm", prefix="sandbox-test-"))
    try:
        log, got = fake_run(tmp_path, [str(rw / ".lake")], ["/"], str(tmp_path / "jobtmp"))
    finally:
        shutil.rmtree(rw)
    assert got == "/tmp" and "/tmp" in tmpfs(log)


@needs_unshare
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
def test_inside_the_sandbox_the_command_can_hold_no_capability_after_an_exec(tmp_path):
    """The command runs as the user namespace's root, so a capability left in its inheritable, ambient or bounding set
    is one it holds again after an `exec`: all three are empty (that it holds none now, the refused mounts below show)."""
    r = subprocess.run([*UNSHARE, "--", "bash", str(SANDBOX), str(tmp_path / ".lake"), "--", "setpriv", "--dump"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    sets = dict(line.split(": ", 1) for line in r.stdout.splitlines() if "capabilit" in line.lower())
    assert sets == dict.fromkeys(("Inheritable capabilities", "Ambient capabilities", "Capability bounding set"),
                                 "[none]"), sets


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
def test_inside_the_sandbox_a_read_only_dir_is_read_and_never_written(tmp_path):
    """`--ro DIR`, as `check.sampled_recompile` passes its seal: DIR is seen inside wherever it is (under `/tmp`, the host's
    `/tmp` is kept), and it is read-only there."""
    seal, out = tmp_path / "seal", tmp_path / "work" / "sampled" / "0"
    seal.mkdir()
    (seal / "M.olean").write_bytes(b"sealed")
    r = subprocess.run([*UNSHARE, "--", "bash", str(SANDBOX), "--ro", str(seal), str(out), "--", "bash", "-c",
                        'cat "$1/M.olean" > "$2/got" && ! touch "$1/no" 2>/dev/null', "_", str(seal), str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (out / "got").read_bytes() == b"sealed" and not (seal / "no").exists()


def linked_lake(tmp: Path, target: str) -> tuple[Path, Path]:
    """A tree whose `pkg/.lake` is a link to ``target``, which a change can commit (`.gitignore`'s `**/.lake/` matches only
    directories), beside `src/Main.lean`: (the tree, the package)."""
    tree = tmp / "tree"
    (tree / "src").mkdir(parents=True)
    (tree / "src" / "Main.lean").write_text("orig\n")
    (pkg := tree / "pkg").mkdir()
    (pkg / ".lake").symlink_to(target)
    return tree, pkg


@needs_unshare
@pytest.mark.parametrize("at", [False, True], ids=["plain", "at"])
def test_a_lake_the_tree_holds_as_a_link_to_its_sources_stops_the_sandbox_before_its_command_runs(tmp_path, at):
    """`pkg/.lake -> ../src`: bound read-write at its target, the build would rewrite the tree's sources, and with `--at`
    (`check.bound` sees the link resolve inside the tree) as much. The sandbox refuses the link and the source stays."""
    if at and not Path("/var/tmp").is_dir():
        pytest.skip("no /var/tmp here, where --at refuses")
    tree, pkg = linked_lake(tmp_path.resolve(), "../src")
    r = subprocess.run([*UNSHARE, "--", "bash", str(SANDBOX), *(["--at", FIXED, str(tree)] if at else []),
                        str(pkg / ".lake"), "--", "bash", "-c", "echo pwned > .lake/Main.lean"], cwd=pkg,
                       capture_output=True, text=True)
    assert r.returncode == 3, r.stdout + r.stderr
    assert f"sandbox.sh: {pkg / '.lake'}, a directory the build would write, is a symbolic link" in r.stderr, r.stderr
    assert (tree / "src" / "Main.lean").read_text() == "orig\n"


@needs_unshare
def test_a_lake_the_tree_holds_as_a_link_out_of_it_stops_the_sandbox_before_its_command_runs(tmp_path):
    """`pkg/.lake -> /an/absolute/directory` outside the tree (another slot's build, a cache): bound at its target, the
    build would write there. The sandbox refuses the link and the directory stays as it was."""
    tmp = tmp_path.resolve()
    (cache := tmp / "cache").mkdir()
    (cache / "e").write_text("entry\n")
    _, pkg = linked_lake(tmp, str(cache))
    r = subprocess.run([*UNSHARE, "--", "bash", str(SANDBOX), str(pkg / ".lake"), "--", "bash", "-c",
                        "echo poisoned > .lake/e"], cwd=pkg, capture_output=True, text=True)
    assert r.returncode == 3, r.stdout + r.stderr
    assert "is a symbolic link" in r.stderr, r.stderr
    assert (cache / "e").read_text() == "entry\n"


@needs_unshare
def test_a_read_only_dir_under_a_writable_one_that_is_a_link_stops_the_sandbox_before_its_command_runs(tmp_path):
    """`--ro pkg/.lake/packages`, as `lean_changed.in_sandbox` passes a package's fetched dependencies (it tests `is_dir`,
    which follows a link), where `packages` is a link: a change can commit one (`git add -f` inside the ignored `.lake`),
    and `check.dev_restore` makes one for the length of a `cache.py get`. The link would choose which directory is bound
    read-only, so the sandbox refuses it."""
    tmp = tmp_path.resolve()
    (lake := tmp / "tree" / "pkg" / ".lake").mkdir(parents=True)
    (deps := tmp / "tree" / "deps").mkdir()
    (lake / "packages").symlink_to(deps)
    r = subprocess.run([*UNSHARE, "--", "bash", str(SANDBOX), "--ro", str(lake / "packages"), str(lake), "--", "bash",
                        "-c", 'touch "$1/ran"', "_", str(lake)], capture_output=True, text=True)
    assert r.returncode == 3, r.stdout + r.stderr
    assert f"sandbox.sh: {lake / 'packages'}, a directory under {lake} that the build would read only, is a symbolic " \
           "link" in r.stderr, r.stderr
    assert not (lake / "ran").exists()


def remount_refused(tmp: Path, monkeypatch, refused: Path, binds: list[tuple[Path, Path]],
                    locked: Path | None = None, cwd: Path | None = None, fakes: dict[str, str] | None = None,
                    also: str = "") -> subprocess.CompletedProcess:
    """The sandbox, with the tree `tree` and the RO_DIR `cache` under ``tmp`` and the bound ``binds`` (source, target)
    each a mount of its own, made in a mount namespace of the test's own, where a `mount` before the real one refuses the
    read-only remount of ``refused``, as a node's kernel or mount can; the sandbox starts in ``cwd`` when given, entered
    after the mounts, and ``locked``, when given, is made mode 0 after that. ``fakes`` ({name: body}) are programs found
    before the real ones. CMD writes into each target (`no`), runs ``also``, and marks that it ran (`ran` in the
    RW_DIR)."""
    tree, cache = tmp / "tree", tmp / "cache"
    (tree / "pkg").mkdir(parents=True)
    cache.mkdir()
    binds = [(tree, tree), (cache, cache), *binds]
    (bin_ := tmp / "bin").mkdir()
    (bin_ / "mount").write_text(f'#!/bin/sh\n[ "$*" != "-o remount,bind,ro {refused}" ] || exit 32\n'
                                f'exec {shutil.which("mount")} "$@"\n')
    for name, body in {"mount": None, **(fakes or {})}.items():
        if body is not None:
            (bin_ / name).write_text(f"#!/bin/sh\n{body}\n")
        (bin_ / name).chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join([str(bin_), *os.get_exec_path()]))
    setup = " && ".join([*(f'mount --bind "{s}" "{t}"' for s, t in binds), *([f'cd "{cwd}"'] if cwd else []),
                         *([f'chmod 0 "{locked}"'] if locked else [])])
    writes = " ".join(f'"{t}/no"' for t in (tree / "pkg", *(t for _, t in binds[1:])))
    return subprocess.run(["unshare", "--user", "--map-root-user", "--mount", "--", "bash", "-c", setup + ' && exec "$@"',
                           "_", *UNSHARE, "--", "bash", str(SANDBOX), "--ro", str(cache), str(tree / "pkg" / ".lake"),
                           "--", "sh", "-c", f'touch {writes} 2>/dev/null; {also} touch "{tree}/pkg/.lake/ran"'],
                          capture_output=True, text=True)


def refused_before_cmd(r: subprocess.CompletedProcess, tmp: Path, *untouched: Path) -> None:
    """The sandbox exited 3 naming a writable mount, CMD never ran, and nothing landed in the tree or ``untouched``."""
    assert r.returncode == 3, r.stdout + r.stderr
    assert "these mounts stay writable here, which the build could write through:" in r.stderr, r.stderr
    assert not (tmp / "tree" / "pkg" / ".lake" / "ran").exists(), "CMD never ran"
    assert not (tmp / "tree" / "pkg" / "no").exists()
    for d in untouched:
        assert list(d.iterdir()) == [], d


def host_dir() -> Path:
    """A directory of the host's root (the one holding `sh`): bound as a mount, the command can write nothing through it."""
    system = Path(shutil.which("sh")).resolve().parent
    if system.stat().st_uid == os.getuid() or os.access(system, os.W_OK):
        pytest.skip(f"{system} is the runner's own or writable by it here")
    return system


@needs_unshare
@pytest.mark.parametrize("refused", ["tree", "cache", "locked/inner"])
def test_a_mount_the_command_could_write_whose_read_only_remount_fails_stops_the_sandbox_before_its_command_runs(
        tmp_path, monkeypatch, refused):
    """The tree, an RO_DIR (Lake's artifact cache, or a seal), or a directory of the runner's own under a directory the
    command can't search but owns (so could chmod), whose read-only remount fails. That mount would stay writable inside,
    so the sandbox names it and exits before CMD runs: no write lands in the tree, the cache or the mount."""
    tmp = tmp_path.resolve()
    (tmp / "locked" / "inner").mkdir(parents=True)
    (src := tmp / "src").mkdir()
    try:
        r = remount_refused(tmp, monkeypatch, tmp / refused, [(src, tmp / "locked" / "inner")], tmp / "locked")
    finally:
        (tmp / "locked").chmod(0o755)
    assert r.returncode == 3, r.stdout + r.stderr
    assert f"these mounts stay writable here, which the build could write through: {tmp / refused} (" in r.stderr, r.stderr
    assert not (tmp / "tree" / "pkg" / "no").exists() and not (tmp / "cache" / "no").exists()
    assert list(src.iterdir()) == []
    assert not (tmp / "tree" / "pkg" / ".lake" / "ran").exists(), "CMD never ran"


@needs_unshare
def test_a_mount_the_command_cannot_write_whose_read_only_remount_fails_leaves_the_command_to_run(tmp_path, monkeypatch):
    """A directory of the host's root (the one holding `sh`), bound as a mount whose read-only remount fails, as a
    node's tracefs stays writable: the command, which holds no capability and owns nothing there, can write nothing
    through it, so the sandbox runs it, and its write there fails."""
    system = host_dir()
    tmp = tmp_path.resolve()
    (target := tmp / "system").mkdir()
    r = remount_refused(tmp, monkeypatch, target, [(system, target)])
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tmp / "tree" / "pkg" / ".lake" / "ran").exists()
    assert not (system / "no").exists() and not (tmp / "tree" / "pkg" / "no").exists()


@needs_unshare
@pytest.mark.parametrize("tool, body, refused", [
    ("id", "exit 1", "locked/inner"),
    ("stat", "exit 0", "locked/inner"),
    ("find", "exit 1", "tree"),
    ("find", "echo \"find: '/x': Permission denied\"; echo \"hit $2\"; echo \"find: '/y': Permission denied\"; exit 1",
     "tree"),
], ids=["id-fails", "stat-says-nothing", "find-fails-silently", "find-hit-among-denials"])
def test_a_probe_tool_that_fails_or_says_nothing_stops_the_sandbox_before_its_command_runs(
        tmp_path, monkeypatch, tool, body, refused):
    """The probe decides "out of reach" only on what its tools said. A failing `id` leaves the command's uid unknown, so a
    locked directory it owns (and could chmod) must not read as someone else's; a `stat` that prints no owner, or a `find`
    that fails without a word (its one line is empty, which is no "Permission denied"), likewise: each is doubt, and doubt
    refuses. A `find` that exits 1 over "Permission denied" lines with a hit among them refuses too: every line must be
    a denial."""
    tmp = tmp_path.resolve()
    (tmp / "locked" / "inner").mkdir(parents=True)
    (src := tmp / "src").mkdir()
    try:
        r = remount_refused(tmp, monkeypatch, tmp / refused, [(src, tmp / "locked" / "inner")], tmp / "locked",
                            fakes={tool: body})
    finally:
        (tmp / "locked").chmod(0o755)
    refused_before_cmd(r, tmp, src)


@needs_unshare
def test_a_failing_dirname_stops_the_probe_at_once_and_the_sandbox_before_its_command_runs(tmp_path, monkeypatch):
    """The probe climbs from the mount to `/` with `dirname`. One that fails prints nothing, which never reaches `/`: the
    climb ends at that failure, with the mount writable. The stand-in counts its calls, and prints `/` after 64 so a climb
    that ignores its failure ends at all. On an absolute path outside the test's directory it is the real `dirname`: a
    host mount whose remount fails on this node (tracefs under /sys/kernel/debug on some) is probed too, as in
    production, and adds no call to the count."""
    tmp = tmp_path.resolve()
    count = tmp / "tree" / "pkg" / ".lake" / "dirname-calls"
    monkeypatch.setenv("COUNT", str(count))
    r = remount_refused(tmp, monkeypatch, tmp / "tree", [], fakes={
        "dirname": f'case "$2" in "{tmp}"|"{tmp}/"*|[!/]*|"") ;; *) exec {shutil.which("dirname")} "$@";; esac\n'
                   'n=$(cat "$COUNT" 2>/dev/null || echo 0); echo $((n + 1)) > "$COUNT"; [ "$n" -lt 64 ] || echo /; exit 1'})
    refused_before_cmd(r, tmp)
    assert int(count.read_text()) == 1


@needs_unshare
def test_a_mount_point_spelled_like_a_glob_is_probed_at_its_own_path(tmp_path, monkeypatch):
    """`/mnt/[ab]` beside `/mnt/a`: the stuck mount is `[ab]`, writable, and `a` is a host directory the command can't
    write. Expanded as a glob, the name would probe `a` and let the build run over `[ab]`."""
    system = host_dir()
    tmp = tmp_path.resolve()
    (glob := tmp / "g" / "[ab]").mkdir(parents=True)
    (tmp / "g" / "a").mkdir()
    (src := tmp / "src").mkdir()
    r = remount_refused(tmp, monkeypatch, glob, [(src, glob), (system, tmp / "g" / "a")])
    refused_before_cmd(r, tmp, src)


@needs_unshare
@pytest.mark.parametrize("name, escaped", [("sp ace", "sp\\040ace"), ("t\tab", "t\\011ab"), ("back\\slash", "back\\134slash"),
                                           ("new\nline", "new\\012line")], ids=["space", "tab", "backslash", "newline"])
def test_a_mount_point_mountinfo_escapes_is_probed_at_its_real_path(tmp_path, monkeypatch, name, escaped):
    """mountinfo writes a mount point's space, tab, newline and backslash as octal escapes. The stuck mount `name` is
    writable; a host directory the command can't write sits at the escaped spelling. Read without decoding, the probe
    would judge that directory and let the build run. A name holding a newline can't be passed on as one line: it is
    writable without a probe."""
    system = host_dir()
    tmp = tmp_path.resolve()
    (real := tmp / name).mkdir()
    (tmp / escaped).mkdir()
    (src := tmp / "src").mkdir()
    r = remount_refused(tmp, monkeypatch, real, [(src, real), (system, tmp / escaped)])
    refused_before_cmd(r, tmp, src)


@needs_unshare
def test_a_locked_directory_above_the_commands_own_directory_does_not_put_a_mount_below_it_out_of_reach(
        tmp_path, monkeypatch):
    """The command starts in `locked/work`, so it reaches `locked/work/sub` by a relative path without searching `locked`:
    that `locked` is someone else's (a `stat` stand-in says so) and unsearchable doesn't make the writable mount at
    `sub` out of reach."""
    tmp = tmp_path.resolve()
    (sub := tmp / "locked" / "work" / "sub").mkdir(parents=True)
    (src := tmp / "src").mkdir()
    try:
        r = remount_refused(tmp, monkeypatch, sub, [(src, sub)], tmp / "locked", cwd=tmp / "locked" / "work",
                            fakes={"stat": "echo 4242"}, also="touch sub/no 2>/dev/null;")
    finally:
        (tmp / "locked").chmod(0o755)
    refused_before_cmd(r, tmp, src)


@needs_unshare
@pytest.mark.skipif(not Path("/var/tmp").is_dir(), reason="no /var/tmp here")
def test_a_mount_a_later_one_hides_is_still_reached_from_the_commands_own_directory_inside_it(tmp_path, monkeypatch):
    """The sandbox starts in `/var/tmp`, a writable mount whose read-only remount fails; its private tmpfs then covers
    `/var/tmp`, but the command's own directory is still the mount below, written by a relative path."""
    tmp = tmp_path.resolve()
    (src := tmp / "src").mkdir()
    r = remount_refused(tmp, monkeypatch, Path("/var/tmp"), [(src, Path("/var/tmp"))], cwd=Path("/var/tmp"),
                        also="touch no 2>/dev/null;")
    refused_before_cmd(r, tmp, src)


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


#: check.py's `FIXED`, where a build reads its tree (`--at`)
FIXED = "/var/tmp/verity"
needs_var_tmp = pytest.mark.skipif(not Path("/var/tmp").is_dir(), reason="no /var/tmp here, where --at refuses")


def linked_tree(tmp: Path) -> tuple[Path, Path]:
    """A tree with a package `pkg` and its source, reached through a link as a slot tree is: (the tree, the link)."""
    real, link = tmp / "root" / "build-0", tmp / "build-0"
    (real / "pkg" / "src" / "deep").mkdir(parents=True)
    (real / "pkg" / "src" / "M.lean").write_text(f"source of {tmp.name}\n")
    link.symlink_to(real)
    return real, link


@needs_unshare
@needs_var_tmp
def test_inside_the_sandbox_a_tree_bound_at_the_fixed_path_is_read_there_and_written_only_in_its_rw_dirs(tmp_path):
    """`--at FIXED TREE`, as a build runs it: CMD reads the tree's files under FIXED, writes there only in the RW_DIRs,
    and those writes land in the tree. Nothing is made on the host: FIXED is in the sandbox's own /var/tmp."""
    real, link = linked_tree(tmp_path)
    was = Path(FIXED).exists()
    r = subprocess.run([*UNSHARE, "--", "bash", str(SANDBOX), "--at", FIXED, str(link), str(link / "pkg" / ".lake"), "--",
                        "bash", "-c", 'cat "$1/pkg/src/M.lean" && echo built > "$1/pkg/.lake/out" && '
                        '! touch "$1/pkg/src/no" 2>/dev/null && ! touch "$2/pkg/src/no" 2>/dev/null && cat "$2/pkg/.lake/out"',
                        "_", FIXED, str(real)], cwd=link / "pkg", capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [f"source of {tmp_path.name}", "built"]
    assert (real / "pkg" / ".lake" / "out").read_text() == "built\n"
    assert not (real / "pkg" / "src" / "no").exists()
    assert Path(FIXED).exists() == was


#: each write a build must not make in its tree, as (its name, Python that makes it under the tree's path `b`)
WRITES = {
    "append to a source": 'open(f"{b}/pkg/src/M.lean", "a").write("x")',
    "unlink a source": 'os.unlink(f"{b}/pkg/src/M.lean")',
    "rename a source": 'os.rename(f"{b}/pkg/src/M.lean", f"{b}/pkg/src/N.lean")',
    "chmod a source": 'os.chmod(f"{b}/pkg/src/M.lean", 0o600)',
    "write the lock": 'open(f"{b}/pkg/lean-audit.json", "w").write("{}")',
    "write another package's .lake": 'open(f"{b}/other/.lake/x", "w").write("x")',
    "write under a submount": 'open(f"{b}/sub/x", "w").write("x")',
}


@needs_unshare
@needs_var_tmp
def test_through_the_fixed_path_and_the_trees_own_every_write_outside_the_rw_dirs_fails_as_a_read_only_file_system(
        tmp_path):
    """The bound copy keeps each mount's read-only flag (`--rbind` after every mount is made read-only), so through FIXED,
    as through the tree's own path, a source file can't be appended to, unlinked, renamed or have its mode changed, and
    the lock, another package's `.lake` and a mount inside the tree can't be written: each fails with EROFS, while the
    RW_DIR takes its write. The submount is made in a mount namespace of the test's own, by a bind inside the tree."""
    real, _ = linked_tree(tmp_path)
    (real / "pkg" / "lean-audit.json").write_text("{}\n")
    (real / "other" / ".lake").mkdir(parents=True)
    (real / "sub").mkdir()
    (subsrc := tmp_path / "subsrc").mkdir()
    script = "\n".join([
        "import errno, os, sys",
        "for b in sys.argv[1:]:",
        *(f"    try:\n        {code}\n        print(b, {name!r}, 'written')\n"
          f"    except OSError as e:\n        print(b, {name!r}, errno.errorcode[e.errno])" for name, code in WRITES.items()),
        "    open(f'{b}/pkg/.lake/ok', 'a').write('ok')",
        "    print(b, 'rw', 'written')"])
    r = subprocess.run(["unshare", "--user", "--map-root-user", "--mount", "--", "bash", "-c",
                        'mount --bind "$0" "$1" && exec "${@:2}"', str(subsrc), str(real / "sub"),
                        *UNSHARE, "--", "bash", str(SANDBOX), "--at", FIXED, str(real), str(real / "pkg" / ".lake"), "--",
                        sys.executable, "-c", script, FIXED, str(real)], cwd=real / "pkg", capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.splitlines() == [f"{b} {w} {'written' if w == 'rw' else 'EROFS'}"
                                     for b in (FIXED, str(real)) for w in [*WRITES, "rw"]]
    assert (real / "pkg" / "src" / "M.lean").read_text() == f"source of {tmp_path.name}\n"
    assert not (real / "pkg" / "src" / "N.lean").exists() and (real / "pkg" / "src" / "M.lean").stat().st_mode & 0o777 != 0o600
    assert (real / "pkg" / "lean-audit.json").read_text() == "{}\n" and list((real / "other" / ".lake").iterdir()) == []
    assert list(subsrc.iterdir()) == [] and list((real / "sub").iterdir()) == []
    assert (real / "pkg" / ".lake" / "ok").read_text() == "okok"


@needs_unshare
@needs_var_tmp
@pytest.mark.parametrize("start", ["", "pkg", "pkg/src/deep"])
def test_with_at_cmd_starts_in_the_fixed_paths_copy_of_the_directory_it_was_started_in(tmp_path, start):
    """Lake names a source by the directory it runs in, so CMD starts under FIXED, wherever in the tree it was started
    and however that directory was spelled (a slot tree is a link): its physical directory, which Lean reads, is there."""
    real, link = linked_tree(tmp_path)
    r = subprocess.run([*UNSHARE, "--", "bash", str(SANDBOX), "--at", FIXED, str(real), str(link / "pkg" / ".lake"), "--",
                        "pwd", "-P"], cwd=link / start, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == str(Path(FIXED) / start)


@needs_unshare
def test_at_refuses_a_var_tmp_that_is_a_link(tmp_path):
    """Where /var/tmp is a link, its private tmpfs and FIXED land at the link's target, and a build bound there would name
    its sources under that path, which no other build's name. So `--at` reads CMD's physical directory (`pwd -P`) and
    refuses, with the fix, unless it is under FIXED. Here /var/tmp is made a link in a mount namespace of the test's own,
    by a directory bound over /var that holds the link."""
    if Path("/var").resolve() in [tmp_path.resolve(), *tmp_path.resolve().parents]:
        pytest.skip("the test's directory is under /var, which it covers")
    real, _ = linked_tree(tmp_path)
    (var := tmp_path / "var").mkdir()
    (target := tmp_path / "target").mkdir()
    (var / "tmp").symlink_to(target)
    r = subprocess.run(["unshare", "--user", "--map-root-user", "--mount", "--", "bash", "-c",
                        'mount --bind "$0" /var && readlink /var/tmp && exec "$@"', str(var),
                        *UNSHARE, "--", "bash", str(SANDBOX), "--at", FIXED, str(real), str(real / "pkg" / ".lake"), "--",
                        "pwd", "-P"], cwd=real / "pkg", capture_output=True, text=True)
    assert r.returncode == 2, r.stdout + r.stderr
    assert r.stdout.splitlines() == [str(target)], "only the outer shell's readlink: CMD never ran"
    assert f"the bound tree is at {target.resolve()}/verity/pkg, not under {FIXED}" in r.stderr, r.stderr
    assert "is a link here" in r.stderr and "Make /var/tmp a directory of its own on this host" in r.stderr
    assert list(target.iterdir()) == [], "the sandbox's tmpfs and FIXED were its own"


@needs_unshare
@needs_var_tmp
def test_two_sandboxes_at_once_each_read_their_own_tree_at_the_fixed_path(tmp_path):
    """FIXED is made in each sandbox's private /var/tmp, so two builds on one node, each of its own tree, never meet
    there: each reads its own source while the other is bound at the same path."""
    trees = [linked_tree(tmp_path / n)[0] for n in ("one", "two") if not (tmp_path / n).mkdir()]
    ps = [subprocess.Popen([*UNSHARE, "--", "bash", str(SANDBOX), "--at", FIXED, str(t), str(t / "pkg" / ".lake"), "--",
                            "bash", "-c", 'cat src/M.lean; read -r _; cat src/M.lean'], cwd=t / "pkg",
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for t in trees]
    try:
        first = [p.stdout.readline() for p in ps]       # both are bound at FIXED now, and stay so until told
        rest = [p.communicate("\n") for p in ps]
    finally:
        for p in ps:
            p.kill()
    assert first == ["source of one\n", "source of two\n"], [err for _, err in rest]
    assert [out for out, _ in rest] == first


@needs_unshare
def test_at_refuses_a_fixed_path_outside_var_tmp_a_start_outside_the_tree_and_a_var_tmp_that_stays_the_hosts(tmp_path):
    """A tree bound anywhere but FIXED would compile to other bytes than every other build's, without saying so. So
    `--at` takes one name in /var/tmp and nothing else, and where /var/tmp got no private tmpfs (a directory the sandbox
    was given lies under it, or the host has none) it refuses with the reason and the fix, and tries no other base."""
    real, link = linked_tree(tmp_path)

    def refused(args: list[str], cwd: Path) -> str:
        r = subprocess.run([*UNSHARE, "--", "bash", str(SANDBOX), *args, str(real / "pkg" / ".lake"), "--", "true"],
                           cwd=cwd, capture_output=True, text=True)
        assert r.returncode == 2, r.stdout + r.stderr
        return r.stderr

    for at in ("/srv/verity", "/var/tmp/a/b", "/var/tmp/", "/var/tmp/.."):
        assert "the fixed path is one name in /var/tmp" in refused(["--at", at, str(real)], real / "pkg")
    said = refused(["--ro", "/var/tmp/lean-artifacts-that-need-not-exist", "--at", FIXED, str(real)], real / "pkg")
    assert "would compile to other bytes" in said and "out of /var/tmp" in said
    assert ("stays the host's" if Path("/var/tmp").is_dir() else "/var/tmp is no directory here") in said
    if Path("/var/tmp").is_dir():
        assert "is not under that tree" in refused(["--at", FIXED, str(real)], tmp_path)
    assert not (real / "pkg" / "src" / "no").exists() and not Path("/var/tmp/lean-artifacts-that-need-not-exist").exists()


@needs_unshare
@needs_var_tmp
def test_an_ro_dir_under_an_rw_dir_is_read_only_at_its_own_path_and_through_the_fixed_path(tmp_path):
    """A build's fetched dependencies lie in its `.lake`, which it writes: handed as an RO_DIR, the script binds them
    read-only before it binds the tree at FIXED, whose copy keeps them so, since the command can mount nothing itself. An
    RO_DIR under an RW_DIR that doesn't exist is left alone."""
    real, _ = linked_tree(tmp_path)
    (deps := real / "pkg" / ".lake" / "packages" / "dep").mkdir(parents=True)
    (deps / "D.olean").write_text("kept")
    r = subprocess.run([*UNSHARE, "--", "bash", str(SANDBOX), "--ro", str(deps.parent), "--ro", str(real / "pkg" / ".lake" / "gone"),
                        "--at", FIXED, str(real), str(real / "pkg" / ".lake"), "--", sys.executable, "-c", APPEND,
                        f"{FIXED}/pkg/.lake/packages/dep/D.olean", str(deps / "D.olean"), f"{FIXED}/pkg/.lake/out"],
                       cwd=real / "pkg", capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.splitlines() == ["EROFS", "EROFS", "written"]
    assert (deps / "D.olean").read_text() == "kept" and (real / "pkg" / ".lake" / "out").read_text() == "x"
    assert not (real / "pkg" / ".lake" / "gone").exists()


#: each mount a command in the sandbox may ask for, as (its name, the shell that asks): `$AT` is FIXED, `$TREE` the tree
#: at its own path, `$OWN` the mount that holds the tree there, and `$AT/pkg/.lake/x` a directory of the RW_DIR
ASKS = {
    "remount FIXED read-write": 'mount -o remount,bind,rw "$AT"',
    "remount the tree's own mount read-write": 'mount -o remount,bind,rw "$OWN"',
    "mount a tmpfs over a source directory": 'mount -t tmpfs tmpfs "$AT/pkg/src"',
    "bind the RW_DIR over a source directory": 'mount --bind "$AT/pkg/.lake/x" "$TREE/pkg/src"',
    "unmount FIXED": 'umount -l "$AT"',
}
#: appends to each file named, printing what each append gave
APPEND = ("import errno, sys\nfor f in sys.argv[1:]:\n    try:\n        open(f, 'a').write('x')\n        print('written')\n"
          "    except OSError as e:\n        print(errno.errorcode[e.errno])")


def asked(tmp: Path, ask: str, opens: tuple[str, ...] = ()) -> list[str]:
    """What ``ask`` gave in a sandbox with a tree bound at FIXED, run in the namespaces ``opens`` makes when it is given:
    "not opened" where ``opens`` failed, else "made" or "refused", then what an append to a source gave through FIXED and
    through the tree's own path. Checks that the source and its directory are as they were."""
    real, _ = linked_tree(tmp)
    (real / "pkg" / ".lake" / "x").mkdir(parents=True)
    own = next(p for p in [real, *real.parents] if os.path.ismount(p))
    script = ('if sh -c "$ASK" >/dev/null 2>&1; then echo made; else echo refused; fi; '
              '"$PY" -c "$APPEND" "$AT/pkg/src/M.lean" "$TREE/pkg/src/M.lean"')
    env = {**os.environ, "AT": FIXED, "TREE": str(real), "OWN": str(own), "ASK": ask, "PY": sys.executable,
           "APPEND": APPEND}
    r = subprocess.run([*UNSHARE, "--", "bash", str(SANDBOX), "--at", FIXED, str(real), str(real / "pkg" / ".lake"), "--",
                        "sh", "-c", 'cd / && { "$@" sh -c "$0" || echo "not opened"; }', script, *opens],
                       cwd=real / "pkg", env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (real / "pkg" / "src" / "M.lean").read_text() == f"source of {tmp.name}\n"
    assert sorted(p.name for p in (real / "pkg" / "src").iterdir()) == ["M.lean", "deep"]
    assert list((real / "pkg" / ".lake" / "x").iterdir()) == []
    return r.stdout.splitlines()


@needs_unshare
@needs_var_tmp
@pytest.mark.parametrize("ask", ["remount FIXED read-write", "remount the tree's own mount read-write"])
def test_inside_the_sandbox_a_read_only_mount_cannot_be_remounted_writable(tmp_path, ask):
    """Code that runs while Lean compiles may ask for any mount. The command holds no capability, so `mount -o
    remount,bind,rw` of FIXED, or of the mount that holds the tree at its own path, is refused, and an append to a source
    through either path still fails with EROFS."""
    assert asked(tmp_path, ASKS[ask]) == ["refused", "EROFS", "EROFS"]


@needs_unshare
@needs_var_tmp
@pytest.mark.parametrize("ask", ["mount a tmpfs over a source directory", "bind the RW_DIR over a source directory"])
def test_inside_the_sandbox_no_mount_can_be_made_over_a_read_only_directory(tmp_path, ask):
    """A tmpfs, or a writable directory bound over a source directory, would let the command put other sources where the
    build reads them. Both mounts are refused."""
    assert asked(tmp_path, ASKS[ask]) == ["refused", "EROFS", "EROFS"]


@needs_unshare
@needs_var_tmp
def test_inside_the_sandbox_a_read_only_mount_cannot_be_unmounted(tmp_path):
    """Unmounting FIXED (detached, as a busy mount is) would leave an empty directory of the sandbox's tmpfs there, for
    the command to fill. It is refused, and FIXED stays the bound tree."""
    assert asked(tmp_path, ASKS["unmount FIXED"]) == ["refused", "EROFS", "EROFS"]


@needs_unshare
@needs_var_tmp
@pytest.mark.parametrize("ask", ASKS)
@pytest.mark.parametrize("opens", [("unshare", "--mount"), ("unshare", "--user", "--map-root-user", "--mount")],
                         ids=["mount", "user-and-mount"])
def test_inside_the_sandbox_namespaces_of_the_commands_own_give_it_no_mount(tmp_path, ask, opens):
    """The command can't get a capability back in namespaces of its own: a mount namespace needs one, and the sandbox's
    user namespace may hold no other (a new user namespace would hold every capability over the mount namespace it
    makes). So it never reaches its mount, and the source stays as it was."""
    assert asked(tmp_path, ASKS[ask], opens) == ["not opened"]


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
@needs_var_tmp
def test_a_real_build_whose_compile_time_code_writes_a_source_through_a_linked_lake_is_refused_before_lake_runs(tmp_path):
    """A change commits `pkg/.lake -> ../src` beside a module whose `#eval`, run while it compiles, writes `.lake/Main.lean`,
    which is `src/Main.lean`, and its `lake-manifest.json` (without it Lake's first write, the manifest's, fails in the
    read-only package before anything compiles). In the sandbox, as a build runs (`--at`), the build is refused and the
    source stays; the same `lake build` outside it rewrites the source, so the `#eval` is a writer."""
    tree = tmp_path.resolve() / "tree"
    (src := tree / "src").mkdir(parents=True)
    (src / "Main.lean").write_text("orig\n")
    (pkg := tree / "pkg").mkdir()
    (pkg / "lakefile.toml").write_text('name = "p"\ndefaultTargets = ["P"]\n\n[[lean_lib]]\nname = "P"\n')
    shutil.copy(CONTROL / "lean-toolchain", pkg / "lean-toolchain")
    (pkg / "P.lean").write_text('#eval IO.FS.writeFile ".lake/Main.lean" "pwned\\n"\n')
    (pkg / ".lake").symlink_to("../src")
    env = {**os.environ, "LAKE_ARTIFACT_CACHE": "false"}
    subprocess.run(["lake", "update"], cwd=pkg, env=env, check=True, capture_output=True)
    assert (pkg / "lake-manifest.json").is_file() and (src / "Main.lean").read_text() == "orig\n"
    r = subprocess.run([*UNSHARE, "--", "bash", str(SANDBOX), "--at", FIXED, str(tree), str(pkg / ".lake"), "--", "lake",
                        "build"], cwd=pkg, env=env, capture_output=True, text=True)
    assert r.returncode == 3, r.stdout + r.stderr
    assert f"sandbox.sh: {pkg / '.lake'}, a directory the build would write, is a symbolic link" in r.stderr, r.stderr
    assert (src / "Main.lean").read_text() == "orig\n"
    subprocess.run(["lake", "build"], cwd=pkg, env=env, check=True, capture_output=True)
    assert (src / "Main.lean").read_text() == "pwned\n"


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


SOCKETS = SANDBOX.parent / "sockets.py"
#: run as the command runs: a line per name, `reached` or `refused`
CONNECT = """import socket, sys
for name in sys.argv[1:]:
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect("\\0" + name[1:] if name.startswith("@") else name)
        print(name, "reached")
    except OSError:
        print(name, "refused")
"""


def listening(path: str) -> socket.socket:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind("\0" + path[1:] if path.startswith("@") else path)
    s.listen(8)
    return s


@needs_unshare
@pytest.mark.parametrize("fetch", [False, True], ids=["build", "fetch"])
def test_the_command_reaches_no_unix_socket_it_did_not_make(tmp_path, fetch):
    """A read-only mount doesn't stop `connect`, and the build's network namespace hides only abstract names: a socket
    the host listens on (a user's systemd bus, through which `systemd-run --user` runs anything outside the sandbox) was
    reachable from the command. Here a socket outside every RW_DIR, one in the job's TMPDIR and an abstract name the
    host's network namespace holds (which a fetch keeps) are each refused, and the TMPDIR's is not even seen."""
    (outside := tmp_path / "outside").mkdir()
    (job := tmp_path / "jobtmp").mkdir()
    names = [str(outside / "s"), str(job / "s"), f"@verity-sandbox-test-{os.getpid()}-{tmp_path.name}"]
    held = [listening(n) for n in names]
    try:
        assert all(line.endswith("reached") for line in subprocess.run(
            [sys.executable, "-c", CONNECT, *names], capture_output=True, text=True, check=True).stdout.splitlines())
        unshare = [a for a in UNSHARE if a != "--net"] if fetch else UNSHARE
        r = subprocess.run([*unshare, "--", "bash", str(SANDBOX), *(["--net"] if fetch else []), str(tmp_path / "rw"),
                            "--", "bash", "-c", '[ ! -e "$1" ] && echo unseen; shift; exec "$@"', "_", str(job / "s"),
                            sys.executable, "-I", "-S", "-c", CONNECT, *names],
                           env={**os.environ, "TMPDIR": str(job)}, capture_output=True, text=True)
    finally:
        for s in held:
            s.close()
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == ["unseen", *(f"{n} refused" for n in names)]


def test_the_probe_refuses_a_socket_the_command_could_reach_and_names_it(tmp_path):
    """The probe, run unconfined as a broken filter would leave the command: a listening socket of the inventory is
    reached and named, exit 3, and so is one only a D-Bus address names; a socket file no process listens on is not."""
    live, bus, dead = str(tmp_path / "live"), str(tmp_path / "bus"), str(tmp_path / "dead")
    held = [listening(live), listening(bus)]
    listening(dead).close()
    try:
        r = subprocess.run([sys.executable, "-I", "-S", str(SOCKETS), "probe", f"unix:path={bus},guid=0", ""],
                           capture_output=True, text=True, input=f"{live}\n{dead}\n", env={})
    finally:
        for s in held:
            s.close()
    assert r.returncode == 3, r.stderr
    assert f"{live} (stream)" in r.stderr and f"{bus} (stream)" in r.stderr and dead not in r.stderr


@pytest.mark.skipif(os.uname().machine not in ("x86_64", "aarch64"), reason="no seccomp filter for this machine")
def test_under_the_filter_no_unix_socket_and_no_io_uring_can_be_made_but_a_pair_can(tmp_path):
    """`sockets.py exec`'s filter: `socket(AF_UNIX, ...)` and `io_uring_setup` (whose rings make sockets without
    `socket`) fail with EACCES, a `socketpair` and an inet socket are made, and the filter outlives an `exec`."""
    check = ("import ctypes, socket\n"
             "try:\n    socket.socket(socket.AF_UNIX); print('unix made')\n"
             "except PermissionError: print('unix refused')\n"
             "libc = ctypes.CDLL(None, use_errno=True)\n"
             "p = ctypes.create_string_buffer(120)\n"
             "print('uring', 'refused' if libc.syscall(425, 1, p) == -1 and ctypes.get_errno() == 13 else 'made')\n"
             "a, b = socket.socketpair(); a.sendall(b'x'); print('pair', b.recv(1).decode())\n"
             "socket.socket(socket.AF_INET).close(); print('inet made')\n")
    r = subprocess.run([sys.executable, "-I", "-S", str(SOCKETS), "exec", "--", "sh", "-c", 'exec "$0" -c "$1"',
                        sys.executable, check], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == ["unix refused", "uring refused", "pair x", "inet made"]


@pytest.mark.skipif(os.uname().machine not in ("x86_64", "aarch64"), reason="no seccomp filter for this machine")
def test_under_the_filter_no_pair_sends_to_a_datagram_socket_bound_outside(tmp_path):
    """A unix datagram pair sends to any datagram socket by its address (`sendto`), and the kernel makes a `SOCK_RAW`
    pair a datagram one: under the filter neither pair, nor a seqpacket one, is made, and nothing sent from inside
    arrives at a datagram socket bound outside."""
    path = str(tmp_path / "outside")
    outside = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    outside.bind(path)
    outside.setblocking(False)
    send = ("import socket, sys\n"
            "for name in ('SOCK_DGRAM', 'SOCK_RAW', 'SOCK_SEQPACKET'):\n"
            "    try:\n        a, b = socket.socketpair(socket.AF_UNIX, getattr(socket, name))\n"
            "    except PermissionError:\n        print(name, 'refused'); continue\n"
            "    try:\n        a.sendto(name.encode(), sys.argv[1]); print(name, 'sent')\n"
            "    except OSError as e:\n        print(name, 'made, not sent', e.errno)\n")
    try:
        r = subprocess.run([sys.executable, "-I", "-S", str(SOCKETS), "exec", "--", sys.executable, "-I", "-S", "-c",
                            send, path], capture_output=True, text=True)
        try:
            arrived = outside.recv(64)
        except BlockingIOError:
            arrived = None
    finally:
        outside.close()
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == ["SOCK_DGRAM refused", "SOCK_RAW refused", "SOCK_SEQPACKET refused"]
    assert arrived is None


def test_the_inventory_lists_a_socket_in_a_directory_no_walk_can_list(tmp_path):
    """The kernel lists every bound name, so a socket in a directory one may search but not read, which `find -type s`
    can't see, is still tried."""
    (hid := tmp_path / "hid").mkdir()
    held = listening(str(hid / "s"))
    hid.chmod(0o311)
    try:
        assert not subprocess.run(["find", str(hid), "-type", "s"], capture_output=True, text=True).stdout
        r = subprocess.run([sys.executable, "-I", "-S", str(SOCKETS), "inventory"], capture_output=True, text=True)
    finally:
        hid.chmod(0o755)
        held.close()
    assert r.returncode == 0, r.stderr
    assert str(hid / "s") in r.stdout.splitlines()
