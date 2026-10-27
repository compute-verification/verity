"""`research pods ssh` / `research pods sync`: reach a pod by the name it was given, and put a worktree's code on it.

    research pods ssh TARGET [--print] [-- CMD ...]
    research pods sync TARGET [SRC] [--dest DIR] [--exclude GLOB]... [--dry-run]

TARGET is a `machines.toml` name, a RunPod pod id, or a pod name (`vy-ligerito-sumcheck-2`; a project suffix such as the
`-veritor-campaign` that `pods create` appends is optional, `pods.POD_SUFFIXES`).  The ssh port is looked up at call time (`remote.pod_target`: the REST API's flaky empty
mappings are retried), so nothing hard-codes an ip/port that changes when a pod restarts.

`sync` ships exactly the files git would show for SRC (default: the repository containing cwd): tracked files that still exist
plus untracked files that are not ignored -- no `.git`, no `.venv`, no build products, nothing gitignored.  It ships them to DEST
(default /workspace/src; rsync, or one tar stream when the machine has no rsync yet), removes the files a previous sync shipped that are gone from SRC now (only those: pod-side venvs,
caches and results are never touched), and writes `DEST/.research-source.json` with the commit / tree / dirty facts of SRC as
`research run` would record them (`env.source_conditions`).
"""
from __future__ import annotations

import argparse
import fnmatch
import io
import json
import os
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .. import remote
from . import POD_SUFFIXES, strip_pod_suffix

POD_ID_RE = re.compile(r"^[a-z0-9]{12,16}$")
DEFAULT_DEST = "/workspace/src"
STAMP = ".research-source.json"
SHIPPED = ".research-sync-files"
STAMP_SCHEMA = "research/source-stamp/v1"


# ---- targets --------------------------------------------------------------------------------------------------------------------------------

def resolve_pod(target: str, *, machines: dict[str, dict[str, Any]] | None = None, pods: Callable[[], list[dict]] | None = None) -> tuple[str, str]:
    """(pod id, label) for a machines.toml name, a pod id, or a pod name.  Refused (by name, with the candidates) when nothing or
    more than one pod matches."""
    ms = remote.load_machines() if machines is None else machines
    if target in ms:
        return ms[target]["pod_id"], target
    listing = (pods or _list_pods)()
    by_id = [p for p in listing if p.get("id") == target]
    if by_id:
        return target, by_id[0].get("name") or target
    names = {target, *(target + s for s in POD_SUFFIXES)}
    hits = [p for p in listing if p.get("name") in names]
    if len(hits) == 1:
        return hits[0]["id"], target
    if hits:
        raise remote.Refused(f"{target!r} names {len(hits)} pods ({', '.join(p['id'] for p in hits)}); give the pod id")
    if POD_ID_RE.match(target):                  # an id the listing did not show (another account view, a just-created pod)
        return target, target
    known = sorted({strip_pod_suffix(p.get("name") or "") for p in listing} | set(ms))
    raise remote.Refused(f"no machine or pod named {target!r} (machines.toml names and running pods: {', '.join(known) or 'none'})")


def _list_pods() -> list[dict]:
    from . import runpod

    try:
        return runpod._request("GET", "/pods") or []  # noqa: SLF001  (runpod.py is kept verbatim; its list verb only prints)
    except runpod.PodError as e:
        raise remote.Refused(f"runpod API unavailable: {e}") from None


def ssh_base(target: str) -> list[str]:
    """`ssh ... user@host` for TARGET, ready for a remote command to be appended."""
    m = remote.load_machines().get(target)
    if m is not None and m.get("provider") == "ssh":
        tgt = remote.ssh_host_target({**m, "name": target})
        return ssh_for(tgt["host"], tgt["port"], user=tgt["user"])
    pod_id, label = resolve_pod(target)
    tgt = remote.pod_target({"provider": "runpod", "pod_id": pod_id, "name": label})
    return ssh_for(tgt["host"], tgt["port"])


def ssh_for(host: str, port: int, *, user: str = "root") -> list[str]:
    from . import runpod

    base = runpod.ssh_command(host, port, user=user)
    return [*base[:-1], "-o", "LogLevel=ERROR", base[-1]]


# ---- transports -----------------------------------------------------------------------------------------------------------------------------

class _Transport:
    rsync_probe = "command -v rsync >/dev/null 2>&1"

    def argv(self, script: str) -> list[str]:
        raise NotImplementedError

    def sh(self, script: str, stdin: bytes | None = None, timeout: float = 300) -> subprocess.CompletedProcess:
        return subprocess.run(self.argv(script), input=stdin, capture_output=True, timeout=timeout)


@dataclass
class SshTransport(_Transport):
    """The pod side of a sync: shell commands over ssh, and the rsync `-e` / destination spelling."""
    ssh: list[str]

    def argv(self, script: str) -> list[str]:
        return [*self.ssh, script]

    def rsync_args(self, dest: str) -> list[str]:
        return ["-e", shlex.join(self.ssh[:-1]), f"{self.ssh[-1]}:{dest}/"]


@dataclass
class LocalTransport(_Transport):
    """The same protocol against a local directory (tests; a pod-side worktree refresh).  `rsync=False` plays a machine without
    rsync, so the tar stream is what ships."""
    rsync: bool = True

    def __post_init__(self) -> None:
        if not self.rsync:
            self.rsync_probe = "false"

    def argv(self, script: str) -> list[str]:
        return ["sh", "-c", script]

    def rsync_args(self, dest: str) -> list[str]:
        return [f"{dest}/"]


# ---- sync -----------------------------------------------------------------------------------------------------------------------------------

def git_toplevel(path: Path) -> Path:
    r = subprocess.run(["git", "-C", str(path), "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if r.returncode != 0:
        raise remote.Refused(f"{path} is not inside a git worktree ({r.stderr.strip()})")
    return Path(r.stdout.strip())


def git_visible_files(src: Path, excludes: list[str] | None = None) -> list[str]:
    """Tracked files that exist + untracked files that are not ignored, sorted; minus `excludes` (fnmatch on the path, or a
    directory prefix).  Gitlinks (submodules) and tracked-but-deleted paths drop out on the existence check."""
    r = subprocess.run(["git", "-C", str(src), "ls-files", "-z", "--cached", "--others", "--exclude-standard"], capture_output=True)
    if r.returncode != 0:
        raise remote.Refused(f"git ls-files failed in {src}: {r.stderr.decode(errors='replace').strip()}")
    out = []
    for rel in sorted({p for p in r.stdout.decode("utf-8", errors="surrogateescape").split("\0") if p}):
        if "\n" in rel:
            raise remote.Refused(f"{rel!r}: a path with a newline cannot travel in an rsync file list")
        if any(fnmatch.fnmatch(rel, g) or rel.startswith(g.rstrip("/") + "/") for g in excludes or []):
            continue
        p = src / rel
        if p.is_symlink() or p.is_file():
            out.append(rel)
    return out


def source_stamp(src: Path, n_files: int, dest_label: str) -> dict[str, Any]:
    from ..env import source_conditions

    c = source_conditions(src)
    branch = subprocess.run(["git", "-C", str(src), "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    return {"schema": STAMP_SCHEMA, "commit": c["source.commit"], "tree": c["source.tree"], "dirty": c["source.dirty"],
            "dirty_digest": c["source.dirty_digest"], "branch": branch, "from": f"{os.uname().nodename}:{src}", "to": dest_label,
            "files": n_files, "synced_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def _tar(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, data in files.items():
            ti = tarfile.TarInfo(name)
            ti.size, ti.mtime, ti.mode = len(data), int(time.time()), 0o644
            tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def _ship_rsync(src: Path, files: list[str], dest: str, transport: SshTransport | LocalTransport) -> None:
    mk = transport.sh(f"mkdir -p {shlex.quote(dest)}")
    if mk.returncode != 0:
        raise remote.Refused(f"mkdir -p {dest} failed: {mk.stderr.decode(errors='replace').strip()[-300:]}")
    *opts, dest_spec = transport.rsync_args(dest)
    with tempfile.NamedTemporaryFile("w", suffix=".files", delete=False, encoding="utf-8", errors="surrogateescape") as fl:
        fl.write("".join(p + "\n" for p in files))
    try:
        r = subprocess.run(["rsync", "-a", f"--files-from={fl.name}", *opts, f"{src}/", dest_spec], capture_output=True, text=True)
    finally:
        os.unlink(fl.name)
    if r.returncode != 0:
        raise remote.Refused(f"rsync failed (rc {r.returncode}): {r.stderr.strip()[-400:]}")


def _ship_tar(src: Path, files: list[str], dest: str, transport: SshTransport | LocalTransport, timeout: float = 1800) -> None:
    """A machine without rsync (a fresh image): every file, streamed as one tar written by Python (no platform tar quirks: no
    AppleDouble `._*` members, no xattr headers), unpacked by the machine's `tar`."""
    q = shlex.quote(dest)
    p = subprocess.Popen(transport.argv(f"mkdir -p {q} && tar -xf - -C {q}"), stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                         stderr=subprocess.PIPE)
    assert p.stdin is not None and p.stderr is not None
    try:
        with tarfile.open(fileobj=p.stdin, mode="w|", format=tarfile.PAX_FORMAT) as tf:
            for rel in files:
                tf.add(src / rel, arcname=rel, recursive=False)
    except BrokenPipeError:
        pass # the far side died; its stderr says why
    finally:
        p.stdin.close()
    err = p.stderr.read().decode(errors="replace")
    if p.wait(timeout=timeout) != 0:
        raise remote.Refused(f"tar stream to {dest} failed (rc {p.returncode}): {err.strip()[-400:]}")


def sync(src: Path, dest: str, transport: SshTransport | LocalTransport, *, excludes: list[str] | None = None, dry_run: bool = False,
         dest_label: str | None = None) -> dict[str, Any]:
    """Ship git-visible files of `src` to `dest` through `transport` (rsync when the machine has it, else a tar stream); prune what
    the previous sync shipped and SRC no longer has; write the source stamp and the shipped list.  Returns the report."""
    if not dest.startswith("/"):
        raise remote.Refused(f"--dest must be an absolute path on the machine, got {dest!r}")
    t0 = time.monotonic()
    files = git_visible_files(src, excludes)
    q = shlex.quote(dest)
    prev = transport.sh(f"({transport.rsync_probe} && echo rsync || echo tar); cat {q}/{SHIPPED} 2>/dev/null || true")
    if prev.returncode != 0:
        raise remote.Refused(f"cannot read {dest}/{SHIPPED}: {prev.stderr.decode(errors='replace').strip()[-300:]}")
    method, *listed = prev.stdout.decode("utf-8", errors="surrogateescape").split("\n")
    before = {p for p in listed if p}
    stale = sorted(before - set(files))
    stamp = source_stamp(src, len(files), dest_label or dest)
    rep: dict[str, Any] = {"src": str(src), "dest": dest_label or dest, "files": len(files), "pruned": stale, "stamp": stamp, "dry_run": dry_run,
                           "via": method}
    if dry_run:
        rep["seconds"] = round(time.monotonic() - t0, 2)
        return rep
    (_ship_rsync if method == "rsync" else _ship_tar)(src, files, dest, transport)
    payload = _tar({STAMP: (json.dumps(stamp, indent=2, sort_keys=True) + "\n").encode(),
                    SHIPPED: "".join(p + "\n" for p in files).encode("utf-8", errors="surrogateescape"),
                    ".research-sync-prune": "".join(p + "\0" for p in stale).encode("utf-8", errors="surrogateescape")})
    fin = transport.sh(f"cd {q} && tar -xf - && xargs -0 -r rm -f -- < .research-sync-prune; rc=$?; rm -f .research-sync-prune; exit $rc", stdin=payload)
    if fin.returncode != 0:
        raise remote.Refused(f"writing the stamp / pruning failed: {fin.stderr.decode(errors='replace').strip()[-300:]}")
    rep["seconds"] = round(time.monotonic() - t0, 2)
    return rep


def render(rep: dict[str, Any]) -> str:
    s = rep["stamp"]
    what = f"{(s['commit'] or '?')[:12]}{'+dirty' if s['dirty'] else ''}" + (f" ({s['branch']})" if s.get("branch") else "")
    verb = "would ship" if rep["dry_run"] else "synced"
    pruned = f"; {'would prune' if rep['dry_run'] else 'pruned'} {len(rep['pruned'])} stale file(s)" if rep["pruned"] else ""
    return f"{verb} {rep['files']} files of {rep['src']} @ {what} -> {rep['dest']} (via {rep['via']}){pruned} in {rep['seconds']:.1f}s"


# ---- CLI --------------------------------------------------------------------------------------------------------------------------------------

def main_ssh(argv: list[str]) -> int:
    cmd: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, cmd = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(prog="research pods ssh", description="ssh to a pod by machine name, pod id or pod name; `-- CMD` runs CMD there")
    ap.add_argument("target", metavar="TARGET")
    ap.add_argument("--print", action="store_true", help="print the ssh command line instead of running it")
    ns = ap.parse_args(argv)
    try:
        base = ssh_base(ns.target)
    except remote.Refused as e:
        print(f"research pods ssh: {e}", file=sys.stderr)
        return 1
    if ns.print:
        print(shlex.join(base + cmd))
        return 0
    os.execvp(base[0], base + cmd)
    return 0  # pragma: no cover  (execvp does not return)


def main_sync(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="research pods sync", description=__doc__.split("\n\n", 2)[2], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", metavar="TARGET")
    ap.add_argument("src", nargs="?", default=None, metavar="SRC", help="a directory inside the worktree to ship (default: cwd); its git top level is what ships")
    ap.add_argument("--dest", default=DEFAULT_DEST, help=f"absolute directory on the machine (default {DEFAULT_DEST})")
    ap.add_argument("--exclude", action="append", default=[], metavar="GLOB", help="skip matching paths / directory prefixes (repeatable)")
    ap.add_argument("--dry-run", action="store_true", help="report what would ship and be pruned; change nothing")
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    try:
        src = git_toplevel(Path(ns.src or os.getcwd()).expanduser())
        transport = SshTransport(ssh_base(ns.target))
        rep = sync(src, ns.dest, transport, excludes=ns.exclude, dry_run=ns.dry_run, dest_label=f"{ns.target}:{ns.dest}")
    except (remote.Refused, subprocess.TimeoutExpired) as e:
        print(f"research pods sync: {e}", file=sys.stderr)
        return 1
    print(json.dumps(rep, indent=2, sort_keys=True) if ns.json else render(rep))
    return 0
