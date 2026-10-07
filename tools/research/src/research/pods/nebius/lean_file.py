#!/usr/bin/env python3
"""`vy-lean-file`: whether one Lean file of a commit on main elaborates, with an agent's patch applied, against node 1's warm
dependencies and build, in a tree of its own inside `vy-lean-sandbox`; what an agent would otherwise SSH to node 1 for.

    sudo -n vy-lean-file SHA FILE [--patch P] --out DIR [--slot-wait S] [--max-sec N]

1. `vy-lean-sandbox checkout SHA --tree JOBS/<tag>`: a fresh tree of this job's own, never the publisher's /build/verity nor one
   another job writes. SHA must be on node 1's main; P (`git diff` against SHA, `research lean-file` sends the working tree's)
   is applied to it with `git apply` as root.
2. FILE's package is the innermost directory above it with a lakefile. Its fetched dependencies are the daily publisher's
   copy for its `lake-manifest.json` and `lean-toolchain` (`run_daily.deps_key`, under DEPS), bound read-only at the package's
   `.lake/packages`; a package that has dependencies and no copy for its key is refused, since the build has no network.
3. Inside `vy-lean-sandbox run` (a `build` slot, as user lean-build): the publisher's `.lake/build` of the package and of each
   package it requires by path is copied in when the publisher's has the same manifest and toolchain and its daily run isn't
   building (Lake then rebuilds only what the patch and SHA changed since main's build), then `lake lean FILE -- --deps` builds
   FILE's imports, and `lake env lean FILE` elaborates it.
4. DIR gets `lean.log` (everything), `elaboration.txt` (`lake env lean`'s output) and `lean-file.json` (each step's exit and
   seconds), owned by DIR's parent's owner. The tree is removed, whatever happened.

Exit status: 0 FILE elaborated without errors, 1 its imports didn't build or it has errors, 2 refused (nothing ran), and the
sandbox's own 3 (a probe got through) and 4 (no `build` slot within --slot-wait).

Stdlib only: it runs as root from /usr/local/bin, beside vy-lean-sandbox, and its in-sandbox steps run from its own source
(`python3 -I -c`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

OK, FAILED, REFUSED = 0, 1, 2
MARK = "vy-lean-file-steps/v1 "
LAKEFILES = ("lakefile.toml", "lakefile.lean")
MAX_SEC = 3600
#: what `elaboration.txt` and the summary keep of a step's output
OUTPUT_MAX = 1 << 20


class Refused(Exception):
    """Nothing runs."""


def log(msg: str) -> None:
    print(f"vy-lean-file: {msg}", file=sys.stderr, flush=True)


@dataclass
class Node:
    """Where everything is on node 1; the tests point it at a temporary directory."""
    jobs: Path = Path("/build/lean-file")
    deps: Path = Path("/workspace/lean-build/deps")
    publisher: Path = Path("/build/verity")
    publisher_unit: str = "vy-lean-cache-daily.service"
    sandbox: list[str] = field(default_factory=lambda: ["/usr/local/bin/vy-lean-sandbox"])
    python: str = "/usr/bin/python3"


# ---- inside the sandbox, as the build's user

def _step(name: str, argv: list[str], cwd: str) -> dict:
    t0 = time.monotonic()
    p = subprocess.Popen(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = bytearray()
    for line in p.stdout:
        sys.stdout.buffer.write(line)
        sys.stdout.flush()
        if len(out) < OUTPUT_MAX:
            out += line[: OUTPUT_MAX - len(out)]
    rc = p.wait()
    return {"step": name, "rc": rc, "seconds": round(time.monotonic() - t0, 2), "output": out.decode(errors="replace")}


def inside(spec: dict) -> list[dict]:
    """The warm copies, then FILE's imports and FILE itself, each a step; a failed copy leaves that package cold."""
    steps = []
    for src, dst in spec["warm"]:
        s = _step(f"warm {src}", ["cp", "-a", src, dst], "/")
        if s["rc"] != 0:
            shutil.rmtree(dst, ignore_errors=True)
        steps.append(s)
    steps.append(_step("imports", ["lake", "lean", spec["file"], "--", "--deps"], spec["package"]))
    if steps[-1]["rc"] == 0:
        steps.append(_step("elaborate", ["lake", "env", "lean", spec["file"]], spec["package"]))
    return steps


def steps_from(log_text: str) -> list[dict] | None:
    for ln in reversed(log_text.splitlines()):
        if ln.startswith(MARK):
            try:
                return json.loads(ln[len(MARK):])
            except ValueError:
                return None
    return None


# ---- outside, as root

def check_args(sha: str, file: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise Refused(f"{sha!r}: a full commit SHA on node 1's main")
    p = Path(file)
    if p.is_absolute() or ".." in p.parts or p.suffix != ".lean" or not p.parts:
        raise Refused(f"{file!r}: a .lean file's path relative to the repository")


def package(tree: Path, file: str) -> str:
    """FILE's Lake package: the innermost directory above it, below ``tree``, with a lakefile."""
    for d in Path(file).parents:
        if any((tree / d / f).is_file() for f in LAKEFILES):
            return str(d)
    raise Refused(f"{file} is in no Lake package (no {' or '.join(LAKEFILES)} above it)")


def requires(tree: Path, pkg: str) -> list[str]:
    """``pkg`` and the packages it requires by path, transitively (lakefile.toml's `[[require]] path`), relative to ``tree``."""
    out, todo = [], [pkg]
    while todo:
        p = todo.pop()
        if p in out:
            continue
        out.append(p)
        lf = tree / p / "lakefile.toml"
        if lf.is_file():
            for r in tomllib.loads(lf.read_text()).get("require", []):
                if "path" in r:
                    q = os.path.normpath(os.path.join(p, r["path"]))
                    if not q.startswith("..") and (tree / q).is_dir():
                        todo.append(q)
    return out


def deps_key(pkg: Path) -> str | None:
    """`run_daily.deps_key`: what a package's fetched dependencies are a function of; None without a manifest."""
    m, t = pkg / "lake-manifest.json", pkg / "lean-toolchain"
    if not m.is_file() or not t.is_file():
        return None
    return hashlib.sha256(m.read_bytes() + b"\0" + t.read_bytes()).hexdigest()


def needs_deps(pkg: Path) -> bool:
    m = pkg / "lake-manifest.json"
    return m.is_file() and any(p.get("type") != "path" for p in json.loads(m.read_text()).get("packages", []))


def _active(unit: str) -> bool:
    r = subprocess.run(["systemctl", "show", "-p", "ActiveState", "--value", unit], capture_output=True, text=True)
    return r.returncode != 0 or r.stdout.strip() not in ("inactive", "failed")


def warm(n: Node, tree: Path, pkgs: list[str], active=_active) -> tuple[list[list[str]], str]:
    """[[the publisher's `.lake/build`, this tree's]] for each of ``pkgs`` it has with the same manifest and toolchain, and why
    not the rest."""
    if active(n.publisher_unit):
        return [], f"{n.publisher_unit} is building {n.publisher} (or its state is unknown): every package builds cold"
    got, cold = [], []
    for p in pkgs:
        src = n.publisher / p / ".lake" / "build"
        if src.is_dir() and not src.is_symlink() and deps_key(n.publisher / p) == deps_key(tree / p):
            got.append([str(src), str(tree / p / ".lake" / "build")])
        else:
            cold.append(p)
    return got, f"cold: {', '.join(cold)} (no build in {n.publisher} with the same manifest and toolchain)" if cold else ""


def _owner(out: Path) -> tuple[int, int]:
    st = out.parent.stat()
    return st.st_uid, st.st_gid


def _write(out: Path, name: str, text: str, owner: tuple[int, int]) -> None:
    p = out / name
    p.write_text(text)
    os.chown(p, *owner)


def _run_sandbox(argv: list[str], log_path: Path) -> tuple[int, str]:
    """``argv`` with its output tee'd to ``log_path``; on SIGTERM, SIGINT or SIGHUP it is passed on and waited for, so the
    sandbox stops its unit before the tree goes."""
    p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    old = {s: signal.signal(s, lambda signum, _f: p.send_signal(signum)) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    text = bytearray()
    try:
        with open(log_path, "ab") as f:
            for line in p.stdout:
                sys.stdout.buffer.write(line)
                sys.stdout.flush()
                f.write(line)
                text += line
        rc = p.wait()
    finally:
        for s, h in old.items():
            signal.signal(s, h)
    return rc, text.decode(errors="replace")


def check(n: Node, sha: str, file: str, patch: Path | None, out: Path, slot_wait: float | None = None,
          max_sec: int = MAX_SEC, geteuid=os.geteuid, active=_active, run_sandbox=_run_sandbox) -> int:
    if geteuid() != 0:
        raise Refused("run as root (`sudo -n vy-lean-file ...`): vy-lean-sandbox starts the build as lean-build")
    check_args(sha, file)
    if patch is not None and not patch.is_file():
        raise Refused(f"--patch {patch}: not a file")
    out.mkdir(parents=True, exist_ok=True)
    owner = _owner(out)
    os.chown(out, *owner)
    lp = out / "lean.log"
    lp.touch()
    os.chown(lp, *owner)
    t0 = time.monotonic()
    n.jobs.mkdir(mode=0o755, parents=True, exist_ok=True)
    tree = n.jobs / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{os.getpid()}-{os.urandom(3).hex()}"
    summary: dict = {"schema": "lean-file/v1", "sha": sha, "file": file, "tree": str(tree)}
    try:
        rc, _ = run_sandbox([*n.sandbox, "checkout", sha, "--tree", str(tree)], lp)
        if rc != 0:
            raise Refused(f"vy-lean-sandbox checkout exited {rc} (its reason is in {lp})")
        if patch is not None and patch.stat().st_size:
            r = subprocess.run(["git", "-C", str(tree), "-c", f"safe.directory={tree}", "apply", "--whitespace=nowarn",
                                str(patch.resolve())], capture_output=True, text=True)
            if r.returncode != 0:
                raise Refused(f"the patch doesn't apply to {sha[:12]}: {r.stderr.strip()[-600:]}")
            summary["patch"] = {"sha256": hashlib.sha256(patch.read_bytes()).hexdigest(), "bytes": patch.stat().st_size}
        if not (tree / file).is_file():
            raise Refused(f"{file} isn't in {sha[:12]}" + (" with the patch" if "patch" in summary else ""))
        pkg = package(tree, file)
        key = deps_key(tree / pkg)
        deps = n.deps / key if key else None
        bind = []
        if needs_deps(tree / pkg):
            if deps is None or not deps.is_dir():
                raise Refused(f"{pkg}: no fetched dependencies for its manifest and toolchain ({key and key[:16]}) in {n.deps}; "
                              f"the build has no network to fetch them (the daily publisher makes them for main's)")
            mount = tree / pkg / ".lake" / "packages"
            mount.mkdir(mode=0o755, exist_ok=True)
            bind = ["--ro", f"{deps}:{mount}"]
        copies, why = warm(n, tree, requires(tree, pkg), active)
        summary.update(package=pkg, deps=str(deps) if bind else None, warm=[c[0] for c in copies])
        if why:
            log(why)
            summary["cold"] = why
        spec = {"package": str(tree / pkg), "file": os.path.relpath(file, pkg), "warm": copies}
        rc, text = run_sandbox([*n.sandbox, "run", "--tree", str(tree), *bind, "-E", "LAKE_ARTIFACT_CACHE=false",
                                "--max-sec", str(max_sec), *(["--slot-wait", str(slot_wait)] if slot_wait is not None else []),
                                "--by", "lean-file", "--", n.python, "-I", "-c", Path(__file__).read_text(), "_inside",
                                json.dumps(spec)], lp)
        steps = steps_from(text)
        summary.update(sandbox_rc=rc, steps=[{k: v for k, v in s.items() if k != "output"} for s in steps or []])
        if steps is None:
            log(f"vy-lean-sandbox run exited {rc} before the steps reported")
            return rc if rc in (REFUSED, 3, 4) else FAILED
        elab = next((s for s in steps if s["step"] == "elaborate"), None)
        _write(out, "elaboration.txt", elab["output"] if elab else steps[-1]["output"], owner)
        ok = elab is not None and elab["rc"] == 0
        log(f"{file}: {'elaborated' if ok else 'FAILED'} ("
            + ", ".join(f"{s['step']} {s['seconds']:.1f} s" + (f" exit {s['rc']}" if s["rc"] else "") for s in steps) + ")")
        return OK if ok else FAILED
    except Refused as e:
        summary["refused"] = str(e)
        raise
    finally:
        shutil.rmtree(tree, ignore_errors=True)
        summary["wall_seconds"] = round(time.monotonic() - t0, 2)
        _write(out, "lean-file.json", json.dumps(summary, indent=1) + "\n", owner)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="vy-lean-file", description=__doc__.split("\n\n")[0])
    ap.add_argument("sha")
    ap.add_argument("file", help="the .lean file's path relative to the repository")
    ap.add_argument("--patch", type=Path, default=None, help="a `git diff` against SHA, applied with `git apply`")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--slot-wait", type=float, default=None, metavar="S", help="give up (exit 4) without a build slot after S s")
    ap.add_argument("--max-sec", type=int, default=MAX_SEC)
    return ap


def main(argv: list[str], node: Node | None = None, **kw) -> int:
    if argv[:1] == ["_inside"]:
        steps = inside(json.loads(argv[1]))
        print(MARK + json.dumps(steps), flush=True)
        return OK
    ns = parser().parse_args(argv)
    for s in (signal.SIGTERM, signal.SIGHUP):                  # so the tree goes however the job is stopped
        signal.signal(s, lambda signum, _f: sys.exit(128 + signum))
    try:
        return check(node or Node(), ns.sha, ns.file, ns.patch, ns.out, ns.slot_wait, ns.max_sec, **kw)
    except Refused as e:
        log(f"refused: {e}")
        return REFUSED


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
