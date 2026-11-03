"""PORTCHK: seed private Lean trees' Security packages' `.lake/packages` from an idle `build-I` slot tree of this node.

    python3 seed_node.py REPO TREE...      (REPO: a checkout at the port's head, for the manifests)

Copies into `packages.portchk-tmp`, checks every fetched package's checkout is its manifest's rev, then renames to
`packages`, so a running `lean_changed.py --build` sees either none or a whole copy."""
from __future__ import annotations

import fcntl
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

PKGS = ["Security", "Security/Properties", "Security/Proofs"]
RECORDS = Path.home() / ".cache" / "verity-check" / "lean-records"


def deps_key(pkg_dir: Path) -> str | None:
    m, t = pkg_dir / "lake-manifest.json", pkg_dir / "lean-toolchain"
    if not (m.is_file() and t.is_file()):
        return None
    doc = json.loads(m.read_text())
    fetched = [d for d in doc.get("packages", []) if d.get("type") != "path"]
    canon = json.dumps({"packagesDir": doc.get("packagesDir"), "packages": fetched}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode() + b"\0" + t.read_bytes()).hexdigest()


def revs_ok(pkg_dir: Path, packages: Path) -> list[str]:
    doc = json.loads((pkg_dir / "lake-manifest.json").read_text())
    bad = []
    for d in doc.get("packages", []):
        if d.get("type") == "path":
            continue
        at = packages / d["name"].strip("«»")
        p = subprocess.run(["git", "-C", str(at), "rev-parse", "HEAD"], capture_output=True, text=True)
        if p.returncode or p.stdout.strip() != d.get("rev"):
            bad.append(f"{d['name']}: {p.stdout.strip() or p.stderr.strip()[:80]} != {d.get('rev')}")
    return bad


def idle(tree: Path) -> bool:
    try:
        with open(tree / ".lock", "a") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fh, fcntl.LOCK_UN)
        return True
    except BlockingIOError:
        return False


def main(argv: list[str]) -> int:
    repo, trees = Path(argv[0]), [Path(t) for t in argv[1:]]
    srcs = sorted(RECORDS.glob("build-*"), key=lambda s: (not idle(s), s.name))
    for pkg in PKGS:
        want = deps_key(repo / pkg)
        src = next((s / pkg / ".lake" / "packages" for s in srcs
                    if deps_key(s / pkg) == want and (s / pkg / ".lake" / "packages" / "mathlib").is_dir()), None)
        if src is None:
            print(f"{pkg}: no slot tree with key {want[:12]}", flush=True)
            continue
        if bad := revs_ok(repo / pkg, src):
            print(f"{pkg}: {src} has checkouts off its manifest: {bad}", flush=True)
            continue
        for tree in trees:
            dst = tree / pkg / ".lake" / "packages"
            if dst.is_dir():
                print(f"{tree.name} {pkg}: already has packages", flush=True)
                continue
            tmp = dst.with_name("packages.portchk-tmp")
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.parent.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            p = subprocess.run(["cp", "-a", str(src), str(tmp)], capture_output=True, text=True)
            if p.returncode or (bad := revs_ok(repo / pkg, tmp)):
                print(f"{tree.name} {pkg}: copy from {src} failed: {p.stderr[-300:]} {bad if not p.returncode else ''}", flush=True)
                shutil.rmtree(tmp, ignore_errors=True)
                continue
            tmp.rename(dst)
            print(f"{tree.name} {pkg}: copied from {src.parents[2].name} ({want[:12]}) in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
