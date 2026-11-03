"""PORTCHK step 3, on a node: each PR's merged-and-converted tree, one after another, built on one private Lean tree.

    python3 $RESEARCH_RUN_DIR/inputs/node_job.py            (cwd: a `--cwd clone` of the port's head)

The inputs tarball holds `manifest.json` ({"port": SHA, "tree": NAME, "prs": [{"pr": N, "patch": "N.patch", "files": [...]}]})
and one patch per PR (`git diff --binary PORT TREE`). For each PR the clone goes back to the port's head, takes the patch
with `git apply --index`, and `tools/verity/lean/lean_changed.py --build FILE... --tree TREE` builds the PR's Lean files'
modules, batched per package, in a `build` Lean slot, on `lean-records/NAME` (this run's own tree, so no lane's warm slot
tree is resynced to the ported tree). Results go to $OUT/results.jsonl, each PR's Lake output to $OUT/logs/N.log.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

RUN = Path(os.environ["RESEARCH_RUN_DIR"])
OUT = RUN / "out"
LINE = "lean-build: "


def git(*a, check=True):
    p = subprocess.run(["git", *a], capture_output=True, text=True)
    if check and p.returncode:
        raise RuntimeError(f"git {' '.join(a)}: {p.stderr[-1500:]}")
    return p


def build(files: list[str], tree: Path, log: Path) -> tuple[int, dict]:
    doc = {}
    with open(log, "a") as fh:
        p = subprocess.Popen(["python3", "tools/verity/lean/lean_changed.py", "--build", *files, "--tree", str(tree)],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
        for line in p.stdout:
            fh.write(line)
            if line.startswith(LINE):
                try:
                    doc = json.loads(line[len(LINE):])
                except ValueError:
                    pass
        return p.wait(), doc


#: the packages that fetch Mathlib and ArkLib; where the node keeps no `lean-deps/<key>` for them, `--build` skips them
DEPS_PKGS = ["Security", "Security/Properties", "Security/Proofs"]


def deps_key(pkg_dir: Path) -> str | None:
    """`lean_audit.deps_key`: the manifest's fetched entries and the toolchain."""
    m, t = pkg_dir / "lake-manifest.json", pkg_dir / "lean-toolchain"
    if not (m.is_file() and t.is_file()):
        return None
    doc = json.loads(m.read_text())
    fetched = [d for d in doc.get("packages", []) if d.get("type") != "path"]
    canon = json.dumps({"packagesDir": doc.get("packagesDir"), "packages": fetched}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode() + b"\0" + t.read_bytes()).hexdigest()


def seed(tree: Path) -> dict:
    """Each of `DEPS_PKGS`' `.lake/packages` copied into ``tree`` from a `build-I` slot tree of this node whose manifest
    and toolchain give the same key (the lean-proofs skill's second-tree copy), where ``tree`` has none yet; {pkg: how}."""
    done = {}
    for pkg in DEPS_PKGS:
        dst = tree / pkg / ".lake" / "packages"
        if dst.is_dir():
            done[pkg] = "kept"
            continue
        want = deps_key(Path(pkg))
        for src_tree in sorted(tree.parent.glob("build-*")):
            src = src_tree / pkg / ".lake" / "packages"
            if src.is_dir() and deps_key(src_tree / pkg) == want and (src / "mathlib").is_dir():
                dst.parent.mkdir(parents=True, exist_ok=True)
                t0 = time.time()
                p = subprocess.run(["cp", "-a", "--reflink=auto", str(src), str(dst.parent) + "/"], capture_output=True, text=True)
                if p.returncode:
                    shutil.rmtree(dst, ignore_errors=True)
                    done[pkg] = f"copy from {src_tree.name} failed: {p.stderr[-300:]}"
                    continue
                done[pkg] = f"copied from {src_tree.name} in {time.time() - t0:.0f} s"
                break
        else:
            done.setdefault(pkg, "no slot tree here holds them")
    return done


def main() -> int:
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="portchk-"))
    with tarfile.open(RUN / "inputs" / "portchk.tgz") as t:
        t.extractall(work)
    man = json.loads((work / "manifest.json").read_text())
    tree = Path(os.environ.get("CHECK_CACHE") or Path.home() / ".cache" / "verity-check") / "lean-records" / man["tree"]
    base = git("rev-parse", "HEAD").stdout.strip()
    print(f"portchk: {len(man['prs'])} PRs on {tree}, each from {base}", flush=True)
    tree.mkdir(parents=True, exist_ok=True)
    seeded = seed(tree)
    print(f"portchk: dependencies: {json.dumps(seeded)}", flush=True)
    (OUT / "seeded.json").write_text(json.dumps(seeded, indent=1))
    for item in man["prs"]:
        n, t0 = item["pr"], time.time()
        log = OUT / "logs" / f"{n}.log"
        rec = {"pr": n, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        try:
            git("checkout", "-q", "-f", "--detach", base)
            git("clean", "-q", "-fd")
            empty = (work / item["patch"]).stat().st_size == 0
            rec["empty_patch"] = empty
            a = None if empty else git("apply", "--index", "--binary", str(work / item["patch"]), check=False)
            if a is not None and a.returncode:
                rec.update(rc=None, error=f"patch: {a.stderr[-1500:]}")
            else:
                files = [f for f in item["files"] if Path(f).is_file()]
                rc, doc = build(files, tree, log)
                if rc and any("import clash" in str(v) for v in doc.get("skipped", {}).values()):
                    rec["clash_retry"] = True
                    rc, doc = build(files, tree, log)
                rec.update(rc=rc, doc=doc, files=len(files))
        except Exception as e:  # recorded; the next PR goes on
            rec.update(rc=None, error=str(e)[-1500:])
        rec["seconds"] = round(time.time() - t0, 1)
        with open(OUT / "results.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"portchk: #{n}: rc {rec.get('rc')} in {rec['seconds']} s "
              f"failed {len((rec.get('doc') or {}).get('failed', {}))} {rec.get('error', '')[:200]}", flush=True)
    git("checkout", "-q", "-f", "--detach", base, check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
