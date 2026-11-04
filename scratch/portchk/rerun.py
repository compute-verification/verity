"""Relaunch PORTCHK's node job from the patches already shipped (ship/{A..D}/N.patch and their manifests' file lists).

    python3 /tmp/portchk/rerun.py NODE TREE [--timeout T] [--delete-tree] L:N ...

Same job as launch.py (node_job.py through `research run --cwd clone` from a clean checkout of the port's head), on the
research pool, with TMPDIR=/workspace/tmp; --delete-tree has the job delete lean-records/TREE when it ends."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path("/agent/repos/verity")
PORT = "27ce82e6010eefb5064f2a497b8a9c4500a9a4c2"
SRC = Path("/tmp/portchk/src-port")
SHIP = Path("/tmp/portchk/ship")
BIN = "/agent/repos/verity/.git/verity-auth/bin:/agent/repos/verity/.venv/bin"


def main(argv):
    node, tree, rest = argv[0], argv[1], argv[2:]
    timeout, delete = "14h", False
    while rest and rest[0].startswith("--"):
        if rest[0] == "--timeout":
            timeout, rest = rest[1], rest[2:]
        elif rest[0] == "--delete-tree":
            delete, rest = True, rest[1:]
        else:
            raise SystemExit(f"unknown flag {rest[0]}")
    if not SRC.exists():
        subprocess.run(["git", "worktree", "add", "-q", "--detach", str(SRC), PORT], cwd=REPO, check=True)
    out = SHIP / f"rerun-{node}-{tree}"
    out.mkdir(parents=True, exist_ok=True)
    man = {"port": PORT, "tree": tree, "delete_tree": delete, "prs": []}
    items = {}
    for letter in "ABCD":
        for p in json.loads((SHIP / letter / "manifest.json").read_text())["prs"]:
            items[(letter, p["pr"])] = p
    for a in rest:
        letter, n = a.split(":")
        p = items[(letter, int(n))]
        (out / p["patch"]).write_bytes((SHIP / letter / p["patch"]).read_bytes())
        man["prs"].append(p)
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    tgz = out / "portchk.tgz"
    with tarfile.open(tgz, "w:gz") as t:
        for f in ["manifest.json", *(p["patch"] for p in man["prs"])]:
            t.add(out / f, arcname=f)
    cmd = ["research", "run", "--on", node, "--pool", "research", "--as", "proofs", "--project", "verity",
           "--source", str(SRC), "--cwd", "clone", "--send", str(tgz), "--send", "/tmp/portchk/node_job.py",
           "--timeout", timeout, "--pool-wait", "1800", "--custody-ttl", "24h", "--mem-gb", "80",
           "--env", "LEAN_NUM_THREADS=16", "--env", "TMPDIR=/workspace/tmp", "--",
           "bash", "-c", 'python3 "$RESEARCH_RUN_DIR/inputs/node_job.py"']
    env = {**os.environ, "PATH": BIN + ":" + os.environ["PATH"]}
    p = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    sys.stdout.write(p.stdout[-3000:])
    sys.stderr.write(p.stderr[-3000:])
    return p.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
