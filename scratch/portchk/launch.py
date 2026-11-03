"""Launch PORTCHK's node job: python3 /tmp/portchk/launch.py NODE TREE [--timeout T] PR...

Packs each PR's patch (port head -> its merged-and-converted tree, refs/portchk/{merged,resolved}/N) and the files to
build into portchk.tgz, and starts node_job.py through `research run --on NODE --pool proofs --cwd clone` from a clean
checkout of the port's head. Prints the run id."""
from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path("/agent/repos/verity")
PORT = "27ce82e6010eefb5064f2a497b8a9c4500a9a4c2"
SRC = Path("/tmp/portchk/src-port")
POOL = __import__("os").environ.get("PORTCHK_POOL", "proofs")
BIN = "/agent/repos/verity/.git/verity-auth/bin:/agent/repos/verity/.venv/bin"


def git(*a, cwd=REPO):
    return subprocess.run(["git", *a], cwd=cwd, capture_output=True, check=True).stdout


def main(argv):
    node, tree, rest = argv[0], argv[1], argv[2:]
    timeout = "8h"
    if rest[:1] == ["--timeout"]:
        timeout, rest = rest[1], rest[2:]
    prs = [int(a) for a in rest]
    if not SRC.exists():
        git("worktree", "add", "-q", "--detach", str(SRC), PORT)
    ship = Path(f"/tmp/portchk/ship/{node}-{tree}")
    ship.mkdir(parents=True, exist_ok=True)
    man = {"port": PORT, "tree": tree, "prs": []}
    for n in prs:
        rec = json.loads(Path(f"/tmp/portchk/out/{n}.json").read_text())
        tree_commit = rec.get("fixed_commit") or rec["tree_commit"]
        (ship / f"{n}.patch").write_bytes(git("diff", "--binary", PORT, tree_commit))
        man["prs"].append({"pr": n, "patch": f"{n}.patch", "files": rec["own_lean"], "kind": rec["tree_kind"]})
    (ship / "manifest.json").write_text(json.dumps(man, indent=1))
    tgz = ship / "portchk.tgz"
    with tarfile.open(tgz, "w:gz") as t:
        for f in ["manifest.json", *(p["patch"] for p in man["prs"])]:
            t.add(ship / f, arcname=f)
    cmd = ["research", "run", "--on", node, "--pool", POOL, "--as", "proofs", "--project", "verity",
           "--source", str(SRC), "--cwd", "clone", "--send", str(tgz), "--send", "/tmp/portchk/node_job.py",
           "--timeout", timeout, "--pool-wait", __import__("os").environ.get("PORTCHK_POOL_WAIT", "1800"), "--custody-ttl", "12h", "--mem-gb", "80", "--env", "LEAN_NUM_THREADS=16", "--",
           "bash", "-c", 'python3 "$RESEARCH_RUN_DIR/inputs/node_job.py"']
    env = {**__import__("os").environ, "PATH": BIN + ":" + __import__("os").environ["PATH"]}
    p = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    sys.stdout.write(p.stdout[-3000:])
    sys.stderr.write(p.stderr[-3000:])
    return p.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
