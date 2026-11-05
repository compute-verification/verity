"""For each commit given, list the graph's test-file family at that commit (main's lister, the commit's tree) and save
{task id: [key, n_inputs, suite]} to OUT/<sha12>.json. Run from a checkout whose venv has verity_graph."""
import json
import subprocess
import sys
from pathlib import Path

from verity_graph import graph

repo = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
for c in sys.argv[3:]:
    sha = subprocess.check_output(["git", "-C", str(repo), "rev-parse", c], text=True).strip()
    dst = out / f"{sha[:12]}.json"
    if dst.exists():
        continue
    wt = Path(f"/tmp/cp/wt-{sha[:12]}")
    subprocess.run(["git", "-C", str(repo), "worktree", "add", "-q", "--detach", str(wt), sha], check=True)
    try:
        try:
            doc = graph.listing(wt, only={"test-file"})
        except Exception as e:
            print(sha[:12], "ERROR", str(e)[:200], flush=True)
            continue
        keys = {t["id"]: [t.get("key"), len(t.get("inputs") or []), (t.get("vars") or {}).get("suite")]
                for t in doc["tasks"]}
        dst.write_text(json.dumps({"commit": sha, "envs": {k: v.get("digest") for k, v in doc["envs"].items()},
                                   "tasks": keys}))
        print(sha[:12], len(keys), flush=True)
    finally:
        subprocess.run(["git", "-C", str(repo), "worktree", "remove", "--force", str(wt)], check=False)
