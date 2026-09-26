#!/usr/bin/env bash
# The input bundle of the recorded agreement job (ci-pod.sh): the upstream sources and every stored input, so the pod needs
# no credentials.
#
#   bash ci-bundle.sh FLOCK_TREE OUT.tar.gz
#
# FLOCK_TREE is upstream flock at b684b12 with PR #83's flock-link and flock-gpu-link patches applied and crates/flock-live
# in its workspace members (its target/ and live crate are left out). Run from a checkout of this branch whose object store
# has PR #83's commits. The store inputs are fetched with `research data fetch` (store credentials on this machine only).
set -euo pipefail
FLOCK=$(realpath "$1")
OUT=$(realpath -m "$2")
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/flock" "$T/live" "$T/store"
(cd "$FLOCK" && tar --exclude=./target --exclude=./crates/flock-live -cf - .) | tar -xf - -C "$T/flock"
# PR #83 commit : vectors patch : the commit of this branch whose patch fits it
for spec in 9294e161:netlist-vectors.patch:HEAD 19c7269a:circuit-vectors.patch:9605a77d fd02e847:circuit-vectors.patch:HEAD; do
  IFS=: read -r c p at <<<"$spec"
  git -C "$REPO" worktree add -q --detach "$T/wt-$c" "$c"
  git -C "$REPO" show "$at:backends/flock/verifier/$p" > "$T/$c.patch"
  git -C "$T/wt-$c" apply "$T/$c.patch"
  cp -r "$T/wt-$c/backends/flock/live" "$T/live/$c"
  git -C "$REPO" worktree remove --force "$T/wt-$c"
done
python3 - "$HERE/vectors.json" "$T/store" <<'EOF'
import json, subprocess, sys
from pathlib import Path
v, store = json.load(open(sys.argv[1])), Path(sys.argv[2])
arts = set()
for s in v["replayable"]["sets"]:
    for k, x in s["statement"].items():
        if isinstance(x, str) and x.startswith("art:"):
            arts.add(x.split("/", 1)[0])
    arts |= {x["dir"].split("/", 1)[0] for x in s["sessions"]}
    if "forgeries" in s:
        arts.add(s["forgeries"].split("/", 1)[0])
for a in sorted(arts):
    subprocess.run([sys.executable, "-m", "research", "data", "fetch", a, "--to", str(store / a.split(":", 1)[1])], check=True,
                   capture_output=True)
    print("fetched", a)
EOF
tar -czf "$OUT" -C "$T" flock live store
ls -la "$OUT"
