#!/usr/bin/env bash
# The input bundle of the recorded agreement job (ci-pod.sh): the upstream sources and every stored input, so the pod needs
# no credentials.
#
#   bash ci-bundle.sh FLOCK_TREE OUT.tar.gz
#
# FLOCK_TREE is upstream flock at b684b12 with M0's flock-link and flock-gpu-link patches applied and crates/flock-live
# in its workspace members (its target/ and live crate are left out). Run from a checkout of this branch whose object store
# has M0's commits. The store inputs are fetched with `research data fetch` (store credentials on this machine only).
set -euo pipefail
FLOCK=$(realpath "$1")
OUT=$(realpath -m "$2")
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/flock" "$T/live" "$T/store"
(cd "$FLOCK" && tar --exclude=./target --exclude=./crates/flock-live -cf - .) | tar -xf - -C "$T/flock"
# M0 commit : vectors patch : the commit of this branch whose patch fits it. A version whose Flock needs its own patch
# (SHA-512 Merkle trees) ships it as live/<commit>/flock.patch with the crate's features in live/<commit>/features.
for spec in 9294e161:netlist-vectors.patch:HEAD 19c7269a:circuit-vectors.patch:9605a77d fd02e847:circuit-vectors.patch:a2c4e9fd \
    631567f7:circuit-vectors.patch:HEAD eb90718f:circuit-vectors-eb90718f.patch:HEAD e51e2b86:circuit-vectors-eb90718f.patch:HEAD \
    25519ba1:circuit-vectors-25519ba1.patch:HEAD 967b8d06:circuit-vectors-967b8d06.patch:HEAD; do
  IFS=: read -r c p at <<<"$spec"
  git -C "$REPO" worktree add -q --detach "$T/wt-$c" "$c"
  git -C "$REPO" show "$at:verity/ml/flock/verifier/$p" > "$T/$c.patch"
  git -C "$T/wt-$c" apply "$T/$c.patch"
  cp -r "$T/wt-$c/verity/ml/flock/live" "$T/live/$c"
  if [ -f "$T/wt-$c/verity/ml/flock/flock-sha512-b684b12.patch" ]; then
    cp "$T/wt-$c/verity/ml/flock/flock-sha512-b684b12.patch" "$T/live/$c/flock.patch"
    # eb90718f's and e51e2b86's selftest records pin seed_injection: true, so their replay binaries are the harness build
    case "$c" in eb90718f|e51e2b86|25519ba1|967b8d06) echo "sha512 seed-injection" > "$T/live/$c/features" ;; *) echo sha512 > "$T/live/$c/features" ;; esac
  fi
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
    arts |= {x["dir"].split("/", 1)[0] for x in s.get("sessions", [])}
    if "forgeries" in s:
        arts.add(s["forgeries"].split("/", 1)[0])
    arts |= {x.split("/", 1)[0] for x in s.get("selftest", [])}
for a in sorted(arts):
    subprocess.run([sys.executable, "-m", "research", "data", "fetch", a, "--to", str(store / a.split(":", 1)[1])], check=True,
                   capture_output=True)
    print("fetched", a)
EOF
tar -czf "$OUT" -C "$T" flock live store
ls -la "$OUT"
