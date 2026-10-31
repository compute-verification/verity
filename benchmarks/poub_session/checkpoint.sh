#!/usr/bin/env bash
# The pinned checkpoint of ROLE (default QWEN25_32B_BF16) on this node, for a bound run (cwd = the source tree): the
# files core's pin names (`verity/core/protocols/pouw/lean/check/models.json`) at its repo and revision in the HF cache
# (HF_HOME, default /workspace/hf), any missing fetched from the Hub; then $HF_HOME/pinned/ROLE-REV8, a directory of hard
# links to the cache's blobs (regular files, which `poub.store.check` takes and a snapshot's symlinks aren't:
# `verifier.py --checkpoint`), every file of it held to the pin's SHA-512 (`store.check`, WORKERS processes); the
# snapshot (`node.py --model-dir`) and the pinned directory each get a retention record (owner OWNER,
# KEEP_EXPIRES). CHECK_ONLY=1 fetches and links nothing: it holds what is there to the pin. Writes $OUT/checkpoint.json.
#
#   OUT=DIR bash benchmarks/poub_session/checkpoint.sh      # on the GPU node
set -uo pipefail
: "${OUT:?}"
ROLE=${ROLE:-QWEN25_32B_BF16}
PIN=verity/core/protocols/pouw/lean/check/models.json
export HF_HOME=${HF_HOME:-/workspace/hf}
PY=${PY:-python3}
read -r REPO REV < <("$PY" -c 'import json, sys; e = json.load(open(sys.argv[1]))[sys.argv[2]]; print(e["repo"], e["revision"])' \
  "$PIN" "$ROLE") || { echo "core's pin $PIN has no role $ROLE"; exit 3; }
SNAP=$HF_HOME/hub/models--${REPO//\//--}/snapshots/$REV
PINNED=$HF_HOME/pinned/$ROLE-${REV:0:8}
mapfile -t FILES < <("$PY" -c 'import json, sys; print("\n".join(sorted(json.load(open(sys.argv[1]))[sys.argv[2]]["files"])))' \
  "$PIN" "$ROLE")
echo "role $ROLE: $REPO@$REV, ${#FILES[@]} files; snapshot $SNAP; pinned $PINNED"
df -h "$HF_HOME" | tail -1
if [ -z "${CHECK_ONLY:-}" ]; then
  missing=()
  for f in "${FILES[@]}"; do [ -f "$SNAP/$f" ] || missing+=("$f"); done
  echo "missing from the snapshot: ${#missing[@]} of ${#FILES[@]}"
  if [ ${#missing[@]} -gt 0 ]; then
    t0=$(date +%s)
    HF_HUB_OFFLINE=0 HF_HUB_ENABLE_HF_TRANSFER=1 uv run -q --no-project --with 'huggingface_hub[hf_transfer]>=0.34' python - \
      "$REPO" "$REV" "${missing[@]}" <<'EOF' || { echo "the fetch failed"; exit 4; }
import sys
from huggingface_hub import snapshot_download
repo, rev, files = sys.argv[1], sys.argv[2], sys.argv[3:]
print(snapshot_download(repo, revision=rev, allow_patterns=files, max_workers=8))
EOF
    echo "fetched ${#missing[@]} files in $(($(date +%s) - t0)) s"
  fi
  mkdir -p "$PINNED" || exit 3
  for f in "${FILES[@]}"; do
    ln -f "$(readlink -f "$SNAP/$f")" "$PINNED/$f" || { echo "can't link $SNAP/$f into $PINNED"; exit 3; }
  done
fi
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
# a file, not stdin: store.check's workers are spawned, and each imports the main module again
cat > "$T/check.py" <<'EOF'
import json, sys
from pathlib import Path
from verity.core.protocols.poub import store


def main() -> int:
    pinned, pin, role, workers, snap = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]), Path(sys.argv[5])
    p = store.read_pin(pin, role)
    doc = {"role": role, "checkpoint_sha512": p["checkpoint_sha512"], "files": len(p["checkpoint"]),
           "pinned": str(pinned), "snapshot": str(snap),
           "snapshot_entries": sorted(x.name for x in snap.iterdir()) if snap.is_dir() else [],
           "bytes": sum((pinned / f).stat().st_size for f in p["checkpoint"] if (pinned / f).is_file())}
    try:
        store.check(pinned, p, workers)
        doc["check"] = "every file is the pin's"
    except store.Refused as e:
        doc["check"] = f"refused: {e}"
    print(json.dumps(doc, indent=1))
    return 0 if doc["check"] == "every file is the pin's" else 5


if __name__ == "__main__":
    sys.exit(main())
EOF
t0=$(date +%s)
PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH} "$PY" "$T/check.py" "$PINNED" "$PIN" "$ROLE" "${WORKERS:-8}" "$SNAP" \
  > "$OUT/checkpoint.json"
rc=$?
echo "store.check rc $rc in $(($(date +%s) - t0)) s"
cat "$OUT/checkpoint.json"
[ $rc = 0 ] || exit $rc
if [ -z "${CHECK_ONLY:-}" ]; then
  for p in "$(dirname "$(dirname "$SNAP")")" "$PINNED"; do
    research keep "$p" --owner "${OWNER:-@memory-accounting}" --expires "${KEEP_EXPIRES:-14d}" --priority high \
      --regenerate "ROLE=$ROLE bash benchmarks/poub_session/checkpoint.sh" \
      --why "$ROLE's pinned checkpoint ($REPO@$REV, core's models.json) for the remote memory audit's bound runs" \
      || { echo "research keep $p failed"; exit 6; }
  done
  research keep show "$PINNED"
fi
df -h "$HF_HOME" | tail -1
