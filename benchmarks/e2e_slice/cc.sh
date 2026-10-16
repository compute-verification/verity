#!/usr/bin/env bash
# circuit-check on the slice's Definitions (verity/core/protocols/pouw/circuit/segment.py), from a clone of the tree's root
# (`research run --cwd clone`): the workspace synced by uv into a scratch venv, then `circuit-check TARGETS` with the pins
# measured now (--update-pins); the report, the pins file and the log under $OUT.
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-13 --source . --cwd clone -- bash benchmarks/e2e_slice/cc.sh
# CC_TARGETS (the seven pieces, PouwSegCheck and PouwSegCheckDither at KS = 4 and 128), CC_ARGS (extra circuit-check flags),
# CC_JOBS (8).
set -uo pipefail
O=${OUT:?}
git rev-parse HEAD > "$O/commit.txt" 2>&1
S=$(mktemp -d /tmp/e2e-cc.XXXXXX)
trap 'rm -rf "$S"' EXIT
UV=$(command -v uv || ls "$HOME"/.local/bin/uv "$HOME"/.cargo/bin/uv 2>/dev/null | head -1)
[ -n "$UV" ] || { echo "no uv on $(hostname)"; exit 3; }
{ nproc; grep MemTotal /proc/meminfo; "$UV" --version; } > "$O/host.txt" 2>&1
UV_PROJECT_ENVIRONMENT="$S/venv" "$UV" sync --all-packages --extra torch-cpu --frozen > "$O/sync.log" 2>&1 \
  || { tail -30 "$O/sync.log"; exit 2; }
T=${CC_TARGETS:-PouwI8Add_v2 PouwI8Pm1_v2 PouwI8AddBit_v2 PouwSegDot_v2 PouwDuoSeg_v2 PouwSel32_v2 PouwSegOk_v2 PouwSegCheck_v1\{KS=4\}
  PouwSegCheck_v1\{KS=128\} PouwSegCheckDither_v1\{KS=4\} PouwSegCheckDither_v1\{KS=128\}}
# shellcheck disable=SC2086
"$S/venv/bin/circuit-check" $T -j "${CC_JOBS:-8}" --update-pins --out "$O/report.json" ${CC_ARGS:-} > "$O/cc.log" 2>&1
echo "circuit-check rc $?" | tee "$O/rc.txt"
cp verity/core/protocols/pouw/circuit/segment.circuit-check.json "$O/" 2>/dev/null
git status --short > "$O/status.txt"
git diff > "$O/pins.diff"
tail -40 "$O/cc.log"
