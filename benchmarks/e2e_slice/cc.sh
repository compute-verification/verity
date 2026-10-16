#!/usr/bin/env bash
# circuit-check on the slice's Definitions (verity/core/protocols/pouw/circuit/segment.py), from a clone of the tree's root
# (`research run --cwd clone`): the workspace synced by uv into a scratch venv, then `circuit-check TARGETS` with the pins
# measured now (--update-pins); the report, the pins file and the log under $OUT.
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-13 --source . --cwd clone -- bash benchmarks/e2e_slice/cc.sh
# CC_TARGETS (the twelve pieces, PouwSegCheck_v2 under duo at S = 1, 4, 8, 16, 32 and XOR-in at S = 1, 8, 32,
# PouwSegCheckDither_v2 at S = 8), CC_ARGS (extra circuit-check flags; the default lets S = 32's ~540K ANDs lower),
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
D="PouwI8Add_v2 PouwI8Pm1_v2 PouwI8AddBit_v2 PouwSegDot_v2 PouwSel8_v2 PouwXor8_v2 PouwSel32_v2 PouwDuoAbsorb_v2 PouwDuoFinal_v2
  PouwXinAbsorb_v2 PouwXinFinal_v2 PouwWinOk_v2"
for s in 1 4 8 16 32; do D="$D PouwSegCheck_v2{S=$s,FOLD=duo}"; done
for s in 1 8 32; do D="$D PouwSegCheck_v2{S=$s,FOLD=xin}"; done
T=${CC_TARGETS:-$D PouwSegCheckDither_v2{S=8,FOLD=duo}}
# shellcheck disable=SC2086
"$S/venv/bin/circuit-check" $T -j "${CC_JOBS:-8}" --update-pins --out "$O/report.json" ${CC_ARGS:---lower-max-ands 800000} > "$O/cc.log" 2>&1
echo "circuit-check rc $?" | tee "$O/rc.txt"
cp verity/core/protocols/pouw/circuit/segment.circuit-check.json "$O/" 2>/dev/null
git status --short > "$O/status.txt"
git diff > "$O/pins.diff"
tail -40 "$O/cc.log"
