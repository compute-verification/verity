#!/usr/bin/env bash
# The red team's resume attack against the streamed-label wipe and the band fill, on CPU, as one Attempt. cwd =
# a clone (the Lean verdict builds into it):
#
#   bash benchmarks/pous/wipe_resume/run.sh [--labels N] [--k K] [--blocks B]
#
#   --labels N   the streamed fill in labels (default 2^19: one 32 MiB segment)
#   --k K        reads a session (default 2152)
#   --blocks B   blocks of the band and dense segments the state attack rebuilds at 64 KiB (default 6)
#
# Outside any lease (CPU only): uv sync, the Lean verdict streamed-wipe-verdict, then resume.py (every prover's session,
# judged by the Lean and the Python) and pi2_state_attack.py. Outputs under $OUT (a run's out/; else ./out): env.txt,
# lake-build.log, resume.json, resume.txt, pi2_state_attack.json, pi2_state_attack.txt. Scratch (the test signers'
# leaf states, the session log) is a mktemp -d, removed on exit. Exit 0 when every session ends as expected and every
# band and dense block is rebuilt.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$ROOT"
export O=${OUT:-${RESEARCH_RUN_DIR:-$PWD}/out}
mkdir -p "$O"
LABELS=$((1 << 19)) K=2152 BLOCKS=6
while [ $# -gt 0 ]; do
  case "$1" in
    --labels) LABELS=$2; shift ;;
    --k) K=$2; shift ;;
    --blocks) BLOCKS=$2; shift ;;
    *) echo "run.sh: unknown option $1 (see the header)"; exit 2 ;;
  esac
  shift
done
{ echo "commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"; date -u +%FT%TZ; hostname
  lscpu | grep -E "Model name|^CPU\(s\)"; free -g; uptime; } > "$O/env.txt" 2>&1
uv sync -q --all-packages --extra torch-cpu
L=verity/core/devices/memory_challenger/lean
export PATH="$HOME/.elan/bin:$PATH"
command -v lake >/dev/null 2>&1 || bash tools/verity/lean/setup.sh "$L" > "$O/lake-setup.log" 2>&1
(cd "$L" && lake build streamed-wipe-verdict) > "$O/lake-build.log" 2>&1 \
  || { tail -30 "$O/lake-build.log"; echo "REFUSED: streamed-wipe-verdict doesn't build"; exit 3; }
rc=0
uv run --no-sync python benchmarks/pous/wipe_resume/resume.py --out "$O" --labels "$LABELS" --k "$K" --decide both \
  2>&1 | tee "$O/resume.txt" || rc=$?
uv run --no-sync python benchmarks/pous/wipe_resume/pi2_state_attack.py --blocks "$BLOCKS" --out "$O" \
  2>&1 | tee "$O/pi2_state_attack.txt" || rc=$((rc ? rc : $?))
exit $rc
