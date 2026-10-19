#!/usr/bin/env bash
# kernels-maxtput's job.sh on each of MACHINES (default the three cards) from the pushed HEAD through `research run`, waits on each
# run, then publishes its per-arm results as hash-loop rows (row.py --publish). Run it from a clean checkout of the branch:
#
#   bash benchmarks/hash/maxtput/launch.sh [ROW]                                   # Qwen3-8B, stock + both commit arms, all three cards
#   MACHINES=vy-dc-b200 ARMS=stock,blake3 bash benchmarks/hash/maxtput/launch.sh   # a subset
#
# ARMS, MODEL, REQUESTS, CYCLES, REPS, MAXTPUT_ARGS and CHECK_TESTS pass through to job.sh; EXTRA_ENV="K=V ..." to every engine;
# TIMEOUT (s, default 7000: node 2's queue caps a run at 120 min); WORKLOAD (the `arm` label's workload, default maxtput, or
# maxtput-<model> for another model); PUBLISH=0 skips publishing. Qwen3-8B runs pinned on the H200 and the RTX PRO 6000 under the
# loop's Builds of record (PIN_ART, PIN_ART_RTX, as iterate.sh); another model runs unpinned unless PIN_ART / PIN_ART_RTX name its
# Builds. The B200 is unpinned (cc 10.0 is on the compiled Builds' refusal list).
set -uo pipefail
main() {
ROW=${1:-11}
cd "$(git rev-parse --show-toplevel)"
BRANCH=$(git rev-parse --abbrev-ref HEAD) HEAD_SHA=$(git rev-parse HEAD)
[ -z "$(git status --porcelain)" ] || { echo "launch: the tree is dirty; commit (and push) first" >&2; exit 2; }
git fetch -q origin "$BRANCH" && [ "$(git rev-parse FETCH_HEAD)" = "$HEAD_SHA" ] \
  || { echo "launch: HEAD $HEAD_SHA is not origin/$BRANCH's tip; push it first" >&2; exit 2; }
MODEL=${MODEL:-qwen3-8b}
declare -A ENVS=(
  [vy-dc-h200]="WORK=/workspace/cp VERITY_BENCH_VENV=/workspace/cp/venv312 NATIVE_COLLECT_BUILD=/workspace/cp/nc_build HF_HOME=/workspace/hf"
  [vy-dc-b200]="WORK=/workspace/cp VERITY_BENCH_VENV=/workspace/cp/venv312 NATIVE_COLLECT_BUILD=/workspace/cp/nc_build HF_HOME=/workspace/hf"
  [vy-nebius-2]="WORK=/workspace VERITY_BENCH_VENV=/workspace/jobs/venv312 NATIVE_COLLECT_BUILD=/workspace/cp/nc_build HF_HOME=/workspace/hf"
)
declare -A LAUNCH=(
  [vy-dc-h200]="--mem-gb 64"
  [vy-dc-b200]="--mem-gb 64"
  [vy-nebius-2]="--queue --gpus 1 --gpu-model rtx-pro-6000 --cpus 16 --mem-gb 64 --max-min 120 --preemptible"
)
if [ "$MODEL" = qwen3-8b ]; then
  PIN_ART=${PIN_ART-art:25a1a1e2b13beec1f5bf76334932fb89a5460cfb2f5f0c29ce3a456922f9fcb7}
  PIN_ART_RTX=${PIN_ART_RTX-art:6f6bce9b8153587587cfe0b3d30ce5b7240ccb513926893b7f98fffc38e44c74}
fi
declare -A PIN=([vy-dc-h200]=${PIN_ART:-} [vy-nebius-2]=${PIN_ART_RTX:-})
WORKLOAD=${WORKLOAD:-$([ "$MODEL" = qwen3-8b ] && echo maxtput || echo "maxtput-$MODEL")}
QUESTION="kernels-maxtput: served vLLM's maximum throughput, native compiled against the Commit's arms on the same saturated load"
LOGS=$(mktemp -d)
for m in "${!PIN[@]}"; do
  art=${PIN[$m]}
  [ -n "$art" ] || continue
  research data fetch "$art" --path out/build.json --to "$LOGS/pin-$m" > /dev/null && [ -f "$LOGS/pin-$m/out/build.json" ] \
    || { echo "launch: could not fetch $m's Build of record $art out/build.json" >&2; exit 2; }
  mv "$LOGS/pin-$m/out/build.json" "$LOGS/pin-$m/build.json"
done
for m in ${MACHINES:-vy-dc-h200 vy-dc-b200 vy-nebius-2}; do
  envs=(--env "MODEL=$MODEL")
  for kv in ${ENVS[$m]}; do envs+=(--env "$kv"); done
  for k in ARMS REQUESTS CYCLES REPS MAXTPUT_ARGS; do [ -n "${!k:-}" ] && envs+=(--env "$k=${!k}"); done
  [ -n "${CHECK_TESTS+x}" ] && envs+=(--env "CHECK_TESTS=$CHECK_TESTS")
  for kv in ${EXTRA_ENV:-}; do envs+=(--env "$kv"); done
  [ -n "${PIN[$m]:-}" ] && envs+=(--send "$LOGS/pin-$m/build.json" --env PIN_BUILD_NAME=build.json \
    --env "PIN_BUILD_ART=${PIN[$m]}:out/build.json")
  read -r -a launch <<< "${LAUNCH[$m]}"
  [[ " ${launch[*]} " == *" --queue "* ]] && launch+=(--question "$QUESTION")
  research run --on "$m" --project verity --campaign hash-loop --source . --cwd source --timeout "${TIMEOUT:-7000}" --wait "${launch[@]}" \
    "${envs[@]}" -- bash benchmarks/hash/maxtput/job.sh > "$LOGS/$m.log" 2>&1 &
done
echo "launch: logs in $LOGS"
wait
runs=()
for f in "$LOGS"/*.log; do
  r=$(grep -o 'run r2026[0-9]*-[0-9]*-[0-9a-f]*' "$f" | head -1 | cut -d' ' -f2)
  tail -1 "$f"
  [ -n "$r" ] && runs+=("$r")
done
[ "${PUBLISH:-1}" = 1 ] && [ ${#runs[@]} -gt 0 ] && python3 benchmarks/hash/maxtput/row.py --publish "${runs[@]}" --row "$ROW" \
  --workload "$WORKLOAD" --branch "$BRANCH" ${HYP:+--hypothesis "$HYP"}
echo "maxtput row $ROW at $BRANCH@$HEAD_SHA: ${runs[*]}"
}
main "$@"; exit $?
