#!/usr/bin/env bash
# One hash-loop iteration end to end: served_commit (job.sh) on the three fixed GPUs from the pushed HEAD, through `research run`,
# then every run and its result labeled (label.py). Run it from a clean checkout of the loop branch:
#
#   bash benchmarks/hash/loop/iterate.sh N "one-line hypothesis"            # all three GPUs
#   MACHINES="vy-dc-b200" bash benchmarks/hash/loop/iterate.sh N "..."      # a subset (a rerun after a preemption)
#
# ARMS, ARM_NAMES (bench arm=loop arm, for an arm collect.py doesn't name) and REPS pass through to job.sh. The launches return
# at once; this script then waits on each run (`--wait`). If this VM stops waiting, the runs go on: label them later with
# `python3 benchmarks/hash/loop/label.py RUN... --row N --hypothesis "..."`.
set -uo pipefail
SELF=$0
# one function, parsed whole before it runs: editing this file while an iteration waits cannot change what it does
main() {
[ $# -ge 2 ] || { sed -n '2,11p' "$SELF"; exit 2; }
N=$1 HYP=$2
cd "$(git rev-parse --show-toplevel)"
BRANCH=$(git rev-parse --abbrev-ref HEAD) HEAD_SHA=$(git rev-parse HEAD)
[ -z "$(git status --porcelain)" ] || { echo "iterate: the tree is dirty; commit (and push) the iteration first" >&2; exit 2; }
git fetch -q origin "$BRANCH" && [ "$(git rev-parse FETCH_HEAD)" = "$HEAD_SHA" ] \
  || { echo "iterate: HEAD $HEAD_SHA is not origin/$BRANCH's tip; push it first" >&2; exit 2; }

declare -A ENVS=(
  [vy-dc-h200]="WORK=/workspace/cp VERITY_BENCH_VENV=/workspace/cp/venv312 NATIVE_COLLECT_BUILD=/workspace/cp/nc_build HF_HOME=/workspace/hf"
  [vy-dc-b200]="WORK=/workspace/cp VERITY_BENCH_VENV=/workspace/cp/venv312 NATIVE_COLLECT_BUILD=/workspace/cp/nc_build HF_HOME=/workspace/hf"
  [vy-nebius-2]="WORK=/workspace VERITY_BENCH_VENV=/workspace/jobs/venv312 NATIVE_COLLECT_BUILD=/workspace/cp/nc_build HF_HOME=/workspace/hf"
)
# the dc VMs are ours alone: job.sh takes the GPU with gpu-lease; node 2 is shared, so the queue places the job, a preemptible
# guest, inside its own lease (memory there is reserved by the check slots, which refuse a plain --on --mem-gb)
declare -A LAUNCH=(
  [vy-dc-h200]="--mem-gb 64"
  [vy-dc-b200]="--mem-gb 64"
  [vy-nebius-2]="--queue --gpus 1 --gpu-model rtx-pro-6000 --cpus 16 --mem-gb 64 --max-min 75 --preemptible"
)
QUESTION="hash-loop: what each committer change does to served commit overhead (TPOT, throughput) per GPU and hash"
LOGS=$(mktemp -d)
for m in ${MACHINES:-vy-dc-h200 vy-dc-b200 vy-nebius-2}; do
  envs=()
  for kv in ${ENVS[$m]}; do envs+=(--env "$kv"); done
  [ -n "${ARMS:-}" ] && envs+=(--env "ARMS=$ARMS")
  [ -n "${REPS:-}" ] && envs+=(--env "REPS=$REPS")
  [ -n "${ARM_NAMES:-}" ] && envs+=(--env "ARM_NAMES=$ARM_NAMES")
  [ -n "${CHECK_TESTS+x}" ] && envs+=(--env "CHECK_TESTS=$CHECK_TESTS")
  read -r -a launch <<< "${LAUNCH[$m]}"
  [[ " ${launch[*]} " == *" --queue "* ]] && launch+=(--question "$QUESTION")
  research run --on "$m" --project verity --campaign hash-loop --source . --cwd source --timeout 7200 --wait "${launch[@]}" \
    "${envs[@]}" --env "ROW=$N" --env "HYP=$HYP" -- bash benchmarks/hash/loop/job.sh > "$LOGS/$m.log" 2>&1 &
done
wait
runs=()
for f in "$LOGS"/*.log; do
  r=$(grep -o 'run r2026[0-9]*-[0-9]*-[0-9a-f]*' "$f" | head -1 | cut -d' ' -f2)
  tail -1 "$f"
  [ -n "$r" ] && runs+=("$r")
done
python3 benchmarks/hash/loop/label.py "${runs[@]}" --row "$N" --hypothesis "$HYP" --branch "$BRANCH"
echo "iteration $N at $BRANCH@$HEAD_SHA: ${runs[*]}"
}
main "$@"
exit
