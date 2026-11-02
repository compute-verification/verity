#!/usr/bin/env bash
# One-stage audit, run A5, its CPU half over GLM-5.3's TP8 Commit (a5_glm53_commit.sh), run from a VM that reaches both machines
# over ssh: the Commit's handoff (its record and served windows, private bodies included: prover only) copied from the
# GPU machine to the CPU machine, every file's sha256 compared on both ends, then ONE run there, a5_tp.sh: flock-circuit
# and flock-verify built from this tree, each rank's window of record, its Lean draw, M0's --zk sessions and Lean verify --zk.
#   bash benchmarks/one_stage/a5_glm53_cpu.sh GPU_RUN [GPU_MACHINE]      (GPU_MACHINE: the Commit's)
#   A5_ON=MACHINE (the CPU machine)   A5_JOBS=2 (ranks at once, ~56 GB each)   RESEARCH=research
# Each rank's Programs of record are read in place: the row's build/ on the machine that built it, a staged copy
# on the other (A5_BUILD elsewhere). The run waits up to --mem-wait for the machine's memory and returns once the run starts.
set -euo pipefail
RUN=${1:?the GPU Commit run id}; FROM=${2:-vy-glm53-eu2}; ON=${A5_ON:-vy-glm53-cpu-2}; JOBS=${A5_JOBS:-2}; R=${RESEARCH:-research}
ROW=glm53__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager
case $ON in
  vy-glm53-cpu-2) BUILD=${A5_BUILD:-/workspace/fs/glm53-failfast/sweep-b3/$ROW/build} ;;
  vy-nebius-cpu-2) BUILD=${A5_BUILD:-/workspace/research/glm53-a5/build} ;;
  *) BUILD=${A5_BUILD:?"the Programs of record of each rank on $ON (rank<r>/)"} ;;
esac
A=/workspace/research/glm53-a5; D=$A/handoff-$RUN
SRC_SSH=$($R pods ssh "$FROM" --print); DST_SSH=$($R pods ssh "$ON" --print)
$SRC_SSH "test -s $D/commit/runs.jsonl && test -s $D/commit/versions.json && test -s $D/serving-rows/tp.json" \
  || { echo "no complete handoff at $FROM:$D (did the Commit reach its serving rows? see the GPU run's out/commit.out)"; exit 2; }
$SRC_SSH "nice -n 19 python3 -c 'import json; d = json.load(open(\"$D/serving-rows/tp.json\")); print(\"tp.json result\", d[\"result\"], d.get(\"why\"))'"
$DST_SSH "for r in 0 1 2 3 4 5 6 7; do test -d $BUILD/rank\$r || exit 1; done" || { echo "no rank<r>/ Programs of record under $ON:$BUILD"; exit 2; }
t0=$(date +%s)
$SRC_SSH "nice -n 19 ionice -c3 tar -C $D -cf - commit serving-rows" | $DST_SSH "mkdir -p $D && tar -C $D -xf -"
a=$($SRC_SSH "cd $D && nice -n 19 ionice -c3 find commit serving-rows -type f -print0 | sort -z | xargs -0 sha256sum")
b=$($DST_SSH "cd $D && find commit serving-rows -type f -print0 | sort -z | xargs -0 sha256sum")
[ "$a" = "$b" ] || { echo "SHA256 MISMATCH between $FROM and $ON under $D"; diff <(echo "$a") <(echo "$b") | head; exit 1; }
echo "handoff OK: $(echo "$a" | wc -l) files in $(( $(date +%s) - t0 )) s -> $ON:$D"
cd "$(git rev-parse --show-toplevel)"
exec $R run --on "$ON" --project verity --source . --cwd clone --mem-gb $((56 * JOBS + 16)) --mem-wait 7200 --timeout 21600 -- \
  env A5_TP="$D" A5_BUILD="$BUILD" A5_JOBS="$JOBS" A5_FLOCK_WORK=$A/flock-work bash benchmarks/one_stage/a5_tp.sh
