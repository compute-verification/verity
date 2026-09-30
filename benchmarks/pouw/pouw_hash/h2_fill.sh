#!/usr/bin/env bash
# fill: owner=bc-7442ca43-9389-5653-ae57-2cbf499a9569 gpus=1 max_min=8 cpus=8 project=pous prio=10 on=@DIE@ mem_gb=16
# GPU 2 (hashing, bc-7442ca43): the -h2 hash bench on die @DIE@ (@UUID@) at the 13 panel shapes (hashing-forms-by-shape.md's),
# for the served path's per-die hashing numbers; -h1 beside it in the same process, both variants. Untimed screen: diagnostic
# CUDA-event and CUDA-graph medians at the node's clocks (locked-2100 unless the clock record says otherwise), not panel rows.
# Instantiated from benchmarks/pouw/pouw_hash/h2_fill.sh, its placeholders filled by sed.
# Ship: @SHIP@ (h2_ship.sh at @SRC@: pearl_c_sm120.cubin sha256 @CUBIN@, the in-kernel -h2 segment keys of 997ac6bd, and
# #596's decode spread). Each chunk is one pass at one shape (every shape's first pass, then every shape's second): a fresh
# process of h2_bench.py, run.py's fixture check, then per (variant, format) the bit-exact gate of every hashed buffer (sampled,
# its seed the die, try and pass) before any timing, then timing rounds for its share of BUDGET seconds; a failed gate exits 4
# with nothing timed. Output in @OUT@/<chunk>.jsonl (the bench's rows, then a "pass" line), chunks.txt (lease UUID and
# CUDA_VISIBLE_DEVICES per start).
# Restartable per chunk: exit 99 while chunks remain, 143 on SIGTERM (the chunk reruns from its start), 0 when all are done, 4
# when any chunk failed (its rows kept, the chunk marked .failed); a chunk started 5 times without finishing fails the job (1).
set -uo pipefail
S=@SHIP@
O=@OUT@
DIE=@DIE@
WANT=@UUID@
BUDGET=140 REPS=9
mkdir -p "$O"
kid= name= st=0
trap '[ -n "$kid" ] && kill -TERM "$kid" 2>/dev/null; wait; [ -n "$name" ] && echo "$((st - 1))" > "$O/$name.starts"; exit 143' TERM
[ "${GPU_LEASE_UUID:-}" = "$WANT" ] || { echo "lease UUID ${GPU_LEASE_UUID:-none} is not die $DIE's $WANT" >&2; exit 1; }
(cd "$S" && sha256sum -c --quiet MANIFEST.sha256) || { echo "the ship's manifest does not check" >&2; exit 1; }
[ "$(sha256sum < "$S/pearl_c_sm120.cubin" | cut -c1-64)" = @CUBIN@ ] || { echo "not the ship's cubin" >&2; exit 1; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=8 PYTHONPATH="$S/py" PYTHONDONTWRITEBYTECODE=1
# name|SHAPE: decode first, then prefill
SHAPES=(
"m32-n8192-k8192|32x8192x8192"
"m32-n10240-k8192|32x10240x8192"
"m32-n57344-k8192|32x57344x8192"
"m32-n8192-k28672|32x8192x28672"
"m2048-n8192-k8192|2048x8192x8192"
"m2048-n10240-k8192|2048x10240x8192"
"m2048-n57344-k8192|2048x57344x8192"
"m2048-n8192-k28672|2048x8192x28672"
"cube2048|2048"
"cube4096|4096"
"cube8192|8192"
"cube16384|16384"
"cube32768|32768"
)
CHUNKS=()
for p in 1 2; do for s in "${SHAPES[@]}"; do CHUNKS+=("${s%%|*}-p$p|${s#*|}|$p"); done; done
remaining() {
  local c n
  for c in "${CHUNKS[@]}"; do n=${c%%|*}; [ -e "$O/$n.done" ] || [ -e "$O/$n.failed" ] || return 0; done
  return 1
}
for c in "${CHUNKS[@]}"; do
  IFS='|' read -r name shape p <<< "$c"
  if [ -e "$O/$name.done" ] || [ -e "$O/$name.failed" ]; then name=; continue; fi
  st=$(( $(cat "$O/$name.starts" 2>/dev/null || echo 0) + 1 )); echo "$st" > "$O/$name.starts"
  if [ "$st" -gt 5 ]; then echo "chunk $name started 5 times without finishing" >&2; exit 1; fi
  echo "chunk $name try=$st start=$(date -u +%FT%TZ) uuid=${GPU_LEASE_UUID:-?} cvd=${CUDA_VISIBLE_DEVICES:-?} job=${FILL_JOB:-?}" | tee -a "$O/chunks.txt"
  : > "$O/$name.part.jsonl"; : > "$O/$name.err"
  ts=$(date -u +%FT%TZ)
  PEARLC_EXPECT_UUID=$WANT SHAPE=$shape BUDGET=$BUDGET REPS=$REPS SEED="die$DIE-try$st-pass$p" \
    python3 "$S/h2_bench.py" "$S/check/check.json" >> "$O/$name.part.jsonl" 2>> "$O/$name.err" &
  kid=$!; wait "$kid"; rc=$?; kid=
  printf '{"kind": "pass", "chunk": "%s", "pass": %d, "rc": %d, "start": "%s", "end": "%s", "gpu_lease_uuid": "%s", "cuda_visible_devices": "%s", "die": %d}\n' \
    "$name" "$p" "$rc" "$ts" "$(date -u +%FT%TZ)" "${GPU_LEASE_UUID:-}" "${CUDA_VISIBLE_DEVICES:-}" "$DIE" >> "$O/$name.part.jsonl"
  mv "$O/$name.part.jsonl" "$O/$name.jsonl"
  if [ "$rc" = 0 ]; then touch "$O/$name.done"; else echo "$rc" > "$O/$name.failed"; tail -3 "$O/$name.err" >&2; fi
  echo "chunk $name rc=$rc end=$(date -u +%FT%TZ)" | tee -a "$O/chunks.txt"
  name=
  break
done
if remaining; then exit 99; fi
if ls "$O"/*.failed >/dev/null 2>&1; then echo "failed chunks: $(cd "$O" && ls *.failed | tr '\n' ' ')" >&2; exit 4; fi
exit 0
