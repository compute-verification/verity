#!/usr/bin/env bash
# run.py's diagnostic timing (not panel rows) as a node-2 fill job's chunks, one (shape, form set) a chunk:
#   forms_fill.sh CHUNKS OUT SHIP [CUBIN_SHA256]
# CHUNKS holds a chunk a line, `name|SHAPES|VARIANTS|REPS|PEARLC_A_FORMS|PEARLC_FORMING|PEARLC_EPILOGUE|budget s|max passes`
# (blank and `#` lines skipped); SHIP is build.sh's output, its MANIFEST.sha256 checked (and its cubin's hash when given).
# An invocation runs the first chunk without OUT/<name>.done: run.py's check against SHIP's fixture, then its timing, under
# the lease's GPU (PEARLC_EXPECT_UUID = GPU_LEASE_UUID), in fresh processes until `budget` seconds have passed or `max passes`
# have run (at least one), a "pass" line after each; then OUT/<name>.jsonl and OUT/<name>.done.  Exits 99 while any chunk
# has no .done and 0 once every chunk has one.  A failed pass exits 4 at once, the chunk's name and its error's tail in the
# log and its rows and stderr kept as OUT/<name>.try<N>.*; the fill runner's restart runs that chunk again, and a chunk's sixth
# start exits 1.  Nothing else exits 4.
set -uo pipefail
[ $# -ge 3 ] || { echo "usage: forms_fill.sh CHUNKS OUT SHIP [CUBIN_SHA256]"; exit 2; }
chunks=$1 O=$2 S=$3 want=${4:-}
mkdir -p "$O"
(cd "$S" && sha256sum -c --quiet MANIFEST.sha256) || { echo "the ship's manifest does not check: $S"; exit 1; }
if [ -n "$want" ] && [ "$(sha256sum < "$S/pearl_c_sm120.cubin" | cut -c1-64)" != "$want" ]; then
  echo "$S/pearl_c_sm120.cubin is not $want"; exit 1
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=8
mapfile -t rows < <(grep -v '^[[:space:]]*\(#\|$\)' "$chunks")
[ "${#rows[@]}" -gt 0 ] || { echo "no chunks in $chunks"; exit 1; }

next=""
for c in "${rows[@]}"; do [ -e "$O/${c%%|*}.done" ] || { next=$c; break; }; done
if [ -z "$next" ]; then echo "all ${#rows[@]} chunks done in $O"; exit 0; fi

IFS='|' read -r name shapes variants reps forms forming epi budget maxp <<< "$next"
st=$(( $(cat "$O/$name.starts" 2>/dev/null || echo 0) + 1 )); echo "$st" > "$O/$name.starts"
if [ "$st" -gt 5 ]; then echo "chunk $name started 5 times without finishing"; exit 1; fi
echo "chunk $name try=$st start=$(date -u +%FT%TZ) uuid=${GPU_LEASE_UUID:-?} cvd=${CUDA_VISIBLE_DEVICES:-?} job=${FILL_JOB:-?}" | tee -a "$O/chunks.txt"
part=$O/$name.part.jsonl
: > "$part"; : > "$O/$name.err"
t0=$SECONDS; p=0; rc=0
while [ "$p" -lt "$maxp" ] && { [ "$p" = 0 ] || [ $((SECONDS - t0)) -lt "$budget" ]; }; do
  p=$((p + 1)); ts=$(date -u +%FT%TZ)
  PEARLC_EXPECT_UUID=${GPU_LEASE_UUID:?} PEARLC_A_FORMS=$forms PEARLC_FORMING=$forming PEARLC_EPILOGUE=$epi VARIANTS="$variants" \
    SHAPES="$shapes" REPS=$reps python3 "$S/run.py" "$S/check/check.json" >> "$part" 2>> "$O/$name.err"; rc=$?
  printf '{"kind": "pass", "chunk": "%s", "pass": %d, "rc": %d, "start": "%s", "end": "%s", "gpu_lease_uuid": "%s", "cuda_visible_devices": "%s", "forms": "%s", "forming": "%s", "epilogue": "%s"}\n' \
    "$name" "$p" "$rc" "$ts" "$(date -u +%FT%TZ)" "${GPU_LEASE_UUID:-}" "${CUDA_VISIBLE_DEVICES:-}" "$forms" "$forming" "$epi" >> "$part"
  [ "$rc" = 0 ] || break
done
if [ "$rc" != 0 ]; then
  mv "$part" "$O/$name.try$st.jsonl"; mv "$O/$name.err" "$O/$name.try$st.err"
  echo "chunk $name failed: pass $p exited $rc at $(date -u +%FT%TZ); rows and stderr in $O/$name.try$st.*; stderr ends:" | tee -a "$O/chunks.txt"
  tail -5 "$O/$name.try$st.err"
  exit 4
fi
mv "$part" "$O/$name.jsonl"
touch "$O/$name.done"
echo "chunk $name passes=$p rc=0 end=$(date -u +%FT%TZ)" | tee -a "$O/chunks.txt"
for c in "${rows[@]}"; do [ -e "$O/${c%%|*}.done" ] || exit 99; done
echo "all ${#rows[@]} chunks done in $O"
exit 0
