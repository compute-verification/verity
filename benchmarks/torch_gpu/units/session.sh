#!/usr/bin/env bash
# The units' run on one H200 or RTX PRO 6000, from build.sh's ship (cwd = the source tree's root):
#
#   tar -C SHIP -cf IN/units-ship.tar .
#   OUT=DIR IN=IN bash benchmarks/torch_gpu/units/session.sh   (SWEEP_ONLY=1 for the sweeps alone)
#
# Needs CUDA torch and `gpu-lease` on PATH. IN (default a run directory's inputs/, RESEARCH_RUN_DIR) holds
# units-ship.tar and, optionally, units-old.tar.
#
# Outside the lease: the ship (units-ship.tar) unpacked into a scratch directory and held to its MANIFEST.sha256, the
# build's records and the fixture's summary copied to $OUT, the card and the Python recorded; units-old.tar, another
# build's ship, likewise.  Under one lease of at most 30 minutes: check.py (the sweeps, the check, then bench.py's timing)
# on the shipped cubin of the card's arch, its JSON lines in $OUT/check.jsonl; then, with units-old.tar, bench.py on the old
# ship's modules and cubin ($OUT/bench-old.jsonl), so both builds are timed by one ruler on one card.  Every step's exit
# code is in $OUT/rc.txt; exits non-zero if any failed.
# VERITY_BENCH_VENV names the Python with numpy and CUDA torch (default the first of /workspace/cp/venv312,
# /workspace/venv312 and /workspace/jobs/venv312); GPU_LEASE_WHO the lease's holder (default units/ and the run's id);
# LEASE_MIN the lease's bound (default 30); check.py's environment (SWEEP_ONLY, BATCHES, ...) passes through.
set -uo pipefail
O=${OUT:?}; IN=${IN:-${RESEARCH_RUN_DIR:?set IN}/inputs}
export O GPU_LEASE_WHO=${GPU_LEASE_WHO:-units/${RESEARCH_RUN_ID:-local}}
V=${VERITY_BENCH_VENV:-}
if [ -z "$V" ]; then
  for c in /workspace/cp/venv312 /workspace/venv312 /workspace/jobs/venv312; do [ -x "$c/bin/python" ] && V=$c && break; done
fi
export PY=${V:-/usr}/bin/python3
# the source tree's verity, which check.py reads only to diagnose a sweep chunk that disagrees (sweep.model)
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
S=$(mktemp -d); trap 'rm -rf "$S"' EXIT
export S
: > "$O/rc.txt"
tar -C "$S" -xf "$IN/units-ship.tar" && (cd "$S" && sha256sum -c --quiet MANIFEST.sha256)
echo "ship rc $?" >> "$O/rc.txt"
export OLD=
if [ -f "$IN/units-old.tar" ]; then
  mkdir "$S/old" && tar -C "$S/old" -xf "$IN/units-old.tar" && (cd "$S/old" && sha256sum -c --quiet MANIFEST.sha256)
  echo "old ship rc $?" >> "$O/rc.txt"
  cp "$S/old/source.txt" "$O/old-source.txt"
  OLD=$S/old
fi
grep -qv " rc 0$" "$O/rc.txt" && { cat "$O/rc.txt"; exit 2; }
cp "$S"/build-check.*.json "$S"/harness-gate.*.txt "$S/toolchain.txt" "$S/source.txt" "$S/MANIFEST.sha256" \
  "$S/check/fixture.json" "$O/"
cp "$S"/harness-gate.*.json "$O/" 2>/dev/null
git rev-parse HEAD > "$O/commit.txt" 2>&1
"$PY" -c 'import sys, numpy, torch; print(sys.executable, "numpy", numpy.__version__, "torch", torch.__version__, torch.version.cuda)' \
  > "$O/python.txt" 2>&1
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${LEASE_MIN:-30}" -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,power.limit --format=csv -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  timeout 20m "$PY" "$S/check.py" "$S/check" > "$O/check.jsonl" 2> "$O/check.err"
  echo "check rc $?" >> "$O/rc.txt"
  if [ -n "$OLD" ]; then
    timeout 8m "$PY" "$S/bench.py" "$OLD" > "$O/bench-old.jsonl" 2> "$O/bench-old.err"
    echo "bench-old rc $?" >> "$O/rc.txt"
  fi'
echo "lease rc $?" >> "$O/rc.txt"
cat "$O/rc.txt"
cat "$O/check.err" "$O/bench-old.err" 2>/dev/null
! grep -qv " rc 0$" "$O/rc.txt"
