#!/usr/bin/env bash
# overlap.py on one leased H200 (cwd = the source tree's root): the serve-time commit beside ALLP-H200's served step at the
# frozen shape, alone, in today's loop, and overlapped at both priorities. Circuits' shared-row Layout JSON (CIR-113,
# art:f6eb8ebd...) is sent; today's FAMILIES layout is e2e_h200's own.
#
#   research data fetch art:f6eb8ebda6ebef5a2cab7bfcfd37e8266660d8241e46a5037fe615a2040b695f --to DIR
#   research run --on vy-dc-h200 --project verity --campaign e2e-demo --source . --cwd source --timeout 60m \
#     --env GPU_LEASE_WHO=LANE --send DIR/families-shared.layout.json -- bash benchmarks/torch_gpu/commit_slots/overlap.sh \
#     [overlap.py's options]
#
# Before the lease: uv sync, attn_ref's cubin, the SHA-512 engine's build. Then one GPU lease of at most LEASE_MIN (30)
# minutes for overlap.py, its JSON and summary into $OUT. Exit: overlap.py's (0 every run ran and every overlapped root is
# today's loop's), 3 the setup failed, else the lease's.
set -uo pipefail
D=${RESEARCH_RUN_DIR:?run under research run}; OUT=$D/out; IN=$D/inputs; mkdir -p "$OUT"
fail() { tail -20 "$2"; echo "$1 3" | tee "$OUT/status.txt"; exit 3; }
uv sync -q --all-packages --extra torch > "$OUT/sync.txt" 2>&1 || fail sync "$OUT/sync.txt"
for c in /usr/local/cuda-13.0 /usr/local/cuda; do [ -x "$c/bin/nvcc" ] && export CUDA=$c && break; done
export PATH="${CUDA:-/usr/local/cuda}/bin:$PATH"
CUBIN_ONLY=1 bash benchmarks/torch_gpu/attn_ref/build.sh "$OUT/attn_ship" > "$OUT/attn_build.txt" 2>&1 || fail attn "$OUT/attn_build.txt"
export ATTN_REF_CUBIN="$OUT/attn_ship/attn_ref.sm_90a.cubin"
uv run --no-sync python -c "import sys; sys.path.insert(0, 'benchmarks/torch_gpu/commit_slots'); import slots_gpu; print(slots_gpu.build('9.0'))" \
  > "$OUT/engine_build.txt" 2>&1 || fail engine "$OUT/engine_build.txt"
git rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || echo "${RESEARCH_SOURCE_SHA:-unknown}" > "$OUT/commit.txt"
SHARED=$IN/families-shared.layout.json
[ -f "$SHARED" ] && sha256sum "$SHARED" > "$OUT/inputs.sha256"
gpu-lease status > "$OUT/leases-before.txt" 2>&1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OUT SHARED
echo "overlap.sh: asking for one GPU at $(date -u +%H:%M:%SZ)"
# shellcheck disable=SC2016
gpu-lease 1 --wait --mem-gb 0 --max-min "${LEASE_MIN:-30}" -- bash -c '
  echo "overlap.sh: leased $GPU_LEASE_UUID at $(date -u +%H:%M:%SZ)"
  q=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,power.limit,memory.used,memory.total
  nvidia-smi --query-gpu=$q --format=csv -i "$GPU_LEASE_UUID" > "$OUT/gpu.txt" 2>&1
  uv run --no-sync python benchmarks/torch_gpu/commit_slots/overlap.py --out "$OUT" --shared "$SHARED" "$@"; rc=$?
  nvidia-smi --query-gpu=$q --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$OUT/gpu.txt" 2>&1
  exit $rc' _ "$@"
rc=$?
echo "overlap.sh: lease ended ($rc) at $(date -u +%H:%M:%SZ)"
echo "overlap $rc" | tee "$OUT/status.txt"
exit $rc
