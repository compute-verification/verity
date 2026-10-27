#!/usr/bin/env bash
# A shim: attn_bf16's build.sh into OUT (CUBIN_ONLY and CUDA pass through), with its sm_90a cubin also as
# attn_ref.sm_90a.cubin, the name e2e_h200's percall_job.sh and commit_slots' overlap.sh set $ATTN_REF_CUBIN to.
# Usage: bash benchmarks/torch_gpu/attn_ref/build.sh [OUT]   (default /tmp/attn-ref-ship)
set -euo pipefail
OUT=${1:-/tmp/attn-ref-ship}
bash "$(cd "$(dirname "$0")" && pwd)/../attn_bf16/build.sh" "$OUT"
cp "$OUT/attn_bf16.sm_90a.cubin" "$OUT/attn_ref.sm_90a.cubin"
