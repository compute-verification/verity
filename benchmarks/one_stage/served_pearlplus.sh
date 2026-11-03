#!/usr/bin/env bash
# PoUW's piece with Pearl+, `served_verdicts.sh`'s POUW_CMD under POUW=pearlplus (H1 off): Pearl+'s own served run on the
# card after the window's serve, every pass retained and each one replayed by the reference verifier:
#   POUW_CMD='bash benchmarks/one_stage/served_pearlplus.sh "$1" "$2" "$3"'      (served_verdicts.sh runs it in the tree)
#   bash benchmarks/one_stage/served_pearlplus.sh SERVE_DIR WINDOW PIECE_DIR
# Everything but PIECE_DIR's files goes to SERVE_DIR/private/pearlplus, each step's log among it:
#   1. benchmarks/harness/setup/pearl_c_sm90.sh builds pearl_c_sm90's cubin from this tree, with its SASS gates;
#   2. under `gpu-lease 1` (PEARLPLUS_LEASE_MIN, default 30), `served_passes.py serve` serves PEARLPLUS_SPEC's Pearl-C arm
#      (default benchmarks/specs/llama32-1b-1gpu-pouw-protocol-served-h200.toml: pouw:pearl-c-h100-v1-h3 over Llama 3.2 1B's
#      BF16 checkpoint) a warm-up, then PEARLPLUS_REQUESTS requests (default 32) of PEARLPLUS_PROMPT_TOKENS seeded random
#      tokens (default 1,024) and PEARLPLUS_OUTPUT_TOKENS outputs (default 1) at PEARLPLUS_T tokens a step (default 8,192:
#      a prefill-heavy load, 8 prompts a step), each engine step one pass retained in passes/;
#   3. `served_passes.py verify` replays each pass on the CPU in PEARLPLUS_JOBS processes (default nproc), on fast_tile's
#      batch forms, a replayed tile whose id is a multiple of 256 on the scalar reference too (a scalar replay takes
#      minutes a tile at the 1B's K), or with PEARLPLUS_FAST set empty on the scalar reference alone;
#   4. `served_verdicts.py pearlplus` writes the piece (PIECE_DIR/development.json): its verdict, its shares and why it
#      credits nothing (served_verdicts.PEARLPLUS_WHY: its passes are not the window sampled proofs audited, so WINDOW
#      names nothing here).
# The passes hold the run's activations, so they stay in SERVE_DIR/private, under served_verdicts.sh's `research keep`.
# env: served_verdicts.sh's (VENV, HF_HOME, CUDA_HOME), PEARLPLUS_* above. Its lines name no window. Exit 0 once step 4
# wrote the piece.
set -uo pipefail
mkdir -p "${3:?PIECE_DIR}" && SD=$(cd "${1:?SERVE_DIR}" && pwd) && P=$(cd "$3" && pwd) || exit 1
cd "$(dirname "$0")/../.." || exit 1
TREE=$PWD TPY=$PWD/.venv/bin/python
[ -x "$TPY" ] || TPY=${PY:-python3}
Q=$SD/private/pearlplus SPEC=${PEARLPLUS_SPEC:-benchmarks/specs/llama32-1b-1gpu-pouw-protocol-served-h200.toml}
FAST=${PEARLPLUS_FAST-1}
rm -rf "$Q" && (umask 077; mkdir -p "$Q") || exit 1
say() { echo "POUW $*"; }
took() { echo "$(( ($(date +%s) - $1) / 60 )) min"; }
export VERITY_BENCH_VENV=${VERITY_BENCH_VENV:-${VENV:-/workspace/cp/venv312}}
# shellcheck source=/dev/null
. benchmarks/harness/env.sh
# Pearl+ served as vLLM serves it, not under the window serve's batch invariance
unset VLLM_BATCH_INVARIANT

t=$(date +%s)
( export RESEARCH_RUN_DIR=$Q
  # shellcheck source=/dev/null
  . benchmarks/harness/setup/pearl_c_sm90.sh && printf '%s\n' "$PEARLC_KERNEL" "$PEARLC_CUBIN" > "$Q/setup.paths" ) \
  > "$Q/setup.log" 2>&1
rc=$?
if [ "$rc" = 0 ] && [ -s "$Q/setup.paths" ]; then
  { read -r PEARLC_KERNEL; read -r PEARLC_CUBIN; } < "$Q/setup.paths"
  export PEARLC_KERNEL PEARLC_CUBIN
  say "cubin: pearl_c_sm90.sh built it and its SASS gates passed after $(took "$t") (cubin sha256 $(sha256sum "$PEARLC_CUBIN" | cut -c1-16))"
else
  say "cubin: pearl_c_sm90.sh failed (exit $rc after $(took "$t")): $(tail -1 "$Q/setup.log" | cut -c1-400)"
fi

if [ -n "${PEARLC_CUBIN:-}" ]; then
  t=$(date +%s)
  GPU_LEASE_WHO=${GPU_LEASE_WHO:-vllm-pearlplus} gpu-lease 1 --wait --max-min "${PEARLPLUS_LEASE_MIN:-30}" -- \
    "$VENV/bin/python" benchmarks/pouw/pearl_c_vllm/served_passes.py serve --spec "$SPEC" --passes "$Q/passes" \
    --out "$Q/served.json" --requests "${PEARLPLUS_REQUESTS:-32}" --prompt-tokens "${PEARLPLUS_PROMPT_TOKENS:-1024}" \
    --output-tokens "${PEARLPLUS_OUTPUT_TOKENS:-1}" --max-num-batched-tokens "${PEARLPLUS_T:-8192}" > "$Q/serve.log" 2>&1
  rc=$?
  say "serve: served_passes.py serve exit $rc after $(took "$t"): $(grep -m1 '^served ' "$Q/serve.log" | cut -c1-300)"
fi

if [ -s "$Q/served.json" ]; then
  t=$(date +%s)
  "$VENV/bin/python" benchmarks/pouw/pearl_c_vllm/served_passes.py verify --passes "$Q/passes" --out "$Q/verify.json" \
    --jobs "${PEARLPLUS_JOBS:-$(nproc)}" ${FAST:+--fast} > "$Q/verify.log" 2>&1
  rc=$?
  say "replay: served_passes.py verify exit $rc after $(took "$t"): $(tail -1 "$Q/verify.log" | cut -c1-300)"
fi

"$TPY" benchmarks/one_stage/served_verdicts.py pearlplus "$P" --served "$Q/served.json" --verify "$Q/verify.json" \
  > "$Q/pearlplus.log" 2>&1
rc=$?
grep '^POUW-DEV ' "$Q/pearlplus.log"
say "piece: served_verdicts.py pearlplus exit $rc"
exit "$rc"
