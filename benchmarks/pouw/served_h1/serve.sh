#!/usr/bin/env bash
# served_h1's window on an RTX PRO 6000 node (serve.py): noise.cu built for the node's card (arch.sh) on the CPU, then
# serve.py on one leased GPU for at most 28 minutes.  Extra arguments go to serve.py.
#   research run --on MACHINE --project verity --source <tree> --cwd source \
#     --env GPU_LEASE_WHO=LANE -- bash benchmarks/pouw/served_h1/serve.sh
# Coins off the node: a gated window's session (POUW_CHALLENGER_GATE) is refused, exit 13, since the gated served window
# of record is vllm's H1 path, integrations/vllm/verity_vllm/protocols/pouw/h1/__init__.py, not this one.
# For development and timing, a session opened before the run: POUW_CHALLENGER=
# 127.0.0.1:PORT and POUW_CHALLENGER_SESSION=ID (with POUW_CHALLENGER_SECRET_FILE for a session with a secret), its layout
# the window's (S = --steps, default --max-tokens + 2; EPOCH = serve.py's --epoch, default 1), waited for with `ss -ltn`
# (never a probe connect: the challenger serves the first connection whose hello it accepts); its salt coin opens after
# the registration root, which a gate session's challenger refuses (it opens the salt only after the session's root).
# Without one the coins are in-process; with one of the pair alone it exits 13.  The verifier's check reads the session as
# the gate publishes it (`ssh GATE fetch SESSION`), every head of the auditor's ledger it holds (`ssh GATE ledger`, one
# --ledger each), the host's pins log infra published, its own first-parent history of main and the challenger's key
# record:
#   python3 benchmarks/pouw/served_h1/coins.py check --challenger-out SESSION_DIR --ledger LEDGER --pins PINS \
#     --main MAIN --manifest THIS_RUN/out/manifest.json --key-record KEY_RECORD --period P --root R
set -uo pipefail
D=${RESEARCH_RUN_DIR:?}; OUT=$D/out; mkdir -p "$OUT"
mkdir -p /workspace/pouw
S=$(mktemp -d /workspace/pouw/tmp-served-h1.XXXXXX); trap 'rm -rf "$S"' EXIT
ROOT=$PWD
git rev-parse HEAD | tee "$OUT/tree.txt"
CUDA=${CUDA:-/usr/local/cuda-13.0}
. "$ROOT/benchmarks/pouw/served_h1/arch.sh"
"$CUDA/bin/nvcc" -O3 -std=c++17 -gencode "$GENCODE" -I benchmarks/codec_tensor -cubin \
  -o "$S/noise.cubin" benchmarks/pouw/served_h1/noise.cu > "$OUT/nvcc.txt" 2>&1 \
  || { echo "served-h1: NVCC_FAIL"; tail -20 "$OUT/nvcc.txt"; exit 11; }
# the challenger's coins are the Lean verifier's (pouw.verifier.coins): built before the lease, not inside it
command -v lake >/dev/null || export PATH="$HOME/.elan/bin:$PATH"
(cd verity/core/protocols/pouw/lean/check && lake build pouw-check-difftest) > "$OUT/lean-coins-build.log" 2>&1 \
  || { echo "served-h1: LEAN_BUILD_FAIL"; tail -20 "$OUT/lean-coins-build.log"; exit 12; }
if [ -n "${POUW_CHALLENGER:-}${POUW_CHALLENGER_SESSION:-}" ] && { [ -z "${POUW_CHALLENGER:-}" ] || [ -z "${POUW_CHALLENGER_SESSION:-}" ]; }; then
  echo "served-h1: NO_CHALLENGER: a challenger session is POUW_CHALLENGER and POUW_CHALLENGER_SESSION together"; exit 13
fi
if [ -n "${POUW_CHALLENGER:-}" ]; then
  port=${POUW_CHALLENGER##*:}
  for _ in $(seq $((${CHALLENGER_WAIT_S:-900} * 20))); do
    ss -Hltn "sport = :$port" | grep -q . && break
    sleep 0.05
  done
  ss -Hltn "sport = :$port" | grep -q . || { echo "served-h1: NO_CHALLENGER: nothing listens on $POUW_CHALLENGER"; exit 13; }
  echo "served-h1: coins from challenger session $POUW_CHALLENGER_SESSION at $POUW_CHALLENGER (development: its root isn't this window's)"
elif [ -n "${POUW_CHALLENGER_GATE:-}" ]; then
  echo "served-h1: NO_CHALLENGER: POUW_CHALLENGER_GATE is refused; the gated served window of record is integrations/vllm/verity_vllm/protocols/pouw/h1/__init__.py (#3053)"; exit 13
else
  echo "served-h1: coins=in-process (not certified)"
fi
cd integrations/vllm
PY=${PY:-/workspace/jobs/venv312/bin/python3}
export PYTHONPATH=.:$ROOT:$ROOT/tools/research/src:$ROOT/tools/config:$ROOT/benchmarks/pouw/served_h1
export HF_HOME=${HF_HOME:-/workspace/hf} FLASHINFER_CUDA_ARCH_LIST=$FLASHINFER_ARCH
echo "served-h1: asking for one GPU at $(date -u +%H:%M:%SZ)"
gpu-lease 1 --wait --max-min 28 -- timeout 1650 "$PY" "$ROOT/benchmarks/pouw/served_h1/serve.py" --out "$OUT" \
  --cubin "$S/noise.cubin" --scratch "$S" --at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --by compute-accounting "$@"
rc=$?
echo "served-h1: serve.py exited $rc at $(date -u +%H:%M:%SZ)"
if [ -f "$S/prover.tar" ]; then
  if "$PY" -m research data put --kind pouw-h1-prover/v1 --file "$S/prover.tar" --preserve > "$OUT/prover-put.txt" 2>&1 \
     || python3 -m research data put --kind pouw-h1-prover/v1 --file "$S/prover.tar" --preserve >> "$OUT/prover-put.txt" 2>&1; then
    echo "served-h1: prover data put: $(tail -1 "$OUT/prover-put.txt")"
  else
    # a job's environment holds no store key; custody uploads all of $OUT after the run (2.1 GB at 18 steps, under 4 GiB)
    mv "$S/prover.tar" "$OUT/prover.tar" && sha256sum "$OUT/prover.tar" > "$OUT/prover.sha256"
    echo "served-h1: prover data in OUT/prover.tar for custody ($(tail -1 "$OUT/prover-put.txt" | cut -c1-120))"
  fi
fi
exit "$rc"
