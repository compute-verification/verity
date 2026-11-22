#!/usr/bin/env bash
# circuit-check on the tile with its core in TN + 1 units at the served qkv place (K = 256, 1 × 4 of 512 columns):
#   research run --on auto --cwd source --wait --mem-gb 120 --cpus 8 -- bash benchmarks/pouw/credited_v4_circuit_check.sh
set -u
S="K=256,TM=1,TR=1,TN=4,N=512,RHO=400,RHOR=100,DEV='h100'"
uv sync -q --all-packages --extra torch-cpu
uv run circuit-check -j 4 --out "$OUT/circuit-check-v4.json" \
  "PearlPlusTileCoreWords_v7{$S}" "PearlPlusTileCoreWired_v4{$S}" \
  "PearlPlusTileRowBWords_v1{HEAD=1,K=256,DEV='h100',TN=4}" "PearlPlusTileRowBWords_v1{HEAD=0,K=256,DEV='h100',TN=4}" \
  "PearlPlusTileCoreRestWords_v1{$S}" "PearlPlusTileSeed_v5{K=256,TN=4}"
echo "circuit-check rc $?"
