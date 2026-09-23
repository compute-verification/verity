#!/bin/bash
# Verifier timing on vy-cpu2 (12 of 32 threads; shared pod).  Proofs/statements from the a-verifier lane live in
# /workspace/averifier/proofs/.  MODES: env-var settings to A/B (space separated "NAME=VALUE" or "-" for none).
set -e
export PATH=$HOME/.cargo/bin:$PATH
R=$RESEARCH_RUN_DIR
PR=/workspace/averifier/proofs
S=$PR/r20260922-091240-4bf9/stmt
P4096=$PR/r20260922-091240-4bf9/proof4096_torch.bin
P64=$PR/r20260922-090318-2864/proof64_torch.bin
MODES=${MODES:-"-"}
THREADS=${THREADS:-"12 1"}
REPS=${REPS:-2}
(cd backends/gkr/verifier && CARGO_TARGET_DIR=/workspace/cargo-target-a-verifier cargo build --release 2>&1 | tail -1)
if [ -n "$TEST" ]; then (cd backends/gkr/verifier && CARGO_TARGET_DIR=/workspace/cargo-target-a-verifier cargo test --release 2>&1 | grep -E 'test result|panicked|FAILED'); fi
V=/workspace/cargo-target-a-verifier/release/verity-gkr-verify
uptime; nproc; grep -m1 "model name" /proc/cpuinfo
for mode in $MODES; do
  for th in $THREADS; do
    for rep in $(seq 1 $REPS); do
      if [ "$mode" = "-" ]; then env_=""; else env_="$mode"; fi
      echo "=== mode=$mode threads=$th rep=$rep B=4096"
      env $env_ $V verify --dir $S --proof $P4096 --threads $th --json $R/v4096_${mode//=/_}_t${th}_r${rep}.json | cut -c1-400
      [ "$th" = "1" ] && [ "$rep" != "1" ] && continue
    done
    echo "=== mode=$mode threads=$th B=64"
    env $env_ $V verify --dir $S --proof $P64 --vus 64 --threads $th --json $R/v64_${mode//=/_}_t${th}.json | cut -c1-300
  done
done
if [ -n "$NEG" ]; then
  echo "=== negatives"; $V negatives --dir $S --proofs $PR/r20260922-091240-4bf9/negproofs --threads 12 --json $R/neg.json | tail -2
  echo "=== mutate B=64 (sampled)"; $V mutate --dir $S --proof $P64 --vus 64 --threads 2 --parallel 6 --sample ${SAMPLE:-200} --json $R/mut64.json | tail -3
fi
