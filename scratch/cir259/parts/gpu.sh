#!/usr/bin/env bash
# PARTSRETIRE: main's new phase_profile digests (MAIN_ZK, MAIN_M0) for the census W13 default, on vy-nebius-2 (sm_120).
# Setup, build and stage outside the lease; ident alone on one leased card, at most 30 min.
# research run --on vy-nebius-2 --cwd source --send gpu.sh -- bash $RESEARCH_RUN_DIR/inputs/gpu.sh
set -uo pipefail
export KW=${KW:-/workspace/jobs/partsretire-phase} SM=${SM:-120} AS=base
O=$RESEARCH_RUN_DIR/out; mkdir -p $O
echo "gpu.sh: source $(git rev-parse HEAD) KW $KW SM $SM start $(date -u +%H:%M:%SZ)" | tee -a $O/gpu.log
STEPS=setup,build,stage bash benchmarks/one_stage/phase_profile.sh; echo "PREP rc=$? $(date -u +%H:%M:%SZ)" | tee -a $O/gpu.log
GPU_LEASE_WHO=PARTSRETIRE gpu-lease 1 --wait --max-min 30 -- env STEPS=ident bash benchmarks/one_stage/phase_profile.sh
echo "IDENT rc=$? $(date -u +%H:%M:%SZ)" | tee -a $O/gpu.log
for n in m1 m0; do
  echo "DIGEST $n $(grep -ho '"transcript_sha256": *"[0-9a-f]*"' $O/ident-base-$n.txt 2>/dev/null | sort -u | tr '\n' ' ')" | tee -a $O/gpu.log
done
