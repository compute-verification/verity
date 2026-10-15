#!/usr/bin/env bash
# job_pearlc.sh's data-pattern lease (ETA_LEASE=lease_data.sh): the board's clock settings read-only (nvidia-smi -q), then
# eta.py's `data` mode, gemm_fast held on each operand pattern twice over (the order reversed the second time), so the clock the
# GPU picks on its own under each pattern, its power and throttle reasons.  Nothing here sets a clock or a power limit.
set -uo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID PEARLC_EXPECT_UUID="$GPU_LEASE_UUID"
echo "lease $GPU_LEASE_UUID index $CUDA_VISIBLE_DEVICES $(date -u +%FT%TZ)" >&2
nvidia-smi -q -i "$GPU_LEASE_UUID" -d CLOCK,PERFORMANCE,POWER,VOLTAGE,TEMPERATURE > "$OUT/smi_q_lease.txt" 2>&1 || true
cd "$SHIP"
P='honest zeros ones alt_sign ternary onebit pow2 sparse8 a_zero uniform honest'
ETA_MODE=data ETA_HOLD_S=4 ETA_PATTERNS="$P" python3 eta.py > "$OUT/eta_data.jsonl" 2> "$OUT/eta_data.err" \
  || echo "eta data rc=$?" >&2
ETA_MODE=data ETA_HOLD_S=4 ETA_PATTERNS="$(echo $P | tr ' ' '\n' | tac | tr '\n' ' ')" python3 eta.py \
  > "$OUT/eta_data_rev.jsonl" 2> "$OUT/eta_data_rev.err" || echo "eta data rev rc=$?" >&2
ETA_MODE=data ETA_HOLD_S=12 ETA_PATTERNS='zeros honest ternary' python3 eta.py > "$OUT/eta_data_long.jsonl" \
  2> "$OUT/eta_data_long.err" || echo "eta data long rc=$?" >&2
nvidia-smi -q -i "$GPU_LEASE_UUID" -d CLOCK,PERFORMANCE,POWER > "$OUT/smi_q_after.txt" 2>&1 || true
echo "data done $(date -u +%FT%TZ)" >&2
exit 0
