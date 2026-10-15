#!/usr/bin/env bash
# R5 (CMP-12 red team of M5): can a prover run vy-nebius-2's GPUs above P8's 2,430 MHz? Read-only: it changes no
# clock, offset or power limit. It reads the driver's clock limits, supported clocks and the VF offset range.
#   research run --on vy-nebius-2 --project verity --campaign pouw --source <tree> --cwd clone \
#     --send r5_clocks.sh -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/r5_clocks.sh"'
set -uo pipefail
OUT=${OUT:?run under research run}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
cat >"$WORK/vf.py" <<'EOF'
import ctypes, json
nv = ctypes.CDLL("libnvidia-ml.so.1")
print(json.dumps({"init": nv.nvmlInit_v2()}))
n = ctypes.c_uint()
nv.nvmlDeviceGetCount_v2(ctypes.byref(n))
for i in range(n.value):
    h = ctypes.c_void_p()
    nv.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(h))
    r = {"probe": "vf", "index": i}
    for name in ("nvmlDeviceGetGpcClkMinMaxVfOffset", "nvmlDeviceGetMemClkMinMaxVfOffset"):
        lo, hi = ctypes.c_int(), ctypes.c_int()
        try:
            r[name] = [getattr(nv, name)(h, ctypes.byref(lo), ctypes.byref(hi)), lo.value, hi.value]
        except AttributeError:
            r[name] = "missing"
    for name in ("nvmlDeviceGetGpcClkVfOffset", "nvmlDeviceGetMemClkVfOffset"):
        v = ctypes.c_int()
        try:
            r[name] = [getattr(nv, name)(h, ctypes.byref(v)), v.value]
        except AttributeError:
            r[name] = "missing"
    for clk, label in ((0, "graphics"), (1, "sm"), (2, "mem")):
        v = ctypes.c_uint()
        r["max_" + label] = [nv.nvmlDeviceGetMaxClockInfo(h, clk, ctypes.byref(v)), v.value]
        v2 = ctypes.c_uint()
        r["maxcustomerboost_" + label] = [nv.nvmlDeviceGetMaxCustomerBoostClock(h, clk, ctypes.byref(v2)), v2.value]
    print(json.dumps(r))
EOF
cat >"$WORK/gpu.sh" <<EOF
set -uo pipefail
nvidia-smi --query-gpu=index,name,clocks.sm,clocks.max.sm,clocks.applications.graphics,clocks.default_applications.graphics,power.limit,power.max_limit,power.default_limit,clocks_event_reasons.active --format=csv >"$OUT/smi.csv" 2>&1
nvidia-smi -i 0 -q -d CLOCK,POWER,PERFORMANCE >"$OUT/q-clock.txt" 2>&1
nvidia-smi -i 0 -q -d SUPPORTED_CLOCKS >"$OUT/q-supported.txt" 2>&1
python3 "$WORK/vf.py" >"$OUT/vf.jsonl" 2>"$OUT/vf.err"
nvidia-smi --version >"$OUT/smi-version.txt" 2>&1
EOF
gpu-lease 1 --wait --max-min 5 -- bash "$WORK/gpu.sh"
echo "lease exit $?" >>"$OUT/lease.log"
grep -E "Graphics|SM|Memory" "$OUT/q-supported.txt" | head -6 >"$OUT/supported-top.txt"
cat "$OUT/smi.csv" "$OUT/vf.jsonl" "$OUT/supported-top.txt" 2>/dev/null
