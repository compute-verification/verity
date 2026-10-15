#!/usr/bin/env bash
# CMP-12 W7: PoUW η of Pearl-C sm120 v1 (-h3) on one leased RTX PRO 6000 (tier 2a diagnostic; the fill queue keeps running).
# A: build.sh's ship (nvcc 13.0, SASS-gated, fixture) off the lease.  B, in one lease: run.py's fixture check on the card (the
# gate: nothing below runs unless it passes), run.py's split, eta.py's schedules and batch curve, the clock hold, then ncu
# (--clock-control none: the node's lock stays as it is) kernel by kernel and graph by graph.  CSVs from the reports after.
# ETA_LEASE names B's script instead (lease_pearlc2.sh: the SHA-512 probe, par<c>, the power holds; lease_pearlc3.sh:
# serial-r32; lease_data.sh: gemm_fast on low-power operands; lease_fast.sh: hash_fast.cuh's gates and timings;
# lease_cores.sh: the served call at the 64 x 64 tile, hash CTAs beside the GEMM's).
# Run from a clone of the pushed commit: research run ... --cwd clone --send job_pearlc.sh -- bash "$RESEARCH_RUN_DIR/inputs/job_pearlc.sh"
set -euo pipefail
d=$(mktemp -d); trap 'rm -rf "$d"' EXIT
nice -n 19 bash benchmarks/pouw/pearl_c_sm120/build.sh "$d/ship"
cp "$d/ship/toolchain.txt" "$d/ship/source.txt" "$d/ship/sass.json" "$d/ship/MANIFEST.sha256" "$OUT/"
cp benchmarks/pouw_eta/eta.py "$d/ship/"
git rev-parse HEAD > "$OUT/eta_commit.txt"
# ETA_PROBE_VARIANTS ("NTxMINB ..."): hash_probe.cu per launch-bound variant, with build.sh's flags, for eta.py's sha2 mode
ETA_PROBES=""
for v in ${ETA_PROBE_VARIANTS:-}; do
  nice -n 19 /usr/local/cuda-13.0/bin/nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -fmad=false -Xptxas -v -cubin \
    -DRS_FAST_NT="${v%x*}" -DRS_FAST_MINB="${v#*x}" -o "$d/probe_$v.cubin" benchmarks/pouw_eta/hash_probe.cu \
    2> "$OUT/probe_ptxas_$v.txt" && ETA_PROBES="$ETA_PROBES $v=$d/probe_$v.cubin" || echo "probe $v failed to build" >&2
done
export ETA_PROBES
export SHIP="$d/ship" NCU=/usr/local/cuda-13.0/bin/ncu
nvidia-smi --query-gpu=index,uuid,clocks.sm,clocks.max.sm,clocks.applications.graphics,clocks.default_applications.graphics,power.draw,power.limit,power.max_limit,clocks_event_reasons.active,utilization.gpu \
  --format=csv > "$OUT/smi_all.csv" 2>&1 || true
gpu-lease 1 --wait --max-min 25 -- bash "benchmarks/pouw_eta/${ETA_LEASE:-lease_pearlc.sh}"
for r in ncu_kernels ncu_graphs; do
  [ -f "$OUT/$r.ncu-rep" ] && "$NCU" --import "$OUT/$r.ncu-rep" --csv --page raw > "$OUT/$r.csv" || echo "no $r report" >&2
done
ls -la "$OUT"
