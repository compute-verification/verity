#!/usr/bin/env bash
# sqfloor.cu on one GPU (cwd = the source tree): the squaring's rate mod 2^b - c at b = $BITS (default 768, 1536 and
# 2048) in the codec lanes' column form, a row form and one level of Karatsuba, gated against Python, with the SASS
# opcode mix and ncu's pipe counts. Results in $RESEARCH_RUN_DIR/out, per width.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --campaign resource-accounting \
#     --declared-output 'out/*' --env GPU_LEASE_WHO=ra-rate-redteam -- taskset -c 48-95 bash benchmarks/rate_redteam/sqfloor.sh
set -uxo pipefail
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
export PATH=/usr/local/cuda-13.0/bin:$PATH
export D=benchmarks/rate_redteam
export B; B=$(mktemp -d /tmp/rate-redteam-sq.XXXXXX); trap 'rm -rf "$B"' EXIT
export BITS=${BITS:-768 1536 2048}
for b in $BITS; do
  nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xptxas -v -DSQ_BITS=$b -o "$B/sqfloor-$b" $D/sqfloor.cu \
    > "$O/build-$b.log" 2>&1 || { tail -40 "$O/build-$b.log"; exit 4; }
  cuobjdump -sass "$B/sqfloor-$b" > "$B/sqfloor-$b.sass" 2>&1
  python3 - "$B/sqfloor-$b.sass" > "$O/sass-mix-$b.json" <<'EOF'
import collections, json, re, sys
out, fn = {}, None
for line in open(sys.argv[1]):
    m = re.search(r"Function : (\S+)", line)
    if m:
        fn = m.group(1); out[fn] = {"base": collections.Counter(), "full": collections.Counter()}; continue
    m = re.match(r"\s+/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_]*)(\.[A-Z0-9_.]+)?", line)
    if m and fn is not None:
        out[fn]["base"][m.group(1)] += 1; out[fn]["full"][m.group(1) + (m.group(2) or "")] += 1
srt = lambda c: dict(sorted(c.items(), key=lambda kv: -kv[1]))
print(json.dumps({f: {"total": sum(c["base"].values()), "base": srt(c["base"]), "full": srt(c["full"])}
                  for f, c in out.items()}, indent=1))
EOF
done
LEASE=(gpu-lease 1 --wait --max-min "${MAXMIN:-25}" --)
[ -n "${GPU_LEASE_UUID:-}" ] && LEASE=()
# shellcheck disable=SC2016
"${LEASE[@]}" bash -c '
  nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,clocks.applications.graphics,power.draw \
    --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  "$B/sqfloor-${BITS%% *}" clock > "$O/clock-before.json" 2>&1
  for b in $BITS; do
    for f in 0 1 2; do
      for k in 1 777 1000; do
        "$B/sqfloor-$b" gate $f $k 512 > "$B/gate-$b-$f-$k.txt" 2>> "$O/errors.txt" || echo "gate $b $f $k rc $?" >> "$O/errors.txt"
        python3 $D/sqfloor_gate.py < "$B/gate-$b-$f-$k.txt" > "$O/gate-b$b-f$f-k$k.json" 2>&1
      done
    done
    for f in 0 1 2; do
      for w in 2 4; do
        "$B/sqfloor-$b" time $f 4000 128 $w 5 >> "$O/time.jsonl" 2>> "$O/errors.txt" || echo "time $b $f $w rc $?" >> "$O/errors.txt"
      done
    done
  done
  "$B/sqfloor-${BITS%% *}" clock > "$O/clock-after.json" 2>&1
  nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  for b in $BITS; do
    for f in 0 1 2; do
      ncu --section ComputeWorkloadAnalysis --section InstructionStats --section Occupancy --csv --page details \
        -k regex:chain -c 1 "$B/sqfloor-$b" time $f 400 128 2 1 > "$O/ncu-b$b-f$f.csv" 2>&1 || echo "ncu $b $f rc $?" >> "$O/errors.txt"
      ncu --metrics smsp__inst_executed.sum,sm__inst_executed_pipe_alu.sum,sm__inst_executed_pipe_fma.sum,sm__inst_executed_pipe_fmaheavy.sum,sm__inst_executed_pipe_fmalite.sum,smsp__issue_active.avg.pct_of_peak_sustained_active \
        --csv -k regex:chain -c 1 "$B/sqfloor-$b" time $f 400 128 2 1 > "$O/ncu-metrics-b$b-f$f.csv" 2>&1 || echo "ncu metrics $b $f rc $?" >> "$O/errors.txt"
      ncu --metrics sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_lsu.sum,l1tex__t_bytes_pipe_lsu_mem_local.sum \
        --csv -k regex:chain -c 1 "$B/sqfloor-$b" time $f 400 128 2 1 > "$O/ncu-pipes-b$b-f$f.csv" 2>&1 || echo "ncu pipes $b $f rc $?" >> "$O/errors.txt"
    done
  done'
