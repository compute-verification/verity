#!/usr/bin/env bash
# What the ZK server adds to one lane statement: served.sh on a stand-in window (H1S_SMOKE=K), lane 0 alone, no negatives,
# once under the ZK server (H1S_FIREWALL=1) and once without it (0), each lane Lean-checked. out/summary.json has each
# arm's job-cgroup CPU seconds, the flock-firewall processes' CPU seconds (summed over each one's last /proc sample: the
# prover never waits for them, so its timed.py record leaves them out) and summed peak RSS, and served.sh's summary.
#   research run --on auto --cpus 16 --mem-gb 240 --timeout 240m --custody-ttl 12h --cwd source -- bash benchmarks/pouw/h1_lanes/zk_cost.sh
# env: ZC_K (2048)  ZC_ARMS ("1 0")  ZC_MEM_GB (served.sh's H1L_MEM_GB, 230)  HZ_CACHE (/workspace/research/cache/pouw-hidden-zk)
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
OUT=${RESEARCH_RUN_DIR:?}/out
mkdir -p "$OUT"
B=$(mktemp -d "${TMPDIR:-/tmp}/zk-cost.XXXXXX") || exit 1
trap 'kill "${SAMPLER:-}" 2>/dev/null; rm -rf "$B"' EXIT
CG=/sys/fs/cgroup$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup)
cg_cpu() { awk '$1 == "usage_usec" {print $2}' "$CG/cpu.stat" 2>/dev/null; }
HZ=$(getconf CLK_TCK) MINE=$(cat /proc/self/cgroup)
( declare -A cpu; arm= peak=0
  while :; do
    a=$(cat "$B/arm" 2>/dev/null)
    if [ "$a" != "$arm" ]; then
      if [ -n "$arm" ]; then t=0; for p in "${!cpu[@]}"; do t=$(( t + cpu[$p] )); done
        echo "$arm $(( t / HZ )) ${#cpu[@]} $peak" >> "$OUT/firewall.txt"; fi
      cpu=() arm=$a peak=0
    fi
    r=0
    for p in $(pgrep -x flock-firewall); do
      [ "$(cat "/proc/$p/cgroup" 2>/dev/null)" = "$MINE" ] || continue
      s=$(cut -d' ' -f14,15 "/proc/$p/stat" 2>/dev/null) || continue
      [ -n "$s" ] && cpu[$p]=$(( ${s% *} + ${s#* } ))
      r=$(( r + $(awk '$1 == "VmRSS:" {print $2}' "/proc/$p/status" 2>/dev/null || echo 0) ))
    done
    [ "$r" -gt "$peak" ] && peak=$r
    sleep 1
  done ) &
SAMPLER=$!
for fw in ${ZC_ARMS:-1 0}; do
  echo "fw$fw" > "$B/arm"
  c0=$(cg_cpu)
  RESEARCH_RUN_DIR=$B/fw$fw H1L_MEM_GB=${ZC_MEM_GB:-230} H1S_SMOKE=${ZC_K:-2048} H1L_LANES=0 H1S_SKIP=tampered-zy,other-call-a \
    H1S_FIREWALL=$fw H1L_PAR=1 H1L_LPAR=1 HZ_CACHE=${HZ_CACHE:-/workspace/research/cache/pouw-hidden-zk} \
    bash benchmarks/pouw/h1_lanes/served.sh > "$OUT/fw$fw.log" 2>&1
  rc=$? c1=$(cg_cpu)
  mkdir -p "$OUT/fw$fw" && cp -r "$B/fw$fw/out/." "$OUT/fw$fw/" 2>/dev/null
  echo "fw$fw exit $rc cgroup_cpu_s $(( (c1 - c0) / 1000000 ))" | tee -a "$OUT/arms.txt"
done
echo done > "$B/arm"
sleep 3
cat "$OUT/arms.txt" "$OUT/firewall.txt" 2>/dev/null
for fw in ${ZC_ARMS:-1 0}; do
  echo "== fw$fw"; grep -h -E 'proved lane0|lean lane0|FAILED|prepare:' "$OUT/fw$fw.log" | cut -c1-400
  cat "$OUT/fw$fw/lane0/prove.time" "$OUT/fw$fw/lane0/serve.time" 2>/dev/null; echo
done
