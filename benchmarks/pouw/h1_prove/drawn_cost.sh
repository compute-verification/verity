#!/usr/bin/env bash
# drawn.sh's sessions under the ZK server against the unfirewalled --zk path (H1_FIREWALL=0), on its test's served
# window (test_pouw_h1_drawn.py's `served` fixture under VERITY_VLLM_H1_DRAWN_J10R=1: J10-R's own tile unit at K 64, draw
# K 2): each arm stages, proves and has credit check (drawn.py check, the Lean flock-verify) the window's sampled calls.
# out/summary.json has each arm's sessions: the job cgroup's CPU seconds over each (the ZK server's Lean among them), both
# sides' timed.py records, the LIVE verdicts, credit's line, and the peak RSS summed over the flock-firewall processes.
#   research run --on auto --cpus 32 --mem-gb 320 --timeout 180m --cwd source --wait -- bash benchmarks/pouw/h1_prove/drawn_cost.sh
# env: DC_MEM_GB (drawn.sh's H1_DRAWN_MEM_GB, 400)  DC_SERVE_GB (H1_ZK_SERVER_GB, 125)  DC_LEAN_GB (H1_ZK_LEAN_GB, 145)
#      DC_ARMS ("1 0": H1_FIREWALL of each arm, in order)  HZ_CACHE (/workspace/research/cache/pouw-hidden-zk)
#      DC_REAL (1; 0 the test's stand-in J10-R tile, a circuit of a few hundred gates over J10-R's ports: each arm proves
#      and checks the test's own staging, H1_DRAWN_STAGED, which drawn.py can't restage on the stand-in)
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
OUT=${RESEARCH_RUN_DIR:?}/out
mkdir -p "$OUT"
uv sync -q --all-packages --extra torch-cpu > "$OUT/sync.log" 2>&1 || { tail -20 "$OUT/sync.log"; exit 1; }
B=$(mktemp -d "${TMPDIR:-/tmp}/drawn-cost.XXXXXX") || exit 1
trap 'kill "${SAMPLER:-}" 2>/dev/null; rm -rf "$B"' EXIT
REAL=${DC_REAL:-1}
(cd adapters/vllm && VERITY_VLLM_H1_DRAWN_J10R=$REAL ../../.venv/bin/python -m pytest -q -p no:cacheprovider --basetemp "$B/t" \
  tests/protocols/test_pouw_h1_drawn.py -k staged_proved_and_credits -m slow) > "$OUT/fixture.log" 2>&1 \
  || { tail -30 "$OUT/fixture.log"; exit 1; }
T=$(ls -d "$B"/t/test_a_served* | head -1)
echo "fixture: $T (DC_REAL=$REAL)" | tee "$OUT/arms.txt"
STAGED=
[ "$REAL" = 1 ] || STAGED=$T/work
# the flock-firewall processes' summed RSS, its peak per arm (they are the prover's children it never waits for)
MINE=$(cat /proc/self/cgroup)
( peak=0 arm=
  while :; do
    a=$(cat "$B/arm" 2>/dev/null)
    [ "$a" != "$arm" ] && { [ -n "$arm" ] && echo "$arm $peak" >> "$OUT/firewall-rss.txt"; arm=$a peak=0; }
    r=0
    for p in $(pgrep -x flock-firewall); do
      [ "$(cat "/proc/$p/cgroup" 2>/dev/null)" = "$MINE" ] || continue
      r=$(( r + $(awk '$1 == "VmRSS:" {print $2}' "/proc/$p/status" 2>/dev/null || echo 0) ))
    done
    [ "$r" -gt "$peak" ] && peak=$r
    sleep 2
  done ) &
SAMPLER=$!
for fw in ${DC_ARMS:-1 0}; do
  echo "fw$fw" > "$B/arm"
  H1_FIREWALL=$fw H1_CIRCUITS=$T/circuits H1_WEIGHTS=$T/weights.npz H1_WEIGHTS_KEY=$T/wkey H1_DRAW_K=2 H1_DRAWN_ONLY=1 \
    H1_DRAWN_MEM_GB=${DC_MEM_GB:-400} H1_ZK_SERVER_GB=${DC_SERVE_GB:-125} H1_ZK_LEAN_GB=${DC_LEAN_GB:-145} \
    HZ_CACHE=${HZ_CACHE:-/workspace/research/cache/pouw-hidden-zk} PY=$PWD/.venv/bin/python H1_DRAWN_WORK=$B/work$fw \
    H1_DRAWN_STAGED=$STAGED bash benchmarks/pouw/h1_prove/drawn.sh "$T/serve" 0 "$OUT/fw$fw" > "$OUT/fw$fw.log" 2>&1
  echo "fw$fw exit $?" | tee -a "$OUT/arms.txt"
  # the fixture's witness is the test's own, so its provers' stderr may leave the node, which drawn.sh's never does
  for e in "$B"/work"$fw"/h1-drawn.*/out/*/prove.err; do
    [ -e "$e" ] && { echo "== $e"; tail -20 "$e"; } >> "$OUT/fw$fw-prove-err.txt"
  done
done
echo done > "$B/arm"
sleep 3
.venv/bin/python - "$OUT" <<'EOF' | tee "$OUT/summary.json"
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
rss = dict(l.split() for l in (out / "firewall-rss.txt").read_text().splitlines()) if (out / "firewall-rss.txt").exists() else {}
def rd(p):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None
doc = {}
for arm in sorted(out.glob("fw*/")):
    log = (out / f"{arm.name}.log").read_text(errors="replace").splitlines()
    sessions = {}
    for s in sorted((arm / "drawn" / "sessions").glob("*/")):
        sessions[s.name] = {"cgroup": rd(s / "cgroup.json"), "prove": rd(s / "prove.time"), "serve": rd(s / "serve.time"),
                            "live": [json.loads(l.split("\t", 1)[1]) .get("accepted") for l in
                                     (s / "verdicts.txt").read_text().splitlines() if "\t" in l]
                            if (s / "verdicts.txt").exists() else None}
    doc[arm.name] = {"sessions": sessions, "credit": [l[:400] for l in log if l.startswith(("PROVED", "REFUSED"))],
                     "firewall_peak_rss_gb": round(int(rss.get(arm.name, 0)) / 2**20, 2)}
print(json.dumps(doc, indent=1))
EOF
