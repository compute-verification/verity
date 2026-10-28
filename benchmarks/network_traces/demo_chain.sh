#!/usr/bin/env bash
# Network accounting for the one-H200 demo by the certifier-only path: the demo's endpoint (demo_endpoint.py, its
# engine behind the OpenAI API, writing serve-openai's records), the certifier rehearsal against it on varied traffic
# (certifier_serve.py rehearse, honest and the tampers), the rehearsal kept as the gate keeps a certifier run
# (request.json, out/, end.json with each file's SHA-512, as auditor.certifier_run writes them), then chain_record.sh on
# it with the endpoint's records and CR_PATH=certifier-only. Off-host, from a clone, in one of the network slots, as a
# job of at most 240 minutes on a CPU node:
#   env CHECK_SLOTS=/workspace/research/locks/network/slots python3 tools/verity/check/slot.py --by LANE -- \
#     env CR_MEM_GB=128 CR_N2_ARGS="--draw 3" bash benchmarks/network_traces/demo_chain.sh
# With DC_UPSTREAM the endpoint is that one (the H200's) and none is started here; CR_ENDPOINT then names its records.
# The endpoint's records hold request values (requests.jsonl's prompt and streamed ids, advice_trace.jsonl's arrivals and
# rows, windows.jsonl's users, served_rows/'s salts), so the endpoint writes them in DC_KEEP/endpoint, 0700 and outside
# the run dir and the tree, which chain_record.sh reads; $OUT/endpoint-records gets only serve.json and summary.json.
# env: CR_MEM_GB and chain_record.sh's own (CR_N2_ARGS, CR_DEPLOYED, ...)
#      DC_KEEP (the private dir, /workspace/research/lanes/network-accounting/demo-chain/RUN by default)
#      DC_ENGINE (demo_endpoint's --engine, seeded:phase1)  DC_SEED (its --seed, 0)  DC_EOS (its --eos and the plan's
#      --eos-ids, 0)  DC_UPSTREAM (an endpoint already serving)  DC_REQUESTS (24)  DC_TOKENS (16)  DC_PROMPTS (1:16, the
#      prompt lengths: the demo's context is 32)  DC_RATE (2)  DC_TRAFFIC_SEED (0)  DC_HEIGHT (the LMS keys' height, 20)
#      DC_STRATEGIES (honest,late,extra,token: late and extra the rehearsal's check refuses, token's changed bytes the
#      I/O tie)
set -uo pipefail
SRC=$(pwd)
OUT=${RESEARCH_RUN_DIR:?}/out
mkdir -p "$OUT" "${TMPDIR:-/tmp}"
D=$(mktemp -d "${TMPDIR:-/tmp}/demo-chain.XXXXXX")
log() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$OUT/job.log"; }
EP_PID=
trap '[ -z "$EP_PID" ] || kill "$EP_PID" 2>/dev/null; rm -rf "$D"' EXIT

export HZ_CONFIGS=chain HZ_MEM_GB=${CR_MEM_GB:?}
CR_CACHE=${CR_CACHE:-/workspace/research/cache/pouw-hidden-zk}
if mkdir -p "$CR_CACHE" 2>/dev/null && [ -w "$CR_CACHE" ]; then export HZ_CACHE=$CR_CACHE; else export HZ_CACHE=$D/cache; fi
(source benchmarks/pouw/hidden_zk/job.sh && setup && echo "$PY" > "$D/py") || { log "demo-chain: setup FAILED"; exit 7; }
PY=$(cat "$D/py")
export PATH=$HOME/.elan/bin:$PATH
export PYTHONPATH=$SRC:$SRC/verity/ml/flock/python:$SRC/integrations/vllm:$SRC/demos/h200:$SRC/tools/verity/circuit_check/src:$SRC/benchmarks/numerical/python:$SRC/tools/research/src

EOS=${DC_EOS:-0}
RUN=certifier-$(date -u +%Y%m%dT%H%M%SZ)-offhost
G=$D/gates/$RUN
KEEP=$(realpath -m "${DC_KEEP:-/workspace/research/lanes/network-accounting/demo-chain/$(basename "$RESEARCH_RUN_DIR")}")
case "$KEEP/" in "$(realpath -m "$RESEARCH_RUN_DIR")/"*|"$(realpath -m "$SRC")/"*)
  log "demo-chain: DC_KEEP $KEEP is inside the run dir or the tree: give a directory outside both"; exit 2;; esac
EP=$KEEP/endpoint
mkdir -p "$G/out" "$EP" && chmod 700 "$KEEP" "$EP" || { log "demo-chain: no private dir at $KEEP (DC_KEEP)"; exit 2; }
if [ -n "${DC_UPSTREAM:-}" ]; then URL=$DC_UPSTREAM
else
  "$PY" benchmarks/network_traces/demo_endpoint.py --out "$EP" --engine "${DC_ENGINE:-seeded:phase1}" \
    --seed "${DC_SEED:-0}" --eos "$EOS" --run-id "$RUN" > "$D/endpoint.out" 2> "$OUT/demo-endpoint.log" &
  EP_PID=$!
  for _ in $(seq 600); do grep -q listening "$D/endpoint.out" 2>/dev/null && break; kill -0 "$EP_PID" 2>/dev/null || break
    sleep 0.1; done
  URL=$("$PY" -c 'import json, sys; print(json.loads(open(sys.argv[1]).readline())["listening"])' "$D/endpoint.out") \
    || { log "demo-chain: the endpoint didn't listen"; tail -20 "$OUT/demo-endpoint.log"; exit 3; }
fi
log "demo-chain: endpoint $URL ($(head -c 300 "$D/endpoint.out"))"

STRATEGIES=${DC_STRATEGIES:-honest,late,extra,token}
ARGS=(--upstream "$URL" --traffic varied --requests "${DC_REQUESTS:-24}" --tokens "${DC_TOKENS:-16}"
      --prompt-range "${DC_PROMPTS:-1:16}" --rate "${DC_RATE:-2}" --seed "${DC_TRAFFIC_SEED:-0}" --eos-ids "$EOS"
      --height "${DC_HEIGHT:-20}" --strategies "$STRATEGIES")
"$PY" - "$G/request.json" "$URL" "${ARGS[@]}" <<'EOF'
import json, os, sys
json.dump({"kind": "certifier-rehearse", "where": "off-host: a pool slot, not the gate's host", "upstream": sys.argv[2],
           "engine": None if os.environ.get("DC_UPSTREAM") else os.environ.get("DC_ENGINE", "seeded:phase1"),
           "args": sys.argv[3:], "source": os.environ.get("RESEARCH_SOURCE_SHA")}, open(sys.argv[1], "w"), indent=1)
EOF
log "demo-chain: certifier_serve.py rehearse ${ARGS[*]}"
"$PY" benchmarks/network_traces/certifier_serve.py rehearse "${ARGS[@]}" --out "$G/out" > "$OUT/rehearse.log" 2>&1
RC=$?
tail -n 2 "$OUT/rehearse.log" | cut -c1-600 | tee -a "$OUT/job.log"
if [ -n "$EP_PID" ]; then
  kill -TERM "$EP_PID"; wait "$EP_PID"; EP_PID=
  tail -n 1 "$D/endpoint.out" | cut -c1-600 | tee -a "$OUT/job.log"
  mkdir -p "$OUT/endpoint-records"
  cp "$EP"/serve.json "$EP"/summary.json "$OUT/endpoint-records/" 2>/dev/null
  log "demo-chain: the endpoint's other records stay in $EP: $(ls "$EP" | tr '\n' ' ')"
  "$PY" -m research keep "$KEEP" --owner @network-accounting --expires 7d --priority low \
    --why "demo_chain.sh's endpoint records, which hold request values" >> "$OUT/job.log" 2>&1 \
    || log "demo-chain: research keep $KEEP failed"
fi
"$PY" - "$G" "$RC" <<'EOF'
import hashlib, json, sys, time
from pathlib import Path
root, rc = Path(sys.argv[1]), int(sys.argv[2])
out = root / "out"
s = json.loads((out / "summary.json").read_text()) if (out / "summary.json").is_file() else {}
runs = {k: {"verdict": r.get("verdict"), "network": r.get("network_verdict"), "audit": r.get("audit_accepted"),
            "salt": r.get("salt_accepted"), "salt_again": r.get("salt_again")} for k, r in (s.get("runs") or {}).items()}
files = {str(f.relative_to(out)): hashlib.sha512(f.read_bytes()).hexdigest() for f in sorted(out.rglob("*")) if f.is_file()}
now = time.time()
json.dump({"run": root.name, "rc": rc, "ended": now, "pinned": None, "runs": runs, "files": files,
           "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))}, open(root / "end.json", "w"), indent=1)
print(json.dumps({"rehearsal": root.name, "rc": rc, "runs": {k: v["verdict"] for k, v in runs.items()}}))
EOF
[ -f "$G/out/honest/plan.json" ] || { log "demo-chain: the rehearsal kept no honest run (rehearse.log)"; exit 3; }

log "demo-chain: chain_record.sh on $RUN"
CR_GATES=$D/gates CR_ENDPOINT=${CR_ENDPOINT:-$EP} CR_PATH=${CR_PATH:-certifier-only} CR_DEPLOYED=${CR_DEPLOYED-1} \
  bash benchmarks/network_traces/chain_record.sh
rc=$?
log "demo-chain: done, rehearsal rc $RC, chain rc $rc"
exit "$rc"
