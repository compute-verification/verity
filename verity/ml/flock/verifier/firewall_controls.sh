#!/usr/bin/env bash
# The firewall's negative controls through the Lean session loop: one honest firewalled CPU `prove --zk` session (FC_FIREWALL=1
# FC_HOST_WITNESS=1), then one session per FC_FIREWALL_TAMPER control (the device dishonest in one named way, live
# PROTOCOL.md §10.1), each against the same `serve --zk`. The honest session must be accepted, and every control must end in a
# visible refusal: the verifier does not accept, and the prover's record carries the refusal (`prover_stopped`, or the
# verdict's `why`). A `cut-MODE` control (a worker that fails) runs at its table's device rounds 2, half way and the last of
# the honest session's (`firewall_device_rounds`). Run from the source tree's root:
#   research run --on vy-nebius-1 --project verity --source . --cwd source --declared-output 'out/*' --cpus 16 --mem-gb 48 \
#     -- bash verity/ml/flock/verifier/firewall_controls.sh
# FC_CONTROLS names the controls (default every one); the builds, staging and config are firewall_finish_agree.sh's
# (FF_CONFIG, default hidden:256). Results in $RESEARCH_RUN_DIR/out: controls.json and each session's logs.
set -uo pipefail
SRC=$(pwd)
CFG=${FF_CONFIG:-hidden:256}
export HZ_CONFIGS=$CFG HZ_N=${FF_N:-1} HZ_MEM_GB=${HZ_MEM_GB:-48}
export HZ_CACHE=${HZ_CACHE:-/workspace/research/cache/pouw-hidden-zk}
# shellcheck source=../../../../benchmarks/pouw/hidden_zk/job.sh
source "$SRC/benchmarks/pouw/hidden_zk/job.sh"
setup
FIREWALL=${LEAN%/flock-verify}/flock-firewall
[ -x "$FIREWALL" ] || { log "missing: $FIREWALL"; exit 1; }
role=${CFG%%:*} k=${CFG#*:} tag=${CFG%%:*}-k${CFG#*:}
[ "$role" != h1 ] || { log "the controls run on a hidden tile's config, not $CFG"; exit 1; }
D=$W/$tag O=$OUT/$tag
mkdir -p "$D" "$O"
stage_config "$role" "$k" "$tag" "$D" "$O" || { log "$tag: staging FAILED: $(tail -1 "$O/stage.err")"; exit 1; }
pin=$(cat "$D/pin.txt") port=$(free_port)
"$BIN" serve --zk --listen 127.0.0.1:$port --out "$D/sessions" --instances "$D/pub.bin" --circuit "$D/circuit.txt" --pin "$pin" \
  --operator loopback --draw "subset:$N" > "$O/serve.log" 2>&1 &
srv=$!
until grep -q '^SERVING ' "$O/serve.log" 2>/dev/null; do kill -0 $srv 2>/dev/null || { log "serve: $(tail -1 "$O/serve.log")"; exit 1; }; sleep 1; done
# one session: its name, then FC_FIREWALL_TAMPER and FC_FIREWALL_CUT_AT (empty: none); its LIVE record, with the control's
# name and the prover's exit, goes to $O/controls.jsonl
session() {
  local name=$1 tamper=$2 at=$3 env=(FC_FIREWALL=1 FC_HOST_WITNESS=1 FC_FIREWALL_LEAN=$FIREWALL)
  [ -n "$tamper" ] && env+=(FC_FIREWALL_TAMPER=$tamper)
  [ -n "$at" ] && env+=(FC_FIREWALL_CUT_AT=$at)
  log "$tag: session $name"
  env "${env[@]}" "$BIN" prove --zk --verifier 127.0.0.1:$port --instances "$D/inst.bin" --circuit "$D/circuit.txt" \
    --runs 1 --warm 0 > "$O/$name.out" 2> "$O/$name.err"
  local rc=$?
  "$PY" - "$name" "$rc" "$O/$name.out" "$O/$name.err" >> "$O/controls.jsonl" <<'EOF'
import json, sys
name, rc, out, err = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
v = next((json.loads(l[5:]) for l in open(out) if l.startswith("LIVE\t")), None)
stops = [l.rstrip("\n")[:300] for l in open(err, errors="replace") if l.startswith("frames: ")]
if v is None:
    rec = {"control": name, "rc": rc, "error": [l.rstrip("\n")[:300] for l in open(err, errors="replace")][-3:]}
else:
    why = (v.get("verdict") or {}).get("why") or v.get("why") or ""
    dev = max([b.get("firewall_device_rounds") or 0 for b in v.get("buckets") or []] or [0])
    rec = {"control": name, "rc": rc, "accepted": v.get("accepted"), "why": str(why)[:300],
           "prover_stopped": str(v.get("prover_stopped") or "")[:300], "device_rounds": dev, "frame_stops": stops}
print(json.dumps(rec))
EOF
}
session honest "" ""
dev=$("$PY" -c "import json, sys; print(json.loads(open(sys.argv[1]).readline()).get('device_rounds', 0))" "$O/controls.jsonl")
log "$tag: the honest session's device rounds: $dev"
for c in ${FC_CONTROLS:-root witness label opening opening-frame algebra algebra-proof shape cut-rc cut-rc240 cut-panic cut-silent cut-decode cut-tree}; do
  case $c in
    cut-tree) for at in 0 $((dev / 2)); do session "$c@$at" "$c" "$at"; done ;;
    cut-*) for at in 2 $((dev / 2)) $((dev - 1)); do session "$c@$at" "$c" "$at"; done ;;
    *) session "$c" "$c" "" ;;
  esac
done
kill $srv 2>/dev/null; wait $srv 2>/dev/null
"$PY" - "$O/controls.jsonl" "$OUT/controls.json" <<'EOF' | tee -a "$OUT/job.log"
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
for r in rows:
    honest = r["control"] == "honest"
    refused = bool(r.get("prover_stopped") or r.get("why"))
    r["ok"] = r.get("accepted") is True if honest else (r.get("accepted") is False and refused)
    print(("OK " if r["ok"] else "BAD ") + json.dumps(r))
ok = bool(rows) and rows[0]["control"] == "honest" and all(r["ok"] for r in rows)
json.dump({"ok": ok, "rows": rows}, open(sys.argv[2], "w"), indent=1)
print("controls:", "every one as required" if ok else "NOT as required")
sys.exit(0 if ok else 1)
EOF
