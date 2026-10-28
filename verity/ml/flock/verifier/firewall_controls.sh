#!/usr/bin/env bash
# The firewall's negative controls through the Lean session loop: one honest firewalled CPU `prove --zk` session (FC_FIREWALL=1
# FC_HOST_WITNESS=1), then one session per FC_FIREWALL_TAMPER control (the device dishonest in one named way, live
# PROTOCOL.md §10.1), each against the same `serve --zk`. The honest session must be accepted, and every control must end in a
# visible refusal at its `Finish`, never before: the verifier's verdict (its `SESSION` line) does not accept, after as many
# rounds as the honest session's, and the prover's record carries the control's own refusal (`EXPECT` below:
# `prover_stopped`'s, the firewall's, and the verdict's `why`, the Lean frame's stop where it stopped a rep). A `cut-MODE` control (a worker that fails) runs at its table's device rounds 2, half way and the last of
# the honest session's (`firewall_device_rounds`). Run from the source tree's root:
#   bash verity/ml/flock/verifier/firewall_controls.sh
#   (a job run from the source tree, 16 CPUs, 48 GB, keeping out/*)
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
n=0
session() {
  local name=$1 tamper=$2 at=$3 env=(FC_FIREWALL=1 FC_HOST_WITNESS=1 FC_FIREWALL_LEAN=$FIREWALL)
  n=$((n + 1))
  [ -n "$tamper" ] && env+=(FC_FIREWALL_TAMPER=$tamper)
  [ -n "$at" ] && env+=(FC_FIREWALL_CUT_AT=$at)
  log "$tag: session $name"
  env "${env[@]}" "$BIN" prove --zk --verifier 127.0.0.1:$port --instances "$D/inst.bin" --circuit "$D/circuit.txt" \
    --runs 1 --warm 0 > "$O/$name.out" 2> "$O/$name.err"
  local rc=$? i=0
  # serve's SESSION line for the session, once it has written it (the sessions run one at a time)
  until [ "$(grep -c '^SESSION' "$O/serve.log")" -ge $n ] || [ $i -ge 120 ]; do sleep 1; i=$((i + 1)); done
  grep '^SESSION' "$O/serve.log" | sed -n "${n}p" | cut -f2 > "$O/$name.auditor"
  "$PY" - "$name" "$rc" "$O/$name.out" "$O/$name.err" "$O/$name.auditor" >> "$O/controls.jsonl" <<'EOF'
import json, sys
name, rc, out, err, aud = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
line = open(aud).read().strip()
auditor = {k: v for k, v in json.loads(line).items() if k in ("accepted", "why", "rounds")} if line else None
v = next((json.loads(l[5:]) for l in open(out) if l.startswith("LIVE\t")), None)
stops = [l.rstrip("\n")[:300] for l in open(err, errors="replace") if l.startswith("frames: ")]
if v is None:
    rec = {"control": name, "rc": rc, "auditor": auditor, "error": [l.rstrip("\n")[:300] for l in open(err, errors="replace")][-3:]}
else:
    why = (v.get("verdict") or {}).get("why") or v.get("why") or ""
    dev = max([b.get("firewall_device_rounds") or 0 for b in v.get("buckets") or []] or [0])
    rec = {"control": name, "rc": rc, "accepted": v.get("accepted"), "why": str(why)[:300],
           "prover_stopped": str(v.get("prover_stopped") or "")[:300], "device_rounds": dev, "auditor": auditor,
           "frame_stops": stops[:3] + ([f"... {len(stops) - 3} more"] if len(stops) > 3 else [])}
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
# each control's refusal: prover_stopped's prefix (the firewall's own, "" where it caught nothing), and whether the verdict's
# `why` is the Lean frame's stop of a rep
FW, DEV, FRAME = "FIREWALL-REFUSED: ", "FIREWALL-REFUSED: the device failed: ", "PROVER-STOPPED: the Lean frame stopped table 0 rep "
EXPECT = {"root": (FW + "level-0 root: the device committed ", True), "witness": (FW + "level-0 root: the device committed ", True),
          "label": (FW + "the proof's exposed value ", True), "opening": ("", True), "opening-frame": ("", True),
          "algebra": (FW + "circuit: the constraints ", False), "algebra-proof": (FW + "the proof's exposed value ", False),
          "shape": (FW + "the proof's shape ", False), "cut-rc": (DEV + "CUDA circuit prover returned error ", None),
          "cut-rc240": (DEV + "PROVER-STOPPED: degenerate coins: ", None), "cut-panic": (DEV + "panicked: ", None),
          "cut-silent": (DEV + "silent past the firewall's deadline", None), "cut-decode": (DEV + "the device's output does not decode", None),
          "cut-tree": ("", True)}
rows = [json.loads(l) for l in open(sys.argv[1])]
rounds = (rows[0].get("auditor") or {}).get("rounds") if rows else None
for r in rows:
    a = r.get("auditor") or {}
    if r["control"] == "honest":
        r["ok"] = r.get("accepted") is True and a.get("accepted") is True and bool(rounds)
    else:
        stopped, frame = EXPECT[r["control"].split("@")[0]]
        why, ps = r.get("why") or "", r.get("prover_stopped") or ""
        r["ok"] = (r.get("accepted") is False and a.get("accepted") is False and a.get("rounds") == rounds
                   and (ps.startswith(stopped) if stopped else ps == "") and frame in (None, why.startswith("Ok(Refused(\"" + FRAME)))
    print(("OK " if r["ok"] else "BAD ") + json.dumps(r))
ok = bool(rows) and rows[0]["control"] == "honest" and all(r["ok"] for r in rows)
json.dump({"ok": ok, "rows": rows}, open(sys.argv[2], "w"), indent=1)
print("controls:", "every one as required" if ok else "NOT as required")
sys.exit(0 if ok else 1)
EOF
