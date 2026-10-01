#!/usr/bin/env bash
# Console overnight loop (cloud successor). Every 5 min:
#  - open PRs on danielreuter/verity -> node 1 /workspace/usage/overnight/console.json
#  - the PR captain's store file -> node 1 pr-captain.json when its sha changes (schema/owner checked)
# The website row keeps the laptop's last count (07:15:50Z): danielreuter/website is not readable from this VM.
set -u
STORE=/cursor/stores/bc-7f347b4b-6175-4b6e-84c6-731add2f8589
CAPTAIN=$STORE/internal/hygiene/overnight-pr-captain.json
W=$HOME/console-overnight
LOG=$W/loop.log
STOP_UTC=${STOP_UTC:-2026-10-01T17:05:00Z}   # 10:05 AM PDT
cd "$W"
eval "SSH=($(cd /workspace && uv run --quiet --package research research pods ssh vy-nebius-1 --print))"

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$LOG"; }

put() {  # put LOCAL NAME: write NAME on node 1 atomically, mode 664
  "${SSH[@]}" "cd /workspace/usage/overnight && cat > $2.tmp && chmod 664 $2.tmp && mv $2.tmp $2" < "$1"
}

last_captain=$(cat "$W/captain.sha" 2>/dev/null || echo none)

while [[ "$(date -u +%Y-%m-%dT%H:%M:%SZ)" < "$STOP_UTC" ]]; do
  now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  if v=$(timeout 60 gh pr list -R danielreuter/verity --state open -L 500 --json isDraft 2>>"$LOG"); then
    python3 - "$now" "$v" > console.json.new <<'EOF'
import json, sys
now, v = sys.argv[1], json.loads(sys.argv[2])
row = lambda m, x, t, s: {"goal": "open-prs", "metric": m, "value": x, "unit": "PRs", "as_of": t, "source": s}
src = "gh pr list --state open"
print(json.dumps({"schema": "verity/overnight-numbers/v0", "owner": "console", "updated_utc": now, "numbers": [
  row("open PRs, verity", len(v), now, src),
  row("of them drafts", sum(p["isDraft"] for p in v), now, src),
  row("open PRs, website", 1, "2026-10-01T07:15:50Z", src)]}))
EOF
    if put console.json.new console.json 2>>"$LOG"; then
      mv console.json.new console.json
      log "console.json open=$(python3 -c 'import json;d=json.load(open("console.json"));print(d["numbers"][0]["value"],"drafts",d["numbers"][1]["value"])')"
    else
      log "ERROR console.json put failed"
    fi
  else
    log "ERROR gh pr list failed"
  fi

  sha=$(sha256sum "$CAPTAIN" 2>/dev/null | cut -d' ' -f1)
  if [[ -n "$sha" && "$sha" != "$last_captain" ]]; then
    if python3 - "$CAPTAIN" <<'EOF'
import json, re, sys
d = json.load(open(sys.argv[1]))
ok = d.get("schema") == "verity/overnight-numbers/v0" and d.get("owner") == "pr-captain" and isinstance(d.get("numbers"), list)
ok = ok and all(re.fullmatch(r"[a-z0-9-]{1,40}", n.get("goal", "")) for n in d["numbers"])
sys.exit(0 if ok else 1)
EOF
    then
      cp "$CAPTAIN" captain.json.copy
      if put captain.json.copy pr-captain.json 2>>"$LOG"; then
        last_captain=$sha; echo "$sha" > captain.sha
        log "pr-captain.json copied sha=${sha:0:12} updated_utc=$(python3 -c 'import json;print(json.load(open("captain.json.copy"))["updated_utc"])')"
      else
        log "ERROR pr-captain.json put failed"
      fi
    else
      log "SKIP pr-captain.json sha=${sha:0:12}: not schema verity/overnight-numbers/v0 owner pr-captain"
      last_captain=$sha
    fi
  fi
  # overnight notes frozen at 16:01Z (top-level, 9:03 AM PDT: the Morning set is live; the overnight set's numbers stand)
  # goals.md's Morning set -> node 1 morning-set.json (panel verity/overnight-morning-set)
  if python3 "$W/morning_set.py" "$STORE/docs/goals.md" > morning-set.json.new 2>morning.err && [[ -s morning-set.json.new ]]; then
    msha=$(sha256sum morning-set.json.new | cut -d' ' -f1)
    if [[ "$msha" != "$(cat morning.sha 2>/dev/null)" ]]; then
      if put morning-set.json.new morning-set.json 2>>"$LOG"; then
        mv morning-set.json.new morning-set.json; echo "$msha" > morning.sha
        log "morning-set.json copied sha=${msha:0:12} $(python3 -c 'import json;d=json.load(open("morning-set.json"));n=d["numbers"];print(len({r["goal"] for r in n}),"goals",sum(1 for r in n if r["metric"]=="status"),"notes",sum(1 for r in n if r["metric"]=="done" and r["value"]=="yes"),"done")')"
      else
        log "ERROR morning-set.json put failed"
      fi
    fi
  else
    log "ERROR morning set: $(tr '\n' ' ' < morning.err | cut -c1-300)"
  fi
  sleep 300
done
log "loop stopped at STOP_UTC"
