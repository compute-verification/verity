#!/usr/bin/env bash
# A gate run's steps through the two auditor hosts' gates under the lane's gate key (GATE_KEY), as a job on the machine
# that holds it, cwd = a clone of COMMIT, the commit the verifier host's verifier is pinned at.
# `gate_run.sh` launches it; one VERB a run, everything into $OUT:
#
#   probe  read-only requests: the challenger host's `pubkey` (its public key's SHA-512 must be PIN), `pins` and `ledger`
#          (PERIOD unused, no period of LANE open), and the verifier host's `status` (pinned at COMMIT, no unit running);
#          the circuit, `verity/remote-bandwidth/v0 verifier tree <tree>` of COMMIT's remote_bandwidth tree (auditor_line's
#          `descriptor`). COMMIT, and so the verifier host's pin, must be on origin/main's first-parent line, fetched as
#          auditor.sh does (main.probe.txt; a fetch that fails refuses), since a session credits only under a verifier
#          there. Writes probe.json; exit 3 on any mismatch, before anything is written to either host.
#   open   `period-open --name PERIOD --cap CAP --circuit CIRCUIT --model MODEL`, the ledger read before and after.
#   host   the verifier gate's `run`, gate run 7's request (PROVER_NODE as its peer, PORT, PERIOD, PIN, PACED at
#          PACE_PASS_MS, MINUTES, JITTER_MS, QUANTILE, MODEL with ENGINES arenas of ARENA_BLOCKS); its `status` every 30 s
#          until it ends, then the run's directory without its transcripts (`fetch`) into host/RUN. Exits with the run's rc.
#   close  `period-close --name PERIOD`, then the closed ledger.
#   audit  `auditor.sh` over the host run HOST_RUN (default the one `host` fetched), into $OUT itself: ledger.jsonl,
#          pins.jsonl, auditor_line.json, credit.md, which `gate_whatif.sh` reads.
#   fire   wait until PROVER_RUN holds ENGINES GPUs on PROVER_NODE (its `gpu-lease status`, over that node's line of
#          /etc/vy/peers, or PROVER_SSH's ssh arguments where that line's private address doesn't route, as from
#          a CPU node off the GPU node's network), then probe, open, host, close and audit. A host run that fails leaves the
#          period open for a rerun in it and runs no auditor. With LEASE_BY (a UTC time) it opens nothing for a lease
#          that began after LEASE_BY, and stops waiting 2 minutes after it.
#
# With DRY=1 no request reaches either gate: each is written to gate.requests and printed, the key need not exist, and
# what doesn't need it runs (the circuit, main's first-parent line, the peers line to PROVER_NODE and its lease status,
# the verifier host's host key).
set -u
: "${OUT:?}"
VERB=${1:?probe, open, host, close, audit or fire}
GATE_KEY=${GATE_KEY:-$HOME/.ssh/vy-gate-memory-accounting}
CHALLENGER=${CHALLENGER:-research@204.12.178.84}
AUDITOR=${AUDITOR:-research@81.85.1.145}
AUDITOR_FP=${AUDITOR_FP:-SHA256:s8UdOpJZGMXJR1DjgmOyTsWpTqQX7GDJq/yQCYhSJhs}
LANE=${LANE:-memory-accounting} MODEL=${MODEL:-QWEN25_32B_BF16} ENGINES=${ENGINES:-8} ARENA_BLOCKS=${ARENA_BLOCKS:-16252928}
PORT=${PORT:-7951} CAP=${CAP:-4} PACE_PASS_MS=${PACE_PASS_MS:-85} MINUTES=${MINUTES:-30}
PACED=${PACED:-1:60000:33273:2 1:60000:33273:2 1:1000:2773:7 1:1000:2773:7}
JITTER_MS=${JITTER_MS:-50} QUANTILE=${QUANTILE:-max} PROVER_NODE=${PROVER_NODE:-vy-nebius-2}
SSH=(ssh -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=15
  -i "$GATE_KEY")

gate() {
  # HOST WORDS...: one request to that host's gate, its answer on stdout
  local host=$1
  shift
  echo "$(date -u +%FT%TZ) $host $*" >> "$OUT/gate.requests"
  if [ -n "${DRY:-}" ]; then
    echo "DRY: ssh -i $GATE_KEY $host $*" >&2
    return 0
  fi
  "${SSH[@]}" "$host" "$@"
}

circuit() {
  local head tree
  head=$(git rev-parse HEAD) || return 3
  [ "$head" = "${COMMIT:?}" ] || { echo "this clone is $head, not COMMIT $COMMIT" >&2; return 3; }
  tree=$(git rev-parse "$COMMIT:verity/core/protocols/remote_bandwidth") || return 3
  printf 'verity/remote-bandwidth/v0 verifier tree %s' "$tree" | sha512sum | cut -d' ' -f1
}

main_line() {
  # origin/main's first-parent history into $OUT/main.probe.txt, fetched as auditor.sh fetches it
  local unshallow=()
  [ "$(git rev-parse --is-shallow-repository)" = true ] && unshallow=(--unshallow)
  { git fetch -q "${unshallow[@]}" origin +refs/heads/main:refs/remotes/origin/main \
      || git fetch -q "${unshallow[@]}" origin +refs/remotes/origin/main:refs/remotes/origin/main; } \
    >> "$OUT/fetch.probe.log" 2>&1 && git rev-list --first-parent refs/remotes/origin/main > "$OUT/main.probe.txt"
}

probe() {
  local c
  c=$(circuit) || exit 3
  echo "circuit $c (COMMIT $COMMIT)"
  [ -z "${CIRCUIT:-}" ] || [ "$CIRCUIT" = "$c" ] || { echo "CIRCUIT $CIRCUIT is not COMMIT's $c"; exit 3; }
  CIRCUIT=$c
  main_line || {
    echo "can't read origin/main's first-parent history (fetch.probe.log), so COMMIT can't be checked on it"; exit 3; }
  grep -qxF "$COMMIT" "$OUT/main.probe.txt" || {
    echo "COMMIT $COMMIT is not on main's first-parent line, so no session would credit: pick a commit on it"; exit 3; }
  if [ -n "${DRY:-}" ]; then
    [ -f "$GATE_KEY" ] && echo "DRY: the gate key $GATE_KEY is here" || echo "DRY: no gate key at $GATE_KEY"
    ssh-keygen -F "${AUDITOR#*@}" -l 2>&1 | grep -q -F "$AUDITOR_FP" \
      && echo "DRY: the verifier host's host key $AUDITOR_FP is known here" \
      || echo "DRY: the verifier host's host key isn't in known_hosts here; the first request accepts it and checks $AUDITOR_FP"
  fi
  gate "$CHALLENGER" pubkey > "$OUT/challenger.pubkey.json" || { echo "pubkey failed"; exit 3; }
  gate "$CHALLENGER" pins > "$OUT/challenger.pins.jsonl" || { echo "pins failed"; exit 3; }
  gate "$CHALLENGER" ledger > "$OUT/ledger.probe.jsonl" || { echo "ledger failed"; exit 3; }
  gate "$AUDITOR" status > "$OUT/host.status.json" || { echo "the verifier host's status failed"; exit 3; }
  if [ -z "${DRY:-}" ]; then
    ssh-keygen -F "${AUDITOR#*@}" -l 2>&1 | tee -a "$OUT/hostkeys.txt" | grep -q -F "$AUDITOR_FP" \
      || { echo "the verifier host's host key isn't $AUDITOR_FP"; exit 3; }
  fi
  [ -n "${DRY:-}" ] && { echo "{\"dry\": true, \"circuit\": \"$CIRCUIT\", \"commit\": \"$COMMIT\"}" > "$OUT/probe.json"; return 0; }
  python3 - "$OUT" "${PIN:?}" "${PERIOD:?}" "$LANE" "$COMMIT" "$CIRCUIT" <<'EOF' || exit 3
import hashlib, json, sys
out, pin, period, lane, commit, circuit = sys.argv[1:]
why = []
key = json.load(open(f"{out}/challenger.pubkey.json"))
got = hashlib.sha512(bytes.fromhex(key["public"])).hexdigest()
if got != pin:
    why.append(f"the challenger's public key is {got[:16]}..., not PIN {pin[:16]}...")
pins = [json.loads(x) for x in open(f"{out}/challenger.pins.jsonl") if x.strip()]
entries = [json.loads(x) for x in open(f"{out}/ledger.probe.jsonl") if x.strip()]
used = [e["seq"] for e in entries if e["kind"] in ("period-open", "period-close") and e["body"].get("period") == period]
if used:
    why.append(f"period {period} is used (seq {used}): name a new PERIOD")
opened = {}
for e in entries:
    if e["kind"] == "period-open":
        opened[e["body"]["lane"]] = e["body"]["period"]
    elif e["kind"] == "period-close":
        opened = {k: v for k, v in opened.items() if v != e["body"]["period"]}
if lane in opened:
    why.append(f"lane {lane} has period {opened[lane]} open: close it first (gate.sh close, PERIOD={opened[lane]})")
main = {x.strip() for x in open(f"{out}/main.probe.txt") if x.strip()}
host = json.load(open(f"{out}/host.status.json"))
pinned = host.get("pinned")
if pinned != commit and pinned in main:
    why.append(f"the verifier host is pinned at {pinned}, not COMMIT {commit}: rerun with COMMIT={pinned}")
elif pinned != commit:
    why.append(f"the verifier host is pinned at {pinned}, off main's first-parent line, where no session credits: "
               f"repin it at COMMIT {commit} (vy-verifier-pin) and rerun")
if host.get("busy"):
    why.append(f"the verifier host runs {host['busy']}: one unit at a time")
doc = {"commit": commit, "circuit": circuit, "pin": pin, "period": period, "main_commits": len(main),
       "challenger_pin": pins[-1] if pins else None,
       "challenger_pinned_at_commit": bool(pins) and pins[-1].get("commit") == commit, "ledger_entries": len(entries),
       "host_pinned": pinned, "host_pinned_on_main": pinned in main, "host_pin_log": host.get("pins", [])[-1:],
       "host_busy": host.get("busy"), "refused": why}
json.dump(doc, open(f"{out}/probe.json", "w"), indent=1)
print(json.dumps(doc, indent=1))
if not doc["challenger_pinned_at_commit"]:
    print(f"note: the challenger host's last pin is {doc['challenger_pin']}, not COMMIT; pinned_session needs it on "
          "main's first-parent line, which auditor.sh checks")
sys.exit(3 if why else 0)
EOF
}

open_period() {
  gate "$CHALLENGER" ledger > "$OUT/ledger.before.jsonl"
  gate "$CHALLENGER" period-open --name "${PERIOD:?}" --cap "$CAP" --circuit "${CIRCUIT:?}" --model "$MODEL" \
    > "$OUT/period-open.out" 2> "$OUT/period-open.err"
  local r=$?
  echo "period-open rc $r"
  cat "$OUT/period-open.out" "$OUT/period-open.err"
  gate "$CHALLENGER" ledger > "$OUT/ledger.after.jsonl"
  tail -n 1 "$OUT/ledger.after.jsonl"
  return $r
}

host_run() {
  local words r run
  read -ra words <<< "$PACED"
  gate "$AUDITOR" run --peer "$PROVER_NODE" --port "$PORT" --period "${PERIOD:?}" --pin "${PIN:?}" --paced "${words[@]}" \
    --pace-pass-ms "$PACE_PASS_MS" --minutes "$MINUTES" --jitter-ms "$JITTER_MS" --quantile "$QUANTILE" \
    --model "$MODEL" --engines "$ENGINES" --arena-blocks "$ARENA_BLOCKS" > "$OUT/host-run.json" 2> "$OUT/host-run.err"
  r=$?
  echo "host run rc $r"
  cat "$OUT/host-run.json" "$OUT/host-run.err"
  [ -n "${DRY:-}" ] && return 0
  [ $r = 0 ] || return $r
  run=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["run"])' "$OUT/host-run.json") || return 3
  echo "$run" > "$OUT/host-run.txt"
  while :; do
    gate "$AUDITOR" status "$run" > "$OUT/host-status.last.json" 2>> "$OUT/host-status.err"
    python3 - "$OUT/host-status.last.json" >> "$OUT/host-status.jsonl" <<'EOF'
import json, sys, time
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    print(json.dumps({"t": time.time(), "error": str(e)}))
    sys.exit(0)
print(json.dumps({"t": time.time(), "active": d.get("active"), "listening": d.get("listening"),
                  "end": d.get("end") is not None, "log": (d.get("log") or [])[-3:]}))
EOF
    python3 -c 'import json, sys; sys.exit(0 if json.load(open(sys.argv[1])).get("end") is not None else 1)' \
      "$OUT/host-status.last.json" 2> /dev/null && break
    sleep 30
  done
  gate "$AUDITOR" fetch "$run" > "$OUT/host.tgz" 2> "$OUT/host-fetch.err" || { echo "fetch $run failed"; return 3; }
  mkdir -p "$OUT/host" && tar -xzf "$OUT/host.tgz" -C "$OUT/host" && rm "$OUT/host.tgz"
  r=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["end"]["rc"])' "$OUT/host-status.last.json")
  echo "host run $run rc $r"
  return "$r"
}

close_period() {
  gate "$CHALLENGER" period-close --name "${PERIOD:?}" > "$OUT/period-close.out" 2> "$OUT/period-close.err"
  local r=$?
  echo "period-close rc $r"
  cat "$OUT/period-close.out" "$OUT/period-close.err"
  gate "$CHALLENGER" ledger > "$OUT/ledger.closed.jsonl"
  return $r
}

leased() {
  # how many GPUs PROVER_RUN holds on PROVER_NODE and when its newest began ("N SINCE", SINCE - when it holds none), or
  # "gone" when it neither holds nor waits for any
  local to st
  read -ra to <<< "${PROVER_SSH:-$(grep "^$PROVER_NODE " /etc/vy/peers | cut -d' ' -f2-)}"
  [ ${#to[@]} -gt 0 ] || { echo "no $PROVER_NODE in /etc/vy/peers and no PROVER_SSH" >&2; return 3; }
  # no waiter leaves no wait file, so only the status decides the rc
  st=$(ssh -o BatchMode=yes -o ConnectTimeout=20 "${to[@]}" \
    "/workspace/pouw/infra/bin/gpu-lease status && { cat /run/gpu-lease/wait.* 2> /dev/null;
      python3 -c 'import json, sys; print(\"state=\" + json.load(open(sys.argv[1]))[\"state\"])' \
        /workspace/research/runs/${PROVER_RUN:?}/status.json 2> /dev/null; true; }" 2>&1) \
    || { tail -n 3 <<< "$st" >&2; return 3; }
  local lines held since
  lines=$(grep -E "^[0-9]+ held .* run=${PROVER_RUN:?}( |$)" <<< "$st")
  held=$(grep -c . <<< "$lines")
  since=$(grep -o 'since=[^ ]*' <<< "$lines" | cut -d= -f2 | sort | tail -n 1)
  # a lease waiting unqueued (out of a booked window) writes no wait file, so a run still running counts as waiting
  if [ "$held" = 0 ] && ! grep -q "run=$PROVER_RUN\b" <<< "$st" && ! grep -qx "state=running" <<< "$st"; then
    echo gone
  else
    echo "$held ${since:--}"
  fi
}

fire() {
  local l n since k fails=0 by=
  if [ -n "${LEASE_BY:-}" ]; then by=$(date -u -d "$LEASE_BY" +%s) || { echo "LEASE_BY $LEASE_BY isn't a time"; exit 2; }; fi
  echo "fire $(date -u +%FT%TZ): waiting for $PROVER_RUN to hold $ENGINES GPUs on $PROVER_NODE${by:+, leased by $LEASE_BY}"
  for ((k = 0; ; k++)); do
    if ! l=$(leased); then
      # one failed read waits for the next; ten running (five minutes) end the wait
      fails=$((fails + 1))
      [ $fails -lt 10 ] || { echo "can't read $PROVER_NODE's leases $fails times running: nothing opened"; exit 3; }
      sleep 30
      continue
    fi
    fails=0
    read -r n since <<< "$l"
    [ $((k % 20)) = 0 ] && echo "$(date -u +%FT%TZ) $PROVER_RUN holds $n${since:+ (since $since)}"
    [ -n "${DRY:-}" ] && { echo "DRY: $PROVER_RUN holds $n; would wait until it holds $ENGINES"; break; }
    [ "$n" = gone ] && { echo "$PROVER_RUN neither holds nor waits for a lease: nothing opened"; exit 3; }
    if [ "$n" -ge "$ENGINES" ]; then
      [ -z "$by" ] || [ "$(date -u -d "$since" +%s)" -le "$by" ] && break
      echo "$PROVER_RUN's lease began at $since, after LEASE_BY $LEASE_BY: nothing opened"
      exit 3
    fi
    if [ -n "$by" ] && [ "$(date -u +%s)" -gt $((by + 120)) ]; then
      echo "$PROVER_RUN holds $n of $ENGINES GPUs, 2 minutes past LEASE_BY $LEASE_BY: nothing opened"
      exit 3
    fi
    sleep 30
  done
  echo "$(date -u +%FT%TZ) leased"
  probe || exit 3
  open_period || exit 3
  host_run
  local r=$?
  if [ $r != 0 ]; then
    echo "the host run failed (rc $r): period $PERIOD stays open for a rerun in it; no close, no auditor"
    exit $r
  fi
  close_period || exit 3
  audit
}

audit() {
  local run=${HOST_RUN:-$OUT/host/$(cat "$OUT/host-run.txt" 2> /dev/null || echo DRY)}
  # PEER empty: the sessions' challenger peer is the host record's (the verifier host), not the prover's node
  if [ -n "${DRY:-}" ]; then
    echo "DRY: PEER='' GATE=$CHALLENGER GATE_KEY=$GATE_KEY RUN=$run PERIOD=$PERIOD PIN=$PIN COMMIT=$COMMIT" \
      "bash benchmarks/remote_session/auditor.sh"
    return 0
  fi
  PEER='' GATE=$CHALLENGER GATE_KEY=$GATE_KEY RUN=$run bash benchmarks/remote_session/auditor.sh
}

: > "$OUT/gate.requests"
case $VERB in
  probe) probe ;;
  open) open_period ;;
  host) host_run ;;
  close) close_period ;;
  audit) audit ;;
  fire) fire ;;
  *) echo "VERB is probe, open, host, close, audit or fire, not $VERB"; exit 2 ;;
esac
