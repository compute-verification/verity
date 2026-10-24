#!/bin/bash
# PRF-131 mainv: PRF-131 shares' K=4096 statements re-verified by flock-verify built from this run's tree (origin/main),
# beside the shares' own Lean checks, without touching the shares.
#   research run --on NODE --project verity --source . --cwd source --mem-gb 130 --timeout 960m --custody-ttl 24h \
#     --send mainv.sh --env SHARES='RUN[@USER@HOST]=TAG,TAG ...' --env CIRCUIT=[USER@HOST:]PATH --env WANT_SHA512=... \
#     -- bash -c 'exec bash $RESEARCH_RUN_DIR/inputs/mainv.sh'
# One collector per share (on this node, or over ssh at USER@HOST) snapshots each of its tags' sessions once the share
# marks the session complete (its out/TAG/sessions.du, written after its prover and server exit; the share's scratch lives
# until it exits) with the tag's public file, registered reads, program and partition from the share's plan.json, into
# this run's scratch. The verifier runs main's flock-verify on each snapshot in turn. The circuit: CIRCUIT read in place
# (waited for, then held open on fd 7 so no removal can free it), or with USER@HOST: a copy pulled from there. Its
# SHA-512 must be WANT_SHA512, and every share's plan must pin it, before every verify and at the end. Each tag's row,
# with its share's own verdict: out/summary.{json,md}.
set -uo pipefail
OUT=${RESEARCH_RUN_DIR:?}/out
mkdir -p "$OUT/v" "$OUT/plans" "$OUT/share-lean"
log() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$OUT/job.log"; }
SHARES=${SHARES:?} WANT=${WANT_SHA512:?} CIRCUIT=${CIRCUIT:?} MAXWAIT=${MAXWAIT:-57600} POLL=${MAINV_POLL:-30}
RUNS=${SHARE_RUNS:-/workspace/research/runs} SCR=${MAINV_TMP:-/workspace/tmp}
COMMIT=${RESEARCH_SOURCE_SHA:-$(git rev-parse HEAD 2>/dev/null || basename "$PWD")}
on() {   # on HOST CMD: CMD on HOST (empty: this node)
  if [ -n "$1" ]; then
    ssh -i ~/.ssh/vy-cluster -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
      -o ServerAliveInterval=30 "$1" "$2"
  else bash -c "$2"; fi
}
alive() {   # alive HOST RUN
  [ "$(on "$1" "python3 -c 'import json; print(json.load(open(\"$RUNS/$2/status.json\"))[\"state\"])'")" = running ]
}
KEYS=()   # RUN.TAG, in SHARES' order
for s in $SHARES; do
  spec=${s%%=*} run=${spec%%@*}
  IFS=, read -ra tags <<< "${s#*=}"
  for tag in "${tags[@]}"; do KEYS+=("$run.$tag"); done
done
log "source $COMMIT; shares: $SHARES; circuit $CIRCUIT; pin $WANT"
mkdir -p "$SCR" && T=$(mktemp -d "$SCR/mainv.XXXXXX") || exit 1
mkdir -p "$T/in"
CPIDS=()
trap 'for p in "${CPIDS[@]}"; do kill "$p" 2>/dev/null; done; rm -rf "$T"' EXIT

export PATH="$HOME/.elan/bin:$PATH"
C=${HZ_CACHE:-/workspace/research/cache/pouw-hidden-zk} && mkdir -p "$C"
key=$(find verity/core/service/flock -type f -not -path '*/.lake/*' -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-16)
D=$C/lean-$key
( flock 9; [ -e "$D/.done" ] || { rm -rf "$D.tmp" && mkdir -p "$D.tmp" \
    && (cd verity/core/service/flock && tar --exclude=.lake -cf - .) | tar -xf - -C "$D.tmp" && (cd "$D.tmp" && lake build) \
    && rm -rf "$D" && mv "$D.tmp" "$D" && touch "$D/.done"; } ) 9> "$C/.lock-lean-$key" > "$OUT/build-lean.log" 2>&1
LEAN=$D/.lake/build/bin/flock-verify
[ -x "$LEAN" ] || { tail -30 "$OUT/build-lean.log"; log "build FAILED"; exit 1; }
BINSHA=$(sha256sum "$LEAN" | cut -d' ' -f1)
echo "$BINSHA  $LEAN" | tee "$OUT/binaries.sha256" | tee -a "$OUT/job.log"

field() {   # field RUN TAG KEY: the share's plan's check field
  python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['checks'][sys.argv[2]][sys.argv[3]])" "$OUT/plans/$1.json" "$2" "$3"
}
snapshot() {   # snapshot HOST RUN TAG: the tag's sessions and verifier inputs into $T/in/RUN.TAG
  local h=$1 run=$2 tag=$3 P=$T/in/$2.$3.partial d pub reg prog part
  d=$(field "$run" "$tag" dir) pub=$(field "$run" "$tag" pub) reg=$(field "$run" "$tag" registered)
  prog=$(field "$run" "$tag" program) part=$(field "$run" "$tag" partition)
  rm -rf "$P" && mkdir -p "$P/x" || return 1
  on "$h" "tar -C / -cf - ${d#/}/sessions ${pub#/} ${reg#/} ${prog#/} ${part#/}" | tar -xf - -C "$P/x" || return 1
  mv "$P/x/$d/sessions" "$P/sessions" && mv "$P/x$pub" "$P/pub.bin" && mv "$P/x$reg" "$P/registered.json" \
    && mv "$P/x$prog" "$P/program.json" && mv "$P/x$part" "$P/partition.json" && rm -rf "$P/x" || return 1
  (cd "$P" && find . -type f | sort | xargs sha256sum) > "$OUT/v/$run.$tag.inputs.sha256"
  mv "$P" "$T/in/$run.$tag"
}
collect() {   # collect HOST RUN TAG...: the share's plan, then each tag's snapshot once its session is complete
  local h=$1 run=$2 SR=$RUNS/$2 tag ready pending end=$(( SECONDS + MAXWAIT ))
  shift 2
  pending=" $* "
  until on "$h" "test -s $SR/out/plan.json"; do
    if ! alive "$h" "$run" || [ $SECONDS -ge $end ]; then
      for tag in $pending; do echo "the share ended without a plan.json" > "$T/in/$run.$tag.missing"; done
      log "collector $run: no plan.json; no session for:$pending"; return 0
    fi
    sleep $(( POLL * 2 ))
  done
  on "$h" "cat $SR/out/plan.json" > "$OUT/plans/$run.json.tmp" && mv "$OUT/plans/$run.json.tmp" "$OUT/plans/$run.json"
  on "$h" "cat $SR/out/binaries.sha256" > "$OUT/plans/$run.binaries.sha256" 2>/dev/null
  for tag in $pending; do
    local pin
    pin=$(field "$run" "$tag" circuit_sha512 2>/dev/null)
    if [ "$pin" != "$WANT" ]; then
      echo "the share's plan pins circuit ${pin:-(none)}, not $WANT" > "$T/in/$run.$tag.missing"
      log "collector $run: $tag's plan pins ${pin:-no circuit}; not verified"; pending=${pending/ $tag / }
    fi
  done
  log "collector $run: plan read; waiting on:$pending"
  while :; do
    ready=$(on "$h" "ls $SR/out/*/sessions.du 2>/dev/null")
    for tag in $pending; do
      if grep -qx "$SR/out/$tag/sessions.du" <<< "$ready"; then
        if snapshot "$h" "$run" "$tag"; then
          log "snapshot $run $tag: $(ls "$T/in/$run.$tag/sessions" | tr '\n' ' ')"; pending=${pending/ $tag / }
        else log "snapshot $run $tag FAILED; retrying"; fi
      fi
    done
    [ -z "${pending// /}" ] && return 0
    if ! alive "$h" "$run"; then
      for tag in $pending; do echo "the share ended without a complete session" > "$T/in/$run.$tag.missing"; done
      log "collector $run: the share ended; no session for:$pending"; return 0
    fi
    sleep "$POLL"
  done
}
for s in $SHARES; do
  spec=${s%%=*} run=${spec%%@*} h=
  [ "$spec" != "$run" ] && h=${spec#*@}
  tl=${s#*=}
  collect "$h" "$run" ${tl//,/ } &
  CPIDS+=($!)
done

case $CIRCUIT in
  *:*) CH=${CIRCUIT%%:*} CP=${CIRCUIT#*:}
       log "pulling the circuit $CP from $CH"
       until on "$CH" "test -s $CP"; do sleep $(( POLL * 2 )); done
       on "$CH" "cat $CP" > "$T/circuit.txt" || { log "circuit pull FAILED"; exit 4; }
       exec 7< "$T/circuit.txt" ;;
  *)   CP=$CIRCUIT end=$(( SECONDS + MAXWAIT ))
       until [ -s "$CP" ]; do [ $SECONDS -lt $end ] || { log "no circuit at $CP in time"; exit 4; }; sleep $(( POLL * 2 )); done
       exec 7< "$CP" ;;
esac
CIRC=/dev/fd/7   # each child opens its inherited fd 7 afresh
hash_circuit() { sha512sum "$CIRC" | cut -d' ' -f1; }
got=$(hash_circuit)
log "circuit $CIRCUIT ($(stat -L -c %s "$CIRC") B, read via fd 7): sha512 $got"
[ "$got" = "$WANT" ] || { log "REFUSED: the circuit is not the pinned one; no verify"; exit 5; }
echo "$got" > "$OUT/circuit.sha512.before"

TIMEV=$(command -v /usr/bin/time >/dev/null && echo "/usr/bin/time -v")
verify() {   # verify RUN.TAG
  local k=$1 I=$T/in/$1 t0=$SECONDS s h
  h=$(hash_circuit)
  echo "$h" > "$OUT/v/$k.circuit.sha512"
  [ "$h" = "$WANT" ] || { log "verify $k: REFUSED to run, the circuit's sha512 is $h"; echo 5 > "$OUT/v/$k.rc"; return; }
  s=$(ls -d "$I"/sessions/*/ | tr '\n' ' ')
  log "verify $k: $LEAN verify --zk --circuit $CIRC --statement verity/flock-circuit --public $I/pub.bin" \
      "--partition $I/partition.json --program $I/program.json --registered $I/registered.json --session $s"
  # shellcheck disable=SC2086
  $TIMEV "$LEAN" verify --zk --circuit "$CIRC" --statement verity/flock-circuit --public "$I/pub.bin" \
    --partition "$I/partition.json" --program "$I/program.json" --registered "$I/registered.json" --session $s \
    > "$OUT/v/$k.out" 2> "$OUT/v/$k.err"
  echo $? > "$OUT/v/$k.rc"
  log "verify $k: exit $(cat "$OUT/v/$k.rc"), $(( SECONDS - t0 )) s:" \
      "$(grep -h VERDICT "$OUT/v/$k.out" | tail -1 | grep -o '"accepted":[a-z]*\|"why":"[^"]*"' | tr '\n' ' ')"
  rm -rf "$I"
}
while :; do
  cdead=1
  for p in "${CPIDS[@]}"; do kill -0 "$p" 2>/dev/null && cdead=0; done   # read before the pass, which then sees all they left
  left=0
  for k in "${KEYS[@]}"; do
    [ -e "$OUT/v/$k.rc" ] || [ -e "$OUT/v/$k.missing" ] && continue
    if [ -d "$T/in/$k" ]; then verify "$k"
    elif [ -e "$T/in/$k.missing" ]; then cp "$T/in/$k.missing" "$OUT/v/$k.missing"
    else left=$(( left + 1 )); fi
  done
  [ $left = 0 ] && break
  [ $cdead = 1 ] && { log "the collectors exited with $left tags unseen"; break; }
  sleep "$POLL"
done
wait
hash_circuit > "$OUT/circuit.sha512.after"

log "waiting for the shares' own verdicts (their out/lean/TAG.time), while they run"
for s in $SHARES; do
  spec=${s%%=*} run=${spec%%@*} h=
  [ "$spec" != "$run" ] && h=${spec#*@}
  tl=${s#*=}
  for tag in ${tl//,/ }; do
    until [ -e "$OUT/v/$run.$tag.missing" ] || on "$h" "test -e $RUNS/$run/out/lean/$tag.time"; do
      alive "$h" "$run" || break
      sleep $(( POLL * 2 ))
    done
    on "$h" "cat $RUNS/$run/out/lean/$tag.out" > "$OUT/share-lean/$run.$tag.out" 2>/dev/null
  done
done

python3 - "$OUT" "$COMMIT" "$BINSHA" "$WANT" "${KEYS[@]}" <<'EOF'
import json, sys
from pathlib import Path
o, commit, binsha, want, *keys = sys.argv[1:]
o = Path(o)
def verdict(p):
    v = [l for l in p.read_text().splitlines() if l.startswith("VERDICT")] if p.exists() else []
    return json.loads(v[-1].split(" ", 1)[1]) if v else None
def text(p):
    return p.read_text().strip() if p.exists() else None
before, after = text(o / "circuit.sha512.before"), text(o / "circuit.sha512.after")
rows = []
for k in keys:
    run, tag = k.split(".", 1)
    plan = o / "plans" / f"{run}.json"
    check = json.loads(plan.read_text())["checks"].get(tag) if plan.exists() else None
    expect = None if check is None else ("ACCEPT" if check["expect"] == "ACCEPT" else "REFUSE")
    v, sv = verdict(o / "v" / f"{k}.out"), verdict(o / "share-lean" / f"{k}.out")
    got = None if v is None else ("ACCEPT" if v.get("accepted") is True else "REFUSE")
    rc = text(o / "v" / f"{k}.rc")
    csha = text(o / "v" / f"{k}.circuit.sha512")
    rows.append({"tag": tag, "share_run": run, "expect": expect, "verdict": got, "why": v and v.get("why"),
                 "rc": None if rc is None else int(rc), "missing": text(o / "v" / f"{k}.missing"),
                 "share_verdict": None if sv is None else ("ACCEPT" if sv.get("accepted") is True else "REFUSE"),
                 "binary_sha256": binsha, "source_commit": commit, "circuit_sha512": csha,
                 "session": Path(v["session"].rstrip("/")).name if v and isinstance(v.get("session"), str) else None,
                 "ok": got is not None and got == expect and csha == want})
s = {"source_commit": commit, "binary_sha256": binsha, "circuit_sha512_pin": want, "circuit_sha512_before": before,
     "circuit_sha512_after": after, "rows": rows, "ok": all(r["ok"] for r in rows) and before == after == want}
(o / "summary.json").write_text(json.dumps(s, indent=1))
md = ["| lane | session run | session | expect | main verdict | share verdict | rc | binary sha256 | source commit | circuit sha512 |",
      "|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    md.append(f"| {r['tag']} | {r['share_run']} | {r['session']} | {r['expect']} | {r['verdict'] or r['missing']} | "
              f"{r['share_verdict']} | {r['rc']} | {binsha[:16]}… | {commit[:10]} | {(r['circuit_sha512'] or '-')[:16]}… |")
(o / "summary.md").write_text("\n".join(md) + "\n")
print("SUMMARY " + json.dumps({"ok": s["ok"], "rows": [(r["share_run"], r["tag"], r["verdict"], r["expect"]) for r in rows]}))
sys.exit(0 if s["ok"] else 1)
EOF
rc=$?
log "done (exit $rc)"
exit $rc
