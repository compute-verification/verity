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
# Two more forms of a share in SHARES: `RUN~KEEPER[@USER@HOST]=TAGS` reads RUN's tags as keep.sh's run KEEPER kept them
# (KEEP/RUN.TAG, removed after its verify), and `?AFTER[@HOSTS]=TAGS` (HOSTS: `USER@HOST` or `-` for this node, joined
# by `+`) finds, per tag, the earliest share started after AFTER (YYYYMMDD-HHMMSS) whose plan checks that tag; its rows'
# key is found.TAG. `RUN%SRC[@USER@HOST]=TAGS` reads RUN's tags from SRC's out/ (TAG/sessions, ready at its sessions.du or
# a harvester's "kept TAG's sessions" log line, and verifier/), with SRC's plan.json or share-plan.json as RUN's plan.
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
KEEP=${KEEP:-/workspace/research/proofs-mainv/keep}
pfx() { case $1 in \?*) echo found ;; *) local r=${1%%@*}; r=${r%%~*}; echo "${r%%%*}" ;; esac; }   # pfx SPEC: its keys' prefix
KEYS=()   # PREFIX.TAG, in SHARES' order
for s in $SHARES; do
  IFS=, read -ra tags <<< "${s#*=}"
  for tag in "${tags[@]}"; do KEYS+=("$(pfx "${s%%=*}").$tag"); done
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
snapshot() {   # snapshot HOST RUN TAG KEY: the tag's sessions and verifier inputs into $T/in/KEY
  local h=$1 run=$2 tag=$3 k=$4 P=$T/in/$4.partial d pub reg prog part
  d=$(field "$run" "$tag" dir) pub=$(field "$run" "$tag" pub) reg=$(field "$run" "$tag" registered)
  prog=$(field "$run" "$tag" program) part=$(field "$run" "$tag" partition)
  rm -rf "$P" && mkdir -p "$P/x" || return 1
  on "$h" "tar -C / -cf - ${d#/}/sessions ${pub#/} ${reg#/} ${prog#/} ${part#/}" | tar -xf - -C "$P/x" || return 1
  mv "$P/x/$d/sessions" "$P/sessions" && mv "$P/x$pub" "$P/pub.bin" && mv "$P/x$reg" "$P/registered.json" \
    && mv "$P/x$prog" "$P/program.json" && mv "$P/x$part" "$P/partition.json" && rm -rf "$P/x" || return 1
  (cd "$P" && find . -type f | sort | xargs sha256sum) > "$OUT/v/$k.inputs.sha256"
  mv "$P" "$T/in/$k"
}
read_plan() {   # read_plan HOST RUN PFX PLANPATH: the share's plan into plans/RUN.json; drops from $pending the tags it
  local h=$1 run=$2 x=$3 tag pin   # doesn't pin to WANT
  on "$h" "cat $4" > "$OUT/plans/$run.json.tmp" && mv "$OUT/plans/$run.json.tmp" "$OUT/plans/$run.json"
  on "$h" "cat $RUNS/$run/out/binaries.sha256" > "$OUT/plans/$run.binaries.sha256" 2>/dev/null
  for tag in $pending; do
    echo "$run $h" > "$OUT/v/$x.$tag.share"
    pin=$(field "$run" "$tag" circuit_sha512 2>/dev/null)
    if [ "$pin" != "$WANT" ]; then
      echo "the share's plan pins circuit ${pin:-(none)}, not $WANT" > "$T/in/$x.$tag.missing"
      log "collector $run: $tag's plan pins ${pin:-no circuit}; not verified"; pending=${pending/ $tag / }
    fi
  done
}
collect() {   # collect HOST RUN PFX TAG...: the share's plan, then each tag's snapshot once its session is complete
  local h=$1 run=$2 x=$3 SR=$RUNS/$2 tag ready pending end=$(( SECONDS + MAXWAIT ))
  shift 3
  pending=" $* "
  until on "$h" "test -s $SR/out/plan.json"; do
    if ! alive "$h" "$run" || [ $SECONDS -ge $end ]; then
      for tag in $pending; do echo "the share ended without a plan.json" > "$T/in/$x.$tag.missing"; done
      log "collector $run: no plan.json; no session for:$pending"; return 0
    fi
    sleep $(( POLL * 2 ))
  done
  read_plan "$h" "$run" "$x" "$SR/out/plan.json"
  log "collector $run: plan read; waiting on:$pending"
  while :; do
    ready=$(on "$h" "ls $SR/out/*/sessions.du 2>/dev/null")
    for tag in $pending; do
      if grep -qx "$SR/out/$tag/sessions.du" <<< "$ready"; then
        if snapshot "$h" "$run" "$tag" "$x.$tag"; then
          log "snapshot $run $tag: $(ls "$T/in/$x.$tag/sessions" | tr '\n' ' ')"; pending=${pending/ $tag / }
        else log "snapshot $run $tag FAILED; retrying"; fi
      fi
    done
    [ -z "${pending// /}" ] && return 0
    if ! alive "$h" "$run"; then
      for tag in $pending; do echo "the share ended without a complete session" > "$T/in/$x.$tag.missing"; done
      log "collector $run: the share ended; no session for:$pending"; return 0
    fi
    sleep "$POLL"
  done
}
collect_kept() {   # collect_kept HOST RUN KEEPER TAG...: RUN's tags as keep.sh's run KEEPER kept them in KEEP
  local h=$1 run=$2 kr=$3 tag k P pending dead
  shift 3
  pending=" $* "
  until on "$h" "test -s $KEEP/$run.plan.json"; do
    if ! alive "$h" "$kr"; then
      for tag in $pending; do echo "the keeper $kr ended without the share's plan" > "$T/in/$run.$tag.missing"; done
      log "collector $run~$kr: no plan kept; no session for:$pending"; return 0
    fi
    sleep $(( POLL * 2 ))
  done
  read_plan "$h" "$run" "$run" "$KEEP/$run.plan.json"
  log "collector $run~$kr: plan read; waiting on:$pending"
  while :; do
    alive "$h" "$kr" && dead=0 || dead=1   # read before the pass: the keeper leaves each tag kept or missing first
    for tag in $pending; do
      k=$run.$tag P=$T/in/$run.$tag.partial
      if on "$h" "test -d $KEEP/$k"; then
        rm -rf "$P" && mkdir -p "$P"
        if on "$h" "tar -C $KEEP/$k -cf - ." | tar -xf - -C "$P" \
            && (cd "$P" && find . -type f | sort | xargs sha256sum) > "$OUT/v/$k.inputs.sha256" \
            && on "$h" "cat $KEEP/$k.inputs.sha256" | cmp -s - "$OUT/v/$k.inputs.sha256"; then
          echo "${h:--} $KEEP/$k" > "$T/in/$k.kept" && mv "$P" "$T/in/$k"
          log "snapshot $run $tag (kept by $kr): $(ls "$T/in/$k/sessions" | tr '\n' ' ')"; pending=${pending/ $tag / }
        else log "snapshot $run $tag (kept by $kr) FAILED or differs from the keeper's hashes; retrying"; fi
      elif on "$h" "test -e $KEEP/$k.missing"; then
        on "$h" "cat $KEEP/$k.missing" > "$T/in/$k.missing"; pending=${pending/ $tag / }
      fi
    done
    [ -z "${pending// /}" ] && return 0
    if [ $dead = 1 ]; then
      for tag in $pending; do echo "the keeper $kr ended without keeping it" > "$T/in/$run.$tag.missing"; done
      log "collector $run~$kr: the keeper ended; no session for:$pending"; return 0
    fi
    sleep "$POLL"
  done
}
collect_out() {   # collect_out HOST RUN SRC TAG...: RUN's tags from SRC's out/ (a split share's own, or a harvester's):
  local h=$1 run=$2 src=$3 S=$RUNS/$3/out tag k P pending dead plan ready   # TAG/sessions and verifier/
  shift 3
  pending=" $* "
  local end=$(( SECONDS + MAXWAIT ))
  until plan=$(on "$h" "ls $S/plan.json $S/share-plan.json 2>/dev/null | head -1") && [ -n "$plan" ]; do
    # a share still in its node's memory queue has a request and no status.json yet
    if { ! alive "$h" "$src" && ! on "$h" "test ! -e $RUNS/$src/status.json -a -e $RUNS/../requests/$src/request.json"; } \
        || [ $SECONDS -ge $end ]; then
      for tag in $pending; do echo "$src ended without a plan" > "$T/in/$run.$tag.missing"; done
      log "collector $run%$src: no plan; no session for:$pending"; return 0
    fi
    sleep $(( POLL * 2 ))
  done
  read_plan "$h" "$run" "$run" "$plan"
  log "collector $run%$src: plan read ($plan); waiting on:$pending"
  while :; do
    alive "$h" "$src" && dead=0 || dead=1   # read before the pass: SRC marks each tag before it exits
    ready=$(on "$h" "ls $S/*/sessions.du 2>/dev/null; grep -o \"kept [^ ]*'s sessions\" $S/job.log 2>/dev/null")
    for tag in $pending; do
      grep -qx -e "$S/$tag/sessions.du" -e "kept $tag's sessions" <<< "$ready" || continue
      k=$run.$tag P=$T/in/$run.$tag.partial
      rm -rf "$P" && mkdir -p "$P"
      if on "$h" "cd $S && tar -cf - $tag/sessions verifier/$(basename "$(field "$run" "$tag" pub)") \
            verifier/$(basename "$(field "$run" "$tag" registered)") verifier/$(basename "$(field "$run" "$tag" program)") \
            verifier/$(basename "$(field "$run" "$tag" partition)")" | tar -xf - -C "$P" \
          && mv "$P/$tag/sessions" "$P/sessions" \
          && mv "$P/verifier/$(basename "$(field "$run" "$tag" pub)")" "$P/pub.bin" \
          && mv "$P/verifier/$(basename "$(field "$run" "$tag" registered)")" "$P/registered.json" \
          && mv "$P/verifier/$(basename "$(field "$run" "$tag" program)")" "$P/program.json" \
          && mv "$P/verifier/$(basename "$(field "$run" "$tag" partition)")" "$P/partition.json" \
          && rmdir "$P/$tag" "$P/verifier"; then
        (cd "$P" && find . -type f | sort | xargs sha256sum) > "$OUT/v/$k.inputs.sha256" && mv "$P" "$T/in/$k"
        log "snapshot $run $tag (from $src's out): $(ls "$T/in/$k/sessions" | tr '\n' ' ')"; pending=${pending/ $tag / }
      else log "snapshot $run $tag (from $src's out) FAILED; retrying"; fi
    done
    [ -z "${pending// /}" ] && return 0
    if [ $dead = 1 ]; then
      for tag in $pending; do echo "$src ended without the tag's sessions" > "$T/in/$run.$tag.missing"; done
      log "collector $run%$src: $src ended; no session for:$pending"; return 0
    fi
    sleep "$POLL"
  done
}
FINDPY='import json, os, sys
R, after, tag = sys.argv[1:4]
for r in sorted(os.listdir(R)):
    if not r.startswith("r") or r <= "r" + after:
        continue
    try:
        p = json.load(open(f"{R}/{r}/out/plan.json"))
        st = json.load(open(f"{R}/{r}/status.json"))["state"]
    except Exception:
        continue
    if tag in p.get("checks", {}) and (st == "running" or os.path.exists(f"{R}/{r}/out/{tag}/sessions.du")):
        print(r)
        break'
collect_found() {   # collect_found AFTER HOSTS TAG: the earliest share after AFTER on HOSTS checking TAG, then collect
  local after=$1 hosts=$2 tag=$3 hh h r end=$(( SECONDS + MAXWAIT ))
  log "finder $tag: waiting for a share started after $after on ${hosts//+/ or }"
  while :; do
    for hh in ${hosts//+/ }; do
      h=$hh && [ "$h" = - ] && h=
      r=$(on "$h" "python3 -c $(printf %q "$FINDPY") $RUNS $after $tag")
      if [ -n "$r" ]; then log "finder $tag: $r on $hh"; collect "$h" "$r" found "$tag"; return 0; fi
    done
    if [ $SECONDS -ge $end ]; then
      echo "no share started after $after checked $tag in time" > "$T/in/found.$tag.missing"
      log "finder $tag: no share in time"; return 0
    fi
    sleep $(( POLL * 2 ))
  done
}
for s in $SHARES; do
  spec=${s%%=*} tl=${s#*=} h=
  case $spec in
    \?*) a=${spec%%@*} hs=-
         [ "$spec" != "$a" ] && hs=${spec#*@}
         for tag in ${tl//,/ }; do collect_found "${a#\?}" "$hs" "$tag" & CPIDS+=($!); done ;;
    *)   run=${spec%%@*}
         [ "$spec" != "$run" ] && h=${spec#*@}
         if [ "$run" != "${run%%~*}" ]; then collect_kept "$h" "${run%%~*}" "${run#*~}" ${tl//,/ } &
         elif [ "$run" != "${run%%%*}" ]; then collect_out "$h" "${run%%%*}" "${run#*%}" ${tl//,/ } &
         else collect "$h" "$run" "$run" ${tl//,/ } & fi
         CPIDS+=($!) ;;
  esac
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
  if [ -e "$I.kept" ]; then read -r kh kd < "$I.kept"; [ "$kh" = - ] && kh=; on "$kh" "rm -rf $kd"; rm -f "$I.kept"; fi
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
for k in "${KEYS[@]}"; do
  [ -e "$OUT/v/$k.share" ] || continue
  read -r run h < "$OUT/v/$k.share"
  tag=${k#*.}
  until [ -e "$OUT/v/$k.missing" ] || on "$h" "test -e $RUNS/$run/out/lean/$tag.time"; do
    alive "$h" "$run" || break
    sleep $(( POLL * 2 ))
  done
  on "$h" "cat $RUNS/$run/out/lean/$tag.out" > "$OUT/share-lean/$k.out" 2>/dev/null
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
    share = o / "v" / f"{k}.share"
    if share.exists():
        run = share.read_text().split()[0]
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
