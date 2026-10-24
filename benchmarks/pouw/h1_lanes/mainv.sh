#!/bin/bash
# PRF-131 mainv: one PRF-131 share's K=4096 statements re-verified by flock-verify built from this run's tree (origin/main),
# beside the share's own Lean checks (its 4a555ddc1d-era verifier), without touching the share.
#   research run --on NODE --project verity --source . --cwd source --mem-gb 130 --timeout 16h --send mainv.sh \
#     --env SHARE_RUN=RUN --env SHARE_HOST=[user@ip] --env TAGS='lane0 ...' --env WANT_SHA512=... \
#     -- bash -c 'exec bash $RESEARCH_RUN_DIR/inputs/mainv.sh'
# A collector snapshots each tag's session once the share marks it complete (its out/TAG/sessions.du, written after its
# prover and server exit; the share's scratch lives until it exits, at least one Lean check later) with the tag's public
# file, registered reads, program and partition from the share's plan.json, into this run's scratch. The verifier runs
# main's flock-verify on each snapshot. The circuit: with SHARE_HOST empty, the share's own file read in place, held open
# (fd 7) so the share's removal of its scratch can't free it; else a copy pulled from SHARE_HOST. Its SHA-512 must be
# WANT_SHA512 before every verify and at the end. Then each tag's row with the share's own verdict: out/summary.{json,md}.
set -uo pipefail
OUT=${RESEARCH_RUN_DIR:?}/out
mkdir -p "$OUT/v"
log() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$OUT/job.log"; }
RUN=${SHARE_RUN:?} H=${SHARE_HOST:-} WANT=${WANT_SHA512:?} TAGS=${TAGS:?} MAXWAIT=${MAXWAIT:-57600} POLL=${MAINV_POLL:-30}
SR=${SHARE_RUNS:-/workspace/research/runs}/$RUN SCR=${MAINV_TMP:-/workspace/tmp}
COMMIT=${RESEARCH_SOURCE_SHA:-$(git rev-parse HEAD 2>/dev/null || basename "$PWD")}
on() {   # on CMD: CMD on the share's node
  if [ -n "$H" ]; then
    ssh -i ~/.ssh/vy-cluster -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
      -o ServerAliveInterval=30 "$H" "$1"
  else bash -c "$1"; fi
}
alive() { [ "$(on "python3 -c 'import json; print(json.load(open(\"$SR/status.json\"))[\"state\"])'")" = running ]; }
log "source $COMMIT; share $RUN on ${H:-this node}; tags: $TAGS; pin $WANT"
mkdir -p "$SCR" && T=$(mktemp -d "$SCR/mainv.XXXXXX") || exit 1
CPID=
trap '[ -n "$CPID" ] && kill $CPID 2>/dev/null; rm -rf "$T"' EXIT

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
on "cat $SR/out/binaries.sha256" > "$OUT/share-binaries.sha256" 2>/dev/null

log "waiting for the share's plan.json (its prepare done)"
end=$(( SECONDS + MAXWAIT ))
until on "test -s $SR/out/plan.json"; do
  alive || { log "the share ended without a plan.json"; exit 4; }
  [ $SECONDS -lt $end ] || { log "no plan.json in time"; exit 4; }
  sleep $(( POLL * 2 ))
done
on "cat $SR/out/plan.json" > "$OUT/share-plan.json" || exit 4
field() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['checks'][sys.argv[2]][sys.argv[3]])" "$OUT/share-plan.json" "$1" "$2"; }
for tag in $TAGS; do
  [ "$(field "$tag" circuit_sha512)" = "$WANT" ] || { log "REFUSED: $tag's plan pins $(field "$tag" circuit_sha512), not $WANT"; exit 5; }
done
first=${TAGS%% *}
CP=$(on "readlink -f $(field "$first" dir)/circuit.txt")
if [ -n "$H" ]; then
  log "pulling the circuit $CP from $H"
  on "cat $CP" > "$T/circuit.txt" || { log "circuit pull FAILED"; exit 4; }
  exec 7< "$T/circuit.txt"
else
  exec 7< "$CP" || { log "no circuit at $CP"; exit 4; }
fi
CIRC=/dev/fd/7   # each child opens its inherited fd 7 afresh
hash_circuit() { sha512sum "$CIRC" | cut -d' ' -f1; }
got=$(hash_circuit)
log "circuit $CP ($(stat -L -c %s "$CIRC") B, ${H:+copied, }read via fd 7): sha512 $got"
[ "$got" = "$WANT" ] || { log "REFUSED: the circuit is not the pinned one; no verify"; exit 5; }
echo "$got" > "$OUT/circuit.sha512.before"

snapshot() {   # snapshot TAG: the tag's sessions and verifier inputs into $T/in/TAG
  local tag=$1 P=$T/in/$1.partial d pub reg prog part
  d=$(field "$tag" dir) pub=$(field "$tag" pub) reg=$(field "$tag" registered) prog=$(field "$tag" program) part=$(field "$tag" partition)
  rm -rf "$P" && mkdir -p "$P/x" || return 1
  on "tar -C / -cf - ${d#/}/sessions ${pub#/} ${reg#/} ${prog#/} ${part#/}" | tar -xf - -C "$P/x" || return 1
  mv "$P/x/$d/sessions" "$P/sessions" && mv "$P/x$pub" "$P/pub.bin" && mv "$P/x$reg" "$P/registered.json" \
    && mv "$P/x$prog" "$P/program.json" && mv "$P/x$part" "$P/partition.json" && rm -rf "$P/x" || return 1
  (cd "$P" && find . -type f | sort | xargs sha256sum) > "$OUT/v/$tag.inputs.sha256"
  mv "$P" "$T/in/$tag"
}
collect() {
  local pending=" $TAGS " ready tag
  while :; do
    ready=$(on "ls $SR/out/*/sessions.du 2>/dev/null")
    for tag in $pending; do
      if grep -qx "$SR/out/$tag/sessions.du" <<< "$ready"; then
        if snapshot "$tag"; then log "snapshot $tag: $(ls "$T/in/$tag/sessions" | tr '\n' ' ')"; pending=${pending/ $tag / }
        else log "snapshot $tag FAILED; retrying"; fi
      fi
    done
    [ -z "${pending// /}" ] && return 0
    if ! alive; then
      for tag in $pending; do echo "the share ended without a complete session" > "$T/in/$tag.missing"; done
      log "collector: the share ended; no session for:$pending"; return 0
    fi
    sleep "$POLL"
  done
}
mkdir -p "$T/in"
collect &
CPID=$!

TIMEV=$(command -v /usr/bin/time >/dev/null && echo "/usr/bin/time -v")
verify() {   # verify TAG
  local tag=$1 I=$T/in/$1 t0=$SECONDS s h
  h=$(hash_circuit)
  echo "$h" > "$OUT/v/$tag.circuit.sha512"
  [ "$h" = "$WANT" ] || { log "verify $tag: REFUSED to run, the circuit's sha512 is $h"; echo 5 > "$OUT/v/$tag.rc"; return; }
  s=$(ls -d "$I"/sessions/*/ | tr '\n' ' ')
  log "verify $tag: $LEAN verify --zk --circuit $CIRC --statement verity/flock-circuit --public $I/pub.bin" \
      "--partition $I/partition.json --program $I/program.json --registered $I/registered.json --session $s"
  # shellcheck disable=SC2086
  $TIMEV "$LEAN" verify --zk --circuit "$CIRC" --statement verity/flock-circuit --public "$I/pub.bin" \
    --partition "$I/partition.json" --program "$I/program.json" --registered "$I/registered.json" --session $s \
    > "$OUT/v/$tag.out" 2> "$OUT/v/$tag.err"
  echo $? > "$OUT/v/$tag.rc"
  log "verify $tag: exit $(cat "$OUT/v/$tag.rc"), $(( SECONDS - t0 )) s:" \
      "$(grep -h VERDICT "$OUT/v/$tag.out" | tail -1 | grep -o '"accepted":[a-z]*\|"why":"[^"]*"' | tr '\n' ' ')"
  rm -rf "$I"
}
while :; do
  cdead=0
  kill -0 $CPID 2>/dev/null || cdead=1   # read before the pass, so the pass sees all the collector left
  left=0
  for tag in $TAGS; do
    [ -e "$OUT/v/$tag.rc" ] || [ -e "$OUT/v/$tag.missing" ] && continue
    if [ -d "$T/in/$tag" ]; then verify "$tag"
    elif [ -e "$T/in/$tag.missing" ]; then cp "$T/in/$tag.missing" "$OUT/v/$tag.missing"
    else left=$(( left + 1 )); fi
  done
  [ $left = 0 ] && break
  [ $cdead = 1 ] && { log "the collector exited with $left tags unseen"; break; }
  sleep "$POLL"
done
wait $CPID 2>/dev/null
hash_circuit > "$OUT/circuit.sha512.after"

log "waiting for the share's own verdicts (its out/lean/TAG.time), while it runs"
share_done() {
  local tag
  for tag in $TAGS; do [ -e "$OUT/v/$tag.missing" ] || on "test -e $SR/out/lean/$tag.time" || return 1; done
}
until share_done; do
  alive || break
  sleep $(( POLL * 2 ))
done
mkdir -p "$OUT/share-lean"
for tag in $TAGS; do on "cat $SR/out/lean/$tag.out" > "$OUT/share-lean/$tag.out" 2>/dev/null; done

python3 - "$OUT" "$RUN" "$COMMIT" "$BINSHA" "$WANT" "$TAGS" <<'EOF'
import json, sys
from pathlib import Path
o, run, commit, binsha, want, tags = sys.argv[1:]
o = Path(o)
plan = json.loads((o / "share-plan.json").read_text())["checks"]
def verdict(p):
    v = [l for l in p.read_text().splitlines() if l.startswith("VERDICT")] if p.exists() else []
    return json.loads(v[-1].split(" ", 1)[1]) if v else None
before, after = (o / "circuit.sha512.before").read_text().split()[0], (o / "circuit.sha512.after").read_text().split()[0]
rows = []
for tag in tags.split():
    v, sv = verdict(o / "v" / f"{tag}.out"), verdict(o / "share-lean" / f"{tag}.out")
    rc = o / "v" / f"{tag}.rc"
    expect = "ACCEPT" if plan[tag]["expect"] == "ACCEPT" else "REFUSE"
    got = None if v is None else ("ACCEPT" if v.get("accepted") is True else "REFUSE")
    sess = Path(v["session"].rstrip("/")).name if v and isinstance(v.get("session"), str) else None
    csha = (o / "v" / f"{tag}.circuit.sha512").read_text().strip() if (o / "v" / f"{tag}.circuit.sha512").exists() else None
    rows.append({"tag": tag, "share_run": run, "expect": expect, "verdict": got, "why": v and v.get("why"),
                 "rc": int(rc.read_text()) if rc.exists() else None,
                 "missing": (o / "v" / f"{tag}.missing").read_text().strip() if (o / "v" / f"{tag}.missing").exists() else None,
                 "share_verdict": None if sv is None else ("ACCEPT" if sv.get("accepted") is True else "REFUSE"),
                 "binary_sha256": binsha, "source_commit": commit, "circuit_sha512": csha, "session": sess,
                 "ok": got == expect and csha == want})
s = {"share_run": run, "source_commit": commit, "binary_sha256": binsha, "circuit_sha512_pin": want,
     "circuit_sha512_before": before, "circuit_sha512_after": after, "rows": rows,
     "ok": all(r["ok"] for r in rows) and before == after == want}
(o / "summary.json").write_text(json.dumps(s, indent=1))
md = ["| lane | session run | session | expect | main verdict | share verdict | rc | binary sha256 | source commit | circuit sha512 |",
      "|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    md.append(f"| {r['tag']} | {run} | {r['session']} | {r['expect']} | {r['verdict'] or r['missing']} | {r['share_verdict']} | {r['rc']} | "
              f"{binsha[:16]}… | {commit[:10]} | {(r['circuit_sha512'] or '-')[:16]}… |")
(o / "summary.md").write_text("\n".join(md) + "\n")
print("SUMMARY " + json.dumps({"ok": s["ok"], "rows": [(r["tag"], r["verdict"], r["expect"]) for r in rows]}))
sys.exit(0 if s["ok"] else 1)
EOF
rc=$?
log "done (exit $rc)"
exit $rc
