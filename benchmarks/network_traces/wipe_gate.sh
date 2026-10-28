#!/usr/bin/env bash
# Network accounting's side of a memory wipe: one warden session through the auditor's gate (`warden-rehearse --scan
# --prompts`, the toy's class) spanning the wipe's session, under the lane's gate key (GATE_KEY), inside a job on the
# machine that holds it (site.json's auditor), cwd = the source tree; everything into $OUT. GATE_KEY, CHALLENGER, PEER
# and KEEP default to site.json's gate_key (under ~/.ssh), gate, peer and keep.
#
# With AT (a UTC time) it first waits until then, so a run launched early fires on the clock with no VM awake: give the
# run a time limit past AT, MINUTES, the fetch and the leak guard (below). Launched more than LATE_S (60) seconds
# after AT it sends nothing. It
# sends the gate REQUESTS prompts of the toy's class (`demo_stub.py prompts`), the toy (`demo_stub.py serve`) already
# listening on PEER's loopback PORT, and keeps the gate's answer in KEEP/RUN/start.json alone (0600), never published:
# it is the request's public fields and the run's systemd unit, vy-certifier@RUN.service, which the warden's witness also
# records (stalls.json's cgroup paths) and no public file holds, so the leak guard would hold it as a piece of the
# witness. A refusal (another rehearsal still runs), an answer naming no certifier run, goes to the log: a visible
# failure, never retried. Then `status` every 15 s until the run ends (status.json), and its bundle with
# witness/, the lane's own (`fetch`), to KEEP/RUN/fetch.tar.gz (0700, outside the run dir: custody publishes $OUT), and
# to bundle.tgz only what `fetch` gives any other key, the bundle less witness/; summary.json is the session's. Exits
# with the gate run's rc.
#
# Everything, this job's own stdout and stderr too (wipe-gate.log), is written to a hold, KEEP/held/ID (ID the research
# run's id), and reaches $OUT only through the record's leak guard (leak_guard.py), which on every exit reads the run dir
# and the hold against the kept fetch, moves what holds a piece of the witness to KEEP/quarantine/ID and only then
# releases the rest (exit 3 when it found one). A signal, a guard past its budget or one that fails releases nothing:
# the outputs stay in the hold, out of the record, and the job fails. PY is the guard's interpreter (python3).
#
# 1700 requests, the gate's most (`auditor.WARDEN_REQUESTS`), on the gate's most prompts, 400 (`WARDEN_PROMPTS`; request
# j sends the (j mod n)-th), are 1064 link-windows at its fixed seed (`warden_serve.prompt_traffic`): 3404.8 s from the
# session's origin, which is about 16 s after the request at HEIGHT 13. At about 5 one-time signatures a link-window
# they take about 5300 of height 13's 8192 (height 12's 4096 sign about 819 windows). A gate pinned before 1700 takes
# 400 requests at most, 259 link-windows: give it REQUESTS=400 HEIGHT=12 MINUTES=20. A wipe whose [t0, t1) lies within
# [origin + 1 s, origin + 3403.8 s) is covered; then
#
#   python benchmarks/network_traces/wipe_egress.py $OUT/bundle.tgz --t0-ns T0 --t1-ns T1 --out egress.json
#
# writes its `wipe-egress/v1` record.
set -euo pipefail
: "${OUT:?}"
site() { python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$SITE" "$1"; }
SITE=$(dirname "${BASH_SOURCE[0]}")/site.json
GATE_KEY=${GATE_KEY:-$HOME/.ssh/$(site gate_key)}
CHALLENGER=${CHALLENGER:-$(site gate)}
PEER=${PEER:-$(site peer)} PORT=${PORT:-18776} REQUESTS=${REQUESTS:-1700} HEIGHT=${HEIGHT:-13} MINUTES=${MINUTES:-60}
PROMPTS=$((REQUESTS < 400 ? REQUESTS : 400))
LATE_S=${LATE_S:-60}
SSH=(ssh -T -o BatchMode=yes -o IdentitiesOnly=yes -o ServerAliveInterval=15 -i "$GATE_KEY" "$CHALLENGER")
KEEP=$(realpath -m "${KEEP:-$(site keep)}")
case "$KEEP/" in "$(realpath -m "${RESEARCH_RUN_DIR:-$OUT}")/"*|"$(realpath -m "$OUT")/"*)
  echo "KEEP $KEEP is inside the run dir, which custody publishes: give a directory outside it"; exit 2;; esac
REQUEST="warden-rehearse --peer $PEER --port $PORT --minutes $MINUTES --requests $REQUESTS --height $HEIGHT --scan"
REQUEST+=" --prompts --tokens 24"

ID=${RESEARCH_RUN_ID:-$$} PUB=$OUT SIGNALED= RUN=
HELD=$KEEP/held/$ID Q=$KEEP/quarantine/$ID
release() {   # on every exit: the leak guard on the run dir and the hold, which releases the hold to $PUB once it passes
  local rc=$? g gates=()
  set +e
  trap - EXIT; trap 'SIGNALED=SIGTERM' TERM; trap 'SIGNALED=SIGINT' INT
  exec 1>&3 2>&4
  if [ -n "$SIGNALED" ]; then
    echo "wipe gate: $SIGNALED ended the run before the leak guard read it: nothing released, the outputs stay in $HELD"
    exit "$rc"
  fi
  [ -z "$RUN" ] || [ ! -e "$KEEP/$RUN/fetch.tar.gz" ] || gates=(--gates "$KEEP/$RUN/fetch.tar.gz")
  "${PY:-python3}" benchmarks/network_traces/leak_guard.py "${RESEARCH_RUN_DIR:-$(dirname "$PUB")}" "$Q" \
    ${gates[@]+"${gates[@]}"} --hold "$HELD" --release-to "$PUB" --report "$PUB/leak-guard.json"
  g=$?
  if [ -n "$SIGNALED" ]; then echo "wipe gate: $SIGNALED ended the leak guard (exit $g): nothing more released"; exit 143
  elif [ "$g" != 0 ] && [ "$g" != 3 ]; then
    echo "wipe gate: the leak guard failed (exit $g): nothing released, the outputs stay in $HELD"; exit 3
  fi
  [ ! -f "$PUB/summary.json" ] || cat "$PUB/summary.json"
  [ "$g" = 0 ] || [ "$rc" != 0 ] || rc=3
  exit "$rc"
}
trap 'SIGNALED=SIGTERM; exit 143' TERM
trap 'SIGNALED=SIGINT; exit 130' INT
mkdir -p "$HELD" && chmod 700 "$KEEP" "$KEEP/held" "$HELD"
exec 3>&1 4>&2
trap release EXIT
echo "$(date -u +%FT%TZ): every output and this job's own output held in $HELD until the leak guard passes them"
OUT=$HELD
exec >>"$OUT/wipe-gate.log" 2>&1

if [ -n "${AT:-}" ]; then
  at=$(date -u -d "$AT" +%s) || { echo "AT $AT isn't a time"; exit 2; }
  now=$(date -u +%s)
  [ "$now" -le $((at + LATE_S)) ] || { echo "$(date -u +%FT%TZ): past AT $AT by more than $LATE_S s: nothing sent"; exit 3; }
  echo "$(date -u +%FT%TZ): waiting until $AT"
  [ "$now" -ge "$at" ] || sleep $((at - now))
fi
echo "$(date -u +%FT%TZ): $REQUEST"
rc=0
reply=$(PYTHONPATH=$PWD python3 benchmarks/network_traces/demo_stub.py prompts --count "$PROMPTS" \
  | "${SSH[@]}" "$REQUEST") || rc=$?
RUN=$([ "$rc" = 0 ] && python3 -c 'import json, sys; print(json.loads(sys.argv[1])["run"])' "$reply" 2>/dev/null) || RUN=
# `auditor.CERTIFIER_RUN`, which plain python3 can't import; RUN names a directory under KEEP
if ! [[ "$RUN" =~ ^c[0-9]{8}T[0-9]{6}Z-[0-9a-f]{4}$ ]]; then
  echo "$(date -u +%FT%TZ): the gate didn't start the session (exit $rc): ${reply:0:400}"; exit $((rc ? rc : 1))
fi
mkdir -p "$KEEP/$RUN" && chmod 700 "$KEEP" "$KEEP/$RUN"
(umask 077 && printf '%s\n' "$reply" > "$KEEP/$RUN/start.json")
echo "$(date -u +%FT%TZ): gate run $RUN, its start reply in $KEEP/$RUN/start.json alone"
until "${SSH[@]}" "status $RUN" > "$OUT/status.json" \
    && python3 -c 'import json, sys; sys.exit(json.load(open(sys.argv[1]))["end"] is None)' "$OUT/status.json"; do
  sleep 15
done
(umask 077 && "${SSH[@]}" "fetch $RUN" > "$KEEP/$RUN/fetch.tar.gz")
python3 - "$OUT/status.json" "$KEEP/$RUN/fetch.tar.gz" "$OUT/bundle.tgz" <<'EOF' | tee "$OUT/summary.json"
import json, sys, tarfile
end = json.load(open(sys.argv[1]))["end"]
with tarfile.open(sys.argv[2], "r:gz") as tar, tarfile.open(sys.argv[3], "w:gz") as public:
    for m in tar:
        if "witness" not in m.name.split("/") and (m.isfile() or m.isdir()):
            public.addfile(m, tar.extractfile(m) if m.isfile() else None)
with tarfile.open(sys.argv[3], "r:gz") as tar:
    dep = json.load(tar.extractfile(next(m for m in tar if m.name.endswith("/out/deployment.json"))))
n, w = dep["windows"], dep["params"]["window"] * dep["params"]["bucket_ms"] / 1000
print(json.dumps({"run": end.get("run"), "rc": end.get("rc"), "ended_utc": end.get("ended_utc"),
                  "accepted": (end.get("verdict") or {}).get("accepted"), "origin_unix": dep["origin_unix"],
                  "windows": n, "session_s": round(n * w, 1)}))
sys.exit(end.get("rc") or 0)
EOF
