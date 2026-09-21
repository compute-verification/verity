#!/usr/bin/env bash
# Acceptance fixtures for telemetry v0.1 increment A (brief §12) on a Linux pod (cp-bk: cgroup v2 read-only, shared pod scope, no GPU).
#   run_fixtures.sh <out-root> [cases...]      cases default: all
# Each case is one attempt directory under <out-root>/<case>/ produced by the reference harness `tele run`; SUMMARY.txt collects the
# classifier's class/certainty/limitations and the case-specific checks.  Needs python3 >= 3.11; runs as root on the pod.
set -uo pipefail
OUT=${1:?out-root}; shift || true
CASES=${*:-oom_own timeout guardian bystander forkshare writer sampler_killed sampler_denied truncated spans}
HERE=$(cd "$(dirname "$0")" && pwd); PKG=$(cd "$HERE/.." && pwd)             # .../research/telemetry
export PYTHONPATH=$(cd "$PKG/../.." && pwd)                                          # .../src, so `research.telemetry` imports
PY=${PY:-python3}
TELE="$PY -m research.telemetry"
mkdir -p "$OUT"; SUM=$OUT/SUMMARY.txt; : > "$SUM"
say() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$SUM"; }
summ() {  # <attempt-dir>  -> one line of class/certainty + limitations + incidents
  $PY - "$1" <<'EOF'
import json, sys
d = sys.argv[1]
f = json.load(open(f"{d}/failure.json"))
print(f"  class={f['class']} certainty={f['certainty']} rc={f['termination']['rc']} signal={f['termination']['signal']} partial={f['telemetry_partial']}")
print(f"  why: {f['why']}")
for i in f["incidents"]: print(f"  incident: {i.get('kind')} scope={i.get('scope')} delta={i.get('counters_delta')}")
for l in f["limitations"]: print(f"  limitation: {l}")
for e in f["evidence"]:
    if e["kind"] in ("oom.attribution", "cancel_intent", "cancel_result", "cgroup.sampled"): print(f"  evidence: {json.dumps(e)[:300]}")
EOF
}
say "fixtures start on $(hostname): cgroup $(stat -fc %T /sys/fs/cgroup) memory.max=$(cat /sys/fs/cgroup/memory.max 2>/dev/null) memory.current=$(cat /sys/fs/cgroup/memory.current 2>/dev/null) oom.group=$(cat /sys/fs/cgroup/memory.oom.group 2>/dev/null) swaps=$(tail -n +2 /proc/swaps | wc -l) nvidia-smi=$(which nvidia-smi || echo none) python=$($PY --version 2>&1)"
say "oom_kill counter before: $(grep oom_kill /sys/fs/cgroup/memory.events)"

for c in $CASES; do
  D=$OUT/$c; rm -rf "$D"; mkdir -p "$D"
  case $c in
  oom_own)
    say "== oom_own: hog in the job's tree, shared pod cgroup (oom_score_adj=1000 -> the hog is the kernel's victim)"
    $TELE run --attempt-dir "$D" --stage fx.oom --timeout 300 --project fixtures --job oom_own --lane mprof --policy "$PKG/policy/r17-rules-1.toml" \
      -- $PY "$HERE/hog.py" --step-mib 256 --sleep 0.25 2>>"$D/run.err"; rc=$?
    say "  harness rc=$rc; last hog line: $(tail -1 "$D/stdout.log")"; summ "$D" | tee -a "$SUM"
    say "  EXPECT CGROUP_OOM_ATTRIBUTED (inferred; shared scope limitations) -- oom_kill now: $(grep oom_kill /sys/fs/cgroup/memory.events)";;
  timeout)
    say "== timeout: sleep 60 under --timeout 5 (marker actor=timeout before SIGTERM)"
    $TELE run --attempt-dir "$D" --stage fx.timeout --timeout 5 --grace 3 --project fixtures --job timeout --lane mprof -- sleep 60 2>>"$D/run.err"; rc=$?
    say "  harness rc=$rc"; summ "$D" | tee -a "$SUM"; say "  EXPECT STAGE_TIMEOUT (established: delivered marker)";;
  guardian)
    say "== guardian: external cancel-intent --actor guardian --deliver SIGKILL on the workload pid (identity verified by starttime)"
    $TELE run --attempt-dir "$D" --stage fx.guardian --timeout 60 --project fixtures --job guardian --lane mprof -- sleep 60 2>>"$D/run.err" & RUNPID=$!
    for i in $(seq 1 40); do [ -f "$D/status.json" ] && $PY -c "import json,sys; s=json.load(open('$D/status.json')); sys.exit(0 if s.get('pid') else 1)" && break; sleep 0.25; done
    sleep 3
    WPID=$($PY -c "import json; print(json.load(open('$D/status.json'))['pid'])"); ST=$($PY -c "import json; print(json.load(open('$D/status.json'))['starttime_ticks'])")
    $TELE cancel-intent --attempt-dir "$D" --actor guardian --target-pid "$WPID" --expect-starttime "$ST" --signal 9 --reason "fixture: guardian-style kill (per-proc cap)" --controller mem_guardian-fixture --deliver --confirm-exit-s 2; say "  cancel-intent rc=$?"
    wait $RUNPID; rc=$?; say "  harness rc=$rc"; summ "$D" | tee -a "$SUM"; say "  EXPECT CANCELLED_GUARDIAN (established)"
    say "  identity refusal check: a recycled-pid guard --"; $TELE cancel-intent --attempt-dir "$D" --actor guardian --target-pid $$ --expect-starttime 1 --signal 0 --reason "fixture: wrong starttime must be refused" --deliver; say "  refusal rc=$? (expect 3)";;
  bystander)
    say "== bystander: this job = sleep 30 (tree pid only); a hog OUTSIDE the tree is OOM-killed in the shared pod cgroup meanwhile"
    $TELE run --attempt-dir "$D" --stage fx.bystander --timeout 120 --project fixtures --job bystander --lane mprof -- sleep 30 2>>"$D/run.err" & RUNPID=$!
    sleep 4; $PY "$HERE/hog.py" --step-mib 256 --sleep 0.25 > "$D/outside_hog.log" 2>&1; say "  outside hog rc=$? (137 = SIGKILL by the kernel): $(tail -1 "$D/outside_hog.log")"
    wait $RUNPID; rc=$?; say "  harness rc=$rc"; summ "$D" | tee -a "$SUM"; say "  EXPECT SUCCESS + cgroup_oom incident in shared scope, NO attribution to this job"
    say "  bystander KILLED variant: job = sleep killed by SIGKILL (manual, no marker) while an outside hog OOMs -> must be OOM_UNRESOLVED_ORIGIN, not attributed"
    D2=$OUT/bystander_killed; rm -rf "$D2"; mkdir -p "$D2"
    $TELE run --attempt-dir "$D2" --stage fx.bystander2 --timeout 120 --project fixtures --job bystander_killed --lane mprof -- sleep 60 2>>"$D2/run.err" & RUNPID=$!
    sleep 4; ( $PY "$HERE/hog.py" --step-mib 256 --sleep 0.25 > "$D2/outside_hog.log" 2>&1 ) & HOGPID=$!
    for i in $(seq 1 200); do grep -q "hog 3328" "$D2/outside_hog.log" 2>/dev/null && break; sleep 0.1; done   # kill the job just as the hog nears the limit
    WPID=$($PY -c "import json; print(json.load(open('$D2/status.json'))['pid'])"); kill -9 "$WPID"; wait $HOGPID; say "  outside hog rc=$?"
    wait $RUNPID; rc=$?; say "  harness rc=$rc"; summ "$D2" | tee -a "$SUM"; say "  EXPECT OOM_UNRESOLVED_ORIGIN (ambiguous) or UNKNOWN_SIGNAL if the windows did not coincide -- never CGROUP_OOM_ATTRIBUTED";;
  oom_discriminator)
    say "== oom_discriminator (rev d2): a LARGER bystander (adj 0, ~2 GiB, held) survives; the job's SMALLER hog (adj 1000) is the kernel's victim"
    say "   -> the largest RSS is an outsider (recorded as corroboration that disagrees, never a veto); disappearance + no outsider vanished attributes; oom_score agrees"
    $PY "$HERE/hog.py" --step-mib 256 --sleep 0.1 --stop-mib 2048 --oom-score-adj 0 --hold 90 > "$D/outside_big.log" 2>&1 & BIGPID=$!
    for i in $(seq 1 300); do grep -q "hog 2048" "$D/outside_big.log" 2>/dev/null && break; sleep 0.1; done
    say "  outside big holder: $(tail -1 "$D/outside_big.log")"
    $TELE run --attempt-dir "$D" --stage fx.oom_discr --timeout 300 --project fixtures --job oom_discriminator --lane mprof --policy "$PKG/policy/r17-rules-1.toml" \
      -- $PY "$HERE/hog.py" --step-mib 128 --sleep 0.3 2>>"$D/run.err"; rc=$?
    say "  harness rc=$rc; last job hog line: $(tail -1 "$D/stdout.log")"
    kill -TERM $BIGPID 2>/dev/null; wait $BIGPID 2>/dev/null; say "  outside big holder after: $(tail -1 "$D/outside_big.log") (must NOT have been killed by the kernel)"
    summ "$D" | tee -a "$SUM"
    $PY - "$D" <<'EOF' | tee -a "$SUM"
import json, sys
f = json.load(open(f"{sys.argv[1]}/failure.json"))
att = next((e for e in f["evidence"] if e["kind"] == "oom.attribution"), {})
hi, lg = att.get("highest_oom_score_before") or {}, att.get("largest_before") or {}
co = att.get("corroboration") or {}
print(f"  primary: tree_gone={att.get('tree_gone')} outsiders_gone={att.get('outsiders_gone')} age_s={att.get('attribution_sample_age_s')}")
print(f"  corroboration: highest_oom_score_before={hi.get('membership')}/{hi.get('pid')}/{hi.get('comm')} score={hi.get('oom_score')} adj={hi.get('oom_score_adj')} agrees={co.get('highest_oom_score_is_the_vanished_tree_process')}"
      f" | largest_before={lg.get('membership')}/{lg.get('pid')} anon_shmem={lg.get('anon_shmem_bytes')} agrees={co.get('largest_anon_shmem_is_the_vanished_tree_process')}")
ok = f["class"] == "CGROUP_OOM_ATTRIBUTED" and hi.get("membership") == "tree" and lg.get("membership") == "outsider" and co.get("highest_oom_score_is_the_vanished_tree_process") is True
print("  " + ("PASS" if ok else "FAIL") + ": expected CGROUP_OOM_ATTRIBUTED by disappearance, oom_score corroborating (tree), largest RSS an outsider (recorded, not a veto)")
EOF
    say "  EXPECT CGROUP_OOM_ATTRIBUTED (inferred) by disappearance + no outsider vanished; oom_score corroborates -- oom_kill now: $(grep oom_kill /sys/fs/cgroup/memory.events)";;
  forkshare)
    say "== forkshare: MAP_SHARED 512 MiB touched by parent + 3 children; 128 MiB private per child"
    $TELE run --attempt-dir "$D" --stage fx.forkshare --timeout 120 --pss-every 2 --project fixtures --job forkshare --lane mprof -- $PY "$HERE/forkshare.py" --shared-mib 512 --private-mib 128 --children 3 --hold 10 2>>"$D/run.err"; rc=$?
    say "  harness rc=$rc"; summ "$D" | tee -a "$SUM"
    $PY - "$D" <<'EOF' | tee -a "$SUM"
import json, sys
d = sys.argv[1]
best = None
for line in open(f"{d}/resources.jsonl"):
    r = json.loads(line)
    if r["kind"] == "pss" and len(r["processes"]) >= 4:
        if best is None or r["rss_sum_bytes"] > best["rss_sum_bytes"]:
            best = r
if not best:
    print("  FAIL no pss sample with parent+3 children"); sys.exit(0)
mib = lambda b: b / 2**20
print(f"  steady-state pss sample: procs={len(best['processes'])} rss_sum={mib(best['rss_sum_bytes']):.0f} MiB pss_sum={mib(best['pss_sum_bytes']):.0f} MiB")
print(f"  EXPECT rss_sum ~ 4*512+3*128 = 2432 MiB (shared counted 4x), pss_sum ~ 512+3*128 = 896 MiB (shared counted once)")
ok = 2100 < mib(best["rss_sum_bytes"]) < 2800 and 750 < mib(best["pss_sum_bytes"]) < 1100
print("  " + ("PASS" if ok else "FAIL") + " PSS distinguishes shared from duplicated memory; summed RSS would overstate by", f"{mib(best['rss_sum_bytes'] - best['pss_sum_bytes']):.0f} MiB")
EOF
    ;;
  writer)
    say "== writer: 1 GiB buffered, no fsync, hold 8 s"
    $TELE run --attempt-dir "$D" --stage fx.writer --timeout 120 --project fixtures --job writer --lane mprof -- $PY "$HERE/writer.py" --path "$D/blob.bin" --mib 1024 --chunk-mib 64 --hold 8 2>>"$D/run.err"; rc=$?
    say "  harness rc=$rc"; summ "$D" | tee -a "$SUM"
    $PY - "$D" <<'EOF' | tee -a "$SUM"
import json, sys
d = sys.argv[1]
dirty = wb = []; dirty=[]; wb=[]; wbytes=[]; pw=[]
for line in open(f"{d}/resources.jsonl"):
    r = json.loads(line)
    if r["kind"] == "cgroup" and r["cgroup"].get("memory_stat"):
        ms = r["cgroup"]["memory_stat"]; dirty.append(ms.get("file_dirty", 0)); wb.append(ms.get("file_writeback", 0))
        io = r["cgroup"].get("io_stat") or {}; wbytes.append(sum(v.get("wbytes", 0) for v in io.values()))
    if r["kind"] == "process":
        pw.append(sum((p.get("io") or {}).get("write_bytes") or 0 for p in r["processes"]))
mib = lambda b: b / 2**20
print(f"  cgroup file_dirty: min {mib(min(dirty)):.0f} max {mib(max(dirty)):.0f} MiB; file_writeback max {mib(max(wb)):.0f} MiB; io.stat wbytes delta {mib(wbytes[-1]-wbytes[0]):.0f} MiB (summed over the stacked devices the cgroup reports: overlay -> md -> nvme partitions count the same bytes once per layer); process write_bytes max {mib(max(pw)):.0f} MiB")
print("  observation: process write accounting (write_bytes) is not completed writeback; file_dirty/file_writeback are the in-flight quantities; io.stat is the cgroup's device-level count -- three different meanings, none labelled bandwidth")
ok = (max(dirty) > 0 or max(wb) > 0) and (wbytes[-1] - wbytes[0]) > 0 and max(pw) >= 1000 * 2**20
print("  " + ("PASS" if ok else "FAIL") + " dirty/writeback and I/O observed; the classifier states observation only (class SUCCESS, no cause claimed)")
EOF
    ;;
  sampler_killed)
    say "== sampler_killed: kill -9 the sidecar mid-run; the workload and its evidence must survive; telemetry marked partial"
    $TELE run --attempt-dir "$D" --stage fx.skill --timeout 60 --project fixtures --job sampler_killed --lane mprof -- sh -c 'echo workload-start; sleep 8; echo workload-evidence-ok' 2>>"$D/run.err" & RUNPID=$!
    for i in $(seq 1 40); do [ -f "$D/sampler.json" ] && break; sleep 0.25; done; sleep 2
    SPID=$($PY -c "import json; print(json.load(open('$D/sampler.json'))['pid'])"); kill -9 "$SPID"; say "  killed sampler pid $SPID"
    wait $RUNPID; rc=$?; say "  harness rc=$rc; workload stdout: $(tr '\n' '|' < "$D/stdout.log")"; summ "$D" | tee -a "$SUM"; say "  EXPECT SUCCESS, partial=True, limitation 'never wrote sampler_end'";;
  sampler_denied)
    say "== sampler_denied: sidecar runs as nobody against a root workload -> smaps_rollup / io permission denied -> null + reason, sampler survives"
    chmod 777 "$D"
    $TELE run --attempt-dir "$D" --stage fx.denied --timeout 60 --no-sampler --project fixtures --job sampler_denied --lane mprof -- sleep 8 2>>"$D/run.err" & RUNPID=$!
    for i in $(seq 1 40); do [ -f "$D/status.json" ] && $PY -c "import json,sys; sys.exit(0 if json.load(open('$D/status.json')).get('pid') else 1)" && break; sleep 0.25; done
    WPID=$($PY -c "import json; print(json.load(open('$D/status.json'))['pid'])")
    setpriv --reuid=65534 --regid=65534 --clear-groups env PYTHONPATH=$PKG TELEMETRY_STREAM_SUFFIX=nobody $PY -m telemetry sample --attempt-dir "$D" --cadence 1 --pss-every 2 --root-pid "$WPID" --linger 1 > "$D/sampler_nobody.log" 2>&1; say "  nobody-sampler rc=$?"
    wait $RUNPID; rc=$?; $TELE classify --attempt-dir "$D" >/dev/null; say "  harness rc=$rc"; summ "$D" | tee -a "$SUM"
    $PY - "$D" <<'EOF' | tee -a "$SUM"
import json, sys, glob
d = sys.argv[1]
den = tot = 0; reasons = set()
for f in glob.glob(f"{d}/resources*.jsonl"):
    for line in open(f):
        r = json.loads(line)
        if r["kind"] == "pss":
            for p in r["processes"]:
                tot += 1
                if p.get("pss_bytes") is None: den += 1; reasons.add(p.get("reason"))
print(f"  pss readings: {tot}, denied: {den}, reasons: {sorted(reasons)}")
print("  " + ("PASS" if den and den == tot else "FAIL") + " permission denial is recorded as null + reason (never 0), sampler kept running")
EOF
    ;;
  truncated)
    say "== truncated: a good run, then a truncated last line appended to events.jsonl and resources.jsonl; re-classify"
    $TELE run --attempt-dir "$D" --stage fx.trunc --timeout 60 --project fixtures --job truncated --lane mprof -- sh -c 'echo ok; sleep 3' 2>>"$D/run.err"; rc=$?
    C1=$($PY -c "import json; print(json.load(open('$D/failure.json'))['class'])")
    printf '{"kind": "mark", "stage": "fx.trunc", "na' >> "$D/events.jsonl"; printf '{"schema": "telemetry/resources/v0.1", "kind": "cgroup", "sco' >> "$D/resources.jsonl"
    $TELE classify --attempt-dir "$D" > /dev/null; say "  harness rc=$rc; class before $C1"; summ "$D" | tee -a "$SUM"
    N=$($PY -c "import json; print(len(json.load(open('$D/failure.json'))['inputs']))"); say "  EXPECT class unchanged ($C1), partial=True, two 'truncated last line' limitations; readers recovered ($N input files hashed)";;
  spans)
    say "== spans: nested / repeated / overlapping / unclosed spans through the real append helper"
    $PY "$HERE/spans.py" --attempt-dir "$D" 2>&1 | tee -a "$SUM"; say "  spans rc=${PIPESTATUS[0]} (expect 0)";;
  *) say "unknown case $c";;
  esac
done
say "oom_kill counter after: $(grep oom_kill /sys/fs/cgroup/memory.events)"
say "fixtures done -> $SUM"
