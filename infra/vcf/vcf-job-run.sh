#!/bin/bash
# vcf-job-run.sh NAME PARTITION MINUTES JOBSCRIPT [SBATCH_ARG...]: run on the control pod, deployed as /data/vcf/vcf-job-run.sh.
# Submits JOBSCRIPT as one Slurm batch job on VCF's cluster under Daniel's account, waits for it, copies its output to OUT
# (default /data/vcf-jobs/NAME-STAMP/) on the pod, and removes the job's directory under the account's home. Only the pod holds
# the key; nothing of ours but JOBSCRIPT and SEND's payload is sent there. MINUTES is at most 30 (an unreserved job); a job
# still pending after PEND_MIN minutes (default 20) is cancelled. sacct.txt records its node, submit, start and end.
# With SEND (a payload.sh archive), each run directory the job leaves under result/runs/ is published to the evidence store
# with run custody (`research data custody RUN_DIR --publish`), by the payload's own tools/research and r2.env's key.
#   SEND=/data/vcf/payloads/x.tar.gz vcf-job-run.sh native-peak main 30 /data/vcf/vcf-bench.sh --gres=gpu:1 --cpus-per-task=16 --mem=64G
set -u
NAME=${1:?name}; PART=${2:?partition}; MIN=${3:?minutes}; JOB=${4:?job script}; shift 4
[ "$MIN" -le 30 ] 2>/dev/null || { echo "MINUTES: a whole number, at most 30" >&2; exit 2; }
[ -f "$JOB" ] || { echo "no job script at $JOB" >&2; exit 2; }
case "$NAME" in *[!A-Za-z0-9_-]*) echo "NAME: letters, digits, - and _ only" >&2; exit 2;; esac
PEND=${PEND_MIN:-20}
H=vcf.lucidcomputing.ai; U=daniel_reuter; KEY=/data/state/secrets/vcf-b300-login
WANT='SHA256:025kTncRbW96LL+49dZHfO7+TivsfEWBHJurRwyT208'
KH=/data/state/vcf-known-hosts
if [ ! -s "$KH" ]; then
  T=$(mktemp); ssh-keyscan -T 15 -t ed25519 "$H" > "$T" 2>/dev/null
  [ "$(ssh-keygen -lf "$T" 2>/dev/null | awk '{print $2}' | head -n 1)" = "$WANT" ] || { rm -f "$T"; echo "the host's key is not the one recorded: stopped" >&2; exit 3; }
  install -m 600 "$T" "$KH"; rm -f "$T"
fi
S=(ssh -o BatchMode=yes -o IdentitiesOnly=yes -o IdentityAgent=none -o ForwardAgent=no -o ClearAllForwardings=yes -o ConnectTimeout=15
   -o PreferredAuthentications=publickey -o StrictHostKeyChecking=yes -o UserKnownHostsFile="$KH" -o GlobalKnownHostsFile=/dev/null
   -i "$KEY" "$U@$H")
STAMP=$(date -u +%Y%m%dT%H%M%SZ); D="verity-jobs/$NAME-$STAMP"; OUT=${OUT:-/data/vcf-jobs/$NAME-$STAMP}
mkdir -p "$OUT" || exit 4
say() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$OUT/vcf-job.log"; }
say "submit $NAME on $PART for $MIN min, extra: $*"
if [ -n "${SEND:-}" ]; then
  [ -f "$SEND" ] || { say "no file at SEND=$SEND"; exit 2; }
  "${S[@]}" "mkdir -p '$D' && cat > '$D/payload.tar.gz'" < "$SEND" || { say "the payload's copy failed"; "${S[@]}" "rm -rf '$D'"; exit 5; }
  say "payload sent: $(stat -c %s "$SEND") bytes"
fi
ID=$("${S[@]}" "mkdir -p '$D' && cat > '$D/job.sh' && cd '$D' && sbatch --parsable --job-name='$NAME' --partition='$PART' --time='$MIN' --output=out.txt --error=err.txt $* job.sh" < "$JOB" 2> "$OUT/submit.err" | tr -dc '0-9;' | cut -d';' -f1)
[ -n "$ID" ] || { say "not submitted: $(tr '\n' ' ' < "$OUT/submit.err" | cut -c1-300)"; "${S[@]}" "rm -rf '$D'; rmdir verity-jobs 2>/dev/null; true"; exit 5; }
say "job $ID"
n=0
while [ $n -lt $(((PEND + MIN) * 4 + 40)) ]; do
  n=$((n + 1)); sleep 15
  st=$("${S[@]}" "squeue -h -j $ID -o %T 2>/dev/null" 2>/dev/null | head -n 1)
  [ -n "$st" ] || break
  if [ "$st" = PENDING ] && [ $n -ge $((PEND * 4)) ]; then
    say "still pending after $PEND min: cancelled"; "${S[@]}" "scancel $ID" 2>/dev/null; sleep 15; break
  fi
  [ $((n % 8)) = 0 ] && say "state $st"
done
"${S[@]}" "sacct -n -X -P -j $ID -o JobID,State,ExitCode,NodeList,Submit,Start,End,Elapsed,AllocTRES 2>/dev/null | head -n 1" > "$OUT/sacct.txt" 2>/dev/null
say "ended: $(cut -c1-200 < "$OUT/sacct.txt")"
"${S[@]}" "cd '$D' && tar -cf - out.txt err.txt \$(ls result 2>/dev/null | sed 's#^#result/#')" 2>/dev/null | tar --no-same-owner -C "$OUT" -xf - || say "the copy of the output failed"
"${S[@]}" "rm -rf '$D'; rmdir verity-jobs 2>/dev/null; true" 2>/dev/null
say "output in $OUT: $(ls "$OUT" | tr '\n' ' ')"
[ -n "${SEND:-}" ] || exit 0
RUNS=$(ls -d "$OUT"/result/runs/*/ 2>/dev/null)
[ -n "$RUNS" ] || { say "no run directory came back: nothing to publish"; exit 6; }
# the local store stays on /data, never the pod's /root
R=$(mktemp -d -p /data/vcf) && tar -C "$R" -xzf "$SEND" tree/tools/research/src || { say "no tools/research in $SEND"; rm -rf "$R"; exit 6; }
rc=0
for d in $RUNS; do
  if ( set -a; . /root/.config/verity/r2.env; set +a
       cd "$R/tree/tools/research" && PYTHONPATH=src LC_ALL=C.UTF-8 python3.12 -m research data custody "${d%/}" --publish --store "${VCF_STORE:-/data/vcf/store}"
     ) >> "$OUT/publish.log" 2>&1; then
    say "published $(basename "$d"): $(tail -n 1 "$OUT/publish.log" | cut -c1-200)"
  else
    rc=7; say "publish of $(basename "$d") failed: $(tail -n 1 "$OUT/publish.log" | cut -c1-200)"
  fi
done
rm -rf "$R"
exit $rc
