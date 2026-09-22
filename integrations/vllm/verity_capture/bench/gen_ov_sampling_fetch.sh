#!/usr/bin/env bash
# [ov-sampling] fetch the SWARM-20260917 small-artifact list of the lane's run dirs (never capture/values, snapshots, or logs;
# files over 20 MB are skipped on the pod).  Streams one tarball per run dir over the pod's ssh.
#   bash verity_capture/bench/gen_ov_sampling_fetch.sh <run-dir-name ...>      # e.g. temp07 temp07_ovs
set -uo pipefail
SSHF=out/capture/pods/cp-ov-sampling.ssh; RUNS=/workspace/cp/runs/ov-sampling; DST=out/gen/lanes/ov-sampling/runs
FILES="card.json CARD.md gates.json stages.json stages.full.json run.json fold_summary.json fold_summary_2.json program.json instances.jsonl accesses.jsonl.gz replay_a.json replay_chain.json wiring.json root_policy.json census_reconcile.json resolution.json resolution.md compare.json golden.json ov_recompute.json quarantine.json quarantine_lint.json tests.json nonint/noninterference.json"
for r in "$@"; do
  mkdir -p "$DST/$r"
  bash "$SSHF" "cd $RUNS/$r 2>/dev/null && for f in $FILES; do [ -f \$f ] && [ \$(stat -c %s \$f) -lt 25000000 ] && echo \$f; done | tar czf - -T -" 2>/dev/null | tar xzf - -C "$DST/$r" 2>/dev/null
  echo "$r: $(find "$DST/$r" -type f | wc -l | tr -d ' ') files, $(du -sh "$DST/$r" | cut -f1)"
done
