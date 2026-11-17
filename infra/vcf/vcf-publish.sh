#!/bin/bash
# vcf-publish.sh OUT PAYLOAD: on the control pod, publish each run directory a VCF job left under OUT/result/runs/ to the
# evidence store with run custody (`research data custody RUN_DIR --publish`), by PAYLOAD's own tools/research and r2.env's
# key. vcf-job-run.sh runs it after its copy; run it again to retry a publish that failed. Its output: OUT/publish.log.
set -u
OUT=${1:?OUT}; SEND=${2:?PAYLOAD}
RUNS=$(ls -d "$OUT"/result/runs/*/ 2>/dev/null)
[ -n "$RUNS" ] || { echo "no run directory under $OUT/result/runs: nothing to publish" >&2; exit 6; }
# the local store stays on /data, never the pod's /root
R=$(mktemp -d -p /data/vcf) && tar -C "$R" -xzf "$SEND" tree/tools/research || { echo "no tools/research in $SEND" >&2; rm -rf "$R"; exit 6; }
rc=0
for d in $RUNS; do
  if ( set -a; . /root/.config/verity/r2.env; set +a
       cd "$R/tree/tools/research" && PYTHONPATH=src LC_ALL=C.UTF-8 python3.12 -m research data custody "${d%/}" --publish --store "${VCF_STORE:-/data/vcf/store}"
     ) >> "$OUT/publish.log" 2>&1; then
    echo "published $(basename "$d"): $(tail -n 1 "$OUT/publish.log" | cut -c1-200)"
  else
    rc=7; echo "publish of $(basename "$d") failed: $(tail -n 1 "$OUT/publish.log" | cut -c1-200)"
  fi
done
rm -rf "$R"
exit $rc
