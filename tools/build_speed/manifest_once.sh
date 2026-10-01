#!/usr/bin/env bash
# One B=1 manifest build (`manifest build`, as the Build's required-manifest step runs it) of a kept request Program dir (PROGRAM: its
# instances.json.gz, result.json, artifact.json, descriptor.json.gz), reading the unit rules from RULES: its wall, its processes' CPU time, the peak RSS of its
# largest process, the manifest's digest and the sha256 of its file, and its log, to $RESEARCH_RUN_DIR/out/. Measurement only.
set -euo pipefail
ROW=$1 PROGRAM=$2 RULES=$3
SRC=$(pwd)
RID=${RESEARCH_RUN_ID:?}
SCR=/workspace/research/build-speed/runs/$RID
mkdir -p "$SCR/tmp" "$RESEARCH_RUN_DIR/out"
trap 'rm -rf "$SCR"' EXIT
export TMPDIR=$SCR/tmp PY=/workspace/jobs/venv312/bin/python OMP_NUM_THREADS=16 BUILD_JOBS=auto VERITY_UNIT_RULE_CACHE=$RULES
cd "$SRC/integrations/vllm"
export PYTHONPATH=.:$SRC/packages/verity/src:$SRC/tools/research/src
"$PY" -m compileall -q -j 16 "$SRC/packages/verity/src" verity_vllm "$SRC/tools/research/src" > /dev/null 2>&1 || true
"$PY" - "$PROGRAM" "workloads/$ROW.json" "$SCR/manifest.json" "$RESEARCH_RUN_DIR/out/manifest.log" << 'EOF' | tee "$RESEARCH_RUN_DIR/out/manifest_once.json"
import hashlib, json, resource, subprocess, sys, time
program, workload, out, log = sys.argv[1:5]
t0 = time.time()
with open(log, "w") as f:
    rc = subprocess.call([sys.executable, "-m", "verity_vllm.pipeline.cli", "manifest", "build", "--program", program, "--workload", workload,
                          "--norm-scales", "--router-softmax", "--vocab-range", "--guarded-max", "--out", out], stdout=f, stderr=subprocess.STDOUT)
wall = time.time() - t0
if rc:
    sys.exit(rc)
m = json.load(open(out))
h = hashlib.sha256()
with open(out, "rb") as f:
    while b := f.read(8 << 20):
        h.update(b)
ru = resource.getrusage(resource.RUSAGE_CHILDREN)
print(json.dumps({"wall": round(wall, 1), "cpu_s": round(ru.ru_utime + ru.ru_stime, 1), "peak_rss_gb": round(ru.ru_maxrss / 2**20, 2),
                  "manifest_digest": m.get("manifest_digest"), "program_digest": m.get("program_digest"), "manifest_sha256": h.hexdigest()}, indent=1))
EOF
