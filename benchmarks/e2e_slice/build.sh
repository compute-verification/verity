#!/usr/bin/env bash
# The slice's CPU-side builds, from the source tree's root (`research run --cwd source`): C-Flock's flock-circuit (the proving
# build and the selftest build) and the numpy venv, by hidden_zk's own builders and cache (benchmarks/pouw/hidden_zk/job.sh,
# sourced; keyed by verity/ml/flock's sources under $HZ_CACHE), so a later run of the slice finds them by the same key. Writes
# $OUT/builds.json: the binaries' paths and sha256, the venv's python.
set -uo pipefail
export HZ_CONFIGS=none HZ_MEM_GB=${HZ_MEM_GB:-64}
export HZ_CACHE=${HZ_CACHE:-/workspace/research/cache/pouw-hidden-zk}
source benchmarks/pouw/hidden_zk/job.sh
key=$(tree_key verity/ml/flock/live verity/ml/flock/*.patch)
F=$(cached flock "$key" build_circuit)
V=$(cached venv numpy2.5.3-blake3-1.0.9 build_python)
for p in "$F/flock-circuit-prove" "$F/flock-circuit-selftest" "$V/bin/python"; do [ -x "$p" ] || { log "missing $p"; tail -30 "$OUT"/build-*.log; exit 1; }; done
"$V/bin/python" - "$F" "$V" "$key" > "$OUT/builds.json" <<'EOF'
import hashlib, json, sys
f, v, key = sys.argv[1:]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
print(json.dumps({"flock_key": key, "prove": f"{f}/flock-circuit-prove", "prove_sha256": sha(f"{f}/flock-circuit-prove"),
                  "selftest": f"{f}/flock-circuit-selftest", "python": f"{v}/bin/python"}))
EOF
cat "$OUT/builds.json"
