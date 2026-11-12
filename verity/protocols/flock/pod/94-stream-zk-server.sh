#!/usr/bin/env bash
# `prove --zk --stream -` behind the ZK server on real builds: verity/protocols/flock/prover/tests/test_stream_zk_server.py
# against this tree's proving build of flock-circuit (60-circuit.sh MODE=build GPU=0), on a CPU machine. The test builds
# its Lean itself: the Lean verifier's two executables, and the challenger's and its audit's. Run from the verity tree:
#   research run --on auto --cwd source --project verity -- bash verity/protocols/flock/pod/94-stream-zk-server.sh
# Kept in $OUT: mach, sync.log, build/ (the prover build's logs and binary.sha256), pytest.txt, junit.xml and sessions/
# (each arm's session records, its challenger's record, release log and audit, the Lean processes each prover started and
# the logs); never an instance or public file, an input set, a circuit, a table or a proof.
# env: FLOCK_WORK (a kept build directory for 60-circuit.sh; default the job's scratch)
set -uo pipefail
REPO=$(pwd); O=${RESEARCH_RUN_DIR:-/tmp/stream-zk-server}/out; mkdir -p "$O/build" "$O/sessions"
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$HOME/.elan/bin:$PATH
{ nproc; free -g | head -2; git rev-parse HEAD HEAD:verity/protocols/flock/prover HEAD:verity/protocols/flock/verifier; } > "$O/mach" 2>&1
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
uv sync --all-packages --extra torch-cpu > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 5; }
FW=${FLOCK_WORK:-$W}
RESEARCH_RUN_DIR=$O/build FLOCK_WORK=$FW MODE=build GPU=0 bash verity/protocols/flock/pod/60-circuit.sh > "$O/build/build.log" 2>&1 \
  || { tail -40 "$O/build/build.log"; exit 6; }
export FLOCK_CIRCUIT=$FW/flock-gpu-link/flock/target/release/flock-circuit RAYON_NUM_THREADS=$(nproc)
"$REPO/.venv/bin/python" -m pytest -p no:cacheprovider -v --basetemp="$W/pytest" --junitxml="$O/junit.xml" \
  verity/protocols/flock/prover/tests/test_stream_zk_server.py > "$O/pytest.txt" 2>&1
rc=$?
tail -40 "$O/pytest.txt"
(cd "$W/pytest" 2>/dev/null && find . -type f ! -name '*.bin' ! -name circuit.txt ! -name loop.sh ! -path '*/set/*' ! -path '*/tables/*' -print0 \
  | xargs -0 -r -I{} cp --parents {} "$O/sessions/")
exit $rc
