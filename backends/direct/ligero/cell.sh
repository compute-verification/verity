#!/usr/bin/env bash
# A B-interactive Table 2 cell (bench.cell's driver drivers/b_interactive.py): the TABLES sweep of one B-Ligero statement, the
# prover on the line's GPU against a live verifier on a separate pod of the same datacenter.  Runs from the shipped tree
# (research run --source . --cwd source); KEY=VALUE arguments become the environment.
#
#   ROLE=verifier  [PORT=7000] [IDLE=1800] [JOBS=<vCPUs>]
#       CPU torch + ligero-verify from this tree, then `live serve` until IDLE seconds pass with no session.  The session index,
#       log and every session's verdict files are the run's output "live-verdicts" (run-files/v1).
#   ROLE=prover  REL=<rel>+<leaf> VERIFIER=host:port [SET=<set tarball under inputs/ | set dir>] [SET_ART=art:...] [START=1024]
#                [RUNS=5] [L=4096] [PP=4] [SCREEN="4096:4 8192:2"] [SCREEN_VUS=2048] [LOOPBACK=1] [OPERATOR=<lane>]
#       pod_bootstrap.sh; the open-connection RTT / bandwidth probe to the verifier; optionally a per-proof screen (local coins,
#       2 reps, the fastest (l, p) sweeps); the sweep (sweep_vu, capped at the set's size, every point bench-vu --verifier
#       --no-local-verify --commit-per-rep); the pod's pinned ligero-verify on the plateau dump (a producer check); the loopback
#       probe at the plateau; result.json = the cell (backends.direct.ligero.cell finish), outputs.json = the other points.
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uo pipefail
REPO=$(pwd); RD=${RESEARCH_RUN_DIR:?}; W=/workspace
ROLE=${ROLE:-prover}
say() { echo "##### $(date -u +%H:%M:%SZ) $*"; }
Q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo -1); PER=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || echo 100000)
if [ -r /sys/fs/cgroup/cpu.max ]; then read -r Q PER < /sys/fs/cgroup/cpu.max; [ "$Q" = max ] && Q=-1; fi
NT=$(( Q > 0 ? Q / PER : $(nproc) )); [ "$NT" -lt 1 ] && NT=1
export PYTHONPATH="$REPO/packages/verity/src:$REPO/backends/numerical/python:$REPO/tools/research/src:$REPO"
export RESEARCH_GIT_COMMIT=${RESEARCH_SOURCE_SHA:-$(python3 -c "import json;print(json.load(open('.research-source.json')).get('commit',''))" 2>/dev/null)}

if [ "$ROLE" = verifier ]; then
  PY=$W/venv312/bin/python; PORT=${PORT:-7000}
  export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH" CARGO_TARGET_DIR=$W/cargo-target
  mkdir -p $W/bin
  say "verifier setup ($NT vCPU of $(nproc), DC ${RUNPOD_DC_ID:-?}, tree $RESEARCH_GIT_COMMIT)"
  command -v cargo >/dev/null 2>&1 || curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable >/tmp/rustup.log 2>&1
  ( cd backends/ligero-verify && cargo build --release 2>&1 | tail -2 && cp $CARGO_TARGET_DIR/release/ligero-verify $W/bin/ligero-verify ) > $RD/build.log 2>&1 &
  B=$!
  if ! "$PY" -c "import torch, numpy, blake3" 2>/dev/null; then
    command -v uv >/dev/null 2>&1 || curl -fsSL https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
    uv python install 3.12 >/dev/null 2>&1; [ -x "$PY" ] || uv venv $W/venv312 --python 3.12 >/dev/null
    uv pip install --python "$PY" --index-url https://download.pytorch.org/whl/cpu "torch==2.6.0" 2>&1 | tail -1
    uv pip install --python "$PY" numpy blake3 2>&1 | tail -1
  fi
  wait $B || { cat $RD/build.log; echo "BUILD FAILED"; exit 1; }
  sha256sum $W/bin/ligero-verify | tee $RD/ligero_verify.sha256
  say "live serve on :$PORT (idle exit ${IDLE:-1800}s)"
  # a finished session's statements and proofs (GBs per cell) leave the disk once its verdict is written; the verdicts stay
  ( while sleep 60; do for d in $RD/sessions/s*/; do [ -f "$d/verdict.json" ] && rm -f "$d"/*.proof "$d"/*.hproof "$d"/*.stmt; done; done ) &
  CLEAN=$!
  "$PY" -m backends.direct.ligero.live serve --listen 0.0.0.0:$PORT --out $RD/sessions --ligero-verify $W/bin/ligero-verify \
      --jobs ${JOBS:-$NT} --threads 1 --target-bits 128 --idle-exit ${IDLE:-1800} > $RD/serve.out 2>&1
  rc=$?
  kill $CLEAN 2>/dev/null
  tail -3 $RD/serve.out
  "$PY" - "$RD" <<'EOF'
import json, sys
from pathlib import Path
rd = Path(sys.argv[1])
files = [str(p.relative_to(rd)) for p in sorted((rd / "sessions").rglob("*")) if p.is_file() and p.suffix in (".json", ".jsonl", ".log")]
files += [f for f in ("serve.out", "ligero_verify.sha256", "build.log") if (rd / f).is_file()]
(rd / "outputs.json").write_text(json.dumps({"schema": "research/outputs/v0.1", "outputs": [
    {"name": "live-verdicts", "kind": "run-files/v1", "files": files,
     "meta": {"label": "B-interactive live verifier: session index, log and every session's verdicts (the proofs stay with the prover's dump)"}}]}))
print(f"outputs.json: {len(files)} verdict files")
EOF
  exit $rc
fi

# ---- prover -------------------------------------------------------------------------------------------------------------------
REL=${REL:?}; VERIFIER=${VERIFIER:?}; L=${L:-4096}; PP=${PP:-4}; RUNS=${RUNS:-5}
say "bootstrap"
SRC=$REPO RELS= bash backends/direct/ligero/pod_bootstrap.sh > $RD/bootstrap.log 2>&1; echo "bootstrap rc=$?"; tail -n 6 $RD/bootstrap.log
source $W/env.sh; cd "$REPO"
export OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT VY_CPU_THREADS=$NT
export MALLOC_MMAP_MAX_=0 MALLOC_TRIM_THRESHOLD_=1000000000000 LIGERO_GPU_STRICT=1 LIGERO_GRAPH_STRICT=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# the warm-up sub-batch's host gen_step cross-check is minutes of Python per folded sub-batch; the relation's gadget gate runs it,
# the prover's constraint self-check on the warm-up sub-batch stays on
export LIGERO_REFERENCE_HINTS=${LIGERO_REFERENCE_HINTS:-0}
V=$W/bin/ligero-verify
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee $RD/gpu.txt
SETARGS=(); CAP=""
if [ -n "${SET:-}" ]; then
  S=$SET; [ -e "$S" ] || S=$RD/inputs/$SET
  if [ -f "$S" ]; then mkdir -p $RD/set && tar -xf "$S" -C $RD/set && S=$(dirname "$(find $RD/set -name manifest.json | head -1)"); fi
  CAP=$($PY -c "import json,sys;print(json.load(open(sys.argv[1]+'/manifest.json'))['n'])" "$S") || { echo "no input set at $S"; exit 2; }
  SETARGS=(--input-set "$S" ${SET_ART:+--input-set-art $SET_ART})
  say "input set $S: $CAP instances"
fi
say "probe tcp://$VERIFIER (waits for the verifier's setup)"
for i in $(seq 90); do
  $PY -m backends.direct.ligero.live probe --verifier tcp://$VERIFIER --pings 50 --mb 64 --repeat 3 --out $RD/probe.json > $RD/probe.log 2>&1 && break
  sleep 20
done
[ -f $RD/probe.json ] || { tail $RD/probe.log; echo "PROBE FAILED: no verifier at $VERIFIER"; exit 3; }
tail -n 1 $RD/probe.log
BENCH=(--zk --mode interactive --auth included-hash --commit-per-rep --target -128 --device cuda --instance-procs 16 "${SETARGS[@]}")
if [ -n "${SCREEN:-}" ]; then
  best=0
  for c in $SCREEN; do
    IFS=: read -r l p <<<"$c"; o=$RD/screen/l$l-p$p; mkdir -p $o
    say "screen l$l p$p at ${SCREEN_VUS:-2048} VUs (local coins, 2 reps)"
    $PY -u -m backends.direct.ligero.run --relation $REL bench-vu "${BENCH[@]}" --batch $l --pipeline $p --total-vus ${SCREEN_VUS:-2048} \
        --reps 2 --out $o/result.json > $o/bench.log 2>&1
    r=$?; v=$($PY -c "import json;m={x['name']:x['value'] for x in json.load(open('$o/result.json'))['measurements']};print(m['e2e.vu_per_second'], m.get('mem.peak_device_bytes'))" 2>/dev/null)
    echo "rc=$r vu/s,peak=$v $(grep -m1 -o 'OutOfMemoryError' $o/bench.log)"; v=${v%% *}
    [ $r -eq 0 ] && [ -n "$v" ] && $PY -c "import sys;sys.exit(0 if float('$v')>float('$best') else 1)" && { best=$v; L=$l; PP=$p; }
  done
  [ "$best" = 0 ] && { echo "SCREEN: nothing ran"; exit 7; }
  say "screen chose l$L p$PP ($best VU/s)"
fi
say "sweep $REL l$L p$PP from ${START:-1024}${CAP:+, capped at the set size $CAP}"
$PY -m backends.direct.ligero.sweep_vu --relation $REL --out-dir $RD/sweep --start ${START:-1024} ${CAP:+--cap $CAP} --dump plateau -- \
    "${BENCH[@]}" --batch $L --pipeline $PP --reps $RUNS --verifier tcp://$VERIFIER --no-local-verify > $RD/sweep.log 2>&1
src=$?; echo "sweep rc=$src"; grep -E "point|rc=|plateau|stopped" $RD/sweep.log | tail -n 14 | cut -c1-300
read -r PDIR NPL < <($PY -c "import json;d=json.load(open('$RD/sweep/sweep.json'));p=[p for p in d['points'] if p['point']==d['plateau_point']][0];print(p['dir'],p['total_vus'])") \
  || { echo "no plateau"; exit 6; }
( cd $RD/sweep/$PDIR/proofs || exit 1
  $V system-digest --system system.bin > rust_digest.json 2> rust_digest.err; echo "system-digest rc=$? $(head -c 300 rust_digest.json)"
  $V batch --system system.bin --dir rep1 --jobs $NT --threads 1 --target-bits 128 --json rust_batch.json > rust_batch.out 2>&1
  echo "producer rust batch rc=$? $(tail -n 1 rust_batch.out | cut -c1-300)"; sha256sum $V > ligero_verify.sha256 )
LB=()
if [ "${LOOPBACK:-1}" = 1 ]; then
  say "loopback probe: $NPL VUs against a live verifier on 127.0.0.1"
  mkdir -p $RD/loopback
  $PY -m backends.direct.ligero.live serve --listen 127.0.0.1:7001 --out $RD/loopback/sessions --ligero-verify $V --jobs $NT \
      --threads 1 --idle-exit 60 > $RD/loopback/serve.out 2>&1 &
  LS=$!; sleep 3
  $PY -u -m backends.direct.ligero.run --relation $REL bench-vu "${BENCH[@]}" --batch $L --pipeline $PP --total-vus $NPL --reps 2 \
      --verifier tcp://127.0.0.1:7001 --no-local-verify --out $RD/loopback/result.json > $RD/loopback/bench.log 2>&1
  echo "loopback rc=$? $(grep -m1 't.total_live' $RD/loopback/bench.log | cut -c1-200)"
  kill $LS 2>/dev/null; wait $LS 2>/dev/null
  [ -f $RD/loopback/result.json ] && LB=(--loopback $RD/loopback/result.json)
fi
say "the cell's result"
$PY -m backends.direct.ligero.cell finish --sweep $RD/sweep --probe $RD/probe.json "${LB[@]}" --verifier $VERIFIER \
    --operator "${OPERATOR:-}" --run-dir $RD --out $RD/result.json | tail -n 30
$PY - "$RD" "${OPERATOR:-bligero-real-k}" <<'EOF'
import json, sys
from pathlib import Path
rd, lane = Path(sys.argv[1]), sys.argv[2]
sw = json.loads((rd / "sweep" / "sweep.json").read_text())
outs = []
for p in sw["points"]:
    rj = rd / "sweep" / p["dir"] / "result.json"
    if p["point"] == sw["plateau_point"] or p.get("vu_per_second") is None or not rj.is_file():
        continue
    meta = json.loads(rj.read_text())
    meta.update(lane=lane, label=f"{lane} {sw['relation']} sweep point {p['point']} ({p['total_vus']} VUs)")
    outs.append({"name": f"sweep-p{p['point']}", "kind": "bench-result/v1", "meta": meta})
(rd / "outputs.json").write_text(json.dumps({"schema": "research/outputs/v0.1", "outputs": outs}, indent=1, default=str))
print(f"outputs.json: {len(outs)} non-plateau sweep points")
EOF
exit $src
