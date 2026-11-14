#!/usr/bin/env bash
# One cold session a member of a kept window behind the ZK server, on one card and on the CPU prover in turn
# (cold_sessions.py), with the builds as zk_cost.sh makes them. Run from the verity tree on a GPU node, as a GPU host job:
#   bash benchmarks/one_stage/cold_sessions.sh COLD_STAGE=probe|build|run COLD_ARCHIVE=FILE COLD_MEMBERS=K,K,... COLD_CARD=I
# env: COLD_ARCHIVE (a .tar.zst holding audit/window/{m<k>/,session-m<k>/draw.json,partition.json,program.json,summary.json}
#        and window/circuit.json, the window's registered circuit record)
#      COLD_MEMBERS (run in this order, each member's arms one after the other)  COLD_ARMS=gpu,cpu  COLD_CARD (an index)
#      COLD_CPUS=0-9 (every session's processes)  COLD_LEAN_CPUS=10-11 (the Lean verdicts, one at a time, beside the sessions)
#      COLD_OPEN (epoch: the run sleeps on one CPU until then)  COLD_END (epoch: no session starts whose estimate ends after
#        it; the estimate is COLD_EST_<arm>_<k> seconds, else COLD_EST_<arm>, default 1800)
#      COLD_SCRATCH=/dev/shm/proofs-cold (the window unpacked and the sessions' files; removed at the end)
#      FLOCK_WORK=/workspace/jobs/proof-throughput, PT_WORK=/workspace/cache/proof-throughput (the builds, as zk_cost.sh's)
#      a file $FLOCK_WORK/cold.stop stops the run before its next session
# probe: the archive listed, the window unpacked and planned (cold_sessions.py plan), the toolchains and the builds' cache
# looked at; nothing built, proved or leased. build: the builds alone, no card. run: the builds (cached), then the sessions.
# Kept in $OUT: job.log, cold.log (every COLD and VERDICT-COLD line), table.md and table.json, each session's records and
# logs under 8 MB, the stamps, binaries.sha256 and the command line; never a circuit, a proof or an archive.
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uo pipefail
REPO=$(pwd); O=${OUT:-${RESEARCH_RUN_DIR:?}/out}; mkdir -p "$O"
STAGE=${COLD_STAGE:?probe, build or run}
export FLOCK_WORK=${FLOCK_WORK:-/workspace/jobs/proof-throughput}; mkdir -p "$FLOCK_WORK" || exit 1
PT_WORK=${PT_WORK:-/workspace/cache/proof-throughput}
mkdir -p "$PT_WORK" 2>/dev/null && [ -w "$PT_WORK" ] || PT_WORK=$FLOCK_WORK/verifier
SCR=${COLD_SCRATCH:-/dev/shm/proofs-cold}
CPUS=${COLD_CPUS:-0-9}; LCPUS=${COLD_LEAN_CPUS:-10-11}
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$O/job.log"; }
export PYTHONDONTWRITEBYTECODE=1
unset CARGO_TARGET_DIR
echo "$0 $*" > "$O/command.txt"; env | grep -E '^COLD_' | sort >> "$O/command.txt"
git rev-parse HEAD > "$O/commit.txt" 2>/dev/null || echo "${RESEARCH_SOURCE_SHA:-}" > "$O/commit.txt"
if [ -n "${COLD_OPEN:-}" ]; then
  taskset -cp "${CPUS##*-}" $$ > /dev/null
  log "start guard: asleep on CPU ${CPUS##*-} until $(date -u -d "@$COLD_OPEN" +%FT%TZ); nothing unpacked, no lease"
  while now=$(date +%s); [ "$now" -lt "$COLD_OPEN" ]; do d=$(( COLD_OPEN - now )); sleep $(( d < 300 ? d : 300 )); done
fi
taskset -cp "$CPUS" $$ > /dev/null
log "cold-sessions $STAGE: source $(cat "$O/commit.txt"), CPUs $CPUS, builds under $FLOCK_WORK and $PT_WORK"
{ lscpu | grep -E 'Model name|^CPU\(s\)'; nproc; free -g; df -h / /dev/shm; } > "$O/host.txt" 2>&1

PY=$FLOCK_WORK/flock-circuit/py/bin/python3
if [ ! -x "$PY" ]; then
  UV=$(command -v uv || echo "$HOME/.local/bin/uv"); mkdir -p "$FLOCK_WORK/flock-circuit"
  (exec 8>"$FLOCK_WORK/flock-circuit/venv.lock"; flock 8; [ -x "$PY" ] || { "$UV" venv -q --python 3.12 "$FLOCK_WORK/flock-circuit/py" \
    && "$UV" pip install -q --python "$PY" numpy blake3; })
fi
export PYTHONPATH=$REPO:$REPO/verity/protocols/proofs/flock/prover/python:$REPO/adapters/vllm:$REPO/tools/verity/circuit_check/src:$REPO/benchmarks/numerical/python
COLD() { "$PY" benchmarks/one_stage/cold_sessions.py "$@"; }

W=$SCR/window/audit/window; IFACE=$SCR/window/window/circuit.json
unpack() {
  [ -s "${COLD_ARCHIVE:?}" ] || { log "no archive at $COLD_ARCHIVE"; exit 1; }
  rm -rf "$SCR/window"; mkdir -p "$SCR/window"
  local t0; t0=$(date +%s)
  if command -v zstd > /dev/null; then zstd -dc -T2 "$COLD_ARCHIVE" | tar -xf - -C "$SCR/window"
  else "$PY" -c 'import sys, tarfile, zstandard
with open(sys.argv[1], "rb") as f, tarfile.open(fileobj=zstandard.ZstdDecompressor().stream_reader(f), mode="r|") as t:
    t.extractall(sys.argv[2], filter="tar")' "$COLD_ARCHIVE" "$SCR/window"; fi
  log "unpacked $COLD_ARCHIVE ($(du -sh "$SCR/window" | cut -f1)) in $(( $(date +%s) - t0 )) s"
  COLD plan --window "$W" --members "${COLD_MEMBERS:?}" | tee "$O/plan.txt"
  [ "${PIPESTATUS[0]}" = 0 ] || { log "the window's plan names missing files (plan.txt)"; return 1; }
  [ -s "$IFACE" ] || { log "no registered circuit record at $IFACE"; return 1; }
}

if [ -x /workspace/verity-guest/bin/gpu-lease ] && [ -e /run/gpu-lease/pool.gpus ]; then
  LEASE=(env GPU_LEASE_DIR=/run/gpu-lease GPU_LEASE_ALLOWED_FILE=/run/gpu-lease/pool.gpus /workspace/verity-guest/bin/gpu-lease)
else
  LEASE=(bash "$REPO/tools/research/src/research/pods/sh/gpu_lease.sh")
fi
others() {   # the cards other runs' leases hold (kernels' fused.sh)
  local n; n=$(ls "${GPU_LEASE_DIR:-/run/gpu-lease}" 2>/dev/null | sed -n 's/^\([0-9]*\)\.lock$/\1/p' | sort -n | tail -1)
  GPU_LEASE_COUNT=$(( ${n:--1} + 1 )) bash "$REPO/tools/research/src/research/pods/sh/gpu_lease.sh" status 2>/dev/null \
    | grep -E '^[0-9]+ held' | grep -v " run=${RESEARCH_RUN_ID:-no-run-id} " | sed 's/ cmd=.*//' | cut -c1-200
}
apps() {   # every card's compute processes, "GPU i pid P M"
  local map out u p m i
  map=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | tr -d ' ') && [ -n "$map" ] \
    && out=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | tr -d ' ') \
    || { echo "nvidia-smi gave no answer"; return; }
  echo "$out" | while IFS=, read -r u p m; do
    [ -n "$u" ] || continue; i=$(echo "$map" | awk -F, -v u="$u" '$2 == u {print $1}')
    echo "GPU ${i:-?} pid $p $m"
  done
}
card_free() {   # card COLD_CARD holds no other run's lease and no compute process
  local o
  o=$(others | awk -v c="${COLD_CARD:?}" '$1 == c || $1 !~ /^[0-9]+$/'); [ -z "$o" ] || { log "REFUSED: another run's lease holds card $COLD_CARD: $o"; return 1; }
  o=$(apps | awk -v c="$COLD_CARD" '$2 == c || $2 !~ /^[0-9]+$/'); [ -z "$o" ] || { log "REFUSED: card $COLD_CARD is not free of compute processes: $o"; return 1; }
}

if [ "$STAGE" = probe ]; then
  log "probe: archive $(ls -l "${COLD_ARCHIVE:?}" | awk '{print $5}') B, sha256 $(sha256sum "$COLD_ARCHIVE" | cut -c1-16)"
  unpack; rc=$?
  for f in verity/protocols/proofs/flock/prover/pod/70-class-sweep.sh benchmarks/proof_throughput/verify.sh benchmarks/one_stage/member_seconds.py; do
    [ -f "$f" ] || { log "missing $f"; rc=1; }
  done
  KEY=$(cat verity/protocols/proofs/flock/prover/pod/20-gpu-link.sh verity/protocols/proofs/flock/prover/pod/60-circuit.sh verity/protocols/proofs/flock/prover/*.patch verity/protocols/proofs/flock/prover/cuda_*_patch.py \
    $(find verity/protocols/proofs/flock/prover/cuda verity/protocols/proofs/flock/prover/live -type f | LC_ALL=C sort) | sha256sum | cut -c1-16)
  LKEY=$("$PY" verity/protocols/proofs/flock/prover/pod/lake_closure.py key verity/protocols/proofs/flock/verifier 2>&1 | tail -1)
  {
    echo "prover build key $KEY: $(ls -d "$FLOCK_WORK"/flock-circuit/bin-"$KEY"-* 2>/dev/null | tr '\n' ' ')"
    echo "Lean build key $LKEY: $(ls "$PT_WORK/lean-$LKEY/.lake/build/bin/" 2>/dev/null | tr '\n' ' ')"
    echo "nvcc 13.3: $(ls /usr/local/cuda-13.3/bin/nvcc "$FLOCK_WORK"/cuda-root/usr/local/cuda-13.3/bin/nvcc 2>/dev/null | tr '\n' ' ')"
    echo "lake: $(command -v lake || ls "$HOME"/.elan/bin/lake 2>/dev/null)  uv: $(command -v uv || ls "$HOME"/.local/bin/uv 2>/dev/null)  zstd: $(command -v zstd)"
    echo "lease: ${LEASE[*]}"; echo "pool: $(cat /run/gpu-lease/pool.gpus 2>/dev/null | tr '\n' ' ')"
    echo "other leases:"; others; echo "compute processes:"; apps
  } | tee -a "$O/job.log"
  COLD prove --help > /dev/null && COLD verify --help > /dev/null && log "cold_sessions.py parses" || rc=1
  rm -rf "$SCR"
  log "probe: rc $rc"; exit "$rc"
fi

for c in "$FLOCK_WORK/cuda-root" /workspace/cache/flock/cuda-root; do
  [ ! -x /usr/local/cuda-13.3/bin/nvcc ] && [ -z "${CUDA_TK:-}" ] && [ -x "$c/usr/local/cuda-13.3/bin/nvcc" ] && export CUDA_TK=$c/usr/local/cuda-13.3
done
[ -x /usr/local/cuda-13.3/bin/nvcc ] || [ -n "${CUDA_TK:-}" ] || { log "no CUDA 13.3 here (zk_cost.sh unpacks one)"; exit 1; }
C13=${CUDA_TK:-/usr/local/cuda-13.3}; export LD_LIBRARY_PATH=$C13/compat:$C13/lib64:${LD_LIBRARY_PATH:-}
OUT=$O/build bash verity/protocols/proofs/flock/prover/pod/70-class-sweep.sh BUILD_ONLY=1 GPU=1 SM="${COLD_SM:-90}" > "$O/build.txt" 2>&1 &
OUT=$O/lean bash benchmarks/proof_throughput/verify.sh BUILD_ONLY=1 PT_WORK="$PT_WORK" > "$O/lean-build.txt" 2>&1 &
wait
BIN=$(sed -n 's/^BINARY=//p' "$O/build.txt" | tail -1)
LEAN=$(sed -n 's/.*BUILD_ONLY: \(\S*\) .*/\1/p' "$O/lean/job.log" 2>/dev/null | tail -1)
FW=${LEAN%/*}/flock-firewall
if [ -x "$LEAN" ] && [ ! -x "$FW" ]; then
  command -v lake > /dev/null || export PATH="$HOME/.elan/bin:$PATH"
  (exec 8>"$PT_WORK/lean.lock"; flock 8; [ -x "$FW" ] || (cd "$(readlink -f "${LEAN%/build/bin/*}")/.." && lake build flock-firewall)) \
    > "$O/lake-firewall.log" 2>&1
fi
for p in "$BIN" "$LEAN" "$FW" "$PY"; do
  [ -x "$p" ] || { log "missing: '$p'"; tail -20 "$O/build.txt" "$O/lean-build.txt" "$O/lake-firewall.log" 2>/dev/null; exit 1; }
done
sha256sum "$BIN" "$LEAN" "$FW" | tee "$O/binaries.sha256" | tee -a "$O/job.log"
[ "$STAGE" = build ] && { log "build: done"; exit 0; }

card_free || exit 11
trap 'rm -rf "$SCR"' EXIT
unpack || exit 1
S=$SCR/sessions; mkdir -p "$S"
read -r WBIN WLEAN WFW < <("$PY" benchmarks/one_stage/member_seconds.py wrap "$SCR/stamped" "$BIN" "$LEAN" "$FW" | tr '\n' ' ')
export GPU_LEASE_WHO=${GPU_LEASE_WHO:-proofs/cold-sessions/${RESEARCH_RUN_ID:-local}}
END=${COLD_END:-$(( $(date +%s) + 86400 ))}
est() { local v="COLD_EST_$1_$2" w="COLD_EST_$1"; echo "${!v:-${!w:-1800}}"; }
keep() {
  (cd "$S" && find . -type f -size -8M ! -name circuit.txt ! -name 'inst*.bin' ! -name 'pub*.bin' ! -path '*/archive/*' \
    \( ! -path '*/sessions/*/*' -o -name session.json \) -print0 | xargs -0 -r -I{} cp --parents {} "$O/sessions/")
  cp "$SCR/stamped/stamps.txt" "$O/" 2>/dev/null
}
mkdir -p "$O/sessions"
VQ=$SCR/verify.lock
rc=0
log "run: members $COLD_MEMBERS, arms ${COLD_ARMS:-gpu,cpu}, card $COLD_CARD, until $(date -u -d "@$END" +%H:%MZ)"
for k in ${COLD_MEMBERS//,/ }; do
  for arm in ${COLD_ARMS//,/ }; do
    [ -e "$FLOCK_WORK/cold.stop" ] && { log "stopped by $FLOCK_WORK/cold.stop"; break 2; }
    e=$(est "$arm" "$k"); now=$(date +%s)
    if [ $(( now + e )) -gt "$END" ]; then log "m$k $arm: skipped, its estimate ${e} s would end past $(date -u -d "@$END" +%H:%MZ)"; continue; fi
    args=(--window "$W" --interface "$IFACE" --member "$k" --arm "$arm" --out "$S")
    mm=$(( (e + 59) / 60 + 15 )); left=$(( (END - now) / 60 )); [ "$mm" -gt "$left" ] && mm=$left
    log "m$k $arm: start (estimate ${e} s, limit ${mm} min)"
    if [ "$arm" = gpu ]; then
      card_free || { rc=11; break 2; }
      "${LEASE[@]}" 1 --on "$COLD_CARD" --cores "$CPUS" --max-min "$mm" -- bash -c '"$@" --device "$CUDA_VISIBLE_DEVICES" --card "$GPU_LEASE_UUID"' _ \
        "$PY" benchmarks/one_stage/cold_sessions.py prove "${args[@]}" --bin "$WBIN" --firewall "$WFW" \
        --stamps "$SCR/stamped/stamps.txt" --timeout $(( mm * 60 - 30 )) >> "$O/cold.log" 2>> "$O/cold.err"
      x=$?
    else
      taskset -c "$CPUS" "$PY" benchmarks/one_stage/cold_sessions.py prove "${args[@]}" --bin "$WBIN" --firewall "$WFW" \
        --stamps "$SCR/stamped/stamps.txt" --timeout $(( mm * 60 )) >> "$O/cold.log" 2>> "$O/cold.err"
      x=$?
    fi
    log "m$k $arm: exit $x; $(grep -c "^COLD .*\"k\": $k, .*\"arm\": \"$arm\"" "$O/cold.log") record"
    [ "$x" = 75 ] && { log "the lease refused card $COLD_CARD (75)"; rc=11; break 2; }
    # the verdict on the Lean CPUs, one at a time, beside the next session
    ( flock 7; taskset -c "$LCPUS" "$PY" benchmarks/one_stage/cold_sessions.py verify "${args[@]}" --lean "$WLEAN" \
        >> "$O/cold.log" 2>> "$O/cold.err" ) 7> "$VQ" &
  done
done
log "sessions done; waiting for the verdicts"
wait
COLD table --out "$S" > /dev/null; cp "$S/table.md" "$S/table.json" "$O/" 2>/dev/null
keep
log "cold-sessions: done, rc $rc"
exit "$rc"
