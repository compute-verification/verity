#!/usr/bin/env bash
# The wide level-0 commitment leaf against today's leaf on one card, on statement.toml's statement (wleaf.py):
#   research run --on NODE --project verity --source . --cwd source -- bash benchmarks/kernels/wide_leaf/run.sh [dry]
#
#   card    (the default) on a GPU node, in one of its GPU slots: the prover built by the steady harness's setup
#           (benchmarks/harness/setup/flock_prover.sh, CUDA 13.3) in FLOCK_WORK on the node's disk; the statement
#           fetched from the store into a scratch where server_rate.py looks (z/NAME/c{n}/stage) and its circuit held to
#           its pin; then one gpu-lease of one card near the run's CPUs, pinned to them, for LEASE_MIN (100) minutes,
#           running STEPS (a-own,b-wide,c-wide,d-own,e-own,f-wide; SESSIONS 120 timed sessions after WARM 3 warm, one
#           prover); after the lease each step's ten checks and the result (wleaf.py record). Under a run that already
#           holds its card (GPU_LEASE_UUID set) the steps run on it without a second lease.
#   dry     on a CPU node with nothing timed: the same setup with SM=90, server_rate.py's and wleaf.py's argument
#           parsing under the setup's Python, and the statement's small files fetched and held to its pin.
#
# $OUT: setup/ (the setup's logs), host.json, steps/TAG/ (rate/, the card's cotenants, step.json), summary.json, run.log.
set -uo pipefail
MODE=${1:-card}
RD=${RESEARCH_RUN_DIR:?run under research run}; O=$RD/out; mkdir -p "$O"
ROOT=$PWD; HERE=$ROOT/benchmarks/kernels/wide_leaf
STEPS=${STEPS:-a-own,b-wide,c-wide,d-own,e-own,f-wide}
SESSIONS=${SESSIONS:-120}; WARM=${WARM:-3}; AHEAD=${AHEAD:-2}; STEP_MIN=${STEP_MIN:-12}; LEASE_MIN=${LEASE_MIN:-100}
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$O/run.log"; }
toml() { python3 -c 'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))[sys.argv[2]])' "$HERE/statement.toml" "$1"; }
NAME=$(toml name); PIN=$(toml pin); ART=$(toml art); ART_PATH=$(toml path)
rfetch() {   # rfetch STORE ARGS...: the statement's files through the tree's research CLI, its store in scratch
  local st=$1; shift
  PYTHONPATH=$ROOT/tools/research/src:$ROOT/tools/config:${PYTHONPATH:-} python3 -m research data fetch "$ART" --store "$st" "$@"
}

cuda_fetch() {   # a node with no CUDA 13.3 and no apt source for it: the packages cuda-13.3.sh takes, from NVIDIA's
  # repository by name and version, each held to its Packages SHA256, unpacked to FLOCK_WORK/cuda-root, where it finds them
  [ -x /usr/local/cuda-13.3/bin/nvcc ] || [ -x "$FLOCK_WORK/cuda-root/usr/local/cuda-13.3/bin/nvcc" ] && return 0
  apt-cache show cuda-nvcc-13-3 > /dev/null 2>&1 && return 0
  local R=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64 d=$FLOCK_WORK/cuda-debs pv p v f s
  mkdir -p "$d" && curl -fsS "$R/Packages.gz" | gunzip > "$d/Packages" || { log "cuda: NVIDIA's Packages could not be fetched"; return 1; }
  for pv in cuda-cudart-13-3=13.3.29-1 cuda-cudart-dev-13-3=13.3.29-1 cuda-culibos-dev-13-3=13.3.33-1 cuda-driver-dev-13-3=13.3.29-1 \
            cccl-13-3=13.3.3.4.1-1 cuda-crt-13-3=13.3.73-1 cuda-nvcc-13-3=13.3.73-1 libnvptxcompiler-13-3=13.3.73-1 libnvvm-13-3=13.3.73-1; do
    p=${pv%%=*}; v=${pv#*=}
    read -r f s < <(awk -v p="$p" -v v="$v" '/^Package: /{a=$2} /^Version: /{b=$2} /^Filename: /{f=$2} /^SHA256: /{s=$2} /^$/{if(a==p&&b==v){print f, s; exit} a=b=f=s=""}' "$d/Packages")
    [ -n "${f:-}" ] || { log "cuda: $p $v is not in NVIDIA's Packages"; return 1; }
    curl -fsS -o "$d/${f##*/}" "$R/${f#./}" && [ "$(sha256sum "$d/${f##*/}" | cut -d' ' -f1)" = "$s" ] \
      || { log "cuda: $p $v did not fetch or is not its SHA256"; return 1; }
    dpkg-deb -x "$d/${f##*/}" "$FLOCK_WORK/cuda-root" || return 1
  done
  rm -rf "$d"; log "cuda: CUDA 13.3's packages from $R unpacked to $FLOCK_WORK/cuda-root"
}

setup() {
  # the setup writes under RESEARCH_RUN_DIR/out/setup, and it and the CUDA unpacker it sources read unset names
  export RESEARCH_RUN_DIR=$O/setup-run; mkdir -p "$RESEARCH_RUN_DIR"; set +u
  # shellcheck disable=SC1091
  . benchmarks/harness/setup/flock_prover.sh || { export RESEARCH_RUN_DIR=$RD; log "REFUSED: the setup failed ($O/setup-run/out/setup)"; exit 4; }
  set -u; export RESEARCH_RUN_DIR=$RD
  export PYTHONPATH=${PYTHONPATH:-}:$ROOT/benchmarks/harness:$ROOT/tools/config PY=$VERITY_BENCH_STEADY_PY
  export CARD_TOOLKIT; CARD_TOOLKIT=$(nvcc --version | sed -n 's/.*, V\([0-9.]*\)$/\1/p' | tail -1)
  log "setup: prover $VERITY_BENCH_STEADY_BIN sha256 $(sha256sum "$VERITY_BENCH_STEADY_BIN" | cut -c1-16); nvcc $CARD_TOOLKIT; $($PY --version)"
}

case $MODE in
dry)
  export FLOCK_WORK; FLOCK_WORK=$(mktemp -d); trap 'rm -rf "$FLOCK_WORK"' EXIT; export SM=${SM:-90}
  log "dry: $(git rev-parse --short=12 HEAD 2>/dev/null || echo "${RESEARCH_SOURCE_SHA:-?}"), no card; $NAME, pin ${PIN:0:16}"
  setup
  "$PY" benchmarks/one_stage/server_rate.py --help > "$O/server-rate-help.txt" 2>&1 \
    && log "dry: server_rate.py --help: $(grep -c -- '--wide-leaf' "$O/server-rate-help.txt") line(s) of --wide-leaf" || log "dry: server_rate.py --help FAILED"
  for c in lease host step record; do "$PY" "$HERE/wleaf.py" "$c" --help > /dev/null 2>&1 && log "dry: wleaf.py $c --help" || log "dry: wleaf.py $c --help FAILED"; done
  rfetch "$FLOCK_WORK/store" --to "$FLOCK_WORK/st" --path "$ART_PATH/stage.json" --path "$ART_PATH/pin.txt" > "$O/fetch.txt" 2>&1
  sj=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("circuit_sha512",""))' "$FLOCK_WORK/st/$ART_PATH/stage.json" 2>/dev/null)
  [ "$(tr -d ' \n' < "$FLOCK_WORK/st/$ART_PATH/pin.txt" 2>/dev/null)" = "$PIN" ] && [ "$sj" = "$PIN" ] \
    && log "dry: the statement's pin.txt and stage.json give its pin" || { log "dry: the statement is NOT its pin (stage.json ${sj:0:16})"; exit 1; }
  log "dry: done"; exit 0 ;;
inside)
  END=$(( $(date +%s) + LEASE_MIN * 60 - 120 ))
  export CARD_DRIVER; CARD_DRIVER=$(nvidia-smi --id="${CUDA_VISIBLE_DEVICES:-0}" --query-gpu=driver_version --format=csv,noheader | tr -d ' ')
  "$PY" "$HERE/wleaf.py" host 2>&1 | tee -a "$O/run.log"; [ "${PIPESTATUS[0]}" = 0 ] || { log "REFUSED: the lease is not of one card"; exit 2; }
  for s in ${STEPS//,/ }; do
    [ $(( (END - $(date +%s)) / 60 )) -ge "$STEP_MIN" ] || { log "step $s: skipped, under $STEP_MIN min left in the lease"; continue; }
    w=; case $s in *-wide) w=--wide ;; esac
    "$PY" "$HERE/wleaf.py" step "$s" $w --staged "$STAGED" --type "$NAME" --pin "$PIN" --scratch "$SCRATCH" \
      --sessions "$SESSIONS" --warm "$WARM" --ahead "$AHEAD" 2>&1 | tee -a "$O/run.log"
  done
  exit 0 ;;
card)
  export FLOCK_WORK=${FLOCK_WORK:-/workspace/tmp/kernels-wide-leaf}; mkdir -p "$FLOCK_WORK/scratch"
  log "card: $(git rev-parse HEAD 2>/dev/null || echo "${RESEARCH_SOURCE_SHA:-?}") on $(hostname); $NAME; steps $STEPS"
  cuda_fetch || { log "REFUSED: no CUDA 13.3 here"; exit 4; }
  setup
  export SCRATCH; SCRATCH=$(mktemp -d "$FLOCK_WORK/scratch/run-XXXXXX"); export STAGED=$SCRATCH/z
  trap 'rm -rf "$SCRATCH"' EXIT
  rfetch "$SCRATCH/store" --to "$SCRATCH/st" --path "$ART_PATH/*" > "$O/fetch.txt" 2>&1 || { log "REFUSED: the statement's fetch failed"; exit 4; }
  n=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["n"])' "$SCRATCH/st/$ART_PATH/stage.json")
  STMT=$STAGED/$NAME/c$n/stage
  mkdir -p "$(dirname "$STMT")" && mv "$SCRATCH/st/$ART_PATH" "$STMT" && rm -rf "$SCRATCH/store"
  [ "$(sha512sum "$STMT/circuit.txt" | cut -d' ' -f1)" = "$PIN" ] || { log "REFUSED: the statement's circuit is not its pin ${PIN:0:16}"; exit 4; }
  log "statement: $NAME x $n, its circuit's SHA-512 computed here is its pin, $(du -sh "$STMT" | cut -f1)"
  if [ -n "${GPU_LEASE_UUID:-}" ]; then bash "$0" inside; rc=$?
  else "$PY" "$HERE/wleaf.py" lease --max-min "$LEASE_MIN" -- bash "$0" inside; rc=$?; fi
  log "the steps ended (exit $rc)"
  "$PY" "$HERE/wleaf.py" record --stage "$STMT" 2>&1 | tee -a "$O/run.log"
  exit "$rc" ;;
*) echo "run.sh [card|dry|inside]" >&2; exit 2 ;;
esac
