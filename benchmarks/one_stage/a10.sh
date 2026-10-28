#!/usr/bin/env bash
# One-stage audit, run A10, on a CPU machine: A11's kinds and A9's routed-expert launches on rank 0's committed values (a10.py),
# on hidden_zk's builds as a11.sh runs them. Two modes:
#   values reads the tree, so it runs from the source, on the node holding the Build's row dir:
#     A10_MODE=values A10_MEM_GB=256 bash benchmarks/one_stage/a10.sh
#     (a job run from the source tree, 256 GB, up to 4 h)
#   It fetches rank 0's replay bundle, the Commit's records, and A11's and A9's slices from the store into $A10_CACHE (kept on
#   the node), reads the Build's rank 0 Program and the required manifest from the Build's row dir ($A10_ROW, on the
#   machine that built it), runs `a10.py values`, and keeps $OUT/compare.json, the logs and $OUT/saved-a11.tar, $OUT/saved-a9.tar
#   (a11.py --load-values' directories).
#   audit builds the sampled-proofs Lean in the tree, so it runs from a clone:
#     A10_MODE=audit A10_MEM_GB=128 A10_VALUES=art:ID A10_SET=a11 A10_ARGS="--only K1,K2 --tamper none" bash benchmarks/one_stage/a10.sh
#     (a job run from a clone of the tree, 128 GB, up to 4 h)
#   It runs `a10.py audit` (a11.py with the committed window) on $A10_SET's saved directory from the values run's record
#   (A10_VALUES), its records in $OUT/a11 as a11.sh keeps them.
# env: A10_MODE  A10_MEM_GB (the job's --mem-gb)  A10_ARGS (more a10.py arguments)  A10_ROW  A10_CACHE  A11_CACHE (the build cache)
set -uo pipefail
export HZ_CONFIGS=a10 HZ_MEM_GB=${A10_MEM_GB:?}
export HZ_CACHE=${A11_CACHE:-$HOME/.cache/verity-a11}
mkdir -p "$HZ_CACHE"
source benchmarks/pouw/hidden_zk/job.sh
export PATH=$HOME/.elan/bin:$PATH
export PYTHONPATH=$SRC:$SRC/verity/ml/flock/python:$SRC/integrations/vllm:$SRC/tools/verity/circuit_check/src:$SRC/benchmarks/numerical/python
C=${A10_CACHE:-$HOME/.cache/verity-a10}
mkdir -p "$C"
# the store CLI from the tree (a node's job has none on its PATH), with the run's custody key in its environment only
R() { PYTHONPATH=$SRC/benchmarks/proof_throughput python3 -c 'import os, sys
from verity_proof_throughput.store import store_env
os.execvpe(sys.executable, [sys.executable, "-m", "research", *sys.argv[1:]], store_env())' "$@"; }
R keep "$HZ_CACHE" --owner @circuits --expires 4d --priority low --root / \
  --why "A10's builds (flock-circuit, flock-verify, the venv), keyed by their inputs" >> "$OUT/job.log" 2>&1

BUNDLE=art:6437727df08be44eba654589780a7d307a4e8038d70b373146eba4b83878034e    # rank 0's replay bundle, Commit r20261019-074856-7c3a
ROWMETA=art:7d815b1d963c7e9cf42aaeff8793901b38a301d10ed8c814d2ffe292b757d9ea   # the row's records (commit/)
BUILD=art:8be49d1e6f506b6a2105dfba80222db6aa6b9542613548c1c9012af1ebcca4f4     # the Build (build/rank0, manifest.json), without the Programs' instances
# the Build's row dir where it ran (vy-glm53-cpu-2), read in place: its rank 0 Program's instances aren't in the art
ROWDIR=${A10_ROW:-/workspace/cp/circuits/glm-aot/r20261018-070557-2172/glm53__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager}
SLICES="art:2767c3a80137ce6cd7782373e1cba8b4231d4598f1b902caaa7bc1604edf09c8 art:22e7456dcc5018b85c511d507ff3c33ffa5be2b1946e3106de1ce5727f0d3106
art:84ae2056669e502abb2751cd38d2125167728fd580193df7f7432338877551ff art:c1ec2634adc0f5b548bec6af41cca82995d11b977fdb67d716895695a0b38cf8"
KINDS=art:9ec1d8b37286bdd5ec3cee7d39da27d97866eece6018d994c4921d068bc49abb     # A11's kinds table (kinds.json)

fetch() {   # fetch ART DEST [GLOB...]: ART's files (those matching the globs) under DEST, once per node; DEST.done marks it whole
  local art=$1 dest=$2; shift 2
  [ -e "$dest.done" ] && return 0
  rm -rf "$dest" && mkdir -p "$dest"
  local paths=() g
  for g in "$@"; do paths+=(--path "$g"); done
  R data fetch "$art" --to "$dest" "${paths[@]}" >> "$OUT/fetch.log" 2>&1 && touch "$dest.done"
}

build_python_torch() {   # build_python_torch DEST: build_python's venv with CPU torch (the replay's `tp.commit` imports it)
  build_python "$1" && uv pip install -q --python "$1/bin/python" --index-url https://download.pytorch.org/whl/cpu "torch>=2.4"
}

case ${A10_MODE:?} in
values)
  PY=$(cached venv numpy2.5.3-blake3-1.0.9-torch-cpu build_python_torch)/bin/python
  [ -x "$PY" ] || { log "a10: no venv"; tail -20 "$OUT/build-venv.log"; exit 2; }
  log "a10: fetching into $C"
  fetch "$BUNDLE" "$C/bundle" &
  fetch "$ROWMETA" "$C/rowmeta" 'commit/runs.jsonl' 'commit/replay_deferred.json' 'commit/summary.json' &
  if [ -d "$ROWDIR/build/rank0" ]; then BD=$ROWDIR/build/rank0 MF=$ROWDIR/manifest.json; else
    fetch "$BUILD" "$C/build" 'build/rank0/build_request_LP256_T31/*' 'manifest.json' & BD=$C/build/build/rank0 MF=$C/build/manifest.json; fi
  fetch "$KINDS" "$C/kinds" 'kinds.json' &
  for a in $SLICES; do fetch "$a" "$C/slice-${a:4:8}" 'out/*.tar' & done
  wait
  for d in bundle rowmeta kinds $(for a in $SLICES; do echo "slice-${a:4:8}"; done); do
    [ -e "$C/$d.done" ] || { log "a10: fetch of $d failed"; tail -20 "$OUT/fetch.log"; exit 2; }
  done
  [ -e "$BD/build_request_LP256_T31/instances.json.gz" ] \
    || { log "a10: $BD has no rank 0 Program instances (the Build art doesn't carry them): set A10_ROW to the Build's row dir"; exit 2; }
  if [ ! -e "$C/values.done" ]; then
    rm -rf "$C/values" && mkdir -p "$C/values"
    for t in "$C"/slice-*/out/*.tar; do tar -xf "$t" -C "$C/values" || { log "a10: can't extract $t"; exit 2; }; done
    touch "$C/values.done"
  fi
  R keep "$C" --owner @circuits --expires 2d --priority low --root / \
    --why "A10's inputs from the store (rank 0's bundle, the Commit's records, the Build, A11/A9 slices)" >> "$OUT/job.log" 2>&1
  log "a10: inputs $(du -sh "$C" | cut -f1), the Build from $BD; values"
  # shellcheck disable=SC2086
  "$PY" benchmarks/one_stage/a10.py values --values "$C/values" --bundle "$C/bundle" --commit "$C/rowmeta/commit" \
    --build "$BD" --manifest "$MF" \
    --roles "$BD/build_request_LP256_T31/derive-report.json" --kinds "$C/kinds/kinds.json" --out "$W/a10" \
    --try RMSNormCudaVector8_v1 ${A10_ARGS:-} 2>&1 | tee -a "$OUT/a10.log" | cut -c1-4000
  rc=${PIPESTATUS[0]}
  cp "$W/a10/compare.json" "$OUT/" 2>/dev/null
  for s in a11 a9; do
    [ -d "$W/a10/saved-$s" ] && tar -cf "$OUT/saved-$s.tar" -C "$W/a10/saved-$s" .
    [ -d "$W/a10/$s" ] && cp -r "$W/a10/$s" "$OUT/$s"
  done
  log "a10: values done, rc $rc"
  exit "$rc"
  ;;
audit)
  setup
  (cd verity/core/service/sampled_proofs && lake build sampled-proofs-verifier) > "$OUT/build-sampled-proofs.log" 2>&1 \
    || { tail -40 "$OUT/build-sampled-proofs.log"; exit 7; }
  S=$W/saved && mkdir -p "$S"
  R data fetch "${A10_VALUES:?}" --to "$W/fetch" --path "out/saved-${A10_SET:?}.tar" >> "$OUT/fetch.log" 2>&1 \
    && tar -xf "$W/fetch/out/saved-$A10_SET.tar" -C "$S" || { log "a10: no saved-$A10_SET.tar in $A10_VALUES"; exit 2; }
  if [ "$A10_SET" = a9 ]; then KARGS=(--kinds "$S/kinds.json"); else
    fetch "$KINDS" "$C/kinds" 'kinds.json'; KARGS=(--kinds "$C/kinds/kinds.json" --try RMSNormCudaVector8_v1); fi
  log "a10: audit of saved-$A10_SET ($(du -sh "$S" | cut -f1))"
  # shellcheck disable=SC2086
  "$PY" benchmarks/one_stage/a10.py audit --bin "$BIN" --lean "$LEAN" --load-values "$S" "${KARGS[@]}" --out "$W/a11" \
    ${A10_ARGS:-} 2>&1 | tee -a "$OUT/a11.log" | cut -c1-4000
  rc=${PIPESTATUS[0]}
  mkdir -p "$OUT/a11"
  (cd "$W/a11" && find . -type f ! -name circuit.txt ! -name 'inst-*' ! -path '*/archive/*' -print0 | xargs -0 -I{} cp --parents {} "$OUT/a11/")
  log "a10: audit done, rc $rc"
  exit "$rc"
  ;;
esac
