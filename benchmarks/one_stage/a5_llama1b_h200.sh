#!/usr/bin/env bash
# One-stage audit, run A5, over a GPU-served window: Llama-3.2-1B int8 batch-invariant eager on one H200 (ROW, default the B4
# I256/O32 h200 row), its Build and Match of record and a deferred config-run Commit under rows-sha512 whose
# --serving-rows window (llama32_1b_rope_d64_l0.json: every RoPEHead_v1{D=64} head of layer 0 at steps 1-7, by subset:1) is
# committed at its C2 site before the bundle is sealed (`c2_replay.serving_rows`): a `served` window under the Commit's 64-byte
# run root. Then the Commit's C2 replay on CPU from the sealed bundle, and a5.sh (WINDOW=served, A5_ROW_DIR) on that bundle and
# window, in the same job, so the window's private bodies (m*-private.bin, v*-private.bin: rows and salts) never leave the node.
#   research run --on vy-dc-h200 --project verity --source TREE --cwd clone --mem-gb 192 --timeout 6h \
#     --env GPU_LEASE_WHO=LANE -- bash benchmarks/one_stage/a5_llama1b_h200.sh
# Needs on the node what served_verdicts.sh does: vLLM's venv /workspace/cp/venv312, the int8 checkpoint in /workspace/hf, the
# sm_90 taps under /workspace/research/vllm-composed-h200int8 and gpu-lease (the row leases the GPU for
# its Match capture and its Commit's engine only). The uv venv, Lean's verifiers and M0's flock-circuit build in the background
# while the GPU stages run (M0 in A5_ROOT/flock-work, kept for a rerun). The row dir and the window stay on the node under
# `research keep`; $OUT keeps the row's records, the window's public files, a5.sh's out ($OUT/a5) and summary.txt, whose A5 line
# is the audit's verdict and whose REGISTERED line the window's kind, commitment and run root length.
set -uo pipefail
OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}; mkdir -p "$OUT"
ROW=${ROW:-llama32-1b__int8__h200__tp1__b4__i256__o32__mixed__greedy__bi-eager}
ROLE=LLAMA32_1B_INT8 REPO=verity-int8/Llama-3.2-1B-INT8 REV=15b0976d400d796ee589e7737d03a919526bf800
TREE=$PWD
H=/workspace/research/vllm-composed-h200int8
FA3=${FA3_SO:-$H/fa2/build/fa3_matReq/verity_fa3_matReq.so} FA3G=${FA3G_SO:-$H/fa2/build/fa3_matReqG/verity_fa3_matReqG.so}
NT=${NORM_TAP_SO:-$H/norm_tap/build/verity_norm_tap.so}
PART=$TREE/benchmarks/one_stage/llama32_1b_rope_d64_l0.json
ROOT=${A5_ROOT:-/workspace/research/vllm-sha512-a5}
BASE=$ROOT/${RESEARCH_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
D=$BASE/$ROW WD=$BASE/serving-rows FW=$ROOT/flock-work
VENV=/workspace/cp/venv312
CLEAN_PATH=$HOME/.local/bin:$HOME/.cargo/bin:$HOME/.elan/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
for f in "$FA3" "$FA3G" "$NT" "$VENV/bin/python" "$PART"; do [ -e "$f" ] || { echo "a5_llama1b_h200.sh: missing $f" >&2; exit 2; }; done
mkdir -p "$D" "$ROOT/cache" "$BASE/sealed" "$FW"

# the CPU side's builds, in the background: the tree's venv, Lean's flock-verify, flock-firewall and sampled-proofs-verifier, M0
( export PATH=$CLEAN_PATH; unset PYTHONPATH PY PY312 VIRTUAL_ENV; cd "$TREE" || exit 1
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
  command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
  uv sync --all-packages --extra torch-cpu || exit 5
  (cd verity/core/service/flock && lake build flock-verify flock-firewall) || exit 7
  (cd verity/core/service/sampled_proofs && lake build sampled-proofs-verifier) || exit 7
  RESEARCH_RUN_DIR=$OUT/prebuild FLOCK_WORK=$FW MODE=build GPU=0 bash verity/ml/flock/pod/60-circuit.sh || exit 6
) > "$OUT/prebuild.log" 2>&1 &
PB=$!

export PY=$VENV/bin/python PY312=$VENV/bin/python HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
if [ -z "${CUDA_HOME:-}" ]; then
  for c in /usr/local/cuda-12.9 /usr/local/cuda-12 /usr/local/cuda-13.0 /usr/local/cuda; do [ -x "$c/bin/nvcc" ] && { CUDA_HOME=$c; break; }; done
fi
export CUDA_HOME TORCH_EXTENSIONS_DIR=$ROOT/cache/torch-extensions TRITON_CACHE_DIR=$ROOT/cache/triton VLLM_CACHE_ROOT=$ROOT/cache/vllm
export HIDDEN_GPU_BUILD=$ROOT/cache/jit/hidden_gpu NATIVE_COLLECT_BUILD=$ROOT/cache/jit/nc
export PATH=$VENV/bin:$CUDA_HOME/bin:$PATH
export PYTHONPATH=$TREE:$TREE/integrations/vllm:$TREE/tools/research/src
export GPU_LEASE=${GPU_LEASE:-gpu-lease} GPU_LEASE_WHO=${GPU_LEASE_WHO:-vllm-sha512}
export VERITY_WEIGHTS_HASH=${VERITY_WEIGHTS_HASH:-device-checked}
unset CONFIG_RUN REPLAY_K REPLAY_DEFERRED SERVING_ROWS
rs() { "$PY" -m research "$@" 2>&1 || research "$@" 2>&1; }
{ id; hostname; git -C "$TREE" rev-parse HEAD 2>/dev/null; echo "row dir $D"; echo "CUDA_HOME $CUDA_HOME"; nvidia-smi -L
  sha256sum "$FA3" "$FA3G" "$NT"; } | tee "$OUT/machine.txt"
rs keep "$ROOT" --owner @vllm --expires 7d --priority normal \
  --why "A5 over a GPU-served rows-sha512 window (Llama-3.2-1B int8 H200): the row dir, its sealed bundle and window" | tail -2

row() {  # NAME ARGS...: one `verity-vllm row run` of the row dir, its output in $OUT/NAME.out
  local name=$1; shift
  echo "$(date -u +%H:%M:%SZ) $name: $*" | tee -a "$OUT/steps.txt"
  (cd "$TREE/integrations/vllm" && "$PY" -m verity_vllm.pipeline.cli row run "$ROW" "$ROLE" "$REPO" "$REV" --sweep-dir "$BASE" \
     --hidden-so-fa3 "$FA3" --hidden-so-fa3-guarded "$FA3G" --norm-tap-so "$NT" "$@") > "$OUT/$name.out" 2>&1
  local rc=$?
  echo "$(date -u +%H:%M:%SZ) $name rc=$rc" | tee -a "$OUT/steps.txt"
  tail -5 "$OUT/$name.out"
  return $rc
}
records() {  # the row's records into $OUT (never the bundle, the store or the capture)
  for f in stages.txt row.log verdict.json build_summary.json manifest.json config_record.json gpu_lease.json effective_config.json \
    commit.log match.log build.log replay.log manifest.log; do cp "$D/$f" "$OUT/" 2>/dev/null; done
  mkdir -p "$OUT/commit"
  find "$D/commit" -maxdepth 1 -type f -size -50M -exec cp {} "$OUT/commit/" \; 2>/dev/null
  [ -d "$D/commit/replay_bundle_p0" ] && mkdir -p "$OUT/commit/replay_bundle_p0" \
    && cp "$D/commit/replay_bundle_p0/MANIFEST.json" "$D/commit/replay_bundle_p0/context.json" "$OUT/commit/replay_bundle_p0/" 2>/dev/null
  du -sh "$D" "$D"/* "$WD" 2>/dev/null > "$OUT/sizes.txt"
}
window_off() {  # the Commit's window out of the run dir (custody uploads it), its public files and error into $OUT
  local rd=${RESEARCH_RUN_DIR:-.}/serving-rows/$ROW
  [ -d "$rd" ] && mv "$rd" "$WD"
  [ -d "$WD" ] || return 0
  mkdir -p "$OUT/window"
  cp "$WD"/*.json "$WD"/m*-public.bin "$WD"/error.txt "$OUT/window/" 2>/dev/null
  return 0
}
stop() { window_off; records; echo "STOPPED at $1" | tee -a "$OUT/summary.txt"; exit "$2"; }

row build --stages build --config-run 1 --replay-k 460 || stop build 10
row match --stages match --config-run 0 || stop match 11
row commit --stages commit --config-run 1 --replay-k 460 --replay-deferred 1 --serving-rows "$PART" || stop commit 12
window_off
[ -s "$WD/registration.json" ] || stop "commit (no window at $WD: $(head -c 400 "$WD/error.txt" 2>/dev/null))" 13
B=$D/commit/replay_bundle_p0
[ -s "$B/MANIFEST.json" ] || stop "commit (no sealed bundle $B)" 13
cp -al "$B" "$BASE/sealed/" || stop "linking the sealed bundle aside" 14
row replay --stages replay --config-run 1 --replay-k 460; rr=$?
[ -d "$B" ] || cp -al "$BASE/sealed/replay_bundle_p0" "$D/commit/" || stop "restoring the sealed bundle" 15
(cd "$B" && sha256sum MANIFEST.json) | tee "$OUT/bundle.sha256"
records

echo "$(date -u +%H:%M:%SZ) a5: waiting on the builds" | tee -a "$OUT/steps.txt"
wait "$PB"; pb=$?
echo "$(date -u +%H:%M:%SZ) builds rc=$pb" | tee -a "$OUT/steps.txt"
[ "$pb" = 0 ] || { tail -30 "$OUT/prebuild.log"; stop "the builds (rc $pb, prebuild.log)" 16; }
( export PATH=$CLEAN_PATH; unset PYTHONPATH PY PY312 VIRTUAL_ENV; cd "$TREE" || exit 1
  OUT=$OUT/a5 COMMIT_SCHEME=rows-sha512 WINDOW=served A5_ROW_DIR=$D A5_PARTITION=$PART A5_WINDOW=$WD A5_FLOCK_WORK=$FW \
    bash benchmarks/one_stage/a5.sh ) > "$OUT/a5.out" 2>&1
ar=$?
echo "$(date -u +%H:%M:%SZ) a5 rc=$ar" | tee -a "$OUT/steps.txt"

"$PY" - "$WD/registration.json" "$D/commit" <<'PY' | tee -a "$OUT/summary.txt"
import json, pathlib, sys
w = (json.loads(pathlib.Path(sys.argv[1]).read_text()).get("window") or {})
run = str(w.get("run") or "")
roots = set()
for p in pathlib.Path(sys.argv[2]).glob("*.json*"):
    for line in p.read_text(errors="replace").splitlines():
        if '"weight_registration"' in line:
            def walk(x):
                if isinstance(x, dict):
                    if isinstance(x.get("weight_registration"), dict) and x["weight_registration"].get("root"):
                        roots.add(x["weight_registration"]["root"])
                    for v in x.values():
                        walk(v)
                elif isinstance(x, list):
                    for v in x:
                        walk(v)
            try:
                walk(json.loads(line))
            except ValueError:
                pass
print(f"REGISTERED kind={w.get('kind')} commitment={w.get('commitment')} run_root_bytes={len(run) // 2} run={run[:32]}...")
print("WEIGHTS root_bytes=" + ",".join(sorted({str(len(r) // 2) for r in roots}) or ["?"]))
PY
grep -a '^A5 ' "$OUT/a5/a5.log" 2>/dev/null | tail -1 | cut -c1-4000 | tee -a "$OUT/summary.txt"
tail -8 "$OUT/stages.txt" 2>/dev/null | tee -a "$OUT/summary.txt"
echo "replay rc=$rr a5 rc=$ar" | tee -a "$OUT/summary.txt"
exit $(( rr != 0 ? 17 : ar != 0 ? 18 : 0 ))
