#!/usr/bin/env bash
# One-stage audit, run A5, over a GPU-served window: Llama-3.2-1B int8 batch-invariant eager on one H200 (ROW, default the B4
# I256/O32 h200 row), its Build and Match of record and a deferred config-run Commit under rows-sha512 whose
# --serving-rows window (llama32_1b_rope_d64_l0.json: every RoPEHead_v1{D=64} head of layer 0 at steps 1-7, by subset:1; or
# A5_PARTITION's file, whose sampling design gives the sample) is committed at its C2 site before the bundle is sealed
# (`c2_replay.serving_rows`): a `served` window under the Commit's 64-byte run root. Then the Commit's C2 replay on CPU from
# the sealed bundle, and the window's audit, in the same job, so the window's private bodies (m*-private.bin,
# v*-private.bin: rows and salts) never leave the machine. The audit is A5_COINS's:
#  - dev (default): a5.sh (WINDOW=served, A5_ROW_DIR) on the bundle and window: the Lean verifier's sample on in-process development coins,
#    each Flock --zk session on a development challenger of its own, accepted by Flock's per-session Lean verify --zk and
#    the Lean verifier's sample; SPOT's one-stage check (OneStageZk) does not run, and refuses such a case (it names no challenger).
#  - challenger: records_audit.py audit of the window on ONE local challenger session (`--challenger local --deployed-key`:
#    the auditor device's service run on this machine, a development stand-in for the auditor's, its key made for the run at
#    the deployed LMS parameters), at the rate rate:K/N (the partition's subset:K over the window's N proof blocks), every
#    session's coin tree from that session's release log (a5.window_trees), Flock --zk behind the Lean ZK server; then SPOT's
#    one-stage check (spot-one-stage-zk, `Spot.OneStageZk` v2) on the case the audit wrote, whose V2 line is the verdict.
#    A5_PLAIN=1 makes its sessions plain (`records_audit.py --plain`, a5's plain audit) at the partition's sampling design, which must be
#    a stratified rate (stratified-rate:NUM/DEN, which the Commit registers as the window's stratified design): one session
#    on its own keyed coin tree from a development challenger of its own (FC_COINS=tree, FLOCK_CHALLENGER=1), served the
#    challenger session's stratified sample and held to it by Lean's `verify --coins tree --stratified K`; no ZK server, and
#    no OneStageZk, which checks --zk sessions only. The COINS line reads where the sessions' coin trees came from off the
#    audit's own summary (``coin_trees``: window, the challenger session's release log; own, a challenger of each
#    session's own), and a --zk challenger audit whose summary says other than window fails the run (rc 20) with no line.
#    summary.json's ``epsilon`` is the window's sampling design's bound at several deltas (`Stratified.count_bound`).
#
#   research run --on MACHINE --project verity --source TREE --cwd clone --mem-gb 32 --timeout 3h \
#     --env GPU_LEASE_WHO=LANE --env A5_CORES=CPUS -- bash benchmarks/one_stage/a5_llama1b_h200.sh
#   or, on a machine of your own, from the tree's root: bash benchmarks/one_stage/a5_llama1b_h200.sh   (its record in ./out)
# Memory: it checks none; proof_layers.sh, which runs it three times, asks for 96 GB of host memory available.
# Uses: about 3 GB of disk under A5_ROOT
#   while it runs (the row dir and the window, removed at exit) and 0.3 GB for M0's Flock build (A5_FLOCK_WORK, kept), and 13-16 min on one H200 with M0's
#   build kept (the uv, Lean and M0 builds run behind the GPU stages; a machine's first run builds Flock and Lean cold).
# Needs, which vllm's one bootstrap makes (refused in one line each before any build, naming it):
#   (cd adapters/vllm && bash verity_vllm/ops/pod_bootstrap.sh --cases LLAMA32_1B_INT8)
#   vLLM's Python environment at VENV (default WORK/venv312), the int8 checkpoint under HF_HOME (default WORK/hf) and the
#   sm_90 taps under A5_TAPS (default WORK/cp: the guarded FA3 build and the norm tap; the unguarded FA3 build only where it
#   is there, as the row runs the guarded one). WORK defaults to the bootstrap's, /workspace. With gpu-lease the row leases
#   the GPU for its Match capture and its Commit's engine only (A5_CARD=N leases card N alone, on a machine whose other
#   cards are others'), and every lease names the CPUs A5_CORES gives (gpu-lease --cores, e.g. 40-59: the machine's map
#   gives each lane its cores, and a lease without them holds every other lane out), refused before any build when unset;
#   on a machine with no gpu-lease it takes card A5_CARD (default 0) directly, and its COINS line says so.
# The uv venv, Lean's verifiers and M0's flock-circuit build in the background while the GPU stages run (M0 in
# A5_ROOT/flock-work, or A5_FLOCK_WORK, kept for a rerun; a build killed mid-clone leaves one that 20-gpu-link.sh takes as
# built, so a rerun after one names another). The row dir, the window and the audit's private files are under
# A5_ROOT/RUN (A5_ROOT default WORK/a5-llama1b) while it runs, and removed at exit, success or failure, unless A5_KEEP=1
# keeps them for a rerun (one copy of the sealed bundle: sealed/ is hard links); the last line says which, where and how
# large. $OUT, published as the run's record, keeps the row's records that
# adapters/vllm/tools/workload/published.sh lists (each with the reason it holds nothing of a request) and verdicts.txt,
# the window's public files, the audit's out without its private files or the challenger's key ($OUT/a5) and summary.txt,
# whose A5 line is the audit's verdict, whose REGISTERED line the window's kind, commitment and run root length, and whose
# COINS line (also summary.json's ``coins``) the coins, the card and which check accepted. No other file of the row is
# published (with A5_KEEP=1 they stay on the machine): match/spec.json holds the prompts, commit/ the token record, the bundle and
# the slim keep (the committed store with its salts, the weights key, the opened weights, the requests and served tokens),
# and stages.txt, row.log and the stages' logs can quote a served token on a FAIL. Under KEPT, beside the row dir: each
# stage's output, the row dir's sizes and what `row run` writes for a research run (result.json, its evidence/ copies of
# the row's records and the Commit's window, which the stages get KEPT for, never the published run dir).
set -uo pipefail
OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}; mkdir -p "$OUT"
ROW=${ROW:-llama32-1b__int8__h200__tp1__b4__i256__o32__mixed__greedy__bi-eager}
ROLE=LLAMA32_1B_INT8 REPO=verity-int8/Llama-3.2-1B-INT8 REV=15b0976d400d796ee589e7737d03a919526bf800
TREE=$PWD
. "$TREE/adapters/vllm/tools/workload/published.sh" || exit 2
WORK=${WORK:-/workspace}
H=${A5_TAPS:-$WORK/cp}
FA3=${FA3_SO:-$H/fa2/build/fa3_matReq/verity_fa3_matReq.so} FA3G=${FA3G_SO:-$H/fa2/build/fa3_matReqG/verity_fa3_matReqG.so}
NT=${NORM_TAP_SO:-$H/norm_tap/build/verity_norm_tap.so}
PART=${A5_PARTITION:-$TREE/benchmarks/one_stage/llama32_1b_rope_d64_l0.json}
ROOT=${A5_ROOT:-$WORK/a5-llama1b}
BASE=$ROOT/${RESEARCH_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
D=$BASE/$ROW WD=$BASE/serving-rows FW=${A5_FLOCK_WORK:-$ROOT/flock-work} KEPT=$BASE/unpublished
VENV=${VENV:-$WORK/venv312} HF_HOME=${HF_HOME:-$WORK/hf} A5_COINS=${A5_COINS:-dev} A5_PLAIN=${A5_PLAIN:-}
CLEAN_PATH=$HOME/.local/bin:$HOME/.cargo/bin:$HOME/.elan/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
BOOT="(cd adapters/vllm && WORK=$WORK bash verity_vllm/ops/pod_bootstrap.sh --cases $ROLE)"
refuse() { echo "a5_llama1b_h200.sh: $1" >&2; exit 2; }
case $A5_COINS in dev|challenger) ;; *) refuse "A5_COINS=$A5_COINS: dev or challenger";; esac
case ${A5_KEEP:-} in ''|1) ;; *) refuse "A5_KEEP=$A5_KEEP: 1 keeps the row dir and the window at exit, or unset";; esac
case $A5_COINS/$A5_PLAIN in */|challenger/1) ;; *) refuse "A5_PLAIN=$A5_PLAIN: 1 with A5_COINS=challenger, or unset";; esac
[ -x "$VENV/bin/python" ] || refuse "no vLLM Python environment at $VENV/bin/python (VENV): make it with $BOOT"
CKPT=$(PYTHONPATH=$TREE/adapters/vllm "$VENV/bin/python" -c 'import sys; from verity_vllm.config import checkpoint_dir
print(checkpoint_dir(*sys.argv[1:]) or "")' "$REPO" "$REV" "$HF_HOME" 2>/dev/null)
[ -n "$CKPT" ] && [ -d "$CKPT" ] || refuse "no $REPO@${REV:0:10} checkpoint under $HF_HOME (HF_HOME): make it with $BOOT"
[ -e "$FA3G" ] || refuse "no guarded FA3 tap at $FA3G (A5_TAPS): make it with $BOOT"
[ -e "$NT" ] || refuse "no norm tap at $NT (A5_TAPS): make it with $BOOT"
[ -e "$PART" ] || refuse "no partition file at $PART (A5_PARTITION): a copy of benchmarks/one_stage/llama32_1b_rope_d64_l0.json with its sampling design (its law key)"
[ -z "$A5_PLAIN" ] || "$VENV/bin/python" -c 'import json, sys; sys.exit(not json.load(open(sys.argv[1]))["law"].startswith("stratified-rate:"))' "$PART" \
  || refuse "A5_PLAIN=1 takes a partition whose sampling design (its law key) is stratified-rate:NUM/DEN (a plain audit's sample is stratified, and the Commit registers the partition's sampling design): $PART"
if [ -n "${GPU_LEASE-}" ] || command -v gpu-lease >/dev/null; then
  [[ ${A5_CORES:-} =~ ^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$ ]] || refuse "A5_CORES=${A5_CORES:-}: every gpu-lease names its CPUs (--cores), and a lease without them holds every other lane out: A5_CORES=CPUS, the cores this machine's map gives you (e.g. 40-59)"
fi
command -v nvidia-smi >/dev/null || refuse "no nvidia-smi: the workload serves on one H200 (sm_90), which this machine doesn't show"
UNGUARDED=(); [ ! -e "$FA3" ] || UNGUARDED=(--hidden-so-fa3 "$FA3")
mkdir -p "$D" "$ROOT/cache" "$BASE/sealed" "$FW" "$KEPT"
finish() {  # at exit, success or failure: BASE (the row dir, the window, the audit's private files) removed, or with A5_KEEP=1 kept
  local rc=$? size
  size=$(du -sh "$BASE" 2>/dev/null | cut -f1)
  if [ "${A5_KEEP:-}" = 1 ]; then
    echo "kept: the row dir $D and the window $WD, under $BASE ($size)"
  else
    rm -rf "$BASE" && echo "removed: $BASE ($size: the row dir, the window and the audit's private files; A5_KEEP=1 keeps them)"
  fi | tee -a "$OUT/summary.txt"
  exit "$rc"
}
trap finish EXIT
if [ -n "${GPU_LEASE-}" ] || command -v gpu-lease >/dev/null; then
  CARD="card ${A5_CARD:-any} and CPUs $A5_CORES leased by gpu-lease for Match and Commit"
  real=$(command -v "${GPU_LEASE:-gpu-lease}")
  mkdir -p "$BASE/bin"
  printf '#!/usr/bin/env bash\n[ "$1" = status ] && exec %q "$@"\nn=$1; shift\nexec %q "$n" %s--cores %q "$@"\n' \
    "$real" "$real" "${A5_CARD:+--on $(printf %q "$A5_CARD") }" "$A5_CORES" > "$BASE/bin/gpu-lease" && chmod +x "$BASE/bin/gpu-lease"
  GPU_LEASE=$BASE/bin/gpu-lease
else  # an unleased row (pipeline.gpu_lease: no GPU_LEASE, nothing changes) on the one card it is shown
  CARD="card ${A5_CARD:-0} taken directly, no gpu-lease on this machine"
  GPU_LEASE= CUDA_VISIBLE_DEVICES=${A5_CARD:-0}; export CUDA_VISIBLE_DEVICES
fi

# the CPU side's builds, in the background: the tree's venv, Lean's verity-verify-proof, flock-zk-server and spot-verifier, M0
( export PATH=$CLEAN_PATH; unset PYTHONPATH PY PY312 VIRTUAL_ENV; cd "$TREE" || exit 1
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
  command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
  uv sync --all-packages --extra torch-cpu || exit 5
  (cd verity/protocols/proofs/flock/verifier && lake build verity-verify-proof flock-zk-server) || exit 7
  (cd verity/protocols/proofs/spot/verifier/lean && lake build spot-verifier spot-difftest spot-one-stage-zk) || exit 7
  RESEARCH_RUN_DIR=$OUT/prebuild FLOCK_WORK=$FW MODE=build GPU=0 bash verity/protocols/proofs/flock/prover/pod/60-circuit.sh || exit 6
) > "$OUT/prebuild.log" 2>&1 &
PB=$!

export PY=$VENV/bin/python PY312=$VENV/bin/python HF_HOME HF_HUB_OFFLINE=1
if [ -z "${CUDA_HOME:-}" ]; then
  for c in /usr/local/cuda-12.9 /usr/local/cuda-12 /usr/local/cuda-13.0 /usr/local/cuda; do [ -x "$c/bin/nvcc" ] && { CUDA_HOME=$c; break; }; done
fi
if [ ! -f "$CUDA_HOME/include/cusparse.h" ]; then  # a toolkit without the math libraries' headers: the venv's wheels hold them
  for lib in cusparse cusolver cufft curand; do
    for d in "$VENV"/lib/python*/site-packages/nvidia/"$lib"/include; do [ -d "$d" ] && CPATH=${CPATH:+$CPATH:}$d; done
  done
  export CPATH
fi
export CUDA_HOME TORCH_EXTENSIONS_DIR=$ROOT/cache/torch-extensions TRITON_CACHE_DIR=$ROOT/cache/triton VLLM_CACHE_ROOT=$ROOT/cache/vllm
export NATIVE_COLLECT_BUILD=$ROOT/cache/jit/nc
export PATH=$VENV/bin:$CUDA_HOME/bin:$PATH
export PYTHONPATH=$TREE:$TREE/adapters/vllm:$TREE/adapters/torch:$TREE/tools/config:$TREE/verity/protocols/proofs/flock/prover/python:$TREE/tools/research/src
export GPU_LEASE GPU_LEASE_WHO=${GPU_LEASE_WHO:-vllm-sha512}
export VERITY_WEIGHTS_HASH=${VERITY_WEIGHTS_HASH:-device-checked}
unset CONFIG_RUN REPLAY_K REPLAY_DEFERRED SERVING_ROWS
rs() { "$PY" -m research "$@" 2>&1 || research "$@" 2>&1; }
{ id; hostname; git -C "$TREE" rev-parse HEAD 2>/dev/null; echo "row dir $D"; echo "CUDA_HOME $CUDA_HOME"; nvidia-smi -L
  sha256sum ${UNGUARDED[1]+"${UNGUARDED[1]}"} "$FA3G" "$NT"; echo "HF_HOME $HF_HOME VENV $VENV A5_COINS $A5_COINS${A5_PLAIN:+ A5_PLAIN 1}: $CARD"; } | tee "$OUT/machine.txt"
rs keep "$ROOT" --owner @vllm --expires 7d --priority normal \
  --why "A5 over a GPU-served rows-sha512 window (Llama-3.2-1B int8 H200): M0's Flock build, and a row dir kept by A5_KEEP=1" | tail -2

row() {  # NAME ARGS...: one `verity-vllm row run` of the row dir, its output in KEPT/NAME.out; prints its stages' verdict words
  local name=$1; shift
  echo "$(date -u +%H:%M:%SZ) $name: $*" | tee -a "$OUT/steps.txt"
  (cd "$TREE/adapters/vllm" && "$PY" -m verity_vllm.pipeline.cli row run "$ROW" "$ROLE" "$REPO" "$REV" --sweep-dir "$BASE" \
     --research-run-dir "$KEPT" ${UNGUARDED[@]+"${UNGUARDED[@]}"} --hidden-so-fa3-guarded "$FA3G" --norm-tap-so "$NT" "$@") > "$KEPT/$name.out" 2>&1
  local rc=$?
  echo "$(date -u +%H:%M:%SZ) $name rc=$rc" | tee -a "$OUT/steps.txt"
  verdicts "$D/stages.txt" "$OUT"
  return $rc
}
records() {  # the row's records into $OUT, which the run publishes: only the files adapters/vllm/tools/workload/published.sh lists
  publish "$D" "$OUT"
  verdicts "$D/stages.txt" "$OUT" > /dev/null
  du -sh "$D" "$D"/* "$WD" 2>/dev/null > "$KEPT/sizes.txt"
}
window_off() {  # the Commit's window from KEPT, the stages' run dir, to WD, its public files and error into $OUT
  local rd=$KEPT/serving-rows/$ROW
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
challenger_audit() {  # A5_COINS=challenger: records_audit.py audit of WD on one local challenger session, then OneStageZk v2
  local A=$BASE/audit TPY=$TREE/.venv/bin/python leanb=$TREE/verity/protocols/proofs/flock/verifier/.lake/build/bin
  local run law n rate rc
  run=$("$TPY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["run_root"])' "$B/binding_map.json") || return 21
  law=$("$TPY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["law"])' "$PART")
  n=$("$TPY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["population"])' "$WD/registration.json") || return 22
  # the audit's rate is the one the Commit registered the window under: subset:K as rate:K/N over the window's N proof blocks
  case ${A5_PLAIN:+plain}/$law in
    /subset:*) rate=rate:${law#subset:}/$n ;;
    /rate:*|plain/stratified-rate:*) rate=$law ;;
    *) echo "A5_COINS=challenger takes a partition whose sampling design (its law key) is subset:K or rate:NUM/DEN, and A5_PLAIN=1 one that is stratified-rate:NUM/DEN: $law" >&2
       return 22 ;;
  esac
  git -C "$TREE" rev-list --first-parent HEAD > "$BASE/main.txt" || return 23
  "$TPY" -m verity_numerical.bench.generate --subcircuit "${A5_SUBCIRCUIT:-rope-head/d64/neox-bf16}" --n 4 --seed 1 \
    --out "$BASE/sets" > "$OUT/a5/sets.log" 2>&1 || return 24
  FLOCK_CHALLENGER=1 RAYON_NUM_THREADS=$(nproc) RUST_MIN_STACK=$((64 << 20)) "$TPY" benchmarks/prove_path/records_audit.py audit --window "$WD" \
    --sampling-design "$rate" --circuit "$(sha512sum "$WD/circuit.json" | cut -c1-128)" --run "$run" --template-sets "$BASE/sets" \
    --challenger local --deployed-key ${A5_PLAIN:+--plain} --main "$BASE/main.txt" \
    --m0-pythonpath "$TREE:$TREE/benchmarks/numerical/python:$TREE/verity/protocols/proofs/flock/prover/python:$TREE/adapters/vllm:$TREE/adapters/torch" \
    --bin "$FW/flock-gpu-link/flock/target/release/flock-circuit" --lean "$leanb/verity-verify-proof" \
    --zk-server "$leanb/flock-zk-server" --out "$A" > "$OUT/a5/a5.log" 2> "$OUT/a5/audit.err"
  rc=$?
  (cd "$A" && find . -type f ! -name circuit.txt ! -name 'inst-*' ! -name 'm*-inst-*' ! -name '*private*' ! -path './*/archive/*' \
     ! -path '*/challenger-work/*' \( ! -path './challenger/*' -o -name windows.jsonl -o -name pins.jsonl \) -print0 \
    | xargs -0 -r -I{} cp --parents {} "$OUT/a5/")
  [ -z "$A5_PLAIN" ] || return $rc
  # the case the audit wrote (a5.write_case: it names the challenger, its window root and coins), as SPOT's verifier reads it;
  # v2_case.py's rebuild from the kept files carries no coins, which v2 refuses (NO_CHALLENGER)
  "$TPY" - "$A/window/case-v2.json" "$BASE/v2" "$TREE/verity/protocols/proofs/spot/verifier/lean/.lake/build/bin/spot-one-stage-zk" \
    > "$OUT/a5/v2.log" 2>&1 <<'PY'
import json, subprocess, sys
from pathlib import Path

sys.path.insert(0, "benchmarks/one_stage")
from composite import canonical, write_private

case, out, exe = json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2]), sys.argv[3]
files = {"params.json": canonical(case["params"]), "commitment.bin": bytes.fromhex(case["commitment"]),
         "evidence.json": bytes.fromhex(case["evidence"])}
write_private(out, files)
p = subprocess.run([exe, "check", *(str(out / f) for f in files), *case["blobs"]], capture_output=True, text=True, check=False)
line = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
if not line:
    print(f"V2 no answer (exit {p.returncode}): {p.stderr.strip()[-400:]}")
    sys.exit(p.returncode or 2)
write_private(out, {"answer.json": (line + "\n").encode()})
answer = json.loads(line)
print(f"V2 verity/sampled-proofs/one-stage/v2 {answer['verdict']}" + (f": {answer['reason']}" if answer.get("reason") else ""))
sys.exit(p.returncode)
PY
  echo "V2RC $?" >> "$OUT/a5/v2.log"
  return $rc
}
mkdir -p "$OUT/a5"
if [ "$A5_COINS" = challenger ]; then
  ( export PATH=$CLEAN_PATH; unset PYTHONPATH PY PY312 VIRTUAL_ENV; cd "$TREE" || exit 1; challenger_audit )
else
  ( export PATH=$CLEAN_PATH; unset PYTHONPATH PY PY312 VIRTUAL_ENV; cd "$TREE" || exit 1
    OUT=$OUT/a5 COMMIT_SCHEME=rows-sha512 WINDOW=served A5_ROW_DIR=$D A5_PARTITION=$PART A5_WINDOW=$WD A5_FLOCK_WORK=$FW \
      bash benchmarks/one_stage/a5.sh ) > "$OUT/a5.out" 2>&1
fi
ar=$?
echo "$(date -u +%H:%M:%SZ) a5 ($A5_COINS) rc=$ar" | tee -a "$OUT/steps.txt"

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
grep -a -E '^V2(CASE)? ' "$OUT/a5/v2.log" 2>/dev/null | cut -c1-2000 | tee -a "$OUT/summary.txt"
A5_COINS=$A5_COINS A5_PLAIN=$A5_PLAIN CARD=$CARD WD=$WD "$TREE/.venv/bin/python" - "$OUT/a5" <<'PY' | tee -a "$OUT/summary.txt"
import json, os, pathlib, sys
from verity.protocols.proofs.spot.verifier.one_stage.audit import core_sampling_design
d = pathlib.Path(sys.argv[1])
s = next((p for p in (d / "a5" / "summary.json", d / "summary.json") if p.is_file()), d / "summary.json")
def read(p):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}
summ = {**read(d / "window" / "summary.json"), **read(s)}
# records_audit.py's law is its rate text; the window's is the partition's {law, k}
law = next((x for x in (read(s).get("law"), read(d / "window" / "summary.json").get("law")) if isinstance(x, dict)), {})
rate, card = read(s).get("law") if isinstance(read(s).get("law"), str) else None, os.environ["CARD"]
lines = [x for x in (d / "a5.log").read_text(errors="replace").splitlines() if x.startswith("A5 ")] if (d / "a5.log").is_file() else []
audit = next((json.loads(x[3:]).get("audit") for x in reversed(lines) if '"audit"' in x), None) or {}
accepted = "accepted" if audit.get("accepted") is True else "NOT accepted"
TREES = {"window": "every session's coin tree from the release log of the window's one challenger session",
         "own": "each session's coin tree from a development challenger of its own"}
trees = summ.get("coin_trees")
if trees not in TREES or (os.environ["A5_COINS"] == "challenger" and not os.environ["A5_PLAIN"] and trees != "window"):
    sys.exit(f"a5_llama1b_h200.sh: no COINS line: the audit's summary says coin_trees={trees!r}, and a --zk audit on the "
             f"challenger's coins must have every session's tree from that challenger session's release log (window)")
trees = f"coin trees {trees}: {TREES[trees]}"
sample = (f"{len(summ['draw']) if isinstance(summ.get('draw'), list) else '?'} of {summ.get('population', '?')} proof blocks "
          f"sampled (sampling design {law.get('law', '?')} k={law.get('k', '?')}, delta {summ.get('delta', '?')}"
          + (f", at {rate}" if rate else "") + ")")
if os.environ["A5_PLAIN"]:
    line = (f"COINS draw: one local challenger session (the auditor device's service run on this machine, a development "
            f"stand-in for the auditor's, its key made for the run at the deployed LMS parameters), {sample}; sessions: Flock "
            f"plain, no ZK server ({trees}); the audit {accepted}" + (", by Flock's plain per-session Lean verification, "
            "with --coins tree and --stratified K, and the Lean verifier's sampler" if audit.get("accepted") is True else "") + "; SPOT's one-stage check does not "
            f"apply (OneStageZk checks --zk sessions); {card}")
elif os.environ["A5_COINS"] == "challenger":
    v2 = [x for x in (d / "v2.log").read_text(errors="replace").splitlines()] if (d / "v2.log").is_file() else []
    rc = next((x.split()[1] for x in v2 if x.startswith("V2RC ")), None)
    verdict = {"0": "accepted", "1": "rejected", "2": "undecided or no case"}.get(rc, "did not run")
    line = (f"COINS draw: one local challenger session (the auditor device's service run on this machine, a development "
            f"stand-in for the auditor's, its key made for the run at the deployed LMS parameters), {sample}; sessions: Flock "
            f"--zk behind the Lean ZK server ({trees}); the audit {accepted}; SPOT's one-stage check (OneStageZk v2, "
            f"spot-one-stage-zk) {verdict}; {card}")
else:
    line = (f"COINS draw: development coins (in-process secrets.token_bytes), no auditor's challenger, by the Lean verifier's sampler, {sample}; "
            f"sessions: Flock {'plain' if summ.get('plain') else '--zk'} ({trees}), arm {summ.get('arm', '?')} "
            f"({'the ZK server' if summ.get('firewall') else 'no ZK server'}); {accepted} by Flock's per-session Lean verification "
            f"with --zk and the Lean verifier's sampler, not by SPOT's one-stage check (OneStageZk did not run; it refuses a case naming no "
            f"challenger); {card}")
print(line)
if s.is_file():  # benchmarks' printer reads summary.json's coins, an object: the line rides as its text
    doc = read(s)
    doc["coins"] = {**(doc.get("coins") if isinstance(doc.get("coins"), dict) else {}), "text": line,
                    **({"source": "challenger", "challenger": "local"} if os.environ["A5_COINS"] == "challenger" else {})}
    # the window's sampling design as registered, a stratified one with its strata (a stratum a template)
    reg = next((read(p) for p in (d / "window" / "registration.json", d / "a5" / "window" / "registration.json",
                                  pathlib.Path(os.environ["WD"]) / "registration.json") if p.is_file()), {})
    design, n = reg.get("law"), reg.get("population")
    profile = core_sampling_design(design, n) if isinstance(design, dict) and isinstance(n, int) else None
    if profile is not None:
        doc["epsilon"] = {"design": design, "population": n,
                          "rule": "except with probability delta, an accepted audit has at most wrong_units incorrect proof "
                          "blocks of the population's proof blocks: the window's sampling design's Stratified.count_bound(delta), a stratum a template; "
                          "epsilon = wrong_units/population",
                          "table": [{"delta": x, "wrong_units": profile.count_bound(x), "epsilon": round(profile.count_bound(x) / n, 6)}
                                    for x in (0.5, 0.1, 0.01, 1e-3, 1e-6, 1e-9)]}
    s.write_text(json.dumps(doc, indent=1, sort_keys=True))
PY
cs=${PIPESTATUS[0]}; [ "$cs" != 0 ] || cs=
tail -8 "$OUT/verdicts.txt" 2>/dev/null | tee -a "$OUT/summary.txt"
vr=; [ "$A5_COINS" != challenger ] || [ -n "$A5_PLAIN" ] || { vr=$(awk '/^V2RC /{print $2}' "$OUT/a5/v2.log" 2>/dev/null); vr=${vr:-2}; }
echo "replay rc=$rr a5 rc=$ar${vr:+ v2 rc=$vr}${cs:+ coins rc=$cs}" | tee -a "$OUT/summary.txt"
exit $(( rr != 0 ? 17 : ar != 0 ? 18 : vr != 0 ? 19 : cs != 0 ? 20 : 0 ))
