#!/usr/bin/env bash
# A split Match's Commit on an 8x H200 node that holds nothing of the row, every input from the store.  MATCH_FULL is
# match_full.sh's run, which labels itself glm_v2.match_run with the vllm.match run commit.sh cites once its Match passes; given
# MATCH_RUN, that Match has passed, and without it the pod is readied while the Match runs, then waits for MATCH_FULL's match-cpu
# set and that label (MATCH_WAIT_MIN, a failed Match stops it).  SETUP=1 (a fresh RunPod pod, the default) installs the CUDA
# 12.9.1 toolkit at /usr/local/cuda-12.9, which /usr/local/cuda then links (the image's 12.4 nvcc can't build DeepGEMM's kernels),
# gpu-lease, and the venv and taps (pod_bootstrap.sh --gpu).  Then the device facts the Build pins (8 GPUs, cc 9.0, 132 SMs); then,
# side by side, the checkpoint (stage_ckpt.sh) and the row dir: the Build of record (BUILD_PARTS) first, then the legs' small files
# (LEGS_PARTS) and their values (VALUES_SETS), GETS sets at a time, each through store_tar.sh; then the passing Match (MATCH_FULL's
# match-cpu); the legs against match_gpu.sh's list (LEGS, every file but its .tree); then commit.sh over MATCH_RUN.  Each step's
# records are under $OUT.
#   research run --on NODE --mem-gb 600 --timeout 14h --custody-ttl 24h --source . --cwd source --project verity --send LEGS.sha256.gz -- env ROW=... BUILD_RUN=... MATCH_FULL=RUN [MATCH_RUN=RUN] BUILD_PARTS=RUN:build LEGS_PARTS=RUN:legs-small VALUES_SETS="RUN:values-r0 ..." LEGS=LEGS.sha256.gz NODE=... EVICT_FREE_GB=300 bash integrations/vllm/manifests/glm_v2_tree/jobs/pod_commit.sh
# env: ROW, BUILD_RUN, MATCH_FULL, [MATCH_RUN], [MATCH_WAIT_MIN=600], BUILD_PARTS, LEGS_PARTS, VALUES_SETS, LEGS (a path, else
# $RESEARCH_RUN_DIR/inputs/LEGS; gzipped or not), [NODE], [SETUP=1], [GETS=4], store_tar.sh's [EVICT_FREE_GB], [FETCH_*], commit.sh's.
set -uo pipefail
: "${ROW:?ROW}" "${BUILD_RUN:?BUILD_RUN}" "${MATCH_FULL:?MATCH_FULL}" "${BUILD_PARTS:?BUILD_PARTS}" \
  "${LEGS_PARTS:?LEGS_PARTS}" "${VALUES_SETS:?VALUES_SETS}" "${LEGS:?LEGS}"
MATCH_RUN=${MATCH_RUN:-}
TREE=$PWD J=$PWD/integrations/vllm/manifests/glm_v2_tree/jobs
source "$J/model.sh"
D=$SW/$ROW O=$OUT PY=/workspace/cp/venv312/bin/python
step() { echo "$(date -u +%FT%TZ) [pod] $*"; }
[ -f "$LEGS" ] || LEGS=$RESEARCH_RUN_DIR/inputs/$LEGS
[ -s "$LEGS" ] || { step "no legs list $LEGS (match_gpu.sh's match.sha256)"; exit 3; }
[ -z "$(find "$D" -type f 2>/dev/null | head -1)" ] || { step "$D holds files already: this run fills an empty row dir"; exit 3; }
df -h /workspace | tail -1 | tee "$O/disk_at_start.txt"

if [ "${SETUP:-1}" = 1 ]; then
  mkdir -p "$O/setup"
  if [ ! -x /usr/local/cuda-12.9/bin/nvcc ]; then
    step "the CUDA 12.9.1 toolkit"
    S=$(mktemp -d /workspace/cuda-install.XXXXXX) || exit 3
    curl -fsSL -o "$S/cuda.run" https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_575.57.08_linux.run ||
      { step "the toolkit's download failed"; exit 3; }
    sha256sum "$S/cuda.run" | tee "$O/setup/cuda_run.sha256"
    sh "$S/cuda.run" --silent --toolkit --toolkitpath=/usr/local/cuda-12.9 --no-opengl-libs --no-man-page --no-drm --override --tmpdir="$S" \
      >"$O/setup/cuda_install.log" 2>&1 || { tail -20 "$O/setup/cuda_install.log"; step "the toolkit's install failed"; exit 3; }
    rm -rf "$S"
  fi
  /usr/local/cuda-12.9/bin/nvcc --version | tail -2 | tee "$O/setup/nvcc.txt"
  grep -q "release 12.9" "$O/setup/nvcc.txt" || { step "nvcc is not 12.9"; exit 3; }
  if [ -L /usr/local/cuda ]; then ln -sfn /usr/local/cuda-12.9 /usr/local/cuda
  elif [ -e /usr/local/cuda ]; then mv /usr/local/cuda "/usr/local/cuda.before-$RESEARCH_RUN_ID" && ln -s /usr/local/cuda-12.9 /usr/local/cuda
  else ln -s /usr/local/cuda-12.9 /usr/local/cuda; fi
  readlink -f /usr/local/cuda | tee "$O/setup/cuda_link.txt"
  mkdir -p /run/gpu-lease && install -m 0755 tools/research/src/research/pods/sh/gpu_lease.sh /usr/local/bin/gpu-lease || { step "gpu-lease's install failed"; exit 3; }
  step "the venv and taps (pod_bootstrap.sh --gpu)"
  mkdir -p /workspace/cp && (cd integrations/vllm && WORK=/workspace/cp VENV=/workspace/cp/venv312 HF_HOME=/workspace/cp/hf CUDA_HOME=/usr/local/cuda-12.9 \
    PATH=/usr/local/cuda-12.9/bin:$PATH bash verity_vllm/ops/pod_bootstrap.sh --gpu --cases "" --out "$O/bootstrap") >"$O/bootstrap.log" 2>&1
  grep -q '^BOOTSTRAP-OK' "$O/bootstrap.log" || { tail -30 "$O/bootstrap.log"; step "pod_bootstrap.sh --gpu failed"; exit 5; }
fi
[ -x "$PY" ] && [ -x /usr/local/bin/gpu-lease ] || { step "no $PY or /usr/local/bin/gpu-lease: SETUP=1 makes them"; exit 3; }

step "the device facts"
{ nvidia-smi --query-gpu=index,name,pci.device_id,driver_version,memory.total,power.limit,clocks.max.sm --format=csv
  "$PY" - <<'EOF'
import torch
n = torch.cuda.device_count()
ps = [torch.cuda.get_device_properties(i) for i in range(n)]
for i, p in enumerate(ps):
    print(i, p.name, f"cc {p.major}.{p.minor}", p.multi_processor_count, "SMs", p.total_memory >> 20, "MiB")
print("torch", torch.__version__, "cuda", torch.version.cuda)
assert n == 8 and all((p.major, p.minor, p.multi_processor_count) == (9, 0, 132) for p in ps), "not 8 GPUs at cc 9.0 with 132 SMs"
EOF
} >"$O/device_facts.txt" 2>&1; rc=$?
tail -11 "$O/device_facts.txt"
[ $rc = 0 ] || { step "the device facts differ from the Build's pins (device_facts.txt): no Commit"; exit 3; }

step "the checkpoint (stage_ckpt.sh), beside the row dir's sets"
mkdir -p "$O/ckpt"
(OUT=$O/ckpt JOBS=32 HF_MAX_WORKERS=32 PY=$PY bash "$J/stage_ckpt.sh" >"$O/ckpt/stage_ckpt.log" 2>&1; echo $? >"$O/ckpt/rc") &
ck=$!
get() {  # get RUN:NAME: one set into the row dir, its records under $O/get-NAME
  local o=$O/get-${1#*:} rc
  mkdir -p "$o" && OUT=$o PARTS=$1 bash "$J/store_tar.sh" get >"$o/get.log" 2>&1; rc=$?
  echo "$(date -u +%FT%TZ) [pod] $1: rc $rc ($(tail -1 "$o/get.log"))"; return $rc
}
export -f get; export O J SW
get "$BUILD_PARTS" || { wait $ck; exit 5; }
[ -s "$D/manifest.json" ] || { step "no Build of record in $D"; wait $ck; exit 3; }
printf '%s\n' $LEGS_PARTS $VALUES_SETS | xargs -P "${GETS:-4}" -I{} bash -c 'get "$1"' _ {}; grc=$?
wait $ck; crc=$(cat "$O/ckpt/rc" 2>/dev/null || echo 9)
tail -3 "$O/ckpt/stage_ckpt.log"
[ "$grc" = 0 ] || { step "a set's get failed (xargs rc $grc): no Commit"; exit 5; }
[ "$crc" = 0 ] && [ -e "$LOCAL_CK/.verified" ] || { step "the checkpoint's staging failed (rc $crc): no Commit"; exit 5; }
if [ -z "$MATCH_RUN" ]; then
  step "readied; waiting for $MATCH_FULL's Match (its match-cpu set, then glm_v2.match_run)"
  mkdir -p "$O/match-wait"
  OUT=$O/match-wait PARTS=$MATCH_FULL:match-cpu WAIT_MIN=${MATCH_WAIT_MIN:-600} bash "$J/store_tar.sh" wait >"$O/match-wait/wait.log" 2>&1; rc=$?
  tail -1 "$O/match-wait/wait.log"
  [ "$rc" = 0 ] || { step "$MATCH_FULL's Match did not pass (store_tar.sh wait rc $rc): no Commit"; exit 8; }
  MATCH_RUN=$(OUT=$O/match-wait RUN=$MATCH_FULL KEY=glm_v2.match_run WAIT_MIN=30 bash "$J/store_tar.sh" value 2>>"$O/match-wait/wait.log" | tail -1)
  [[ $MATCH_RUN =~ ^r[0-9]{8}-[0-9]{6}-[0-9a-f]+$ ]] || { step "no glm_v2.match_run label on $MATCH_FULL ('$MATCH_RUN'): no Commit"; exit 8; }
  step "the Match passed: $MATCH_RUN"
fi
get "$MATCH_FULL:match-cpu" || { step "the passing Match's get failed: no Commit"; exit 5; }
step "the legs against $LEGS, every file but its .tree"
case $LEGS in *.gz) gzip -dc "$LEGS" ;; *) cat "$LEGS" ;; esac | awk 'substr($0, 67) !~ /\.tree$/' >"$O/legs.sha256"
mkdir -p "$O/legs-check" && split -n l/32 -d -a 2 "$O/legs.sha256" "$O/legs-check/part."
ls "$O"/legs-check/part.* | xargs -P 32 -I{} sh -c 'cd "$1" && sha256sum --quiet -c "$2"' _ "$SW" {} >"$O/legs-check.log" 2>&1 ||
  { head -20 "$O/legs-check.log"; step "the legs in $D/match are not the ones match_gpu.sh hashed: no Commit"; exit 3; }
step "$(wc -l <"$O/legs.sha256") legs files equal match_gpu.sh's list"
[ -s "$D/match/tp2_match.json" ] || { step "no match/tp2_match.json in $D after match-cpu"; exit 3; }
df -h /workspace | tail -1 | tee "$O/disk_before_commit.txt"

step "the Commit over $MATCH_RUN"
mkdir -p "$O/commit" && OUT=$O/commit MATCH_RUN=$MATCH_RUN NODE=${NODE:-$(hostname)} bash "$J/commit.sh"; rc=$?
step "the Commit ended (rc $rc); bundles: $O/commit/bundles.json"
exit "$rc"
