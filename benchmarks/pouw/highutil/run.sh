#!/usr/bin/env bash
# The highutil period on one leased GPU (an RTX PRO 6000, sm_120, by default; H200: below), cwd = the source tree's root under
# `research run`:
# the build and the config (the checkpoint's weights, params) outside the lease, then under it capture.py's activations
# (a W8A8 forward of the checkpoint), period.py's diag (one step's openings on the GPU against the prover's recomputed rows)
# and its interleaved protocol and plain periods and the negative one, and the machine model's θ beside them.
#
#   research run --on MACHINE --project verity --source . --cwd source \
#     --env GPU_LEASE_WHO=LANE --send main.txt [--send challenger-key.json] -- \
#     env CHECK_SLOTS=/workspace/research/locks/gpu/slots python3 tools/verity/check/slot.py --by LANE \
#     -- bash benchmarks/pouw/highutil/run.sh
#
# COINS: local (default; a development session on this machine, not of record) or gate:USER@HOST (the challenger, with
# PEER, CHALLENGER_KEY, the deployment's key record, and GATE_KEY); MAIN (default inputs/main.txt): the verifier's copy of
# main's first-parent line. CARD (default inputs/card.toml, sent with --send; skipped when absent): θ's card, verity/hardware's
# card file or its SKU, for summary.json's certified share at the card's clock ceiling and, with NSYS=1, terms.json (terms.py:
# where the rest of θ x wall goes); nvidia-smi's clock event reasons go to clocks-{before,after}.txt. PLAIN (default
# lt,int_mm): period.py's --plain, the plain arm's baselines, gamma's first: cuBLASLt (v31's lt_setup) and torch._int_mm. CKPT: the pinned checkpoint's snapshot directory (default the node's HF cache), or `seeded`
# (seeded A and B, no capture). BLOCKS, STEPS, Q, PERIODS (0: none), NEGATIVE: period.py run's; ROW (default 8192), LP, LP_MAX
# and L (default 512, 512 and 16384: S = 1 at every layer, down as four K-slices at L' 512; LP_MAX=1024 runs it as two at
# L' 1,024, 1.7% more GPU time a step on the RTX): prep's; NSYS=1: after the
# periods, Nsight Systems over one plain period of each baseline and one protocol period (their
# kernels' share of the GPU's time); NCU=1: Nsight Compute over two steps of each arm (pipes.py: each kernel's tensor-pipe
# active share and INT8 MACs, kernels serialized; pipes-{plain,proto}.json); LEASE (default gpu-lease 1 --wait --max-min 25 --); PY (default uv run -q --no-project
# --with numpy python) for the prep, TORCH_PY for the rest. v31's V31_* knobs come from the environment when the Provers are
# made; V31_LTW defaults to 1 (a call at S = 1 takes cuBLASLt's int32 X W'^T in place of v31's GEMM, the same leaf bytes).
# TEXT: wikitext-2-raw-v1's test split for capture.py (default the datasets cache's arrow file, else its parquet at the same
# revision, fetched from the hub into the run's scratch). On an H200: GENCODE=arch=compute_90a,code=sm_90a
# NVFLAGS=-DV31_WG=1 THETA_DEVICE=sm_90 (with V31_SCHED=2 V31_LMINB=4 V31_SAHEAD=1), CARD its card file, and the checkpoint built into the node's HF cache first
# (benchmarks/hash/dither/pin.sh).
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
D=benchmarks/pouw/highutil
PY=${PY:-uv run -q --no-project --with numpy python}
export PYTHONPATH=.
GENCODE=${GENCODE:-arch=compute_120a,code=sm_120a} NVFLAGS=${NVFLAGS:-} THETA_DEVICE=${THETA_DEVICE:-sm_120}
TORCH_PY=${TORCH_PY:-uv run -q --no-project --with torch --with numpy --with safetensors --with tokenizers --with pyarrow python}
CKPT=${CKPT:-/workspace/hf/hub/models--verity-int8--Llama-3.1-8B-INT8/snapshots/723d342ac443b040931242c768df342b0f6f15a7}
ROW=${ROW:-8192} LP=${LP:-512} LP_MAX=${LP_MAX:-512} L=${L:-16384} BLOCKS=${BLOCKS:-4} STEPS=${STEPS:-300} Q=${Q:-2048} PERIODS=${PERIODS:-2} NEGATIVE=${NEGATIVE-skip:2@2}
COINS=${COINS:-local} MAIN=${MAIN:-${RESEARCH_RUN_DIR:-.}/inputs/main.txt}
CARD=${CARD:-${RESEARCH_RUN_DIR:-.}/inputs/card.toml}
CARDA=()
if [ -f "$CARD" ] || [ "${CARD%.toml}" = "$CARD" ]; then CARDA=(--card "$CARD"); fi
W=$(mktemp -d /dev/shm/highutil.XXXXXX)
export W
trap 'rm -rf "$W"' EXIT
git rev-parse HEAD > "$O/commit.txt" 2>&1
[ -n "${HYPOTHESIS:-}" ] && printf '%s\n' "$HYPOTHESIS" > "$O/hypothesis.txt"
: > "$O/rc.txt"
export V31_LTW=${V31_LTW-1}
LIB=$W/libv31.so
# shellcheck disable=SC2086
nvcc -O3 -std=c++17 -gencode "$GENCODE" -Xptxas -v -DSHA_ADD=0 -DSHA_ROT=0 -DSHA_ROLL=1 -DV31_LIB -DCOMMIT_GPU_DOMS=8 $NVFLAGS \
  -Xcompiler -fPIC -shared -o "$LIB" $D/hu_lib.cu -lcuda -lcublasLt > "$O/build.log" 2>&1
echo "build rc $?" >> "$O/rc.txt"
[ -s "$LIB" ] || { cat "$O/rc.txt"; tail -30 "$O/build.log"; exit 1; }
A=benchmarks/pouw/accel_v31
sha256sum $A/v31.cu $A/wg_proto.cuh $A/v31.py $A/v31lib.py $D/hu_lib.cu $D/period.py $D/capture.py $D/terms.py $D/pipes.py > "$O/sources.sha256"
if [ "$CKPT" = seeded ]; then SRC=(--seeded); else SRC=(--checkpoint "$CKPT"); fi
uv run -q --no-project --with numpy --with safetensors python $D/period.py prep "$W/cfg" "${SRC[@]}" --row "$ROW" --lp "$LP" --lp-max "$LP_MAX" --L "$L" \
  > "$O/prep.log" 2>&1
r=$?
echo "prep rc $r" >> "$O/rc.txt"
[ $r = 0 ] || { cat "$O/rc.txt"; tail -30 "$O/prep.log"; exit 1; }
cp "$W/cfg/prep.json" "$O/prep.json"
$TORCH_PY -c "import torch, safetensors, tokenizers, pyarrow; print('torch', torch.__version__, torch.version.cuda)" \
  >> "$O/prep.log" 2>&1
(cd benchmarks/pouw/machine_model && PYTHONPATH=. uv run -q --no-project --with scipy --with numpy python -m verity_machine_model.cli \
  --device "$THETA_DEVICE" theta) > "$O/theta.txt" 2>&1
echo "theta rc $?" >> "$O/rc.txt"
WIKI_REV=b08601e04326c79dfdd32d625aee71d232d685c3
TEXT=${TEXT:-/workspace/hf/datasets/Salesforce___wikitext/wikitext-2-raw-v1/0.0.0/$WIKI_REV/wikitext-test.arrow}
if [ "$CKPT" != seeded ] && [ ! -f "$TEXT" ]; then
  TEXT=$W/wikitext-test.parquet
  curl -sfL -o "$TEXT" "https://huggingface.co/datasets/Salesforce/wikitext/resolve/$WIKI_REV/wikitext-2-raw-v1/test-00000-of-00001.parquet"
  echo "text rc $?" >> "$O/rc.txt"
fi
GATE=()
if [ "$COINS" != local ]; then
  GATE=(--peer "${PEER:?}" --challenger-key "${CHALLENGER_KEY:?}" --gate-key "${GATE_KEY:?}")
fi
LEASE=${LEASE-gpu-lease 1 --wait --max-min 25 --}
export D PY TORCH_PY CKPT TEXT LIB BLOCKS STEPS Q PERIODS NEGATIVE COINS MAIN PLAIN=${PLAIN:-lt,int_mm} NSYS=${NSYS:-0} NCU=${NCU:-0}
export GATE_ARGS="${GATE[*]}" CARD_ARGS="${CARDA[*]}"
[ -f "$CARD" ] && cp "$CARD" "$O/card.toml"
$LEASE bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,clocks.mem,power.limit --format=csv \
    ${GPU_LEASE_UUID:+-i "$GPU_LEASE_UUID"} > "$O/gpu.txt" 2>&1
  nvidia-smi -q -d PERFORMANCE,CLOCK ${GPU_LEASE_UUID:+-i "$GPU_LEASE_UUID"} > "$O/clocks-before.txt" 2>&1
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.gr,clocks.mem,power.draw,utilization.gpu,temperature.gpu --format=csv -lms 500 \
    ${GPU_LEASE_UUID:+-i "$GPU_LEASE_UUID"} > "$O/smi.csv" 2>&1 &
  smi=$!
  if [ "$CKPT" != seeded ]; then
    $TORCH_PY $D/capture.py "$W/cfg" --checkpoint "$CKPT" --text "$TEXT" \
      > "$O/capture.log" 2>&1
    r=$?
    echo "capture rc $r" >> "$O/rc.txt"
    [ $r = 0 ] || { kill $smi 2>/dev/null; exit $r; }
    cp "$W/cfg/capture.json" "$O/capture.json"
  fi
  $TORCH_PY $D/period.py diag "$W/cfg" "$LIB" "$O" > "$O/diag.log" 2>&1
  echo "diag rc $?" >> "$O/rc.txt"
  r=0
  [ "$PERIODS" = 0 ] || $TORCH_PY $D/period.py run "$W/cfg" "$LIB" "$O" --coins "$COINS" --main "$MAIN" $GATE_ARGS $CARD_ARGS --blocks "$BLOCKS" --steps "$STEPS" \
    --q "$Q" --periods "$PERIODS" --negative "$NEGATIVE" --plain "$PLAIN" > "$O/period.log" 2>&1
  r=$?
  echo "period rc $r" >> "$O/rc.txt"
  nvidia-smi -q -d PERFORMANCE,CLOCK ${GPU_LEASE_UUID:+-i "$GPU_LEASE_UUID"} > "$O/clocks-after.txt" 2>&1
  command -v nsys > /dev/null || { [ "$NSYS" = 1 ] && echo "nsys absent rc 0" >> "$O/rc.txt"; NSYS=0; }
  command -v ncu > /dev/null || { [ "$NCU" = 1 ] && echo "ncu absent rc 0" >> "$O/rc.txt"; NCU=0; }
  if [ "$NSYS" = 1 ]; then
    for arm in plain proto; do
      HU_CLOCK=0 nsys profile -o "$O/nsys-$arm" --force-overwrite true --trace cuda $TORCH_PY $D/period.py run "$W/cfg" "$LIB" "$W/nsys-$arm" \
        --coins "$COINS" --main "$MAIN" $GATE_ARGS --blocks 1 --steps 64 --q 64 --periods 1 --negative "" --arms $arm >> "$O/nsys.log" 2>&1
      echo "nsys $arm rc $?" >> "$O/rc.txt"
      nsys stats --report cuda_gpu_kern_sum --format csv -o "$O/nsys-$arm" "$O/nsys-$arm.nsys-rep" >> "$O/nsys.log" 2>&1
      nsys export --type sqlite --force-overwrite true -o "$O/nsys-$arm.sqlite" "$O/nsys-$arm.nsys-rep" >> "$O/nsys.log" 2>&1
      $PY $D/timeline.py "$O/nsys-$arm.sqlite" > "$O/timeline-$arm.json" 2>> "$O/nsys.log"
    done
    if [ -s "$O/summary.json" ] && [ -n "$CARD_ARGS" ]; then
      $PY $D/terms.py "$O/summary.json" "$O/timeline-proto.json" > "$O/terms.json" 2>> "$O/nsys.log"
      echo "terms rc $?" >> "$O/rc.txt"
    fi
  fi
  if [ "$NCU" = 1 ]; then
    for arm in plain proto; do
      ncu --clock-control none --csv --page raw --print-units base --log-file "$O/ncu-$arm.csv" -c 600 \
        --metrics gpu__time_duration.sum,sm__ops_path_tensor_op_imma_src_int8.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__cycles_elapsed.avg.per_second \
        --kernel-name "regex:^(Kernel2|sm[0-9]+_xmma|leaf_persist|salt_persist|noise_kernel|fill_a_kernel|fill_b_kernel|top_small|bits_kernel|perm_kernel|level_kernel)" \
        $TORCH_PY $D/period.py run "$W/cfg" "$LIB" "$W/ncu-$arm" --coins "$COINS" --main "$MAIN" $GATE_ARGS --blocks 1 --steps 2 --q 16 --periods 1 \
        --negative "" --arms $arm >> "$O/ncu.log" 2>&1
      echo "ncu $arm rc $?" >> "$O/rc.txt"
      $PY $D/pipes.py "$O/ncu-$arm.csv" $CARD_ARGS > "$O/pipes-$arm.json" 2>> "$O/ncu.log"
      echo "pipes $arm rc $?" >> "$O/rc.txt"
    done
  fi
  kill $smi 2>/dev/null
  exit $r
'
echo "lease rc $?" >> "$O/rc.txt"
cat "$O/rc.txt"
tail -40 "$O/period.log"
grep -q "^period rc 0" "$O/rc.txt"
