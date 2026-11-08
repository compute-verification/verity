#!/usr/bin/env bash
# A served GLM-5.3 replay block of node A's (vy-spot-glm53's) replayed whole on vy-glm53-h8 from an empty engine with every hook
# on (spot_canonical.sh), cross-machine: node A's CUDA toolkit and package versions (h8_setup.sh), and the all-reduces summed in
# node A's served tree (TREE_AR, tree_allreduce.py). BLOCK names the serve; its record arm comes from the store (--store-read).
#   BLOCK=c343|72b6 [STEPS=s,...] bash spot-send/h8_replay.sh      under research run --gpus 8 --store-read --cwd clone
set -uo pipefail
D=${RESEARCH_RUN_DIR:?run this under research run}; IN=$D/inputs; OUT=$D/out; mkdir -p "$IN" "$OUT"
HERE=$(cd "$(dirname "$0")" && pwd)
CU=/workspace/cp/cuda-12.9; OV=/workspace/cp/overlay-nodeA; V=/workspace/cp/venv312
BY=https://cursor.com/agents/bc-5609d8d8-985e-50b9-9216-15dfd5276546
TREE='(((((((0,2),3),1),5),6),4),7)'
SRC='art:3ffc7fc21dc9978b0a071b8da7f6838594f356f9455ee5330d12f36de195f2e9 (kernels probe r20261107-192422-508b, vy-spot-glm53)'
case ${BLOCK:?BLOCK: the served block} in
  c343) ART=art:c5a054e9b1eb5232a58217a9d131864148b6459902807571e74bbc91060e21fe; ARMDIR=out/boundary; LOADN=saturated
        EEF=out/engine_env.txt; STEPS=${STEPS:-5833}; MLA=8 ;;
  72b6) ART=art:e4098b2bb706d8c42c5eff1ff545515bc20620c51b3719005cb3c0b0fd908216; ARMDIR=out/pinned/glm53/record; LOADN=saturated
        EEF=out/serve_engine_env.txt; STEPS=${STEPS:-163,1321}; MLA=8 ;;
  *) echo "replay: unknown BLOCK $BLOCK"; exit 2 ;;
esac
echo "replay: $BLOCK on $(hostname) $(date -u +%FT%TZ) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset}"
for f in serve_logprobs.patch replay_aborts.patch dsa_topk.py dsa_topk_kernels.py capture.py stage.py spot_job.sh spot_tp8.sh \
    gpu-lease sitecustomize.py long_rows.py spot_canonical.sh tree_allreduce.py node_record.sh; do cp "$HERE/$f" "$IN/" || exit 2; done
X=/workspace/spot-inputs/$BLOCK
if [ ! -f "$X/$ARMDIR/meta.json" ]; then
  research data fetch "$ART" --to "$X" --path "$ARMDIR/**" --path "$EEF" --path out/topk-serve.json \
    --path "out/served/**" --path out/served.json >/dev/null || { echo "replay: fetching $ART failed"; exit 3; }
  research keep "$X" --owner @proofs --expires 2026-11-15 --priority low --art "$ART" \
    --why "$BLOCK's served record arm for OVERHEADDEF's cross-machine replays on this node" --by "$BY" >/dev/null || true
fi
find "$X" -type f | sort | xargs sha256sum | sed "s|$X/||" > "$OUT/served_inputs.sha256"; wc -l < "$OUT/served_inputs.sha256"
"$CU/bin/nvcc" --version | grep -q V12.9.86 || { echo "replay: no CUDA 12.9.86 toolkit at $CU: run h8_setup.sh"; exit 2; }
# the serve's engine environment names node A's toolkit path; here the same toolkit (V12.9.86) is at $CU
sed "s|^CUDA_HOME=/usr/local/cuda-12.9\$|CUDA_HOME=$CU|" "$X/$EEF" > "$OUT/engine_env.h8.txt"
diff "$X/$EEF" "$OUT/engine_env.h8.txt"
export CUDA_HOME=$CU PATH="$CU/bin:$PATH" PYTHONPATH="$OV${PYTHONPATH:+:$PYTHONPATH}"
VENV=$V CUDA_HOME=$CU bash "$IN/node_record.sh" "$OUT/node" 2>&1 | tail -2
VENV=$V SERVED_OUT=$X/out REC=$X/$ARMDIR LOAD=$LOADN ENGINE_ENV_FILE=$OUT/engine_env.h8.txt OLD=/nonexistent \
  STEPS=$STEPS MIN_FREE_G=${MIN_FREE_G:-30} KEEP_TENSORS=opened MLA_TOKENS=$MLA SAMPLER=1 GATE_EVERY=0 STAGE=0 \
  ALLREDUCE_TREE="$TREE" ALLREDUCE_TREE_SOURCE="$SRC" TREE_AR="$TREE" TREE_AR_SOURCE="$SRC" \
  bash "$IN/spot_canonical.sh"
rc=$?
echo "replay: done rc=$rc $(date -u +%FT%TZ) $(df -BG --output=avail /workspace | tail -1) free"
exit $rc
