# Two replays of a served ladder window with topk_probe loaded (TOPK_PROBE=sort unless set), under research run with
# --cwd source at the ladder's tree; SRC is the served run's model directory (record/ and its plan). EXTRA_SITE=DIR goes on
# PYTHONPATH ahead of the probe (another plugin's dist-info and modules), and TOPK_PROBE_REAL=module:function names the
# top-k the probe calls and records in place of vLLM's.
#   research run --on NODE --project verity --source TREE --cwd source --send topk_probe.py --send compare.py \
#     --send probe.sh -- bash -c 'SRC=<served run>/out/pinned/glm53 bash "$RESEARCH_RUN_DIR/inputs/probe.sh"'
set -uo pipefail
D=${RESEARCH_RUN_DIR:?}; OUT=$D/out
SRC=${SRC:?the model directory of the served run}
VENV=${VENV:-/workspace/venv312}; PY=$VENV/bin/python
export PATH="$VENV/bin:$PATH"
export HF_HOME=${HF_HOME:-/workspace/hf} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.9}
export PATH="$CUDA_HOME/bin:$PATH"
unset NCCL_P2P_DISABLE VLLM_USE_DEEP_GEMM
export LADDER_ARM=pinned TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0
d=$OUT/pinned/glm53; mkdir -p "$d"
cp -a "$SRC/plan.jsonl" "$SRC/serve.json" "$SRC/arm_env.txt" "$d/" && cp -a "$SRC/record" "$d/record" || exit 2
rm -rf "$d/record/route_capture"
cache=$(mktemp -d)
export VLLM_CACHE_ROOT=$cache/vllm TORCHINDUCTOR_CACHE_DIR=$cache/inductor TRITON_CACHE_DIR=$cache/triton
site=$(mktemp -d); mkdir -p "$site/verity_topk_probe-0.0.dist-info"
printf 'Metadata-Version: 2.1\nName: verity-topk-probe\nVersion: 0.0\n' > "$site/verity_topk_probe-0.0.dist-info/METADATA"
printf '[vllm.general_plugins]\nverity_topk_probe = topk_probe:register\n' > "$site/verity_topk_probe-0.0.dist-info/entry_points.txt"
cp "$D/inputs/topk_probe.py" "$site/"
export PYTHONPATH="${EXTRA_SITE:+$EXTRA_SITE:}$site${PYTHONPATH:+:$PYTHONPATH}" TOPK_PROBE=${TOPK_PROBE:-sort}
echo "TOPK_PROBE=$TOPK_PROBE TOPK_PROBE_REAL=${TOPK_PROBE_REAL:-} EXTRA_SITE=${EXTRA_SITE:-} SRC=$SRC" | tee "$d/probe_env.txt"
"$PY" -c 'from vllm.utils.flashinfer import has_flashinfer; print(f"has_flashinfer={has_flashinfer()}")' >> "$d/probe_env.txt"
rc=0
for i in 1 2; do
  export TOPK_PROBE_OUT=$d/probe-$i
  echo "== probe replay $i $(date -u +%FT%TZ)"
  "$PY" benchmarks/replay_ladder/ladder.py replay --model glm53 --out "$d" --index "$i" --logprobs > "$d/replay-$i.log" 2>&1 \
    || { tail -n 40 "$d/replay-$i.log"; rc=1; }
  grep '^{"mode"' "$d/replay-$i.log" | tail -n 1
  find "$cache" -type f -iname '*autotun*' -exec sha256sum {} + | sed "s|^|replay-$i |" >> "$d/autotune.sha256"
done
"$PY" "$D/inputs/compare.py" "$d/probe-1" "$d/probe-2" | tee "$d/probe_compare.json"
rm -rf "$cache" "$site"
exit $rc
