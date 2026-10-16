#!/usr/bin/env bash
# noise_ft.py on one leased GPU of vy-nebius-2 (cwd = the clone's root), one config after another, each under its own
# bounded lease: the GPU selftest, then the untuned evaluation, the noise-aware LoRA fine-tune and its checkpoints'
# evaluations (16 WikiText-2 test windows, NFT_SEEDS seeds).  Training data: C4's English train shard 0 (fallback:
# WikiText-2's train split), downloaded into scratch.  Clocks are read, never set.
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-13 --source . --cwd clone --custody-ttl 12h \
#     --env GPU_LEASE_WHO=... --env NFT_CONFIGS="lin-u5 lin-u8" [--env NFT_TRAIN_MIN=40 --env NFT_EVAL_AT=...] \
#     --send benchmarks/fp8_instantiation/run_noise_ft.sh -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/run_noise_ft.sh"'
set -uo pipefail
export O=${RESEARCH_RUN_DIR:?}/out
mkdir -p "$O"
export PY=/workspace/jobs/venv312/bin/python
export SNAP=/workspace/hf/hub/models--unsloth--Llama-3.1-8B/snapshots/3f0d51f8e5640f98f1a96ea9044a0e55c0a83814
export HF_HUB_OFFLINE=1
CAPTURE=/workspace/pouw/fp8-security/llama8b-w0/ids.npy
WT=/workspace/hf/datasets/Salesforce___wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
S=benchmarks/fp8_instantiation/noise_ft.py
SCR=$(mktemp -d)
trap 'rm -rf "$SCR"' EXIT
"$PY" -c "import pyarrow" 2>/dev/null || "$PY" -m pip install -q --target "$SCR/pa" pyarrow > "$O/pip.txt" 2>&1 \
  || uv pip install --python "$PY" --target "$SCR/pa" pyarrow >> "$O/pip.txt" 2>&1 || exit 4
export PYTHONPATH=$SCR/pa
"$PY" benchmarks/fp8_instantiation/windows.py "$SNAP" "$WT/wikitext-test.arrow" "$CAPTURE" "${NFT_WINDOWS:-16}" "$O/windows.npy" \
  > "$O/windows.json" 2>> "$O/pip.txt" || exit 4
N=$(( ${NFT_STEPS:-1000} * ${NFT_ACCUM:-2} + 64 ))
C4=https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz
if curl -sfL --max-time 300 "$C4" -o "$SCR/c4.json.gz" && "$PY" "$S" --corpus "$SNAP" "$SCR/corpus.npy" "$N" 2048 "$SCR/c4.json.gz" > "$O/corpus.json"; then
  echo "c4-train.00000-of-01024 $(sha256sum "$SCR/c4.json.gz" | cut -c1-16)" >> "$O/corpus.json"
else
  "$PY" "$S" --corpus "$SNAP" "$SCR/corpus.npy" "$N" 2048 "$WT/wikitext-train.arrow" > "$O/corpus.json" || exit 5
fi
sha256sum "$SCR/corpus.npy" | cut -c1-64 >> "$O/corpus.json"
export CORPUS=$SCR/corpus.npy
rc=0
for cfg in ${NFT_CONFIGS:?}; do
  export CFG=$cfg
  # shellcheck disable=SC2016
  gpu-lease 1 --wait --max-min "${NFT_MAX_MIN:-75}" -- bash -c '
    mkdir -p "$O/$CFG"
    nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,clocks.mem,power.limit --format=csv -i "$GPU_LEASE_UUID" > "$O/$CFG/gpu.txt" 2>&1
    "$PY" "$S" --selftest > "$O/$CFG/selftest.json" 2> "$O/$CFG/selftest_stderr.txt" || exit 3
    "$PY" "$S" "$SNAP" "$O/windows.npy" "$CORPUS" "$O" "$CFG" 2> "$O/$CFG/stderr.txt"
    r=$?
    nvidia-smi --query-gpu=index,clocks.sm,clocks.mem --format=csv -i "$GPU_LEASE_UUID" >> "$O/$CFG/gpu.txt" 2>&1
    exit $r'
  r=$?
  echo "$cfg $r" >> "$O/status.txt"
  [ $r -ne 0 ] && rc=$r
  tail -3 "$O/$cfg/selftest_stderr.txt" "$O/$cfg/stderr.txt" 2>/dev/null
done
"$PY" "$S" --table "$O"/*/results.jsonl > "$O/table.md" 2>> "$O/status.txt"
exit $rc
