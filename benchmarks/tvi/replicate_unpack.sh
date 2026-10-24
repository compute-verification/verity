# On a node without the venv: unpack replicate_pack.sh's tarball (sent as inputs/tvi-venv.tar.zst, its sha256 in $TVI_SHA256)
# at the same absolute paths, check the imports, and write READY from the sent inputs/READY plus a replica line.
set -euo pipefail
: "${TVI_SHA256:?set TVI_SHA256 to the sha256 the pack run printed}"
F="$RESEARCH_RUN_DIR/inputs/tvi-venv.tar.zst"
echo "$TVI_SHA256  $F" | sha256sum -c
test ! -e /workspace/jobs/tvi/venv/bin/python || { echo "a venv is already here; refusing to overwrite"; exit 3; }
[ -d /workspace/jobs ] || sudo -n install -d -o research -g research -m 755 /workspace/jobs
( cd / && time tar -I zstd -xf "$F" -C / )
/workspace/jobs/tvi/venv/bin/python -c 'import torch, trl, peft, vllm, transformers, math_verify; from trl import GRPOTrainer, SFTTrainer; print(torch.__version__, trl.__version__, peft.__version__, vllm.__version__, transformers.__version__)' | tee "$OUT/imports.txt"
{ cat "$RESEARCH_RUN_DIR/inputs/READY"; printf '\n%s replica (run %s): unpacked from tarball sha256 %s, imports checked\n' "$(hostname)" "$RESEARCH_RUN_ID" "$TVI_SHA256"; } > /workspace/jobs/tvi/venv/READY
cp /workspace/jobs/tvi/venv/READY "$OUT/READY"
