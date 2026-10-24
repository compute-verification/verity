# On a node with a READY venv: pack /workspace/jobs/tvi/{venv,templates} and the uv interpreter tree into the run dir's
# pack/ (outside $OUT, so custody lists it without uploading it). Then, from the VM: `research fetch RUN --all --on NODE`,
# and launch replicate_unpack.sh on the other node with `--send .../pack/tvi-venv.tar.zst`.
set -euo pipefail
test -e /workspace/jobs/tvi/venv/READY || { echo "no READY: the venv isn't final"; exit 3; }
mkdir -p "$RESEARCH_RUN_DIR/pack"
cd /
time tar -I "zstd -T0 -3" --exclude='__pycache__' --exclude='workspace/jobs/tvi/venv/READY' -cf "$RESEARCH_RUN_DIR/pack/tvi-venv.tar.zst" \
  workspace/jobs/tvi/venv workspace/jobs/tvi/templates workspace/jobs/cache/python
sha256sum "$RESEARCH_RUN_DIR/pack/tvi-venv.tar.zst" | tee "$OUT/sha256.txt"
cp /workspace/jobs/tvi/venv/READY "$OUT/READY"
