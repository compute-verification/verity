# Install the templates into /workspace/jobs/tvi/templates and write READY (sent as $RESEARCH_RUN_DIR/inputs/READY).
# Runs with --cwd source on the commit whose benchmarks/tvi/ the templates come from; refuses if READY already exists.
set -euo pipefail
T=/workspace/jobs/tvi
test -x "$T/venv/bin/python" || { echo "no venv at $T/venv"; exit 3; }
test ! -e "$T/venv/READY" || { echo "READY exists; the venv is frozen"; exit 3; }
mkdir -p "$T/templates"
cp benchmarks/tvi/grpo_train.py benchmarks/tvi/sft_train.py benchmarks/tvi/eval_pass_at_k.py benchmarks/tvi/tvi_common.py \
   benchmarks/tvi/smoke.sh "$T/templates/"
git rev-parse HEAD 2>/dev/null > "$T/templates/COMMIT" || basename "$PWD" > "$T/templates/COMMIT"
"$T/venv/bin/python" "$T/templates/grpo_train.py" --help > /dev/null
"$T/venv/bin/python" "$T/templates/sft_train.py" --help > /dev/null
cp "$RESEARCH_RUN_DIR/inputs/READY" "$T/venv/READY"
cat "$T/venv/READY" | tee "$OUT/READY"
ls -l "$T/templates" | tee "$OUT/templates.txt"
sha256sum "$T"/templates/* | tee -a "$OUT/templates.txt"
