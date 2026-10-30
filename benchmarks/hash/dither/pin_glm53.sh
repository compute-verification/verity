#!/usr/bin/env bash
# The int8 GLM-5.3 pin (checkpoints-int8.json's GLM53_INT8, recipe verity-int8-channel127/v3) in a CPU node's HF cache: the
# FP8-block base pin (GLM53, zai-org/GLM-5.3) downloaded into $HF_HOME/hub if missing, then `verity-vllm int8-checkpoint`:
#
#   pin   make the checkpoint from the base and write its new entry to $OUT/entry.json, then the per-tensor error report against
#         the base's exact FP8 dequant ($OUT/error-report.json) from a scratch copy of the manifests holding that entry;
#   make  re-make the registered pin and check every file's sha256 against checkpoints-int8.json.
#
# Both outputs stay on the node (about 705 GB of base and 757 GB of pin), recorded with `research keep` under pin_glm53.toml's
# owner until its expiries; the pin entry's resolved_by is its `by`. cwd = the source tree.
#
#   bash benchmarks/hash/dither/pin_glm53.sh pin|make      (on the CPU host whose HF cache serves it, up to 8 hours)
set -euo pipefail
MODE=${1:?usage: pin_glm53.sh pin|make}
ROLE=GLM53_INT8 REPO=verity-int8/GLM-5.3-INT8-V3 RECIPE=verity-int8-channel127/v3 JOBS=${JOBS:-10}
export HF_HOME=${HF_HOME_PIN:-/workspace/hf}
OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}
mkdir -p "$OUT" "$HF_HOME/hub"
stamp() { echo "[$(date -u +%H:%M:%SZ)] $*"; }

stamp "uv sync"
uv sync -q --all-packages --extra torch-cpu
VV=(uv run --no-sync verity-vllm int8-checkpoint)
pin_data() { uv run --no-sync python -c 'import functools, sys, tomllib
print(functools.reduce(dict.__getitem__, sys.argv[2:], tomllib.load(open(sys.argv[1], "rb"))))' \
  benchmarks/hash/dither/pin_glm53.toml "$@"; }
OWNER=$(pin_data owner)
BY=$(pin_data by)
BASE_EXPIRES=$(pin_data expires base)
PIN_EXPIRES=$(pin_data expires pin)

stamp "base download"
base_dir=$(uv run --no-project -q --with 'huggingface_hub[hf_xet]' python - <<'EOF'
import json, os
from huggingface_hub import snapshot_download
base = next(c for c in json.load(open("integrations/vllm/manifests/checkpoints.json"))["checkpoints"] if c["role"] == "GLM53")
print(snapshot_download(base["repo"], revision=base["revision"], allow_patterns=["*.json", "*.safetensors"],
                        cache_dir=os.environ["HF_HOME"] + "/hub", max_workers=16))
EOF
)
research keep "$HF_HOME/hub/models--zai-org--GLM-5.3" --owner "$OWNER" --expires "$BASE_EXPIRES" --priority low \
  --regenerate "huggingface_hub snapshot_download zai-org/GLM-5.3 at checkpoints.json's GLM53 revision" \
  --why "the FP8-block base of GLM53_INT8, read by int8-checkpoint make and report" || echo "warning: research keep failed: the path stays unrecorded (never deleted by gc)"
stamp "base at $base_dir"

pin_dir="$HF_HOME/hub/models--${REPO/\//--}"
mkdir -p "$pin_dir"
research keep "$pin_dir" --owner "$OWNER" --expires "$PIN_EXPIRES" --priority normal \
  --regenerate "verity-vllm int8-checkpoint make --role $ROLE --hf-home $HF_HOME --jobs $JOBS" \
  --why "GLM-5.3 int8 ($RECIPE) for the Dither leg's serve" || echo "warning: research keep failed: the path stays unrecorded (never deleted by gc)"

if [ "$MODE" = make ]; then
  stamp "make $ROLE"
  "${VV[@]}" make --role "$ROLE" --hf-home "$HF_HOME" --jobs "$JOBS" | tee "$OUT/make.json"
  stamp "done"
  exit 0
fi

stamp "pin $ROLE"
"${VV[@]}" pin --base-role GLM53 --role "$ROLE" --repo "$REPO" --recipe "$RECIPE" --hf-home "$HF_HOME" \
  --by "$BY" --at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --jobs "$JOBS" > "$OUT/entry.json"
du -sb "$pin_dir"/snapshots/* | tee "$OUT/pin-size.txt"

stamp "report"
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
cp integrations/vllm/manifests/checkpoints.json integrations/vllm/manifests/checkpoints-int8.json "$scratch/"
python3 - "$scratch/checkpoints-int8.json" "$OUT/entry.json" <<'EOF'
import json, sys
path, entry = sys.argv[1:]
doc = json.load(open(path))
doc["checkpoints"].append(json.load(open(entry)))
json.dump(doc, open(path, "w"), indent=1)
EOF
"${VV[@]}" report --role "$ROLE" --checkpoints "$scratch/checkpoints.json" --hf-home "$HF_HOME" --jobs "$JOBS" \
  --out "$OUT/error-report.json" | tee "$OUT/error-summary.json"
stamp "done"
