#!/usr/bin/env bash
# prep.sh: covert_witness's fixed side on one node, run there in its CPU pool (fixed). From an agent VM's committed checkout:
#
#   python3 benchmarks/covert_witness/starts_store.py pack --out /tmp/cw-starts.tar   # where the store's key is
#   research run --on NODE --project verity --source . --pool CPU_POOL --as training --pool-wait 1800 --timeout 80m \
#     --declared-output 'out/*' --send /tmp/cw-starts.tar -- bash benchmarks/covert_witness/prep.sh
#
# It installs the starts (from the sent tar, else the store), checks data version `sealed.DATA` (v3: `data_v3.py build
# --push` installs it on every node from one; here `data_v3.py verify --keep` checks each file against its record, the
# sets also against prepare.SET_SHA256, and leaves them read-only) and Pythia's stream, registers the sealed start
# (prepare.register_pythia) and records their retention, each step skipped when its output is already in place, so on a
# prepared node it only checks (and needs no tar). The node needs learning_frontier's GPT-2 stream first. List the node
# in nodes.txt (`NODE pool=GPU_POOL cpu=CPU_POOL`) once this passes.
set -eu
cd "$(dirname "$0")"
P=${CW_PYTHON:-/workspace/jobs/venv312/bin/python}
R=${CW_ROOT:-/workspace/research/scratch/covert-witness}
S=${CW_SEALED:-/workspace/research/te/sealed}
LF=${CW_LF_CORPUS:-/workspace/research/scratch/learning-frontier/fineweb-edu-gpt2-v1}
OUT=${OUT:-$(mktemp -d)}
[ -x "$P" ] || { echo "prep: no python at $P; set CW_PYTHON to one with torch and transformers"; exit 2; }
[ -f "$LF/train.bin" ] || { echo "prep: no GPT-2 training stream at $LF/train.bin; build it on this node with" \
  "benchmarks/learning_frontier/prep.py --dest $LF (or set CW_LF_CORPUS)"; exit 2; }

echo "--- the python, with a private overlay of what it lacks"
T="${RESEARCH_RUN_DIR:-$OUT}/pylib"
need=$("$P" - <<'EOF'
import importlib.util
print(" ".join(p for m, p in [("pyarrow", "pyarrow"), ("tiktoken", "tiktoken==0.14.0"), ("regex", "regex"),
                              ("huggingface_hub", "huggingface_hub"), ("transformers", "transformers"),
                              ("safetensors", "safetensors")] if importlib.util.find_spec(m) is None))
EOF
)
echo "missing: ${need:-none}"
if [ -n "$need" ]; then uv pip install --python "$P" --target "$T" --no-deps $need; fi
export PYTHONPATH="$T${PYTHONPATH:+:$PYTHONPATH}"

echo "--- starts"
tar="${RESEARCH_RUN_DIR:-}/inputs/cw-starts.tar"
if [ -f "$tar" ]; then python3 starts_store.py install --tar "$tar" --keep; else python3 starts_store.py install --keep; fi

echo "--- data v3 (the held-out sets, the training stream and its tokenizer) and Pythia's stream"
data_json=$(CW_DEVICE=cpu "$P" -c "import prepare; print(prepare.DATA_JSON)")
sets=$(CW_DEVICE=cpu "$P" -c "import prepare; print(' '.join(str(d) for d in prepare.SET_DIRS.values()))")
[ -f "$data_json" ] || { echo "prep: no $data_json: build data v3 once, on one node, with" \
  "\`data_v3.py build --push OTHER_NODES...\` (README), then run this on each"; exit 2; }
[ -f "$R/train/pythia.bin" ] || { echo "prep: no Pythia stream at $R/train/pythia.bin (data v2.2's data.py built it):" \
  "copy it from a prepared node"; exit 2; }
"$P" data_v3.py verify --keep
chmod -R a-w $sets
CW_DEVICE=cpu "$P" - <<'EOF'
import prepare
for split in (prepare.SEARCH_POOL, prepare.PROMOTION, prepare.FINAL):
    if set(prepare.manifest(split)["tokenizers"]) != set(prepare.SET_SHA256[split]):
        raise SystemExit(f"prep: the {split} set's tokenizers are not prepare.SET_SHA256's")
    for tok in prepare.manifest(split)["tokenizers"]:
        prepare.held_out(split, tok)
for tok in ("gpt2", prepare.data_v3.TOKENIZER):
    toks, man = prepare.held_out(prepare.SEARCH_POOL, tok)
    if int(prepare.token_bytes(tok)[toks[1:]].sum()) != man["bytes"]:
        raise SystemExit(f"prep: the search pool's {tok} tokens' bytes are not its documents' bytes")
print("held-out sets match their manifests and prepare.SET_SHA256")
EOF

echo "--- the sealed start"
if [ -f "$S/starts/pythia160m/start.json" ]; then echo "registered: $S/starts/pythia160m"
else "$P" -c "import prepare; prepare.register_pythia()"; fi
CW_FINAL=1 CW_DEVICE=cpu "$P" - <<'EOF'
import json

import torch

import prepare

net, rec, theta = prepare.load_start("pythia160m", device="cpu")
rows = prepare.TrainData("pythia", seed=1).next(2, device="cpu")
with torch.no_grad():
    logits = net.logits(rows[:, :-1])
loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), rows[:, 1:].reshape(-1))
print(json.dumps({"start": rec["name"], "digest": rec["digest"], "positions": len(prepare.positions(rec)),
                  "train_stream_nats_per_token": round(float(loss), 4)}))
EOF

echo "--- retention"
python3 -m research keep "$S" --owner @training --expires never --priority keep \
  --regenerate "bash benchmarks/covert_witness/prep.sh (values/ and decisions.jsonl are not regenerable)" \
  --why "covert_witness's sealed side: the promotion and final sets, the sealed start, the sealed values"
python3 -m research keep "$R/heldout" --owner @training --expires never --priority keep \
  --regenerate "bash benchmarks/covert_witness/prep.sh" --why "covert_witness's search pool"
python3 -m research keep "$R/train/pythia.bin" --owner @training --expires never --priority keep \
  --regenerate "bash benchmarks/covert_witness/prep.sh" --why "covert_witness's Pythia training stream"
echo "prep: ready"
