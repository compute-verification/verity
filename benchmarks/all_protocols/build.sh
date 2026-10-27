#!/usr/bin/env bash
# What the sampled-proofs roles of `verity run CONFIG` run (verity_all_protocols.parties), built from this tree as run.sh
# builds them: M0's flock-circuit (in DIR/flock), Lean's flock-verify and flock-firewall, the sampled-proofs verifier of
# record and the product over a configuration's components (onsite's `product`), in the clone; their paths in
# DIR/tools.json. Run from the repository's root: bash benchmarks/all_protocols/build.sh DIR
set -uo pipefail
D=${1:?usage: build.sh DIR}; mkdir -p "$D"; D=$(cd "$D" && pwd); REPO=$(pwd)
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$HOME/.elan/bin:$PATH
command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
(cd verity/core/service/flock && lake build flock-verify flock-firewall) > "$D/build-lean.log" 2>&1 &
LP=$!
mkdir -p "$D/m0-build"
RESEARCH_RUN_DIR=$D/m0-build FLOCK_WORK=$D/flock MODE=build GPU=0 bash verity/ml/flock/pod/60-circuit.sh > "$D/m0-build/build.log" 2>&1 \
  || { tail -40 "$D/m0-build/build.log"; exit 6; }
wait $LP || { tail -40 "$D/build-lean.log"; exit 7; }
(cd verity/core/service/sampled_proofs && lake build sampled-proofs-verifier) > "$D/build-sampled-proofs.log" 2>&1 \
  || { tail -40 "$D/build-sampled-proofs.log"; exit 7; }
(cd verity/core/protocols/onsite && lake build product) > "$D/build-product.log" 2>&1 || { tail -40 "$D/build-product.log"; exit 7; }
BIN=$D/flock/flock-gpu-link/flock/target/release/flock-circuit
LEAN=$REPO/verity/core/service/flock/.lake/build/bin/flock-verify
PRODUCT=$REPO/verity/core/protocols/onsite/.lake/build/bin/product
M0PATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/ml/flock/python:$REPO/integrations/vllm
sha256sum "$BIN" "$LEAN" "${LEAN%/*}/flock-firewall" "$PRODUCT" | tee "$D/binaries.sha256"
python3 -c 'import json, sys; print(json.dumps(dict(zip(["flock-circuit", "flock-verify", "flock-firewall", "product",
  "m0-pythonpath"], sys.argv[1:]))))' "$BIN" "$LEAN" "${LEAN%/*}/flock-firewall" "$PRODUCT" "$M0PATH" > "$D/tools.json"
