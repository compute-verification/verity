#!/usr/bin/env bash
# Regenerate tests/fixtures/lean_band_vectors.json from the Lean model.  Needs a Lake package with the POUS trusted
# layer (`lean/pous`) plus, as extra root modules, DenseMeets, DenseReCert, SpongeDense, ChainEPF,
# HintCard.PartialOutput, PubEncoderModel, PubEncoder, BandChainModel, BandChain, BandEncDec and BandPubEncoder
# (lean/submissions/{dense,sponge-dense,hintcard,public-encoder,band-chain}), built with
# `lake build BandEncDec BandPubEncoder`.
#   bash tests/lean/generate_band.sh LAKE_PACKAGE_DIR
set -euo pipefail
pkg=${1:?usage: generate_band.sh LAKE_PACKAGE_DIR}
here=$(cd "$(dirname "$0")" && pwd)
cd "$pkg"
cp "$here/PousBandCheck.lean" PousBandCheck.lean
lake env lean --run PousBandCheck.lean | python3 -c '
import json, sys
data = json.loads(sys.stdin.read())
data = {"source": "tests/lean/PousBandCheck.lean via tests/lean/generate_band.sh", **data}
json.dump(data, open(sys.argv[1], "w"), indent=1)
' "$here/../fixtures/lean_band_vectors.json"
