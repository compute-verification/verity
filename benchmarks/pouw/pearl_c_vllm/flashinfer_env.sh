# Sourced by setup.sh, window.sh and served_profile.sh, with WORK set.  FlashInfer keys its JIT cache by version string and
# arch only ($FLASHINFER_WORKSPACE_BASE/.cache/flashinfer/<version>/<arch>, under $HOME by default), so on a node's shared
# cache two builds that both call themselves one version overwrite each other's kernels, and a window would rebuild them
# (vy-nebius-2, 2026-10-05).  The base here is keyed by the venv's FlashInfer build: its dist-info RECORDs' sha256.  No
# RECORD fails the caller (set -euo pipefail).  The RECORDs (flashinfer-python's, -cubin's, -jit-cache's) are hashed in C
# order, which no pod's locale changes.
fi_records=$(LC_ALL=C ls "$WORK"/venv312/lib/python3*/site-packages/flashinfer*.dist-info/RECORD)
export FLASHINFER_WORKSPACE_BASE=$WORK/flashinfer/$(cat $fi_records | sha256sum | cut -c1-16)
