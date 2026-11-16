#!/usr/bin/env bash
# nvcc for a library a job builds off the card and carries to it, outside a ship's frame: ship_nvcc.sh with the same
# arguments, so the cubin doesn't follow where the tree or the source's directory sits (its header says how), with the
# toolkit CUDA names and a TMPDIR of this call's own, at a path of one length, which builds running side by side don't
# share. It is not the frame: nothing here holds a rebuild to the same bytes from run to run (ship.sh's header).
set -euo pipefail
: "${CUDA:?the toolkit whose nvcc builds}"
tmp=$(mktemp -d /tmp/job_nvcc.XXXXXXXX); trap 'rm -rf "$tmp"' EXIT
TMPDIR=$tmp "$(dirname "${BASH_SOURCE[0]}")/ship_nvcc.sh" "$@"
