#!/usr/bin/env bash
# `verity bench rate_squaring NAME [ARGS]`: this directory's one runner (cwd = the source tree). It hands ARGS to NAME's
# own script, unchanged. Each script is the command of a `research run`, as its header says.
#
#   sq      sq.sh: sq.cu on one GPU, every (FOLD, ILP) form gated against Python's pow and then timed
#   ifma    ifma.sh: ifma.c on the host's CPUs, each compiler's build gated and then timed
set -euo pipefail
here=$(dirname "$0")
case "${1:-}" in
  sq | ifma) name=$1; shift; exec bash "$here/$name.sh" "$@" ;;
  *) echo "usage: verity bench rate_squaring sq|ifma [ARGS]" >&2; exit 2 ;;
esac
