#!/usr/bin/env bash
# `verity bench bandwidth NAME [ARGS]`: this directory's one runner (cwd = the source tree). It hands ARGS to NAME's own
# script, unchanged. Each script is one Attempt, the command of a `research run`, as its header says.
#
#   chip          run_chip.sh: what one card holds on chip (chip.cu, bwchip)
#   write_gpu     run_write_gpu.sh: the write ceilings' GPU half (write.cu, bwwrite)
#   write_host    run_write_host.sh: the write ceilings' host half (write_host.c, bwwrite_host)
set -euo pipefail
here=$(dirname "$0")
case "${1:-}" in
  chip | write_gpu | write_host) name=$1; shift; exec bash "$here/run_$name.sh" "$@" ;;
  *) echo "usage: verity bench bandwidth chip|write_gpu|write_host [ARGS]" >&2; exit 2 ;;
esac
