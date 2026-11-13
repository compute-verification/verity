#!/usr/bin/env bash
# commit_gpu's runner: the directory's one job, `gpu.sh` beside this file, started as it is with the caller's arguments
# and environment. The job is the command of a `research run` on an RTX PRO 6000 node (cwd = the source tree's root,
# `$OUT` set); `gpu.sh`'s header has the full command, its knobs and what it leases, and README.md what it writes.
#
#   bash benchmarks/pouw/commit_gpu/run.sh
set -uo pipefail
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gpu.sh" "$@"
