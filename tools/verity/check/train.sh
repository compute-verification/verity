#!/usr/bin/env bash
# Goes once no launcher names this path: a launcher from before the move runs a candidate's check from it.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/tools/check/train.sh" "$@"
