#!/usr/bin/env bash
# Goes once the architecture branch lands: main's verified queue launches a candidate's check from this path.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/tools/check/train.sh" "$@"
