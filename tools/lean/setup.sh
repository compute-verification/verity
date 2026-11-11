#!/usr/bin/env bash
# The layout move put this script at tools/verity/lean/setup.sh. The Cloud Agent environment's install command (in the
# Cursor dashboard) still runs `bash tools/lean/setup.sh OLD_PACKAGE_DIR`, and while the file is missing every environment
# build fails, so new agent VMs boot from the last snapshot built before the move. This runs the moved script with the
# package path as given: that script finds where a path of an earlier layout is now, from the recorded moves, so this
# one reads no file a move removes. Delete it once the install command names tools/verity/lean/setup.sh.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
exec bash "$root/tools/verity/lean/setup.sh" "$@"
