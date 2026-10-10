#!/usr/bin/env bash
# The layout move put this script at tools/verity/lean/setup.sh. The Cloud Agent environment's install command (Daniel's
# dashboard) still runs `bash tools/lean/setup.sh OLD_PACKAGE_DIR`, and while the file is missing every environment build
# fails, so new agent VMs boot from the 30 Sep snapshot. This maps the old package path through `tools/move/layout_map.toml`
# and runs the moved script. Delete it once the install command names tools/verity/lean/setup.sh.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
pkg=$(python3 - "$root" "${1:?usage: setup.sh PACKAGE_DIR}" <<'EOF'
import pathlib, sys, tomllib
root, old = pathlib.Path(sys.argv[1]), sys.argv[2].strip("/") + "/"
paths = tomllib.loads((root / "tools/move/layout_map.toml").read_text())["paths"]
hit = max((k for k in paths if k.endswith("/") and old.startswith(k)), key=len, default=None)
print((paths[hit] + old[len(hit):] if hit else old).rstrip("/"))
EOF
)
[ "$pkg" = "${1%/}" ] || echo "tools/lean/setup.sh: $1 is $pkg since the layout move"
exec bash "$root/tools/verity/lean/setup.sh" "$pkg"
