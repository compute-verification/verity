#!/usr/bin/env bash
# Put the redacting env/printenv ahead of /usr/bin on this VM. Idempotent; /usr/bin/env (shebangs) is untouched.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
sudo install -m 755 "$here/env_guard.py" /usr/local/bin/env
sudo ln -sf /usr/local/bin/env /usr/local/bin/printenv
hash -r 2>/dev/null || true
echo "env guard installed: $(command -v env), $(command -v printenv)"
