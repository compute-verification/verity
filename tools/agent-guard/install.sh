#!/usr/bin/env bash
# Put the redacting env/printenv and the credential-URL git guard ahead of /usr/bin on this VM. Idempotent; /usr/bin/env
# (shebangs) and /usr/bin/git are untouched. The git guard is tested where it is installed and removed if a check fails.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
sudo install -m 755 "$here/env_guard.py" /usr/local/bin/env
sudo ln -sf /usr/local/bin/env /usr/local/bin/printenv
sudo install -m 755 "$here/git_guard.sh" /usr/local/bin/git
if ! bash "$here/test_git_guard.sh" /usr/local/bin/git >/dev/null; then
  sudo rm -f /usr/local/bin/git
  echo "git guard failed its self-test and was removed: bash $here/test_git_guard.sh" >&2
  exit 1
fi
hash -r 2>/dev/null || true
echo "env guard installed: $(command -v env), $(command -v printenv); git guard installed: $(command -v git)"
