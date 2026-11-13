#!/usr/bin/env bash
# Put the redacting env/printenv, the credential-stripping git guard (with the branch rule's pre-push hook it gives every push)
# and the `research` shim (research.sh) ahead of /usr/bin on this VM. Idempotent; /usr/bin/env (shebangs) and /usr/bin/git
# are untouched. Both guards are tested where they are installed; the git guard is removed if a check fails, the env guard kept.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
sudo install -m 755 "$here/env_guard.py" /usr/local/bin/env
sudo ln -sf /usr/local/bin/env /usr/local/bin/printenv
if ! bash "$here/test_env_guard.sh" /usr/local/bin/env >/dev/null; then
  echo "env guard failed its self-test (it stays installed: redacting too much is the safe failure): bash $here/test_env_guard.sh" >&2
  exit 1
fi
# the branch rule's pre-push hook, which the git guard gives every push (pre_push.py)
hooks=/usr/local/lib/agent-guard/hooks
sudo install -d "$hooks"
printf '#!/bin/sh\nexec python3 '"'"'%s/pre_push.py'"'"' "$@"\n' "$here" | sudo tee "$hooks/pre-push.new" >/dev/null
sudo chmod 755 "$hooks/pre-push.new" && sudo mv -f "$hooks/pre-push.new" "$hooks/pre-push"
sudo install -m 755 "$here/git_guard.sh" /usr/local/bin/git
if ! bash "$here/test_git_guard.sh" /usr/local/bin/git >/dev/null; then
  sudo rm -f /usr/local/bin/git
  echo "git guard failed its self-test and was removed: bash $here/test_git_guard.sh" >&2
  exit 1
fi
sudo install -m 755 "$here/research.sh" /usr/local/bin/research
hash -r 2>/dev/null || true
echo "env guard installed: $(command -v env), $(command -v printenv); git guard installed: $(command -v git), pushes through $hooks/pre-push;" \
  "research: $(command -v research), origin/main's unless the checkout changes it"
