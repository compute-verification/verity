#!/usr/bin/env bash
# Put the redacting env/printenv and the credential-URL git guard ahead of /usr/bin on this VM. Idempotent; /usr/bin/env
# (shebangs) and /usr/bin/git are untouched. Both guards are tested where they are installed; the git guard is removed if a
# check fails, the env guard kept.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
multi=$(python3 -c 'import os; print(" ".join(sorted(k for k, v in os.environ.items() if "\n" in v and not k.startswith("BASH_FUNC_"))))')
if [ -n "$multi" ]; then   # names only: a multi-line value prints whole through any name filter, so it must not be here at all
  echo "agent-guard: this VM's environment holds a multi-line value in: $multi. Delete that Cursor secret (Dashboard -> Cloud" \
       "Agents -> Secrets) or store it as one line (NAME_B64, base64 -w0), then start a fresh VM; every shell command is refused" \
       "until then (shell_hook.py)" >&2
  exit 1
fi
sudo install -m 755 "$here/env_guard.py" /usr/local/bin/env
sudo ln -sf /usr/local/bin/env /usr/local/bin/printenv
if ! bash "$here/test_env_guard.sh" /usr/local/bin/env >/dev/null; then
  echo "env guard failed its self-test (it stays installed: redacting too much is the safe failure): bash $here/test_env_guard.sh" >&2
  exit 1
fi
sudo install -m 755 "$here/git_guard.sh" /usr/local/bin/git
if ! bash "$here/test_git_guard.sh" /usr/local/bin/git >/dev/null; then
  sudo rm -f /usr/local/bin/git
  echo "git guard failed its self-test and was removed: bash $here/test_git_guard.sh" >&2
  exit 1
fi
hash -r 2>/dev/null || true
echo "env guard installed: $(command -v env), $(command -v printenv); git guard installed: $(command -v git)"
