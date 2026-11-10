#!/usr/bin/env bash
# Put the redacting env/printenv, the credential-URL git guard, the masking process lister (procs.py: `procs`, and the
# `ps` and `pgrep` that mask what the real ones print) and the `research` shim (research.sh) ahead of /usr/bin on this VM.
# Idempotent; /usr/bin/env (shebangs), /usr/bin/git, /usr/bin/ps and /usr/bin/pgrep are untouched. Each guard is tested
# where it is installed; the git guard, or `ps` and `pgrep`, are removed if a check fails, the env guard kept.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
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
sudo install -m 755 "$here/research.sh" /usr/local/bin/research
sudo install -m 755 "$here/procs.py" /usr/local/bin/procs
sudo ln -sf /usr/local/bin/procs /usr/local/bin/ps
sudo ln -sf /usr/local/bin/procs /usr/local/bin/pgrep
if ! bash "$here/test_procs.sh" /usr/local/bin/procs >/dev/null; then
  sudo rm -f /usr/local/bin/ps /usr/local/bin/pgrep
  echo "the process lister failed its self-test; its ps and pgrep were removed: bash $here/test_procs.sh" >&2
  exit 1
fi
hash -r 2>/dev/null || true
echo "env guard installed: $(command -v env), $(command -v printenv); git guard installed: $(command -v git);" \
  "process lister installed: $(command -v procs), $(command -v ps), $(command -v pgrep);" \
  "research: $(command -v research), origin/main's unless the checkout changes it"
