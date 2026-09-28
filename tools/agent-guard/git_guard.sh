#!/usr/bin/env bash
# `git` that keeps credentials out of remote URLs: installed ahead of /usr/bin on agent VMs (tools/agent-guard/install.sh).
#
# - `remote -v`, `remote get-url`, `remote show` and `ls-remote --get-url` print URLs with the credential removed. Git prints
#   URLs after `insteadOf` rewriting, so on a Cursor VM these print cursor[bot]'s token from ~/.gitconfig, and a clone whose
#   own URL holds a token prints that. Removing the userinfo keeps the URL usable: git rewrites it again when it is used.
#   When the repository's own config holds the credential, a warning names the remote (never the URL).
# - `clone`, `remote add` and `remote set-url` refuse a URL that carries a credential (exit 2): use a credential helper.
# Anything else runs /usr/bin/git unchanged. Not covered: `git config` output, reading .git/config. VERITY_GIT_GUARD=off disables.
REAL=${VERITY_REAL_GIT:-/usr/bin/git}
[ "${VERITY_GIT_GUARD:-on}" = off ] && exec "$REAL" "$@"

pre=() i=0 sub=""
args=("$@")
while [ $i -lt ${#args[@]} ]; do
  a=${args[$i]}
  case $a in
    -C|-c|--git-dir|--work-tree|--namespace|--config-env|--super-prefix) pre+=("$a" "${args[$((i + 1))]}"); i=$((i + 2)) ;;
    -*) pre+=("$a"); i=$((i + 1)) ;;
    *) sub=$a; i=$((i + 1)); break ;;
  esac
done
rest=("${args[@]:$i}")

# userinfo with a password, or a token as the user name (GitHub's prefixes, or anything 30+ characters long)
CRED='://([^/@[:space:]]*:[^/@[:space:]]*|gh[pousr]_[A-Za-z0-9_]+|github_pat_[A-Za-z0-9_]+|[A-Za-z0-9_-]{30,})@'
strip() { sed -E "s#${CRED}#://#g"; }

refuse_creds() {
  local a
  for a in "$@"; do
    if printf '%s' "$a" | grep -Eq "$CRED"; then
      echo "git guard: refusing a remote URL that carries a credential ($(printf '%s' "$a" | strip)). Keep tokens out of" \
           "URLs: use a credential helper (research-notes: kb/cloud-lane-setup.md section 1; verity: the GitHub broker)." >&2
      exit 2
    fi
  done
}

filtered() {
  "$REAL" "$@" | strip
  local rc=${PIPESTATUS[0]}
  local stored
  stored=$("$REAL" "${pre[@]}" config --get-regexp '^remote\..*\.url$' 2>/dev/null | grep -E "$CRED" | cut -d' ' -f1 | tr '\n' ' ')
  [ -n "$stored" ] && echo "git guard: ${stored}holds a credential in this repository's config; move it to a credential helper" \
                          "and set the URL without it (kb/cloud-lane-setup.md section 1)." >&2
  return "$rc"
}

case $sub in
  clone) refuse_creds "${rest[@]}" ;;
  remote)
    verb="" ; for a in "${rest[@]}"; do case $a in -*) ;; *) verb=$a; break ;; esac; done
    case $verb in
      add|set-url) refuse_creds "${rest[@]}" ;;
      ""|get-url|show) filtered "$@"; exit ;;
    esac ;;
  ls-remote)
    for a in "${rest[@]}"; do [ "$a" = --get-url ] && { filtered "$@"; exit; }; done ;;
esac
exec "$REAL" "$@"
