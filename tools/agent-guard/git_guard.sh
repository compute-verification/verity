#!/bin/bash
# `git` that keeps credentials out of what git prints: installed ahead of /usr/bin on agent VMs (tools/agent-guard/install.sh).
#
# - `remote -v`, `remote get-url`, `remote show` and `ls-remote --get-url` print URLs with the credential removed. Git prints
#   URLs after `insteadOf` rewriting, so on a Cursor VM these print cursor[bot]'s token from ~/.gitconfig, and a clone whose
#   own URL holds a token prints that. Removing the userinfo keeps the URL usable: git rewrites it again when it is used.
#   When the repository's own config holds the credential, a warning names the remote (never the URL).
# - `config` that reads (`--list`, `--get`, `--get-all`, `--get-regexp`, `--get-urlmatch`, a lone NAME, `list`, `get`; with
#   `--show-origin`, `--show-scope`, `--name-only` or `-z` too) and `var -l` print keys, values and git's errors with the
#   credential removed. Cursor's token sits in a key, `url.https://x-access-token:TOKEN@github.com/.insteadOf`, which
#   ~/.gitconfig or GIT_CONFIG_KEY_n sets. Also removed: what follows `Authorization:` and its scheme word (the value of
#   `http.*.extraheader`), and a GitHub token standing alone. Everything else prints as it is, so a program that reads a
#   helper, a merge driver or a URL gets what git uses. A `config` that writes (NAME VALUE, `--add`, `--unset`, `--unset-all`,
#   `--replace-all`, `--rename-section`, `--remove-section`, `set`, `unset`), `--get-color` and, on a terminal, `--edit` run
#   unchanged; off a terminal the editor still edits, and what it prints is filtered.
# - `clone`, `remote add` and `remote set-url` refuse a URL that carries a credential (exit 2): use a credential helper.
# - `push` runs the branch rule's pre-push hook (pre_push.py, which install.sh puts in $AGENT_GUARD_GIT_HOOKS, default
#   /usr/local/lib/agent-guard/hooks) whoever runs it, script or child process: `-c core.hooksPath=` that directory after the
#   caller's own options, and `--no-verify` dropped; the hook then runs the hook the push would have run otherwise.
# Anything else runs /usr/bin/git unchanged. Not covered: .git/config and ~/.gitconfig read as files, a git alias that runs
# `config`, a header other than `Authorization:`, a token that is not GitHub's outside a URL, and the environment's
# GIT_CONFIG_* variables, of which env_guard.py redacts only GIT_CONFIG_KEY_n (by its name). VERITY_GIT_GUARD=off disables.
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

# In a URL: userinfo with a password, or a token as the user name (GitHub's prefixes, or anything 30+ characters long; an
# installation token has dots). Anywhere: an `Authorization:` header's credential, and a GitHub token.
CRED='://([^/@[:space:]]*:[^/@[:space:]]*|(gh[pousr]|github_pat)_[A-Za-z0-9._-]+|[A-Za-z0-9._-]{30,})@'
AUTH='(authorization:[[:blank:]]*([A-Za-z0-9-]+[[:blank:]]+)?)[^\n]+'
BARE='(gh[pousr]|github_pat)_[A-Za-z0-9._-]{20,}'
# sed -z: a record ends at NUL, so no match spans two records of `-z` output, and no pattern matches across a newline. sed
# is found on a PATH that lacks it (a test's fake bin).
strip() {
  LC_ALL=C PATH=$PATH:/usr/bin:/bin sed -z -E -e "s#${CRED}#://#g" -e "s#${AUTH}#\1[REDACTED]#gI" -e "s#${BARE}#[REDACTED]#g"
}

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

# git "$@" with its stdout and its stderr each through strip, and git's exit status
stripped() {
  local out
  { "$REAL" "$@" 2>&1 >&"$out" {out}>&- | strip >&2 {out}>&-; exit "${PIPESTATUS[0]}"; } {out}>&1 | strip
  return "${PIPESTATUS[0]}"
}

filtered() {
  stripped "$@"
  local rc=$?
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
  config)
    # Unchanged: what surely prints no setting. A subcommand or an action that writes, a color, the help, the editor on a
    # terminal (off one, an editor such as `cat` prints the file), and NAME VALUE after nothing but a scope or a file
    # (NAME has a dot, which no subcommand has). Every other form is taken as a read, an option this file doesn't know
    # included: filtering a write changes nothing it does.
    case ${rest[0]-} in
      set|unset|rename-section|remove-section) exec "$REAL" "$@" ;;
      edit) [ -t 1 ] && exec "$REAL" "$@" ;;
    esac
    j=0 scoped=1
    while [ $j -lt ${#rest[@]} ]; do
      case ${rest[$j]} in
        --add|--unset|--unset-all|--replace-all|--rename-section|--remove-section|--get-color|--get-colorbool|-h|--help)
          exec "$REAL" "$@" ;;
        --edit|-e) [ -t 1 ] && exec "$REAL" "$@"; scoped=0 j=$((j + 1)) ;;
        --local|--global|--system|--worktree) j=$((j + 1)) ;;
        -f|--file|--blob) j=$((j + 2)) ;;
        -t|--type|--default) scoped=0 j=$((j + 2)) ;;
        -*) scoped=0 j=$((j + 1)) ;;
        *) break ;;
      esac
    done
    [ $scoped = 1 ] && [ $((${#rest[@]} - j)) -ge 2 ] && [[ ${rest[$j]} == *.* ]] && exec "$REAL" "$@"
    stripped "$@"; exit ;;
  var)
    for a in "${rest[@]}"; do [ "$a" = -l ] && { stripped "$@"; exit; }; done ;;
  push)
    # The branch rule's pre-push hook (pre_push.py) for every push, after the caller's own -c and through --no-verify; it
    # then runs the hook the push would have run otherwise (AGENT_GUARD_CHAIN_HOOKS: none under --no-verify).
    hooks=${AGENT_GUARD_GIT_HOOKS:-/usr/local/lib/agent-guard/hooks}
    if [ -x "$hooks/pre-push" ]; then
      chain=$("$REAL" "${pre[@]}" rev-parse --path-format=absolute --git-path hooks 2>/dev/null)
      # git before 2.31 echoes the option it doesn't know and prints a relative path: then no hook is chained, and the push
      # still runs, judged
      [[ $chain == /* && $chain != *$'\n'* ]] || chain=
      [ -n "$chain" ] && [ "$chain" -ef "$hooks" ] && chain=
      kept=()
      for a in "${rest[@]}"; do
        if [ "$a" = --no-verify ]; then chain=; else kept+=("$a"); fi
      done
      AGENT_GUARD_CHAIN_HOOKS=$chain exec "$REAL" "${pre[@]}" -c "core.hooksPath=$hooks" push "${kept[@]}"
    fi ;;
esac
exec "$REAL" "$@"
