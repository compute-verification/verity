#!/usr/bin/env bash
# Self-test of git_guard.sh, offline (127.0.0.1:9 refuses at once; nothing is fetched). install.sh runs it against the
# installed wrapper and uninstalls the wrapper if it fails. Usage: test_git_guard.sh [GUARD] (default: beside this file).
set -u
G=${1:-$(cd "$(dirname "$0")" && pwd)/git_guard.sh}
REAL=${VERITY_REAL_GIT:-/usr/bin/git}
t=$(mktemp -d) fails=0
trap 'rm -rf "$t"' EXIT
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fails=$((fails + 1)); fi; }
g() { bash "$G" "$@"; }

"$REAL" init -q "$t/r" && cd "$t/r" || exit 1
g remote add a "http://u:pw@127.0.0.1:9/x.git" 2>"$t/err"; rc=$?
check "remote add with a password is refused" '[ $rc = 2 ] && ! "$REAL" config --get remote.a.url >/dev/null'
check "the refusal never prints the password" '! grep -q pw "$t/err"'
g clone -q "http://ghp_FAKEnotarealtoken000000000000000000@127.0.0.1:9/x.git" "$t/c" 2>"$t/err"; rc=$?
check "clone with a token as user name is refused" '[ $rc = 2 ] && [ ! -e "$t/c" ] && ! grep -q ghp_ "$t/err"'
g remote set-url a "http://u:pw@127.0.0.1:9/x.git" 2>/dev/null; rc=$?
check "set-url with a password is refused" '[ $rc = 2 ]'

"$REAL" remote add s "http://u:pw@127.0.0.1:9/x.git"
out=$(g remote -v 2>"$t/err")
check "remote -v removes a stored credential" '! grep -q pw <<<"$out" && grep -q "s	http://127.0.0.1:9/x.git (fetch)" <<<"$out"'
check "and names the remote that holds it" 'grep -q "remote.s.url" "$t/err" && ! grep -q pw "$t/err"'
"$REAL" remote remove s

"$REAL" remote add b "http://example.invalid/x.git"
out=$(g -c "url.http://x-access-token:tok@127.0.0.1:9/.insteadOf=http://example.invalid/" remote get-url b 2>"$t/err")
check "get-url removes a credential that insteadOf put in" '[ "$out" = "http://127.0.0.1:9/x.git" ] && [ ! -s "$t/err" ]'
out=$(g -c "url.http://x-access-token:tok@127.0.0.1:9/.insteadOf=http://example.invalid/" ls-remote --get-url b)
check "ls-remote --get-url too" '[ "$out" = "http://127.0.0.1:9/x.git" ]'

g remote add n "https://notes@github.com/danielreuter/research-notes.git"; rc=$?
check "a plain user name is allowed and kept" '[ $rc = 0 ] && [ "$(g remote get-url n)" = "https://notes@github.com/danielreuter/research-notes.git" ]'
g remote get-url nosuch 2>/dev/null; rc=$?
check "git's exit status survives the filter" '[ $rc != 0 ]'
check "other commands pass through" '[ "$(g -C "$t/r" rev-parse --is-inside-work-tree)" = true ]'
check "even on a PATH without bash (a test's fake bin)" '[ "$(PATH=/nonexistent "$G" -C "$t/r" rev-parse --is-inside-work-tree)" = true ]'
check "VERITY_GIT_GUARD=off disables it" 'VERITY_GIT_GUARD=off bash "$G" remote add o "http://u:pw@127.0.0.1:9/y.git" && "$REAL" remote remove o'

[ $fails = 0 ] && echo "git guard: all checks pass" || { echo "git guard: $fails check(s) failed"; exit 1; }
