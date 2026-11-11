#!/usr/bin/env bash
# Self-test of git_guard.sh, offline (127.0.0.1:9 refuses at once; nothing is fetched) and on dummy credentials only: git
# reads no configuration of this machine, only what the test writes. install.sh runs it against the installed wrapper and
# uninstalls the wrapper if it fails. Usage: test_git_guard.sh [GUARD] (default: beside this file).
set -u
G=${1:-$(cd "$(dirname "$0")" && pwd)/git_guard.sh}
REAL=${VERITY_REAL_GIT:-/usr/bin/git}
t=$(mktemp -d) fails=0
trap 'rm -rf "$t"' EXIT
export HOME="$t/home" GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_COUNT=0
unset GIT_CONFIG_GLOBAL GIT_CONFIG_PARAMETERS XDG_CONFIG_HOME
mkdir "$HOME"
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fails=$((fails + 1)); fi; }
g() { bash "$G" "$@"; }
# Dummy credentials in the shapes a Cursor VM holds, each built from parts so that this file holds none: an installation
# token (three parts joined by dots), a basic Authorization credential and a personal access token.
tok=$(printf %s gh s_ 1210556_ dummyheader . dummypayload . dummysignature)
basic="dummy+not/to/be/copied=="
pat=$(printf %s github_ pat_ dummydummydummydummydummy)
secret() { grep -a -q -e dummy -e copied "$@"; }

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
out=$(g -c "url.http://$tok@127.0.0.1:9/.insteadOf=http://example.invalid/" remote get-url b)
check "an installation token as the user name goes, dots and all" '[ "$out" = "http://127.0.0.1:9/x.git" ]'

g remote add n "https://notes@github.com/danielreuter/research-notes.git"; rc=$?
check "a plain user name is allowed and kept" '[ $rc = 0 ] && [ "$(g remote get-url n)" = "https://notes@github.com/danielreuter/research-notes.git" ]'
g remote get-url nosuch 2>/dev/null; rc=$?
check "git's exit status survives the filter" '[ $rc != 0 ]'
check "other commands pass through" '[ "$(g -C "$t/r" rev-parse --is-inside-work-tree)" = true ]'
check "even on a PATH without bash (a test's fake bin)" '[ "$(PATH=/nonexistent "$G" -C "$t/r" rev-parse --is-inside-work-tree)" = true ]'
check "VERITY_GIT_GUARD=off disables it" 'VERITY_GIT_GUARD=off bash "$G" remote add o "http://u:pw@127.0.0.1:9/y.git" && "$REAL" remote remove o'

# `config`: the token in a key, as Cursor's insteadOf rule holds it, and a credential in two values
helper='!f() { echo username=x-access-token; echo "password=$SCRATCH"; }; f'
"$REAL" config --global "url.https://x-access-token:$tok@github.com/.insteadOf" "https://github.com/"
"$REAL" config --global "http.https://github.com/.extraheader" "AUTHORIZATION: basic $basic"
"$REAL" config --global scratch.pat "$pat"
"$REAL" config --global scratch.tok "$tok"
"$REAL" config --global credential.helper "$helper"
"$REAL" config scratch.port "http://host:8080"
"$REAL" config --add scratch.multi "http://host:8080"
"$REAL" config --add scratch.multi "me@example.invalid"
records() { tr -cd '\0\n' <"$1" | wc -c; }
forms="--list
-l
--global --list
--file $HOME/.gitconfig --list
--list --show-origin
--list --show-scope
--list --name-only
--list -z
--list --null
-lz
--name-only -z --list
--show-origin --show-scope -z --list
--get-regexp .
--get-regexp credential\.|url\..*insteadof
--name-only --get-regexp url
--show-origin --show-scope --get-regexp http
-z --get-regexp .
--get http.https://github.com/.extraheader
--get-all http.https://github.com/.extraheader
http.https://github.com/.extraheader
--show-origin --get scratch.pat
-z scratch.pat
--get-urlmatch http https://github.com/x
--get-urlmatch http.extraheader https://github.com/x"
subcommands=""      # git 2.46 on: `config list`, `get`, `set`, `unset`, `edit`
"$REAL" config list >/dev/null 2>&1 && subcommands="
list
list --show-origin --name-only
get scratch.pat
get --all --regexp --show-names .
get --url=https://github.com/x http.extraheader"
while read -r -a f; do
  "$REAL" config "${f[@]}" >"$t/raw" 2>/dev/null; raw=$?
  g config "${f[@]}" >"$t/out" 2>"$t/err"; rc=$?
  check "config ${f[*]} prints no credential, in as many records" \
    'secret "$t/raw" && ! secret "$t/out" "$t/err" && [ $rc = $raw ] && [ "$(records "$t/out")" = "$(records "$t/raw")" ]'
done <<<"$forms$subcommands"
out=$(GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0="url.https://x-access-token:$tok@github.com/.insteadOf" GIT_CONFIG_VALUE_0=https://github.com/ \
  g config --show-origin --get-regexp insteadof)
check "nor a key the environment sets" \
  '! secret <<<"$out" && grep -q "^command line:[[:space:]]url.https://github.com/.insteadof https://github.com/$" <<<"$out"'
g var -l >"$t/out" 2>/dev/null
check "var -l lists the configuration without one" '! secret "$t/out" && grep -q "^scratch.pat=\[REDACTED\]$" "$t/out"'
check "a key keeps its URL" '[ "$(g config --get-regexp insteadof)" = "url.https://github.com/.insteadof https://github.com/" ]'
check "an Authorization header keeps its scheme word" '[ "$(g config http.https://github.com/.extraheader)" = "AUTHORIZATION: basic [REDACTED]" ]'
check "a token standing alone is replaced" '[ "$(g config --get scratch.pat) $(g config scratch.tok)" = "[REDACTED] [REDACTED]" ]'
check "a credential helper reads as it is" '[ "$(g config --get credential.helper)" = "$helper" ]'
check "and on a PATH without sed" '[ "$(PATH=/nonexistent "$G" config --get credential.helper)" = "$helper" ]'
check "-z records that hold no credential come out byte for byte" \
  'cmp -s <(g config -z --get-all scratch.multi) <("$REAL" config -z --get-all scratch.multi)'
check "VERITY_GIT_GUARD=off prints what git prints" '[ "$(VERITY_GIT_GUARD=off bash "$G" config scratch.pat)" = "$pat" ]'
g config --get nosuch.key >"$t/out" 2>"$t/err"; rc=$?
check "a read that finds nothing exits as git does" '[ $rc = 1 ] && [ ! -s "$t/out" ] && [ ! -s "$t/err" ]'
"$REAL" config --type=int --get scratch.pat >/dev/null 2>&1; raw=$?
g config --type=int --get scratch.pat >"$t/out" 2>"$t/err"; rc=$?
check "git's error stays on stderr, without the value it quotes" \
  '[ $rc = $raw ] && [ $rc != 0 ] && [ ! -s "$t/out" ] && [ -s "$t/err" ] && ! secret "$t/err"'

g config scratch.a 1 && g config --add scratch.a 2 && g config --global scratch.g "$pat"; rc=$?
check "NAME VALUE and --add write, a token as it is" \
  '[ $rc = 0 ] && [ "$("$REAL" config --get-all scratch.a | tr "\n" " ")" = "1 2 " ] && [ "$("$REAL" config --global scratch.g)" = "$pat" ]'
g config --replace-all scratch.a 3 && g config --type=bool scratch.b yes; rc=$?
check "--replace-all writes, and so does a write that goes through the filter" \
  '[ $rc = 0 ] && [ "$("$REAL" config --get-all scratch.a)" = 3 ] && [ "$("$REAL" config scratch.b)" = true ]'
g config --unset scratch.b && g config --unset-all scratch.multi && g config --rename-section scratch moved; rc=$?
check "--unset, --unset-all and --rename-section write" \
  '[ $rc = 0 ] && [ "$("$REAL" config --local --get-regexp "^(scratch|moved)\." | tr "\n" " ")" = "moved.port http://host:8080 moved.a 3 " ]'
GIT_EDITOR='sed -i s/8080/8081/' g config --edit; rc=$?
check "--edit off a terminal still edits" '[ $rc = 0 ] && [ "$("$REAL" config moved.port)" = http://host:8081 ]'
out=$(GIT_EDITOR=cat g config --global --edit 2>&1)
check "and an editor that prints the file shows no credential" '! secret <<<"$out" && grep -q "extraheader = AUTHORIZATION: basic \[REDACTED\]" <<<"$out"'
g config --remove-section moved; rc=$?
check "--remove-section writes" '[ $rc = 0 ] && ! "$REAL" config --local --get-regexp "^moved\." >/dev/null'
g config --unset nosuch.key 2>/dev/null; rc=$?
check "a write that fails exits as git does" '[ $rc = 5 ]'
if [ -n "$subcommands" ]; then
  g config set scratch.n 1; rc=$?
  check "set writes" '[ $rc = 0 ] && [ "$("$REAL" config get scratch.n)" = 1 ]'
  g config unset scratch.n; rc=$?
  check "unset writes" '[ $rc = 0 ] && ! "$REAL" config get scratch.n >/dev/null'
  check "edit off a terminal shows no credential either" '! GIT_EDITOR=cat g config edit --global 2>&1 | secret'
fi

[ $fails = 0 ] && echo "git guard: all checks pass" || { echo "git guard: $fails check(s) failed"; exit 1; }
