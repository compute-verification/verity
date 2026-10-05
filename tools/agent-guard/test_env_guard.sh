#!/usr/bin/env bash
# Self-test of env_guard.py with dummy values only, in an environment of its own. install.sh runs it against the installed
# guard. Usage: test_env_guard.sh [GUARD] (default: beside this file).
set -u
G=${1:-$(cd "$(dirname "$0")" && pwd)/env_guard.py}
t=$(mktemp -d) fails=0
trap 'rm -rf "$t"' EXIT
ln -s "$G" "$t/printenv"
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fails=$((fails + 1)); fi; }
blob=$(printf 'dummy-first-line\nNOTANAME_LEAK\nnot-to-be-copied')
# run CMD... in an environment holding only these variables (LANG keeps python3 from adding LC_CTYPE)
in_env() { /usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 PLAIN_VAR=hello DUMMY_API_TOKEN=not-to-be-copied DUMMY_BLOB="$blob" "$@"; }
names=$'DUMMY_API_TOKEN\nDUMMY_BLOB\nLANG\nPATH\nPLAIN_VAR'

out=$(in_env python3 "$G" | cut -d= -f1)
check "env | cut -d= -f1 prints names only" '[ "$out" = "$names" ]'
out=$(in_env python3 "$G" -0 | cut -z -d= -f1 | tr '\0' '\n')
check "env -0 | cut -z -d= -f1 lists the names" '[ "$out" = "$names" ]'
out=$(in_env python3 "$G")
check "env redacts a KEY|TOKEN name and a multi-line value" \
  'grep -qx "DUMMY_API_TOKEN=\[REDACTED\]" <<<"$out" && grep -qx "DUMMY_BLOB=\[REDACTED\]" <<<"$out" && ! grep -q not-to-be-copied <<<"$out"'
check "and shows an ordinary value" 'grep -qx "PLAIN_VAR=hello" <<<"$out"'
out=$(in_env python3 "$G" -u PLAIN_VAR; in_env python3 "$G" --unset=PLAIN_VAR)
check "env -u NAME and --unset=NAME print the rest redacted" \
  '! grep -q -e NOTANAME_LEAK -e not-to-be-copied -e PLAIN_VAR <<<"$out" && grep -q "DUMMY_BLOB=\[REDACTED\]" <<<"$out"'
out=$(in_env python3 "$G" -v)
check "an option the guard doesn't know still prints redacted" \
  '! grep -q -e NOTANAME_LEAK -e not-to-be-copied <<<"$out" && grep -q "DUMMY_BLOB=\[REDACTED\]" <<<"$out"'
out=$(in_env python3 "$G" EXTRA=1 | grep EXTRA)
check "env NAME=VALUE without a command prints the assignment" '[ "$out" = "EXTRA=1" ]'
out=$(in_env python3 "$G" EXTRA=1 sh -c 'echo "$EXTRA"')
check "env NAME=VALUE CMD runs CMD" '[ "$out" = 1 ]'
out=$(in_env python3 "$G" -i ONLY=1 /usr/bin/printenv ONLY)
check "env -i NAME=VALUE CMD runs CMD in that environment" '[ "$out" = 1 ]'
out=$(in_env python3 "$t/printenv" | cut -d= -f1)
check "printenv | cut -d= -f1 prints names only" '[ "$out" = "$names" ]'
out=$(in_env python3 "$t/printenv" -0 | cut -z -d= -f1 | tr '\0' '\n')
check "printenv -0 keeps NUL separators" '[ "$out" = "$names" ]'
out=$(in_env python3 "$t/printenv" DUMMY_API_TOKEN PLAIN_VAR DUMMY_BLOB)
check "printenv NAME redacts secrets and shows the rest" '[ "$out" = $'"'"'[REDACTED]\nhello\n[REDACTED]'"'"' ]'
out=$(in_env bash -c 'compgen -e' | grep -vx -e PWD -e SHLVL -e OLDPWD -e _ | sort)
check "compgen -e lists names without the guard" '[ "$out" = "$names" ]'

[ "$fails" = 0 ] && echo "env guard: all checks pass" || { echo "env guard: $fails check(s) failed"; exit 1; }
