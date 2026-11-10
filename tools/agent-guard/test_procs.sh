#!/usr/bin/env bash
# Self-test of procs.py with dummy values only: as the lister, and as the `ps` and `pgrep` it stands in front of. install.sh
# runs it against the installed file and takes the two wrappers off PATH if it fails. Usage: test_procs.sh [PROCS] (default:
# beside this file). The process it looks at is this script itself, started again with a dummy secret in each spelling among
# its arguments, so there is nothing to wait for and no other process's line is read out.
set -u
mark=selftest-$$                # short: a longer word with digits in it is itself masked as an id
if [ "${2:-}" != held ]; then
  G=${1:-$(cd "$(dirname "$0")" && pwd)/procs.py}
  hex=0123456789abcdef
  exec bash "$0" "$G" held "$mark" --auth-token not-to-be-copied --api-key=not-to-be-copied "http://u:pw@127.0.0.1:9/x" \
    Bearer not-to-be-copied "$hex$hex" plain-word
fi
G=$1 mark=$3
t=$(mktemp -d) fails=0
trap 'rm -rf "$t"' EXIT
ln -s "$G" "$t/ps" && ln -s "$G" "$t/pgrep"
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fails=$((fails + 1)); fi; }
# what must not come out, and what must: the flags' names, the program and an ordinary word
leaks() { grep -q -e not-to-be-copied -e u:pw -e 0123456789abcdef <<<"$1"; }
kept() { grep -q -e "--auth-token \[REDACTED\] --api-key=\[REDACTED\] http://\[REDACTED\]@127.0.0.1:9/x Bearer \[REDACTED\] \[REDACTED\] plain-word" <<<"$1"; }

raw=$(/usr/bin/ps -o args= -p $$)
check "the real ps shows the dummy values (so the checks below can fail)" 'leaks "$raw"'
out=$(python3 "$G" "$mark")
check "procs PATTERN lists this process masked" 'kept "$out" && ! leaks "$out" && grep -q "^ *$$ " <<<"$out"'
out=$(python3 "$G" -p $$)
check "procs -p PID lists it masked" 'kept "$out" && ! leaks "$out" && [ "$(wc -l <<<"$out")" = 1 ]'
out=$(python3 "$G" "$mark.*not-to-be"; echo "rc=$?")
check "a pattern that matches only the raw line matches nothing" '[ "$out" = "rc=1" ]'
out=$("$t/ps" -o args= -p $$)
check "ps -o args= prints the line masked" 'kept "$out" && ! leaks "$out"'
out=$("$t/ps" -o pid=,comm= -p $$)
check "ps -o pid=,comm= prints what the real one does" '[ "$out" = "$(/usr/bin/ps -o pid=,comm= -p $$)" ]'
out=$("$t/pgrep" -af "$mark")
check "pgrep -af prints the line masked" 'kept "$out" && ! leaks "$out"'
"$t/pgrep" -f "no-such-process-$mark-x" >/dev/null; rc=$?
check "pgrep's exit status passes through" '[ $rc = 1 ]'
out=$("$t/pgrep" -af "no-such-process-$mark-x"; echo "rc=$?")
check "a pgrep that lists doesn't find itself" '[ "$out" = "rc=1" ]'

[ "$fails" = 0 ] && echo "procs: all checks pass" || { echo "procs: $fails check(s) failed"; exit 1; }
