#!/usr/bin/env bash
# `lean_ready.sh DIR EXE CMD...`: CMD once a piece's background build of the Lean program EXE, whose lock is DIR/EXE.lock,
# has ended; where that build or one before it failed, the build's own error (DIR/failed) and exit 7.
set -u
D=${1:?usage: lean_ready.sh DIR EXE CMD...} X=${2:?EXE}
shift 2
flock -s "$D/$X.lock" true
[ -e "$D/$X.ok" ] || { cat "$D/failed" >&2; exit 7; }
exec "$@"
