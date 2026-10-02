#!/usr/bin/env bash
# The shipped tree (build.sh's OUT, or $1) as a one-commit git repo at $2 (default /tmp/pearl-c4-ship-repo) for
# `research run --source`: every file 100644, author, committer, dates and message pinned, so the same tree gives the same
# commit and the launch path /workspace/research/src/<sha>/run.sh is known before the launch.  Prints the sha.
set -euo pipefail
SRC=$(cd "${1:-/tmp/pearl-c4-ship}" && pwd); DEST=${2:-/tmp/pearl-c4-ship-repo}
(cd "$SRC" && sha256sum -c --quiet MANIFEST.sha256)
rm -rf "$DEST"; mkdir -p "$DEST"
cp -r "$SRC"/. "$DEST"/
cd "$DEST"
find . -type f -exec chmod 644 {} +
git init -q
git add -f -A .
export GIT_AUTHOR_NAME=pouw-pearl-c4 GIT_AUTHOR_EMAIL=pearl-c4@pouw.invalid GIT_AUTHOR_DATE=2026-09-30T00:00:00Z
export GIT_COMMITTER_NAME=pouw-pearl-c4 GIT_COMMITTER_EMAIL=pearl-c4@pouw.invalid GIT_COMMITTER_DATE=2026-09-30T00:00:00Z
git -c commit.gpgsign=false commit -q -m "pearl-c4 sm_120a ship: pearl_c4.cubin, the check fixture and the runner"
git rev-parse HEAD
