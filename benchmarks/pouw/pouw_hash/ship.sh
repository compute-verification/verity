#!/usr/bin/env bash
# The shipped tree (build.sh's OUT, or $1) as a one-commit git repo at $2 (default /tmp/pouw-hash-ship-repo) for
# `research run --source`.  Every file 100644, and the author, committer, dates and message pinned, so the same tree gives the
# same commit and the launch path /workspace/research/src/<sha>/run.sh is known before the launch.  Prints the sha.
set -euo pipefail
SRC=$(cd "${1:-/tmp/pouw-hash-ship}" && pwd); DEST=${2:-/tmp/pouw-hash-ship-repo}
(cd "$SRC" && sha256sum -c --quiet MANIFEST.sha256)
rm -rf "$DEST"; mkdir -p "$DEST"
cp -r "$SRC"/. "$DEST"/
cd "$DEST"
git init -q
git add -f -A .
git ls-files -z | xargs -0 git update-index --chmod=-x --
export GIT_AUTHOR_NAME=pouw-hash GIT_AUTHOR_EMAIL=pouw-hash@pouw.invalid GIT_AUTHOR_DATE=2026-09-29T00:00:00Z
export GIT_COMMITTER_NAME=pouw-hash GIT_COMMITTER_EMAIL=pouw-hash@pouw.invalid GIT_COMMITTER_DATE=2026-09-29T00:00:00Z
git -c commit.gpgsign=false commit -q -m "pouw-hash ship: pouw_hash_bench, the check vectors and the runner"
git rev-parse HEAD
