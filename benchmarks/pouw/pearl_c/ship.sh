#!/usr/bin/env bash
# The shipped tree (build.sh's OUT, or $1) as a one-commit git repo at $2 (default /tmp/pearl-c-ship-repo) for
# `research run --source`.  Every file 100644, and the author, committer, dates and message pinned, so the same tree gives the
# same commit and the launch path /workspace/research/src/<sha>/run.sh is known before the launch.  Prints the sha.
set -euo pipefail
SRC=$(cd "${1:-/tmp/pearl-c-ship}" && pwd); DEST=${2:-/tmp/pearl-c-ship-repo}
(cd "$SRC" && sha256sum -c --quiet MANIFEST.sha256)
rm -rf "$DEST"; mkdir -p "$DEST"
cp -r "$SRC"/. "$DEST"/
cd "$DEST"
git init -q
git add -f -A .
git ls-files -z | xargs -0 git update-index --chmod=-x --
export GIT_AUTHOR_NAME=pous-pearl-c GIT_AUTHOR_EMAIL=pearl-c@pous.invalid GIT_AUTHOR_DATE=2026-09-29T00:00:00Z
export GIT_COMMITTER_NAME=pous-pearl-c GIT_COMMITTER_EMAIL=pearl-c@pous.invalid GIT_COMMITTER_DATE=2026-09-29T00:00:00Z
git -c commit.gpgsign=false commit -q -m "pearl-c h100 ship: pearl_c.cubin, the check fixture and the runner"
git rev-parse HEAD
