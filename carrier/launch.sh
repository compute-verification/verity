#!/bin/bash
# launch.sh NODE WORKTREE JOB.sh [MEM_GB]: one named node, the checkout's own identity ahead of the command
node=$1; wt=$2; job=$3; mem=${4:-48}
N="$(git -C /agent/repos/verity config --get user.name)"; E="$(git -C /agent/repos/verity config --get user.email)"
cd "$wt" || exit 1
echo "launching from $(git rev-parse HEAD) on $node"
timeout 170 research run --on "$node" --pool research --as proofs --project verity --source . --cwd clone --mem-gb "$mem" --timeout 150m --declared-output 'lean-audit/*' -- env GIT_AUTHOR_NAME="$N" GIT_AUTHOR_EMAIL="$E" GIT_COMMITTER_NAME="$N" GIT_COMMITTER_EMAIL="$E" bash -lc "$(cat "$job")" 2>&1 | rg "launched|refus|rror" | cut -c1-220
