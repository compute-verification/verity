#!/bin/bash
# launch-auto.sh WORKTREE JOB.sh BASE [MEM_GB] [MAX_MIN]: the job, with BASE written in for its @BASE@, on the node
# `research run --on auto` picks, in the shipped tree (--cwd source). The job commits nothing, so it names no identity.
# Prints the launch's lines: where the job went, why, and when it should start.
wt=$1; job=$2; base=$3; mem=${4:-32}; max=${5:-240}
cd "$wt" || exit 1
[ -z "$(git status --short)" ] || { echo "the worktree has changes that are not committed: the run ships the commit"; exit 2; }
git merge-base --is-ancestor "$base" HEAD || { echo "HEAD does not hold $base"; exit 2; }
body="$(sed "s/@BASE@/$base/g" "$job")"
echo "launching from $(git rev-parse HEAD), base $base"
timeout 170 research run --on auto --cwd source --mem-gb "$mem" --max-min "$max" --project verity --source . -- bash -lc "$body" 2>&1 | cut -c1-260 | tail -12
