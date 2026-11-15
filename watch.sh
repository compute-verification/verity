#!/bin/bash
# Fetch only, during the hold: what moved on each branch since the pre-rename tip.
cd /home/ubuntu/wt-collect || exit 1
B=${B:-df6a96232ff}
git fetch -q origin architecture cursor/na-architecture-b51f || exit 1
echo "time $(TZ=America/Los_Angeles date +%H:%M:%S) arch $(git rev-parse --short=12 origin/architecture) (+$(git rev-list --count $B..origin/architecture)) collect $(git rev-parse --short=12 origin/cursor/na-architecture-b51f) (+$(git rev-list --count $B..origin/cursor/na-architecture-b51f))"
git log --reverse --format='C %h %s' $B..origin/cursor/na-architecture-b51f | cut -c1-150
