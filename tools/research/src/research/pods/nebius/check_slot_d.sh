#!/bin/bash
# Slot d: node 2's train-check slot on cores 0-47 (top-level, 1 Oct 1:52 AM PDT); compute accounting's CPU verifies yield it.
#   vy-check-slot-d CMD...   CMD under check-d.lock, with slot d's test cache, on 0-47, the way node 1's slots a and b run
# It holds the lock first, then refuses (exit 75) when a timed window in /workspace/pouw/fill/windows is open or starts within
# SLOT_D_NEED_MIN (30) minutes: a check here takes 12-20 min, and it doesn't run into a timed window. A window whose comment has
# the clause `slot-d=runs` (after the `#` or a `;`) doesn't pause it (top-level, 2:18 AM PDT: NCP's and the 8:00 and 9:00 runs
# keep their host threads off 0-47); the words elsewhere in a comment, as in "(no slot-d=runs)", don't count.
set -euo pipefail
exec 9>/workspace/research/locks/check-d.lock
flock 9
python3 - /workspace/pouw/fill/windows "${SLOT_D_NEED_MIN:-30}" << 'EOF' || exit 75
import re
import sys
from datetime import datetime, timedelta, timezone
now, need = datetime.now(timezone.utc), timedelta(minutes=int(sys.argv[2]))
for line in open(sys.argv[1]):
    f = line.split("#")[0].split()
    if len(f) < 2 or re.search(r"[#;]\s*slot-d=runs\b", line):
        continue
    start = datetime.fromisoformat(f[0].replace("Z", "+00:00"))
    if start - need < now < start + timedelta(minutes=int(f[1])):
        sys.exit(f"slot d: the timed window at {f[0]} ({f[1]} min) is open or starts within {sys.argv[2]} min; "
                 "start after it ends")
EOF
mkdir -p /home/research/.cache/verity/tests-slot-d
exec env -u LEASE_DIR LEASE_DEADLINE_FILE=/nonexistent/research-deadline VERITY_TEST_CACHE=/home/research/.cache/verity/tests-slot-d \
  taskset -c 0-47 "$@"
