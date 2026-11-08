[2026-11-08T06:32Z] c83a44eb9167
BINDTERM 06:36Z: piece 1, #8015, is marked ready at cc8706a8be and admitted; the next sync merges it onto `next`. The quick suites passed at that head: r20261108-060419-d2cc on cpu-9, 57 suites passed, 0 failed. They also passed at 0e35fffcc7 (r20261108-055931-03c2, 54 passed). The preflight merged it onto `next` with one conflict, the two-stage pin, which it rebuilt from the merged tree. You can mark your stacked PRs now. Piece 1's head won't move unless a red names it, and then I'd add a commit on top and tell you first.

