[2026-11-10T00:54Z] a2282a350c73
From the network-accounting coordinator, Mon 9 Nov 5:05 PM PST. This changes NET-285's statements.

Measured on Monday's kept case (widths on NET-285): the scan's state is tied in two places. Tie 1 is unit t's `sin` = unit t-1's `o1` (4,512,608 bits a phase), registered and read by the check (`stateRead`). Tie 2 is egress unit t's `state` = ingress unit t's `o2` (25,600 bits), neither registered nor read.

Soundness ruled on PRF-483 that tie 2 must be registered before the coins and read by the period check. For your three statements:
- Tie 2 is its own premise, apart from tie 1, discharged by the check's read once NET-286 is built.
- Both ties enter through main's checked inputs (`CheckedUnique`), per PRF-480's ruling 3.
- The statement requires every scan unit proved. It doesn't rely on Monday's draw of 64 of 64.
- With tie 2 absent the count charges the 25,600 view bits: Monday's figure is 26,400, and 1,696 is the figure with both ties.
- Read NET-285's comments and PRF-483 before you finish the statements.
- Each hour, run `python3 tools/agent-guard/inbox.py pull <your bc id> --repo <your worktree>` and read `/tmp/agent-inbox/<your bc id>.md`: the shell hook that delivers these is dead.

