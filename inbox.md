[2026-11-10T00:54Z] 8d17532c7c42
From the network-accounting coordinator, Mon 9 Nov 5:05 PM PST. This changes NET-284.

Soundness ruled on PRF-483: the scan's egress view tie (egress unit t's `state` is ingress unit t's `o2`, 25,600 bits a phase) must be in the scan's registration before the coins and read by the period check. Monday's commitment doesn't state it.

So:
- Monday's record is restated at K = 26,400 as checked (384 + 384 + 25,600 + 32). 1,696 is given only as conditional on the view tie, a named open premise. NET-284's title says so now.
- The check's change is two parts: the entries charged once (your group A), and the view rows charged where the registration doesn't state the tie.
- Don't build the registration's new field or the check's read of it. That is NET-286, after yours, in the same files.
- Read NET-284's comments and PRF-483 before group B. If this changes what you have pushed, say what in your report.
- Each hour, run `python3 tools/agent-guard/inbox.py pull <your bc id> --repo <your worktree>` and read `/tmp/agent-inbox/<your bc id>.md`: the shell hook that delivers these is dead.

