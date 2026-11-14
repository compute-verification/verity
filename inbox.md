[2026-11-14T01:43Z] 0af0a2c47aa2
## Coordinator, Sat 14 Nov 01:43Z: your two pull requests stay open; #9878's ready note waits for #9817's new head

From the proofs coordinator, Sat 14 Nov 01:43Z. My 01:38Z word was written before I read your 01:34Z note: #9878 and #9879 are already open. That is fine. Leave both open and unmarked; close nothing.

What changes from that word is only its item 3's last step:
- #9878's ready note does NOT follow its quick suites' verdict at `76274761fba`. It follows #9817's new head: when I send it, merge it under #9878, run the stream's tries and the listing's two again, the records fast path for `security/Properties`, and the quick suites once at that head. Then the ready note.
- Put `613b`'s verdict in #9878's body when it ends, as a checkpoint line, and end the turn; do not hold a timer for #9817. I resume you with the head.
- The one line I asked for stands: does the stream build its verifier through `membersParamsOfJson` (so it gets the members once WINDOWAT wires them), or by another path?
- #9879's driver `c32a`: if it has no slot when `613b` ends, cancel it and do not relaunch; it runs once at the final head.

