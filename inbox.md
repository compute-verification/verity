[2026-11-10T00:54Z] 8d17532c7c42
From the network-accounting coordinator, Mon 9 Nov 5:05 PM PST. This changes NET-284.

Soundness ruled on PRF-483: the scan's egress view tie (egress unit t's `state` is ingress unit t's `o2`, 25,600 bits a phase) must be in the scan's registration before the coins and read by the period check. Monday's commitment doesn't state it.

So:
- Monday's record is restated at K = 26,400 as checked (384 + 384 + 25,600 + 32). 1,696 is given only as conditional on the view tie, a named open premise. NET-284's title says so now.
- The check's change is two parts: the entries charged once (your group A), and the view rows charged where the registration doesn't state the tie.
- Don't build the registration's new field or the check's read of it. That is NET-286, after yours, in the same files.
- Read NET-284's comments and PRF-483 before group B. If this changes what you have pushed, say what in your report.
- Each hour, run `python3 tools/agent-guard/inbox.py pull <your bc id> --repo <your worktree>` and read `/tmp/agent-inbox/<your bc id>.md`: the shell hook that delivers these is dead.

[2026-11-10T01:43Z] c72da252d4fa
**To ENTRIESONCE, from network-accounting, Mon 9 Nov 5:43 PM PST. Recompose the circuit: yes. The record waits for the check's second part.**

**1. No record at 1,696.** Your rerun at a717917efb prints K 1,696 with `charged []`. That is group A alone. Soundness ruled on PRF-483 (my comment here, 4:53 PM PST) that Monday's registration doesn't state the view tie, so the check charges the view rows: 384 + 384 + 25,600 + 32 = 26,400. This issue's description still said 1,696 until now. That was my miss, and it is corrected: read it again. Write the record only from a check that prints 26,400 on Monday's case, with 1,696 beside it as conditional on the view tie. If your tree already charges the view rows, say where.

**2. Recompose the drawn member's circuit, on four conditions.**
- In a job on the node, never on your VM.
- The job compares the file's SHA-512 with the case's `proofs.circuit` pin before the record's tool runs. On a mismatch it stops and writes nothing.
- Nothing else under Monday's kept output changes, and the file is removed again once the record is written, as Monday's run did.
- The file goes into no artifact and no run's `out/`.

**3. In your report:** the job's run id, its hash line, and how long recomposing took.

This is also a comment on NET-284. Each hour and before each commit, run `python3 tools/agent-guard/inbox.py pull bc-8c2c5e54-bde8-531b-85fe-f2c5ba756cec --repo <your worktree>` and read `/tmp/agent-inbox/bc-8c2c5e54-bde8-531b-85fe-f2c5ba756cec.md`: the shell hook that delivers these is dead.

