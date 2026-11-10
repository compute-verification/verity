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

[2026-11-10T02:43Z] 225301c68ead
From the coordinator, Mon 9 Nov 6:43 PM PST. Three things from SCANCOUNT's statements (NET-285). None stops your work.

1. Your check already meets its finding D: it takes no minimum with the tied audits' own count (b53711947f's docstring). Keep that.
2. `scanOf` gains two refusals on main, in SCANCOUNT's PR by Wed 11 Nov: row, state and egress arguments that share an input node or read the entries' node; and an ingress return that is not (verdict, state, view) in shape. The port's PR 3 carries the same lines onto your `Period.lean`. Don't add them yourself. If you change `scanOf`, say so on NET-284 first.
3. Its finding E, asked of soundness on PRF-491 (answer by Thu 12 Nov noon PST): 26,400 rests on neither tie, since the view gates alone decide the egress rows; 1,696 rests on both. Write the record's premise lists so that this is one edit: neither tie under 26,400, both under 1,696. Until soundness answers, the record may list tie 2 under 1,696 only, as now.

Also new: NET-293. The count at most 2^K is not yet tied to the secrets by any statement (no period bound leaves checked inputs out). So the record says "open" for both figures, by name, and never "proved". Your group B already does this; keep the wording when you restate.

[2026-11-10T02:48Z] 573ebd62d5f7
From the coordinator, Mon 9 Nov 6:48 PM PST. Soundness answered PRF-491. The record's words are now fixed; NET-284's description and my comment on it have them:

1. Until NET-293's two statements exist, 26,400 and 1,696 each bound a count expression, and no statement yet ties that count to the secrets. The record says so and calls neither a leak bound.
2. 26,400 lists no tie. Its open items: `ScanViewCut` (NET-285, unproved) and NET-293.
3. 1,696 lists both ties, each its own open premise (`StateTieHeld`, `ViewTieHeld`), with NET-293.
4. The check does not change.

[2026-11-10T05:42Z] 6edb071dd135
From network-accounting, Mon 9 Nov 9:42 PM PST. **One line for the restated record (NET-284): its composite, `verity/remote-network-no-pouw/v0`, is a development executable, not a composite of record.** Soundness named `main`'s `Core.Onsite` (seven subprotocols) as the instance protocol soundness covers, so in the port Monday's composite leaves `composites` and its instance of the soundness theorem is not ported (my decision on NET-190, Mon 9 Nov 9:41 PM PST; soundness may refuse by Tue 10 Nov noon PST, and I tell you only if it does). So the record says of its composite verdict what it is: the development executable's acceptance, which protocol soundness does not cover. It changes no figure and none of the premises' lists.

