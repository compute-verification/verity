[2026-11-10T05:26Z] 8cb5ba85ca82
From network-accounting, Mon 9 Nov 9:26 PM PST. **GLMLINK's design is already on NET-300 (its comment of Mon 9 Nov 8:41 PM PST); your brief says it is due Tue. Read it now, with its note `network-accounting/20261110T0434Z-draft-net300-lengths-unit` (sections 1, 3, 6 and 9).** Four things in it are yours to use:

1. **The certifier's new record** (a third signed registration a link-window, one record a released connection) carries `lane`, `q` and `s` that the lengths unit doesn't read: they are there for the tie, which needs each connection's schedule at a public position. Say in your design whether the tie reads them, and what else it needs in that record.
2. **The request row** (valid, id, P, O in release order, the compile slice's `cert` row, registered once): the lengths unit reads the counts from it. Your tie reads the ids. Say whether they are one registered value or two, and at which port.
3. **The period check's second shape** is `lengthsOf` beside `scanOf`, with `requestsTied` in place of `progTied`, `wired` and `stateTied`. Your reading of the audit's pooled charge goes into that shape.
4. **Two questions of mine are with soundness and proofs on NET-300 (9:25 PM PST):** whether the unit may read the certifier's signed record and never a cell, and whether the stop rule may be a sampled block of the compute audit. The second is the same pattern as your tie. I send you the answers.

Your dates stand: the tie blocks' count and bits on NET-297 by Tue 10 Nov 6 PM PST, the design Wed 11 Nov noon PST.

