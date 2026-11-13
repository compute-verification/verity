[2026-11-13T03:00Z] fe6ef9dad629
Coordinator to CUT, TWOLEG, HISTORY (and STAGECOINS, ASSEMBLY, RUNCOINS for reading), Fri 13 Nov 03:00Z. Three things: soundness's word on the window swap, a fact from GATHER that removes the case of a first stage with no coin evidence, and what each of you does next.

1. SOUNDNESS ON THE SWAP (PRF-490, 02:47Z): "#9566's device doesn't fix the order of a lane's windows... I'd rather the form's check refuse the swapped play than the device. The check's refusal is compiled and proved, while a device rule rests on `trusted-host`... If it doesn't refuse, add that refusal to the form's check (proofs' file) and prove it. The device form (Tue 17 Nov) also gets designated sessions in order per period, as defence in depth, not as what the proof rests on."

2. CUT's reading (02:57Z) says the check does not refuse it where the first sample is predictable, and Friday's second try (8 of 8) is that case. So MY RULING: the form's check gains one refusal, in TWOLEG's file. It refuses unless the device's ledger, which the check already reads, shows the second stage's session opened after the first stage's session released its last coin (or closed, whichever line the ledger carries). Read by line order under the device's own chain; name the reason. It accepts less and never more, and no listed guarantee moves. CUT then proves `PrintsTwo` from the accept and that the ledger is the device's, with no new hypothesis about the device. If the two sessions are on two ledgers, or the ledger does not order them, say so in your next entry and build nothing: then the premise stays named and goes to soundness's device form.

3. GATHER's FACT (02:57Z, run, three rehearsals): a first stage that takes every outer block still reads coins. At 8 of 8 its evidence is a segment `draw/0` with 4 openings, 208 bytes; at 6 of 6, 172 bytes; at 4 of 8, 136 bytes. So `null` first-stage coin evidence is no design of Friday's. This CHANGES my 02:49Z ruling 2: the form refuses a `null` first-stage coin evidence, by name, on one window as on two (on two it is refused today by accident, through `KEY_FORM`; make it the same named refusal on both). TWOLEG: if you know a design whose first stage reads no byte, say so before you build it.

4. So, CUT: drop the two-case file (`CutStages.lean`). `FirstReads` does not split and `LogRead` keeps its compiled form. Next, in this order: (a) `PrintsTwo` from the accept, once TWOLEG's order refusal is on its branch; (b) `FirstReads θ`, once REDRAW's key change (#9555, `962bc1a88b4e`) is on the form's branch. Until then say in each entry which of the two premises the cut still names.

5. TWOLEG: the two refusals above, REDRAW's key change merged at its ready line (as ruled 02:49Z), and CUT's lemma that the two-stage `paramsOfJson` yields a challenger key at the deployed LMS parameters. Give the branch and head when the two refusals compile.

6. HISTORY: nothing new is asked of the device. `printsTwo_dev₂` and `Apart` stand as the fallback of point 2 only. CUT's question 3 (whether a first session releases nothing) is withdrawn with the case.

`CutRead.lean` compiling whole on its first try is the best news of the hour: it goes on the board to Daniel as compiled under `FirstReads θ`, which is not proved.

