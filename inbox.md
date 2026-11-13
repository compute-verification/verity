[2026-11-13T14:05Z] 2972d23fcffe
WALKER, Fri 13 Nov, 6:05 AM PST.

**s-t20 reads with the rule on: 5,387.4 s with a highest resident memory of 58.8 GiB, against 12,460.4 s and 137.6 GiB under #9676 alone. But the rule does not remove all of the reading's memory, and my earlier wording said it did: the reading still grows by about 46 GiB at s-t20.**

- **s-t20's reading** (RAN: r20261113-114851-b24e on cpu-5; #9558 plus #9676 plus #9730; the reading before #9558's pass):
  - Build: 912.6 s, peak 57.5 GiB.
  - Reading: 5,387.4 s. Resident memory was 12.9 GiB at its first heartbeat, 58.8 GiB at its highest, and 56.8 GiB when the pass started.
  - Before (RAN: 54ec, same machine, earlier): build 1,115.5 s; reading 12,460.4 s; 15.6 GiB at its first heartbeat, 137.6 GiB at its highest.
  - INFERRED: the machine was less loaded for b24e (its build was 18% faster), so the time ratio overstates the rule a little. The memory doesn't depend on load.
- **Correction.** At 3:27 AM I wrote that with the rule on "the reading adds nothing measurable above the build's own peak", and #9730's body said the rule "removes the reader's memory growth". Both compared process peaks, and the build's peak hides the reading. RAN: resident memory falls when the build ends and the reading grows it back. The reading's growth above its start:

  | shape | before (#9676) | after (rule on) |
  |---|---|---|
  | s(2,256,512) | about 8 GiB (5.0 to 12.99) | between 2.8 and 5.3 GiB (4.9 to at least 7.7, under 10.15) |
  | tiny(2,1024,64) | about 7 GiB (3.5 to 10.46) | under 0.9 GiB (2.2 to at least 2.8, under 3.09) |
  | s-t20 | about 122 GiB | about 46 GiB |

  So the rule removes about 60% of the reading's growth at s-t20, not all of it. #9730's body now says this: the lead, the before/after table with each reading's start and highest, and the s-t20 row.
- **What is left standing.** INFERRED: the 46 GiB is the Reader-level `_leaves` cache and the rest of what fixes 1 and 2 were aimed at; nothing here traces it. The trace you asked 3 to give is a traced reading of #9730's head at s(2,256,512). I have not launched it: you kept the machines for chain 2. Say go and I launch it.
- **`Work` equal, wider than I reported.** RAN: at s(2,256,512) and tiny(2,1024,64) the `Work` JSON has one sha256 across rule on (twice), rule off and #9676's own reader. The body's table says so and now lists the runs.
- **The pass** (RAN): b24e is in #9558's pass. Heartbeat at 7,945 s: RSS 64.0 GiB, 247 products passed, nonfinite 12, against a stop at 144 GiB. 54ec stopped at 753 products after growing 22.8 GiB in the pass. INFERRED: this one has room to finish, which would give s-t20's first claims and `Work`. I'll send its end.
- No merge of `main`. The three heads are unchanged.

