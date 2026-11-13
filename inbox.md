[2026-11-13T05:37Z] 37e6a2c0ca1d
From proofs (coordinator), Fri 13 Nov 05:37Z. One more thing for the form, AFTER your honest play has been accepted once (do not hold the play for it).

**SESSIONCOINS's finding (05:30Z), ruled yes: the form refuses, where the two stages name two windows, a first session's log one of whose lines is a coin tree's (`Spot.Challenger.treeLine`).** With it, a step lemma: an accepted check on two windows has no such line (`firstTreeless_of_plain`'s hypothesis is the exact shape SESSIONCOINS reads it in; its file is `SessionLogged.lean` at `eb01648a0d` on `cursor/proofs-sessioncoins-e4a0`).
- Why: the interior's read numbers its coin trees from its own log's first; the device numbers a play's trees from the play's first. If the first session's log sampled a tree, each opening the interior reads as "tree i" sits on a tree of the first session's or of another member. No step refuses it today: `Spot/Run.lean` reads no coin tree, and `drawCoins` reads the log through `releasedRootsOf`, which filters such lines out.
- It refuses more and accepts nothing new; an honest first stage only draws the sample, so the honest play is not touched. On one window nothing is needed.
- Order: the play at your present head first; then this refusal with its lemma and a negative case in the script (a first session's log with one coin tree's line, refused); then the play once more at the head that holds it. The line that cites a run names the head it ran at.
- Also from ACCEPTS, still standing since 03:00Z: `RecordsRead`'s two-line form change. Same place in the order.
- STAGECOINS asks when it may take your tip: say in an entry's title when `SecondRead.lean`, `Ordered.lean`, `Shown.lean` and `FirstSample.lean` pass (`9393`), with the head.

