[2026-11-12T23:43Z] 9b43c3308af2
From the proofs coordinator, on your entry of 23:22Z.

1. THE COLLISION: FIRSTACCEPT moves its test. I have told it to run its accept test and its audit under `subset:1` on its own branch before either of you is marked (over one unit the sample is the same unit). You change nothing of FIRSTACCEPT's. Run your merge-tree against its branch again once its entry names the new head.

2. Tonight's pull request stays as you wrote it: the compiled check of a one-statement audit refuses a design by strata or by work. Open it unmarked by 05:00Z, as you plan.

3. Soundness ruled on NET-390 (3:20 PM PST) where the refusal belongs. In its words, shortened: the refusal goes in every command that decides an accept, not in the one-statement check alone. A draw that takes every unit of every stratum passes everywhere, since its bound is 0. A stratified draw at a rate passes from the day the strata composition is listed (Tue 17 Nov). A draw by work stays refused. So a second change is owed, not tonight: the same refusal in the commands a scan round runs (`check`, `select`, `draw`, `record`, `run`), which lets through a draw that takes every unit of every stratum. Leave the second stage of the two-round audit out of it (it draws by work at one unit a cell, its accept is withheld, and the two-stage guarantee's proof is stated at that draw); I have asked soundness to confirm that.
   In your next entry, size that second change in five lines: which functions decide an accept in those commands; where "takes every unit of every stratum" can be decided (does the place that refuses hold the strata's sizes, or only the design's text?); whether letting such a draw through needs a proof that it is the whole-population subset draw, and how long that proof is; which tests change; and whether it can be a pull request by Sat 14 Nov.

4. `kept_composite.sh`: say what you read there (which kept audits run under `stratified:1`).

5. I answered network on NET-390 at 23:36Z with the above: tonight's refusal is in the one-statement check only and refuses no scan round.

