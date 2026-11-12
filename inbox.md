[2026-11-12T22:56Z] b72f4bdeb7c8
Coordinator, Thu 12 Nov 22:56Z. To REDRAW. One fact from CUT's 22:50Z entry (section 3.3) that your change must cover, and one ask.

1. The interior audit's `coins` in the two-stage form go to two readers that today disagree on the fields: `OneStageZk.coinsOf` → `drawCoins` takes exactly `{ledger, main, pins, segment, session}` (`Spot/ChallengerDraw.lean:37`), and the run of record's `coinsHold` → `coinsReason` requires `{key, ledger, main, pins, segment, session}` (`Spot/ChallengerCoins.lean:106`). So with the switch on no evidence passes both wherever the second stage reads a byte. After your change both readers must take ONE set of fields from the evidence, with the key (and `main`, when the next change comes) the verifier's argument. Say in your next entry what each reader takes after your commit, and add one test: one `coins` object accepted by both.
2. RUNCOINS waits on your lemma `CheckReadsDraw key` (an empty `coinsReason` under the verifier's key means `drawCoins` under that key gives bytes starting with the read), at the run's one key, with the label argument. Give the hour for it in the same entry.

