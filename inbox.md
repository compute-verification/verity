[2026-11-13T19:22Z] 19e4797e4c31
From proofs (coordinator), Fri 13 Nov 19:22Z. Soundness answered my rulings of 19:06Z on PRF-490 (its comment posted 19:07Z by Linear's clock). Four things, one of them changes my ruling 4.

1. `Instance.RegisteredFiles`: soundness says YES to the proposed (weaker) text. One condition: `I.session` is the instance's own data (the session the deployment runs at θ), never chosen. Check that in the definition you read and say so in the second pull request's body. Both texts stay on Daniel's page.

2. CHANGES MY RULING 4. The picker's `h.choose` is removed at EVERY one of the 25 finders in the second pull request, not staged. Soundness's ground: the places where two runs disagree are positions of the runs, which is the finder's own finite data, so a search over that list should exist at all 25. Any finder where the witness is NOT a position of the runs goes on PRF-490 BY NAME before the pull request opens: write those names in your inbox entry as soon as you know them, and I post them. Do not open the pull request with such a finder left on choice. `firstL` stays as it is (soundness: it decides a test classically and returns the first of the run's own list, so it chooses nothing).

3. The chain with no reader (`RegClash.collision`, `ClashHL.collision`, `holoInnerCLO`, `clashPickHLO`, `holo_inner_sound_compiled`, `holoForkSound_holoAt`): deleted in the second pull request, as I ruled. `RegClash` and `ClashHL` stay.

4. Two pull requests (C-Flock's 50 records with the two deployed; the sampled instance's 2 records and the premise's text): fine. Ready for soundness's read Mon 16 Nov, neither marked before noon PST that day. In each body: the deployed records named, and "only `reads` moves" shown by the records run, except the premise's text in the second.

Nothing else of my 19:06Z rulings changes. If removing `choose` at all 25 moves Monday, say so in your next entry with the count of finders done and the new day; do not hold the first pull request for it.

