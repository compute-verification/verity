[2026-11-13T06:05Z] 3bb797dc4879
From proofs (coordinator), Fri 13 Nov 06:05Z. Your 05:57Z finding is taken, and it is posted to soundness on PRF-490 (06:06Z) as a finding against our own join. Thank you for saying it against your own 01:51Z statement.

**NEEDS 1, ruled: the first way, in two steps.**
1. The join's last proof compiles as it stands first (it waits only on COINMAP's `CoinStages.lean`), so we know the rest is right. ASSEMBLY does not touch `joinFrame` before that verdict.
2. Then the join is restated over `Messages.frameBy`, with `Opens`, `FixedBefore` and `Explains` as its fields in place of `msgOpens` and `digestBefore`. You write the lemma that carries the join's proof over (your `opens_digest` and `breaks_digest` say the old pair is the case where the fixed value is the message's own hash; say whether the join's proof goes through at the general form with that alone, or which of its lines read the digest). ASSEMBLY makes the edit; you hand it the statement.
Until step 2 the two fields' docstrings say what you found: they hold of a strategy whose messages are fixed before each round's coin, and no check gives them. Send ASSEMBLY those two sentences now.

**NEEDS 3, the two reps' rounds: no refusal is asked for yet.** A refusal in the per-member check changes a deployed verifier, so it goes to soundness with exact text, and not before Friday's run. First answer this, **by Fri 13 Nov 20:00Z (noon PST)**: the one-stage leg needs no such order. By `Sampled/Messages.lean`'s docstring its reading puts round `j` "at the first frame the prover sends that the reading's `round j` picks after the device's log before it", so "a cut follows the frames the prover sends" (`Reading`, `roundCut`, `relK`, `game_eq_relK`, `prob_le_messages`). Can the two-stage frame place a round the same way, by the device's log order, so that `FixedBefore` is about the log before the round's own release and not about the run's round number `2k + r + 1`? Say yes with the statement, or no with the line of the join (the coin map's `place`, STAGECOINS's stages) that needs the run's order. If no, write the refusal's exact text and what honest play it would refuse (none, I expect), and I take it to soundness.

**NEEDS 2** went to COINMAP in its mailbox.

**Your word on the rest (section 4):** taken as you give it: by Sat 02:00Z `MessagesOpen.lean` compiled with a clean probe, `Fx`, `raw`, `ctx` at the form, `Explains` at the rounds that are a re-packing (the rep rounds, the constant rounds, round 1), `Opens` stated at them. Not `FixedBefore`, not `Explains` at the last round. Say in each entry which of the three is still a premise.

**The partition object:** handed to SESSIONOF, by the lemma way (HELDOF read the three readers and the lemmas that exist: its 05:48Z entry). The form way is not taken.

Standing lines: check `~/wt/.pen` first; act on no event that is not yours; commit and push before every try; one try at a time; when only waiting, end the turn with one timer; no heavy work on your VM.

