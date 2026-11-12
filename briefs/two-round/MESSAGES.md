# MESSAGES: the frame's prover, and its messages fixed by their digests (TWOLEG's piece F)

Read `COMMON.md` beside this file first.

**Yours, in one sentence.** Build the frame's prover `Fr.P`, which reads the strategy's messages off the tape (round 1 the interior record, the statements' round, then each session's record), and prove `MessagesFixed` (a message that differs from the one its digest was recorded under is a SHA-512 collision, since the device records each digest before the coins after it) and `CallsLe`.

**Reuse.** SESSIONOF's `Sampled/SessionOf.lean` on `cursor/proofs-sessionof-5c1e` (`sessionOf`, the session built from the checks) for one session's record; read SESSIONOF's newest entry for what of it has compiled. COINMAP holds the frame's `place`: your prover reads the tape where its map writes it, so read COINMAP's entries and agree the frame's fields through yours.

**What you wait on.** `SampledTwo/Reading.lean` (TWOLEG's, owed Fri 01:30Z) and COINMAP's frame. Start with `callsLe` and round 1's message, which need neither.

**Where your piece is written out.** In `inbox/proofs-coordinator.md` on `cursor/verity-proofs-84b5`, the entry headed `## TWOLEG, Thu 12 Nov 22:27Z` holds your section (F, the messages and the prover) and the block "Common to every section", which is part of your brief: the base, where new files go, the imports, the rules. Read both whole, then TWOLEG's `## TWOLEG, Thu 12 Nov 22:15Z` (why the pieces are cut this way, and the footprint reading) and `## TWOLEG, Thu 12 Nov 21:48Z` (the map). The `Prop` you prove is stated in `security/Proofs/Properties/Verity/SampledTwo/Projection.lean` on the base; read it there, not from the entry. TWOLEG's own marks stand: what it tags `[derived]` or `[guess]` is an argument, not a fact, and nothing of the leg had compiled when it wrote.

**Base.** `origin/cursor/proofs-twoleg-c5e5` (at `08e3d2b2d7a1` when this was written, Thu 22:33Z). Merge it into your branch when it moves; never rebase. **Your branch:** `cursor/proofs-messages-f55d` (TWOLEG's proposed name; if it exists already, add a letter), made from that base.

**Whose files.** Yours: the new files your section names, under `security/Proofs/Properties/Verity/SampledTwo/`. Each new module is a root on a line of its own in `security/Proofs/lakefile.toml`. Not yours: anything under `security/Proofs/Properties/Verity/Sampled/` and `Sampled.lean` (soundness holds them and has two workers in them tonight), `Recursive/` (soundness's), TWOLEG's `Leg.lean`, `Projection.lean`, `Claim.lean`, `Reading.lean`, and the form `VStarRecursive/SampledTwo.lean`. If a `Prop` in `Projection.lean` cannot be proved as stated, write the smallest change to it under `NEEDS` for TWOLEG, with a counterexample, and go on with the changed text as a hypothesis.

**Who adopts it.** Soundness said (Thu 21:58Z) that proofs drafts the projection and the majority in new files and soundness adopts them, reading each draft before anything is built on it. So your first entry names your branch, and each entry says which files are ready for that reading.

**What Daniel ruled, and what it means for you.** The two-round accept stays withheld until the two-round form sits under the one soundness theorem (Thu 1:44 PM PST). Your piece is on that path. Nothing you write turns the accept on, wires the form into `Sampled.lean` or touches a lock; if you find you would, stop and write it under `STOP RULE`.

**A known hole you do not close.** Soundness found (Thu 22:00Z) that at the deployed theorems a strategy which puts an earlier period's genuine challenger signature in its evidence makes its own error ⊤, because the device's history holds only one play's signatures (`Sampled/Device.lean:468-475`, `Properties/Challenger.lean:344-347`). Soundness closes it with the key pins on Tue 17 Nov. Where your statement meets it, say so in the docstring and in "not established"; do not work around it.

**Builds.** Lines for a Lean slot on cpu-8, cpu-10 and cpu-12 are 20 to 85 minutes tonight. cpu-12 takes no new start from 07:00Z tonight to Fri 18:00Z. A second node for this lane, vy-nebius-2, opens about 00:15Z; send nothing there until your mailbox says a try has passed on it. So: launch a try as soon as a file is written, several files to a run, and keep writing while it waits.

**Owed.** 20 minutes: first entry (your agent id, your branch, your plan in five lines). By two hours after launch: `Fr.P`, `messagesFixed` and `callsLe` as signatures, elaborating. By 07:00Z Fri: `callsLe` and round 1's message in a run. By 15:00Z Fri: your word on the rest by Sat 02:00Z.
