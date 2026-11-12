# SESSIONCOINS: each second-stage session's coins are its own tree's (TWOLEG's piece E)

Read `COMMON.md` beside this file first.

**Yours, in one sentence.** Prove `coinsRead_sessions` and `stmtReadsNone`: where the two-round form's check accepts, each proof session's coins are its own member tree's, the trees open in the log's order, and each tree closes after its evidence's SHA-512; and the statement round reads no coin.

**Reuse, by import.** The form computes the second stage's verdicts with one-stage v2's member form on the interior's audit (`OneStageZk.membersRun`). The lemmas about it are on `main`: `security/Proofs/Properties/Verity/Sampled/MemberForm/Order.lean` (`opens_in_order`, `closing_read`, `evidence_before`) and `MemberForm/Evidence.lean` (`evidenceBytes_inj`). Those files are soundness's and FIRSTACCEPT works beside them tonight: read, import, never edit. The round of the run that a session's coins are is `slotOf` in `SpotFlock.V`.

**What you wait on.** `SampledTwo/Reading.lean` (TWOLEG's, owed Fri 01:30Z). Until it is on the base, state both over parameters standing for `formReading` and `FormAccepts` and prove what is about `membersRun` alone.

**Where your piece is written out.** In `inbox/proofs-coordinator.md` on `cursor/verity-proofs-84b5`, the entry headed `## TWOLEG, Thu 12 Nov 22:27Z` holds your section (E, the sessions' coins) and the block "Common to every section", which is part of your brief: the base, where new files go, the imports, the rules. Read both whole, then TWOLEG's `## TWOLEG, Thu 12 Nov 22:15Z` (why the pieces are cut this way, and the footprint reading) and `## TWOLEG, Thu 12 Nov 21:48Z` (the map). The `Prop` you prove is stated in `security/Proofs/Properties/Verity/SampledTwo/Projection.lean` on the base; read it there, not from the entry. TWOLEG's own marks stand: what it tags `[derived]` or `[guess]` is an argument, not a fact, and nothing of the leg had compiled when it wrote.

**Base.** `origin/cursor/proofs-twoleg-c5e5` (at `08e3d2b2d7a1` when this was written, Thu 22:33Z). Merge it into your branch when it moves; never rebase. **Your branch:** `cursor/proofs-sessioncoins-e4a0` (TWOLEG's proposed name; if it exists already, add a letter), made from that base.

**Whose files.** Yours: the new files your section names, under `security/Proofs/Properties/Verity/SampledTwo/`. Each new module is a root on a line of its own in `security/Proofs/lakefile.toml`. Not yours: anything under `security/Proofs/Properties/Verity/Sampled/` and `Sampled.lean` (soundness holds them and has two workers in them tonight), `Recursive/` (soundness's), TWOLEG's `Leg.lean`, `Projection.lean`, `Claim.lean`, `Reading.lean`, and the form `VStarRecursive/SampledTwo.lean`. If a `Prop` in `Projection.lean` cannot be proved as stated, write the smallest change to it under `NEEDS` for TWOLEG, with a counterexample, and go on with the changed text as a hypothesis.

**Who adopts it.** Soundness said (Thu 21:58Z) that proofs drafts the projection and the majority in new files and soundness adopts them, reading each draft before anything is built on it. So your first entry names your branch, and each entry says which files are ready for that reading.

**What Daniel ruled, and what it means for you.** The two-round accept stays withheld until the two-round form sits under the one soundness theorem (Thu 1:44 PM PST). Your piece is on that path. Nothing you write turns the accept on, wires the form into `Sampled.lean` or touches a lock; if you find you would, stop and write it under `STOP RULE`.

**A known hole you do not close.** Soundness found (Thu 22:00Z) that at the deployed theorems a strategy which puts an earlier period's genuine challenger signature in its evidence makes its own error ⊤, because the device's history holds only one play's signatures (`Sampled/Device.lean:468-475`, `Properties/Challenger.lean:344-347`). Soundness closes it with the key pins on Tue 17 Nov. Where your statement meets it, say so in the docstring and in "not established"; do not work around it.

**Builds.** Lines for a Lean slot on cpu-8, cpu-10 and cpu-12 are 20 to 85 minutes tonight. cpu-12 takes no new start from 07:00Z tonight to Fri 18:00Z. A second node for this lane, vy-nebius-2, opens about 00:15Z; send nothing there until your mailbox says a try has passed on it. So: launch a try as soon as a file is written, several files to a run, and keep writing while it waits.

**Owed.** 20 minutes: first entry (your agent id, your branch, your plan in five lines). By two hours after launch: both statements as signatures, elaborating. By 07:00Z Fri: one session's coins in a run. By 15:00Z Fri: your word on all sessions by Sat 02:00Z.
