# HISTORY: the device's history for an audit of two challenger sessions

Read `COMMON.md` beside this file first, and its last section ("what moved Thu 22:49Z to 23:00Z") before anything in this brief: where they differ, that section wins.

**Yours, in one sentence.** Define the challenger device's history for one two-round audit, which is two challenger sessions (the first stage's, and the interior's), and prove at it the two facts CUT's piece takes as hypotheses: the device printed the log (`PrintsTwo`) and the device signed what the evidence shows (`SignsPlay`).

**Why it is a piece of its own.** The one-stage history is `security/Proofs/Properties/Verity/Sampled/Device.lean` (1,218 lines, about 70 declarations: `dev`, `printsLog_dev`, `holds_dev`, `binds_dev`, `signer`). It holds one session's run. The two-round form runs two sessions, each with its own window. COINMAP (which holds the coin map, piece B) sized the two-session history at 400 to 700 lines and 15 to 25 theorems, as large as the coin map itself, and proposed it go to a new worker (its entry `## COINMAP, Thu 12 Nov 23:03Z`, section 4, which is part of your brief: read it whole). The coordinator agreed at 23:22Z.

**The shape COINMAP proposes (yours to change, with a reason).**
- `dev₂ S₁ S₂ split o`: the first session's run printed from `log₁ := (split o).1` under `S₁ o`'s key on window `w₁`, the interior's from `log₂` under `S₂ o` on `w₂`. Its ledger opens each run's id on its own window only, so the two run ids differ. On one window it is the one-stage `dev` at the whole log.
- Reuse the one-stage file by import at `{o with log := logᵢ}` for each session. Never edit it: it is soundness's.
- The one parameter you share with COINMAP: which releases are the first session's (`split`). Agree it through your entries; COINMAP's locator (`Locate`, in `SampledTwo/CoinMap.lean` on `origin/cursor/proofs-coinmap-8b0c`) already says which round each release continues.

**Two problems COINMAP found, both yours to settle with CUT.**
1. `PrintsOn dev w log` speaks of every run the ledger opens on `w`. A history whose `runs` is empty meets it with nothing to show, and the one-stage `dev` drops the run where `printLog` fails (`Device.lean:468-475`). So `PrintsTwo` at your history must say the history holds the two runs, or CUT's `BindsOfPrints` proves nothing. Say in your first statements entry which of the two you state.
2. Printing the interior's releases alone from `PrintState.init` fails if they open a tree whose `Hello` is among the first session's releases: the tree indices are then off. Either the split puts each tree's `Hello` and its openings in one session (check this against the form's sessions in `VStarRecursive/SampledTwo.lean`), or the second run prints from the state after `log₁`. Read the form and say which holds.

**Which challenger the form runs under.** Today each of the driver's two commands makes its own key and ledger under `--challenger local` (GATHER, 23:16Z, section 2: `benchmarks/one_stage/a5_coins.py:750-764`). REDRAW is changing the driver to one local challenger across both stages, and the leg's texts say one challenger key for the run (COMMON's last section). State your history at one key and two windows, with the two keys as parameters `S₁`, `S₂` that may be equal, so both readings are instances.

**A known hole you do not close.** A history may hold earlier runs and ledger lines only on other windows; on the play's own two windows it holds exactly this play's runs. That is the step soundness replaces with the key pins on Tue 17 Nov (row A40 of the list on NET-390). Say so in the docstring and under "not established"; do not work around it.

**Base.** `origin/cursor/proofs-twoleg-c5e5` at its newest head (`b04e0a9fd970` at 23:19Z). Merge `origin/cursor/proofs-cut-9323` for `PrintsTwo` and `SignsPlay` as CUT states them in `SampledTwo/Cut.lean`; if CUT's file does not compile yet, take its two `Prop`s as written and say so. **Your branch:** `cursor/proofs-history-d2e7` (add a letter if it exists), made from that base. Merge the bases when they move; never rebase.

**Whose files.** Yours: `security/Proofs/Properties/Verity/SampledTwo/Device.lean` and any file beside it your piece needs, each a root on a line of its own in `security/Proofs/lakefile.toml`. Not yours: anything under `Sampled/` and `Sampled.lean` (soundness's), CUT's `Cut.lean`, COINMAP's `CoinMap.lean`, TWOLEG's `Leg.lean`, `Projection.lean`, `Claim.lean`, `Reading.lean`, and the form. If a `Prop` of CUT's cannot be met as stated, write the smallest change under `NEEDS` for CUT, with a counterexample, and go on with the changed text as a hypothesis.

**Builds.** vy-nebius-2 is open and is this lane's alone until Sat 02:00Z: `research run --on vy-nebius-2 --pool proofs --as proofs --mem-gb 12 --timeout 3600 --cwd source -- bash -lc 'python3 tools/verity/lean/lean_changed.py --try FILE'` (or `verity lean try FILE --on vy-nebius-2 --pool proofs --as proofs`). Send nothing to cpu-8, cpu-10, cpu-12 or cpu-13. Launch a try as soon as a file is written and keep writing while it waits.

**Owed.** 20 minutes: first entry (your agent id, your branch, your plan in five lines, your answer to problem 2 if the form already shows it). By 01:30Z Fri: `dev₂` and both statements as signatures, in a try. By 07:00Z Fri: `PrintsTwo` at the history, in a run. By 15:00Z Fri: your word on `SignsPlay` by Sat 02:00Z.
