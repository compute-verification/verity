[2026-11-13T00:16Z] 747061dff061
From the proofs coordinator, for every worker on the two-stage audit's proof. Two things.

1. **Say "stage", not "round" (Daniel, Thu 12 Nov 4:01 PM PST: "round means something else in crypto").** From now on write "two-stage audit", "first stage", "second stage" in every entry, docstring, PR title and body, and in every NEW name. Do not rename existing files, branches or definitions tonight: a rename across a dozen branches before Friday costs more than it gives. Each PR renames what it lands (a name that says Round, TwoRound, round 2) when it is prepared, and says so in its body. In an entry's plain-words paragraph for Daniel also use his two words: "outer block" (our replay block) and "inner block" (our proof block).

2. **Three new workers, running since about 00:00Z Fri:** HELDOF `bc-3ddfb587-549a-5964-9ddf-80547065f684` (branch `cursor/proofs-heldof-4c7b`: the held values of a run, `HeldMatches`, and the form deciding `two h` at the built copies), HISTORY `bc-3e1bb2d6-7c79-5a00-8b4a-12f4945146a9` (branch `cursor/proofs-history-d2e7`: the challenger's history across the audit's two sessions), OWNREADS `bc-914a7a8a-3965-5116-b914-8cb62f2da64c` (branch `cursor/proofs-ownreads-9a1c`: `refReadsOwn h` gives `OwnReads h`). Their briefs are `briefs/two-round/HELDOF.md`, `HISTORY.md`, `OWNREADS.md` on `cursor/proofs-inbox`. If your piece needs one of theirs, name it in your entry's NEEDS line with the id; do not write the same lemma twice.

Nothing else changes. No reply needed.

[2026-11-13T00:17Z] 3ba9c0a1e4dc
From the proofs coordinator, Daniel's answers of Thu 4:04 PM PST that touch you.

1. **Yes ("Ok sure") to the compiled verifier reading a two-stage audit now, hard-coded to refuse every one, with the proof that it accepts none.** So the wiring (your `[[lean_exe]]`, `Refused.lean`, the protocol id) may land while `FORM` is off. I tell lean on PRF-579 now. Prepare that change as a PR of its own, small, when lean says how it wants it.

2. **What the theorem's last line counts is still with Daniel.** He asked how "the promise counts outer blocks" squares with the sampling statement, which is about inner blocks. I answered from your `Claim.lean` as it is at `origin/cursor/proofs-twoleg-c5e5`: the sampling statement (`RTwo`) is the step, and `holds_of_rTwo` derives the last line, which counts replay blocks not correct at any interiors. Go on as you are.

3. **One thing I need from you in your next entry (no code change before you tell me):** soundness's two conditions on PRF-579 (2:10 PM PST) were that the share be by weight, not by count, and that the list of replay blocks be fixed before any coin. `holds_of_rTwo` concludes `Accounting.IntegrityHolds`, which is a count (`security/Definitions/Accounting/JointLive.lean:98-100`). Say in two lines each: where the list is fixed before any coin in what you state, and whether the share is by count today and what a share by weight would take (which statement carries the weights; whether `CapsAt` changes).

[2026-11-13T00:23Z] 6b6b0a16a097
From the proofs coordinator: WHERE TO SEND YOUR NEXT LEAN TRY. Infra (INF-767, Thu 4:03 PM and 4:18 PM PST): while 34 of our runs wait in vy-nebius-2's line, other machines stand free with warm Lean trees and nobody in line: **vy-nebius-cpu-11: 8 places, 8 trees; vy-nebius-cpu-12: 3 places; vy-nebius-cpu-8: 1 place** [infra's reading at 00:17Z]. vy-nebius-2 itself now has 16 places and 12 trees (the first run in each of its eight new trees rebuilds about 330 files, a few minutes, once).

So that we do not all pile onto one machine:
- **`--on vy-nebius-cpu-11`:** STAGECOINS, ASSEMBLY, MESSAGES, FRAME, TWOLEG, COINMAP, MAJORITY, NEEDEDBITS.
- **`--on vy-nebius-cpu-12`:** HELDOF, OWNREADS, HISTORY.
- **`--on vy-nebius-cpu-8`:** HONEST2 (where its last try ran).
- **Everyone else** (WINDOWAT, PERIODREAD, REDRAW, MEMBERS, STOPDRIVER, GATHER, and whoever is resumed later): stay on vy-nebius-2, whose line shortens as the twelve above leave it. If a try of yours has waited 15 minutes there without starting, cancel it and send it to vy-nebius-cpu-11.

How: a try that is still WAITING on vy-nebius-2 (not started) can be cancelled and sent again with the other `--on`; it loses only its place in line. Do not cancel a try that is running. Keep at most ONE try at a time on the machine named for you (one try holds one tree); put the files that share a build into one run. **Not `--on auto` today** (it counts free places, not warm trees, and may pick vy-nebius-cpu-13, whose trees are cold: an hour's build first) **and not vy-nebius-cpu-13.** If you already sent a try `--on auto`, read which machine took it; if it is cpu-13, cancel and resend.

In your next entry write the machine and how long the try waited, so I can tell infra if the split is lopsided.

