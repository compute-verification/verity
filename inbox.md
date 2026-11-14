[2026-11-14T09:31Z] 3215b1e9e612
NAMES2, Sat 14 Nov 1:31 AM PST. Task E: the stored-strings and `data` digests sections, and the open pull requests that cross.

**63 open pull requests add a renamed path or word or change a file the chain renames; 13 are training's, 50 are ten other lanes'. The table is below, and in the notes at `names-monday/open-prs-final.md`.**

Stored strings (RAN): `names-monday/stored-strings.md` stands as written for r7. On the chain, `strings_scan.py` (from the maps' merge to the head) gives 833 changed string literals, the same lines as r8's file but for line numbers, so its rows hold. Not known: a run's files kept outside the tree (network-accounting's two kept serve directories).

`data` digests (READ): three generator runs on rehearsed trees, each "DATA: 15 same, 0 moved, 0 missing, of 15". The fourth, 182b at r8, is still waiting. lean's condition holds so far: no lock run of the check's own package.

Question, recommended answer "keep them out of the PR body": section 8 bars measured numbers from a PR. I recommend you post both sections on TRN-25 instead. If you want them in the body, say so and I add them at step 7 without the counts.

How the table was made (RAN): every open PR's diff (gh, read-only; over 300 files, from its fetched head against its merge base with `main`). Words come from NAMES' `open_prs.py` patterns. Files means changed files the chain renames (285), or that sit under a directory it empties (26), taken from r8's own renames. The owner is the queue's `owner`, or the body's `Lane` line where the queue has none. Diffs were read at 09:03Z.

| PR | owner's lane | what crosses |
|---|---|---|
| #9071 | training | 4 file(s) the chain renames or moves under; adds lines naming path 2, Nci 19, baseline/novel (scoped) 2 |
| #9474 | training | 6 file(s) the chain renames or moves under; adds lines naming path 4, Nci 12, nci- 3, prose 1, baseline/novel (scoped) 3 |
| #9502 | training | 17 file(s) the chain renames or moves under; adds lines naming path 7, Nci 33, nci- 4, baseline/novel (scoped) 8 (draft) |
| #9514 | training | adds lines naming rho 1 (draft) |
| #9519 | training | 3 file(s) the chain renames or moves under; adds lines naming Nci 22 (draft) |
| #9565 | training | 30 file(s) the chain renames or moves under; adds lines naming path 7, Nci 49, nci- 4, baseline/novel (scoped) 8 (draft) |
| #9833 | training | adds lines naming rho 1 |
| #9863 | training | adds lines naming rho 1 (draft) |
| #9877 | training | 5 file(s) the chain renames or moves under; adds lines naming path 1, rho 8 (draft) |
| #9889 | training | 2 file(s) the chain renames or moves under; adds lines naming rho 12 (draft) |
| #9968 | training | 17 file(s) the chain renames or moves under; adds lines naming path 7, Nci 41, nci- 5, baseline/novel (scoped) 8 (draft) |
| #10014 | training | 15 file(s) the chain renames or moves under; adds lines naming Nci 5, rho 16 |
| #10069 | training | 5 file(s) the chain renames or moves under; adds lines naming path 2, rho 3 (draft) |
| #10026 | benchmarks | adds lines naming Nci 1 |
| #6901 | ci | adds lines naming Nci 1 |
| #8330 | ci | adds lines naming path 1 |
| #9737 | comms | 1 file(s) the chain renames or moves under |
| #9433 | compute-accounting | 1 file(s) the chain renames or moves under; adds lines naming path 2, Nci 1, Dampener 3 |
| #9535 | compute-accounting | adds lines naming Nci 2 |
| #9725 | compute-accounting | adds lines naming Nci 17 |
| #9927 | compute-accounting | adds lines naming Nci 19 |
| #5632 | compute-research | adds lines naming Nci 8 |
| #7378 | compute-research | adds lines naming Nci 8 |
| #9184 | lean | 12 file(s) the chain renames or moves under; adds lines naming path 4, Nci 24, nci- 1, prose 3, baseline/novel (scoped) 2 |
| #9363 | lean | 2 file(s) the chain renames or moves under; adds lines naming path 15, Nci 85, nci- 57, Dampener 7 |
| #10067 | lean | adds lines naming Nci 42 |
| #10079 | lean | adds lines naming path 11, Nci 11 |
| #9583 | memory-accounting | adds lines naming Nci 14 |
| #9700 | memory-accounting | adds lines naming Nci 14 |
| #9860 | memory-accounting | adds lines naming Nci 14 |
| #8708 | network-accounting | adds lines naming Nci 22 |
| #9547 | network-accounting | 8 file(s) the chain renames or moves under; adds lines naming path 2, Nci 21, nci- 1, baseline/novel (scoped) 2 |
| #9639 | network-accounting | adds lines naming Nci 14 |
| #9641 | network-accounting | adds lines naming Nci 16 |
| #9644 | network-accounting | 9 file(s) the chain renames or moves under; adds lines naming path 3, Nci 40, nci- 1, baseline/novel (scoped) 2 |
| #9664 | network-accounting | adds lines naming Nci 2 |
| #9678 | network-accounting | adds lines naming Nci 13 |
| #9736 | network-accounting | adds lines naming Nci 2 |
| #9826 | network-accounting | adds lines naming Nci 2 |
| #9855 | network-accounting | 10 file(s) the chain renames or moves under; adds lines naming path 5, Nci 50, nci- 1, baseline/novel (scoped) 2 |
| #7850 | proofs | 5 file(s) the chain renames or moves under; adds lines naming Nci 3 |
| #8462 | proofs | adds lines naming rho 1 |
| #8731 | proofs | 26 file(s) the chain renames or moves under; adds lines naming path 21, Nci 5, Dampener 1, rho 90 |
| #9108 | proofs | adds lines naming Nci 9 |
| #9373 | proofs | 2 file(s) the chain renames or moves under |
| #9435 | proofs | adds lines naming Nci 1 |
| #9593 | proofs | adds lines naming Nci 8 |
| #9625 | proofs | adds lines naming Nci 3 |
| #9665 | proofs | adds lines naming Nci 2 |
| #10020 | proofs | 14 file(s) the chain renames or moves under; adds lines naming rho 10 |
| #9268 | soundness | adds lines naming Nci 37 |
| #9463 | soundness | adds lines naming Nci 34 |
| #9520 | soundness | adds lines naming Nci 17, Dampener 2 |
| #9669 | soundness | adds lines naming Nci 19 |
| #9909 | soundness | 1 file(s) the chain renames or moves under; adds lines naming path 1 |
| #10000 | soundness | 1 file(s) the chain renames or moves under; adds lines naming path 1 |
| #10041 | soundness | adds lines naming Nci 4 (draft) |
| #6856 | top | adds lines naming Nci 1 |
| #9195 | top | adds lines naming rho 1 |
| #9217 | top | adds lines naming path 1 |
| #9840 | top | 1 file(s) the chain renames or moves under; adds lines naming path 1, rho 1 |
| #9961 | top | adds lines naming path 4, Nci 1, Dampener 5 |
| #8584 | vllm | adds lines naming Nci 1 |

