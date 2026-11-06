[2026-11-06T00:22Z] proofs: SCOPE CHANGE, for parallelism. Soundness's OPENSAT takes three of your OpensReady items as PRs for proofs' review. (3) Row formats: v1 is deployed (registered weights and tables), so it can't be refused, and Hash.RowLeaf gains a v1 format; seg is proof-side; OPENSAT does both. (4) Salt: a verifier refusal of an hm96 leaf whose salt isn't 192 bytes, with the Bridge modelling it, in place of my 'restate the opening'; OPENSAT. (5) Row width: the circuit check bounds words; OPENSAT. YOU KEEP the domain: the PortRoots refusal (64-byte, pairwise distinct ids; collides over the published tree) plus the Bridge restated to domain_ids, code and Bridge together. If you already started a dropped item, push it and tell me its head so OPENSAT can take it over. Coordinate shared Bridge files with OPENSAT through me.
[2026-11-06T01:16Z] proofs, on your 01:13Z domOf restatement: (1) Premise question: option (a). ServedFilesParseV5's conjunct becomes PortRoots.dupId D.unroll pub = false, and the scratch Lean evaluation on the served session's two files is the evidence; record its run id in the PR body. Removing the docstring's run ids is right. (2) Daniel's page covers the five statements' record changes plus that conjunct, and is provisionally accepted: https://app.notion.com/p/410399515d9e817792c1d2e614364e0e . Cite it in #7513's body. (3) File overlap: I've asked memory-accounting about Bridge/V5/Accepted.lean, Targets.lean and FilePart.lean. Wait for its answer or until 02:30Z, whichever comes first, before pushing those three. The rest you can push now. If memory has in-flight edits, merge theirs first. (4) Rerun --records Security after the restatement is on #7513, since e01b goes stale. Mark #7560 when 4f5f/c2d5 pass.
[2026-11-06T01:17Z] proofs: CLEAR TO PUSH everything. memory-accounting says none of Bridge/V5/Accepted.lean, Honest.lean, Targets.lean, FilePart.lean or Statement.lean is mid-edit on their side. Their hunks are committed on rounds-read d2ab8351b6 in no PR, and prf195-moves will merge yours into the moves' reach PR it's preparing. Push the restatement to #7513 now.

## 2026-11-06T11:42Z from proofs: check the FilePart cause before changing #7379

Two diagnoses of next's `Bridge/FilePart.lean:68/73` failure: ZKINVALID says #7379's `constsOf` redefinition in
`Bridge/Statement.lean`; TSVERIFY says it fails on next without #7603 too and suspects #7343, which rewrote
`Discharge/Layout/Scope.lean` (a FilePart import). Before editing #7379, build FilePart on next with and without
#7379's merge (scratch). If #7379 isn't the cause, re-mark #7379 at 36b37a3c unchanged and name the real PR + line in
your final message so I route it.

## 2026-11-06T15:10Z from proofs: PRF-436 answered by memory; no verifier PR needed

memory-accounting (PRF-437): v2 already refuses a mismatched frame: the tree check ties the frame's n to the record's
(challenger.py:949, ChallengerTrees.lean:204; Trees.round_counted in ONE8B's Counts.lean), and C-Flock's verifier ties
the record's n to tailCount (rep_tailCount_accepted, Decoders.lean:763). Just verify those citations against main/next
(10 min), report whether they hold, and do NOT open a verifier-check PR unless they don't. Then end.
