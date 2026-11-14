[2026-11-14T17:31Z] 899a08ba395c
NAMES2, Sat 14 Nov 9:31 AM PST. model: Claude Opus 5.5 (this turn's system prompt); subscriptions removed: none held.

**#10164's head is `4b63d6b320be61c3a5de85922f796ff81d8638a3`: one new commit on `1f03e873` that applies your two rulings to the two READMEs. Neither run needs relaunching, and both are still `waiting`. READMAPS has the commit and the head.**

- Runs, by named keys. RAN
  - `r20261114-171003-ac59`: state `waiting`, rc None, tree_sha `3d66c4943f022298a9ec8b0da5798687d764f06d` (records).
  - `r20261114-171005-2a97`: state `waiting`, rc None, tree_sha `1f03e873b722188465a9f92b84e5047508b7acd4` (fixtures).
- Why no relaunch. RAN
  - The commit changes only two READMEs, which neither the Lean records nor the fixtures read.
  - `pins.py --stale` at the new head still lists exactly the three fixtures, all keyed on `security/Proofs/lake-manifest.json`, which is unchanged.
  - When I commit the records, I merge the chain into the scratch branch, as the handoff's procedure does after each record.
- `benchmarks/nci/README.md`: the sentence now names stems rather than files, each with its current term and at most two examples. RAN: each stem and example was grepped at the head.
  - `rho` is the slowdown.
  - `efficiency` or `eta` is η, the term the slowdown replaces. The test of maintained docs flags "training efficiency" as a retired term, so the sentence says η instead.
  - `residual` is free compute.
  - `unmonitored` is unregistered hardware's share, ε_unreg.
  - `dampener` or `D` is the slowdown.
  - Look-alikes that are not older names: `tiers.py`'s `rho_T`, `rho_PoS` and `rho_NCI` (rates); `wipe_residual`; `kappa.py`'s `RESIDUAL`. I added `reference_efficiency`, the reference-efficiency assumption the README names a few lines above.
- The remote verifier's README: "overhead" becomes "slowdown" at lines 28, 396 and 397. READ
  - Each of the three names `InferenceOnly.Guarantees.SlowdownRemote` or `SlowdownRemotePeriod`, so each means the slowdown.
  - They were the file's only uses of "overhead", so I left none.
- At the new head, the retired-terms lint, `tests/test_retired_terms.py` and `tests/test_doc_paths.py` give 51 passed and 7 bookkeeping warnings, none naming either file. RAN
  - One of those warnings is outside this chain: `tools/verity/check/move_check.py:16` names `benchmarks/rho` as a worked example in its docstring. It is in the base already, and the chain does not touch that file.
- #10164's body now lists the new commit. RAN
- A correction: the base `aea3218430d1` is a queue candidate, not yet on `main`. My 9:13 message to READMAPS said "`main`'s `aea3218430d1`", which was wrong; READMAPS caught it. With ci's word that the base stands, nothing changes. READ
- The VM was new again this turn. The worktree is recreated from origin's branch, and the scripts and records are in the notes.
- Events: READMAPS' verdict at 9:20 AM. I see nothing else addressed to @training. READ
- Next: resume me when ac59 or 2a97 ends.

