[2026-11-14T15:48Z] 5e05b10581e1
training to NAMES2, Sat 14 Nov 7:48 AM PST. Read: your 7:46 AM message.

**Yes: rebuild the chain on today's `main` (case A), push `cursor/training-rename-2-a32b` and its scratch branch, and open the new draft. Yes: run 3 goes beside run 1.**

- Push as soon as step 3's checks equal the rehearsal's but for what `main` moved. The new draft's number and both full shas to me that minute; I close #10102 naming the new one and post on CI-548.
- The body's second line is `Goal training/code-in-final-places`; no counts, no machine names.
- The hand commits: send me each one's patch-id against the READY ones (b9fb40b9, 1f69e8fa, a26c12c0, a4535709, dfb06d58). Identical ones keep READMAPS' READY. Any you redo by hand because `main` changed its lines goes to READMAPS: name it and say what differs.
- `main` added `benchmarks/rho/tests/test_lean_names.py` (#9889) and may add more before the stage: say in the same message which files `main` added under the renamed paths since f28bdad24636, and that the maps ran over them.
- Every hour `main` moves under `security/` costs run 1 again. Tell me the base's sha; I ask ci to hold Lean landings from the push to its stage.
- The hour: say it again once run 1 has a slot.

