[2026-11-14T16:14Z] 87672b3f3224
NAMES2, Sat 14 Nov 8:14 AM PST.

**The rebuilt chain is pushed and its draft is #10164: chain head `74933509ee16733dd8ef6ad56ec023cc4a82b9a0`, scratch head `24736b8edf6f7ca06e80a5615acba7eadbf2ff05`, base `a116c5614312258aab06a25cf1bb7838f8ea282f`; I launch runs 1 and 3 next.**

- Branches: `cursor/training-rename-2-a32b` (the chain) and `cursor/training-rename-moves-2-a32b` (the chain plus `lean-moves.json`, "EXPERIMENT: …"). Before the push, origin had the old ones unmoved: `cursor/training-rename-a32b` at `86227c4e…` and `cursor/training-rename-moves-a32b` at `c0791fca…`. RAN
- Draft #10164 is linked to TRN-25. Its second line is `Goal training/code-in-final-places`. The body has no counts, no run ids and no machine names. RAN, READ
- Step 3 equals the rehearsal's except where `main` moved. RAN:
  - the script exited rc=0, with 12 moves certified and 5 ordinary maps that change nothing when run again;
  - `lean-moves.json` has 719 entries, byte-identical to the old chain's;
  - globs: 31 STALE;
  - the stragglers equal the old chain's, apart from hashes;
  - `left.py`: UNSORTED 0, after one new rule for the README cure's line;
  - resolve issues: theta 2, inference_only 31;
  - stale pins left: 3; `import_order` is unchanged;
  - `tools/move/tests`: 264 passed;
  - the root four: 72 passed. The one more than 71 is `main`'s new `test_a_name_that_already_says_the_row_s_term_is_not_a_site_of_its_glob`.
- Patch-ids against the READY commits (`git patch-id --verbatim`). RAN
  - All five are identical, so READMAPS' READY holds for each:

    | new commit | READY | patch-id |
    |---|---|---|
    | `2b7c959d0fc9` | b9fb40b9 | 21ce1b669ebc |
    | `9ed147043930` | 1f69e8fa | 15d356fe41f3 |
    | `b4df2d234eeb` | a26c12c0 | 6a06ef3cfe8f |
    | `daf952e003b0` | a4535709 | bf47a8de2698 |
    | `139d1ba7081a` | dfb06d58 | 5dbf7eaf5dad |

  - The other hand commits, NO_VERDICTS and the pin are identical to the old chain's.
  - Only two commits differ, and both are map runs, not hand commits:
    - The calculator's names (`cf4975f73646`): the same lines change, plus `main`'s new test file, and the hunk context moved with `main`'s `lean.py`.
    - The Python names (`3c6695271231`): identical outside `main`'s new test file.
- Files `main` added or changed under the renamed paths since `f28bdad24636`. RAN, READ
  - The two are `benchmarks/rho/rho/lean.py` (changed) and `benchmarks/rho/tests/test_lean_names.py` (added).
  - The maps ran over both: `test_lean_names.py` is in the calculator's move, its names map and its Python names map (`rho.lean.read` became `slowdown.lean.read`, and so on).
  - In the rebuilt tree both sit under `benchmarks/slowdown/`, and neither holds rho. The one Nci-like word left is `nci/arguments.py`, a path in a directory that keeps its name.
- ci's reader's README item. READ, RAN
  - `benchmarks/nci/README.md` line 23 is misleading on the rebuilt tree: it lists `bracket.py`, the module, as an older name.
  - What `bracket.py` actually keeps is the stored key `"rho"` and its `efficiency`.
  - No later map or hand commit cured it. I cured it in a hand commit of its own: `b285f3e7e464`, "benchmarks/nci/README.md: the older names its code keeps, bracket.py's among them". It sits after the `nci` commit, with NO_VERDICTS and the pin replayed on top unchanged.
  - New text: "The code and its outputs keep older names (`GLOSSARY.md`, Older names): `bracket.py`'s stored key `"rho"` and its `efficiency`, and `tiers.py`'s `efficiency` and `residual`."
  - This commit goes to READMAPS. It is also in the script as the `board_readme` group.
- Events: your 8:06 AM model broadcast, which I follow. I see nothing else addressed to @training. READ
- The hour: I'll restate it once run 1 has a slot. INFERRED

