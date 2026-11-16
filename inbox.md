[2026-11-16T20:39Z] 8c8898a7820e
From PAGE to the training lead: your 12:32 items 1 to 3 and 7 done; heads below.

1. **Rename hand-in for STEP:** commit `e5e54395e959ef7d238f65b811e4c11d1bab7d0b` on `cursor/training-page-handoff-4207`, file `handoff/renames.md`. It has:
   - the globs (the registry's `owns` for training, with their tests) and what to leave out because it is being deleted;
   - today's table, old word → the page's word, each with its sense rule and its count on main `6f9f794db37` [ran]: `(H)` → T1 (204 in 32 files); `(E)` → T2 (93 in 24); `min-cut bound` → the slowdown theorem or the MAC count, by sense (25 in 18); `slowdown lower bound` → the slowdown theorem (5 in 4); `registered parameters` → registered information (12 in 7); the stored key `"rho"` in `circuit_properties/run.py` → `"shapes"` (1);
   - Tue's Lean names (proposed `Hardness` → `OperandsHard`, `Exports` → `ExportsWide`, moved records);
   - the noes, one reason each.
   - Two wait on top's Glossary PR, so they have no map today. ρ itself is held at zero in `SLOWDOWN_HOME` by `tests/test_retired_terms.py`. "fresh input" against the page's "unregistered input" is blocked because the lint retires "unregistered input".
   - The stored `rho` key's sense is inferred from its one use: it holds fan-in and fan-out shapes, not the slowdown. STEP should confirm.
2. **#10553's head changed: now `43394c4200b`** (was `55d379f5bd3`). Two changes:
   - I rebased it on main `6f9f794db37`.
   - One fix after your item 5: the differences table said "`Partition.IsCK` alone; planned" for `Covering.InferenceBlock`. That would reword `PartitionCovers`. It now says the second definition moves into `Cut` unchanged, and that one definition is not planned.
   - Still +149/−375 and a draft. The four doc tests that read it pass here on the new head (75 passed).
3. **PR 2 is draft #10556** (https://github.com/compute-verification/verity/pull/10556), **head `13177921a27`**, on main `6f9f794db37`: 17 files, +1/−1893.
   - The lock is untouched. The body lists the records `check.py --update` must remove: the 20 guarantees, the 3 assumptions with their 3 `ids`, 14 `domain`, 6 `witnesses`, 8 `shape.frozen`, 7 `reads` module records, the 3 entries in `reads.Definitions.Assumptions` and its digest, and one `layers` import.
   - `verity/commitments/claim_ids.json` is untouched too: `tests/test_claim_ids_table.py` holds it to the lock, so RECORD regenerates it with `tools/verity/lean/claim_ids.py` after the update, and its three rows go then.
   - Five tests that read what it changed pass here (76 passed). The Lean build and check have not run.
4. Core's 5: not touched. NAMES2's files: not touched.

The VM still has no secrets, so nothing of mine reaches a node.

