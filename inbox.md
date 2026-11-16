[2026-11-16T10:06Z] 2e1737a80c17
model: Opus 5.5 (claude-opus-5-5-high); subscriptions removed: none held

training's worker NAMES2, 02:07 AM PST. This is the report on #9519. It is carried and pushed at b18ca3fe383, a fast-forward from 6c1cbb1c18f, and green there (ran). It is not marked.

- **The merges.** It merged main 354433948d2b, then main 143a5e7c958 (the names pass); both were clean (ran).
  - The move renamed none of our files. Our seven files keep their paths, and there were no hand hunks.
  - stale_names.py, with and without lean_names_pass_map.toml, finds 0 of our lines in old names.
  - The names pass renames none of our names: none is in lean_names_pass_moves.json, and none of our hand-kept entries spells a renamed word. So --moved was not needed.
- **The lock.** I took main's lock and applied our hand entries again exactly (ran): domain Slowdown.Cut.ReadsOn, witnesses for HECharged, the three listed guarantees, and in shape.frozen three Guarantees.HECharged* entries out and one SecurityProofs.HECharged entry in.
- **Records run r20261116-090839-53df: done, SUCCESS (ran).** It ran on vy-nebius-cpu-2, which auto chose.
  - AUDIT PASS for security/Proofs (216341 declarations), security (1066 guarantees) and security/Properties (581).
  - Axioms are only propext, Classical.choice and Quot.sound. The outcome is "regenerated".
- **Suites run r20261116-093609-2b94, fresh:** benchmarks/slowdown 682 passed, benchmarks/inference_only 63, verity-lean-audit 867. Locally, tests/test_lean_packages.py and tests/test_lean_vocabulary.py: 49 passed (ran).
- **The lock against main (ran).**
  - The changed keys are only ours: the three guarantees, the domain entry, the witness, the four frozen entries, and the reads records of Definitions.Slowdown.Cut and Definitions.Slowdown.Guarantees.Cut.
  - No other guarantee's record changed.
  - Our records are byte-equal to the old head 7ecfdc01fc1's.
- **What a reader must read over 7ecfdc01fc1:** the merge of main plus regenerated records, no change of ours. The PR's own patch outside the lock is byte-identical to the old head's (ran).
- **Main moved again,** to 688a98322671: 19 files, none of them Lean, a lock, a move map or our benchmarks (ran). I did not merge it.

Next is #10383. Its records run r20261116-090953-1552 is still running.

