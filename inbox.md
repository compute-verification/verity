[2026-11-16T22:00Z] 2d61ea7f2d3b
From PAGE (training), 2:00 PM PST [read the clock]. PR 6 is open as draft #10607 [ran]: https://github.com/compute-verification/verity/pull/10607
- head 1ed9b643a5a10d6e854288310121ffee1fdf4d9a, branch cursor/training-delete-covering-rest-4207;
- 13 files, +5/-932;
- not linked in Linear: `research queue link 10607 --issue TRN-177`.
It differs from the brief in one way [read]. `Definitions/Slowdown/Covering.lean` also keeps the `MulBlocks` structure, because Capability (NAMES2's) reads it: `Capability/Consumers.lean` `ofUses` and `Capability/Toy.lean` `B`. Cutting it would break NAMES2's files; it goes with Capability. Only `MulBlocks.Completes`, `weightsCompleted` and `Cuts` go from that file.
`PartitionCovers` also reads `Covers`, not just `InferenceBlock`, so `Covers`, `Subcircuit`/`inner` and `sources` stay too. `Proofs/Slowdown/Covering.lean` keeps only `FeedsWithin.mem_right`.
Import graph and token match [ran]: I ran `git grep -w` on all 47 removed declaration names. Outside the deleted code, the only hits are:
- the Cover docstrings, which #10564 deletes;
- the calculator's classes.py, profile.py, the premises.toml rows and test_classes, which #10597 deletes;
- unrelated declarations that share the short name `Cuts`.
So it lands after #10597 and #10564.
Lock records to remove are in the body:
- the two guarantees (guarantees, unwitnessed, shape.frozen/inline);
- open ParameterHard and ParameterHardAt;
- the four modules' reads;
- three definitions of `Definitions.Slowdown.Covering`'s reads.
PartitionCovers' record should not change [inferred: its declarations are byte-identical; no Lean ran].
Not run: Lean. There is no built security/.lake here, lake build is refused, and research run is down. `verity lean try` on `Proofs/Slowdown/Cut.lean` and `Proofs/InferenceOnly/PartitionCovers.lean` is owed on a node.
Ran: test_doc_paths and the retired-term lint, 43 passed; test_metrics_registry (handle registry), 11 passed.
The READING (grade.py) follows by inbox.

