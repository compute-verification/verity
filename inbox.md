[2026-11-13T08:37Z] 7b62b1375646
**Ruling on the `OwnReads.lean -> CanonLine.lean` edge: (b). Not (a), not (c).** Your 08:30Z entry is read.

- **Why not (a):** widening `public_boundary.toml` so the `flock` group may take `Properties/Challenger` would let Flock's proofs depend on the challenger's properties, and through `CanonLine.lean` on `Composition` and on PoUW's `LiveSalt/Canon`. Daniel asked tonight (TOP-415, 12:15 AM PST) for a list of exactly such couplings so they can be removed. I add none.
- **Why not (c):** it puts one more hypothesis on `addressesConnected_of_held` and `ksTwo_of_facts`. Those feed three listed guarantees; no statement on that path gains a premise to make a test pass.
- **(b), as you sized it:** the two lemmas (`canon_inj`, `field?_equiv`) and what they stand on move to the public side under `Flock/Claims`: facts about Flock's canonical text belong with Flock. Move only what the two lemmas need; if `CanonLine.lean`'s imports of `Composition` and `LiveSalt/Canon` are for other lemmas of that file, they stay behind with those lemmas, and say so. The private files then import the public ones.
- **When:** after chain 2 is on `main` and the join is carried, as your first change after the carry. Not tonight: tonight's maps and lean's records run touch `security/`, and new Lean files written before them are carried twice.
- **Until then, by this ruling:** `test_no_public_file_needs_one_the_export_does_not_copy` stays red on your line and on the join for that one edge and no other. You write no ready line and the base does not move for it. Name the edge in every entry that reports a suite, so nobody reads the red as new.
- **The first import: go on as you wrote.** Move `sortedKeys_of_wf` beside `CanonRead`, point `OwnReads.lean` and `SpecTie.lean` at it, one try through `--on auto`, and name the head for ASSEMBLY when it compiles. `test_spot_never_imports_the_network_certifier_in_lean` must pass at that head; run the two tests there and quote both lines.
- **One more thing your last line raises:** you ran the two tests at your head and the base only. When the cut compiles, run the same two tests once at the join's tip (`42bba26494`, or ASSEMBLY's next head if it is named by then) and list every file they name there, so that another worker's import across either boundary is found now and not at the carry.
- HONEST2's 08:32Z entry says `OwnReads` and `PortReads` are gone from every statement of its instance. Say in one line whether `ownReads_of` is still read by `wiring_of_wiringOk` after that, or whether the file can leave the join's path; if it can leave, (b) is not needed and I want to know before anyone writes it.

vy-nebius-cpu-13's `proofs` pool starts nothing after 10:30Z and is closed 11:30Z to 14:30Z.

