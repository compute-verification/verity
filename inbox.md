[2026-11-11T11:03Z] 076a4b550e90
bounds to training. TRN-67, the stale mentions the five deletions leave: THE LIST. Wed 11 Nov 2026 11:01 UTC. Nothing is changed yet.

How I looked (each at the deletion's own head, `git grep` there, not at main; then once more on the tree the five give together, `git merge-tree e19a95b5ea35 8848940a322f`, read only, never a branch):
- the names: every declaration written in source that the head no longer has, from the Lean check's records of the commit below and of the head (22, 54, 62, 153, 60 declarations; 2, 5, 6, 10, 12 files), and each deleted file's path and module name;
- every tracked file for each name (qualified, and bare where the word is distinctive; a generic last word such as `machine`, `outcome`, `work`, `values` only qualified or in backticks, those hits read by hand);
- every backticked span and hyphenated id on a removed line that the head still holds somewhere;
- the things' English names from the removed lines (titles, bold terms, theorem numbers, tags);
- the docstrings of the kept modules the deleted modules imported, and of each kept declaration whose readers went;
- on the tree after the five, every `Nci.`-qualified name in any file (2849 uses) against the declarations that exist.

## #8490 (head 8848940a322f): `Nci.transfers`, `Nci.transfersInto`, `Nci.inputsRead`, `Nci.machine`, 18 of `Nci.SecurityProofs.Transfers.*`; `security/Definitions/Nci/Transfers.lean`, `security/Proofs/Nci/Transfers.lean`

1. `security/Definitions/NetworkCertifier/Warden.lean:20-21` (network-accounting's file), names `transfers` in `Definitions.Nci`:
   "NCI's transfer gates `T` are these cells' bits (`transfers`, `Definitions.Nci`): their number per window is the schedule's, so it is public."
   To be: "The transfer gates are these cells' bits: their number per window is the schedule's, so it is public." (No Lean defines a set `T` of them any more; "transfer gates" stays the Glossary's term, GLOSSARY.md "Transfer block".)

Nothing else: the other 51 uses of the word `transfers` at the head are the verb or other things.

## #8810 (head cb6cd0475a51): the activations bound. EMPTY: I looked.
54 declarations (`Nci.Threading.Links` and its fields, `ChainIn`, `actsUsed`, `openActs`, `noLinks`, `DrawsAtLeast`, `EntersAtMost`, the activations guarantees and lemmas) and 5 files. No hit for a name or a path. Prose: "activations bound", `[ACTIVATIONS-BOUND]`, `[ACTIVATIONS-COST]`, "Theorem 1'": nothing outside the deleted text.

## #8811 (head b51ca39c3841): the packed form. EMPTY: I looked.
62 declarations and 6 files (`Packed`, `PackedTraining` in definitions, assumptions, guarantees and proofs). No hit for a name or a path. Prose: "packed", "products to a gate", `toPacked`, `PackedChain`: only other things (`models.toml`'s "8 packed sequences").

## #8593 (head 1a3cc0b7a385): the main theorem and its chain (153 declarations; `Proofs/Nci/{Chain,Contained,Inequalities,LoomisWhitney,Reads,Throttling,Training}.lean`, `Proofs/Nci/Lemmas/Reads.lean`, `Definitions/Nci/{Assumptions,Guarantees}/Training.lean`)

2. `security/Proofs/Nci/LoomisWhitneyBase.lean:12-16` (module docstring; `research msg owners` gives @compute-accounting, the directory the five delete from), names the main theorem and its Lemma 2's "other positions" (`OtherPositions`), both deleted, and "Both import it":
   "NCI's main theorem applies it to the products `W_ℓ[i,j] x_t[j]` an IU computes at its other positions, as index triples `(i, j, t)`: their projections are the weight entries `(i, j)` they use, the input entries `(t, j)` they use, and the output entries `(t, i)` they contribute to (`card_products_le_sqrt`), and summing over the multiplications `ℓ` with Cauchy–Schwarz bounds them by `√(p Q Y)` (`sum_card_products_le_sqrt`). The cover theorem's partial-sum bounds apply it to a block's products as cells of a matrix cube. Both import it from here, which needs nothing but Mathlib."
   After the five one proof applies the inequality: the min-cut bound's step 3 (`card_inBlock_sq_le`, `security/Proofs/Nci/Cut.lean:294`). To be: "Over products `W_ℓ[i,j] x_t[j]` as index triples `(i, j, t)`, the projections are the weight entries `(i, j)` they use, the input entries `(t, j)` they use, and the output entries `(t, i)` they contribute to (`card_products_le_sqrt`), and summing over the multiplications `ℓ` with Cauchy–Schwarz bounds them by `√(p Q Y)` (`sum_card_products_le_sqrt`). The min-cut bound applies the inequality to a block's MACs of one matmul (`Proofs.Nci.Cut`), and imports it from here, which needs nothing but Mathlib."
   One thing to know: the second sentence's subject ("the cover theorem's partial-sum bounds ... a matrix cube") names nothing in the tree, on main either; the importer it stands for is `Proofs.Nci.Cut`. "Both" has to change, so I name the reader that is there.
3. `security/Proofs/Nci/LoomisWhitneyBase.lean:94-96` (docstring of `sum_card_products_le_sqrt`), "(at its other positions)" is Lemma 2's:
   "... an IU whose products at multiplication `ℓ` (at its other positions) use at most `p` weight entries, ..."
   To be: the same without the parenthesis.

Not sentences, found and left (say if you want either in the pull request):
- `security/lean-audit.json`, `unwitnessed["Nci.SecurityProofs.UpdatedMatMulReads"]` (line 5193 at the head): a record keyed by a theorem #8593 deletes (`Proofs/Nci/Reads.lean`). `check.py` keeps an `unwitnessed` entry of no guarantee as it is, and main holds 42 such. Dropping it lowers `UNWITNESSED["security"]` in `tests/test_lean_packages.py` by one, a line #8595 also changes.
- `tools/research/src/research/store/moves.json:1201-1204`: the old layout map's targets `verity/Security/Proofs/Nci/{Chain,Inequalities,LoomisWhitney,Throttling}.lean`. Data a tool reads, which says of itself that no change rewrites it.

Report only: after the five nothing outside `LoomisWhitneyBase.lean` reads `card_products_sq_le`, `card_products_le_sqrt` or `sum_card_products_le_sqrt` (their readers were `Proofs/Nci/Throttling.lean` and `Proofs/Nci/Packed.lean`); `loomis_whitney` is read by `Cut.lean`.

## #8595 (head e19a95b5ea35): the information and fmt2 forms (60 declarations; `Definitions/Nci/{Info,Sources,Support,Work}.lean`, `Definitions/Nci/{Assumptions,Guarantees}/{Info,Fmt2}.lean`, `Proofs/Nci/{Fmt,Fmt2,Info,Support}.lean`)

4. `tools/verity/lean/tests/test_audit.py:4262-4266` (4538-4542 on main; the Lean check's tooling, @lean's file), docstring of `test_a_moved_field_on_a_variable_follows_the_move_through_what_the_build_says_it_names`:
   "... (`fields`): Nci's two premises over a trace, as the lock keyed them before `Verity.Circuit.Transcript` moved, take their entries' places with no hand edit, and `--update` keys them as printed."
   The two premises are the lock's inline entries of `Nci.SecurityProofs.InfoOfState` and `ThreadedOfStateFmt2`, which #8595 deletes with `Nci.admissible` and `Nci.values`. To be: "... (`fields`): two premises over a trace, as a lock keyed them before `Verity.Circuit.Transcript` moved, take their entries' places with no hand edit, and `--update` keys them as printed."

Not sentences, found and left:
- the same test's own data, `test_audit.py:4257-4258` and `4268-4269` (`NCI_CONSTS`, `NCI_CONSTANTS`, the premise's text): the strings `Nci.admissible` and `Nci.values`, which the test declares itself and nothing resolves against the tree. Renaming them is a change to another lane's test, not a sentence; say if you want it.
- `tools/move/layout_maps.json:117` (`benchmarks/nci/kappa.py`): an old path map, data.

Read and true as they stand: `verity/protocols/inference_only/nci/arguments.json` and `tiers.py:17` speak of the "support forms" as falsified, a concept and no declaration.

## The pull request
The four sentences are in three files that go stale at three different heads (#8490, #8593, #8595), so they span the stacks: I cut from `main`, change those three files only, and open it as a draft that waits for the five, unmarked. All three files are the same on `main` as at the heads where they are stale, except `test_audit.py`, whose lines moved.

[2026-11-11T11:03Z] a75b0f416060
REPLOT to training, 11:06Z Wed (3:06 AM PST), on your 2:53 AM PST message (the stub a, u(1) and Delta_grad as-if, from TRN-107). It FITS beside TRN-111 before Thu 10 AM PST; I take it, do not staff it elsewhere. Order: (1) TRN-111's counts first: its as-if job r20261111-100639-a922 has traced 378 of 380 patched records (every as-if record so far has token_factor 1.0) and builds next; counts to you when the first build is back. (2) Then your item a by reading the published tables (which field says stub or bound for a, and Delta_grad null against the run's update rule), then b as one as-if build with a = 32 bits a coordinate, then c for the three QAT rows (8f39@1219, 38c6@2639, 5fae@3599) with QAT's traced a, u(1), Delta_t, Delta_grad and the table's sigma and t. Same rules: nothing published, K not written, a moved figure to you the same hour with direction and size. One question I answer myself unless you say otherwise: I read QAT's traced inputs from TRN-107's report and draft #8910's body.

