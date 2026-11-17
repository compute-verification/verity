[2026-11-17T01:14Z] e0123dbbc967
From PAGE (training). TRN-177.

1. #10553: head `165ee26c49e4ce56b226eddb9e80cb5fd23351e5`, pushed. One commit on `7ef39626a6c`, in `verity/protocols/compose/verifier/remote/README.md` only.
   - The exact sentence, now at lines 319–320: "**This is the argument's one endpoint.** No Lean states it, and it has no written derivation." It cites nothing. The following sentence, "The Lean bound `SlowdownRemote` is a different, older formula (below), and the calculator's figures (Results per monitoring period) come from a third route.", is unchanged; #10687 deals with it.
   - Before writing "no written derivation" I searched the branch's Markdown for the endpoint ("one endpoint", Ω_*, "fleet-time") [ran]. Besides this file, only `security/Definitions/Slowdown/Capability/README.md`'s "The shape of every bound" sketches it, as arrows with no derivation, and #10611 and #10613 delete that file.
   - The other three citations, each against the new `security/Definitions/Slowdown/README.md` [read]:
     - line 132, "Training's slowdown is stated in `security/Definitions/Slowdown/README.md`": true, its "The slowdown theorem" states it.
     - line 369, the old line 372: it now cites nothing ("…each a new hypothesis to earn."), so it has no fault.
     - line 446, "what a block's c and k count is in `security/Definitions/Slowdown/README.md` ("Inference circuits")": true. That section defines a (c, k)-inference block as at most c gates reading at most k bits outside θ.
   - Ran: `tests/test_doc_paths.py`, `tests/lint/test_repo_retired_terms.py`, `tests/test_glossary_lean_names.py` (46 passed).
   - Order: #10687 edits the same three lines; it drops the `SlowdownRemote` sentence and keeps the old citation. Whichever lands second resolves them to "**This is the argument's one endpoint.** No Lean states it, and it has no written derivation." with the `SlowdownRemote` sentence gone. If #10553 lands first, I do that in #10687's merge of `main`.

2. Threads PR: #10703, head `3438c2ec93a33286a059bcf3f5c9c0fd38f88dbd`, stacked on #10687 (base `cursor/training-delete-tiers-4207`). 28 files deleted; its own diff is +52/−3,850 over 46 files.
   - It deletes threads, layered, fit, piece, `Actual/Threaded`, the verdict properties over threads (`Throttles*` and `DescThrottles*`) and the three witness files.
   - `Guarantees/Actual.lean` (`ServerBlocks`, `ServerBlocksAt`) moves unchanged (R100) to `security/Definitions/ResourceAccounting/Guarantees/Actual.lean`. Its namespace was already `ResourceAccounting.Guarantees.Actual`, so the lock update needs no `--moved` [ran].
   - The lock update removes 35 guarantee records from the security lock and the Properties lock's 8 `Throttles*`/`DescThrottles*` records. The body lists every record.
   - The inference-only files [ran]:
     - `Lemmas.lean` only imports `Definitions.Slowdown.Threads`; the import goes.
     - `ReadsIn.lean`: `freshInput_piece_le_readsIn` is stated over `piece` and goes (its user was `Actual/Threaded`). `freshInput_pull_le_readsIn` is stated over the surviving `pull` and stays, with no reader now.
     - `Verdict.lean` is stated over `Throttles*` and goes whole, with `DescVerdict.lean`.
     - No stop: no surviving file states something outside the thread bound with a lemma of it. `Rate.lean` (`compliesRate_of_complies`), stated over the surviving `CompliesRate`, is likewise left with no reader.
   - `Proofs/Properties/Remote.lean`: the smallest replacement for `FitWitness` is its four declarations that `policyMet_honest` and `policyMet_nontrivial` read (`circuit`, `decl`, `serverBlocks`, `reads_le`). They are copied byte for byte (docstrings aside) into `Properties.Remote.SecurityProofs.PolicyWitness`, and the two statements are unchanged [ran: text compared, names checked for clashes]. That it builds under `Remote.lean`'s opens is [inferred].
   - `benchmarks/nci` is untouched; the body lists its readers. `test_tiers.py::test_reference_layered_at_kimi_k2` fails here (it reads `Assumptions/Layered.lean`) [ran]. `test_fit` and `test_arguments` will fail once the lock update removes `Reach`, `ReachSlots` and `ThreadedReads` [inferred].
   - Root tests at the head: 170 passed [ran].
   - Not run: the Lean build, the Lean check, the lock update, and the `repository` suite whole.

3. #10564's head moved to `6baa6435276` (`main` merged, plus a lock commit for the cover deletion). I merged it into the tiers branch: #10687 is now at `efbdf5ab5f440958bd6a33f2082bcb7204b251d7` with no conflict and its diff unchanged (28 files, +36/−2,604). The import scan finds nothing new, and the root tests pass (170) [ran]. The threads branch merges that head. One addition to #10687's body: `slowdown/params.py:238` names `SlowdownRemoteMonitoringPeriod`, and #10597 rewrites that line away [ran].

Order, unchanged: #10597 → #10564 → #10687 → #10703, all after NAMES2's #10611 and #10613. Both edit `security/Proofs/Slowdown/Witnesses.lean` (one import) where #10703 deletes it; the resolution is the deletion.

