[2026-11-14T15:51Z] bf3c9b6f0f54
PROVENANCE2 to network-accounting (lead), Sat 14 Nov 7:52 AM PST. Item 1.
- Lean check `r20261114-082016-4fd6` (`check.py --build security/Proofs` at `bce89121e5e13a5ee8f7d0c57fa6886e3e214195`): PASS. 204,450 declarations replayed, axioms only propext, Classical.choice and Quot.sound, the lock's 0 guarantees unchanged; no record moves.
- Suites `r20261114-082104-ba73` (`suites.py --changed 100bd8cb330f --quick`, same head): 6 of 8 passed. Two failed, both on one cause and it is my three commits: the pin `tools/pous_grader/TRUSTED.sha256` hashes `security/Proofs/lakefile.toml`, which my new root changes. Suite `verity/protocols/accounting/pous/verifier`, first failing test `tests/test_pous_trusted.py::test_trusted_hashes_are_current`; suite `repository`, `tests/test_pins.py::test_no_pin_is_stale`. Not the base, not the machine.
- Mend: `pins.py --regen` rewrote that one hash line (`pins.py --stale` now clean); commit `cfc0e11b7a8a0bf189c21261e41e8c56dcb8784e`, pushed to `cursor/na-scratch-provenance-7c3e`. Suites again at that head: `r20261114-155042-6416`.
- Item 2 launched: `verity lean try RecordProvenance.lean --on auto` on #10113's head `6635cfa9a943` with my branch merged in (local only, not pushed), `r20261114-154913-5f11`; it builds RecordDeclared and RecordRoles as imports.

[2026-11-14T15:51Z] 80e1344b3cd7
ROWSPROTO2 to network-accounting (lead), Sat 14 Nov 7:55 AM PST. #10108 Lean check: r20261114-084332-1ffe done, exit 0. `check.py --build --update` on the nci package at 2f8b1b9c3682e5386b8ce5e6d70cd965abf0c6b5, whose Lean is identical to head eba54ef543a850c19c1e948b2a3d7a0b61e99ca3 (only the test file changed after it): AUDIT PASS, 1866 declarations in 20 modules, axioms propext, Classical.choice and Quot.sound only, 0 guarantees.

[2026-11-14T15:51Z] 1eb5d9bb2f31
ROWSPROTO2 to network-accounting (lead), Sat 14 Nov 7:55 AM PST. #10108 lean try: r20261114-084423-209f done, exit 0, at 2f8b1b9c3682e5386b8ce5e6d70cd965abf0c6b5 (same Lean as head eba54ef543a850c19c1e948b2a3d7a0b61e99ca3). RowsProtocol.lean: lean exit 0 in 84 s; the axiom probe says it has no theorems. Siblings: none of the 10 built modules that do not import it declares a name it does. 8 were left unchecked because no build of them was at hand: the 6 difftest/fuzz/liveness executables, Difftest and RecordCount.

[2026-11-14T15:51Z] 4c4f17c6c74b
ROWSPROTO2 to network-accounting (lead), Sat 14 Nov 7:55 AM PST. #10108 cost and the network suite on a project machine: r20261114-085720-b853 done, exit 0, at eba54ef543a850c19c1e948b2a3d7a0b61e99ca3. The network suite: the rows test file whole, slow tests included, 15 passed in 74 s. The accept that parses both files takes 3.65 to 3.75 s over 4 passes, at a peak of 0.74 GB. Reading and hashing both blobs adds 0.36 s (0.07 s without blobs, 0.43 s with both). This matches the body (3.4 to 3.8 s), so no figure in the body changes.

[2026-11-14T15:51Z] a1aeded29799
ROWSPROTO2 to network-accounting (lead), Sat 14 Nov 7:55 AM PST. #10108 changed suites: r20261114-085746-3655 failed, exit 1, at eba54ef543a850c19c1e948b2a3d7a0b61e99ca3 (`suites.py --changed 5ae976fa --quick`): 59 of 60 suites passed, 7 of them cached. The one failure is mine: in the root suite, tests/test_public_boundary.py::test_no_public_file_needs_one_the_export_does_not_copy. The new test file is public and names the private nci/lean directory. The fix is cceb77c1b052b3842aaa909c857fbabf8aad5ae9, pushed: public_boundary.toml private glob test_{inference_only_protocol,nci_*}.py widened to test_{inference_only_*,nci_*}.py. With it, public_boundary.py gives 0 edges and the test file passes 56 tests on this VM. main deleted the export, its boundary and that test (26c6c517785), so merging #9855s new head will take the deletion and drop the fix. The suites run after that merge is the one of record. A dry merge of main into the head: test_nci_record_difftest.py (#9855s file, not mine) conflicts, so ROWSMERGE will meet it in #9855.

[2026-11-14T15:51Z] 08ee45ad4921
POLICYROWS to network-accounting (lead), Sat 14 Nov 7:55 AM PST. **`Spot.Bits.check` passes the read's circuit file as today's stager writes it, at three shapes. The two served files on record are refused by the verifier's parse before Bits is reached.** The map rests on the first, so step 1 stays as it is.

**The staged read: passes.** Run r20261114-085433-0473, done with exit 0, on #9547's head `dd6555ae618a6eaa797a2ec607a9f94a63e1a7a2`.
- What ran: `EmbeddingRead_v1{V=256,H}` at H = 4, 96 and 3072, staged by `UnitStager` on the verifier's own template (`structural`, layout `iv/hybrid/4`, bound to its template query's partition, a population of 2). Then `flock-read-tmpl check V H` and my probe program on each file.
- `flock-read-tmpl check`: `ok` at all three.
- `Desc.parseStmt` under `verity/flock-circuit/tmpl` reads each as a templated statement whose unit's template is the read.
- **`Spot.Bits.check` on the unrolled circuit, at the Definition's key: `none` at all three.** On the committed circuit, its template not unrolled: `none` too. `Spot.Bits.readsWhy` at every parameter: `none`.
- The unit's rows read 6,752, 8,224 and 55,840 input columns. Each is `readIn H 8`, the count the read's proofs use.
- The Definition's parameters are `tok` 2 words, `root` 32, `domain` 32, `row` H, `salt` 96, `path` 256, and its return is H words then 1. That is what `readRows` asks, word for word.
- Not covered: any table size but 256 rows, and a population above 2.

**The two served files on record: refused, and not by Bits.** Run r20261114-084936-c47a, exit 3, which is my probe's code for a file it could not read.
- Files: `w13:art:0af6dacba76ca3a3940c603776d377121f7a8aadc9ddf9f292bfbcc83a50d68a` (256 rows of 96 words) and `w2:art:71e9db9a04596503f9e52eb9fbdda2627e397026a5cedba6b65b1d2009c5a5b7` (256 rows of 4 words).
- Each one's META names `format flock-circuit/tmpl` and `statement verity/flock-circuit/tmpl`, and its first line is `flock-circuit/tmpl unit/serving-rows`.
- `Desc.parseStmt` refuses each: "circuit: no template section (TEMPLATE n, then the template's n lines) after META".
- **I was wrong at 12:52 and 12:54 AM PST to call them flat files of another statement.** They name the templated statement and lack its template section. Today's verifier reads neither.
- This is a finding that is not ours: whoever holds those two arts as a served read's circuit holds files this verifier refuses. I mend nothing of it.

**What it changes:** nothing in step 1. The lemma's hypotheses (the templated statement, the read's check on its template) are the ones today's stager meets.

[2026-11-14T15:52Z] 5cb2c73d8b83
PROVENANCE2 to network-accounting (lead), Sat 14 Nov 7:58 AM PST. Item 2. TIMER: none, and I hold no subscription.
- Try `r20261114-154913-5f11`: `verity lean try security/Proofs/Properties/Nci/RecordProvenance.lean --on auto` on #10113's head `6635cfa9a9439e4196d6b4706b895b6e2348bd72` with my branch `cfc0e11b7a8a0bf189c21261e41e8c56dcb8784e` merged in (local merge `baea7973a1a`, not pushed). Verdict: build failed in `RecordDeclared.lean`, so **my lemmas about `inputsOf` do not hold there unchanged**.
- The only errors are five sites in `declared_port` and `declared_opening` (lines 181, 183, 201, 219, 222) that write an `Opening` as a seven-field `⟨reader, table, key, ports, verdict, start, units⟩`; #10113's `Opening.mk` has eight. Lean reported no error past them in the file, and every module before it built, `RecordInputs` (the first walk) included. `RecordRoles` and `RecordProvenance` were not reached.
- The mend I'd make, argued and not run: add `(row : Option String)` to both lemmas, write the constructors with `verdict, row, start`, and pass one more `_` to `declared_opening` in `inputsOf_declared`. The loop's new `let`s and refusals should fall to the `repeat' split` already there. I can run it on a branch of my own on your word; until then nothing of mine changes.
- Waiting on: suites `r20261114-155042-6416` at `cfc0e11b7a8a` (running), and ROWSMERGE's line with #9644's final head (its branch is now at `6b90bd589026`).

