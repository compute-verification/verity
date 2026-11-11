[2026-11-11T20:04Z] 6e28d4f1a35d
COUNTS to training, copy FORM, TRN-130, Wed 11 Nov 12:05 PM PST. THE RECORDS ARE REGENERATED AND COMMITTED, AND FORM'S CANDIDATE c1ac6b770a IS MERGED WITH NO CONFLICT: `cursor/training-digest-form-string-a8e2` at 6d9709d399, pushed. The suites at my head: 39 of 49 passed, none failed, the ten long ones still running.

THE RECORDS (RAN, r20261111-182100-c7e3 on vy-nebius-cpu-1 at 6c1cf5e7a5, `lean_changed.py --records verity/protocols/inference_only/verifier/nci/lean --update`; "done rc=0", OUTCOME "regenerated"; kind lean-fast/v1, so no kernel replay and no tests):
- The check's package: "PASS  984 declarations in 13 modules; axioms propext, Classical.choice, Quot.sound; 0 guarantees"; "check.py exit 0 in 239 s". Regenerated.
- The certifier's verifier, which calls `closureDigest`: "PASS  1484 declarations in 10 modules ... 4 guarantees"; "check.py exit 0 in 38 s". No record moved.
- `security`: first "FAIL", "every failure is a record difference"; then with `--update` "PASS  27021 declarations in 715 modules ... 1077 guarantees"; "check.py --update exit 0 in 2065 s". Regenerated.
- `security/Properties`: the same two steps; "PASS  3088 declarations in 94 modules ... 463 guarantees"; "check.py --update exit 0 in 1146 s". Regenerated.
- `security/Proofs`, built beside each: "PASS  193369 declarations in 3819 modules ... 0 guarantees". No record moved.

THE LOCK LINES THAT MOVED (commit b0fe2bdf4b: the run's three files copied over the locks, their sha256 equal to the run's `record_sha256`; nothing by hand). RAN: the locks before and after compared as data.
- `security/lean-audit.json` and `security/Properties/lean-audit.json`, four paths each, all under `reads / NciCheck.DefinitionDigest`: the module's `digest`; `canonicalEntry` and `closureDigest` rehashed; `isComposite` new. `definitionDigest`, `flatParts` and `written` keep their hashes. No guarantee's statement, premise or trust gap moved.
- `security/lean-audit.json` also prints three entries in another place, same content (`NetTiming.SecurityProofs.PooledLeakCarried`, `Verity.SecurityProofs.ServingBlockSum`, `Verity.SecurityProofs.RefineIncorrect`): the run's order; #8686's head had them out of it.
- The check's package's lock: one line, `"format": 1`, which the tool writes on a lock it rewrites ("rewritten under today's names and layout, saying what it said"). READ: 12 of main's locks carry it; this one did not.

THE SUITES SO FAR (RAN, r20261111-182236-62a9 on vy-nebius-cpu-5 at 6c1cf5e7a5, `suites.py --changed 2177450071`; still running at 12:00 PM PST): "the change from 2177450071 touches 49"; 39 passed, 0 failed. Among them the ones that read a digest through Lean:
- "passed  verity/protocols/certifier/verifier/network 83 passed in 512 s": the period check's pins against the Python mirror (`test_network_period_protocol.py`, `test_slot_entry_pins.py`) and the check's difftests (`test_nci_*_difftest.py`), READ from the suite's directory.
- "passed  verity/protocols/certifier/verifier 1152 passed, 1 skipped in 983 s"; "passed  verity-network-traces 519 passed in 338 s"; "passed  verity-core-composition 27 passed in 4691 s" (it ran because `NciCheck/DefinitionDigest.lean` changed).
- Still to end: verity-numerical, research, verity/circuits, verity/commitments, verity/commitments/windows, verity-core-evaluate, verity/definitions/ops, verity/definitions, verity-flock, verity/protocols/pouw/verifier. I send the last line when the run ends.

THE MERGE (6d9709d399, `git merge` with the lock drivers registered; no conflict):
- RAN: against c1ac6b770a my branch differs in seven files: the digest's module (10 lines), the new proofs file, the umbrella's import, the retired-terms list (as data: my file's entry and the umbrella 14 to 18, nothing of FORM's lost), and the three locks.
- RAN at the merged tree: the retired-terms counts equal the list for the three files of mine (umbrella 18, my proofs file 4, the digest's module 2).
- The check's package's lock merged to FORM's content plus `"format": 1`, in the driver's order and layout, not the tool's. So a records run at the merged head will rewrite that file; I run it and commit what it writes.

RUNNING NOW, one at a time on a node at 6d9709d399: `RecordRead.lean`, `RecordInputs.lean`, `RecordKinds.lean`, `RecordTemplates.lean`, lean's `security/Properties/Nci/Record.lean` and `security/Proofs/Properties/Nci/Record.lean`; then lean's `OpaqueStop.lean` and `Opaque.lean` as untracked copies, removed after. FORM: I tell you the minute one breaks.

THEN: the suites with `--changed c1ac6b770a` at the merged head, as you asked, when the first suites run frees its slot; the records at the merged head; the draft.

ONE QUESTION, yours to rule: the draft opens when items 1 to 4 pass. Item 3's suites at 6c1cf5e7a5 are ten long suites from done. I open the draft at the merged head when that run ends green, with the merged head's runs listed in the body as running. Say if you want it opened only after the merged head's suites.

NOT CHECKED: the ten suites still running; anything at the merged head beyond the lint counts; the package's check with kernel replay at my head (FORM's r20261111-180507-34d9 is at 16c91c972e, before my change).

