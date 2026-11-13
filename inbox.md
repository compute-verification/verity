[2026-11-13T00:53Z] ac8f986b8e18
KEY, Thu 12 Nov, 4:53 PM PST.

**#9502 is at 9ddabfeafb with the `--format` option and its test on it; the twelve served cases ended with the compiled form equal to the reference on every field, and all five suite groups passed; the security proofs package's check and the one suite at the new head are still running.**

Head and branch
- RAN: `cursor/trn147-form-reads-formats-10f7` is at 9ddabfeafb on origin, a fast-forward from ea620a89cf: 4bb4d3a8cb (efd15349eb cherry-picked; its tree is efd15349eb's) and 9ddabfeafb (the test). #9502 is still a draft.
- RAN: #9474's head is b23c853a35 (`git ls-remote`, 00:50Z). The head merges with `origin/main` at 4e7a25c6a7 with no conflict (`git merge-tree`).
- RAN: the diff from ea620a89cf is two files, `verity/protocols/inference_only/verifier/nci/lean/scripts/served_record_case.py` and `verity/protocols/certifier/verifier/network/tests/test_nci_record_check.py`. INFERRED: no Lean input moved (no `.lean`, lakefile, lock or generator a lock's run names), so the Lean runs at ea620a89cf stand.

The served twelve, at efd15349eb (this head's form, reference, command and script)
- RAN r20261112-233108-21af, r20261112-233205-6c00, r20261112-233259-86da: each ended with exit 0 and `4 cases, 0 verdict(s) where the compiled command and the reference differ`.
- RAN: the verdicts are those of the twelve at dbe1d8ede3 on every field. As served: ACCEPT, blocks 4, gates 1,437,386,324, fresh 337,786,912; the count ACCEPTs, bool gates 11,044,318,495,127. One `pearl-a` format at 16 bits, a format for a weight, `registered.formats` dropped, one format removed: `formats`. A schema of another name: `inputs`. Less the smallest and the largest read weight, and no pin: the rate rule ACCEPTs and the count refuses `fresh`. Less a weight not read, and α held: ACCEPT. α held less the smallest: `nci` and `fresh`.
- RAN: a compiled case took 24 to 53 minutes and at most 11.2 GiB.
- The table and the commands a reader runs from the head are in the body.

The test (`test_format_makes_the_case_of_a_record_with_that_format`)
- RAN locally at 9ddabfeafb: `test_nci_record_check.py`, 5 passed. On each of the 15 made cases that state a format, the option's case and verdicts equal those of the files of a record the test itself states so (its own format again, out and in again, out, a schema of another name, the other width).
- RAN locally: it fails under each of three breaks of the option (the run's record not pointed at the new text, the format set on another key, a removal that removes nothing).
- RAN r20261112-234514-8867, the one suite that reads the script (`suites.py verity/protocols/certifier/verifier --changed ea620a89cf --quick`, at 9ddabfeafb): it took slot r0 at 00:38Z after 53 minutes in line, and has no verdict yet. READ: `rg -l served_record_case` names one test file, in that suite.

What you asked for, at ea620a89cf
- RAN, the five suite groups' last lines:
  - r20261112-232245-e582: `12 suites: 12 passed (0 cached), 0 failed; 45.2 min of test time`
  - r20261112-232311-85ba: `12 suites: 12 passed (0 cached), 0 failed; 36.3 min of test time`
  - r20261112-232338-3ab1: `11 suites: 11 passed (0 cached), 0 failed; 37.6 min of test time`
  - r20261112-232351-a95f: `11 suites: 11 passed (3 cached), 0 failed; 14.7 min of test time`
  - r20261112-232403-d24c: `11 suites: 11 passed (0 cached), 0 failed; 21.7 min of test time`
- RAN r20261112-232515-0fba, `RecordRead.lean`: `lean exit 0 in 289 s`; `axiom probe: none of its 16 theorems uses an axiom beyond propext, Classical.choice and Quot.sound`.
- RAN r20261112-232515-de9b, `RecordTemplates.lean`: `lean exit 0 in 56 s`; the same axiom line for its 16 theorems.
- RAN r20261112-232445-9cb4, the security proofs package's check: running on slot r4 since 23:25Z, no line yet. I send its last line when it ends.
- RAN at 9ddabfeafb against `origin/main` (4e7a25c6a7): `ratchet gate 4e7a25c6a766..9ddabfeafb61: 1 change(s), 1 failed`, `adds 3 retired-term site(s)`: `RecordCheckMain.lean` lines 3 and 15 and `lakefile.toml` line 57, all #9474's, none of mine.

vllm's two conditions, of the instance maker (`served_record_case.py`)
- The form's instance carries the pins: yes. READ: `served_case` puts in `pins` the closure digests of the Definitions the record's openings name as readers, unless `--no-pins`. RAN: the served case has 1 pin and 0 fresh opening bits; without it, 1,204,224.
- One α/Γ pair a class: an instance takes one pair, so a run of more than one class is one instance a class. READ: the script reads only the limits file's top-level `alpha` and `gamma`; it does not read `held`. By your ruling I build nothing until vllm's limits file has it. INFERRED: until then, a run of more than one class needs one limits file a class.

Housekeeping
- RAN: I deleted `cursor/trn147-format-option-10f7`, my own holding branch for the two commits. `cursor/trn147-served-cases-10f7` stays at efd15349eb, the served runs' tree.
- The handoff note `lanes/training/20261112T2243Z-handoff-KEY.md` is current.

What's left, in order
1. The last lines of r20261112-232445-9cb4 and r20261112-234514-8867 into the body and to you.
2. If #9474's head moves: merge it, run again what its diff touches, and say so.

In flight: r20261112-232445-9cb4, r20261112-234514-8867. Nothing else.

