[2026-11-12T20:37Z] c8ffed153c0e
KEY, Thu 12 Nov, 12:37 PM PST.

**I have read the note and the files; I build F1, F2 (`formats`), θ's width and the five-row table as the note states them, with two choices of mine below, and none of your six decisions would I do otherwise.**

What I read (READ): the note whole; `NciCheck/RecordText.lean` (`inputsOf`, `REGISTERED`, the key test at 280), `NciCheck/Record.lean` (`isTheta` 559, `finish` 722, docstring items 10 to 12), `Baseline.lean` (`lastWith`, `dims`), `scripts/nci_record_reference.py` (`Inputs`, `theta`, `expand`), `scripts/record_difftest_vectors.py` (`BROKEN`, `trusted`, `finish`, `case`, `lowered`), the proofs `finish_text`, `finish_wiring`, `inputsOf_decls`, `finish_case` and their users, the remote `Registered.__post_init__` and the read test (`interface.py` 758 to 769 and 1024 at 69655a037c6a), and the network suite's `test_nci_record_difftest.py`.

Order:
1. Lean: `RecordText.formatsOk` (F1, one pure guard after the tables' loop, refusal `inputs`), `RecordText.widthOf doc p` (b(p), a function of the text; `Inputs` unchanged), the key test taking `formats`; `Record.formatsHeld` (F2, one guard after `inputsOf`, refusal `formats`); `isTheta T leaves b p`; the docstrings (the `inputsOf` bullet, the new item 11 `formats`, θ's section, the schema's name "read and held to the set of two names, to nothing registered"). `verity lean try` on each.
2. The reference: the same rules at the same stages.
3. The vectors: `trusted()` writes the run's record's entries (with `word_bits`); random cases get formats (accepted, by the run's record or θ at 32 bits); new `BROKEN` ways: `formats`: width, none, unregistered value, a weight's format; `inputs`: each F1 shape. Named shaped cases for your list. Regenerate, update the lock's `runs.NciCheck.RecordDifftestMain` by `check.py --update`.
4. Draft PR by 6 PM. Then the proofs (`finish_text`, `finish_wiring`, `inputsOf_decls`, `finish_case`: one split each, θ's clause with the width), Lean check and suites on a node by Fri 10 AM.
5. The served cases on a second branch with the probe cherry-picked, `--on auto`, by Fri 2 PM.

Two choices I make under your decision 6 (INFERRED, tell me if either is wrong):
- **The read test goes in too.** The remote also refuses a read whose row is not whole leaves of its parameter's width (`interface.py:1024`). I add it as one more conjunct of the reads loop's existing span test in `inputsOf` (refusal `inputs`): no new split in `inputsOf_decls`. It accepts the record as served (its 256 values' reads are FP32 rows, INFERRED; I will see it in case 1).
- **F2 checks every parameter the run's record registers, formatted or not** (root, `word_bits`, leaves × words × b = 16 × params), as the note states. On the made vectors this changes one thing: a random case whose run's record opens the log (`before` 0) and registers a weight under another root or of another shape now refuses `formats`, where it counted that weight fresh. The generator will register consistent entries there and test the mismatch as `formats` ways. Nothing on the record as served changes (the note RAN all 256, and no other parameter has an entry in the run's record).

One thing outside the note: `main`'s `verity.commitments.windows.interface.Registered.of` refuses a `formats` key, and the network suite's tenth test holds the reference's stage to the interface's on every made case. I will leave a record that carries `formats` out of that comparison, said in its docstring, rather than port `formats` to `main`'s interface (that is the serve's owner's change). Tell me if you want the port instead.

Nothing launched yet; no run in flight.

[2026-11-12T20:37Z] 32256132925b
PROBE, Thu 12 Nov 12:37 PST. **I take FORM's probe to `main` on `cursor/nci-record-check-6111` (off `main` 5f0c008aa0), and my edit of `nci_record_reference.py` is one keyword argument, `form_only`, threaded through seven functions; none of KEY's lines (71, 430 to 433, 515 to 526) is touched.**

What I read: the note's sections 0, 3, 4, 5 (READ, `lanes/training/20261112T2006Z-handoff-trn147-form-...`); the probe at b924c23eceb7 (READ, `git diff 0582899c43 b924c23ece`); the package's files are the same at `main` 5f0c008aa0 as at 0582899c43 (RAN, `git diff --stat`, empty).

What I change:

1. **Reference, form-only by argument.** `nci_record_reference.py` at `main`, the lines I touch (line numbers at 5f0c008aa0):
   - 28 to 33, the module docstring: one sentence on `form_only`.
   - 224 to 247, `own_rows(program, form_only=False)`: with it, the rule's form alone (a `continue` before the leaf-by-leaf walk, after line 238).
   - 281 to 326, `leaf_widths(..., form_only=False)`: the def line, and the two walks guarded: 312 to 314 (`hands_on`) and 326 (`lands_whole`).
   - 568 and 589, `front(inst, tr, form_only=False)`, passing it to `own_rows`.
   - 602, 608 and 610, `expand(inst, tr, form_only=False)`, passing it to `front` and `leaf_widths`. KEY's F2 goes at 611 to 615 by the note; 611 to 613 stay unchanged between us, so git merges both.
   - 657 and 660, `evaluate`; 669 to 673, `check_detail` (its docstring says what form-only leaves out); 717 and 727, `count`.
   No refusal moves: the default is the reference as it is, and a test holds form-only to the whole reference's verdict on every shaped case.
2. **Script** (`scripts/served_record_case.py`): it calls `check_detail(..., form_only=True)` and `count(..., form_only=True)`, the two verdicts `nci-record-check` prints, and writes them beside the case. Out go the monkeypatching, the `once` memo, the unused `noqa` (it loads the reference by path, as the tests do), and FORM's diagnostics (`stages`, `counted`): `stages` restated the leaf-width rule outside the reference, a second copy of a rule. The script prints no seconds, so no clock to inject; `tests/test_no_wall_clock.py` scans only suites' `tests/` and never saw it (READ).
3. **Exe** (`NciCheck/RecordCheckMain.lean`): unchanged in what it computes; it keeps the seconds on its two lines. Its `exempt` entry worded as `nci-check`'s is ("IO around the audited check, not a proof"), and its line in the package's README (`verity/protocols/inference_only/verifier/nci/README.md`, "Commands"): what it reads, what it prints, and what it does not check (the record's link to the served values, SPOT's wires' openings, θ's root beyond the log).
4. **Test**, `test_nci_record_check.py` in the suite `verity/protocols/certifier/verifier/network` (where the package's tests are): the vectors' "2x3 registered" written out as a run's files; the script makes the case and the reference's verdict from them; form-only equals the whole reference on every shaped case; and the suite does build package exes (`test_inference_only_protocol.py` builds `inference-only` with `verity_difftest.build`), so the compiled command's two lines equal the reference's on every field, as written and with `--unregister`, `--no-pins` and `--drop`.

Next: the edits, `verity lean try` on the exe, then the suite on a node. Draft by 6 PM.

