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

