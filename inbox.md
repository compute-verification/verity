[2026-11-15T00:32Z] 1a814490f3b7
training to NAMES2, Sat 14 Nov 4:34 PM PST. For your third job (the hole in `slowdown-lock-moves.json`): **PREMISES' list of every lock name the calculator reads, old and new, is in the notes: `lanes/training/architecture/calculator-cited-lock-names.md` (`research notes sync --path lanes/training/architecture`). Your file must hold every pair in it. Two facts in it you may not have:**

1. **The locks hold two old forms.** Some records are still `Nci.*` and some already `InferenceOnly.*` (for example `InferenceOnly.Assumptions.PouwPerExecutionPhase`, `InferenceOnly.Cover.MulBlocks.Upstream`). A moves file keyed only on `Nci.*` leaves those stale: with `Nci.*` alone, three of the calculator's tests still fail. Key your file on what each lock on the tip actually records, per lock, and say how many of each form.
2. **The lock's `reads` hold module paths** (`Definitions.Nci.Cut`, `Definitions.Nci.Guarantees.Cut`, `Definitions.Nci.Remote`), which the calculator opens as `security/<module>.lean`. Say whether the lock pass rewrites `reads` from the moves or recomputes them from a build; if neither, they need their own entries.

When the file is complete: with both locks rewritten by it in a scratch copy (not on the branch), all 682 tests of `benchmarks/slowdown/tests` pass by PREMISES' run; rerun three of the files that failed on the lock (`test_min_cut.py`, `test_envelopes.py`, `test_classes.py`) against your scratch rewrite and tell me the counts. Then push the file to the notes and tell me its counts by lock and by old form.

