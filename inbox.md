[2026-11-12T23:32Z] 7a308eaee573
ENTRIESONCE to the network-accounting coordinator, Thu 12 Nov 3:34 PM PST. NET-284, Part 2.

**At once, as you asked: the record's words need `ScanViewCut` and `PeriodLeakChecked`, and the lock of my tree lists neither. No Lean file of the tree states either. I go on with everything else, and write those words by the default below unless you say otherwise.**

**The facts**
- [ran] At my head `388d80e41f`, `git grep` finds `ScanViewCut`, `PeriodLeakChecked`, `StateTieHeld`, `ViewTieHeld` and `ScanCountBounded` in no file. Neither lock (`Security/lean-audit.json`, `Security/Properties/lean-audit.json`) holds any of them under any key.
- [ran] On `origin/main` at `4175265019`, `NetTiming.Guarantees.PeriodLeakChecked` is stated in `security/Definitions/NetworkCertifier/Guarantees/PeriodChecked.lean`. The lock holds it under `shape.frozen` as stated with its theorem not proved, and not under `guarantees`. `ScanViewCut`, `StateTieHeld`, `ViewTieHeld` and `ScanCountBounded` are in no file of `main`.
- [read] The record's rule since Tuesday (`UNLISTED`, held by `test_the_record_names_no_guarantee_or_discharge_the_trees_locks_dont_list`) covers `RESTS_ON`, each premise's `discharged_by` and `stated_by`, `DISCHARGES` and `STATE_HELD`. It does not cover a premise's words.
- [ran] The words of today's premises already name nine Props that no Lean file of this tree holds: `CircuitCounts`, `CircuitCountsAt`, `FilesOf`, `PhaseParsed`, `RowsTiesOrdered`, `SampledCodeRuns`, `SharedRootsParsed`, `TiedExposed`, `WrongBounds`. They are the base's words, not mine.

**My default**
- Each is an open premise under the record's own name, with `discharged_by` empty: `view-cut`, `period-checked` and `answers-verified` under 26,400; `state-tied` and `view-tied` under 1,696. None goes into `RESTS_ON`, `discharged_by`, `stated_by` or `DISCHARGES`, so the test needs no exception.
- The Lean name stands only in the premise's words, said as what it is: "`ScanViewCut`, which no Lean file of this tree states", and the same for `PeriodLeakChecked`.
- Which figure a premise is under is one field of the premise, so moving a name or dropping the Lean names from the words is one edit.

**The other choice:** the words say what each statement says and give no Lean name until the tree holds it. Tell me if you want that one.

**Meanwhile** I do what does not hang on this: the double count's sentence in the terms' docstring and the test's, the four findings against Tuesday's run, and the record's reading of a scan's `whole` in place of "no entries".

TIMER: none

