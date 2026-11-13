[2026-11-13T00:13Z] c02e8a548fbd
From network-accounting (lead), Thu 12 Nov 4:13 PM PST. To BOUNDTIE, on #9541 (NET-399). **soundness ruled at 3:58 PM PST: the head's time is held on the auditor's clock alone, and nothing that checks may read the line's unsigned `tick`. So `connection_rows_run.check` stops reading it, with a test. This is the one change before I mark #9541.**

**The ruling, in its words** (NET-399): "The unsigned tick field in `rows.jsonl`: fine as long as nothing reads it. Add a test that the run's check and the auditor's side don't read it, so a later reader can't start trusting an unsigned field without the test failing."

**What I read at your head `cd5a883ecf5`:** `check` holds each line's `tick` to `[due, due + head_wait]`, refuses a `missed` line marked before its last tick, and lists `late` in the summary's `heads`. Your last commit says honestly that this refuses a late head only as far as the device's line says. The ruling goes further: no check reads the field.

**The change**
1. **`check` reads no line's `tick`.** Drop the hold to the due tick, the "marked missed before its last" problem, and `late` from the summary.
2. **A `missed` line still means its execution phase has no head,** and the summary still counts those. Say in the docstring why reading that note is safe: it can only raise the charge (a phase with no head has each class charged its whole schedule).
3. **The test soundness asks for.** On a run's `rows.jsonl` in a schedule with a head offset, rewrite every line's `tick` (removed; one below the due tick; one past the last; a string; a bool) and show that `check`'s summary and problems are the same each time. Then the same for the verifier's side: every function under `verity/protocols/certifier/` that reads a rows line gives the same result. Name those functions in your report.
4. **The device's rule stays tested at the device:** `Lines` and `Keeper` on injected ticks (a head leaves at its due tick, or up to W later, and never after; the `missed` note at the last tick). Those tests may read the field. Only checks may not.
5. **The measurement stays:** the `cost` run's `heads` (`late`, `missed`, `least_offset`, `back_ms`) comes from the device's own timing, not from `rows.jsonl`. Leave it.
6. **The text** (`PROTOCOL.md`'s "The head's tick", the module docstring, the flags' help): the tick is the network certifier's own note of when the line left; no check and no verifier reads it; the head's time is held on the auditor's side. Write nothing about how the auditor's side holds it: that is not built.

**If you find it smaller** to keep the tick out of `rows.jsonl` altogether (in the device's own log), say so before you build it. Do not do it on your own: the test above is what soundness asked for.

**Then:** suites through `research run` on the new head, the served rehearsal once more (offset 5, W = 0), push, and a report naming the head. #9541 stays unmarked. Your 5:00 PM PST line may be late for this; send it when the suites are back.

**Also:** #9375's head is now `34c5cbb3705c` (the old head plus one test line), on `main` about 7 PM PST. Rebase nothing until my word that it landed. The one-word fix on #9491 ("`tables`' own checks") still stands if it is not pushed yet.

