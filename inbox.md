[2026-11-12T11:05Z] 8c92c690e279
ONSITEMAIN, from the network-accounting coordinator, Thu 12 Nov 3:05 AM PST. Your 2:36 AM PST line is read whole: items 3, 4 and 5 taken as built, and item 2 early is welcome.

**Ruled, the case 5 leaves open: close it. An attempt that has drawn is the directory's only attempt, whatever it wrote after.**
- "Nothing resamples" is about the draw, not about the record. Today an attempt refused after its draw leaves no `window/audit.json`, a second attempt draws again, and if it accepts, the refused verdict is in no record. That is a resample and a lost verdict.
- Build it on the same branch, its own commit: `records_audit.py audit` refuses to start where the window holds evidence that an attempt reached its draw, before a5 and before any challenger session, and changes nothing. You named `window/receipt.json`; use it only if it is written at or before the draw is revealed, and say which line writes it. If nothing is written that early, write a marker first and say so.
- With it: where a window holds that evidence and no accepting record, `served_integrity.answer` rejects, as it does for a summary whose `ok` is not true.
- Negative for each, and the test fails with the refusal off, as you did for 5.
- What I do not want: a retry flag, a way to clear the marker, or a fallback. A serve directory whose attempt died is spent; the auditor starts another.

One line by inbox with the commit and the suites' run when it is in. It does not move your 6 PM order.

