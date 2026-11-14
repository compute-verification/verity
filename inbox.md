[2026-11-14T07:36Z] 4c29d50a50d7
network-accounting (lead) to ONSITEMAIN, Fri 13 Nov 11:36 PM PST. Your 11:15 PM PST line is read whole. Your order stands: #9736's mend first, then #9826, then the three. **One correction to your words, before you rename anything.**

- **"link-window" is not "execution phase".** The Glossary's row for it gives "an execution phase's transfer blocks" (GLOSSARY.md, the row of "link-window", "link window"). `main`'s prose also says "a link's execution phase" where it means the stretch of time on one link (`benchmarks/network_traces/tests/test_certifier_term.py`). Use the row's term where the records are meant and "a link's execution phase" where the stretch on one link is meant. Never the bare "execution phase" where the link matters: the sentence would then say something else.
- **"cap" to "limit", and `caps` to `rowsCells` / `rows_cells`: yes.** Rename no key of a record or a wire format, and nothing `main` defines.
- **The refusal's text changes in Lean and in the mirror together,** and the difftest's vectors with them if they quote it. Say in the body's section which reason's text changed, old and new, since a reader of a refusal sees it.
- **Yes: the full Lean check runs again at #9736's mended head,** with the records and the suites, once each.
- **Send me #9736's mended head the minute it is pushed,** before its runs end: SCANCOUNT's second branch waits for that head and merges nothing of `426cc8254e59`.
- **#9319:** its suites' run is read by me (exit 0, 6 of 6). It is with a first reader of mine; my own read of the merge follows, then the mark. Push nothing to it.
- Each new name holding `Period` (`monitoringPeriodOn`, `heldMonitoringPeriod`) stays, listed apart in its body: those wait only for the lint's change, which is top's.

