[2026-11-11T02:59Z] d9deca2f1257
Coordinator to ONSITEMAIN, NET-330, Tue 10 Nov 6:59 PM PST. **A correction to my brief: its last "what I already read" line and step 4's premise are wrong.** `protocols.wipe`, `protocols.space` and `protocols.bandwidth` in NET-330's body are rows of the verdicts file, not modules. I looked for directories under `verity/core/protocols/`; ignore that line.

- **What is there** [read by me]: `benchmarks/one_stage/served_verdicts.py` writes the rows `space`, `bandwidth` and `wipe` under `protocols`, on `main` `a930252c4993` (lines 154 to 166) and on staging `00d79b22e4d2` (213 to 225). The record reads that map already: `onsite_record.py:1911`, then `row_verdict(rows, …)` for PoUW and the certifier.
- **Step 4 is then:** what the three entries would be as code in the record (three `row_verdict` reads, `VERDICTS`, `ROWS`, `of_record`'s condition), what each row holds when its piece did not run (so what `of_record` must refuse), which tests of the record would change, and whether anything on the onsite branch stops them being added there now. Say also what fills each row in a run (which piece, whose), read from `served_verdicts.py`; whether those pieces run in the demo is not ours.
- I have said the same on NET-330 (a correction to my 6:52 PM PST comment). The rest of the brief stands.

