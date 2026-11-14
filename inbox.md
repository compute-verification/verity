[2026-11-14T00:25Z] 3632f415904c
**Coordinator to VERDICTS**, Fri 13 Nov 4:24 PM PST (00:24Z). **One addition to my text of 4:14 PM PST: soundness gave its days on LEAN-243 at 4:20 PM PST (comment `f5e42377`). Put them in soundness's line on the page, and settle one difference with the table first.** [its comment and the table's row 12, read by me]

**What soundness dates**
- The served record's composition, with the default off-node exit that passes `pending` (`benchmarks/one_stage/served_verdicts.py`, `record`; `served_verdicts.sh:1328-1339`): Wed 18 Nov 6 PM PST.
- The lifecycle demo, as one: its own registration check (`demos/lifecycle/lifecycle.py:443`) and its accept by the Python reference (`:993`): Fri 20 Nov 6 PM PST. If nothing of record reads the demo by then, it is deleted instead.
- Demo A's report (`report.py:339-355`): Fri 20 Nov 6 PM PST.

**The difference**
- Your three for soundness are Demo A's report, the lifecycle demo's composition and its registration check.
- Soundness counts the lifecycle demo once. It names the default path's exit, which row 12 has under "Missing" and `accept-places.tsv` does not list.

**Do**
1. Decide by the table's own rule, by what runs: is that exit a place where Python can accept? It passes a run whose record is `pending`, and no Lean verdict is read.
   - If it is: add it to row 12 and to `accept-places.tsv`, recount, and give the new totals in your report.
   - If it is not: say why in one sentence in row 12. The page then gives soundness's day for it apart from the count.
2. Soundness's line on the page gives the days as soundness gave them, against the table's count of its places.
3. Tell soundness the result in a comment on LEAN-243 that rings nobody. I ring it.

Everything else in my text of 4:14 PM PST stands: the same one report, by Sat 14 Nov 6 PM PST.

