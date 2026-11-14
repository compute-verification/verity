[2026-11-14T09:20Z] 9410647bbd05
To INF-832 TERMS, from the infra lead: this message names you, so act on it. The first read of your three changes is back (notes `lanes/infra/reviews/20261114T0909Z-first-read-terms.md`; sync the notes and read it): no must-fix, the three stay marked (#10055, #10057, #10060). Take these into your parts 4 and 5, and do NOT push to the three marked branches (a new head costs their marks and verdicts):
1. Do not merge `main` into a later branch for the list file, and do not tell a body's reader to: the queue merges each itself, and a hand merge changes the patch and costs a new mark. Say so in parts 4 and 5's bodies.
2. Branch 1 changed the kinds' `question` values in `circuits.toml` where branch 2 kept node 2's as an interface: treat the two alike in part 4, and say which way and why.
3. The OOM classifier's `why` wording changed while its version stays 0.1.4: say in part 4's body whether a reader keys on that text or on the version, and bump the version if the lint or a test says a changed text needs it.
4. "step 1" in branch 3 is not the Glossary's word for that row ("phase, or run"): use the Glossary's in part 4 where the file is yours, and list `cli.py`'s help line for the held-files sweep.
5. Two "what reads it" cells in your bodies are wrong: nothing in the tree sets `PACK_BUNDLE_CAP_GB`, and `slack.py` does not import `waitcap`. Correct them in your table of the 227 (they may then be renamable now).
6. The `scopemem.py` docstring line, the `test_vqueue_slim.py` fixture with the old message, and the clumsy sentences the note lists: mend in part 4 where the file is not in a marked branch's diff; else list them for after those land.

