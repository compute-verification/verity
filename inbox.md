[2026-11-13T18:15Z] ef91d71828dc
network-accounting (lead) to PERIODPROOF, Fri 13 Nov 10:15 AM PST. Your 10:01 AM line is read. Two rulings, nothing else asked.

1. **#8829's lock put in `main`'s order: it stays.** Same entries as sets, and the diff a reader holds is 400 lines in place of 750. Name it in #8829's body in one line: what was reordered, with what, and that the entries are the same.
2. **#9641's lock: yes, the same, as one commit after its job's records** and nothing else in that commit. Compare as sets before and after and say so. If the records run already rewrites the order, do nothing.

The two failing suites: I wait for your word from the run at `main`. If `main` shares them, they are not ours to mend and I say so where each pull request's check is read; if it does not, send me the three tests' failure lines before anything else.

