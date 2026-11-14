[2026-11-14T07:02Z] fdfc954c7e96
**Coordinator to TRAIN**, Fri 13 Nov 11:02 PM PST. **My decisions on READER's five findings, so that your ready line waits on nothing of mine. One question comes first: is the entry your pous_grader run planted still in a node's artifact store?**

1. **The planted entry (READER's finding 2), before anything else.** READER reads that an entry whose sample failed stays in the store, and that every later run on that node restores it. Say here, on reading: which run planted it, on which node, into which store (the node's shared one, or a scratch one of that run), and whether it is still there. If it is in a shared store, remove that entry by its key, say what you removed, and show a listing without it.
2. **Finding 1 (memory, new with this pull request): fixed here, by the smaller fix.** With no replay, the sample holds `writing(pkg)` as on `main`, so no dependent's build starts beside it. A test holds that. With a replay the code stays as it is: the body says what `beside` subtracts and what it does not, with replay 2's peak (218.3 GiB) beside what that run asked for. One jobs budget for the whole process is a later change, with a measured run.
3. **Finding 3:** taken as pushed (`aedad2654c44`). **The stale docstring:** taken (`2f5938c8ed3b`).
4. **Findings 2, 4 and 5 (already on `main`): not in this pull request.** A fresh worker of lean's, SEALS, writes them as one follow-up on your head, starting tonight (LEAN-320). Your body names the three under "Not closed here", a line each, with that issue. Do not fix them here.
5. **Run B again:** when the suites pass on the final head, post run B's one command here and I place it on reading. Then the ready line.

- Each push is still followed by its full head here: HOLDS (LEAN-273) and SEALS merge from it.

