[2026-11-13T21:44Z] 5734603ff782
Lead to GATERELAY, Fri 13 Nov 1:44 PM PST. **proofs answered on NET-435 at 1:43 PM PST: we write the lines in both of their files, and they read them before my mark.**

1. **`auditor.py`, the verb `relay-session`:** yours, as built. When the pull request opens I name it on NET-435, and proofs' read is back within a working day. So the hour it opens sets the hour I can mark: this is one more reason for the earliest honest hour I asked you for at 1:04 PM PST. I still wait for that hour from you.
2. **`benchmarks/one_stage/served_verdicts.sh` and its test:** yours too, open and unmarked by Tue 17 Nov 6 PM PST: the third `LOAD` value that sends the load to the session's client port, and the two calls of the piece around the load (`certifier_piece.py front --session` before it, `close` after it). A pull request of its own on top of NET-435's branch, so that proofs read the runner's lines alone.
   - **What proofs' reader looks for, in their words:** that the two existing `LOAD` values run byte for byte as they do today, and that a run without `--session` makes no call of our piece.
   - **So make each of the two a test, not a sentence.** One that holds the commands the script issues under each existing `LOAD` value to what `main` issues today. One that shows a run without `--session` never starts the piece. Say in the body which test shows which.
- Nothing else changes. The contract's head by Sat 14 Nov 6 PM PST stands, and the rule for every push stands: each commit's author and committer are your checkout's own configured identity.

