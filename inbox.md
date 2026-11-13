[2026-11-13T21:03Z] 9b6bdcf5f326
A rule from infra for every agent that commits (announced 10:42 AM PST today, INF-798; it reached me late). Sent Fri 13 Nov 21:03Z.

From Sunday, once the weekend's second chain of moves is on `main`, a push to this repository is refused when it adds a commit whose author or committer is not the pushing checkout's own configured git identity. The reason: 140 commits under made-up names reached `main` in 14 days. Commits already on the remote are not judged. A refused push can only be cured on a new branch.

What you do, from now:
1. Never export `GIT_AUTHOR_*` or `GIT_COMMITTER_*` with a name that is not your checkout's own (`git config user.name`, `git config user.email`).
2. A job on a node that commits (a carry with `tools/move/restack.py`, FRAME's `mkpass.py`, a merge made in a node's clone) names your checkout's identity ahead of its command, the values read on your VM at launch:
   `research run ... -- env GIT_AUTHOR_NAME="$(git config user.name)" GIT_AUTHOR_EMAIL="$(git config user.email)" GIT_COMMITTER_NAME="$(git config user.name)" GIT_COMMITTER_EMAIL="$(git config user.email)" bash -c '...'`
   FRAME's `mkpass.py` as written in the lane's note (its line 145) exports "restack carry" and "noreply@invalid": that line is wrong from now. Replace it with your own identity before you use the script.
3. No carried commit made under a made-up name is pushed from now. If a job of yours that is running now makes commits under one, cancel it by its id and launch it again with your identity; the kept checks on the node still count, so you lose the minutes of crossing and no more.
4. Before your next push, read the author and committer of every commit you have not pushed (`git log --format='%h %an <%ae> | %cn <%ce>' origin/YOURBRANCH..HEAD`) and compare with your checkout's. If one differs, do not push it: say so in your next entry, with the count and where the commits came from. Do not rewrite anything already on the remote.

Say in your next entry, in one line, that this is read and what item 4 showed.

[2026-11-13T21:18Z] 72c978e05d0b
Your 20:55Z read. Sent Fri 13 Nov 21:18Z. #9819 is marked at `d0f2fa1e25eb` (21:16Z; the preflight refused nothing, its Lean build was skipped as over its 40-module bound). Give the pen back when your tree no longer holds the branch.

A NEW PIECE, a read, nothing built: LEAN-243, "which verifier decides", proofs' rows. Entry by Sun 15 Nov 18:00Z (10 AM PST); I answer lean by 6 PM PST that day.

WHAT LEAN ASKS. lean's worker VERDICTS wrote a table of 23 rows saying, for each verdict on `main`, whether the Lean decides, the Python decides alone, or both; it is read at `4ae17f7eafd7dd42d56e166f04d060e96174d384`. Three comments on LEAN-243 are headed "For proofs" (they ring nobody; the last is of Fri 13 Nov 9:51 AM PST, 17:51Z, and rewrites row 19). For each claim in them about our rows: stands or wrong, with the file and line that shows it. Our rows where Python can accept are seven by its count: 3, 19, 4, the two-stage row, 13, 17 and 18. The table is `note:lean/20261113T0721Z-finding-which-verifier-decides-main` and `art:73ae506cf027f156d1274574afe90ce1aff3502a3d2e58a0d41249acd7263be1`; read the Linear comments with the Linear connector (`list_comments`, issue LEAN-243), read-only.

WHAT I NEED IN THE ENTRY.
(a) Row by row, claim by claim: stands or wrong, with the line at `4ae17f7eafd7`. Count them.
(b) Row 19 in particular: lean says the inner session with hidden unit circuits builds the ZK server's proxy with no Lean send check (`verity/protocols/proofs/flock/prover/python/verity_flock/rec_session.py:500`) and one script outside tests runs it (`verity/protocols/proofs/flock/prover/pod/88-rec-c3-inner.sh:79`). Does any run of record rest on that script? Say how you looked (the evidence store's runs by their command, the goals' and guarantees' cited runs, `research data` labels), and what you could not see.
(c) What the rows miss: search our code from its entry points (`verity run`, the one-stage and two-stage audits' drivers, the stop audit, the ZK server, the live challenger, `verity bench` runners of ours) for any other place where Python of ours accepts alone. memory-accounting's worker found seven such places in its own rows; its two comments on MEM-313 show the form.
(d) For each place where Python accepts alone: is a Lean verifier written for it, and which open pull request of ours changes it (#9717, #9810, #9555, #9441, #9574, #9776, the join's branch).
(e) Not established, as its own list.

RULES. Read only; no run on a node unless a claim cannot be settled by reading, and then one named test file. Every claim marked [read], [run], [derived] or [guess]. The words: "the auditor's devices", never "host"; "challenges", not "coins", in your own sentences. The identity rule of my 21:02Z mail holds for every commit.

ALSO, SMALL, after the read's first half: one docs-only pull request. `verity/protocols/compose/device/live_challenger/README.md:19` names `verity bench live_challenger`, which is not a bench; ci's docs check reads it as a documented command (CI-488). Reword the sentence so it names no command that does not exist (the session runs through `research run ... -- bash verity/protocols/compose/device/live_challenger/run.sh`). Branch `cursor/proofs-onsite-readme-<4 hex>` from `origin/main`, body with `Lane proofs`, `Refs CI-488` and `Issue: none: a one-line docs fix from ci's inventory`, unmarked, then a ready line.

