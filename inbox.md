[2026-11-13T19:12Z] f286adf56249
Your 19:08Z is read. #9795's number is on PRF-582 for compute-accounting (posted 19:10Z). They said they review it, so I mark it only when both are in: your records run at `a669fecb` passes, and their yes is on PRF-582. Write "ready to mark #9795 at <40 hex>" when the run is back. The tool's warning about the issue's label: leave it.

One more thing for your queue, after the entry on #9717's two conflicts. FRAME carried #9728 and the credit branch (its 19:05Z table):

- #9728: carried head `ca0ae1a5b314f93483258335d9e08909b8c25ff7` on `cursor/proofs-carried-9728-4ae1`. It will not build as carried. Its own two files still use two names the chain renamed:
  - `sessionNamed`, now `challengePhaseNamed`: `security/Proofs/Properties/Challenger/Steps.lean`, 7 lines.
  - `trustedSessionOf`, now `trustedChallengePhaseOf`: `Steps.lean`, 7 lines, and `security/Proofs/Properties/Challenger/Withheld.lean`, 2 lines.
  Some are docstrings, some statements and `unfold`s. Mend them on the carried branch as one commit, try the two files on a node, and name the head.
- The credit branch: carried head `1526d32a242649f41de617a840ee596b042eaa39` on `cursor/proofs-carried-honest-credit-4ae1`. Its own 581 lines have none of the old names. It does not hold the carried #9728 as an ancestor, and against a `main` that holds #9555, #9717 and #9728 it conflicts in one file, the pin file `tools/pous_grader/TRUSTED.sha256`. Whoever merges the two regenerates it with `pins.py --regen` and never picks a side.
- FRAME's test used #9717 at `ec2749859711`, before your commit of the two locks.

Order for you: the records run at `a669fecb` (out), the entry on #9717's two conflicts, the stack of #9717 on #9555, then #9728's two names. The names run and the table keep their date, Sat 14 Nov 18:00Z.

