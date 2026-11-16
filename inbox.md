[2026-11-16T18:34Z] 438fdf83edfe
top, 18:34Z. One correction to your resume's rulings and one added last commit, both from Daniel's word at 10:29 AM PST today: the network subprotocol is **network attestation**, never "network certification" (its device is the network attester). My ruling 1 used the wrong word.

1. **Period entry, first use (replaces ruling 1):** "(for the certifier, T ticks, about 3.2 s)" becomes "(for network attestation, T ticks, about 3.2 s)", not "for network certification". The second use stays "for the network attester". Every other ruling of the resume stands.

2. **One last commit, after your move's commits and after proof (a)-(d) passes on them**, so the proof stays a proof of the move. In `GLOSSARY.md`, `ARCHITECTURE.md` and `AGENTS.md` only, the prose word "certification" becomes "attestation":
   - "network certification" -> "network attestation", keeping a capital ("**Network certification**" -> "**Network attestation**").
   - "certification period" -> "attestation period", and its article with it: "a certification period" -> "an attestation period", "a *certification period*" -> "an *attestation period*".
   - Count to check against: on `main` d4a407c0dbda the three files hold the word "certification" 13 times (ARCHITECTURE.md 8, GLOSSARY.md 3, AGENTS.md 2): 11 as "network certification", 2 as "certification period". After your move it is the same 13 across the files, and 0 after this commit. Put each in the PR body's edits table.
   - In the older-names table change right-hand columns only (the lines for "network accounting" for the subprotocol, "network transparency", and "window" for a span of time). Add no quoted phrase to a left-hand column: I add "network certification" and "certification period" as older names in a follow-up that also changes their other places, so the retired-terms lint's counts do not move in your PR.
   - Leave as they are: "network certificate" and "network certificates" (the record's name is my follow-up too), everything inside backticks, every path, Lean name and identifier.
   - Commit message: `GLOSSARY.md, ARCHITECTURE.md, AGENTS.md: network attestation, for "network certification" (Daniel, 16 Nov)`.
   - Rerun on the new head the named test files you ran for your other edits. If one goes red in a way only an allowlist change would cure, or anything here is unclear, leave this commit out and say so in your report: I then make it on your branch.

Your deadline does not move. If your PR is already open when you read this, push the corrected Period line and this commit to the same branch and say the new head in your report.

