[2026-11-10T16:10Z] 4d67fb859b01
From the coordinator, Tue 10 Nov 8:10 AM PST. **The layout reaches `main` about five hours earlier than your brief says: at about 11:15 AM PST (19:15Z), if top's staging passes its check at about 10:30 AM PST.** `main` is e7b03e4a6c now (#8468, #8636 and #8648 of ours landed) and moves once more at about 9:15 AM PST; then the layout.

- **Your brief's rule holds as written:** once `git log origin/main` shows the commit with the `Layout-move:` line, never `git merge origin/main` into a branch that lacks it.
- **What you have in flight, finish.** A result on the old paths still says whether the content is right. Start no run longer than 30 minutes on the old paths after 10:30 AM PST.
- **When the layout is on `main`:** top posts the base commit and the two commands on TOP-394 (`git merge BASE`, then `restack.py`); read them there and use them.
  - **A branch of yours with no pull request, you cross yourself** (this replaces "you do not cross this turn" where your brief says it): `--dry-run --json` first, then `--push-to` a new branch, and go on there. Tell me the new name by inbox. Then run again on the moved tree what you ran before. Lock records are regenerated there (`restack.py` leaves locks stale by design), so a record you commit before the move is evidence, not the final one.
  - **A pull request's branch (#8537) is crossed by another worker and pushed by me.** CARRYLOCK: do not push a crossed head to it; tell me your head when the layout lands.
  - **SCANLEAN:** unchanged, you follow #8539's branch and tell me when it holds the move.
  - **LAIDCOUNT:** your scratch only moves path: `Security/` becomes `security/` on the moved tree.
  - **TIMEPAIR:** the window's two runs measure the commit they were sent from, so the move does not touch them. Cross after the second run is scored.
- Your dates stay. If the crossing costs you one, say so in your report with the hours it cost.

