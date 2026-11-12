# For TREEFILE (bc-5aa10201-94d6-5942-ba85-c23ccd6dcad7), from top

## Thu 12 Nov 3:03 PM PST

Daniel answered at 3:01 PM PST ("Cool looks great. Can we get this implemented soon so I can share with people?").
Three things in your brief change from open or recommended to ruled:

1. The hardware file is `verity/device.py`, not `site.py`. Write `device.py` in the tree. Take the name out of the
   Open part. What stays open there is only whether a device with parts gets limits of its own in the Lean (owners
   answer on TOP-398 by Fri 13 Nov 10 AM PST).
2. `verity/settings.py` leaves `verity/` for `tools/config/` with `verity_config`: ruled, no longer a recommendation.
3. `verity/claims/` goes. It is a not-yet line: destination "deleted; its two readers (PoUS's `verifier/protocol.py`,
   spot's `device/challenger.py`) keep what they need", owner unowned until Fri 13 Nov noon PST.

Also: he wants it on `main` soon. Keep the PR small enough to read in one sitting, and open it as early as you can
today rather than at 6 PM; the metric commit can follow in a second PR if it slows you.
