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

## Thu 12 Nov 3:08 PM PST

Three more, from reading the tree against your brief:

1. The root also holds `artifacts.json` once `fixtures/` is gone (the not-yet line `fixtures/artifacts.json` -> the
   root). List it among the entries the tree has that `main` does not yet (top; Sun 15 Nov).
2. `tests/` and `pyproject.toml` are plain entries a package may have at every level the test checks, so
   `verity/protocols/tests` is in the tree, and so is `verity/tests/` (the tests of the flat files, a suite of its
   own, once they exist).
3. The dates moved: top's target is Monday 16 Nov 9 AM PST for the root and the top level of `verity/`, with every
   path-only piece run on Sunday 15 Nov evening (TOP-398, 3:05 PM PST). For top's own not-yet lines write
   Sun 15 Nov instead of the days in your brief; leave the other owners' days as the brief has them.

## Thu 12 Nov 3:28 PM PST

#9539 seen, thank you. **Daniel ruled at 3:25 PM PST: the path-only moves go tonight, the rest Saturday morning.** So
the days in the not-yet list change: the lines for `verity/configuration/` (config, deploy, wire), `verity/settings.py`
and the Site's file read Fri 13 Nov; every other line of top's reads Sat 14 Nov. Leave other owners' days as they
gave them. If the pull request is already final, make this one change, push, and end your turn naming the head.
