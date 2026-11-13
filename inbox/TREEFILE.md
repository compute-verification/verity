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

## Thu 12 Nov 3:53 PM PST: #9539 read whole at `05040ddec2`. One change to the test, the lines' owners and days, then I mark

This replaces the 3:28 PM note's days. Make these, push, launch the node run at the new head, and end your turn
naming the head and the run.

**A. Your item 4: a landed move warns, it does not fail.** Add
`test_every_not_yet_line_and_coming_entry_is_still_to_come` to `tests/conftest.py`'s `BOOKKEEPING` tuple for
`tests/test_tree.py`, beside `test_no_line_is_still_arriving`, and say so in the test's docstring, the file's header
and the body. Why: six owners land moves between tonight and Saturday noon, the whole check runs after landing, and
a red root suite reverts its culprit after an hour. A move must not be reverted for a line of this file. The stray
test stays a failure: that one is the guard. Top deletes landed lines, and the metric counts them.

**B. The tree's coming entries** (day: a date, with a time where it isn't 6 PM):
- `verity/party.py`, `verity/channel.py`, `verity/device.py`, `verity/protocol.py`, `verity/config.py`,
  `verity/runtime/`: owner `top`, day `2026-11-13 08:00 PST`. Tonight's move brings the wire's `__init__.py` to
  `verity/channel.py` whole; memory-accounting's later split is inside files and is no line of this file.
- `artifacts.json`: `top`, `2026-11-14 12:00 PST`.
- `verity/protocols/compose/`: `proofs`, `2026-11-14 12:00 PST` (it joins Friday 7 PM's chain).
- `hardware/`: as you have it.

**C. The not-yet lines:**
- `verity/configuration/config/`, `verity/configuration/deploy/`, `verity/settings.py`: `top`,
  `2026-11-13 08:00 PST`.
- `verity/configuration/wire/`: `top`, `2026-11-13 08:00 PST`, to
  `verity/channel.py, and its handshake to verity/runtime/`; drop its note.
- `verity/hardware/site/`: `top`, `2026-11-13 08:00 PST`, to `verity/device.py`.
- `verity/hardware/registration.py`, `verity/hardware/record.py`, `verity/protocols/reporting.py`,
  `verity/definitions/`, `verity/difftest/`, `verity/composite/`, `fixtures/artifacts.json`: `top`,
  `2026-11-14 12:00 PST`. Give `verity/definitions/` the note
  `its served part waits for the vLLM adapter's snapshot and may land 2026-11-16`.
- `verity/claims/`: owner `top`, day `2026-11-14`; drop its note.
- `verity/protocols/remote/`, `verity/protocols/onsite/`: `proofs`, `2026-11-14 12:00 PST`; in `to`, drop the
  clause about the path being named by a date.
- `verity/hardware/cards/__init__.py`, `verity/hardware/nodes/__init__.py`: `compute-research`,
  `2026-11-14 22:00 PST`; drop the note.
- `verity/protocols/inference_only/nci/`: to `benchmarks/nci/`, owner `network-accounting`,
  `2026-11-14 12:00 PST`. `verity/protocols/inference_only/rho/`: to `benchmarks/rho/`, the same.
- `verity/protocols/flock/pod/`, `flock/sigma/`, `flock/agreement/`, `flock/tool.py`, `spot/candidates.py`,
  `spot/costfit.py`, `spot/optimize.py`, `spot/plancheck.py`: owner `proofs`, day `2026-11-18`, to
  `its protocol's verifier/, prover/ or device/, or out of verity/; proofs says which`.
- `verity/hardware/world/` and `verity/periods/` stay unowned.
- So `UNOWNED_MAX = 2`, and the body's counts and its list of unowned lines change with it. `NOT_YET_MAX` stays.

**D. Your other items:** 1 stays as you wrote it (I ask lean once its merge is on `main`). 2, 3, 5, 6, 7 and 8:
taken as you made them.

## Thu 12 Nov 3:57 PM PST: three days change again (lean answered at 3:51 PM)

lean takes a cut on Friday 7 PM PST for the path-only maps, landing Saturday by noon, but a map that moves a Lake
package goes with the chain cut Sunday 8 AM PST (landing Sunday afternoon). So, over the 3:53 PM note:
- `verity/difftest/`: day `2026-11-15`, note `its Lake package moves with Sunday's chain`.
- `verity/definitions/`: day `2026-11-15`, note
  `two Lake packages move with Sunday's chain; its served part waits for the vLLM adapter's snapshot and may land 2026-11-16`.
- `verity/protocols/compose/` (coming), `verity/protocols/remote/` and `verity/protocols/onsite/`: `proofs`,
  `2026-11-15` (the composition package is a Lake package too).
Everything else stands as the 3:53 PM note has it.

## Thu 12 Nov 4:00 PM PST: the last two unowned lines get owners, and the folds a day

- `verity/hardware/world/`: owner `proofs` (the handles registry gives it to proofs), day `2026-11-18`, to
  `where proofs says; a Lake package the certifier's device and the memory challenger require`.
- `verity/periods/` (arrives with #9375): owner `top`, day `2026-11-15`, to `verity/protocol.py`, note
  `the Python mirror of CoreProtocol.Period, folded by hand with reporting.py`.
- `verity/hardware/registration.py`, `verity/hardware/record.py`, `verity/protocols/reporting.py`: these are folds
  by hand, not moves, so day `2026-11-15` (over the 3:53 PM note's Saturday).
- So `UNOWNED_MAX = 0`: every line has an owner and a day, and the body says so in its first paragraph.
