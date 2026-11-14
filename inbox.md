[2026-11-14T09:42Z] bbd6faeff5dc
SCANCOUNT to network-accounting (lead), Sat 14 Nov 1:42 AM PST. Your 1:33 AM PST line is read whole.

**Do not mark #9921 at `7f4b5c45b620`: one proof of it fails. Its head is now `60643bdbebc62126d566a79ef091082a79918f75`, mended and not yet tried. The statement and `Site.linkEnd` are the same at both heads, so your read of them holds.**

**1. #9921's tries** [ran, at the first cut `c5b9aa06690c`]
- `r20261114-081925-7399`, the guarantee's file: passes. The definitions build there, `Placed.onto` and the statement among them. `7f4b5c45b620` changed `Site.linkEnd` to read `host?`, and no run has built that yet.
- `r20261114-081925-4222`, `security/Proofs/Core/SiteSubtree.lean`: fails. One theorem of four, `Site.placesIn_under`: the `simp` left a subtree's membership as it was, and the case split then met a list. It is the lemma behind "limits add", not the guarantee's proof. That file is the same at `7f4b5c45b620`.
- **The mend,** two commits: `Site.mem_under` states a subtree's membership once, and `placesIn_under` is proved from it and `mem_placesIn` as terms, with no `simp` and no rewrite of the goal [argued; not run].
- **Tries at `60643bdbebc6`:** that file and `security/Proofs/Accounting/Actual/RegisteredSite.lean`. Launched 1:36 AM PST; ids in my next line.
- **Withdrawn by me** (`research cancel`, each still waiting, none had started): `r20261114-081123-d7b2` and `r20261114-090145-0e40` on #9921, `r20261114-090837-9e68` and `r20261114-090837-b80a` on the leak scratch. Each builds that file first and could not pass.
- The records run and the suites follow the first green try of both files. The line is still about 80 minutes.

**2. The honest strategy has a gate at a host** [written; not run]. Scratch head `e1792f6d8711`.
- I added one beside the certifier's and did not move it. Moving was the smaller proof, but it leaves the certifier with no gate, and `RecordsWithin` would then be shown of an empty record.
- The cost was small: the toy's circuit now takes its gates' leaves as a parameter, in place of one leaf. No definition more. The honest strategy has gate 0 at `cert0` and gate 1 at `node0`; the nontrivial one keeps its single gate at the outside.
- The docstring says "its gates at a host and at the certifier's place, none at a device or a memory leaf".
- The scratch holds #9921's mend. Its two tries are launched again at that head.

**3. The page on B and C** follows in my next line, within the half hour.

TIMER: none

