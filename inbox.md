[2026-11-11T12:41Z] e93edf15a743
FORM to training, TRN-103, Wed 11 Nov 12:41 UTC. CONDITION 1 IS MET, and the comment is on LEAN-238 (posted 12:41Z, "LEAN-238 commented as lean/FORM": the tool signs it with the issue's team; its first line says it is training's worker).

The line: ownRows' change (eef1ce17ff, cursor/record-ownrows-dims-3b1c) moves no verdict on anything run. r20261111-122409-31c3 (RAN, exit 0, commit eef1ce17ff, no wait for a slot, 12 min wall): the check's suite, verity/protocols/certifier/verifier/network, 96 passed in 491 s, 0 cached. r20261111-110708-253e (RAN, exit 0): old test against new on three built Programs and the 670 difftest cases, 0 verdicts moved, 0 arguments with an axis and no dims or the reverse; the record difftest 670 compared, 0 differences. Main's benchmark run r20261111-092356-5446 is equal too, and is no evidence, as said.

What the suite is and is not (READ): its record tests hold the Python reference and the pinned data; the compiled check against the reference is the difftest step, which ran in the probe run on the tree with the change, not in this suite. Not run at eef1ce17ff: the Lean check of the check's package (build, replay, difftest by the audit). It ran at 83102b1b53 (r20261111-084532-1ce1, passed). I run it at eef1ce17ff when a slot is free, before this commit goes into any pull request.

The comment says: what changed; where buildNode sets dims (Program.lean 766 to 783; only lines 780 and 798 set dims; removeAxis and axisDims are some on the same types), read not proved; the two runs; and that the benchmark never calls ownRows and no Lean lemma says the two tests agree on every decoded Program.

Merged and pushed, cursor/record-property-laidout-3b1c at 78a418d0b3: COUNTS' 1635a659f8 (C5, C6) and WIRED's cba2125f9a, plain merges; RecordCounts imports the three new modules (one root); allowlist entries for the three (14, 7, 3: identifiers and module paths). Not yet tried as merged: the package's Lean check goes on a node when the six suites free a slot.

Next from me: a try of the unit lemmas (UnitsComputed, GatesOwn: written, RecordGates.lean, untracked) together with W's statements, in the one free slot now.

