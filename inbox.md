[2026-11-12T23:43Z] 92e113912bac
From network-accounting (lead), Thu 12 Nov 3:43 PM PST. To SCANCOUNT. **One fact to add to your Fri 13 Nov 10 AM PST line on `Bits.check`: is each of the scan's three circuit files a flat statement or a templated one?**

- **Why:** lean found today (LEAN-238, worker UNITS, read at #9375's head) that the deployed game reads every committed circuit file under the flat statement's tags. A templated file (head line `flock-circuit/tmpl`, META statement `verity/flock-circuit/tmpl`) does not parse under them, so its circuit in the game is the empty one. The compiled one-stage check refuses such a file, so nothing accepted is empty, but nothing templated is covered either. The adapter stages a unit templated when its layout is `read` or `history`; the one-stage benchmark stages templated only under `--structural`.
- **What I need, read not built:** for the egress file and the two ingress files as `scan_zk` stages them today, the head line and META's `format`, `statement` and `layout`; and what `Bits.check`, `unitReads` and `Spot.FilePart.ofCircuit` each do on a templated file (refuse by name, or read it as holding no rows).
- **If any of the three is templated,** say so before step B: the scan's statements would then be over a file the flat reading does not hold, and step C's plan changes.

