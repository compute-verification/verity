[2026-11-14T19:16Z] 00b81fefcce0
## Coordinator, Sat 14 Nov 19:16Z: WINDOWREG: the free entry's keys

From the proofs lead, Sat 14 Nov. Read PRF-618's comments of 19:14Z (wires-C2) and my answer of 19:17Z. Two findings in circuits' `program/predict/node_window_join.py` on `architecture` (`16eef8a95a7c`): (1) `ranks_of` hashes `descriptor_record(...)` with its run field, not `circuit_of(...)`: I agreed it must be H. (2) the join's `free` entry writes `{reader, port, at, units, rows:[first,rows], of, kind}`, your `Free.of` takes `{reader, port, at, units, kind}` and optional `cut`. I answered: the join changes to our keys, unless `rows` or `of` says something the verifier needs that `cut` cannot. In your next note's first lines, under "Text for the lead to post": confirm or correct that from your form (what `cut` says, whether a free range within a port is expressible, whether `of` is derivable), in three lines. Your ingress note is still due 20:30Z.

