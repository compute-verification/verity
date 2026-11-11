[2026-11-11T18:13Z] b05074e56e57
kernels lead, Wed 18:13Z: #8989's fixes are read (nothing to change), but it is not marked: #9034 is marked and will land, and #8989's made-up rows hold cores without numa_nodes and say nine in four places, so its tests would fail in a candidate holding both. After the pairs summary: restack #8989 onto #9034's head c9a5a45a33a1, give the rows numa_nodes, make the four nines ten, run its suites, say the head; I mark it then. Then #8976's README merge. The newest sections of inbox/KSTEADY.md have both.

