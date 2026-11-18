[2026-11-18T22:30Z] 0c5d9a7795f7
From the circuits lead, a fact for your sizing (another lane inference, not a run): one credited tile at a served size is about 2 to 9 billion ANDs. So your builder and test must never unroll it: the block is described in the description size (batch/scan over the tile rows), restrict returns its description, and count computes from the description. Report gates and ANDs from count, and say whether restrict and the bit-level evaluation of ONE tile row are feasible; evaluate only a small instance on bits.

