[2026-11-12T23:53Z] 81f4cbc0904f
From your coordinator (memory-accounting), Thu 12 Nov 3:53 PM PST (23:53Z) (the same text is in your inbox). **SIGNEDRUN: two points make my order of 3:50 PM PST exact. Item 2 as I wrote it would have left `verdict`'s count with nothing to count; this replaces its second bullet. And the certificate gets a second gap line.**

1. **Each round's opening is a row and its path, and `verdict` judges the answer against the row.** `evidence.paths` holds, for round `j`, `{row, path}`: the 64-byte row digest of block `rounds[j].index` and its Merkle path. The check, where `Run.checkAt` reads the table today (lines 88-92 and `expected`, 77-78):
   - there is one opening for each round, else refuse;
   - for each round, `Merkle.verifyPath pub.root q.blocks r.index (rowLeaf pub.salt (r.index / q.blocksPerSegment) (r.index % q.blocksPerSegment) row) path`, else refuse with a reason of its own that names the round (the opening is not a row under the root the setup published);
   - `expected i` is the row of the first round whose index is `i`, and none for an index no round asks;
   - then `verdict schemes trusted expected r.signed sc r.openings`, unchanged. So an answer that is not its block is counted as `wrong` and refused in today's words, and a bad opening is refused in its own. Do not build the leaf from the signed `H(block)`: that folds the two refusals into one and makes `expected` vacuous.
   - Test both refusals, and two rounds that draw one index.
2. **Decision 5: two gap lines, in this order, verbatim.** The first as in my 3:50 PM PST order. The second: `the memory challenger's host saw the whole encoded store at setup`. `certified` stays false while either stands. Daniel is asked whether the device may hold the store (an ask page I file by Fri 13 Nov 10 AM PST); the line goes when he says yes.

