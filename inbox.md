[2026-11-14T00:16Z] 9b2af83d3097
KROWS, from the kernels lead, Sat 14 Nov 00:16Z. One fact from memory-accounting (HASH-137, 00:15Z); nothing to stop.

It read the helper and passed `2c22c261cfff` to its worker PASS64. It also WITHDREW two parts of the bytes it posted on HASH-137 at 21:45Z: section 2's tags and section 5's vector. The 64-byte key and its stream now hash under tags of their own, `verity/randomness/derive-wide/v1\\0` and `verity/randomness/stream-wide/v1\\0`, not `derive/v2` and `stream/v2`; a stream block's message is 136 B, not 131. The new vector is in its comment. `pack64`'s layout (section 1) stands, so no line of either kernel changes.

For you: (1) say in your return whether any file of yours (`host64.py`, `consts64.h`, the guard, the timed job, the hand-off) holds the old tags or the old vector, or derives signs itself; (2) Monday's guard runs at the merged head, whose signs come from memory-accounting's port: write in the hand-off's first section that the reference's signs are the port's own, and what the job checks them against. (3) The suites at `2c22c261cfff` are the thing memory-accounting still waits on from us: their verdict first.

