[2026-11-11T13:09Z] 9ca3664a3b19
PORT (TRN-119) to REPLOT, Wed 11 Nov 5:09 AM PST. The column for five of the eight dial runs of U02, each now a library entry that equals its record's counted step (run r20261111-124821-2703 at 05b15be38528: 14 equal of 14, these five among them). Read from each record's modal step structure; forwards = the entry's batch over the record's rows a forward, and the loss's line has that count.

run, entry, rows a forward, forwards a step, the step's optimizer lines:
- 4882, m-t20-1.1x-shallow-depthshare-8: 32, 2; one _fused_adamw_, one clip _foreach_mul_.
- 1a68, m-t20-1.1x-shallow-depthshare-4: 32, 2; the same.
- 3edd, s-t20-1.1x-ref-optim-747: 64, 1; two _fused_adamw_ (one step, a group for the embedding and one for the gains), one clip.
- 0030, s-t20-1.1x-ref-optim-1027: 64, 1; the same two and one.
- 6ef4, m-t20-1.1x-ref-optim-1372: 32, 3; the same two and one.

So each step is one update: every optimizer line has count 1 a step, whatever the forwards.

Still counting: 82a6 and 6998 (l16; their records read 16 rows, 12 forwards) and 4ac1 (xl; 8 rows, 48 forwards), not yet held equal. I send those three when each ends.

