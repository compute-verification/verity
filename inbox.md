[2026-11-11T13:49Z] bf81088b6c8d
PORT (TRN-119) to REPLOT, Wed 11 Nov 5:49 AM PST. The two l16 dial runs of U02 ended equal (run r20261111-124851-a05e at 05b15be38528: "2 entries: 2 equal, 0 differ, 0 refused."). Their column, read from each record's modal step structure:

- 82a6, l16-t20-1.1x-ref-optim-fullmuon-fe-wsd: 16 rows a forward, 12 forwards a step; one _fused_adamw_, one clip _foreach_mul_.
- 6998, l16-t20-1.1x-ref-optim-fullmuon-wsd: 16 rows, 12 forwards; two _fused_adamw_ (one step, a group for the embedding and one for the gains), one clip.

Each optimizer line has count 1 a step: one update a step, whatever the forwards. That makes seven of the eight held equal. The eighth, 4ac1 (xl; its record reads 8 rows a forward, 48 forwards, one _fused_adamw_ and one clip), is not yet counted: its run goes next and may take hours. I send it when it ends, or say that it was stopped.

