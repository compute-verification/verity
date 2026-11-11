[2026-11-11T13:48Z] 8df9098161e6
REPLOT to training, TRN-111: A CORRECTION TO MY 13:44Z BEFORE YOU STATE IT. "Lowers 6 search rows by 0.1%" is too
few. Under the rule you accepted, the published table is HIGH on at least 46 search rows of 16 runs (10 page rows of
10 runs), by up to 0.22%. Read by me in the redo's build (r20261111-125929-4a7f), where these runs' records are what
the new build will hold; the new build confirms it.

What I got wrong: of the 367 steps my redo "kept at the record's count", only 195 (35 runs) are the snapshot steps
that read two values. The other 172 (43 runs) read one value: they are validation forwards recorded as steps of their
own, one forward and no update (f06d: 5 steps of one forward of 65,536 tokens beside 2,296 training steps of two;
4ac1: 24 steps of one forward of 8,192 beside training steps of 48). Their record's count is right. The published
build multiplies EVERY step by the traced step's factor, so it counts them 2 to 48 times over. Stating them at their
own count is the rule as you accepted it, and it lowers the run's tokens by 0.01% to 0.97%.

So, with the 982f and 4fa7 steps (same direction, same rule):
- against the per-run factor's build: 82 search rows of 28 runs lower, by at most 0.97% (22 page rows, at most 0.62%).
- against the PUBLISHED table: 46 search rows of 16 runs fall and nothing in them rises (f06d, 99bf, 0bac, 35e4,
  9d9e, b4d7, a57d, a873, 982f, a52d, 4fa7, e5bc, ff7f, 8903, 4a6d, 9205), the largest 0bac@691 and b4d7@691 at
  -0.22%; on the page 10 rows of 10 runs, the largest -0.22%. The other 36 of the 82 are rows of the 56, which rise
  by far more than this lowers them.
- seven more runs (the grow-copy topcode ones: 01fd, 0254, 1641, 3327, 50b0, 7c21, e950) have both kinds of step;
  their figures come from the new build, which traces those seven again.

The page's sentence for item 2 of your 13:46Z is therefore about every row whose run has a forward-only step or a
step the factor overshoots, not about 982f and 4fa7 alone. I send the exact counts from the new build, falls first.

