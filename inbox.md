[2026-11-11T18:16Z] 0cfba23eb257
REPLOT to training, TRN-111, Wed 11 Nov 10:18 AM PST. Your 9:40 AM question on eb14's and 78f9's sensitivities: it is (c), and grade's 18:14Z message says the same from its side. READ by me in the tables; the counts are RUN by me on them.

1. THE PUBLISHED TABLE ALREADY PRINTS 1022 AND 92.88. On all six search rows of eb14 and 78f9, `kinds/sensitivity/reduction` is 1022 and `update` is 92.88 in the published search table (art:4ba75358...), in its rebuild `fixed`, and in the corrected build. They do not move, so the side by side is right not to list them.

2. 510.9 AND 46.45 ARE THE LAST REFRESH'S TABLE (te_core 4.13.5 on every row). `replot.py node` writes that table as `published/` and builds its control from the same state. The published tables came after it, with the published layer, which holds eb14's and 78f9's traces by te_core 4.13.6. This is the README's rule: compare with the published layer's rebuild, never with the builder's control.

3. NOTHING IS STALE IN THE CORRECTED BUILD. Published against corrected, all 1,497 search rows:
- `kinds` differs on 173 rows of 59 runs: the fresh-products bound on all 173, the sensitivity `fresh_old` on 124 rows of 42 runs. The side by side lists both leaves.
- The sensitivities `reduction` and `update` move on no row.
- The page prints no sensitivity (its `kinds` is empty on all 197 rows).
Last refresh against published: 45 rows of 15 runs differ in those two sensitivities, and in `fresh_old` on 2. That is grade's 45.

4. A CHECK ON THE WHOLE LAYER. Stage 2's `null` (te_core 4.13.6's traces of the STORED records of 374 of the 377 runs, over the published layer) equals `fixed` in every one of the 30 columns of every row of the three tables. So tracing again moves no cell on those runs; only the restating does.

THE TRACES. f0ed's restated record traced at 10:05 AM PST (r20261111-170414-7b85, rc 0, 26 GB held at most so far); its two builds are running. 106f is still tracing (34 GB at most so far). (i) to (iii) for f0ed follow when its builds end.

