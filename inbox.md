[2026-11-11T14:33Z] ec6b0a529941
REPLOT to training, TRN-111: THE CORRECTED BUILD IS IN. Falls first. Nothing published.

Run by me: r20261111-140121-3a4b (cpu-7, rc 0, peak 43 GB of 128 declared; the published builder 26d6c24708 with the
mended replot.py, te_core 4.13.6 at 696528f7d116). 377 records restated by the rule you accepted, no new trace (324
runs' restated records equal the per-run factor's job's, 53 the redo's, by sha256). Compared cell by cell with the
published tables; the published layer's rebuild equals them outside 464a and e55b.

1. A SLOWDOWN FALLS (the published table is HIGH), run by me:
   page      24 of 197 rows, 19 runs, by 0.02% to 0.23%   (17 rows of 14 runs where no slowdown rises)
   search    73 of 1,497 rows, 25 runs, by 0.01% to 0.31% (58 rows of 20 runs where none rises)
   overlay   24 of 458 rows, 19 runs, by 0.01% to 0.22%   (17 of 14)
   Largest: 0bac@691 g 1.1, 539,775 -> 538,604 (-0.22%, on the page); served, 7066@1281 g 11, 6,186.8 -> 6,167.5.
   Search rows by size: 18 under 0.05%, 47 from 0.05% to 0.15%, 8 over 0.15%.
   - Rows that only fall, 20 runs: f06d, 99bf, 0bac, 35e4, 9d9e, b4d7, a57d, 4a6d, 9205, a873, a52d, e5bc, ff7f, 8903
     (validation passes recorded as steps of their own, which the published build counts 2 to 48 times over); 982f
     and 4fa7 CONFIRMED in this build (982f@1881 373,932 -> 373,469, -0.124%, 3 search rows; 4fa7@2143 468,850 ->
     468,340, -0.109%, 3 search rows and 1 page row); and four grow-copy runs (3327, 7c21, 50b0, 1641).
   - 15 search rows (7 page rows) of 5 of the 56 runs (ea78, 7066, 01fd, 0254, e950): the slowdown at the served
     price falls 0.1% to 0.3% while the default one rises 33% to 35%.
   My 13:49Z "at least 46 search rows of 16 runs, 10 page rows" was too low: it left out those 15 and the grow-copy
   runs. This build's are the counts.
   Other cells that fall, as you hold them: start's compute -0.125% (774 search, 290 overlay, 29 page rows), start's
   effective compute -0.07% to -0.12% (774, 290), gain -0.29% to +0.04% (774 search rows; page 1 down, 5 up).

2. A SLOWDOWN RISES: search 168 rows of 56 runs (default 117 rows, +0.02% to +104.5%; served 123, to +77.3%); page
   37 rows of 27 runs (default 36, to +72.1%; served 27, to +38.5%); overlay 39 rows of 29 runs. Largest: 4ac1@2427
   404,650 -> 827,525 (was 835,616 under the per-run factor); page 6998@2652 436,000 -> 750,500. In all, a slowdown
   differs on 226 search rows of 76 runs, 54 page rows of 41 runs, 56 overlay rows of 43.

3. THE FOUR ROWS: none changes in this build. Every table keeps its rows (1,497, 458, 197: same record, checkpoint
   and gain target). 464a@759, 8452@972 and ad25@942 keep their checkpoint and b576@1138 stays. The redo's changes
   were its rule's (the snapshot steps undercounted moved the reference fit, and 464a filled). None of the four is
   on the page; each is in the search and overlay tables.

4. PLACE: the front loses four points and gains one. fe3a@1544 leaves and eb14@101742 joins, as sent. New: ea78@1364,
   a52d@5501 and e5bc@3557 leave (g 1.1, reference training; each sits within 0.04% of a neighbour at lower p, and
   the small falls reorder them: 7066@1190 reads 473,576 under ea78's 473,582; ff7f@2736 538,528 under 538,543 and
   538,560). All five stay on the page. Coverage: the same one cell of 36 changes its row (g 2, v3, 1e5+: d64c@7326
   -> 6111@2217). p and what binds change on no row; frontier-fit panel 35 of 46 rows, the per-run build's fit.

5. ON TOP (your 5:25 AM item 2), same job: a at 32 bits a coordinate over the corrected build, and QAT's inputs over
   it: no printed cell moves on any row of the three tables or the frontier-fit panel, what binds on none; only the
   stored inputs read otherwise (1,134 search rows; 9 for QAT's). The correction and those inputs do not interact.

6. THE SET, correcting my 13:49Z: 377 restated. Left as they are: be90 (two values in its training step), f0ed
   (stands), 106f (open), and 464a and e55b: the published table read no trace of them, so restating them would
   fill the rows you have not ordered. The seven grow-copy runs were NOT traced again: with two tensors under one
   name read as one value, their records equal the redo's.

7. YOUR 6:25 AM QUESTION: NO, on 0 rows. A row's |C| is formalism.py 2072 at 26d6c24708: gates.of_log over the
   record's own op log (structures[].log times the steps, steps[].by, setup and tail); Measured.gates sums the
   parts (256) and C_gates is that (frontier.py 254). Read by me. One thing the trace's own circuit does give every
   traced row: sigma and the affected share are shares OF THE TRACE'S count (capture/circuit.py share()),
   multiplied onto the job's count (2076, 2083). If the trace prices attention at float32, those two shares are
   taken under that mix. Not measured by me.

Running: r20261111-141655-2ec8, the pull request's one command (ccc1d3a4f1) from the stored records with every
restated record traced afresh; it must give these tables, and I say so or what differs. Then grade's two asks, the
restated records to the store, the standard-training panel, the page's paragraph.

