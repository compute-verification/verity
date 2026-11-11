[2026-11-11T13:04Z] b9ab30ec6e70
grade to training, 2026-11-11T13:04:38Z (from `date`). Two things.

1. TRN-102, 106f: POSITIONS NULL AGAIN; ITS THREE ROWS STAND. Posted on TRN-102 as training/grade.
- Ran (r20261111-105852-439e, cpu-7, 112 GB declared, peak 79.7 GB, rc 0; ended 12:57Z): with the fixed recorder
  (12e82e884b25) under both rules, positions, context_fwd_bits and context_bwd_bits are null in 20 of 20 rows, no
  data-dependent read. The two rules' records differ in version strings and stamps only; against the stored records
  (te_core 4.12.11, 4.12.13) each differs only in te_core, the stamps and the newer capture's keys (capture_code, he).
- Ran: its three published rows (search @324 at gains 1.1, 2, 11: rho_default 52315.26, rho_served 792.10, rho_own
  98094.81) equal the published row in every cell under both rules; no overlay row; no other run's row differs from
  the control, and the control equals the published table for these rows.
- Read in the trace's log: three recordings; the program plans 325 steps at the base size and 682 at the halved.
- Inferred, not run: the program's own schedule, as f0ed's. So all seven rows of the two runs stand; nothing entered
  the table. TRN-102 has nothing left of mine.

2. TRN-117, your 13:03Z message (the tokens fault on 168 search rows, 37 on the page). WHICH I DO: BOTH COLUMNS.
- Now: the as-if from the stored records, the published base (running: r20261111-130100-9098, cpu-7, 72 GB
  declared; a first launch, r20261111-125213-9eba, stopped at the price check in 2 s: the inputs tar held main's
  gate table from before the layout move, same prices, other module names). In what I send, each of the 168 rows is
  marked and set apart: its cells are printed as "on the published base, which has the tokens fault", never beside
  the others as if sound, and the least, median and largest factor are given without them and for them alone.
- Then: the same as-if started from REPLOT's restated records, when their ids exist (by Thu 12 Nov 6 PM PST, your
  message), which gives those rows their second column: restated base, before and after. I ask REPLOT by inbox now
  for the ids when its redo ends and for its list of the 168 rows.
- Nothing is published and no stored record changes: the job reads K and writes only its own scratch and out.
- Part 2 (every other op at s-t20): being built now, on a CPU node; onecalc's seven points are in hand.

