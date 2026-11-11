[2026-11-11T12:32Z] bd670bcc079b
grade, TRN-102, 2026-11-11T12:32:48Z: r20261102-204410-f0ed retraced with the fixed recorder. Positions null again; its published rows stand.

Run r20261111-105829-b845 (vy-nebius-cpu-7, 96 GB declared, peak 57.1 GB, rc 0): f0ed traced with the recorder of commit 12e82e884b25 under both rules (te_core 4.13.5 and 4.13.6), then the tables rebuilt.

- Ran, and read in the records: 17 of 17 rows of each rule's record have positions, context_fwd_bits and context_bwd_bits null; data_dependent is empty. The two rules' records differ in version strings only.
- Ran: against the stored record the published layer reads (te_core 4.12.11, as run r20261103-213702-0693 wrote it), each retraced record differs only in te_core, written_at, written_ts and the keys the newer capture adds (capture_code, he). No traced figure moved.
- Ran: the rebuild with nothing changed is the published table for these rows, and under both rules f0ed's rows equal the published row in every cell: the overlay's @1406 at gain 1.1 (rho 96362.25, served 1770.59, own 96362.25), and the search table's @1406 at gain 1.1 and @3515 at gains 2 and 11 (98878.94 / 1816.83 / 98878.94). No other run's row differs from the control.
- Read in the trace's log: three recordings (the base size, the halved size, the base size again). At the base size the traced step, 1120, is in the program's stage 2 (stages of 536, 583, 639 steps); at the halved size the stages are 1116 and 1204 steps long and the recording ends at step 1117, in stage 1.
- Inferred, not run: the two sizes record different stages of the program (other blocks trained), so their positions can't be matched and the trace keeps windows only. It is the program's own schedule, not the recorder: the fixed recorder gives the same record.

r20261103-140155-106f (run r20261111-105852-439e, 112 GB declared) is still tracing; its answer follows.

Files: /cursor/stores/bc-b43da88b-5d25-4463-9b8c-da4df6f84207/trn102/cells/f0ed-overlay.txt, f0ed.txt; trn102/held-f0ed-records.json.

