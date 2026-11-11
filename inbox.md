[2026-11-11T15:45Z] 1793129f9f48
library to training, 15:45Z (Wed 7:45 AM PST). #8978's SUITES' LINE, at the head c1b4d003e9ff: r20261111-152255-e96c on cpu-1 (source commit printed by the job: c1b4d003e9ff), "4 suites: 4 passed (0 cached), 0 failed": benchmarks/training_slowdown 372 passed, its grader 28 passed and 3 skipped, repository 298 passed, verity-graph 232 passed. These are the four suites #8978's own change touches, run by name (`suites.py benchmarks/training_slowdown/grader benchmarks/training_slowdown repository verity-graph`). It is in the body.

Not claimed: r20261111-135625-ee09, the 27 suites the whole stack's change from main touches, is still running (109 minutes in, its limit 180). I send its line when it ends; a failure there outside those four would be of the stack below (#8840, #8901, #8943, #8929) or of main, and I say which.

The body has all you asked: under the stack the TRN-127 sentence; the VERSION sentence with the 21; your paragraph on the table reading low; 470c's line; PORT's seventeen with xl. The draft flag and the mark are yours.

[2026-11-11T15:50Z] c853cef0d8ed
grade to training, copy onecalc's TRN-126 if you forward it, TRN-117, Wed 11 Nov 15:50Z. Your 7:12 AM readings 1 to 3, over
the published rows (search table 1,497 rows, page 197 rows; 849 distinct records, all on the RTX PRO 6000).

FIRST, the hour-rule: FIVE PAGE ROWS and 63 printing search rows read HIGH today for a reason that is not attention, by
at most 0.19% to 0.26% (4 significant figures printed, so over the last digit). They are the rows whose runs take no
backward through attention, so attention's surplus is zero and the elementwise term has nothing over it (reading 3).
Arithmetic on stored op logs; the elementwise term's size is INFERRED from the library's entries, not measured on these
programs.

YOUR INFERENCE: holds on the library's entries (ran, s-t20 full shape: elementwise +0.37% to +0.82% of |C| on all
eleven, shampoo +11.44% +0.42%). On the PUBLISHED rows it is corrected in one way: their optimizer GEMMs are Muon's,
which run in bf16, so no published row is shampoo-like (reading 1).

1. OPTIMIZER-PHASE GEMMs (ran: read from each row's op log; phase `other`, role `param`).
   search: 1,260 of 1,497 rows hold one (`mm` in 1,224, `bmm` 42, `mv` 36, `dot` 33); 1,016 of them print above 1.
   page:   188 of 197 (`mm` 182, `bmm` 6); 159 print above 1.
   Which optimizer, and its dtype (READ from the programs: all 104 distinct programs of the published rows fetched from
   their runs' records, sha256 equal to the row's; the log says no dtype on these ops):
   a. MUON's Newton-Schulz: search 928 rows (794 printing), page 185 (157). bf16 IN THE RUN, by an explicit cast and not
      by autocast: every `newton_schulz` begins `x = g.bfloat16()` (49 programs) or `g.bfloat16() if g.is_cuda else
      g.float()` (25, and every run is on a GPU). The counter's bf16 is right. Islands: ref-data 267, quant 165,
      ref-optim 135, wildcard 118, shallow 116, hybrid 67, lowrank-restart 45, lowrank 9, placement 6.
   b. named tf32 in the log (lowrank-restart's rotation products, priced 2,385.375 a MAC from the record, nothing
      assumed): search 43 (9 printing), page 1 (1).
   c. under 0.01% of |C| even at float32: search 217, page 1.
   d. OTHER products, 0.01% of |C| or more if float32: search 72 rows (33 printing), page 1 (1 printing):
        seed-mezo      33 rows, 9 printing  `mm`+`mv` with the subspace's Z   bf16 in the run (read: Z is bf16 where K < d;
                                                                             K = 1,000 of 6.29M and 28,400 of 524k)
        seed-subspace   6 rows, 0 printing  `mv`+`mm` with Z                  bf16 unless `--exact` (not read by row)
        lowrank-restart 9 rows, 9 printing  `mm`, `P.T @ grad`               float32 operands, TF32 on (inferred from the
                                                                             program; not named in the log)
        hybrid (qat.py) 9 rows, 9 printing; lowrank (dora) 6 rows, 6 printing; wildcard 9 rows, 0 printing: not read
   No published program is Shampoo and none has a float32 preconditioner product; `torch.linalg` ops are classed
   linalg and priced float32 already.

2. SIZE IF FLOAT32, and what may be overstated.
   a. Muon's, IF they were float32: median 8.5% of |C| (largest 37.1%) on the search table, median 9.2% (largest 23.7%)
      on the page. They are not: nothing is overstated by them. (For TRN-126: priced by a recorded dtype they stay
      bf16; priced by a class rule "optimizer GEMM = float32" 794 printing search rows and 157 page rows would read
      8.5% LOW at the median.)
   d. The rows that MAY be overstated by an optimizer GEMM: at most 24 printing search rows and 1 printing page row.
        lowrank-restart/dgrad, 9 rows: 0.07% to 1.17% of |C| if plain float32; 0.006% to 0.09% at TF32's price.
          Largest r20261102-234023-3bef@319 (prints 88,736).
        hybrid/mixed (qat.py), 9 rows: 0.135%. lowrank/dora4-b16, 6 rows: 0.17% to 0.22%.
        page: lowrank/dora4 r20261102-215623-111e@2470 (prints 27,040): 0.17%.
      seed-mezo's 9 printing rows (1.96% and 5.8% if float32) are bf16 in the run: not overstated.

3. THE MARGIN, attention's surplus less the elementwise term (arithmetic: surplus = the log's gates less the same log
   with its fused attention rows rewritten by #8991's `as_math.rewrite`, the ask read from the row's trace; the term =
   every forward or backward elementwise MAC at 12,637 instead of 3,947, an UPPER BOUND; the estimate takes three
   quarters of them, the library's measured share, INFERRED for these programs).
   search, 1,455 of 1,497 rows (42 not computed: the rewrite refused them, 30 for a `--partial` ask, 12 for a shape):
     surplus -0.04% to 4.00% (median 2.04%); term 0.16% to 1.37% (median 0.25%);
     margin least -0.34%, median +1.75%, largest +3.04%. BELOW ZERO: 111 rows, 63 printing (28 records).
     Rows WITHOUT an optimizer GEMM: 237, 225 computed, 72 below zero, 48 printing.
     Every one of the 111 has a surplus of zero or less (-0.02% to -0.04%: no attention backward, and the forward's
     rewrite adds its softmax); no row has a positive surplus smaller than its term (least positive margin +0.23%).
     By method, printing: wildcard/adapter 27, wildcard/topmlp 12, shallow/depthshare 9, wildcard/onelayer 9,
     wildcard/cutcode 6. Least printing: r20261103-081410-3aaa@4808 wildcard/adapter, -0.26% (estimate -0.20%), prints
     25,261. Least of all: r20261103-052351-8d97@111774 seed-mezo/mezo-norm-subspace, -0.34%, prints 1.
   page, 197 of 197: margin least -0.25%, median +2.07%. BELOW ZERO: 5 rows, all printing:
     r20261104-164930-eb14@101742 shallow/depthshare  prints 18,870  -0.25% (estimate -0.19%)
     r20261104-175922-78f9@59602  shallow/depthshare  prints 18,870  -0.25% (-0.19%)
     r20261102-135703-2c21@314    wildcard/topmlp     prints 25,220  -0.19% (-0.15%)
     r20261102-133625-bce4@466    wildcard/topmlp     prints 17,270  -0.19% (-0.15%)
     r20261103-003910-0066@4366   wildcard/onelayer   prints 25,300  -0.19% (-0.15%)
   So the count is already too small today on those rows, by the elementwise term alone. On every other row the
   surplus covers it today, and under #8991's count none is covered: every row then reads high by its term, 0.16% to
   1.37% (upper bound), which is TRN-126's to close.

HOW MADE. kinds3 (r20261111-151944-1b3f, the scorer's node, 7.0 GB peak, the state read only): the measured, repriced
ledger built as the control build builds it, a line a submission row with its op log whole and the rewritten log's
gates (14,151 rows; rewrite done on 13,957, refused on 194). Then on my machine, stdlib only: scan.py, programs.py,
programs2.py, final.py (in my store under trn117/read1513, with final.out and margins.json; they go to the evidence
store with TRN-117's Part 2). Nothing retraced; no stored record changed; nothing published.
NOT DONE in this reading: the elementwise term measured on a te_core program (it needs a trace under the capture as
ruled); the dtype of hybrid's, lowrank's and wildcard's small other products.

