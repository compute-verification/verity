[2026-11-11T15:53Z] befaafbadcd8
grade to training, TRN-117 (the trace's price), Wed 11 Nov 15:53Z. The 11 AM PST proposal. Each claim says ran, read or
inferred.

ONE LINE FOR ONECALC'S TRN-126: THE ROUTE CASTS ON A CPU TOO. File benchmarks/training_slowdown/training_slowdown/
routes/subset.py, function `step`: `with torch.autocast(x.device.type, dtype=torch.bfloat16):`, the argument
`enabled=x.device.type == "cuda"` removed. It is the only cast: `lora.step` is that function and `ema.step` calls it
(read). The capture wraps nothing. It is its own commit on cursor/training-trace-job-dtype-0cd4, pushed today.

1. R2, WHY THE ROUTE AND NOT THE TRACE (ran: both suites, 342 tests, and six tiny CPU jobs a variant: mlp, whole,
   block, lora, ema, muon, three steps each).
   the route casts (U):       13 tests fail, 14 with the capture's switch. 3 state the float32 program's trace
                              (test_run.py: the routes' step's rows, the update's width twice), 10 are an entry's steps
                              against a float32 reference (test_entries.py, all but ref-optim), 1 asserts the float32
                              `bmm` (test_record.py). All restated in the pull request.
                              A CPU job then RUNS the GPU job's program: every trained digest of the six jobs moves
                              (bits a byte in the sixth digit), and its trace and its count are of what it ran.
   the trace alone casts (T): 3 fail, 4 with the switch. A CPU job's trained digests are today's (ran: equal on all six
                              on one node) while its trace is of the bf16 program: a record whose trace is not its
                              run's, which is what your 7:25 (a) rules out.
   Under both: no stored record's config digest moves (a config's digest covers no code text; no route VERSION is
   bumped, since a bump makes `--retrace` refuse every stored record), and a GPU job is unchanged (`enabled` is true
   there today; read).
   Seen on the way: the same code's CPU job gives different trained digests on vy-nebius-cpu-1 and vy-nebius-cpu-3
   (ran). A CPU record's digests are a node's, today already. Not compared with a GPU: CPU nodes only.

2. THE SWITCH MOVES NO RESULT OF THE STEP (ran; your 6:20 ask). With and without it every checkpoint's trained digest
   is equal, all six jobs, under U and under T; the switch reads False before and after each job. Only the trace
   moves: attention's `bmm` rows read bfloat16/bfloat16 (forward 2, backward 4 in `whole`) where they read float32.

3. STORED RECORDS UNDER YOUR RULE 2.
   raw rows with a float32 operand on a forward or backward GEMM: ONE record found, the stored point's search record
     (capture 4.12.9, 52 rows; 7 rows, 34 of its 64 `bmm` products: forward 14, backward 20; ran, onecalc's count
     again). The stored point's other 7 records hold no trace.
   rows stored already priced: the scorer's ledger, 14,151 submission rows, 11,544 with a trace: every product row of
     every one carries its price and none keeps `dtypes` (ran: r20261111-151944-1b3f). The suite's four
     `.products.json` fixtures are priced rows too (read).
   not enumerated: other job records of training_slowdown in the evidence store (it has no index by format).
   THE CONTROL BUILD UNDER RULE 2 (read, not re-run): the build that gave the published tables cell for cell
     (r20261111-130100-9098, 12,965 checkpoints) runs no line my change touches. benchmarks/covert_witness on REPLOT's
     head names training_slowdown in one place, `te_core/gates_main.py`, which holds `gates.py` and `opnames.py` to
     main's by statement; I change neither. Its rows are priced already, so no rule of mine prices them. I re-run it
     against my branch's training_slowdown when the code is pushed, and send the comparison.
   THE STORED POINT: its search trace is capture 4.12.9, so the older capture's rule prices it as today. A test holds
     that, and I grade the point at #8923's head and at mine and send both files' digests.

4. WHAT THE PULL REQUEST HOLDS (as you ruled 6:20 and 7:03).
   a. capture/trace.py: the one place the capture forces the math backend sets the switch and puts it back as it was,
      also after a refused trace. Done, uncommitted.
   b. capture/products.py: CAPTURE 4.23.0. `gemm_dtype` leaves every signature. The older captures' rule is one
      function, keyed by the record's capture version: older than 4.23.0, a forward or backward GEMM row with operands
      wider than bfloat16 is priced at bfloat16; from 4.23.0 every row at its traced dtype. Its docstring says it goes
      with the rebuild that retraces the stored records. The row's dtype stays its first operand's here: onecalc's
      TRN-126 moves it to the result's.
   c. grade and the grader read captures 4.12.9 and 4.23.0 and refuse any other, as today's message does.
   d. R1's guard in `calculator.slowdown`, R3 (the library's row prints the dtype its circuit prices its products
      at), VERSION 4.23.0, and the body: every pinned digest and test that moves and why; that the switch acts on a
      CPU, with trace C; that the rule of b goes with the retrace; that the counter's assumed dtype is TRN-126's.

5. ONE QUESTION, with my default: THE GUARD'S REACH. Over the trace's own circuit (the table) both sides are mine:
   the guard refuses a trace whose rows the older rule re-priced, names the row, and holds by construction on a
   4.23.0 trace. Over a JOB'S count the other side is the counter, which has no dtype and prices by class. A strict
   row-by-row check there refuses every new lora record (its merge `mm` is float32 in the trace and bf16 in the
   count: 0.04% of |C|, and the bound does not move, ran) and shampoo's. DEFAULT: my guard checks the two prices my
   change owns and leaves counter against trace to TRN-126's (b). Say if you want the strict check in this one.

6. YOUR 7:03 ITEM 5, ONE LINE: of the eleven library entries measured at s-t20, only shampoo's printed job figure
   moves by over a per cent from the counter's assumed dtype (+11.44% of |C|, its optimizer's `mm`; the others at
   most +0.82%, lora's merge +0.04%, adafactor's `mm` +0.003%; ran). shampoo-whole inferred the same; muon's and
   fullmuon's products are bf16 by Muon's own cast, where the counter is right (read).

7. DATES. The pull request Thu 12 Nov 6 PM PST holds. TRN-117 Fri 13 Nov 6 PM PST holds: REPLOT's restated records
   came at 7:38 AM (377 runs; fe07, 8452, f0ed and 106f are not in the set). The six rows (8833@177, a207@168) wait
   for the capture, as you ruled.

