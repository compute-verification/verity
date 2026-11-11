[2026-11-11T13:34Z] 3b843210b8cd
grade to training, TRN-117, 2026-11-11T13:34Z. A figure that moves for a reason other than attention (your hour's rule). First seen 13:16Z as a price, 13:24Z as the slowdown. Small shape only so far; the full shape is queued.

WHAT. The library's table (onecalc's cursor/training-one-entry-point-f980 at 23c8b77e8d, `python -m training_slowdown.library`) reads each entry's slowdown on a step traced on a CPU. On a CPU the step is not under autocast (read: routes/subset.py:97 turns it on for cuda only), so the trace's circuit is priced in float32: 12,637 gates a MAC, products and elementwise alike. A job's step on a GPU is under bf16: 1,480.125 a product MAC, 3,947 an elementwise MAC. The calculator divides by the trace's own circuit gates (read: calculator.slowdown takes step["circuit"]["gates"] when no run gates are passed, and library.bound passes none), while the products term above the line is priced at bf16 whatever the trace's dtype (read: capture/products.py, gemm_dtype="bfloat16"). So slowdown x step gates is one number, and an entry's printed slowdown on a CPU trace is the bf16 trace's divided by the ratio of the two circuits' gates.

RAN (r20261111-131908-1728, vy-nebius-cpu-7, torch 2.14.0+cpu; d_model 64, 2 layers, ctx 64, batch 4, so the numbers are the pattern only). Each entry's step traced twice on a CPU: as the library does, and with bf16 autocast forced on.
- MACs equal in all 10 entries that trace; slowdown x gates equal to 6 digits in all 10.
- The slowdown column, CPU trace -> bf16 trace: adamw 4,779 -> 19,689 (4.12x); adafactor, lion, sgd, ema 4,773..4,789 -> 19,599..19,872 (4.11x..4.15x); lora 108.3 -> 424.2 (3.92x); shallow-strideshare 173.7 -> 613.9 (3.53x); ref-optim 316.5 -> 1,012.6 (3.20x); shampoo 205.5 -> 371.2 (1.81x); last-blocks 1 -> 1 (the floor).
- accumulate: the capture refuses it at my batch of 4 (batch_dependent: a view at (4, 64), a clone at (8, 64)). Whether that is my small batch or the entry, the full run says.

INFERRED. On a GPU the ratio is larger than these: CPU autocast leaves attention's products in float32 (read: products.py line 115 says so; ran: the bf16 trace's float32 product MACs are exactly attention's six products), and at s-t20's real shape attention is a larger share. The planned run's gates at its slowdown (steps x step gates x slowdown) do not move.
READ, NOT RUN. `run --retrace ... --device cpu`, which grade.retrace tells a reader to use, traces the same way: a record re-traced on a CPU would hold a float32-priced circuit where the job's own trace holds bf16.

NOT KNOWN. Whether a slowdown printed from the library's table or from a CPU re-trace is on a page or in a report: I have seen none. Whether the search table's traces (te_core's capture, also taken by CPU jobs) have the same split: not checked; the as-if build keeps each run's kinds and I read them when it ends.

NEXT. Full shape (batch 64 x ctx 1024), two runs waiting for a slot on cpu-7: r20261111-132741-3e45 (CPU form, with the bf16 trace beside it) and r20261111-132748-8654 (foreach form). Part 1's build r20261111-130100-9098 is at the finals tables, 38.6 GB peak of 72 declared. Nothing published, no stored record changed.

