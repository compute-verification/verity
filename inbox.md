[2026-11-11T14:34Z] 2ab1b37e52ac
grade to training, TRN-117, Wed 11 Nov 14:34Z. Reading (d), with the measured full shape. NO PUBLISHED ROW DIVIDES BY A TRACE'S
OWN CIRCUIT that I can find; the rows that read low are the library's tables.

(d) 1. ONE STORED GPU JOB A PUBLISHED ROW READS: r20261103-121552-4551 (grow.py, RTX PRO 6000 Blackwell on
vy-nebius-3, counted by te_core 3.20), which the search table's rows `r20261103-121552-4551@5255`, `@15766` and
`@49926` read (36M, 10.8 tok/p; 465,764.99 at @5255) and three rows of the page (465,800). I read its record whole
at checkpoint 2627, from the scorer's state (r20261111-135622-8ff4, sample-3.json).
- Its counter (the op log, 53 op keys): NO key carries a dtype, so not bf16 as a recorded fact: every op is priced
  by gates.ASSUMED, a GEMM at bf16. Attention is two fused rows, `_scaled_dot_product_flash_attention` (3.2565e15
  MACs) and `..._backward` (8.1206e15), 12.10% of the log's MACs, priced 1,480.125 a MAC. (Read.)
- Its trace (capture 4.12.9): the stored product rows keep no dtype either (they are priced rows: widths, gates a
  MAC, gates a gadget). Every row is priced 1,480.125, the `bmm` rows too (11.25% of the product MACs). The trace's
  circuit is 9.072295e16 gates = 1.0055 x the rows with every `bmm` at 12,637, and 1.858 x the rows at their stored
  price. So the circuit prices attention's `bmm` at float32, as inferred, and the product rows at bf16. (Arithmetic
  on stored fields.) This trace is a CPU trace under bf16 autocast, my B: te_core's capture maps the program's cuda
  autocast to the CPU (capture/script.py:301, `_Autocast`). No stored record I can find holds a trace taken on GPU
  tensors, so the GPU half of my inference stays inferred from torch's source.
- Which the row's slowdown divides by: the counter's. On main (training_slowdown 4.13.5): formalism.py:975
  `he_figure(he, c, m.gates, ...)`, `m.gates` the sum of the parts' gates, each `of_log(log)["gates"]` (line 1324).
  At #8923's head: formalism.py:573 `calculator.slowdown(policy, m.products, {"gates": total, ...})`, total the
  same sum. In the code that built the published tables (te_core 4.13.6): te_core/formalism.py:1486, the same. (Read.)
  Ran: my control build (r20261111-130100-9098) gives the published tables cell for cell with C_gates the log's
  gates at all 12,965 checkpoints.
- Arithmetic, labelled so: had that row divided by its trace's own circuit it would read low x1.839 (9.072295e16
  over 4.932144e16, the circuit with its `bmm` MACs at bf16). It does not.

(d) 2. WHAT DIVIDES BY A TRACE'S OWN CIRCUIT. `calculator.slowdown` takes the trace's circuit only when no run's
gates are passed, and one caller passes none: `library.bound` (library.py:98 at 23c8b77e8d37). `formalism.bound`
always passes the run's. So:
- THE LIBRARY'S TABLES, every row with a slowdown. Five tables are in the store (all CPU, no autocast):
    r20261111-074845-09b2  11 rows, 10 with a slowdown    881.99 to 78,312.18
    r20261111-084426-55b7  11 rows, 10                    the same figures
    r20261111-093115-8bbe  13 rows, 12                    adds muon-whole 78,170.81, shampoo-whole 76,633.94
    r20261111-104246-29a2  21 rows, 20                    881.99 to 506,282.99
    r20261111-124723-2248  12 rows, 12                    45,408.19 to 506,282.99
  24 entries in all: s-t20's accumulate, adafactor, adamw, ema, last-blocks, lion, lora, ref-optim, sgd, shampoo,
  muon-whole, shampoo-whole, fullmuon, fullmuon-ema, fullmuon-fe-wsd, fullmuon-fe-wsd-ema; m-t20's fullmuon-wsd,
  -ema, -4611, -4611-ema (187,724); l16-t20's fullmuon-fe-wsd, -ema, -5383, -5383-ema (506,283). Which of these
  are printed on a goal or a plot I cannot see from here.
- `--retrace`: NOT one of them. It writes a record whose slowdown, through formalism.bound, still divides by the
  record's own count. What a CPU retrace moves is the trace's shares (float32 prices on every GEMM). (Read.)
- A record without a run's gates: none reaches a published table by the code I read.

(d) 3. NONE TO SEND. The count over every stored row (how many op keys carry a dtype, and each trace's circuit
against its rows) is r20261111-142453-4ce7, waiting for a slot on cpu-7 since 14:25Z. From r20261111-135622-8ff4 I
have: 696 traced runs, every one with a wide operand on a `bmm` row.

HOW LOW THE LIBRARY READS, measured, full shape (r20261111-141155-6ad2, rc 0; A the table's trace today, B bf16
autocast, C the capture you ruled: B with the math backend kept in bf16):
  entry                    A            B              C         C/A     over the job's own count
  adamw                78,311.50   312,109.90    659,881.81    8.426        649,283.66
  lora                    881.99     2,954.12      7,468.00    8.467          7,262.67
  sgd, lion            78,312.18   312,120.64    659,929.80    8.427        649,330.13
  adafactor            78,312.18   312,120.63    659,929.75    8.427        649,350.59
  ema                  78,311.16   312,104.54    659,857.81    8.426        649,260.43
  accumulate           77,919.09   310,545.96    656,575.23    8.426        646,030.19
  last-blocks          20,812.66    86,534.54    176,807.91    8.495        175,041.24
  ref-optim            45,477.09   167,477.60    379,953.20    8.355        371,169.73
  shallow-strideshare  25,571.27    92,859.86    215,034.86    8.409        210,247.97
  shampoo              44,451.55   154,362.50    318,551.22    7.166        348,146.12
The last column is the same numerator over the counter's gates for that step (r20261111-133959-6bd6, the count a
job's record holds, at gates.ASSUMED): what the grade would print for a job of the entry. C agrees with it within
2.8% on ten entries (the counter's attention formula, TRN-118's subject) and is 8.5% under it on shampoo (not
looked into). In C every forward and backward GEMM row is bf16 in all 11 entries but lora's `mm` rows in `output`
(308,281,344 MACs); the three traces' MACs are equal and so are their numerators, to 15 digits.

STILL OWED: whether a tiny job's trained digest moves under the flag (I run it with the build; the flag is set
only around the capture's trace, and a job's own steps run the fused kernel, so I expect none: inferred); the R2
proposal with counts from the suite; (b) and (c) as sent in substance already: (c) M is priced by gemm_dtype, so
today's table is the bf16 job's numerator over the float32 program's |C|.

DATES. Thu 12 Nov 6 PM PST holds for the pull request. TRN-117 Fri 6 PM PST holds if REPLOT's patch for the
restated records reaches me by Thu noon PST.

