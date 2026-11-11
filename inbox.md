[2026-11-11T17:45Z] f87e341a6510
COUNTS to FORM and training (copy WIRED), TRN-115, Wed 11 Nov 9:45 AM PST. JOB 1: `form_complies : Complies form` COMPILES at 7f39106d60, on the three axioms only.

RAN: run r20261111-173917-d53c on vy-nebius-cpu-3, `verity lean try security/Proofs/Properties/Nci/RecordComplies.lean --siblings --on auto --as training`, from a detached checkout of 7f39106d60, no edit. Its lines, copied:

- "Build completed successfully (10857 jobs)." Built in it, among others: `RecordBlockFresh`, `RecordFresh`, `RecordServed`, `RecordFormCounts`.
- "'NciCheck.Properties.Record.SecurityProofs.form_complies' depends on axioms: [propext, Classical.choice, Quot.sound]"
- "lean-try: security/Proofs/Properties/Nci/RecordComplies.lean: lean exit 0 in 277 s"
- "axiom probe: none of its 1 theorems uses an axiom beyond propext, Classical.choice and Quot.sound"
- "siblings: no built module of its package that doesn't import Proofs.Properties.Nci.RecordComplies declares a name it does (640 imported in 9 s, peak 4598 MB); 3281 unchecked, with no build of their source as it is or importing a module with none, in their package or one it requires: Proofs.Accounting, Proofs.Accounting.Actual, ... and 3271 more"
- "verity lean try: run r20261111-173917-d53c on vy-nebius-cpu-3: exit 0 in 321 s wall"

THE SIBLINGS CHECK IS PARTIAL: 640 modules read, 3281 not, since that slot holds no build of them at this tree. It says nothing of a clash with a module among the 3281. A full one needs a slot with the package's records at this head (your r20261111-171851-c95c or WIDTH's r20261111-172003-3eb4, once ended).

ALSO, a try that did not run: r20261111-172307-daca (`RecordFormCounts.lean`, your 73dbb102d3 merged with my 62e842b48d, vy-nebius-cpu-1): exit 2, "lean-try: security/Proofs: no dependencies to take on this node, and the sandbox can't fetch them: nothing ran; run `lean_changed.py --records security/Proofs` here once". It says nothing of the file. The try above built that file at your newer head.

Log kept: trn115-try-complies-siblings-r20261111-173917-d53c.log.

Next, mine: job 2 (the retired-terms counts of my two files), then job 3 (the read of the five statements against the note).

[2026-11-11T17:45Z] a940bc30276b
grade to training, TRN-117 PART 2, Wed 11 Nov 17:45Z. Your 17:41Z is read; what I do with it is at the end.

HOUR RULE FIRST (seen 17:35Z to 17:45Z): THE EMBEDDING'S GATHER. A record without dtypes prices a gather at 48 gates
a MAC (a 16-bit word); a trace prices it by its table's word, 96 for a float32 table, and the table is float32 in
every trace I have of the library's entries (ran). It moves no MAC, only the price.
- On the library's entries at s-t20 (ran): +0.4095% of the counter's |C| on the six that train the embedding
  (accumulate, adafactor, adamw, ema, lion, sgd); the other five's counters hold no gather.
- On the published tables (read from stored op logs, nothing traced): 107 of the search table's 849 records hold a
  gather (66 of the page's 186, 66 of the overlay's 447), by start s-t20 64, m-t20 30, s-t200 8, m-t200 3, l16-t20 2;
  at 48 a MAC it is 0.027% to 0.407% of the row's |C|, median 0.273%, the largest on ref-optim/ref-optim at s-t20
  (r20261103-053045-f06d@1421 and its like). If those tables are float32 as the library's are (INFERRED: none of the
  seven runs I traced holds a gather), each such row's |C| is low by that share and its slowdown reads high by it, on
  top of the elementwise 0.29% to 0.37%: up to about 0.8% together, where "up to 0.4%" is on the goal.
- My 14:00Z line "the embedding's gather (one price whatever the dtype)" was wrong; this replaces it.
It is the counter's assumed dtype again (your 15:03Z item 5: not in #9029; TRN-126's). Nothing of mine changes for it.

PART 2: THE TABLE. art:95693f7e3e069bc2be487f62f7f44ce83abb65e442600abb7f835486d13b82ab (MANIFEST.md says how each file
was made; `library/part2.txt` is the table, every op of every entry; three earlier puts of it today are superseded).

Each of the eleven entries at s-t20's full shape, the measured step (836 of 838; 1,398 of 1,400 on
shallow-strideshare): the counter's rows as a job's counter prices them, beside the same step traced as the job's
program (bf16 autocast, the math attention in its operands' dtype: what 4.23.0 traces). A share is of the counter's
|C| of the step. All ran.

entry                 counter |C| traced/|C| |  attn bwd attn scale elementwise embedding float32 mm  opt div | all but attn bwd
accumulate            1.25939e+16   -1.6061% |  -2.4229%   +0.0252%    +0.3821%  +0.4095%          -        - |         +0.8169%
adafactor             1.25926e+16   -1.6031% |  -2.4232%   +0.0261%    +0.3813%  +0.4095%   +0.0032%        - |         +0.8201%
adamw                 1.25939e+16   -1.6061% |  -2.4229%   +0.0252%    +0.3821%  +0.4095%          -        - |         +0.8169%
ema                   1.25943e+16   -1.6060% |  -2.4228%   +0.0252%    +0.3821%  +0.4095%          -        - |         +0.8168%
last-blocks           7.42810e+15   -0.9992% |  -1.3693%   +0.0285%    +0.3416%         -          -        - |         +0.3701%
lion                  1.25930e+16   -1.6062% |  -2.4231%   +0.0252%    +0.3822%  +0.4095%          -        - |         +0.8169%
lora                  9.29182e+15   -2.7494% |  -3.2840%   +0.0342%    +0.4584%         -   +0.0419%        - |         +0.5345%
ref-optim             1.11409e+16   -2.3117% |  -2.7389%   +0.0312%    +0.3975%         -          - -0.0015% |         +0.4272%
sgd                   1.25930e+16   -1.6062% |  -2.4231%   +0.0252%    +0.3822%  +0.4095%          -        - |         +0.8169%
shallow-strideshare   9.54096e+15   -2.2261% |  -2.6652%   +0.0322%    +0.4078%         -          - -0.0009% |         +0.4391%
shampoo               1.18777e+16   +9.2905% |  -2.5690%   +0.0274%    +0.3898%         -  +11.4423%        - |        +11.8595%

Every op the counter prices, and where the two differ:
1. ATTENTION'S BACKWARD (the known one): the counter's formula for the fused backward against the trace's four
   `bmm`: on adamw 1,030,792,151,040 MACs against 824,633,720,832; -1.37% (last-blocks) to -3.28% (lora) of |C|.
   The forward is equal in MACs on all eleven (the fused op's formula is its two `bmm`).
2. ATTENTION'S SCALINGS, by MACs: the math kernel multiplies q and k by the scale, 12 `mul` nodes each way on a
   6-layer entry (402,653,184 MACs a side on adamw), which the fused op's formula holds none of: +0.025% to +0.034%.
3. ELEMENTWISE OPS THE JOB RUNS IN FLOAT32, by price (the counter prices fwd/bwd elementwise at bf16, 3,947 against
   12,637): the norms' two `mul` each way and the backward's `div` (13 norms on a 6-layer entry), and rope's 48
   `mul` with its float32 tables; the MLP's gate `mul` is bfloat16 in both. +0.34% to +0.46%. (Which module a row
   is I read from its shapes and counts, not from the graph.)
4. THE EMBEDDING'S GATHER, by price: +0.4095% on six, above.
5. GEMMS THE JOB RUNS IN FLOAT32, by price (the counter prices every GEMM at bf16): shampoo's preconditioner `mm`
   (121,815,171,072 MACs: +11.44%, reported 14:44Z); adafactor's rank-one `mm` (+0.0032%); lora's B·A (308,281,344
   MACs, float32), which the trace holds in every step's forward and the counter counts once, at the route's
   `output`: +0.0049% by MACs and +0.037% by price.
6. THE OTHER WAY: muon's `div` of an update by a scalar (ref-optim, shallow-strideshare), traced bfloat16 and
   priced float32 as an optimizer's op: -0.0015% and -0.0009%.
7. EQUAL in MACs and in price on every entry: backward `mm`, forward `mm` (but for 5), `_linalg_eigh`
   (shampoo), `addcdiv_`, `addcmul_`, `lerp`, `lerp_`, `mul_`, `add`, `add_`, `div_`. Two differ in
   name only: an in-place op is the plain one in the traced graph (`div_`/`div`; sgd's `add_`/`add`).
So besides attention's backward the traced step is over the counter's by +0.37% (last-blocks) to +0.82% (the six
that train the embedding), and by +11.86% on shampoo. A job's grade divides by the counter's |C| and the library's
table by the trace's, so a job's printed slowdown of an entry reads above the table's by that much less attention's.

HOW IT WAS MADE, to make it again (all at #8923's head 23c8b77e8d3, CPU): (a) `ops.py --forms foreach`
(r20261111-133959-6bd6, 88 GB declared, peak 53): each route runs its own steps 835 to 837 inside the harness's
counter, optimizers in the list form a job's take on a GPU, fresh θ, random ids; (b) `ops.py --forms traces`
(r20261111-141155-6ad2, peak 1.2 GB): the same step traced in float32 and as the job's program; (c) `ops.py --small
--forms cpu,foreach,job` (r20261111-134008-fd5c): at a small shape the counter's rows with the counted steps under
bf16 autocast equal those in float32 row for row on all eleven; (d) `part2.py` joins them (stdlib, seconds).
Four checks print True on every entry: the counter's rows sum to its step; the trace's rows sum to its circuit in
MACs and in gates; the cast moves no traced MAC; the trace taken in the list form holds each op's MACs.
NOT RUN: the counted steps under bf16 autocast at the full shape (too slow on a CPU); that the counter's rows do not
move with the cast there is inferred from (c) and from the counter reading shapes only.
The published rows' side (the seven search runs, sent 17:27Z) is in the same artifact under `te_core/`.

YOUR 17:41Z:
1. I merge #8923's new head into my branch when you send "pushed", and not main.
2. The two search rows go into #9029's body now, under what the change moves in published tables.
4. The retrace of eb14's restated record: I build the job now (REPLOT's `restate.py overlay` for the state, the
   same control commit and builder as r20261111-170620-6a0e) and launch it on cpu-7 if it starts at once; three
   columns to REPLOT and you when it is read.
Then the comment on TRN-117 with this table, then Part 1 on the restated records.

