[2026-11-11T09:33Z] 4689245e2c5a
library to training, Wed 1:34 AM PST. TRN-104's head is pushed: cursor/training-muon-shampoo-whole-6e3e at 4bcb698ecbfa (192f7b6d8bf the step, d382efeac21 the two entries, then #8840's 062c072e6e7b merged in, no conflict; #8840's Goal line is added). PORT can stack on it. One README commit follows the table; the step will not change.

subset's new dials, each default the step as it was, VERSION still 2:
- gain_lr, embed_lr (0: lr): the gains' and the embedding's own rate under any optimizer. lr is the matrices' (--muon-lr, --mat-lr); gain_lr is --lr or --norm-lr; embed_lr is --embed-lr.
- embed = "adamw" | "opt": under muon or shampoo the embedding trains with the gains under the AdamW beside it (undecayed), or is one of the optimizer's matrices.
- schedule = "linear" (today's ramp, hold, last `decay` share linear) | "cosine" | "constant".

One thing you did not ask for and should rule on: config.py digested every dial with its default filled in, so ANY new dial moved every subset config's digest. I added one clause to the route contract: a route may name dials in LATER, each one's default the step without it, and a config digests such a dial only where it sets it away from its default. Checked by script on one thread: all 11 existing entries' digests and their tiny-shape output models are bit-identical before and after; a test pins the nine dials a subset config digests.

fit.md: no row passes cosine; 3 pass constant (shallow/emaconst, class C by --ema, which route ema now takes with it). Seen and not carried: E in bf16 (hybrid --store bf16, sparse norms: 2 B-dial rows), --arms (3 rows), and in class C --embed-wd (4) and --gain-opt sgd (1).

Running at 4bcb698ecbfa: table r20261111-093115-8bbe, suites r20261111-093152-6716 (--changed 062c072e6e7b, slow tests in). The PR opens stacked on #8840 when both are in, draft, not marked.

