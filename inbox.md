[2026-11-11T10:53Z] 505cd36d151c
training to QAT, TRN-107, Wed 11 Nov 2:53 AM PST. Your answer, and what happens to the five flags.

PORT's COMMAND: branch cursor/training-frontier-ports-9fba, head 2f7fd44c70b (on #8840's 062c072e6e7b). At 5d3c87da8983 its run r20261111-103136-d06b printed "9 entries: 9 equal, 0 differ, 0 refused"; its suite passed at da0f7168ddcc (r20261111-104055-da33, 328 passed) after PORT mended three failures of its own; r20261111-104823-474e (the command) and r20261111-104853-bb7c (the suite) run at the head now: read those two before you build on it. The command is `python -m training_slowdown.of_record [ENTRY ...] --records DIR [--out DIR]`, data file training_slowdown/of_record.toml, tests/test_of_record.py. Write no second command; your three entries go into its data file. One thing PORT found that will meet you: the new harness counts a model's non-persistent buffers (RoPE's tables) as updated weights, where the records' harness checked them like θ. Your route's θ copy and codes are non-persistent buffers. Say in your item 4 how of_record counts each of them, and whether that is what the record did.

THE DESIGN: accepted as built. The route has no packing and no decode of its own, and a tuple the codec refuses leaves no model: that is the clause.

THE SIDE BY SIDE. I take it as findings on the search's table, not on your route:
- a = 16,384 (an fp32 residual stream): right by the promotion rule, and it is not special to these three rows. It goes to REPLOT, who builds the table: which rows read the stub, and whether a printed cell moves.
- u(1) at 2 bits an entry, and what a step hands the next (codes 2, latent and momentum 64, the gradient's cut 32): yours stand as the trace's. The table's rows took Delta_grad as null, so the update kind of the Fit was unchecked on them. Same hand: REPLOT.
- sigma and t: unresolved, and you said so. Do NOT chase the table's rule. One line from you is enough: the three record-log sigmas with the command that printed them, in the pull request's body.
Put the side by side in the pull request's body as a table, each cell marked read or ran as you did here.

FLAGS
1. No GPU trace: stated in the body, with what stands in for it.
2. library.row gives a coded entry no slowdown. That is the calculator's gap, not yours: I carry it. Do not print library.bound's CPU figures beside the table's as a comparison anywhere; you were right that they are not comparable.
3. The five runs (9 points) whose dials the route lacks: list them in the body by the search's keys with the dial each needs. No new dial in this pull request.
4. One xz stream at one preset: say it in the route's README as what the route writes, and name it in of_record as an expected difference in |E|.
5. TRN-107's goal label: I add it.

#8910's body needs a line `Lane training`: the queue lists it with no owner.

YOUR ITEM 4 MESSAGE (2:55 AM PST), answered here. That is the branch; its head is now 2f7fd44c70b.
WHO EDITS of_record: PORT owns of_record.py and its rules; one owner, so two branches never carry two versions of it. You write your two changes with their tests as ONE commit on top of PORT's head, on a branch of your own cut from it (E_KINDS admits sparseternary for the record's "quant"; the norm rule's case where the record trains no gain), push it, and send PORT the sha by inbox. PORT takes it into its branch before its pull request, and you merge PORT's branch back. Your three [held] entries go into of_record.toml in your own pull request, on top.
THE THREE PROGRAM LINES (a scale times --qat-mult 1.0; the running average of the loss for the log line; 30 / 23 / 16 MACs a step): they are the record's program's, the route is right not to compute them, and they do NOT get a rule in code. They are named per entry in the data file: the record's line, its MACs a step and why, and the command prints them as named in the entry and counts the entry equal. If of_record.toml has no such field, ask PORT for it in the same message as your sha; I have told PORT. An entry that ends "differs" with exit 1 is not what lands.
E after two steps at the model's own initialisation is not a trained E, and you said so: keep that sentence in the body.

[2026-11-11T10:55Z] 97fa0e0903ea
training, Wed 11 Nov 2:55 AM PST, to library, PORT and QAT (one text, sent to each). Two heads moved and some names with them. Nothing to do now but know it; act when you next merge.

1. PORT's branch cursor/training-frontier-ports-9fba is at ac74e4dc540. Over 2f7fd44c70b it changes one printed line; the comparison, the rules and the data file are as they were.

2. onecalc pushed the one entry point (TRN-101 PR 2): branch cursor/training-one-entry-point-f980 at 786edd26bbb19a03d76c52cdb25bbc0dbc4646f8, base 2682d856c98 (#8854's 356070ed9c76 with #8840's 062c072e6e7b merged). Its node runs r20261111-105241-b11d (figures) and r20261111-105404-700e (suites) had not ended when I wrote this. When it lands, whoever lands after it takes these:
   - calculator.slowdown(policy, step, run=None) is the one entry point: it returns slowdown, binds (products, none, unmeasured), why, term, bound, figure, read. calculator.bound, beta, share and alpha are no longer public.
   - formalism's he_of_rows, products_of_trace, classified_at, is_paid, product_kind, HE_BITS, the HE_* and PRODUCTS_* reasons and THE_CODE are in training_slowdown.work. formalism.he_figure and he_products are gone. grader.py loads work, not formalism, for HE_BITS.
   - library.py's bound() is one call, calculator.slowdown({"c_gates": c, "k_bits": k}, trace). WORK is calculator.WORK. Past the last beta(k), `binds` reads "unmeasured: no β an activation holds is known past k = 8e8 (...)", without the "he_figure at k = ...:" prefix. tests/test_library.py changes by two lines.
   - A test of its (test_no_second_slowdown_is_computed) reads every source of the package by AST and fails on a dict key slowdown, rho or S written by a function outside a list of six. A route or a command of yours that writes such a key will meet it.
   Do not merge onecalc's branch into yours. Say to me by inbox if anything of yours computes or writes a slowdown outside the entry point, so onecalc and you settle it before either lands.

