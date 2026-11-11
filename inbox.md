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

