[2026-11-12T18:44Z] cbeeab79e37a
training to grade, Thu 12 Nov 10:44 AM PST. Two things for your 1 PM PST report; change nothing for either now.

1. **The grader's `tests/conftest.py`.** TRN-144's worker is now STEP (a new worker; SEAT handed over). Its line makes a step at a `cc` wire the fused attention op forward and backward, which holds only under deterministic algorithms and at the captured shapes, so the grader's cc fixture needs two stand-in shapes and `torch.use_deterministic_algorithms(True)`. I told STEP to make that change as its own commit on `cursor/trn144-step-fused-attention-2556`. The file is yours: say in your report if it should be done another way.
2. **The masked scores.** BACKWARD's Definitions are in (seven, branch `cursor/fa2-train-backward-d232` at `cb20f4171f`). At T 1,024, D 64 a head's circuit holds S for the visible pairs only, 33,587,200 multiply-adds, where the card multiplies whole tiles, 37,748,736, and then masks; the other four products hold the 72 tiles the kernel visits, 37,748,736 each, against 67,108,864 for a full rectangle. One line in your report: which of these the count takes today, and what #9374 does about it.

Also for you: Daniel's rule on models is in force for this lane (Opus does the work, Fable plans). Finish #9374's report on the model you have; what follows it goes to a new worker, so end that report with what a successor needs.

