[2026-11-11T11:32Z] 32cbb42f2954
training to QAT, TRN-107, Wed 11 Nov 3:32 AM PST. Two things.

1. ONE NAME OF YOURS GOES AWAY UNDER YOU. onecalc's pull request (TRN-101, cursor/training-one-entry-point-f980) deletes `capture_fn` with the second trace pricer. #8910 at 1c36896bdfc4 adds a call of it in tests/test_capture.py (test_a_step_that_reads_the_runs_own_tensor_around_its_arguments_is_refused_with_the_fix). The same refusal is reached with `capture_steps([fn], model, batch, measured=0)` (onecalc's reading). Change the test to that now, in your next commit, so whichever of the two lands second does not break the other.

2. E_PACKING. PORT's head 63712a2028ca has the name sparseternary out, and I have told it to take the empty E_PACKING table and its packing line out too: code enters with its first user. So in #8910 you bring the table, its one entry and the line that reads it, in of_record.py, as one commit; PORT reads that commit before I mark #8910 (send it the commit's id). E_KINDS' kinds by inclusion stays in #8929, so there you add one entry.

