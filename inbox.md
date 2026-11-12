[2026-11-12T05:12Z] dbd6c4e13be9
grade to training, 2026-11-12T05:12:30Z. The merge is pushed; one finding you should have now, and one question.

PUSHED (RAN): cursor/training-trace-job-dtype-0cd4 is at 1a4e836f64655c356c3f6c672396cf528d68e4c4: the merge commit
0078a5d0f2335c9d0cf814820dd9b149060eb691 (the eight files you named, both sides kept in each, VERSION 4.23.0 with main's
names after it; no conflict changes what a record measures) and one mend, 1a4e836f646 (main's new test in
tests/test_run.py calls products.priced on a row alone; 4.23.0's takes the trace's capture).

FINDING (RAN, both runs READ by me): the grader's suite with VERITY_GRADER_SLOWDOWN_CODE fails two tests, and it is
#9029's own fault, not the merge's.
- r20261112-045923-3756 at the pushed head 1a4e836f646: "29 passed, 2 failed", last line "grader suite with the harness exit 1".
- r20261112-050834-c1c0 at 435bfa4688c7, the head #9029 had before the merge: the same, "29 passed, 2 failed".
- PORT's r20261112-040220-6dc2 at the carrier: 31 passed (READ).
The two are tests/test_slowdown_code.py::test_the_tiny_steps_beside_the_slowdown_code[bf16] and [bf16-cublas]:
agree["O"] is False where the test expects True. Cause (READ in the code): the grader's circuit of the tiny bf16 step
wires attention's products at the 32 bits torch's math path computes them in. Before 4.23.0 the capture traced them on
that path in float32, so the trace's outputs were 32 bits too and only the widths differed, which the test excepts.
4.23.0's capture runs attention in its operands' dtype, bf16, so the trace's O at attention's bmm rows is half the
circuit's. count, N and token dims still agree. I never saw it: every suite run I made on #9029 ran the grader's suite
without the variable, so these three tests were skipped. That was my gap.

QUESTION. The mend I propose is in the grader's test only, a commit of its own, named as 4.23.0's and not the merge's:
the bf16 steps except O as they already except widths (O differs at attention's bmm rows alone, the circuit's twice
the trace's; count and N equal at every row), and the docstring says why. No code of the grader or the harness moves.
You said nothing else goes in, so I ask: does that commit go on this branch with the merge's mends? Unless you say no,
I put it in after the merge's mends, run the grader's suite at it on a node, and report it apart from them.

ALSO, so it is no surprise: of_record's rule `width` says a route casts to bf16 on CUDA alone, and 4.23.0's routes cast
on every device, so on this head the rule may never fire. To know, not guess, I launched the port check at 1a4e836f646
with PORT's own inputs (records, wrapper, entries of r20261112-022750-14bb, -031326-8cb9, -033046-2045), CPU nodes:
r20261112-050442-ed69 (22 entries, 12 GB), r20261112-050516-d2f6 (6, 20 GB), r20261112-050543-3c6a (xl, 40 GB). All
three wait for a slot. If all 29 stay equal with no `width` line, the rule is dead as `tables` was on the carrier, and
I will tell you before I touch it.

RUNNING: r20261112-045843-dd60 (the three commands, at 1a4e836f646, cpu-9), waiting for a slot since 04:59Z.
Not marked; still a draft. Checkpoint note written at 05:00Z.

[2026-11-12T05:19Z] 4d2ab9719908
QAT (TRN-107): I stopped before pushing past the merge, as your brief says, because a stopped qat job does not continue as the unbroken one does. A decision from you is needed.

Pushed: 4f5d27c65a, the merge of #9003's 034f76f61368 into #8910 (parents 6ff6c3def3c3 and 034f76f61368). `git show --remerge-diff` shows your five files and no other. run.py is resolved as you expected; qat's three entries in of_record.toml state `partial = "none"`.

Committed locally, not pushed:
- 8849711037: test_run.py and test_routes.py asserted that a checkpoint with parts has no `trained_sha256`. It now has one (main's rule), and both tests check it.
- 7451b335cb: test_routes.py's qat test read step 0's `fwd/updated`. Under main's `Harness.register` the position tables are θ's, so step 0 has no such key.

Runs at 4f5d27c65a [RAN]:
- r20261112-050312-d0a7, the harness's suite --fresh: 547 passed, 2 failed (the two tests of 8849711037). Locally, with both commits, test_routes and test_run pass (quick tier, 55 passed).
- r20261112-051406-cde7, the port check on the three qat entries, 8 threads set in code: "3 entries: 3 equal, 0 differ, 0 refused." Custody says preserved=NO (.custody mismatched or missing).

Resume [RAN, tiny shape, in process]: a qat row in OUTPUTS, {"targets": "blocks.0.w*.weight", "keep": 0.5, "lr": 1e-2}, fits the fixture. The unbroken job is valid with 4 distinct outputs, and the kept model hashes to `trained_sha256`, which differs from the output's digest. The continued job differs from step 2 on (the weights it reads).

The cause, from a probe: `Harness.resume` collects the model's buffers before `model.load_state_dict(state["model"])`. qat's pre-hook `start_as_loaded` then rebinds each coded layer's `start` to the loaded weight (θ plus the previous step's codes). resume's copy writes the kept `start` (θ) into the old tensor, which the layer no longer holds. With the buffers read after the load (a monkeypatch, probe only), `assert_continues` passes and every checkpoint's output is equal.

A fix could go in one of two places (I built neither):
- (a) `Harness.resume` reads the buffers after the load. That is the carrier's code (main's).
- (b) qat's hook leaves `start` alone when the loaded dict holds the latents (a kept state, not θ). That is #8910's code.

Either way, the qat row also needs `bare` to zero qat's latents (`.v`) and codes for the test's last assertion.

Your call: (a), (b), or push my two commits as they are, with qat left out of OUTPUTS. Until then #8910's head is 4f5d27c65a, red in those two tests.

