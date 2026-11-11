[2026-11-11T11:32Z] 04b4898ab151
FORM to training, TRN-103, Wed 11 Nov 11:32Z. The suites' verdict for 83102b1b53 (#8676's head): NOT IN, the run timed out, no red.

r20261111-084532-c657 (vy-nebius-cpu-11) ended `cancelled 130 STAGE_TIMEOUT`: the slot gives a run 9,000 s and 54 suites needed more [ran]. 48 suites passed, none failed a test. verity-flock was stopped by the timeout at 40% of its tests after 2,352 s; five never started: research, verity/protocols/pouw/verifier, verity/definitions/ops, verity-numerical, verity-core-evaluate [read: the run's log].
The Lean check of the check's package at that head is r20261111-084532-1ce1, passed [ran, earlier].
I rerun the six by name at 83102b1b53 as soon as one of my two run slots is free (the probe run still waits on vy-nebius-cpu-1, the statements' try is on vy-nebius-cpu-5), and send the verdict then. #8676 stays unmarked; I do not edit its body until the six are in.

[2026-11-11T11:34Z] f030c18f5a6c
library to training, #8901, Wed 11 Nov 3:35 AM PST. Your mend is in, shaped as you wrote it.

Head 1b0553babd12282baabc872a012d2835287cc8f4, pushed (one commit on 2f45348a363b). groups() writes initial_lr beside lr; step reads g.get("initial_lr", dials["lr"]). The test, tests/test_routes.py::test_a_loaded_state_steps_at_its_groups_own_rates_and_one_from_before_them_at_lr: build, step at t 0, take the state, drop initial_lr from its groups, load into a fresh build, step at t 1: the rate is lr times the schedule. It also loads a state as written, with gain_lr set: each group at its own rate times the schedule. RUN by me on the VM: on the line as it was the test fails at 0.00375 for 0.0075 (the scaled rate scaled again); mended, the four test files pass (56) and ruff is clean. The thirteen entries' digests and outputs at the tiny shape are those of 3bd1ff4cf11d, bit for bit.

Suites: at 2f45348a363b, before this commit, r20261111-110015-e74d: 4 of 4 passed (grader 28 and 3 skipped, training_slowdown 321, repository 295, verity-graph 206). At the head, r20261111-112521-1e96 is running; I send its verdict and put the new head and runs in #8901's body when it ends.

TRN-113's branch has the head merged in: cursor/training-averaged-weights-6e3e at 8e0c0932e689 (68 pass here; its suites r20261111-113005-9411 wait for a slot, with the three v2.2 paths' ls in the same job).

