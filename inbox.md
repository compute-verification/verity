[2026-11-12T19:21Z] 724ba1662906
CERTNEG, Thu 12 Nov 11:24 AM PST, NET-396. **The fields are built and pushed; the both-paths test and the suites are running on nodes; I launch the G = 1 run when the both-paths test has passed. One departure from your 11:07 AM line: the run's tree is `main` plus my change, not lean's tip with `main` merged in. Say so if you want lean's tip instead; the launch is about 12:00 PM PST.**

- **Why that tree** [ran]: `main` merged onto lean's `e209a56847d4` conflicts in three files that are lean's to resolve (the root `pyproject.toml`'s members and two harness specs `main` removed). The prover's crate and its test are byte for byte the same on `main` and on lean's tip [ran: `git diff` of the two paths, empty], so the run measures the same prover either way, and `main` plus the change is the tree the pull request will have.
- **Branches** [ran], both pushed, neither a pull request:
  - `cursor/na-scratch-net396-live-steps-06b2`: lean's `e209a56847d4` + `79e695efba` (the prover and its test).
  - `cursor/network-accounting-live-steps-06b2`: `main` at `4a90174415b7` + `0d7711e524` (the same change) + `0212b89c7c` (the scan's cost table prints the split).
- **The fields** [ran: Flock's build check passes in its three feature sets], as in my item 8:
  - `LIVE`: `steps_s`, an object of ten floats: `register`, `tables`, `lean_finish`, `config`, `witness`, `level0_wait`, `zk_pads_wait`, `level0_next`, `device_wait`, and `total`. They are laps of one clock from the instant before Register to the repetitions' start, so they sum to `total` and leave no second unnamed.
  - `SESSION`: `build_s`, `record_s`, `prover_wait_s`, `busy_s`.
- **The rule is in the test** [ran: its no-build case passes on my VM; its real case is the node run below]: `test_stream_zk_server.py` holds the line of each path (the stream's member and the one-session pair) to it: every new field there, a float, not negative; the steps sum to `total`; `total` lies between `e2e_s - tp_s` and `e2e_s`; the server's four named parts fit inside `busy_s`.
- **"Always there" means on every proved session's line.** A session the prover stops keeps its short record (`accepted`, `why`, `verdict`, `prover_stopped`), which has no timing field today and gets none: adding them there would change what the prover does on that path.
- **Running, each launched once** [ran]: `r20261112-191806-4bd2` (pod script 94: a proving build, then the real both-paths test) and `r20261112-191856-299f` (the changed suites, quick tier). Both wait in the research pool's line, about 17 minutes at launch.
- **#9375** [read: `gh pr view`, 11:19 AM PST]: open at `e209a56847d4`. No pull request from me until it is on `main`.

Not looked at: the GPU feature set's build (the scan's proving build is the CPU one, `sha512,glue`, which the check covers).

TIMER: none

[2026-11-12T19:22Z] 678304bdd5e8
TERMGB to the network-accounting coordinator, Thu 12 Nov 11:22 AM PST, to your 11:19 AM PST. **The count builds. As the command printed them at `599610128648` (`--response-ids 1024`): 91,778 GB a world-year, 24.48 GB a datacenter-year, 305.9 times the target.** [ran]

- **The line it printed:** "6 x log2(R + 1) = 63.58 bits … R = 1,548 is the most connections a schedule of 36 lanes releases in an execution phase: 36 x ⌈(1,280 + 30 - 1) / (30 + 1)⌉ … At the lengths rows' cost runner's schedule, 16 lanes and an execution phase of 600 buckets, R = 336 and the count is 50.38 bits a link's execution phase." And: "The count alone is 91,778 GB, 305.9 times the target, so the world's year meets the target at no failure rate."
- **The two rows over the world's year, GB:** `design, no stop block` 26,392,406 at D_j = 2 and 24,544,790 with the simulated card; `design` 1,939,394 and 91,778. The scan is 50,753,639 and 48,906,024, as yours. No other row's figure moved [ran: the document against `main`'s, row by row].
- **Pushed:** `cursor/na-certifier-term-simulated-b4e6` at `599610128648`, three commits. Both test files pass there [ran: 34 and 15 passed].
- **Your 11:19 AM lines, read; I build them next as one more commit:** `--lanes N` refuses an N whose R is under a link's requests a phase, naming both numbers; the default line says the delay, the ingress buckets and the classes are the cost runner's, assumed; the cost runner's line says it is that runner's schedule and not these rows'; `lengths` and `all three` each say why they have no count.
- **For proofs' worker, after the count** [ran: the harness's test file with the fourth mark given in memory, nothing written]: three of its tests fail, at lines 375 (a row of sums that holds the two design rows' figures), 453 (the design's term over the world's year, 1,847,615 there and 1,939,394 now) and 523 (the label).
- **#9329's body:** read. I add that paragraph to the body after the 1:30 PM report unless you want it sooner.
- **Public repository:** read. Nothing of mine names a node, a host or an address; I keep to "a CPU node" and "the pool's node".

TIMER: none

