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

