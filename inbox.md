[2026-11-12T20:26Z] 3f67d518eca1
PLAN to training, Thu 12 Nov 12:30 PM PST. The report on the two sizes' plan, for TRN-42.

**The answer: "launches unchanged" does not hold from `main`.** Two of the plan's seven points have a config at their start and gain, three need a config only, and two need a new route. Even the two with a config would not run the plan's runs: they run their records' step counts, and one of them runs four jobs where the plan has one. No node in the queue can place a 656M job today or on Tue 17 Nov unless a node is started or kept past its stop. The node the plan used, vy-nebius-3, is unknown to the queue, and the queue's two nodes are stopped or stop tonight.

Each claim is marked RAN (a run, with its id) or READ (where). The plan is RESUME's stored plan, run r20261106-183414-1416. I read its `PLAN.md` from the store and its points from `plan-slim.json`. The tree is `origin/main` 0582899c43. Nothing the dry run reads changed between the commit RESUME used (9ddae97ba2f7) and that one [READ, `git diff --stat`].

**The seven points [READ, the plan and `attacks/`; RAN, r20261112-184648-ce12 section B].** GPU-hours are the plan's own.

1. 656M, gain 1.1, ref-optim: 4 runs, 73.1 GPU-hours. **Has a config:** `xl-t20-1.1x-ref-optim-fullmuon-fe-wsd`. It runs its record's 2,428 steps. The plan's limit is 1.60 times that record's gates, about 3,883 steps, so the plan's runs need a second config with that budget. At the record's rate, the existing config's 4 jobs are about 45.9 GPU-hours [RAN, section B].
2. 656M, gain 1.1, shallow: 4 runs, 102.7. **Config only.** The route is `subset` with a block pattern and `store = "q8"`. The pattern is on `main` at the small starts (`m-t20-1.1x-shallow-depthshare-4`, `s-t20-1.1x-shallow-strideshare`, each held to one of the plan's two ported runs). The blocks still have to be chosen for 21 layers.
3. 238M, gain 2, ref-data: 4 runs, 119.0. **Config only.** The recipe is `l16-t20-1.1x-fullmuon-fe-wsd-5383`, with steps to the plan's limit of 7.49 times its record's gates (about 40,300 steps, RESUME's arithmetic, which I did not redo). A gain of 2 resolves [RAN, section C, `probe-gain2`].
4. 238M, gain 2, hybrid (grow-depth): 4 runs, 116.7. **New route.** The command refuses a route it lacks [RAN, section C: "no route 'grow' ... (have ema, lora, qat, subset)"]. No entry holds the ported run r20261104-060904-d64c [READ, `of_record.toml`].
5. 238M, gain 1.1, shallow: 4 runs, 19.3. **Config only**, as point 2, at 16 layers.
6. 238M, gain 1.1, ref-optim, the third final seed only: 1 run, 2.3. **Has a config:** `l16-t20-1.1x-ref-optim-fullmuon-fe-wsd`, at its record's 1,805 steps. The plan's limit is 1.25 times that record's gates, about 2,255 steps. Run as it is, the command runs all 4 of its jobs, about 7.5 GPU-hours [RAN, section B]. Seeds are not in a config's digest, so a copy listing only final seed 102 would label its one job exactly as the four-job config labels that job [READ, `config.py`'s docstring].
7. 238M, gain 2, wildcard (topcode): 1 run, 38.4. **New route.** I checked this one against RESUME's note. `main` has `s-t20-1.1x-wildcard-topcode` and `m-t20-1.1x-wildcard-topcode`, but they are the lane's uncoded controls (commit 5d3c87da89: "wildcard's uncoded controls"). They are plain `subset` Muon runs. The topcode run codes each layer's outputs to 2 or 4 bits inside the forward and passes the decode straight through (the archived harness's capture test `Packed`, at 38e60c6778e5). None of `subset`, `ema`, `lora` or `qat` computes that [READ].

**The count.** 2 points have a config: 5 runs, 75.4 GPU-hours (at the plan's limits, not the configs' own step counts). 3 need a config only: 12 runs, 241.0. 2 need a new route: 5 runs, 155.2. The total is 22 runs and 471.6. RESUME's count is right except the last figure: 116.73 + 38.45 = 155.18, so 155.2, not 155.1 [RAN, arithmetic on `plan-slim.json`]. The command's own estimate for the 28 jobs of the seven configs on `main`, at their records' rates, is about 111.8 GPU-hours. That covers training and the old harness's overhead, not the job's trace or its environment [RAN, section B].

**Other things the configs lack [RAN, section B; READ, `config.py`].**
- None of the seven configs states `budget.minutes`, and 3 of the 44 configs on `main` state an `honest` config, none of them at the two sizes. So the grade has nothing to set beside these runs.
- A config's target gain is not in its digest. The gain-2 probe got the digest of `l16-t20-1.1x-fullmuon-fe-wsd-5383` (190150e48b08e09e). So a 2x config at the same dials and steps would be the same jobs under the same labels as the 1.1x config, and only the grade tells them apart [RAN, section C; READ, `config.py`].

**The dry run, its labels, refusals and placement [RAN, r20261112-184648-ce12, vy-nebius-cpu-1, tree 0582899c43].**
- **Each of the seven configs resolves.** Its dry run lists 4 jobs, all `missing`, and exits 3 (pending). Each would queue as `training-attack` with one GPU, preemptible, with `--disk-gb 44` at 656M and 29 at 238M, and no `--max-min`, so each lease is the kind's 60 minutes.
- **The store holds no job of the command at any size.** The node's copy of the labels has 293,341 labels, the newest at 20:12Z, and 0 under the key `training-slowdown`.
- **Refused, by name:** a `[sweep]` table ("no key 'sweep' in the schema"), `route = "grow"` ("no route 'grow'"), and a start given by size ("start is dict, not str"). Accepted: gain 2.
- **Placement, as `cluster submit` computes it.** It is a pure function and placed nothing. The node stops given were the registry's: node 1 Wed 4 Nov 22:45Z, node 2 Fri 13 Nov 01:35Z.
  - Today, a job asking the kind's 60 minutes places on vy-nebius-2.
  - A 656M job asking its 14 hours is refused on both nodes, today and on Tue 17 Nov. Node 1's admission ended 2026-11-04T20:45Z, and node 2's ends 2026-11-12T23:35Z (its stop less a 2-hour drain).
  - On Tue 17 Nov, if node 2 is kept up, the job places there with two warnings: "asks for 840 min, over its max_wall_min of 60" and "a GPU job over 60 min: split it into chunks".
  - If node 1 is up instead, an unnamed job is refused there ("lends GPUs only to a job pinned there (--on vy-nebius-1)"). With `--on vy-nebius-1` it places.
  - A job named to vy-nebius-3 is refused: "unknown node 'vy-nebius-3'".

**What names a stopped node.**
- The plan: each of its 22 runs is on a named card of vy-nebius-1 or vy-nebius-3, in a window that ended Thu 12 Nov 8 AM PST [READ, `PLAN.md`].
- The queue's description lists only vy-nebius-1 and vy-nebius-2. Node 1's admission ended Wed 4 Nov and node 2's ends today at 3:35 PM PST. vy-nebius-3 is not in it [READ, `infra/cluster/descriptions/nebius.toml`; RAN, section F].
- The command's dry run accepts `--on vy-nebius-3 --pool training` and prints "run on vy-nebius-3" for 4 jobs without checking the machine, so a stopped node is named only by whoever passes `--on` [RAN, section A]. I did not launch to see whether the launch refuses, since that would start a job.
- `starts.toml` names four node-local paths: `/workspace/research/te/starts`, the stream `/workspace/research/scratch/covert-witness/train/v3`, its search pool, and the final set `/workspace/research/te/sealed/heldout/v3/final`. A CPU node holds none of them [RAN, section D]. The two starts are in the store with `starts.toml`'s sha256 for both files at both sizes [RAN, section E], so a new node can fetch them. The stream, about 80 GB, is in no store [READ, RESUME's note; not checked by me].

**Item 4: what must be true before the first run starts the hour Daniel says go, in order, each with its owner.**
1. Daniel's word on Tue 17 Nov, and the spend: Daniel, through training's coordinator.
2. The plan restated as configs of the one command, with GPU-hours: training's coordinator.
3. The configs: training.
   - Points 2, 3 and 5 need configs, with the blocks chosen at 16 and 21 layers and the step count at the plan's limits.
   - Points 1 and 6 need configs at the plans' limits, about 3,883 and 2,255 steps, or the coordinator's word that the records' step counts stand.
   - Point 6 runs one job only by a config that lists final seed 102 alone.
   - Each config names its `honest` config.
4. The two routes, grow-depth and topcode: training. Each needs a reference test and an `of_record.toml` entry held to its ported runs. Goal `training/search-frontier-in-library` plans them by Wed 18 and Fri 20 Nov, after Daniel's word, so points 4 and 7 cannot start the hour he says go.
5. A job's time: training, with infra reviewing the kind. Without `budget.minutes`, each job leases 60 minutes and continues across leases. A 656M job is about 11.5 GPU-hours at its record's rate, so it needs a dozen leases or more. The trace after the last step takes 24 minutes at 656M of the 30 that `OUTSIDE_MIN` leaves for the environment, the loads and the trace together. With `budget.minutes` a job asks for 1.2 times that plus 30 minutes. The queue only warns past the kind's 60, and places it where a node's admission window is long enough [READ, `run.py` `run_minutes`; RAN, section F].
6. A GPU node in the queue with an admission window past the longest job: infra. Either node 2 kept up or started again past Fri 13 Nov 1:35 AM UTC, or node 1 started with the command run as `--on vy-nebius-1`, since unnamed jobs never place there. vy-nebius-3 needs registering in the description before any job can name it [RAN, section F].
7. On that node: `starts.toml`'s four paths and unprivileged user namespaces for the job's sandbox: infra. The starts come from the store. The 80 GB stream comes from a stopped node's disk, or a rebuild with the archived builder.
8. A first small job on that GPU: training. It should be a 36M config with one seed. The command has never run a job on a GPU, and this gives the rate `budget.minutes` needs and shows the sandbox, the labels, and a stop and its continuation.
9. Someone runs the command again while jobs are stopped, and decides before the first run what a failed label takes: training. The command never reruns a failed job, and an eviction during the 63-second environment build fails that job under its label [READ, the README].

**The time and memory a step's trace takes [RAN by RESUME, r20261112-174639-330a, vy-nebius-cpu-1, tree 9ddae97ba2f7; I read its stdout and `job.json`].** These are `python -m training_slowdown.library ENTRY` on fake tensors, 8 threads, on an 11-CPU slot, training nothing.
- 238M (`l16-t20-1.1x-ref-optim-fullmuon-fe-wsd`): 4 min 48 s, peak resident memory 4.8 GB (4,796,016 kB).
- 656M (`xl-t20-1.1x-ref-optim-fullmuon-fe-wsd`): 24 min 25 s, peak 19.5 GB (19,517,408 kB).
- Both exited 0, and both logged a fake-tensor `RuntimeError` on stderr while tracing: "Attempting to broadcast a dimension of length 1024".
- Infra's figure, as you relayed it (TOP-412), is about 50 GiB for a traced step of an l16 or xl entry. I have not reconciled the two. Infra's may be a job's trace, taken in the route's process with the model and its optimizer's state still in memory, but I did not run that.
- The queued job's environment builds in 63 s to 5.6 GB, with CUDA torch for sm_120 [RAN by RESUME, r20261112-174859-65b6; I read its stdout].

**The pull request.** Draft #9453, not marked, Refs TRN-75, head 177b3db24a.
- RESUME's commit named the suite `training_slowdown` in its failure message. `suites.py` matches that against two suites and refuses to run [RAN, r20261112-184612-107b]. My second commit names it `benchmarks/training_slowdown`.
- At that head, the message's own command on `tests/test_resume.py`, slow tests included, passed 20 in 204 s on an AMD node, and the scratch check of the message passed [RAN, r20261112-190853-3b46].
- I agree with your ruling on the test, and would not do otherwise.

**What I could not check.**
- Whether any live GPU node holds the stream and the final set. The job allows no GPU node, and vy-nebius-2 takes no `--on auto` job until 3:30 PM PST today.
- Whether a launch with `--on vy-nebius-3` is refused at launch.
- Whether a job's trace costs what the library's does. No job of the command has run on any machine.
- Whether `/workspace/research/te/sealed` outlives node 2's stop (INF-662). I read no answer.
- The step counts at the plan's limits are RESUME's arithmetic. Steps scale with gates only if a step's gates stay fixed.

**Leftovers.** RESUME's scratch check, as first sent, left a `.venv` under `benchmarks/training_slowdown` in vy-nebius-cpu-7's shipped tree of b098faf2f87b. No other run uses that commit. Every run of mine has ended or been cancelled.

