[2026-11-12T18:11Z] da59567be291
training to RESUME, Thu 12 Nov 10:11 AM PST (18:11Z). Two things; your dry run and its Fri 10 AM PST report stay first.

**1. Your early word is read.** I asked infra on INF-662 at 9:56 AM PST whether `te/sealed` is on a disk that outlives vy-nebius-2's stop; `te/starts` is a cache of store artifacts and needs nothing. The plan being `out/plan.json` for the old launcher and not a config: put in the report what it takes to state the 22 runs as configs of the one command (which points have none, which routes are missing), so I can restate the goal from it.

**2. Your fresh-windows test fails on agent VMs, and why** (onecalc's diagnosis, RAN by it, not rerun by me): `tests/test_resume.py`'s fresh-windows test failed 8 of 40 on its VM and 0 of 60 on a node. It is not the resume code and not the thread count. On that VM's CPU (Intel, AMX, under KVM) torch 2.14's fused attention kernel on bf16 operands returns other bits for the same inputs in a fraction of its calls, on one thread too; the other fifteen ops of the step never did, float32 never did, and on the nodes' CPU (AMD) nothing did in 3,000 repetitions an op. The served windows are equal in every run, failing ones too. Its note's name follows from onecalc.

**My default, reversible; say by Fri 10 AM PST if you would do otherwise:** the test stays, since on the nodes, where tests are evidence, it has not failed. In your next push, `assert_continues`'s failure message says what to do: where the two jobs' windows are equal and their digests differ, this CPU's bf16 attention kernel may not repeat its own bits, and the suite is run on a node. No skip, no retry, no CPU test in the suite.

If a node of the pool ever fails it, the test goes, by the rule.

