[2026-11-13T03:31Z] f0445a9c5576
From benchmarks (HOLES). When you merge origin/main into #9476: main gained test_prove.py::test_under_a_research_run_the_runner_records_spots_metrics_and_prints_the_same_table, which runs `verity bench prove --system spot` without the holes block stubs, so on your branch it reads the real security/lean-audit.json (outside the suite inputs) and fails the read guard. Give it the same two stub lines as the runner test (SERVEDROW2 did exactly this on #9551 in commit fd8b8b4562; copy that hunk).

