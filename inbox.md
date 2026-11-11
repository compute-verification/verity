[2026-11-11T07:25Z] f7fb5844782c
resume, Tue 10 Nov 11:25 PM PST (07:25Z). #8775: (a) to (d) are pushed. Head 672d95175ea0016a7902636703562ebdd555f2c9 (tree 9470bc060440), it holds 6991ea8e32f9. Not marked: the node suites are running.

Commits, each pushed as made, no amend or force:
- 814dc5b96c3b merge of #8680's 6991ea8e32f9.
- fba730f851d (a) a stop is read from the record. The job writes run.json with stopped = "told to stop" in its SIGTERM handler, before it ends the route's process. The command reads that file from the run's record (`research data fetch ART --path out/run.json`): any exit status but 0 with that record is stopped and is continued; no record, no run.json, or one that doesn't say so (the budget's own `stopped` sentence among them) is failed. Tests: 124, 143, 137 and 75 with the record are continued; 124 and 75 without are failed.
- 971657caeae (b) the environment is per run under /tmp: the script makes /tmp/training-slowdown.XXXXXX, sets UV_PROJECT_ENVIRONMENT into it, `uv sync --frozen`, and its shell removes the directory at the end (no exec; `trap : TERM` so the shell waits for the job and the run's status stays the job's).
- f001efaa6b4 (c) the route's process gets no AWS_ name and no RESEARCH_STORE_CONFIG; the directory that file is in (where cred.json lies) is added to --hide. One test, dummy values, names only.
- 672d95175ea (d) the kind's comment says it caps no time; max_wall_min = 60 stays (the time of a job with no budget.minutes; as a limit it only warns, read in kinds.py:105 and submit.py:156).

Three things you should know:
1. Found while testing (a): the stop record could not go where `Stopped` was caught. `with Popen(...)` waits for the route's process on the way out, so the record came only after that process ended. It is now written in the handler, first. [run on this VM: the slow test asserts the route's process is still alive and the record already there when it is ended]
2. Limit of (a), by your own rule: a job told to stop before its Python process has its handler (during `uv sync`, the state's fetch, the interpreter's start) writes no record and reads as failed, never run again. [test_job_shell.py runs that case in bash: status 143, no record, directory removed]
3. Not from my change, but it touches the slow resume test: on this VM under load the CPU evaluator (bf16 autocast, main's evaluate.py) gives different bpb for the same weights, call to call: 12 distinct values in 40 calls of θ at 4 threads, 10 at 1 thread, load 3 to 8 on 4 cores. The states' sha256 are equal every time. So the slow test's equality of scores between the whole and the cut job flaked here (passed, failed, passed, failed at the same tree); it passed every time earlier tonight when the VM was quiet. I did not delete or weaken it. It runs on a node by name now; if it fails there I send you the output and do not rerun it.

Runs at 672d95175ea, both launched 07:24Z:
- r20261111-072354-0939 on vy-nebius-cpu-7: suites.py --changed 5c095bb4ab70 --quick.
- r20261111-072429-4502 on vy-nebius-cpu-8: the slow resume test by name.
On this VM, one file a call: test_command 15 passed, test_job_shell 3, test_sandbox 6 (+1 slow), test_resume 10 quick, test_run 13, test_library_imports 1, root doc-paths, wall-clock and repository 23; ruff clean.

Not run by anyone: a queued job with the new script on a GPU node (the real `uv sync` into /tmp, whether /tmp there allows exec, the lease's timeout against the real job). The shell test uses stand-ins for uv and the job.

