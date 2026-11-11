[2026-11-11T13:12Z] 2069c88442c9
From the kernels lead: FIRST, ahead of the verdict module: #8679 no longer merges onto main (ci, 13:10Z; a conflict in benchmarks/one_stage/tests/test_server_rate.py with proofs' 0a0e98be69) and is out of the queue until it does. Read the newest section of inbox/KSTEADY.md on cursor/verity-kernels-0ebe now: merge main into #8679's branch keeping both sides' tests, run its touched suites, end your turn with the head and the run so I mark it again; then #8953 on the new head; then infra's small follow-up on --cores (refuse a list partly outside the machine).

