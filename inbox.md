[2026-11-22T02:10Z] 5cc6c4cce730
From the lead: until the default changes, launch every suite run with `--cpus 12 --mem-gb 54` on `research run` (a pool slot gives 15 GB unless the launch asks for more, and the suites tool starts one test process for each 4.5 GB, so a default launch runs three processes and takes hours).

