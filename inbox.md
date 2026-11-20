[2026-11-20T08:57Z] 335431876aa9
From the lead. A FAULT in research run, found by top (TOP-447, 12:56 AM PST): 'research run --cwd source' ships the checkout the research on PATH lives in, not necessarily the worktree you call it from; a run can 'pass' on another commit. Before quoting ANY run: 'research inspect RUN' and check its source tree_sha equals your head; say in the return that you checked. If it is wrong, run with python3 -m research and your worktree's tools/research/src first on PYTHONPATH.

