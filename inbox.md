[2026-11-17T23:58Z] 0b2377538b05
From the circuits lead, a rule from Daniel today (no run over 2 hours without a reason; everything fast). For any run you have NOT yet launched: no `--fresh` on suites.py (a pass is keyed on its inputs), and do not run `suites.py --changed OLDBASE` over every suite main moved; run the suites your OWN diff touches (`--changed` against the merge base with main, i.e. origin/main), with the cache. The queue candidate check runs the rest. Runs already going: let them finish. Report each run minutes in your return. Re-run your command.

