[2026-11-17T23:58Z] c60396c18bf1
From the circuits lead, a rule from Daniel today (no run over 2 hours without a reason; everything fast). For any run you have NOT yet launched: no `--fresh` on suites.py (a pass is keyed on its inputs), and do not run `suites.py --changed OLDBASE` over every suite main moved; run the suites your OWN diff touches (`--changed` against the merge base with main, i.e. origin/main), with the cache. The queue candidate check runs the rest. Runs already going: let them finish. Report each run minutes in your return. Re-run your command.

[2026-11-18T00:34Z] 44ee6d2219e5
From the circuits lead: your three heads were pushed at 21:27Z and nothing is marked three hours on. If what you still wait for is (a) the run of every suite main moved (`suites.py --changed OLDBASE --fresh`) or (b) a GLM-5.3 rebuild waiting for a 300 GB machine, do not wait: mark each change once the adapter suite three parts, the suites its OWN diff touches, circuit-check on its sidecars and merge-tree are green at its head, say in the body which runs were still going and cite none of them as passed, and return. If something FAILED, return now with it. Re-run your command.

