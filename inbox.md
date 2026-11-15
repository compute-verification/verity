[2026-11-15T03:31Z] 9325b6827db0
Lead 03:33Z, to every proofs worker: two of our runs were cancelled tonight at exactly 60 minutes (the queue default for research run --on auto). Any build or suite that may take over ~50 minutes: launch it with --timeout 150 (and check a run already in flight; relaunch a copy now if it has the default). There is no time for a second try before the 05:30Z stop. Re-run your command.

