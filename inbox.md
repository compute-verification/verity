[2026-11-22T02:09Z] 58acf913b629
Lead: top, 6:09 PM PST (TOP-450): until infra changes the default tomorrow, launch every suites run with `research run --cpus 12 --mem-gb 54`. A default slot gives 15 GB, so `suites.py` runs only three test processes and a whole-suites run passes the two-hour limit (ours did, six times). Applies only if you launch `suites.py`; named test files and your prover runs are unchanged. No reply needed.

