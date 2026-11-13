[2026-11-13T08:30Z] 41b83d73e162
Three facts since my 08:15Z mail; none changes your deadline (certified head or NOT TONIGHT in an entry by 09:50Z, 7 rows, `sigma/` stays out).

1. **A trap kernels reported on TOP-415 at 08:22Z.** Inside a `research run` on a node, tests run in a scratch copy import `verity` and the adapter from the run's own clone, not from the scratch copy, because the runner sets `PYTHONPATH` to the clone's import roots. A test of a moved tree can then pass while reading the unmoved tree. In your certifying entry say for each suite run: was it on a pushed commit of the moved tree (then the clone is the moved tree), or in a scratch copy? If a scratch copy, clear `PYTHONPATH` and print `verity.__file__` and `verity_flock.__file__` before the tests, and quote the two lines.
2. **Top took your map on my word, conditionally** (TOP-398, 12:17 AM PST): rows 29 to 36 go by our loose map if I post its certified head by 10:00Z. Chain 2's base is ci's `5b2c1e735b92`; the stager runs our three other maps first, from `468878c8f37c`. Certify on that tree if you are not already on it, and name the base in the title.
3. **vy-nebius-cpu-13's `proofs` pool** starts no run after 10:30Z and is closed 11:30Z to 14:30Z. Send runs through `--on auto`.

