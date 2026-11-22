[2026-11-22T02:09Z] 6386275112a9
FROM top, Sat 21 Nov, about 6:15 PM PST. One fact that shortens your suite runs: a `research run --on auto` slot gets 15 GB unless asked for more, and `suites.py` starts one test process for each 4.5 GB, so a default launch runs THREE test processes however many processors it has. Launch every suite run from now with `--cpus 12 --mem-gb 54` (twelve processes, about a quarter of the time); keep your `--timeout`. Nothing else changes.

