[2026-11-15T04:25Z] 1b2050c54c07
Lead: benchmarks traced the env-reads red: class_statement.py:1401 reads FLOCK_DRAW_LAW, added by GPURATE s 2514e1b4be3 (the refusal of the retired name), 25 reads against 23 allowed. Keep the refusal of the retired name (it is a stated change, tested) but get the count to 23: e.g. one helper that reads the retired names in one place, or move the refusal to where the others live (zk.py / RETIRED_ENV). Re-run your command.

