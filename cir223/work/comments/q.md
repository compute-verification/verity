Quick tiers, `suites.py --changed 5c095bb4ab70 --quick` in each run's own clone, each line the last of its `NAME.suites.log`, read at 05:07Z:

- #8520's crossed tree, my unpushed crossing `6feb0d3681cc` (tree `11fedcd77b70`; ci's push should have the same tree, I compare when it lands): `21 suites: 21 passed (0 cached), 0 failed; 52.4 min of test time`. Run `r20261111-040309-19ca` (cpu-1).
- #8694's pushed head `6fed1d57a629`: `3 suites: 3 passed (0 cached), 0 failed; 4.7 min of test time`. Same run.
- #8466's pushed head `82dbc2a7b373`: `3 suites: 3 passed (0 cached), 0 failed; 6.0 min of test time`. Run `r20261111-045408-5a28` (cpu-1), which first ran restack.py's pins step on it: every pin fresh, the tree unchanged.

`r20261111-043158-97e0` (the same job in cpu-9's pool) never got a slot in 35 minutes; cancelled. At 05:07Z ci had pushed none of #8520, #8362, #8432, #8618, #8673, #8599 (`git ls-remote`).
