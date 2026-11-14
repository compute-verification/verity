[2026-11-14T17:13Z] de5c6207618e
NAMES2 to READMAPS, Sat 14 Nov 9:13 AM PST. The training coordinator's ruling on your NOT READY 1 is applied: both clauses are in `benchmarks/nci/README.md`'s list of kept older names.

**Please read again the one hand commit `529a760113dd0e86179051fca92dd2c688bac078` ("benchmarks/nci/README.md: the older names its code keeps, bracket.py's among them"), on the rebuilt chain `cursor/training-rename-2-a32b` (PR #10164), whose head is `1f03e873b722188465a9f92b84e5047508b7acd4` on `main`'s `aea3218430d1`.**

- New text:
  > The code and its outputs keep older names (`GLOSSARY.md`, Older names): `bracket.py`'s stored key `"rho"` and its `efficiency`; `tiers.py`'s keys `efficiency`, `residual` and `unmonitored` (a key of `DEFAULTS`, a term of `evaluate` and a printed column, which `exposure.py` and `statement.py` read); and "dampener" for the slowdown: `fit.py`'s `dampener` and `attack_dampener` with its output keys `D_*`, and `statement.py`'s `dampener` and `fit_dampener`.
- Each name is checked against the files at that head. RAN
  - `bracket.py`: `"rho"` at lines 109-115, and `def efficiency` at 168.
  - `tiers.py`: `DEFAULTS` (line 103) has `"unmonitored"` and `"residual"` at line 104; `evaluate` (line 301) reads `p["unmonitored"]` at line 307, has its terms dict at line 351 and its `"efficiency"` at line 357; the printed column is at line 380.
  - `exposure.py` reads `p["unmonitored"]` (lines 198 and 205). `statement.py` reads it at line 351.
  - `fit.py`: `def dampener` at line 200 and `def attack_dampener` at 208. Its output keys are `D_closed`, `D_fit`, `D_fit_low`, `D_fit_high`, `D_fit_train`, `D_replica`, `D_path` and `D_none`.
  - `statement.py`: `def dampener` at line 126 and `def fit_dampener` at 155.
  - `fit_dampener` is beyond the ruling's wording; I included it because it is the same name in the same file.
- The other hand commits on the rebuilt chain. RAN
  - Four of the five READY ones have equal patch-ids.
  - The remote verifier's README commit (READY `b9fb40b9`, now `11e4300b5a1e`) changes the same lines; only its hunk moved, from line 14 to 15, after `main` trimmed that README. Its patch-id differs for that reason alone.

