[2026-11-09T03:58Z] 323102219136
Coordinator 04:16Z (real UTC): finding B now has a check. TWOVERIFY's #8270 (head c4f904f305, stacked on #8119) adds second-values: valuesOk in Spot/SecondStage.lean, with check_ok_values. Round 2 holds each statement's registered file to its template reference's, so a port's registered value name is fixed before the run. Use check_ok_values to discharge the registered-value case instead of a named hypothesis, once #8270 is on your base or merged in (it's unmarked; merge only after it is admitted).

