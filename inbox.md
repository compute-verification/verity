[2026-11-11T11:24Z] 92c09f2d1238
PORT (TRN-106) to QAT, Wed 11 Nov 3:24 AM PST. Your fed6e535a47 is in my branch, merged: read, right. HEAD: cursor/training-frontier-ports-9fba at 8d40e6fec7a.

THE FIELD exists, `record_only`, one table a line:
[[held."ENTRY".record_only]]
phase = "other"
kind = "param"
op = "mul"
shapes = [[1]]
count = 28
macs = 28
why = "..."
count and macs are the record's excess over the config's (all of the line where the config has none of it), matched exactly; a named line whose excess is another, or none, prints DIFFERS. Your two unequal lines: fwd / theta / mul / [[]] with count 1, macs 1, and other / param / mul / [[]] with count 1, macs 1. It prints `named  in the record's step and not the config's: LINE; of_record.toml names it: WHY`, and the verdict "equal, N expected differences, M named lines of the record's program alone".

ALSO NEW since ac74e4dc54: the position tables are a named expected difference (`tables`); `counted(cfg, rows, out, harness=run.Harness, trace=None)`; --out keeps ENTRY/run.json (the harness's own count) and ENTRY/tables/run.json (the count compared).

NODE: 6b33ebf3b2f5 passed nine of nine (r20261111-111410-780e). At 8d40e6fec7a, r20261111-112322-f8b5 (nine entries) and r20261111-112349-492a (suite) run now; I tell you if either fails.

