[2026-11-12T11:59Z] b3a677255e68
From circuits' lead, Thu 12 Nov 4:01 AM PST [12:01Z]. Not urgent: take it while a CIR-253 run waits for a slot, and before Fri 13 Nov noon PST.

#9226 and #9242 are on `main` (`e52d53383d7f`, landed 11:47Z). The `<registered>` lines' pull request for CIR-248 is yours to cut now, as planned:
- Branch `cursor/circuits-cc-heavy-registered-6f04` from origin/main. `heavy.json` only. Body: `Lane circuits`, `Refs CIR-248`, `Goal infra/full-check-45-min`. Unmarked; I mark it.
- First check that the cut you measured still holds: `main` has 826 catalog bindings now against your 818, and the registered roots sort by id, so a new id can move roots between records. Compare `units.registered_entries` at `62985b7a372e` and at origin/main (which ids are in records 0, 6 and 7). If it is the same cut, write your three figures (`:7` 4.47, `:0` 3.82, `:6` out). If it moved, do not guess: say what moved and propose one of (a) one figure for all eight records, with what that does to `_budget` at a 60 GB slot, or (b) the figures as measured, knowing each new Definition can stale them. I decide.
- One run: the suites `tools/verity/circuit_check` and `repository` at its head.
- One comment on CIR-248 when it is open (`--worker CC-RECORDS`), then back to CIR-253. Name the pull request in your CIR-253 return as well.

Also for your CIR-253 design, a fact from the check that landed `e52d53383d7f` (runs r20261112-103655-c9e8 and r20261112-104146-92db, read by me): the suite's session on cpu-9 built all 826 bindings apart (1,390.2 s at 9 workers, tests 1,841 s summed), and the circuit-check step on cpu-5 reused none of 826 bindings and none of 17,189 targets. I have asked ci on CI-485 where records could live so every shard's node reads them. If that happens, the cache you expire may move off the node's home: keep the expiry's directory a parameter, not a constant.

