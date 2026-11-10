[2026-11-10T03:45Z] 2d29798d0ae5
## PROVECUT, Tue 10 Nov 03:55Z: CMP-370 addendum. A Pearl+ word unit at k = 8,192 is measured at k_log 25, so four tables a tile

My 01:32Z sizing estimated k_log 23 to 25 for a word unit at k = 8,192, so one to four tables a tile. `main` already has the measurement:
- `Pc8CapWordUnit_v1{K=8192,DEV='sm120'}` staged at k_log 25, 8 instances in one session [measured].
- That's r20261004-032911-9751, a source run of `benchmarks/numerical/.../unit_costs_prover.py` (`SOURCE`).
- So the word units take the top of my range: four tables a tile at k = 8,192. The rest of the sizing stands.
- **Caveat.** That run is from 4 Oct. `cap.py` has changed since only in two pure moves, but `flock-circuit`'s SHA-512 and hm96 glue may have. A CPU restage would confirm; say if you want one before 12:00Z.

