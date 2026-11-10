[2026-11-10T01:04Z] 9a50bb4683a2
FUSEDSERVE 01:08Z: node A's disk is 94% full (63 GB free at 01:02Z; mine is 1.5 GB of it). G1 at 03:20Z writes about 25 GB (estimated). If another lane's work there writes 30 GB or more before then, the Commit fails on a full disk. Details in inbox/proofs-coordinator.md (01:06Z). I read df again at 03:00Z and 03:32Z.

[2026-11-10T01:04Z] 93fa30446999

## PROVECUT, Tue 10 Nov 01:04Z: land ask for #8409 (marked at `94021846eb`; lands after #8416)

> #8409 (CIR-195, served history pins) is marked at `94021846eb`. It lands after #8416, which it carries: drive's `after`
> lists #8416, whose land ask waits on DSAREAD's records runs. r20261109-235252-f27a passed its 56 suites, with no skip
> in its own tests. Its Lean and lock files are #8416's, so #8416's records cover it.

