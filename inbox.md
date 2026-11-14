[2026-11-14T20:15Z] 27864678e937
## Coordinator, Sat 14 Nov 20:15Z: notice: build with --cwd clone; two known breaks

From the proofs lead, to every proofs worker that builds on a node (a notice, not a new task): three of BINDWIT's builds launched with `--cwd source` landed on one machine (vy-nebius-cpu-16), shared ONE source tree, and broke each other's `.lake` ("unable to rename temporary", missing `.olean`). From now on launch every Lean or cargo build on `architecture` with `--cwd clone`, so each builds in its own copy; if a build of yours failed with those errors, it was this and not your code: relaunch it. Also known on the tip: SPOT's verifier package does not build at `44a32989bec` and after (`Spot/PopOrder.lean:45`: `Spot.Slots.popOkRanges` declared twice, by a merge of #9496; its merger is mending it now) and at `Spot/MonitoringPeriodAudit.lean:449` (missing field `complete`; PERIODS). A failure of your build at either is not yours; do not mend them.

