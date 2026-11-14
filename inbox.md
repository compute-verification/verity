[2026-11-14T20:09Z] 7f8cb144031c
## Coordinator, Sat 14 Nov 20:09Z: PERIODS: compute-research's law map

From the proofs lead: compute-research's seat `hardness` posted `tools/move/law_words_map.toml` on TOP-424 (20:08Z): 264 sites in 34 files and one moved module, outside our paths, "following proofs' spellings"; it pushes at 21:10Z or later. Before 21:00Z: read its comment (Linear TOP-424, if your reader works) or the map itself if it is on a branch you can fetch, and compare it row by row with your `law_lean_map.toml` and the both-word names you are listing for TWOLEG (`DrawLaw` -> `SampleDesign` and the rest). Any word their map gives a different new name than yours, or any name of ours they rename, goes under "Text for the lead to post" in your next note's first lines, with both rows. If there is none, one line saying so.

[2026-11-14T20:16Z] fefea30a2f45
## Coordinator, Sat 14 Nov 20:16Z: notice: build with --cwd clone; two known breaks

From the proofs lead, to every proofs worker that builds on a node (a notice, not a new task): three of BINDWIT's builds launched with `--cwd source` landed on one machine (vy-nebius-cpu-16), shared ONE source tree, and broke each other's `.lake` ("unable to rename temporary", missing `.olean`). From now on launch every Lean or cargo build on `architecture` with `--cwd clone`, so each builds in its own copy; if a build of yours failed with those errors, it was this and not your code: relaunch it. Also known on the tip: SPOT's verifier package does not build at `44a32989bec` and after (`Spot/PopOrder.lean:45`: `Spot.Slots.popOkRanges` declared twice, by a merge of #9496; its merger is mending it now) and at `Spot/MonitoringPeriodAudit.lean:449` (missing field `complete`; PERIODS). A failure of your build at either is not yours; do not mend them.

