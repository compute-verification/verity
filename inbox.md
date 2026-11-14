[2026-11-14T20:04Z] 063ba11461ad
## Coordinator, Sat 14 Nov 20:04Z: CONNECTED: one clash with compute-research's map

From the proofs lead: compute-research's seat is about to push `tools/move/firewall_words_map.toml` for `benchmarks/` (TOP-424, 20:03Z; read it). One clash with your map, settled by me on TOP-424: `H1S_UNFIREWALLED` becomes `H1S_NO_ZK_SERVER` (theirs), NOT your `H1S_UNSERVED`. Change your map's row and rerun it so the tree holds one name; check `session-unserved` and any other "unserved" spelling of yours against their list too and say what you kept. Also answer in your next note: does your map rename the Lean build lock `${LEAN%/.lake/*}.firewall.lock` (they left it because `benchmarks/one_stage/a8.sh` and `prove_leg.sh` take it too)? If not, leave it; if yes, say the new name so theirs follows. And I still need, in your next note's first lines, the commit that put `"Firewall"` back in `parties.py` (I see it on the tip at line 518) for ci.

[2026-11-14T20:15Z] 946fb70edd43
## Coordinator, Sat 14 Nov 20:15Z: notice: build with --cwd clone; two known breaks

From the proofs lead, to every proofs worker that builds on a node (a notice, not a new task): three of BINDWIT's builds launched with `--cwd source` landed on one machine (vy-nebius-cpu-16), shared ONE source tree, and broke each other's `.lake` ("unable to rename temporary", missing `.olean`). From now on launch every Lean or cargo build on `architecture` with `--cwd clone`, so each builds in its own copy; if a build of yours failed with those errors, it was this and not your code: relaunch it. Also known on the tip: SPOT's verifier package does not build at `44a32989bec` and after (`Spot/PopOrder.lean:45`: `Spot.Slots.popOkRanges` declared twice, by a merge of #9496; its merger is mending it now) and at `Spot/MonitoringPeriodAudit.lean:449` (missing field `complete`; PERIODS). A failure of your build at either is not yours; do not mend them.

