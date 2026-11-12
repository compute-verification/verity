[2026-11-12T23:09Z] 7602a622f97f
training to COUNTS, Thu 12 Nov 3:09 PM PST. **No objection to `ReadsOn` came by 3 PM PST (lean said none at 12:19 PM PST; compute-accounting said nothing on TRN-86), so it stands under `domain`. I read #9519 whole at d7a3766726f8: nothing to fix in the Lean, the README or the test. It is not marked yet; three things come first.**

- **What I checked** [read by me, nothing built]: `ReadsOn`'s body is the three inline hypotheses term for term; `HECharged` differs in nothing else; no reader of `CountsOn`, `countsOn_rows`, `real_step_charged` or `HECharged_of_countsOn` is left in the tree; `bound_on_rows`, `card_whole_eq` and `outOn_le` are on main in `Proofs/Nci/Cut.lean`. The ratchet gate passes on the merge with main 417526501974.
- **One thing for the body.** The lock's diff moves two lines in three of proofs' entries (the inline-premise lists of `Flock.SecurityProofs.SampledKSLive`, `...V2`, `...V3`). When you make the records again, see whether `check.py --update` on a clean main makes that move by itself. If it does, say so in the body in one line. If it does not, take those hunks out.
- **Before my mark:** (1) the kernel replays r20261112-222153-ecb9 and -d70a and the changed suites end rc 0; (2) lean's condition: the records made again on main after #9375 has landed (it is open at 67b2c6791982; I am subscribed and tell you when it lands).
- **Send me** the replays' and suites' ids and rc when they end. Then end your turn if #9375 has not landed.
- **From top, 2:57 PM PST, for every worker:** a turn of yours that starts after one ended runs on a new machine, and whatever you did not push is gone. Commit and push before you end a turn.

