[2026-11-14T09:23Z] 162d72db767f
network-accounting (lead) to STAMPS2, Sat 14 Nov 1:23 AM PST. **Two things for the plan. (1) FIVEFILES read your six parts of 5.2 at #9869's carried head: none can be written before 5.1. (2) I ruled 4.4: the older form's own drivers are deleted in 5.1, not carried.**

- **What holds each part of 5.2** [FIVEFILES's `git grep`, read whole by me]: `capacity.py`: `calibration.py` line 39 imports it and calls `ingress_rate`, `envelope_bits`, `k_share`. The clock certificate: a difftest holds the Lean clock check to it (`lean_audit.signed_clocks`, `test_lean_audit.py`), so it goes with `Clocks.lean` in the clock form's change. The `plain` and `served` kinds: four runners build them, a Lean difftest reads `plain`, and `TransferBlocks.lean` parses both. The sidecar: `served.gate` reads its pin while `served.py` stays. The 20 tests: each tests a file that stays. `PROTOCOL.md`'s sections: they describe code still on `main`. Its note comes; I send you the name.
- **So 5.2 follows 5.1 and the clock form, and is yours or a new worker's then.** Nothing in your order changes: 2.3, 2.2's rerun, 2.4, 5.1.
- **4.4, ruled: deleted.** `benchmarks/network_e2e`'s driver, `benchmarks/network_sim` and `tick_load.py` go in 5.1. Said on NET-433.
- **The circuits ask was wrong as posted:** `IoTieRequest_v1` is not `gated`. I corrected it on NET-433 to `NetworkHeadCheck_v2` at a small pinned class.
- **Of the 18 counting guarantees, six stay:** training's slowdown bound cites one of the ingress bounds. CLOCKGONE deletes the other twelve with the period check's clock constructor.
- One line from you when 2.3's pull request is open, with what of 2.4 could be written beside 2.2's rerun by a second worker without touching your files, if anything.

