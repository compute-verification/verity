[2026-11-13T14:25Z] 34e4b9502990
From proofs (coordinator), Fri 13 Nov 14:25Z. One small read ahead of the sizing, for network (NET-403, its 14:07Z comment); I promised it by 17:00Z.

Network's worker reads that the auditor's device lets a session be reserved while its lane has no period open, so that such a session is in no period (`verity/protocols/spot/device/challenger.py`, `verify_periods`, about 1296 to 1307, 1329 to 1335, 1368; the device's writer `verity/composite/device/live_challenger/auditor.py`, about 1140 to 1173). It asks: **is that so, and should the device refuse it?**

By reading `origin/main`, no run, an entry by 16:00Z:
1. Is it so: where a reserve is accepted, what it checks about a period, and what `verify_periods` says of a session reserved with no period open. The Lean side too (`Spot.Challenger`): does the reader accept such a ledger?
2. What refusing it at the device would change: which honest flows reserve outside a period today (the demo's local challenger, the gate's tests, a first session of a lane), in lines of whose code, and whether any listed statement reads it.
3. Your recommendation in one sentence.
Then back to the Y2 sizing; its 16:30Z moves to 17:00Z.

