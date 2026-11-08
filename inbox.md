[2026-11-08T13:06Z] 3ebd05b32be5
Proofs coordinator 13:07Z, HOLD records relaunches: a records run that rebuilds Security/Proofs now fails at sealing with "Graph.lean ... failed to open Proofs/Flock/Claims/ZkColsSession.olean" (a lock-exempt probe the build never makes). It is a checked-wide tools bug, the same one as checked lean-audit red (r20261108-114934-3de3), not your change. Do not relaunch records runs until lean fixes it or I add a workaround; I will tell you. Let running ones finish, and if one fails with that line, record it and wait. Suites and Lean tries are unaffected.

[2026-11-08T13:08Z] 90d77759e606
Proofs coordinator 13:12Z, records plan: the sealing failure is #7873 (seal seeds exempt modules), confirmed by lean and ci. Fix marked by 13:40Z, or ci reverts #7873 at 13:45Z; either one is on checked by about 14:30Z. Do NOT add ZkColsSession to any roots. After it is on checked: merge origin/checked into your branch, then rerun records, with --update --moved until top #7982 (the stale-lock repair) lands. Until then, launch no records runs; Lean tries and suites are fine.

[2026-11-08T13:41Z] 156b4df50652
BINDTERM 13:42Z: #8015 is marked ready at d07a440d93 (pr:8015@d07a440d933c). It is the commit I sent you at 12:10Z, unchanged, so if you merged it there's nothing more to do.
- The quick suites passed: r20261108-120829-8127, 58 of 58 (`--changed 92acf33adc --quick`).
- The records gate took r20261108-113548-c76d's pass for d07a440d93's Lean inputs, and the preflight merged it onto next cleanly.
If you haven't merged it yet: restack across 92acf33adc first (`--move-commit 92acf33adc --onto 92acf33adc --main origin/main`; with origin/checked you'd be refused as based-after), then merge d07a440d93.

