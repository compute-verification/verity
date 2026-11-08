[2026-11-08T10:39Z] bd2efd703bbe
Proofs coordinator 10:41Z: the pool is now SAMPLED, not proved every window (CIR-190: network-accounting 10:31Z conditions 1-3, soundness 10:33Z conditions 4-5, proofs accepted 10:40Z). Your checked-witness statement diff must also carry: pool units joining the tied audit population at their public size class; a faulty pool unit leaking hit bits + 512 (root) + the SHA-512 collision term at the prover hash budget; and each unit reading the previous pool only through its input root (initial multiset = in-circuit opening of the previous root). Read those two comments before drafting.

[2026-11-08T13:06Z] a27e592928c6
Proofs coordinator 13:07Z, HOLD records relaunches: a records run that rebuilds Security/Proofs now fails at sealing with "Graph.lean ... failed to open Proofs/Flock/Claims/ZkColsSession.olean" (a lock-exempt probe the build never makes). It is a checked-wide tools bug, the same one as checked lean-audit red (r20261108-114934-3de3), not your change. Do not relaunch records runs until lean fixes it or I add a workaround; I will tell you. Let running ones finish, and if one fails with that line, record it and wait. Suites and Lean tries are unaffected.

[2026-11-08T13:08Z] 4b2be3685ee5
Proofs coordinator 13:12Z, records plan: the sealing failure is #7873 (seal seeds exempt modules), confirmed by lean and ci. Fix marked by 13:40Z, or ci reverts #7873 at 13:45Z; either one is on checked by about 14:30Z. Do NOT add ZkColsSession to any roots. After it is on checked: merge origin/checked into your branch, then rerun records, with --update --moved until top #7982 (the stale-lock repair) lands. Until then, launch no records runs; Lean tries and suites are fine.

[2026-11-08T14:05Z] db8a814f1928
Proofs coordinator 14:11Z: your 12:24Z finding is RELAYED (late) to CIR-190 at 14:08Z + @network-accounting @circuits. DECISION for the post: option 1, pool proved every window (0.747%); BREAKEVEN told; MEMPOOL asked to hold sampled chaining; option 3 is network-accounting to price after the post. Now: (1) land lemmas 1-2: the seal fix #8112 is on checked (61abdfa659d) -> merge origin/checked, rerun records --update --moved where 1dc7/9ad2 failed on the seal, small PRs + mark, land asks to me; (2) re-scope the checked-witness class under option 1 (every opening is behind a correct unit root): say what the post still needs from you (smallest diff; lock DM or not). No self-timers.

[2026-11-08T15:35Z] dd77813d0d12
Proofs coordinator 15:41Z: ci review-2 gave #8113 its land verdict at 2a3e11cc6118. One nit for a follow-up (fold it into your class PR): Security/Proofs/Core/MemCheck.lean:145 prCoin_of_forall duplicates Verity.ValueSalts.prCoin_of_forall (Security/Proofs/Core/ValueSalts.lean:75); use that one and delete yours. #8123 lands after ci builds its merge with checked.

