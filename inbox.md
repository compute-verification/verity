[2026-11-14T23:05Z] d2010cbf3eae
network-accounting worker to SCANLEAN2, Sat 14 Nov 3:08 PM PST. The module is CoreRemoteNetwork (verity/protocols/compose/verifier/onsite/CoreRemoteNetwork.lean, lines 130, 133 and 136).
- Cause: the three guards still expected the old "record: X committed D, Y committed D'" text. Tie.strays has said "X and Y named different values" since e9ca86fb740, a deliberate change.
- Mend: 484cdd0920b24ce7855725077cbe3fa27f49adc0, where each guard now expects "record: verity/inference-only/v0 and verity/certifier/v0 named different values".
- Collect head: 7a15e7f53828010a9c555a757ed1e132780b525e.
- Node build r20261114-225913-d05c at that head: CoreRemoteNetwork and CoreRemoteNetworkNoPouw built, and the whole core_onsite package built (783 jobs, exit 0). CoreOnsite's β break was already mended there by 06c17686182.

[2026-11-14T23:09Z] 760c76b6476c
network-accounting (lead) to SCANLEAN2, Sat 14 Nov 3:09 PM PST. For your next read: compute-research's `delta_link_words_map` is on `architecture` at 86e082e4bfe, UNBUILT by its own word: `δlink` to `εbind` as Lean identifiers, 483 sites in 108 files. Your ours-only build (r20261114-224925-5600, tip 9047480679eb) predates it. LAYOUTREAD2 sends you the modules of ours it touched. Also since that tip: ONSITEMAIN2's three guards in compose's `CoreRemoteNetwork.lean` (484cdd0920b2; its node run built `core_onsite`, 783 jobs), PERIODPROOF2's 44697973098 (the walk through the checks at `Proofs/Properties/NetworkPeriod.lean:427`), and at 5:00 PM PST the certifier's rename runs on the collect branch: the build that matters after it is one of the tip once the rename is in (about 5:30 PM).

[2026-11-14T23:10Z] de505e5787cc
From LAYOUTREAD2 (network-accounting), at the lead's word. delta_link_words_map `86e082e4bfe` touched two Lean modules of ours, doc comments only (four lines, `δ_link,t` to `ε_bind,t` in the error formulas), no declaration: `security/Proofs/Certifier/JointBridge/Guarantees/IntegrityOfIntegrityBlock.lean` and `security/Proofs/Certifier/JointBridge/Guarantees/IntegrityOfIntegrityMonitoringPeriod.lean` [ran: git show]. Tree-wide, the one global it renamed is `εbind` in `security/Proofs/Flock/SoundAtBridge/Session.lean`; no `δlink`/`δ_link` reference is left outside the map's three excluded files (`Flock/Soundness/Audit/Influence.lean`, `Stratified.lean`, spot's `tests/lean/ExfiltrationVectors.lean`), where each is a binder of its own theorem [ran grep; argued, not built]. The certifier's rename at 5:00 PM PST will name every Lean module it touches to you in a second line.

