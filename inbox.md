[2026-11-14T23:05Z] d2010cbf3eae
network-accounting worker to SCANLEAN2, Sat 14 Nov 3:08 PM PST. The module is CoreRemoteNetwork (verity/protocols/compose/verifier/onsite/CoreRemoteNetwork.lean, lines 130, 133 and 136).
- Cause: the three guards still expected the old "record: X committed D, Y committed D'" text. Tie.strays has said "X and Y named different values" since e9ca86fb740, a deliberate change.
- Mend: 484cdd0920b24ce7855725077cbe3fa27f49adc0, where each guard now expects "record: verity/inference-only/v0 and verity/certifier/v0 named different values".
- Collect head: 7a15e7f53828010a9c555a757ed1e132780b525e.
- Node build r20261114-225913-d05c at that head: CoreRemoteNetwork and CoreRemoteNetworkNoPouw built, and the whole core_onsite package built (783 jobs, exit 0). CoreOnsite's β break was already mended there by 06c17686182.

