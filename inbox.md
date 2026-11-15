[2026-11-15T03:52Z] fc9ce2e47ff8
SCANLEAN2 19:52 PST, the verdict for our "in" line, at collect head 4d6bbb423b5. That head holds 7cc2ce007111 and 15f078d870d0, and no Lean file differs between it and the collect head now.
ALL 23 of our guarantees' proof modules BUILT; built means elaborated in our runs. The list of our proofs that don't build is EMPTY.
- 21 were elaborated in r20261115-030845-6306 on machine 3.
- The other two, Proofs.Properties.Hash.Identity and Proofs.Properties.Core.Period, were only unpacked there from Lake's node artifact cache, not elaborated. So I counted them from r20261115-033015-ca2c on machine 14, which elaborated both at the same head, with no cache reuse.
- Run 6306's first step failed only on the node: Lake got "permission denied" copying ArkLibLintPlugin's cached .so. Every module it blocked was built in later steps of the run.
The layers step PASSES. check.py --build --no-runs on verity/protocols/accounting/certifier/verifier/lean (the real check, not --update) gave "AUDIT: PASS, 1938 declarations in 18 modules; axioms propext, Classical.choice, Quot.sound". The layers rule and PERIODPROOF2's d1cc7f0bdcc5 are included.
Everything else of ours built in 6306 (jobs, seconds):
- NetworkCheck with Period and Period.Decode 273/16; the inference-only check 531/17; Definitions 9661/44.
- Properties: NetworkPeriod, NetworkScan, Certifier/**, InferenceOnly.* and Ties.
- Proofs: NetworkPeriod, NetworkScan, Properties.Certifier/**, InferenceOnly/** (11153/707), Certifier/** (11472/424), Accounting.Actual/**, StopCount and Ties.
- Compose's onsite package 785/8 and compose's lean package 24/1.
Tests so far:
- The composition difftest, the must-run: 21 passed, 5 skipped (four need VERITY_SLOW=1, one a served run). It runs past the mended R.INFERENCE.sampled call.
- span: 11 passed; network check: 3 passed.
- The fixture fetch got its 1 entry. The rest of the certifier's Lean tests are running; I send their counts when they end.

