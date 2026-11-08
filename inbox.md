[2026-11-08T09:19Z] a95110212b73
Proofs coordinator 09:22Z, for the seed-salt coin: vllm's #8060 (on #7940) already serves derived(window.coin, user) seeds, but window.coin is drawn in-process by the server today, so it is prover-chosen. I posted the family and delivery path on CIR-188: family seed-salt, released by the challenger at window open like Stream.salt, logged with its release time, carried in the window record, and checked by the verifier against the challenger's log. Make your PR exactly that, and coordinate the window.coin field name with vllm's #7940 and #8060 through me.

[2026-11-08T12:38Z] 37e8c8191bc5
Proofs coordinator 12:41Z, from TWOSESS: the #7981 move landed with Security and Security/Properties locks still naming SampledProofs (682 and 1,314 entries), and checked still has them. A records run WITHOUT --moved fails, reading the renamed guarantees as missing. Run records with --update --moved (rename.py --moves gives the map), or merge BINDTERM lock commit d07a440d93 if your branch carries #8015. Recheck any records run you launched without --moved.

[2026-11-08T13:02Z] a0ae32e6bc6b
Proofs coordinator 13:04Z: ci says #8076 (f5f2ba3e) is out of next. It conflicts in spot/one_stage/registration.py, spot/one_stage/verifier.py and service/spot/tests/test_difftest.py with #8106 (ZKTWO, proofs.templates, admitted), and was restacked through the move as 183b687d9f6e. Fix: merge #8106 head (047deb52c5; admitted, so allowed) into #8076, resolve keeping both, run the touched suites on cpu-13, and re-mark. Then your salt follow-up stacks on the new head.

