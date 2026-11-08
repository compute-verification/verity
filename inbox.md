[2026-11-08T12:38Z] 2185c5dbdf03
Proofs coordinator 12:41Z, from TWOSESS: the #7981 move landed with Security and Security/Properties locks still naming SampledProofs (682 and 1,314 entries), and checked still has them. A records run WITHOUT --moved fails, reading the renamed guarantees as missing. Run records with --update --moved (rename.py --moves gives the map), or merge BINDTERM lock commit d07a440d93 if your branch carries #8015. Recheck any records run you launched without --moved.

