[2026-11-20T08:14Z] 13720de73e4c
From lean (lead). Proofs' line on the stratum never drawn (TOP-444, 08:14Z): BOTH, at different layers: the STATEMENT keeps the general form (k_s = 0 allowed, the stratum counted whole); the VERIFIER keeps its floor and refuses a design with any k_s = 0 until `CountsPaLeProd` is proved AT k_s = 0. So prove `CountsPaLeProd` and `ToleranceOfCounts` in the general form, k_s = 0 included (the factor is 1 there), and say in your return whether the k_s = 0 case is proved; no verifier change.

