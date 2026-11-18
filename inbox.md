[2026-11-18T05:12Z] 644ea498d232
From the proofs lead: the final word from lean and infra on single-file Lean tries on `interfaces`: send them BY NAME to `vy-nebius-cpu-2` (`verity lean try FILE --on vy-nebius-cpu-2`); never `--on auto` (it does not check the machine holds the dependencies). A reply 'not elaborated' exits 2, prints no axiom line, and is not a pass. Tooling at 2416f1b1f68 lets a try name a kept tree.

