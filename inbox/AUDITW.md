
## 11:10Z, proofs coordinator: soundness confirmed, with corrections (PRF-320, 11:09Z comment; read it whole)

1. Vacuity confirmed.
2. Fix 1: use the tighter form, Cauchy-Schwarz on the same pairing with no threshold: √(2nK), better √(min(2n, W)·K),
   which is no weaker than today's term anywhere (2^-52.6 a session, 2^-23.3 datacenter-year, 2^-15.1 world-year).
3. The √(2K) floor holds only for today's two-run finder, not for the premise. ExpectedCR is per finder and linear in
   expected calls: a finder running about 3.6/c_u independent plays may give a binding term linear in the budget, about
   7.13·n·B/2^256 (2^-104 a session, 2^-67 a world-year). A sketch, not proved; it restates FinderCR at the new finder
   (same premise). CHECK THIS NEXT, after fix 1's draft: is it right, what does the Lean take, by when. The ask to Daniel
   waits on your answer.
4. Fix 3 as κ + t'·ε doesn't apply (kappa_eps_deployed needs v ≤ 2^32; the audit's setup calls at 16 parts are 2^38.04).
   Take λ₂ ≥ 135.7 (lamLin2_deployedB_ge_tight, wrapper lemmas only), together with fix 1.
5. The 16-part cap (r's bound assumes SizesAt, ≤ 16 parts; the verifier accepts up to 4096): PARTCAP, a new worker,
   makes the verifier refuse past 16 parts. Don't do it yourself; keep your Dep.runs_le at SizesAt.
