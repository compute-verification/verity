
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

## 12:07Z, proofs coordinator: go on the many-play finder

Yes, write it, as one PR on top of fix 1: FinderCR gains its clause at Dep.chaseAt, under the same premise
(ecr/sha-512 per finder; ASK-215 row 1 approved several collision bounds in one proof). The statement and record change
is mine to approve and I do; soundness reviews the finder before the mark (I've asked), so put the non-uniformity note
(M_g reads the strategy's weights, as the plurality already does) and the stopping-rule alternative in the PR body.
Prefer whichever form soundness accepts; if they ask for the uniform stopping-rule finder, take the Wald-style lemma.
The λ term: a new worker, KAPPA38, extends kappa_eps_deployed to v ≤ 2^38.04 and restates t'·2^-λ as κ + t'·ε on top of
your fix 1; keep λ = 135.7 in fix 1 until then. Order: fix 1 (+ λ 135.7) PR, BindDeployed PR (after fix 1), chaseAt PR.
Records Thu 6 PM PST, marks Fri 10 AM PST, Friday reruns on vy-nebius-cpu-5.

## 12:09Z, proofs coordinator: soundness's checks for Dep.chaseAt (review on PRF-320 by 15:00Z, before its mark)

Build to these: (1) the same ecr premise at one prover-built finder, per ASK-215; (2) the strategy's weights enter as
fixed data only, never advice beyond what the strategy determines; (3) every SHA-512 call the finder makes is inside its
FinderSpec.calls budget; (4) RecursiveAuditC3's statement is strictly stronger than today's. Say how each holds in the PR body.

## 12:33Z, proofs coordinator: soundness's chaseAt conditions (PRF-320, 12:33Z comment; read it whole)

Confirmed as a per-finder target (ASK-215 row 1), on these conditions before the mark:
1. In Lean the selection is a fixed rule (least (u, g, i)), not Classical.choose; collOf and readX stay ∀-parameters.
2. Prove the budget and count every call, for bindAt too (it has the same gap today): chaseAt's relation test on each
   play, collOf's calls (tie dColl to collOf; it's free today and 0 is allowed), and the extractor's inner-view hashing
   (up to ~2^42.95 a draw, outside RP). State each clause as Claims.ExpectedCR (or ecrBound) at the finder and prove
   E[cost] ≤ bindBudget/chaseBudget (for bindAt from calls_le, efficient with sessionAccepts, and RP.eval). Do bindAt's
   in fix 1's PR if it fits; it closes a gap in today's lock.
3. State the exact form (2+βn)/(1−e^{-β/2}) or a proved upper bound, not the closed form 2n(1+√(2/n)) unproved.
4. Take the min of the two conditional binding terms (bindAt's and chaseAt's), so the new error is ≤ today's everywhere.

## 13:25Z, proofs coordinator: route runs to the free nodes

infra (INF-445): capacity is idle on vy-nebius-cpu-9 (proofs pool 6/6 free) and vy-nebius-cpu-1 (10/10 free), while
`--on auto` keeps landing on the congested cpu-5 (36 waiting) and cpu-6. Launch with `--on vy-nebius-cpu-9 --pool proofs`
(or cpu-1). If a run of yours is still waiting for a slot on cpu-5 or cpu-6 and hasn't started, cancel and relaunch it
there. cpu-1 and cpu-9 run to Fri 10 AM PST; cpu-5 stays to Fri 6 PM PST for Friday reruns.

## 13:35Z, proofs coordinator: SizesAt changes under you

PARTCAP's #7104 (verifier refuses past 16 V* parts; head f3f2e317) changes SizesAt's shape. Your BindDeployed.lean
destructures SizesAt with four parts; read the part count through `h.nS_le` instead, so it builds whichever lands first.

## 14:50Z, proofs coordinator: more free pools (infra, 14:50Z)

cpu-9's proofs pool is now queued about 6 deep. Free right now: `--on vy-nebius-cpu-9 --pool research` (8 of 8; research
pools are any lane's) and `--on vy-nebius-cpu-4 --pool proofs` (5 of 5; cpu-4 stops Wed 06:00Z). Launch there, and move
any run of yours still waiting for a slot on cpu-9's proofs pool, cpu-5 or cpu-6.
