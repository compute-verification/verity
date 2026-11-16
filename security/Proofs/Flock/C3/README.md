# Candidate 3's recursive guarantee

In a recursive audit an outer Flock session proves that the verifier would accept an inner one. Candidate 3, the Holo
design, is the inner protocol whose circuits are hidden: each proof block's template is one of a registered table R's,
opened by the sparse phase and R's registered opening. This directory holds the pieces of the proof that the recursive
audit is sound at candidate 3. The guarantee, what it takes and what stays open are below; [`DESIGN.md`](DESIGN.md)
has the decisions behind the model.

## Try it

~~~sh
bash tools/verity/lean/setup.sh security/Proofs
(cd security/Proofs && lake build Proofs.Flock.C3)
~~~

The first line installs the toolchain and the packages the proofs require and restores `main`'s build from the Lean
cache; the second builds and checks this directory's proofs.

## The guarantee

`Flock.SecurityProofs.RecursiveAuditHolo : Flock.Guarantees.RecursiveAuditHolo` (`Recursive/AuditC3.lean`,
`Recursive/AuditC3Proof.lean`), unlisted, is PublicAudit's proof at candidate 3's table (`holoInnerAt`, R from the registration's
`stacked` entry, V*'s statements over the verifier's own list). It has no `sorry`. It takes VBridge (`_hbr`) and V*'s
parse (`_hob`) as PublicAudit does; `C3/VStarList/Statements.lean` discharges `_hob`, and VBridge stays open (below). V*'s statements are indexed by the
verifier's own list (`vstarCount ish`), which at candidate 3's inner shape is eleven statements in the writer's order:
0–3 `rec-open`, 4–5 `rec-residuals`, 6 `rec-sparse`, 7–9 R's levels `rec-open-stacked`, 10 `rec-opening` at META's
`stacked` shape.

Its soundness (i) is in fork form and states no bits. The recursion's knowledge form, `Claims.Recursion.KS`
(`security/Definitions/Claims/Recursion.lean`), is at `Claims.Recursion.lam`, which no theorem evaluates at the
deployment. Route 2C's bound λ₂ (`lamLin2`) evaluates to at least 135.7 at the deployed records
(`RecDeploy.lamLin2_deployed_ge_dep_tight`; at least 100, `RecDeploy.lamLin2_deployed_ge_dep`), and `KSLin2` at the
deployment's dense statements (`KSLin2On` at `RecDeploy.DenseAt`, at the deployed reader) is proved with no hypothesis
(`Flock.Guarantees.RecursiveKS`, `Flock.SecurityProofs.RecursiveKS` in `Recursive/KSProved.lean`). The listed
recursive audit, `Flock.Guarantees.RecursiveAuditC3` (`Proofs/Properties/Verity/Recursive/Audit.lean`), is that dense
form, the one `verity/sampled-proofs/recursive/v1` accepts, at `λ₂ = 135.7`, and at a statement at the deployment's
tables also at route 2C's split bound `κ + t'·ε`, `κ ≤ 2^{-167.7}` and `ε ≤ 2^{-174.95}` at the run's call caps
(`RecDeploy.kappa_eps_oneTable_cap`). Holo statements are candidate 3's own model, and `BitsLin2` at the dense
statements follows from `KSLin2` there (`bitsLin2On_of_ksLin2On`).

What stays open is VBridge: the proof reads VBridgeCOff as a premise (`Recursive/AuditC3Proof.lean`), and
`C3/Compose/Honest.lean` defines the honest family its discharge is stated at (`C3Family.honest`).

No test replays honest R levels through `verify` with candidate-3 coins, so `checkTopReads`' acceptance of an honest
prover is untested (lean-agreement does not reach it). Its honesty argument is the prover's code: R's level-`l` tops are
`rec_vstage.stage_stacked`'s `rec-top-stacked-L<l>`, and opening `j`'s top is at node `pos[j] >>> H`
(`rec_outer.TopValue.position`).

## Weakness

R's level 0 at `fast100` is queried at the unique-decoding radius (rate 1/2, 218 queries): `udQueryError` is about
2^-88.4, so `holoError` is at least that, against PublicAudit's list-decoding terms. Bits to cite: 88, not `fast100`'s
100, which is its list-decoding figure; 100 bits at the unique-decoding radius needs about 247 level-0 queries for R.

A prover registers R (Decisions), and the statement covers only a registration whose cap the registrant can open at
every level-0 position (`_hreg`): a prover who registers a cap with no preimage is in no event the guarantee bounds.
