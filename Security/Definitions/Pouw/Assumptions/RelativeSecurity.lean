import Definitions.Claims
import Definitions.Pouw.RelativeSecurity
import Definitions.Pouw.RelativeSecurityJ10R

/-!
# Relative security's named premises for the staged window (trusted)

The premises of the floors of record, `Definitions.Pouw.Guarantees.RelativeSecurity.StagedFloorKOneSided` and
`OverlapFloorKOneSided`, each about the analysis's map `run` from an adversary and its own coins to its window run
(`RunMap` on the staged design, `ORunMap` for stages that overlap). `StagedFloorKOneSided`'s three together are
`RoundsHypKStP` (`Proofs.Pouw.RelativeSecurity`) at `φ = toF2` and plane 0, split so that each is judged on its own:
what the service and its proof layer give (`ServiceSoundStaged`), what an adversary's run is (`KScopeOneSided`:
one-sided operands of any form, charge-pure), and what the machine charges for it (`ChargeVF3j16`).
`OverlapFloorKOneSided`'s three are the same split of `RoundsHypKOvP` (`ServiceSoundOverlap`, `KScopeOverlapOneSided`,
`ChargeVF3j16Overlap`). `KScopeAffine` and `KScopeOverlapAffine` are the affine scopes, row-pure, which the sibling
floors `StagedFloorK` and `OverlapFloorK`, proved on their own scope, take for the luck ¾ (`RoundsHypKSt`,
`RoundsHypKOv`). The ledger ids are the premise ledger's.

These are K's two proved scopes, and the floors' constant is `c = 1` in them. The one-sided scope is a sibling of the
ledger's K1, not a narrowing of it. **K1 is not claimed outside the two proved scopes.** There the ledger's K1 at
`c = 1` is false (RS-E-REF r8 VIII): a factor `∏(1 − z_u)` in one operand leaves a word wrong only at `U = 0`, so the
word is uncertified and passes almost always, and K's count binds certified words only. Outside the scopes Theorem O's
priced floor holds on paper only (`c ≥ 1 − δ` given CoincCharged_θ, `¼` without it; P2-FF r7–r8, RS-E-REF Part VI),
and no premise or guarantee here states it. **The verifier can't observe which case applies**: no check of the
service refuses a run outside a scope, so no premise here is a check's outcome.

**The service premises per prover.** `ServiceSoundStaged` and `ServiceSoundOverlap` bound every adversary's binding
break by one `δb`, which is false below 1 at SHA-512: a machine that hard-codes a collision opens a committed leaf two
ways (a red team's side finding). `ServiceSoundStagedPerProver` and
`ServiceSoundOverlapPerProver` restate them in the form C-Flock's link bound has (`LinkCRZC`, `linkBoundZC`): each
round's break is at most a constant plus a weight times the probability that the analysis's finder for that adversary
(`Finders`) finds a collision, and `ecr/sha-512` (`Claims.ExpectedCR`) enters only the floors' bound, at that
adversary's finders, `⊤` where they fail it, capped at 1 a value of its coins (`Claims.ecrBound`, `finderTerm`,
`breakBound`, all in `Definitions`, so other areas read the same break term). The bound reads each finder at a budget
of SHA-512 calls computed from the adversary's time, `c · time M + d` at stated constants per path, never at the
finder's declared cost, so no choice of finder lowers it. The constant forms stay, superseded,
because the floors of `Definitions.Pouw.Guarantees.RelativeSecurity` are proved from them; the floors per prover beside
them are proved from these.

**At call scope under J10-R**. J10-R draws H1's noise once a call, so a call's rows and
columns share their coins across the whole call, and a segment's coin × coin part is one `⟨M, 32, N⟩` product
(`Definitions.Pouw.RelativeSecurityJ10R`). Each premise above has a mirror on that call design, where a stage is a
call and its run one straight-line program over the call's bits: `ServiceSoundStagedJ10R` and its per-prover form,
`KScopeOneSidedJ10R` and `ChargeVF3j16J10R`; overlapped, `ServiceSoundOverlapJ10R`, its per-prover form,
`KScopeOverlapOneSidedJ10R` and `ChargeVF3j16OverlapJ10R`. They say what the per-unit forms say, read at the call: no
premise here is weakened for it. The charge counts every product of the call's program, so a cheaper algorithm for the
call's product, one whose products each serve several rows or columns, is charged fewer products and is outside the
call's one-sided scope (charging purity at `jFrame`): K1 is not claimed there, as it isn't per unit.
-/

namespace Pouw.Assumptions

open Pouw.RelativeSecurity Finset

section Staged

variable {R St n L np B n' : ℕ} {Ar : Type} [CommRing Ar]

/-- `pouw-rs-staged-service` (proposed). **The staged service is sound up to its proof layer.** The service's seed is
the window's live coins (`e`): `R` rounds, each `St` stage draws of the design's bits and one check of a stage and a
cell-block (S4, round protocol with C1 per stage and C4 per draw; S5, live coins on `trusted-host`). For every
adversary in `Mc` that commits first and every value `r` of its own coins, its run's play fixes each stage's error
table at the stage's operand commitment, before its draw (the play's form, `StagedPlay`); every run the verifier
accepts is accepted by the play's game or breaks binding or zero knowledge in some round (N1, X-RO commitments; F1 and
F2, C-Flock's binding and zero knowledge), at most `δb` and `δz` a round.

**Superseded by `ServiceSoundStagedPerProver`** (a red team's side finding).
`M` ranges over all of `Mc` with no cost bound, and one `δb` bounds them all, so the premise is false for every
`δb < 1` at SHA-512: a machine that hard-codes a collision opens a committed leaf two ways, and the third conjunct puts
its accepted runs in `Bind`. It stays only because `StagedFloorK` and `StagedFloorKOneSided` are proved from it. -/
def ServiceSoundStaged (Mc : Machine) (S : Service) (run : RunMap Mc S R St n L np B n' Ar)
    (e : S.Seed ≃ Paths (DStage St n L np B n') R) (δb δz : ℝ) : Prop :=
  ∀ (Ω' : Type) [Fintype Ω'] (M : Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)),
    (∀ σ σ' r, (Mc.eval M (σ, r)).1 = (Mc.eval M (σ', r)).1) → ∀ r : Ω',
      (∀ w, prob ((run Ω' M r).Bind w) ≤ δb) ∧ (∀ w, prob ((run Ω' M r).ZK w) ≤ δz) ∧
      ∀ x, S.verify (Mc.eval M (e.symm x, r)).1 (e.symm x) (Mc.eval M (e.symm x, r)).2 = true →
        (run Ω' M r).play.game.accepts x ∨ ∃ w, (run Ω' M r).Bind w x ∨ (run Ω' M r).ZK w x

/-- `pouw-rs-staged-service` (proposed; restates `ServiceSoundStaged` per prover). **The staged service is sound up to
its proof layer, each break bounded through the adversary's own collision finders.** The seed, the round protocol and
the play's form are `ServiceSoundStaged`'s (S4, round protocol with C1 per stage and C4 per draw; S5, live coins on
`trusted-host`; `StagedPlay`). For every adversary in `Mc` that commits first and every value `r` of its own
coins, every run the verifier accepts is accepted by the play's game or breaks binding or zero knowledge in some round
(N1, X-RO commitments; F1 and F2, C-Flock's binding and zero knowledge), and round `w`'s binding break has probability
at most `δb + βb · Pr[fb Ω' M r w finds a collision of sha]`, its zero-knowledge break `δz + βz · Pr[fz Ω' M r w …]`:
the linear term of `Claims.KSLin` for extraction-based finders, a constant that reads no prover (C-Flock's link has
`1/(eM) + k/(eR_w)` there, at weight `Q_s/(1 − ρ)`, `zk_linkSoundC`) plus the prover's own finder at its weight.

`Bind w` and `ZK w` are the run map's (`run Ω' M r`, the analysis's): events of `M`'s transcript at the live coins,
for its coins `r`. `fb` and `fz` are the analysis's reductions from those events to collisions of `sha`, built from
`M` at `r`, one a round. Binding is at the circuit's cuts, before each round's coin, which live coins allow: a round's
coin is released after the roots it reads, and the fork finder forks there. Under seeded coins nothing is bound before
a round's seed, so the seeded statements bind at the draw instead (`Accounting.Assumptions.Compose`, at
`Declared.atDraw`).

A square-root term folds into this form by AM–GM: `√(γ x) ≤ δ' + γ x/(4δ')` for every `δ' > 0`. So `Claims.KS`'s
`κ + √(γ · PrB)` instantiates the zero-knowledge conjunct at `δz = κ + δ'`, `βz = γ/(4δ')`, `fz` its `B`; a machine
whose `x` is about 1 still pays about `γ/(4δ')`.

The finders and their costs, path by path:
- **Binding: the fork, not a single run.** `Bind w` is the run's opening departing from the error table the play fixes
  from the commitment, a function of `(M, r)` that one run of `M` doesn't compute, so a finder that plays one run and
  reads the collision off it takes the table as advice its cost (`T`, one run's SHA-512 calls) doesn't count. The fork
  finder plays `M` twice from its commitment on fresh draws, and a fork bound `Pr[Bind w]² ≤ |L|² Pr[fork]` (the form
  of `Accounting.Guarantees.TS.BindingOfFork`, there at `|L| = 1`, over a drawn layer `L`) gives, by AM–GM, `δb = δ'`
  and `βb = |L|²/(4δ')`. Its cost is two runs of `M` and the verifier's checks of both openings: `cb` is twice the
  SHA-512 calls one unit of `Mc`'s time can buy, `db` twice the verifier's calls. The other honest instance counts
  the table: its cost is about the draws times `T`.
- **Zero knowledge: extraction first.** The openings a zero-knowledge break equivocates are witness, inside C-Flock's
  proofs, so the finder must extract them before it can compare: its cost is the knowledge extractor's, C-Flock's link
  finders' `2t'(1 + k)` calls at `t'` one run's (`linkBoundZC`), not `2T`. So `cz` is about `2(1 + k)` times the
  calls one unit of `Mc`'s time can buy, and `dz` counts the extractor's own calls. Where both the link finder and `Claims.KS`'s
  `B` are needed, `fz` runs both and outputs the first collision either finds, at their costs summed.

The floors per prover read this premise at the fork finders built from `M` (`forkFinders`: `1` continuation for
binding, `kz` for zero knowledge), never at finders of the analysis's choosing, and conclude at those finders' collision
probabilities (`breakPr`), so collision resistance is applied to them outside the floors.

It assumes nothing of SHA-512. `ecr/sha-512` enters only the floors' bound, adversary by adversary (`breakBound`), at a
budget computed from the adversary: a finder that meets `Claims.ExpectedCR` within `cb · time M + db` SHA-512 calls in
expectation (binding) or `cz · time M + dz` (zero knowledge) gives the budget over `2^256`, any other `⊤`
(`Claims.ecrBound`), and the floors cap each value of the adversary's coins at 1 before averaging. A finder's cost is a
declared field (`Claims.Finder.cost`), so the bound never reads it as a number. That closes the trivial instance: a
finder that outputs a fixed collision on a share `Pr[Bind w]` of its coins and declares `2^256` calls on them meets
`ExpectedCR` and makes this premise a tautology, but it exceeds the budget wherever `Pr[Bind w]` is above the budget
over `2^256`, where the floors then give 1. Below that, it gives the floors the budget's number, as the fork does.

A machine that hard-codes a collision is why. It passes every round while the play's game rejects, so by the third
conjunct its rounds' break bounds sum to about 1, and with `R (δb + δz)` small its finders carry that mass: they find
the collision. Finders that rerun it are cheap and fail `ecr/sha-512`; finders that meet `ecr/sha-512` make about
`2^256/βb` calls in expectation, past any budget at a feasible time. Either way the floors give it 1. Every other
adversary keeps `R (βb (cb · time M + db) + βz (cz · time M + dz)) / 2^256` on top of the constants, where
`ServiceSoundStaged` held for none of them below `δb` near 1.

`δz` is per prover too, because the zero-knowledge break can be hard-coded as well: where `ZK` reads an accepted proof
whose opening fails its relation, it is C-Flock's soundness, and C-Flock's knowledge bound has a collision term of the
prover's (`Claims.KS`'s `√(γ · PrB)`, `Claims.KSLin`'s `β · Pr[Fe]`, the compiled table's `ε_c⁻(τ) + Kr·adv₀(τ)`): a
session state holding two openings of one column-tree leaf passes with extraction failing. Its hiding part, hm96's
(`Claims.Hm96Hiding`), is statistical, a bound on every event at once, so no machine hard-codes it: an analysis whose
`ZK` is only hiding reads `βz = 0` and the constant `δz`. -/
def ServiceSoundStagedPerProver (Mc : Machine) (S : Service) (run : RunMap Mc S R St n L np B n' Ar)
    (e : S.Seed ≃ Paths (DStage St n L np B n') R) (sha : Claims.Msg → Claims.Msg) (fb fz : Finders Mc S R)
    (δb βb δz βz : ℝ) : Prop :=
  ∀ (Ω' : Type) [Fintype Ω'] (M : Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)),
    (∀ σ σ' r, (Mc.eval M (σ, r)).1 = (Mc.eval M (σ', r)).1) → ∀ r : Ω',
      (∀ w, prob ((run Ω' M r).Bind w) ≤ δb + βb * (fb Ω' M r w).Pr sha) ∧
      (∀ w, prob ((run Ω' M r).ZK w) ≤ δz + βz * (fz Ω' M r w).Pr sha) ∧
      ∀ x, S.verify (Mc.eval M (e.symm x, r)).1 (e.symm x) (Mc.eval M (e.symm x, r)).2 = true →
        (run Ω' M r).play.game.accepts x ∨ ∃ w, (run Ω' M r).Bind w x ∨ (run Ω' M r).ZK w x

/-- `pouw-rs-k-scope` (proposed). **Theorem K's proved affine scope.** For every adversary that commits first,
every value of its coins and every play `x`, each round's stage is a straight-line program over `Ar` on the stage's
bits (E3, straight-line; integer MMAs with s32 wrap and unscaled reads, so ring operations only), and:
- every product slot is a product of two bit-affine operands, mixed included (the luck ¾ is proved for them, LA);
- the stage's error table is its committed prefix words (`word`) minus their targets at the plane weights `wt` (S4,
  the check reads the prefix word);
- certified words of distinct rows of a stage share no product (row-pure).

On this scope `c = 1` is proved at the luck ¾, the sibling floor `StagedFloorK`'s. K1 is not claimed outside
the two proved scopes, this one and `KScopeOneSided`'s (one-sided operands of any form, charge-pure, at the parity luck
0.854: the floor of record's), and the verifier can't observe which case applies: no check of the service refuses a
run outside a scope (the ledger's "priced, not prevented"). -/
def KScopeAffine (Mc : Machine) (S : Service) (run : RunMap Mc S R St n L np B n' Ar) (wt : Fin np → Ar) : Prop :=
  ∀ (Ω' : Type) [Fintype Ω'] (M : Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)),
    (∀ σ σ' r, (Mc.eval M (σ, r)).1 = (Mc.eval M (σ', r)).1) → ∀ (r : Ω') x,
      (∀ w s j, ((run Ω' M r).prog x w s).isProd j = true → AffProd (((run Ω' M r).prog x w s).prod j)) ∧
      (∀ w s c, (run Ω' M r).play.errAt x w s c = fun z =>
        ((run Ω' M r).prog x w s).val z ((run Ω' M r).word x w s c) -
          dPrefix ((run Ω' M r).pc x w s) ((run Ω' M r).vq x w s) wt c.1 c.2.1 c.2.2.val z) ∧
      ∀ w s c c', (∀ U : Finset (DBits n L np B n'), U.card ≤ 2 → (run Ω' M r).play.errAt x w s c (bitInd U) = 0) →
        (∀ U : Finset (DBits n L np B n'), U.card ≤ 2 → (run Ω' M r).play.errAt x w s c' (bitInd U) = 0) →
        c.1 ≠ c'.1 → ∀ k, ((run Ω' M r).prog x w s).coef ((run Ω' M r).word x w s c) k ≠ 0 →
          ((run Ω' M r).prog x w s).coef ((run Ω' M r).word x w s c') k = 0

/-- `pouw-rs-k-scope-one-sided` (proposed). **Theorem K's proved one-sided scope, operands of any form.** For every
adversary that commits first, every value of its coins and every play `x`, each round's stage is a straight-line
program over `Ar` on the stage's bits (E3, straight-line; ring operations only, so reads are unscaled), and:
- every product slot is one-sided, of any form (`OneSided`: a function of U's bits times a function of the coins,
  neither side asked to be affine, so wider than K1's `GProd` on the coin side; RS-E-REF r6 VI.2.2);
- the stage's error table is its committed prefix words (`word`) minus their targets at the plane weights `wt` (S4,
  the check reads the prefix word);
- no product is two-sided, mod 2 through `φ`, at the frames (`dFrame`, plane `p₀`) of two cell-blocks of the stage
  whose words read it with odd coefficients (`SlpG.ChargePure`, charging purity, RS-E-REF r6 VI.2.1, in place of row
  purity). Prefix chains, merges and partners are free: a product of block `t'` that a later block's word reads
  reads none of that block's coins. A product two-sided at the frames of two columns of a row, which row purity
  allows, is outside it.

On this scope `c = 1` is proved at the parity luck 0.854, the floor of record `StagedFloorKOneSided`'s. It is a
sibling of the ledger's K1, not a narrowing (RS-E-REF r8 VIII.3 item 6): its coin side may be non-affine, which no
`GProd` is, and charging purity neither implies nor follows from row purity. K1 is not claimed outside the two proved
scopes, this one and `KScopeAffine`'s (affine operands, mixed included, row-pure, at ¾), and the verifier can't
observe which case applies: no check of the service refuses a run outside a scope. -/
def KScopeOneSided (Mc : Machine) (S : Service) (run : RunMap Mc S R St n L np B n' Ar) (wt : Fin np → Ar)
    (φ : Ar →+* ZMod 2) (p₀ : Fin np) : Prop :=
  ∀ (Ω' : Type) [Fintype Ω'] (M : Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)),
    (∀ σ σ' r, (Mc.eval M (σ, r)).1 = (Mc.eval M (σ', r)).1) → ∀ (r : Ω') x,
      (∀ w s j, ((run Ω' M r).prog x w s).isProd j = true → OneSided (((run Ω' M r).prog x w s).prod j)) ∧
      (∀ w s c, (run Ω' M r).play.errAt x w s c = fun z =>
        ((run Ω' M r).prog x w s).val z ((run Ω' M r).word x w s c) -
          dPrefix ((run Ω' M r).pc x w s) ((run Ω' M r).vq x w s) wt c.1 c.2.1 c.2.2.val z) ∧
      ∀ w s, ((run Ω' M r).prog x w s).ChargePure φ p₀ ((run Ω' M r).word x w s)

/-- `pouw-rs-charge-vf3j16` (proposed). **The machine charges a run at least `κ` a product.** For every adversary
that commits first, every value of its coins and every play `x`, the run's charge (`max(|M|, cost)`) is at least `κ`
times its products over every stage of every round, less its noise reads `V x ≤ Vmax`. With `Mc` the sm_120 machine
B_θ this is P1 at the active price vector VF3j-16's prices (decision 17: τ 0.0610, margin 2.000, every price at most
its measured cost, (π₃₂) closed for the whole census, every register load credited 16 a word)
with P4 (B_θ complete), and it sums the stages' products, so it
holds when a stage's interval, from its draw to its outputs' commitment, holds no other draw. Stages that overlap are
`ChargeVF3j16Overlap`'s. -/
def ChargeVF3j16 (Mc : Machine) (S : Service) (run : RunMap Mc S R St n L np B n' Ar)
    (e : S.Seed ≃ Paths (DStage St n L np B n') R) (κ Vmax : ℝ) : Prop :=
  ∀ (Ω' : Type) [Fintype Ω'] (M : Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)),
    (∀ σ σ' r, (Mc.eval M (σ, r)).1 = (Mc.eval M (σ', r)).1) → ∀ (r : Ω') x,
      (run Ω' M r).V x ≤ Vmax ∧
      κ * ∑ w, ∑ s, ((univ.filter fun j => ((run Ω' M r).prog x w s).isProd j = true).card : ℝ) - (run Ω' M r).V x ≤
        Mc.charge M (e.symm x, r)

end Staged

section Overlap

variable {R St L np : ℕ} {n B n' : Fin St → ℕ} {Ar : Type} [CommRing Ar]

/-- `pouw-rs-overlap-service` (proposed). **The overlapped service is sound up to its proof layer.** The service's
seed is the window's live coins (`e`): `R` rounds, each one draw a stage of the stage's design bits, the draws
overlapping at any depth, and one check of a cell-block drawn uniformly among every stage's (S4, round protocol with
C1 (i)–(ii) and C4 per draw; S5, live coins on `trusted-host`). For every adversary in `Mc` that commits first
and every value `r` of its own coins, its run's play fixes each round's error tables, every stage's, before the
round's first draw (the play's form, `OvPlay`; this is E3, the round's run straight-line over every draw's bits,
which K∥ reads in place of C1 (iii) and (iv), and which the service doesn't check); every run the verifier accepts is
accepted by the play's game or breaks binding or zero knowledge in some round (N1, X-RO commitments; F1 and F2,
C-Flock's binding and zero knowledge), at most `δb` and `δz` a round.

The uniform draw is a deployment requirement. `e` is a bijection onto `OSpot`'s paths, so the check's cell-block is
uniform on `OCells`, all `Σ_t n_t n'_t B_t` of them at once. A challenger that draws a stage first and then a
cell-block in it weights a small stage's cell-blocks above a large one's, and is outside this premise unless every
stage has one size.

**Superseded by `ServiceSoundOverlapPerProver`** (a red team's side
finding), for `ServiceSoundStaged`'s reason: one `δb` for every adversary is false below 1 at SHA-512, since a machine
that hard-codes a collision opens a committed leaf two ways. It stays only because `OverlapFloorK` and
`OverlapFloorKOneSided` are proved from it. -/
def ServiceSoundOverlap (Mc : Machine) (S : Service) (run : ORunMap Mc S R St n L np B n' Ar)
    (e : S.Seed ≃ Paths (OSpot St n L np B n') R) (δb δz : ℝ) : Prop :=
  ∀ (Ω' : Type) [Fintype Ω'] (M : Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)),
    (∀ σ σ' r, (Mc.eval M (σ, r)).1 = (Mc.eval M (σ', r)).1) → ∀ r : Ω',
      (∀ w, prob ((run Ω' M r).Bind w) ≤ δb) ∧ (∀ w, prob ((run Ω' M r).ZK w) ≤ δz) ∧
      ∀ x, S.verify (Mc.eval M (e.symm x, r)).1 (e.symm x) (Mc.eval M (e.symm x, r)).2 = true →
        (run Ω' M r).play.game.accepts x ∨ ∃ w, (run Ω' M r).Bind w x ∨ (run Ω' M r).ZK w x

/-- `pouw-rs-overlap-service` (proposed; restates `ServiceSoundOverlap` per prover). **The overlapped service is sound
up to its proof layer, each break bounded through the adversary's own collision finders.** The seed, the round
protocol, the uniform draw and the play's form are `ServiceSoundOverlap`'s (S4, round protocol with C1 (i)–(ii) and C4
per draw; S5, live coins on `trusted-host`; `OvPlay`, which is E3). For every adversary in `Mc` that commits
first and every value `r` of its own coins, every run the verifier accepts is accepted by the play's game or breaks
binding or zero knowledge in some round (N1, X-RO commitments; F1 and F2, C-Flock's binding and zero knowledge), round
`w`'s binding break at most `δb + βb · Pr[fb Ω' M r w finds a collision of sha]` and its zero-knowledge break
`δz + βz · Pr[fz Ω' M r w …]`. Everything `ServiceSoundStagedPerProver` says of its form holds here:
- it assumes nothing of SHA-512;
- `Bind w` and `ZK w` are the run map's, and `fb`, `fz` the analysis's reductions built from `M` at `r`;
- a square-root term folds in by AM–GM;
- the binding finder's honest instance is the fork (`δb = δ'`, `βb = |L|²/(4δ')`, two runs of `M`), and the
  zero-knowledge finder extracts first (C-Flock's link finders, about `2(1 + k)` runs);
- `ecr/sha-512` enters the floors' bound at each adversary's finders (`breakBound`), at budgets `cb · time M + db` and
  `cz · time M + dz` computed from the adversary, never at a finder's declared cost;
- a machine that hard-codes a collision gets 1 there;
- `δz` is per prover because C-Flock's soundness has a collision term of the prover's, while hm96's hiding is
  statistical (`βz = 0` for an analysis whose `ZK` is only hiding). -/
def ServiceSoundOverlapPerProver (Mc : Machine) (S : Service) (run : ORunMap Mc S R St n L np B n' Ar)
    (e : S.Seed ≃ Paths (OSpot St n L np B n') R) (sha : Claims.Msg → Claims.Msg) (fb fz : Finders Mc S R)
    (δb βb δz βz : ℝ) : Prop :=
  ∀ (Ω' : Type) [Fintype Ω'] (M : Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)),
    (∀ σ σ' r, (Mc.eval M (σ, r)).1 = (Mc.eval M (σ', r)).1) → ∀ r : Ω',
      (∀ w, prob ((run Ω' M r).Bind w) ≤ δb + βb * (fb Ω' M r w).Pr sha) ∧
      (∀ w, prob ((run Ω' M r).ZK w) ≤ δz + βz * (fz Ω' M r w).Pr sha) ∧
      ∀ x, S.verify (Mc.eval M (e.symm x, r)).1 (e.symm x) (Mc.eval M (e.symm x, r)).2 = true →
        (run Ω' M r).play.game.accepts x ∨ ∃ w, (run Ω' M r).Bind w x ∨ (run Ω' M r).ZK w x

/-- `pouw-rs-k-scope-overlap-affine` (proposed). **Theorem K∥'s proved affine scope, operands affine per draw.** For
every adversary that commits first, every value of its coins and every play `x`, each round's run is one
straight-line program over `Ar` on every draw's bits (E3), and:
- every product slot is, at every bit point of the other draws', a product of two operands affine in draw `t`'s bits,
  for every draw `t` (`AffMulAt`: affine in each draw's own bits, as `KScopeAffine`'s operands are in the stage's; an
  operand may depend on the other draws' bits in any way, nonlinear included, as the served operands' later stages do
  on earlier draws);
- stage `t`'s error table is its committed prefix words (`word`) minus their targets at the plane weights `wt`, read
  on draw `t`'s bits (S4);
- at every bit point `W` of the draws, certified words of distinct rows, of any stages, share no product (row-pure
  across draws; a word is certified at `W` where its error vanishes on its own draw's bits with at most two ones, the
  other draws' at `W`).

On this scope `c = 1` is proved at the luck ¾ per draw, the sibling floor `OverlapFloorK`'s. K1 is not claimed
outside the two proved scopes across draws, this one and `KScopeOverlapOneSided`'s (one-sided operands of any form
per draw, charge-pure across draws, at the parity luck 0.854: the floor of record's), and the verifier can't observe
which case applies: no check of the service refuses a run outside a scope. -/
def KScopeOverlapAffine (Mc : Machine) (S : Service) (run : ORunMap Mc S R St n L np B n' Ar) (wt : Fin np → Ar) :
    Prop :=
  ∀ (Ω' : Type) [Fintype Ω'] (M : Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)),
    (∀ σ σ' r, (Mc.eval M (σ, r)).1 = (Mc.eval M (σ', r)).1) → ∀ (r : Ω') x,
      (∀ w j, ((run Ω' M r).prog x w).isProd j = true → AffMulAt (((run Ω' M r).prog x w).prod j)) ∧
      (∀ w c, (run Ω' M r).play.errAt x w c = fun z => ((run Ω' M r).prog x w).val z ((run Ω' M r).word x w c) -
        dPrefix ((run Ω' M r).pc x w c.1) ((run Ω' M r).vq x w c.1) wt c.2.1 c.2.2.1 c.2.2.2.val fun v => z ⟨c.1, v⟩) ∧
      ∀ w (W : ODraws St n L np B n') c c',
        (∀ U : Finset (DBits (n c.1) L np (B c.1) (n' c.1)), U.card ≤ 2 →
          (run Ω' M r).play.errAt x w c (mergeAt W c.1 (bitInd U)) = 0) →
        (∀ U : Finset (DBits (n c'.1) L np (B c'.1) (n' c'.1)), U.card ≤ 2 →
          (run Ω' M r).play.errAt x w c' (mergeAt W c'.1 (bitInd U)) = 0) →
        (⟨c.1, c.2.1⟩ : Σ t, Fin (n t)) ≠ ⟨c'.1, c'.2.1⟩ → ∀ k,
          ((run Ω' M r).prog x w).coef ((run Ω' M r).word x w c) k ≠ 0 →
            ((run Ω' M r).prog x w).coef ((run Ω' M r).word x w c') k = 0

/-- `pouw-rs-k-scope-overlap-one-sided` (proposed). **Theorem K∥'s proved one-sided scope, operands of any form per
draw.** For every adversary that commits first, every value of its coins and every play `x`, each round's run is one
straight-line program over `Ar` on every draw's bits (E3), and:
- every product slot is, at every bit point of the other draws', one-sided of any form on draw `t`'s bits, for every
  draw `t` (`OneSidedAt`: `KScopeOneSided`'s products per draw; each side may depend on the other draws' bits in any
  way);
- stage `t`'s error table is its committed prefix words (`word`) minus their targets at the plane weights `wt`, read
  on draw `t`'s bits (S4);
- at every bit point of the draws, no product is two-sided, mod 2 through `φ`, at the frames (plane `p₀`) of two
  cell-blocks, of any stages, whose words read it with odd coefficients (`SlpG.ChargePureAt`, charging purity across
  draws, in place of row purity).

On this scope `c = 1` is proved at the parity luck 0.854 per draw, the floor of record `OverlapFloorKOneSided`'s.
Like `KScopeOneSided` it is a sibling of the ledger's K1, not a narrowing. K1 is not claimed outside the two proved
scopes across draws, this one and `KScopeOverlapAffine`'s (operands affine per draw, row-pure across draws, at ¾),
and the verifier can't observe which case applies: no check of the service refuses a run outside a scope. -/
def KScopeOverlapOneSided (Mc : Machine) (S : Service) (run : ORunMap Mc S R St n L np B n' Ar) (wt : Fin np → Ar)
    (φ : Ar →+* ZMod 2) (p₀ : Fin np) : Prop :=
  ∀ (Ω' : Type) [Fintype Ω'] (M : Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)),
    (∀ σ σ' r, (Mc.eval M (σ, r)).1 = (Mc.eval M (σ', r)).1) → ∀ (r : Ω') x,
      (∀ w j, ((run Ω' M r).prog x w).isProd j = true → OneSidedAt (((run Ω' M r).prog x w).prod j)) ∧
      (∀ w c, (run Ω' M r).play.errAt x w c = fun z => ((run Ω' M r).prog x w).val z ((run Ω' M r).word x w c) -
        dPrefix ((run Ω' M r).pc x w c.1) ((run Ω' M r).vq x w c.1) wt c.2.1 c.2.2.1 c.2.2.2.val fun v => z ⟨c.1, v⟩) ∧
      ∀ w, ((run Ω' M r).prog x w).ChargePureAt φ p₀ ((run Ω' M r).word x w)

/-- `pouw-rs-charge-vf3j16-overlap` (proposed). **The machine charges an overlapped run at least `κ` a product.**
For every adversary that commits first, every value of its coins and every play `x`, the run's charge is at least `κ`
times the products of every round's run, less its noise reads `V x ≤ Vmax`. With `Mc` the sm_120 machine B_θ this is
P1 at VF3j-16's prices with P4, as `ChargeVF3j16`; it counts each product once, whatever draws' intervals it lies in,
so it reads nothing of the schedule. -/
def ChargeVF3j16Overlap (Mc : Machine) (S : Service) (run : ORunMap Mc S R St n L np B n' Ar)
    (e : S.Seed ≃ Paths (OSpot St n L np B n') R) (κ Vmax : ℝ) : Prop :=
  ∀ (Ω' : Type) [Fintype Ω'] (M : Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)),
    (∀ σ σ' r, (Mc.eval M (σ, r)).1 = (Mc.eval M (σ', r)).1) → ∀ (r : Ω') x,
      (run Ω' M r).V x ≤ Vmax ∧
      κ * ∑ w, ((univ.filter fun j => ((run Ω' M r).prog x w).isProd j = true).card : ℝ) - (run Ω' M r).V x ≤
        Mc.charge M (e.symm x, r)

end Overlap

section J10R

variable {R St n L B n' : ℕ} {Ar : Type} [CommRing Ar]

/-- `pouw-rs-staged-service` (proposed) at call scope under J10-R: `ServiceSoundStaged` on the call design. The
service's seed is the window's live coins (`e`): `R` rounds, each `St` call draws of the call's bits (`JStage`: every
row's dither bits, every column's coin bits and `U`'s bytes, drawn once a call) and one check of a call and a
cell-block. For every adversary in `Mc` that commits first and every value `r` of its own coins, its run's play fixes
each call's error table at the call's operand commitment, before its draw; every run the verifier accepts is accepted
by the play's game or breaks binding or zero knowledge in some round, at most `δb` and `δz` a round.

**Superseded by `ServiceSoundStagedJ10RPerProver`**, for `ServiceSoundStaged`'s reason: one `δb` for every adversary is
false below 1 at SHA-512. It mirrors the premise today's floor of
record is proved from, so that the call-scope floor `StagedFloorKOneSidedJ10R` is that floor read at the call. -/
def ServiceSoundStagedJ10R (Mc : Machine) (S : Service) (run : JRunMap Mc S R St n L B n' Ar)
    (e : S.Seed ≃ Paths (JStage St n L B n') R) (δb δz : ℝ) : Prop :=
  ∀ (Ω' : Type) [Fintype Ω'] (M : Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)),
    (∀ σ σ' r, (Mc.eval M (σ, r)).1 = (Mc.eval M (σ', r)).1) → ∀ r : Ω',
      (∀ w, prob ((run Ω' M r).Bind w) ≤ δb) ∧ (∀ w, prob ((run Ω' M r).ZK w) ≤ δz) ∧
      ∀ x, S.verify (Mc.eval M (e.symm x, r)).1 (e.symm x) (Mc.eval M (e.symm x, r)).2 = true →
        (run Ω' M r).play.game.accepts x ∨ ∃ w, (run Ω' M r).Bind w x ∨ (run Ω' M r).ZK w x

/-- `pouw-rs-staged-service` (proposed; per prover) at call scope under J10-R: `ServiceSoundStagedPerProver` on the
call design. The seed, the round protocol and the play's form are `ServiceSoundStagedJ10R`'s; round `w`'s binding
break is at most `δb + βb · Pr[fb Ω' M r w finds a collision of sha]` and its zero-knowledge break
`δz + βz · Pr[fz Ω' M r w …]`. Everything `ServiceSoundStagedPerProver` says of its form, finders and budgets holds
here unchanged: it assumes nothing of SHA-512, and `ecr/sha-512` enters only the floors' bound (`breakBound`). -/
def ServiceSoundStagedJ10RPerProver (Mc : Machine) (S : Service) (run : JRunMap Mc S R St n L B n' Ar)
    (e : S.Seed ≃ Paths (JStage St n L B n') R) (sha : Claims.Msg → Claims.Msg) (fb fz : Finders Mc S R)
    (δb βb δz βz : ℝ) : Prop :=
  ∀ (Ω' : Type) [Fintype Ω'] (M : Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)),
    (∀ σ σ' r, (Mc.eval M (σ, r)).1 = (Mc.eval M (σ', r)).1) → ∀ r : Ω',
      (∀ w, prob ((run Ω' M r).Bind w) ≤ δb + βb * (fb Ω' M r w).Pr sha) ∧
      (∀ w, prob ((run Ω' M r).ZK w) ≤ δz + βz * (fz Ω' M r w).Pr sha) ∧
      ∀ x, S.verify (Mc.eval M (e.symm x, r)).1 (e.symm x) (Mc.eval M (e.symm x, r)).2 = true →
        (run Ω' M r).play.game.accepts x ∨ ∃ w, (run Ω' M r).Bind w x ∨ (run Ω' M r).ZK w x

/-- `pouw-rs-k-scope-one-sided` (proposed) at call scope under J10-R: `KScopeOneSided` on the call design. For every
adversary that commits first, every value of its coins and every play `x`, each round's call is one straight-line
program over `Z/2³²` on the call's bits (E3; ring operations only), and:
- every product slot is one-sided, of any form: a function of the rows' dither bits times a function of the columns'
  coin bits and `U`'s bytes (`OneSided`);
- the call's error table is its committed prefix words (`word`) minus their targets (`jPrefix`: the int8 operands
  `s8 (A + b)` and `s8 (W + c + U)`, as served; S4, the check reads the prefix word);
- no product is two-sided, mod 2, at the frames (`jFrame`: a row's dither bits and a column's coin bits of one
  segment) of two cell-blocks whose words read it with odd coefficients (`SlpG.ChargePureJ`).

A product shared by the call's rows or columns of a segment (a fast algorithm for the call's `⟨M, 32, N⟩` product, or
a selection of shared streams) is two-sided at several frames and outside it. K1 is not claimed outside the scope, and
the verifier can't observe which case applies: no check of the service refuses a run outside it. -/
def KScopeOneSidedJ10R (Mc : Machine) (S : Service) (run : JRunMap Mc S R St n L B n' (ZMod (2 ^ 32))) : Prop :=
  ∀ (Ω' : Type) [Fintype Ω'] (M : Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)),
    (∀ σ σ' r, (Mc.eval M (σ, r)).1 = (Mc.eval M (σ', r)).1) → ∀ (r : Ω') x,
      (∀ w s j, ((run Ω' M r).prog x w s).isProd j = true → OneSided (((run Ω' M r).prog x w s).prod j)) ∧
      (∀ w s c, (run Ω' M r).play.errAt x w s c = fun z =>
        ((run Ω' M r).prog x w s).val z ((run Ω' M r).word x w s c) -
          jPrefix ((run Ω' M r).aq x w s) ((run Ω' M r).wq x w s) c.1 c.2.1 c.2.2.val z) ∧
      ∀ w s, ((run Ω' M r).prog x w s).ChargePureJ toF2 ((run Ω' M r).word x w s)

/-- `pouw-rs-charge-vf3j16` (proposed) at call scope under J10-R. **The machine charges a run at least `κ` a product
of each call's program.** For every adversary that commits first, every value of its coins and every play `x`, the
run's charge (`max(|M|, cost)`) is at least `κ` times the products of every call's straight-line program over the
call's bits (the call's rows' and columns' coins, shared across the call, each segment's coin × coin part one
`⟨M, 32, N⟩` product), summed over every call of every round, less its noise reads `V x ≤ Vmax`. With `Mc` the sm_120
machine B_θ this is P1 at VF3j-16's prices with P4, as `ChargeVF3j16`, read at the call rather than the unit: it
holds when a call's interval, from its draw to its outputs' commitment, holds no other draw. Overlapping calls are
`ChargeVF3j16OverlapJ10R`'s. -/
def ChargeVF3j16J10R (Mc : Machine) (S : Service) (run : JRunMap Mc S R St n L B n' Ar)
    (e : S.Seed ≃ Paths (JStage St n L B n') R) (κ Vmax : ℝ) : Prop :=
  ∀ (Ω' : Type) [Fintype Ω'] (M : Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)),
    (∀ σ σ' r, (Mc.eval M (σ, r)).1 = (Mc.eval M (σ', r)).1) → ∀ (r : Ω') x,
      (run Ω' M r).V x ≤ Vmax ∧
      κ * ∑ w, ∑ s, ((univ.filter fun j => ((run Ω' M r).prog x w s).isProd j = true).card : ℝ) - (run Ω' M r).V x ≤
        Mc.charge M (e.symm x, r)

end J10R

section J10ROverlap

variable {R St L : ℕ} {n B n' : Fin St → ℕ} {Ar : Type} [CommRing Ar]

/-- `pouw-rs-overlap-service` (proposed) at call scope under J10-R: `ServiceSoundOverlap` on the call design. `R`
rounds, each one draw a call of the call's bits, the draws overlapping at any depth, and one check of a cell-block
drawn uniformly among every call's (the uniform draw is `ServiceSoundOverlap`'s deployment requirement); the play fixes
each round's error tables, every call's, before the round's first draw (E3); every accepted run is accepted by the
play's game or breaks binding or zero knowledge in some round, at most `δb` and `δz` a round.

**Superseded by `ServiceSoundOverlapJ10RPerProver`**, for `ServiceSoundOverlap`'s reason. It mirrors the premise
today's overlapped floor of record is proved from. -/
def ServiceSoundOverlapJ10R (Mc : Machine) (S : Service) (run : JORunMap Mc S R St n L B n' Ar)
    (e : S.Seed ≃ Paths (OJSpot St n L B n') R) (δb δz : ℝ) : Prop :=
  ∀ (Ω' : Type) [Fintype Ω'] (M : Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)),
    (∀ σ σ' r, (Mc.eval M (σ, r)).1 = (Mc.eval M (σ', r)).1) → ∀ r : Ω',
      (∀ w, prob ((run Ω' M r).Bind w) ≤ δb) ∧ (∀ w, prob ((run Ω' M r).ZK w) ≤ δz) ∧
      ∀ x, S.verify (Mc.eval M (e.symm x, r)).1 (e.symm x) (Mc.eval M (e.symm x, r)).2 = true →
        (run Ω' M r).play.game.accepts x ∨ ∃ w, (run Ω' M r).Bind w x ∨ (run Ω' M r).ZK w x

/-- `pouw-rs-overlap-service` (proposed; per prover) at call scope under J10-R: `ServiceSoundOverlapPerProver` on the
call design, with `ServiceSoundOverlapJ10R`'s seed, round protocol, uniform draw and play, and each round's breaks
bounded through the adversary's own collision finders as `ServiceSoundOverlapPerProver` states. -/
def ServiceSoundOverlapJ10RPerProver (Mc : Machine) (S : Service) (run : JORunMap Mc S R St n L B n' Ar)
    (e : S.Seed ≃ Paths (OJSpot St n L B n') R) (sha : Claims.Msg → Claims.Msg) (fb fz : Finders Mc S R)
    (δb βb δz βz : ℝ) : Prop :=
  ∀ (Ω' : Type) [Fintype Ω'] (M : Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)),
    (∀ σ σ' r, (Mc.eval M (σ, r)).1 = (Mc.eval M (σ', r)).1) → ∀ r : Ω',
      (∀ w, prob ((run Ω' M r).Bind w) ≤ δb + βb * (fb Ω' M r w).Pr sha) ∧
      (∀ w, prob ((run Ω' M r).ZK w) ≤ δz + βz * (fz Ω' M r w).Pr sha) ∧
      ∀ x, S.verify (Mc.eval M (e.symm x, r)).1 (e.symm x) (Mc.eval M (e.symm x, r)).2 = true →
        (run Ω' M r).play.game.accepts x ∨ ∃ w, (run Ω' M r).Bind w x ∨ (run Ω' M r).ZK w x

/-- `pouw-rs-k-scope-overlap-one-sided` (proposed) at call scope under J10-R: `KScopeOverlapOneSided` on the call
design. For every adversary that commits first, every value of its coins and every play `x`, each round's run is one
straight-line program over `Z/2³²` on every call's bits (E3), and:
- every product slot is, at every bit point of the other calls', one-sided of any form on call `t`'s bits (its rows'
  against its columns'), for every call `t` (`OneSidedAtJ`);
- call `t`'s error table is its committed prefix words (`word`) minus their targets (`jPrefix`), read on call `t`'s
  bits (S4);
- at every bit point of the draws, no product is two-sided, mod 2, at the frames (`jFrame`) of two cell-blocks, of any
  calls, whose words read it with odd coefficients (`SlpG.ChargePureAtJ`).

K1 is not claimed outside the scope, and the verifier can't observe which case applies. -/
def KScopeOverlapOneSidedJ10R (Mc : Machine) (S : Service) (run : JORunMap Mc S R St n L B n' (ZMod (2 ^ 32))) :
    Prop :=
  ∀ (Ω' : Type) [Fintype Ω'] (M : Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)),
    (∀ σ σ' r, (Mc.eval M (σ, r)).1 = (Mc.eval M (σ', r)).1) → ∀ (r : Ω') x,
      (∀ w j, ((run Ω' M r).prog x w).isProd j = true → OneSidedAtJ (((run Ω' M r).prog x w).prod j)) ∧
      (∀ w c, (run Ω' M r).play.errAt x w c = fun z => ((run Ω' M r).prog x w).val z ((run Ω' M r).word x w c) -
        jPrefix ((run Ω' M r).aq x w c.1) ((run Ω' M r).wq x w c.1) c.2.1 c.2.2.1 c.2.2.2.val fun v => z ⟨c.1, v⟩) ∧
      ∀ w, ((run Ω' M r).prog x w).ChargePureAtJ toF2 ((run Ω' M r).word x w)

/-- `pouw-rs-charge-vf3j16-overlap` (proposed) at call scope under J10-R. **The machine charges an overlapped run at
least `κ` a product of each round's program over its calls' bits.** For every adversary that commits first, every
value of its coins and every play `x`, the run's charge is at least `κ` times the products of every round's run, one
straight-line program over every call's bits (each call's rows' and columns' coins shared across the call), less its
noise reads `V x ≤ Vmax`. With `Mc` the sm_120 machine B_θ this is P1 at VF3j-16's prices with P4, as
`ChargeVF3j16Overlap`; it counts each product once, whatever calls' intervals it lies in. -/
def ChargeVF3j16OverlapJ10R (Mc : Machine) (S : Service) (run : JORunMap Mc S R St n L B n' Ar)
    (e : S.Seed ≃ Paths (OJSpot St n L B n') R) (κ Vmax : ℝ) : Prop :=
  ∀ (Ω' : Type) [Fintype Ω'] (M : Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)),
    (∀ σ σ' r, (Mc.eval M (σ, r)).1 = (Mc.eval M (σ', r)).1) → ∀ (r : Ω') x,
      (run Ω' M r).V x ≤ Vmax ∧
      κ * ∑ w, ((univ.filter fun j => ((run Ω' M r).prog x w).isProd j = true).card : ℝ) - (run Ω' M r).V x ≤
        Mc.charge M (e.symm x, r)

end J10ROverlap

end Pouw.Assumptions
