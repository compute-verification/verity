import Definitions.Pouw.RelativeSecurity

/-!
# Relative security's named premises for the staged window (trusted)

The premises of the floors of record, `Models.Pouw.Guarantees.RelativeSecurity.StagedFloorKOneSided` and
`OverlapFloorKOneSided`, each about the analysis's map `run` from an adversary and its own coins to its window run
(`RunMap` on the staged design, `ORunMap` for stages that overlap). `StagedFloorKOneSided`'s three together are
`RoundsHypKStP` (`Proofs.Pouw.RelativeSecurity`) at `φ = toF2` and plane 0, split so that each is judged on its own:
what the service and its proof layer give (`ServiceSoundStaged`), what an adversary's run is (`KScopeOneSided`:
one-sided operands of any form, charge-pure), and what the machine charges for it (`ChargeVF3j16`).
`OverlapFloorKOneSided`'s three are the same split of `RoundsHypKOvP` (`ServiceSoundOverlap`, `KScopeOverlapOneSided`,
`ChargeVF3j16Overlap`). `KScopeAffine` and `KScopeOverlapAffine` are the affine scopes, row-pure, which the sharper
corollaries `StagedFloorK` and `OverlapFloorK` take for the luck ¾ (`RoundsHypKSt`, `RoundsHypKOv`). The ledger ids
are the premise ledger's.

These are K's two proved scopes, and the floors' constant is `c = 1` in them. The one-sided scope is a sibling of the
ledger's K1, not a narrowing of it. **K1 is not claimed outside the two proved scopes.** There the ledger's K1 at
`c = 1` is false (RS-E-REF r8 VIII): a factor `∏(1 − z_u)` in one operand leaves a word wrong only at `U = 0`, so the
word is uncertified and passes almost always, and K's count binds certified words only. Outside the scopes Theorem O's
priced floor holds on paper only (`c ≥ 1 − δ` given CoincCharged_θ, `¼` without it; P2-FF r7–r8, RS-E-REF Part VI),
and no premise or guarantee here states it. **The verifier can't observe which case applies**: no check of the
service refuses a run outside a scope, so no premise here is a check's outcome.
-/

namespace Pouw.Assumptions

open Pouw.RelativeSecurity Finset

section Staged

variable {R St n L np B n' : ℕ} {Ar : Type} [CommRing Ar]

/-- `pouw-rs-staged-service` (proposed). **The staged service is sound up to its proof layer.** The service's seed is
the window's live coins (`e`): `R` rounds, each `St` stage draws of the design's bits and one check of a stage and a
cell-block (S4, round protocol with C1 per stage and C4 per draw; S5, live coins on `trusted-nebius-host`). For every
adversary in `Mc` that commits first and every value `r` of its own coins, its run's play fixes each stage's error
table at the stage's operand commitment, before its draw (the play's form, `StagedPlay`); every run the verifier
accepts is accepted by the play's game or breaks binding or zero knowledge in some round (N1, X-RO commitments; F1 and
F2, C-Flock's binding and zero knowledge), at most `δb` and `δz` a round. -/
def ServiceSoundStaged (Mc : Machine) (S : Service) (run : RunMap Mc S R St n L np B n' Ar)
    (e : S.Seed ≃ Paths (DStage St n L np B n') R) (δb δz : ℝ) : Prop :=
  ∀ (Ω' : Type) [Fintype Ω'] (M : Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)),
    (∀ σ σ' r, (Mc.eval M (σ, r)).1 = (Mc.eval M (σ', r)).1) → ∀ r : Ω',
      (∀ w, prob ((run Ω' M r).Bind w) ≤ δb) ∧ (∀ w, prob ((run Ω' M r).ZK w) ≤ δz) ∧
      ∀ x, S.verify (Mc.eval M (e.symm x, r)).1 (e.symm x) (Mc.eval M (e.symm x, r)).2 = true →
        (run Ω' M r).play.game.accepts x ∨ ∃ w, (run Ω' M r).Bind w x ∨ (run Ω' M r).ZK w x

/-- `pouw-rs-k-scope` (proposed). **Theorem K's proved affine scope.** For every adversary that commits first,
every value of its coins and every play `x`, each round's stage is a straight-line program over `Ar` on the stage's
bits (E3, straight-line; integer MMAs with s32 wrap and unscaled reads, so ring operations only), and:
- every product slot is a product of two bit-affine operands, mixed included (the luck ¾ is proved for them, LA);
- the stage's error table is its committed prefix words (`word`) minus their targets at the plane weights `wt` (S4,
  the check reads the prefix word);
- certified words of distinct rows of a stage share no product (row-pure).

On this scope `c = 1` is proved at the luck ¾, the sharper corollary `StagedFloorK`'s. K1 is not claimed outside
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
its measured cost, (π₃₂) closed for the whole census, every register load credited 16 a word;
note:20261016T1640Z-draft-rs-vf3j-strict-closure) with P4 (B_θ complete), and it sums the stages' products, so it
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
C1 (i)–(ii) and C4 per draw; S5, live coins on `trusted-nebius-host`). For every adversary in `Mc` that commits first
and every value `r` of its own coins, its run's play fixes each round's error tables, every stage's, before the
round's first draw (the play's form, `OvPlay`; this is E3, the round's run straight-line over every draw's bits,
which K∥ reads in place of C1 (iii) and (iv), and which the service doesn't check); every run the verifier accepts is
accepted by the play's game or breaks binding or zero knowledge in some round (N1, X-RO commitments; F1 and F2,
C-Flock's binding and zero knowledge), at most `δb` and `δz` a round.

The uniform draw is a deployment requirement. `e` is a bijection onto `OSpot`'s paths, so the check's cell-block is
uniform on `OCells`, all `Σ_t n_t n'_t B_t` of them at once. A challenger that draws a stage first and then a
cell-block in it weights a small stage's cell-blocks above a large one's, and is outside this premise unless every
stage has one size. -/
def ServiceSoundOverlap (Mc : Machine) (S : Service) (run : ORunMap Mc S R St n L np B n' Ar)
    (e : S.Seed ≃ Paths (OSpot St n L np B n') R) (δb δz : ℝ) : Prop :=
  ∀ (Ω' : Type) [Fintype Ω'] (M : Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)),
    (∀ σ σ' r, (Mc.eval M (σ, r)).1 = (Mc.eval M (σ', r)).1) → ∀ r : Ω',
      (∀ w, prob ((run Ω' M r).Bind w) ≤ δb) ∧ (∀ w, prob ((run Ω' M r).ZK w) ≤ δz) ∧
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

On this scope `c = 1` is proved at the luck ¾ per draw, the sharper corollary `OverlapFloorK`'s. K1 is not claimed
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

end Pouw.Assumptions
