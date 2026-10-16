import Definitions.Pouw.RelativeSecurity

/-!
# Relative security's named premises for the staged window (trusted; draft)

The three premises `Models.Pouw.Guarantees.RelativeSecurity.StagedFloorK` takes, each about the analysis's map `run`
from an adversary and its own coins to its window run on the staged design (`RunMap`). Together they are
`RoundsHypKSt` (`Proofs.Pouw.RelativeSecurity`) at `Ok = AffProd`, split so that each is judged on its own: what the
service and its proof layer give (`ServiceSoundStaged`), what an adversary's run is (`KScopeAffine`), and what the
machine charges for it (`ChargeVF3f`). The three `OverlapFloorK` takes are the same split of `RoundsHypKOv`, for
stages that overlap (`ServiceSoundOverlap`, `KScopeOverlap`, `ChargeVF3fOverlap`, over `ORunMap`). The ledger ids
are the premise ledger's.
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

/-- `pouw-rs-k-scope` (proposed). **Theorem K's scope, affine operands.** For every adversary that commits first,
every value of its coins and every play `x`, each round's stage is a straight-line program over `Ar` on the stage's
bits (E3, straight-line; integer MMAs with s32 wrap and unscaled reads, so ring operations only), and:
- every product slot is a product of two bit-affine operands (K1's operands, narrowed to affine: the luck ¾ is proved
  for affine operands, LA);
- the stage's error table is its committed prefix words (`word`) minus their targets at the plane weights `wt` (S4,
  the check reads the prefix word);
- certified words of distinct rows of a stage share no product (K1, row-pure).

K1 is unobservable: no check of the service refuses a run outside it (the ledger's "priced, not prevented"). -/
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

/-- `pouw-rs-charge-vf3f` (proposed). **The machine charges a run at least `κ` a product.** For every adversary that
commits first, every value of its coins and every play `x`, the run's charge (`max(|M|, cost)`) is at least `κ`
times its products over every stage of every round, less its noise reads `V x ≤ Vmax`. With `Mc` the sm_120 machine
B_θ this is P1 at VF3f's prices (τ 0.1349, margin 1.935) with P4 (B_θ complete), and it sums the stages' products,
so it holds when a stage's interval, from its draw to its outputs' commitment, holds no other draw. Stages that
overlap are `ChargeVF3fOverlap`'s. -/
def ChargeVF3f (Mc : Machine) (S : Service) (run : RunMap Mc S R St n L np B n' Ar)
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
overlapping at any depth, and one check of a cell-block drawn among every stage's (S4, round protocol with C1 (i)–(ii)
and C4 per draw; S5, live coins on `trusted-nebius-host`). For every adversary in `Mc` that commits first and every
value `r` of its own coins, its run's play fixes each round's error tables, every stage's, before the round's first
draw (the play's form, `OvPlay`; this is E3, the round's run straight-line over every draw's bits, which K∥ reads in
place of C1 (iii) and (iv), and which the service doesn't check); every run the verifier accepts is accepted by the
play's game or breaks binding or zero knowledge in some round (N1, X-RO commitments; F1 and F2, C-Flock's binding and
zero knowledge), at most `δb` and `δz` a round. -/
def ServiceSoundOverlap (Mc : Machine) (S : Service) (run : ORunMap Mc S R St n L np B n' Ar)
    (e : S.Seed ≃ Paths (OSpot St n L np B n') R) (δb δz : ℝ) : Prop :=
  ∀ (Ω' : Type) [Fintype Ω'] (M : Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)),
    (∀ σ σ' r, (Mc.eval M (σ, r)).1 = (Mc.eval M (σ', r)).1) → ∀ r : Ω',
      (∀ w, prob ((run Ω' M r).Bind w) ≤ δb) ∧ (∀ w, prob ((run Ω' M r).ZK w) ≤ δz) ∧
      ∀ x, S.verify (Mc.eval M (e.symm x, r)).1 (e.symm x) (Mc.eval M (e.symm x, r)).2 = true →
        (run Ω' M r).play.game.accepts x ∨ ∃ w, (run Ω' M r).Bind w x ∨ (run Ω' M r).ZK w x

/-- `pouw-rs-k-scope-overlap` (proposed). **Theorem K∥'s scope, operands affine per draw.** For every adversary that
commits first, every value of its coins and every play `x`, each round's run is one straight-line program over `Ar` on
every draw's bits (E3), and:
- every product slot is, at every bit point of the other draws', a product of two operands affine in draw `t`'s bits,
  for every draw `t` (`AffMulAt`: K1's operands, narrowed to affine in each draw's own bits, as `KScopeAffine` narrows
  them; an operand may depend on the other draws' bits in any way, nonlinear included, as the served operands' later
  stages do on earlier draws);
- stage `t`'s error table is its committed prefix words (`word`) minus their targets at the plane weights `wt`, read
  on draw `t`'s bits (S4);
- at every bit point `W` of the draws, certified words of distinct rows, of any stages, share no product (K1,
  row-pure across draws; a word is certified at `W` where its error vanishes on its own draw's bits with at most two
  ones, the other draws' at `W`).

K1 is unobservable, as for `KScopeAffine`. -/
def KScopeOverlap (Mc : Machine) (S : Service) (run : ORunMap Mc S R St n L np B n' Ar) (wt : Fin np → Ar) : Prop :=
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

/-- `pouw-rs-charge-vf3f-overlap` (proposed). **The machine charges an overlapped run at least `κ` a product.** For
every adversary that commits first, every value of its coins and every play `x`, the run's charge is at least `κ`
times the products of every round's run, less its noise reads `V x ≤ Vmax`. With `Mc` the sm_120 machine B_θ this is
P1 at VF3f's prices with P4, as `ChargeVF3f`; it counts each product once, whatever draws' intervals it lies in, so
it reads nothing of the schedule. -/
def ChargeVF3fOverlap (Mc : Machine) (S : Service) (run : ORunMap Mc S R St n L np B n' Ar)
    (e : S.Seed ≃ Paths (OSpot St n L np B n') R) (κ Vmax : ℝ) : Prop :=
  ∀ (Ω' : Type) [Fintype Ω'] (M : Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)),
    (∀ σ σ' r, (Mc.eval M (σ, r)).1 = (Mc.eval M (σ', r)).1) → ∀ (r : Ω') x,
      (run Ω' M r).V x ≤ Vmax ∧
      κ * ∑ w, ((univ.filter fun j => ((run Ω' M r).prog x w).isProd j = true).card : ℝ) - (run Ω' M r).V x ≤
        Mc.charge M (e.symm x, r)

end Overlap

end Pouw.Assumptions
