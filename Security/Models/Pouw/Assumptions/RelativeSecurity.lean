import Definitions.Pouw.RelativeSecurity

/-!
# Relative security's named premises for the staged window (trusted; draft)

The three premises `Models.Pouw.Guarantees.RelativeSecurity.StagedFloorK` takes, each about the analysis's map `run`
from an adversary and its own coins to its window run on the staged design (`RunMap`). Together they are
`RoundsHypKSt` (`Proofs.Pouw.RelativeSecurity`) at `Ok = AffProd`, split so that each is judged on its own: what the
service and its proof layer give (`ServiceSoundStaged`), what an adversary's run is (`KScopeAffine`), and what the
machine charges for it (`ChargeVF3f`). The ledger ids are the premise ledger's.
-/

namespace Pouw.Assumptions

open Pouw.RelativeSecurity Finset

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
so it holds when a stage's interval, from its draw to its outputs' commitment, holds no other draw: stages that
overlap are outside it until K∥ (RS-D1E r2) is formalized. -/
def ChargeVF3f (Mc : Machine) (S : Service) (run : RunMap Mc S R St n L np B n' Ar)
    (e : S.Seed ≃ Paths (DStage St n L np B n') R) (κ Vmax : ℝ) : Prop :=
  ∀ (Ω' : Type) [Fintype Ω'] (M : Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)),
    (∀ σ σ' r, (Mc.eval M (σ, r)).1 = (Mc.eval M (σ', r)).1) → ∀ (r : Ω') x,
      (run Ω' M r).V x ≤ Vmax ∧
      κ * ∑ w, ∑ s, ((univ.filter fun j => ((run Ω' M r).prog x w s).isProd j = true).card : ℝ) - (run Ω' M r).V x ≤
        Mc.charge M (e.symm x, r)

end Pouw.Assumptions
