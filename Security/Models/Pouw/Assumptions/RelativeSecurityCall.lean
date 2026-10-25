import Definitions.Pouw.RelativeSecurityCall

/-!
# Relative security's named premises at J10-R's call scope (trusted)

RT-ALGO-17 (note:compute-accounting/20261019T0845Z-draft-rt-algo-r3). `ChargeVF3j16` and `ChargeVF3j16Overlap` charge
κ a product of a stage's straight-line program over the stage's own bits, summed over stages. Under J10-R a call's
units share their noise, so a unit's program is not over coins of its own: a product the machine forms once serves
every unit whose words read it, and a sum over units' programs would charge it once a unit. These premises are the
staged and overlapped premises restated for one straight-line program a call over J10-R's call coins (both sides
shared, `Definitions.Pouw.RelativeSecurityCall`), each product one slot:

- `KScopeOneSidedCall`, `KScopeOverlapOneSidedCall`: Theorem K's one-sided scope at the call, each unit's error its
  word minus its own target read at its J10-R slice of the call's coins, and charging purity across every unit's
  cell-blocks;
- `ChargeVF3j16Call`, `ChargeVF3j16CallOverlap`: P1 at VF3j-16's prices for the call-level program.

The service premises are `ServiceSoundStaged`, `ServiceSoundOverlap` and their per-prover forms at the calls' shapes,
unchanged: they read the run only through its play and its breaks, and their seed is identified with the calls' coins
(`e`), which under J10-R is the random-oracle premise on the call's seed (RT-ALGO-16), not stated here.

What the call scope adds to the unit scope: a product two-sided at the frames of two units' cell-blocks is outside it,
as one two-sided at two cell-blocks of one unit is outside the unit scope. Per unit the units' coins are independent;
under J10-R the call's coin × coin part is ⟨32, 32, 32⟩ ⊗ ⟨mu, 1, nu⟩, a restriction of the units' direct sum, so a
bilinear algorithm sharing products across units is never worse there. Whether one is strictly cheaper at served
sizes is RT-ALGO-17's open half; K1 is not claimed outside the proved scopes, and the verifier can't observe which
case applies.
-/

namespace Pouw.Assumptions

open Pouw.RelativeSecurity Finset

section Call

variable {R St mu nu ur uc L np B : ℕ} {Ar : Type} [CommRing Ar]

/-- `pouw-rs-k-scope-one-sided-call` (proposed). **Theorem K's proved one-sided scope at J10-R's call scope.** For every
adversary that commits first, every value of its coins and every play `x`, each round's stage is one J10-R call, its
run one straight-line program over `Ar` on the call's coins (E3, straight-line; ring operations only), and:
- every product slot is one-sided, of any form (`OneSided`: a function of the call's `U` bits times a function of its
  dither bits, as in `KScopeOneSided`);
- each unit's error table is its committed prefix words (`word … (unitCell u)`) minus its own targets at the plane
  weights `wt`, read at its slice of the call's coins (`unitBit`: its columns' `U`, its rows' `b`) on its slice of the
  call's constants (`unitPc`, `unitVq`): what the unit's own check reads (S4);
- no product is two-sided, mod 2 through `φ`, at the frames (plane `p₀`) of two cell-blocks of the call, of one unit
  or of two, whose words read it with odd coefficients (charging purity across the call's units, in place of
  `KScopeOneSided`'s within a stage).

It is `KScopeOneSided` at the call's shape, read by unit. On this scope `c = 1` is proved at the parity luck 0.854,
the floor `StagedFloorKOneSidedCall`'s. A product the call's units share across two of their cell-blocks, as a
bilinear algorithm for the call's ⟨mu ur, L, nu uc⟩ coin × coin part would form, is outside it; K1 is not claimed
there, and the verifier can't observe which case applies. -/
def KScopeOneSidedCall (Mc : Machine) (S : Service) (run : CallRunMap Mc S R St mu nu ur uc L np B Ar)
    (wt : Fin np → Ar) (φ : Ar →+* ZMod 2) (p₀ : Fin np) : Prop :=
  ∀ (Ω' : Type) [Fintype Ω'] (M : Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)),
    (∀ σ σ' r, (Mc.eval M (σ, r)).1 = (Mc.eval M (σ', r)).1) → ∀ (r : Ω') x,
      (∀ w s j, ((run Ω' M r).prog x w s).isProd j = true → OneSided (((run Ω' M r).prog x w s).prod j)) ∧
      (∀ w s (u : UCells mu nu ur uc B), (run Ω' M r).play.errAt x w s (unitCell u) = fun z =>
        ((run Ω' M r).prog x w s).val z ((run Ω' M r).word x w s (unitCell u)) -
          dPrefix (unitPc ((run Ω' M r).pc x w s) u.1.2) (unitVq ((run Ω' M r).vq x w s) u.1.1) wt u.2.1 u.2.2.1
            u.2.2.2.val fun v => z (unitBit u.1.1 u.1.2 v)) ∧
      ∀ w s (u u' : UCells mu nu ur uc B) i, u ≠ u' →
        φ (((run Ω' M r).prog x w s).coef ((run Ω' M r).word x w s (unitCell u)) i) ≠ 0 →
        TwoSidedAt (dFrame p₀ (unitCell u)) φ (((run Ω' M r).prog x w s).prod i) →
        φ (((run Ω' M r).prog x w s).coef ((run Ω' M r).word x w s (unitCell u')) i) ≠ 0 →
        ¬TwoSidedAt (dFrame p₀ (unitCell u')) φ (((run Ω' M r).prog x w s).prod i)

/-- `pouw-rs-charge-vf3j16-call` (proposed). **The machine charges a J10-R call's run at least `κ` a product of the
call-level program.** For every adversary that commits first, every value of its coins and every play `x`, the run's
charge is at least `κ` times the products of every round's stages, less its noise reads `V x ≤ Vmax`, where a stage's
program is the call's one straight-line program over J10-R's call coins (`CallRunMap`): a product it forms once is one
slot, whichever of the call's units read it, and units' programs are never summed. With `Mc` the sm_120 machine B_θ
this is P1 at VF3j-16's prices with P4, as `ChargeVF3j16`, for the program the machine runs on a call, so it holds
when a call's interval, from its draw to its outputs' commitment, holds no other draw. Calls that overlap are
`ChargeVF3j16CallOverlap`'s. -/
def ChargeVF3j16Call (Mc : Machine) (S : Service) (run : CallRunMap Mc S R St mu nu ur uc L np B Ar)
    (e : S.Seed ≃ Paths (CallStage St mu nu ur uc L np B) R) (κ Vmax : ℝ) : Prop :=
  ∀ (Ω' : Type) [Fintype Ω'] (M : Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)),
    (∀ σ σ' r, (Mc.eval M (σ, r)).1 = (Mc.eval M (σ', r)).1) → ∀ (r : Ω') x,
      (run Ω' M r).V x ≤ Vmax ∧
      κ * ∑ w, ∑ s, ((univ.filter fun j => ((run Ω' M r).prog x w s).isProd j = true).card : ℝ) - (run Ω' M r).V x ≤
        Mc.charge M (e.symm x, r)

end Call

section OverlapCall

variable {R St ur uc L np : ℕ} {mu nu B : Fin St → ℕ} {Ar : Type} [CommRing Ar]

/-- `pouw-rs-k-scope-overlap-one-sided-call` (proposed). **Theorem K∥'s proved one-sided scope at J10-R's call scope.**
For every adversary that commits first, every value of its coins and every play `x`, each round's run is one
straight-line program over `Ar` on every draw's coins (E3), draw `t` one J10-R call of `mu t × nu t` units, and:
- every product slot is, at every bit point of the other draws', one-sided of any form on draw `t`'s coins, for every
  draw `t` (`OneSidedAt`, as in `KScopeOverlapOneSided`);
- each unit's error table is its committed prefix words minus its own targets at the plane weights `wt`, read at its
  slice of its call's coins (`unitBit`) on its slice of the call's constants (`unitPc`, `unitVq`), the other draws' at
  theirs (S4);
- at every bit point of the draws, no product is two-sided, mod 2 through `φ`, at the frames (plane `p₀`) of two
  cell-blocks, of one unit or of two, of one call or of two, whose words read it with odd coefficients (charging
  purity across the round's units).

It is `KScopeOverlapOneSided` at the calls' shapes, read by unit. On this scope `c = 1` is proved at the parity luck
0.854 per draw, the floor `OverlapFloorKOneSidedCall`'s. A product shared across two units' cell-blocks is outside it;
K1 is not claimed there, and the verifier can't observe which case applies. -/
def KScopeOverlapOneSidedCall (Mc : Machine) (S : Service) (run : OCallRunMap Mc S R St mu nu ur uc L np B Ar)
    (wt : Fin np → Ar) (φ : Ar →+* ZMod 2) (p₀ : Fin np) : Prop :=
  ∀ (Ω' : Type) [Fintype Ω'] (M : Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)),
    (∀ σ σ' r, (Mc.eval M (σ, r)).1 = (Mc.eval M (σ', r)).1) → ∀ (r : Ω') x,
      (∀ w j, ((run Ω' M r).prog x w).isProd j = true → OneSidedAt (((run Ω' M r).prog x w).prod j)) ∧
      (∀ w (u : OUCells St mu nu ur uc B), (run Ω' M r).play.errAt x w (oUnitCell u) = fun z =>
        ((run Ω' M r).prog x w).val z ((run Ω' M r).word x w (oUnitCell u)) -
          dPrefix (unitPc ((run Ω' M r).pc x w u.1) u.2.1.2) (unitVq ((run Ω' M r).vq x w u.1) u.2.1.1) wt u.2.2.1
            u.2.2.2.1 u.2.2.2.2.val fun v => z ⟨u.1, unitBit u.2.1.1 u.2.1.2 v⟩) ∧
      ∀ w (W : ODraws St (fun t => nu t * uc) L np B fun t => mu t * ur) (u u' : OUCells St mu nu ur uc B) i, u ≠ u' →
        φ (((run Ω' M r).prog x w).coef ((run Ω' M r).word x w (oUnitCell u)) i) ≠ 0 →
        TwoSidedAt (dFrame p₀ (unitCell u.2)) φ (fun z => ((run Ω' M r).prog x w).prod i (mergeAt W u.1 z)) →
        φ (((run Ω' M r).prog x w).coef ((run Ω' M r).word x w (oUnitCell u')) i) ≠ 0 →
        ¬TwoSidedAt (dFrame p₀ (unitCell u'.2)) φ fun z => ((run Ω' M r).prog x w).prod i (mergeAt W u'.1 z)

/-- `pouw-rs-charge-vf3j16-call-overlap` (proposed). **The machine charges an overlapped run of J10-R calls at least
`κ` a product of its program.** For every adversary that commits first, every value of its coins and every play `x`,
the run's charge is at least `κ` times the products of every round's one program over every call's coins
(`OCallRunMap`), less its noise reads `V x ≤ Vmax`. With `Mc` the sm_120 machine B_θ this is P1 at VF3j-16's prices
with P4, as `ChargeVF3j16Overlap`; it counts each product once, whichever units, calls and draws' intervals it serves,
so it reads nothing of the schedule. -/
def ChargeVF3j16CallOverlap (Mc : Machine) (S : Service) (run : OCallRunMap Mc S R St mu nu ur uc L np B Ar)
    (e : S.Seed ≃ Paths (OCallSpot St mu nu ur uc L np B) R) (κ Vmax : ℝ) : Prop :=
  ∀ (Ω' : Type) [Fintype Ω'] (M : Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)),
    (∀ σ σ' r, (Mc.eval M (σ, r)).1 = (Mc.eval M (σ', r)).1) → ∀ (r : Ω') x,
      (run Ω' M r).V x ≤ Vmax ∧
      κ * ∑ w, ((univ.filter fun j => ((run Ω' M r).prog x w).isProd j = true).card : ℝ) - (run Ω' M r).V x ≤
        Mc.charge M (e.symm x, r)

end OverlapCall

end Pouw.Assumptions
