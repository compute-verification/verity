module

public import Definitions.Claims.SamplingGeneric
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init
import Std.Tactic.BVDecide.Normalize.Prop

/-!
# A session's coins in a compiled run

* `place_of_not_mem`: the run with a session's coins replaced agrees with the run off the session's rounds.
* `coinsOf_place`: where `A` has as many rounds at the session's statement as the session (`Sessions.runs`), the
  session's coins of the run with them replaced are the replacement.
* `place_coinsOf`: replacing a run's session coins by its own changes nothing.
* `card_eq_of_strictMono`, `coinsOf_eq_of_strictMono`: an order embedding of `A`'s rounds onto the session's (the one
  `Claims.Sampling`'s `SessionRuns` gives) makes them as many, and is `emb`, so its coins are `coinsOf`'s.
-/

@[expose] public section

namespace Claims.SamplingGeneric.Compiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem place_of_not_mem (L : Compiled Coin A) {x : Msg} {u : ℕ} {xS : Msg} (r : Fin (L.V.k x) → Coin)
    (d : Fin (A.V.k xS) → Coin) {t : Fin (L.V.k x)} (ht : t ∉ L.rounds x u) : L.place x u xS r d t = r t := by
  unfold Compiled.place
  rw [dite_eq_right fun h => ht h.2]

omit [Fintype Coin] [Nonempty Coin] in
theorem coinsOf_place (L : Compiled Coin A) {x : Msg} {u : ℕ} {xS : Msg} (h : A.V.k xS = (L.rounds x u).card)
    (r : Fin (L.V.k x) → Coin) (d : Fin (A.V.k xS) → Coin) : L.coinsOf x u xS (L.place x u xS r d) = d := by
  funext i
  have hm : L.emb x u xS h i ∈ L.rounds x u := Finset.orderEmbOfFin_mem _ _ _
  have hi : ((L.rounds x u).orderIsoOfFin h.symm).symm ⟨L.emb x u xS h i, hm⟩ = i := by
    rw [OrderIso.symm_apply_eq]
    exact Subtype.ext (Finset.coe_orderIsoOfFin_apply _ _ _).symm
  simp only [Compiled.coinsOf, Compiled.place, dite_eq_left h, dite_eq_left (And.intro h hm), hi]

omit [Fintype Coin] [Nonempty Coin] in
theorem place_coinsOf (L : Compiled Coin A) {x : Msg} {u : ℕ} {xS : Msg} (r : Fin (L.V.k x) → Coin) :
    L.place x u xS r (L.coinsOf x u xS r) = r := by
  funext t
  by_cases h : A.V.k xS = (L.rounds x u).card ∧ t ∈ L.rounds x u
  · have ht : L.emb x u xS h.1 (((L.rounds x u).orderIsoOfFin h.1.symm).symm ⟨t, h.2⟩) = t := by
      show (L.rounds x u).orderEmbOfFin h.1.symm _ = t
      rw [← Finset.coe_orderIsoOfFin_apply, OrderIso.apply_symm_apply]
    simp only [Compiled.place, Compiled.coinsOf, dite_eq_left h, dite_eq_left h.1, ht]
  · simp only [Compiled.place, dite_eq_right h]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem card_eq_of_strictMono (L : Compiled Coin A) {x : Msg} {u : ℕ} {xS : Msg}
    {f : Fin (A.V.k xS) → Fin (L.V.k x)} (hf : StrictMono f) (hr : ∀ t, t ∈ L.rounds x u ↔ ∃ i, f i = t) :
    A.V.k xS = (L.rounds x u).card := by
  have he : L.rounds x u = Finset.univ.image f := by
    ext t
    simp [hr]
  rw [he, Finset.card_image_of_injective _ hf.injective, Finset.card_univ, Fintype.card_fin]

omit [Fintype Coin] [Nonempty Coin] in
theorem coinsOf_eq_of_strictMono (L : Compiled Coin A) {x : Msg} {u : ℕ} {xS : Msg}
    {f : Fin (A.V.k xS) → Fin (L.V.k x)} (hf : StrictMono f) (hr : ∀ t, t ∈ L.rounds x u ↔ ∃ i, f i = t)
    (r : Fin (L.V.k x) → Coin) : L.coinsOf x u xS r = fun i => r (f i) := by
  have h := L.card_eq_of_strictMono hf hr
  have hfe : f = L.emb x u xS h := Finset.orderEmbOfFin_unique h.symm (fun i => (hr _).2 ⟨i, rfl⟩) hf
  rw [hfe]
  funext i
  simp only [Compiled.coinsOf, dite_eq_left h]

end Claims.SamplingGeneric.Compiled
