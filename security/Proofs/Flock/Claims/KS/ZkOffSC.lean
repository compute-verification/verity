module

public import Proofs.Flock.Claims.KS.ZkSessionW
public import Proofs.Flock.Claims.KS.Trace
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
import Proofs.Flock.Claims.KS.OffSC

/-! Off the masked table on a witness lane without a self-clash, at the deployed reader. -/

@[expose] public section

namespace Claims.Flock

open FlockSoundness FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession

local notation "𝔽" => FlockLevel3.GF128

theorem exists_ne_of_offW₀Z_sc {D Sl : Type} {mPts : ℕ} (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts) (cap₀ : Model.Cap D)
    (out : OutZC 𝔽 D Sl) {N : ℕ} (hN : N = 2 ^ (lvOf T.sch 0).logLen) (C : Fin N → List 𝔽) (d : ℕ)
    (os : List (ℕ × List 𝔽)) (hcap : out.cap₀ = cap₀)
    (hver : ∀ x ∈ opens₀Z T.sch out, VerifiedZ Hs.H Hs.E Hs.Lf cap₀ (width₀ T) (Model.pathLen (lvOf T.sch 0)) x)
    (hlt : ∀ x ∈ opens₀Z T.sch out, x.1 < 2 ^ d)
    (hos : ∀ e ∈ os, ∃ x ∈ opens₀Z T.sch out, (x.1, x.2.1.take (2 ^ T.sch.k0)) = e)
    (hin : ∀ x ∈ opens₀Z T.sch out, (x.1, x.2.1.take (2 ^ T.sch.k0)) ∈ os)
    (hoff : OffW₀Z Hs T cap₀ (oracleOf C) out) (hsc : ¬ SelfClashZ Hs T out) :
    ∃ (p : Fin N) (u : List 𝔽), lookCol (distinctBelow d os) p.val = some u ∧ u ≠ C p := by
  classical
  subst hN
  obtain ⟨p, v, hv, i, hi, hne⟩ := hoff
  unfold op₀Z at hv
  obtain ⟨x, hx, rfl⟩ := Option.map_eq_some_iff.1 hv
  have hxm := List.mem_of_mem_head? hx
  rw [List.mem_filter, decide_eq_true_eq] at hxm
  obtain ⟨hxo, hxp, -⟩ := hxm
  obtain ⟨u, hu, hum⟩ := lookCol_distinctBelow_of_mem (hin x hxo) (hlt x hxo)
  obtain ⟨y, hyo, hye⟩ := hos _ hum
  simp only [Prod.mk.injEq] at hye
  obtain ⟨hy1, hy2⟩ := hye
  refine ⟨p, u, by rw [← hxp]; exact hu, fun hu' => ?_⟩
  by_cases hrow : y.2.1 = x.2.1
  · apply hne
    show x.2.1.getD i.val 0 = oracleOf C i.val p.val
    unfold oracleOf
    simp only [p.isLt, ↓reduceDIte]
    rw [← hu', ← hy2, hrow, List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_take_of_lt hi]
  · apply hsc
    left
    rw [hcap]
    exact ⟨y, hyo, x, hxo, hy1, hver y hyo, hver x hxo, hrow⟩

theorem exists_ne_of_zk_entries {D Sl : Type} {mPts : ℕ} (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (cap₀ : Model.Cap D) (out : OutZC 𝔽 D Sl) {N : ℕ} (hN : N = 2 ^ (lvOf T.sch 0).logLen) (C : Fin N → List 𝔽)
    (d : ℕ) (os : List (ℕ × List 𝔽)) (hcap : out.cap₀ = cap₀)
    (hver : ∀ x ∈ opens₀Z T.sch out,
      VerifiedZ Hs.H Hs.E Hs.Lf cap₀ (width₀ T) (Model.pathLen (lvOf T.sch 0)) x ∧ x.1 < 2 ^ d)
    (hent : (opens₀Z T.sch out).map (fun x => (x.1, x.2.1.take (2 ^ T.sch.k0))) = os)
    (hoff : OffW₀Z Hs T cap₀ (oracleOf C) out) (hsc : ¬ SelfClashZ Hs T out) :
    ∃ (p : Fin N) (u : List 𝔽), lookCol (distinctBelow d os) p.val = some u ∧ u ≠ C p :=
  exists_ne_of_offW₀Z_sc Hs T cap₀ out hN C d os hcap (fun x hx => (hver x hx).1) (fun x hx => (hver x hx).2)
    (fun e he => by rw [← hent] at he; exact List.mem_map.1 he)
    (fun x hx => by rw [← hent]; exact List.mem_map_of_mem hx) hoff hsc

end Claims.Flock
