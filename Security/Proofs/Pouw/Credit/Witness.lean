module

public import Proofs.Pouw.Credit
import all Lean.Data.Json.Basic
import all Init.Data.Ord.String
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.Data.Repr
import all Init.Data.String.Defs
import all Init.Data.String.Basic
import all Init.Data.String.Bootstrap
import all Init.Data.ToString.Basic
import all Init.Data.Nat.Bitwise.Basic
import all Init.Data.List.Basic
import all Init.Data.List.Control
import all Init.Data.List.Sort.Basic

/-!
# `GroupedDrawMisses`'s witnesses

The compiled `strata` and `law` run on the lanes window in the kernel (`decide +kernel`, through the `DecidableEq` the
draw's structures derive here), past `strata`'s key sort, which the kernel doesn't reduce (`keys_lanes`, as
`Proofs.Pouw.Period.Accept` reads H1's items); `kPrime`'s `K′` is pinned by `kPrime_least` as `Accept` pins H1's. They
are a module of their own because the kernel's run unfolds the core string and JSON definitions this module imports
whole, which `Proofs.Pouw.Credit`'s proofs must not see.
-/

@[expose] public section

namespace Properties.Pouw.Credit.SecurityProofs

open Lean (Json toJson)
open FlockSoundness FlockSoundness.Game FlockSoundness.Audit Finset
open Properties.Pouw.Strata (unitWork)
open scoped ENNReal NNReal

deriving instance DecidableEq for Flock.Draw.Stratum
deriving instance DecidableEq for Flock.Draw.WorkStratum

theorem ok_of_toOption {ε α : Type} {x : Except ε α} {a : α} (h : x.toOption = some a) : x = .ok a := by
  cases x <;> simp_all [Except.toOption]

theorem keys_lanes : PouwCheck.keys lanesLayout = some PouwCheck.Credit.LAYOUT_KEYS :=
  Properties.Pouw.Period.SecurityProofs.Accept.keys_of (by decide) (by decide +kernel)

theorem strata_lanes : PouwCheck.Credit.strata lanesLayout = .ok lanesStrata := by
  apply ok_of_toOption
  unfold PouwCheck.Credit.strata
  rw [keys_lanes]
  decide +kernel

theorem law_lanes : PouwCheck.Credit.law lanesStrata (toJson lanesK) = .ok lanesLaw :=
  ok_of_toOption (by decide +kernel)

/-- **`K′ ≤ 8` at ε = 1/2 and δ = 1/256 over the whole window's work**: `q = 1/2`, and `q^8 = δ`. -/
theorem kPrime_lanes : ∃ k', PouwWindow.kPrime (1 / 2) (1 / 256) (Flock.Draw.totalWork lanesStrata)
    (Flock.Draw.totalWork lanesStrata) 1 = .ok k' ∧ k' ≤ lanesK := by
  have hd : PouwWindow.KPrimeDomain (1 / 2) (1 / 256) (Flock.Draw.totalWork lanesStrata)
      (Flock.Draw.totalWork lanesStrata) 1 := by
    unfold PouwWindow.KPrimeDomain; decide +kernel
  have h := Properties.Pouw.Window.kPrime_least _ _ _ _ _ hd
  dsimp only at h
  obtain ⟨k, hk, -, -, -, hmin⟩ := h
  exact ⟨k, hk, Nat.le_of_not_lt fun hlt => absurd (hmin 8 (by decide) hlt) (by decide +kernel)⟩

/-- **No leaf's blocks do more than the leaf**: leaf `u < 16`'s 32 lanes do its 32·32·2048, and no block is past
leaf 15. -/
theorem leaves_lanes (u : ℕ) :
    ∑ b ∈ lanesBlocks.filter (fun b => lanesLeaf b = u), lanesWork b ≤ unitWork lanesStrata u := by
  by_cases hu : u < 16
  · interval_cases u <;> decide +kernel
  · have he : lanesBlocks.filter (fun b => lanesLeaf b = u) = ∅ := by
      refine Finset.filter_eq_empty_iff.2 fun b hb => ?_
      simp only [lanesBlocks, Finset.mem_range] at hb
      unfold lanesLeaf
      omega
    rw [he, Finset.sum_empty]
    exact Nat.zero_le _

theorem work_lanes : ∑ b ∈ lanesBlocks, lanesWork b = Flock.Draw.totalWork lanesStrata := by decide +kernel

/-- **`GroupedDrawMisses`'s honest lemma**: its premises hold at the lanes window (`lanesLayout`), sixteen 32 × 32
leaves at K = 2048 each proved as its 32 lanes and their combine (`lanesBlocks`, `lanesLeaf`, `lanesWork`): `strata` and
`law` accept it at `K = 8`, `kPrime` answers `K′ ≤ 8` at ε = 1/2 and δ = 1/256 over its whole work, no leaf's blocks do
more than the leaf, and the blocks do at least ε of the work. -/
theorem groupedDrawMisses_honest :
    ∃ k', PouwCheck.Credit.strata lanesLayout = .ok lanesStrata ∧
      PouwCheck.Credit.law lanesStrata (toJson lanesK) = .ok lanesLaw ∧
      PouwWindow.kPrime (1 / 2) (1 / 256) (Flock.Draw.totalWork lanesStrata) (Flock.Draw.totalWork lanesStrata) 1 =
        .ok k' ∧ k' ≤ lanesK ∧
      (∀ u, ∑ b ∈ lanesBlocks.filter (fun b => lanesLeaf b = u), lanesWork b ≤ unitWork lanesStrata u) ∧
      (1 / 2 : ℚ) * (Flock.Draw.totalWork lanesStrata : ℚ) ≤ ((∑ b ∈ lanesBlocks, lanesWork b : ℕ) : ℚ) := by
  obtain ⟨k', hk', hkK⟩ := kPrime_lanes
  refine ⟨k', strata_lanes, law_lanes, hk', hkK, leaves_lanes, ?_⟩
  rw [work_lanes]
  have : (0 : ℚ) ≤ (Flock.Draw.totalWork lanesStrata : ℚ) := Nat.cast_nonneg _
  linarith

/-- **`GroupedDrawMisses`'s nontrivial lemma**: its conclusion fails under the strata a law that accepts everything may
give, none (`idleLaw`): the draw is empty whatever the streams, so the lanes window's blocks all escape, with
probability 1 > 1/256. -/
theorem groupedDrawMisses_nontrivial :
    ¬ Pouw.RelativeSecurity.prob (fun g => ∃ d, drawnOn idleLaw g = some d ∧ ∀ b ∈ lanesBlocks, lanesLeaf b ∉ d) ≤
      ((1 / 256 : ℚ) : ℝ) := by
  classical
  have hall : ∀ g : (i : Fin idleLaw.size) → Fin idleLaw[i].need → Fin 256,
      ∃ d, drawnOn idleLaw g = some d ∧ ∀ b ∈ lanesBlocks, lanesLeaf b ∉ d := fun g => by
    refine ⟨(#[] : Array ℕ).qsort (· < ·), ?_, fun b _ hb =>
      absurd ((DrawExec.mem_qsort (#[] : Array ℕ) _ _).1 hb) (by simp)⟩
    unfold drawnOn Flock.Draw.stratified
    dsimp only
    rw [Std.Legacy.Range.forIn_eq_forIn_range']
    rfl
  have h1 : Pouw.RelativeSecurity.prob
      (fun g => ∃ d, drawnOn idleLaw g = some d ∧ ∀ b ∈ lanesBlocks, lanesLeaf b ∉ d) = 1 := by
    unfold Pouw.RelativeSecurity.prob
    have hf : ∀ inst : DecidablePred (fun g : (i : Fin idleLaw.size) → Fin idleLaw[i].need → Fin 256 =>
        ∃ d, drawnOn idleLaw g = some d ∧ ∀ b ∈ lanesBlocks, lanesLeaf b ∉ d),
        @Finset.filter _ _ inst Finset.univ = Finset.univ := fun _ => Finset.filter_true_of_mem fun g _ => hall g
    rw [hf, Finset.card_univ]
    exact div_self (Nat.cast_ne_zero.2 Fintype.card_ne_zero)
  rw [h1]
  norm_num

end Properties.Pouw.Credit.SecurityProofs
