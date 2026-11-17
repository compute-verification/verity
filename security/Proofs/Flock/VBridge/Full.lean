module

public import Proofs.Flock.Recursive.Stage
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
import Mathlib.Tactic.ReduceModChar

/-! `VFull` under the full sample (`vFull_of_whole`), and the verifier's U4 check sampling in full each unit that reads a round's commitment (`samplesInFull_of_checkFull_round`). -/

@[expose] public section

namespace FlockVBridge.Full

open FlockSoundness FlockSoundness.Audit FlockSoundness.Discharge FlockSoundness.Discharge.Recursion

def unitL : List (ℕ × ℕ) → ℕ → ℕ
  | [], _ => 0
  | (lo, hi) :: rs, j => if j < hi - lo then lo + j else unitL rs (j - (hi - lo))

theorem unit_eq (s : Flock.Draw.Stratum) (j : ℕ) : s.unit j = unitL s.units.toList j := by
  unfold Flock.Draw.Stratum.unit
  simp only [Id.run]
  rw [← Array.forIn_toList]
  generalize s.units.toList = l
  induction l generalizing j with
  | nil => rfl
  | cons p rs ih =>
    obtain ⟨lo, hi⟩ := p
    simp only [List.forIn_cons, unitL]
    split
    · rfl
    · simpa using ih _

def sizeL (l : List (ℕ × ℕ)) : ℕ := l.foldl (fun acc (lo, hi) => acc + (hi - lo)) 0

theorem foldl_shift (l : List (ℕ × ℕ)) (a : ℕ) :
    l.foldl (fun acc (p : ℕ × ℕ) => acc + (p.2 - p.1)) a = a + sizeL l := by
  induction l generalizing a with
  | nil => simp [sizeL]
  | cons p rs ih =>
    simp only [sizeL, List.foldl_cons] at *
    rw [ih, ih (0 + _)]; omega

theorem size_eq (s : Flock.Draw.Stratum) : s.size = sizeL s.units.toList := by
  unfold Flock.Draw.Stratum.size sizeL
  rw [← Array.foldl_toList]

theorem mem_unitL : ∀ (l : List (ℕ × ℕ)) (u : ℕ), (∃ p ∈ l, p.1 ≤ u ∧ u < p.2) →
    ∃ j < sizeL l, unitL l j = u
  | [], _, ⟨_, h, _⟩ => by simp at h
  | (lo, hi) :: rs, u, ⟨p, hp, h1, h2⟩ => by
    rcases List.mem_cons.mp hp with rfl | hp
    · refine ⟨u - lo, ?_, ?_⟩
      · simp only [sizeL, List.foldl_cons]; rw [foldl_shift]; simp at h1 h2; omega
      · simp only [unitL]; simp at h1 h2; split <;> omega
    · obtain ⟨j, hj, hu⟩ := mem_unitL rs u ⟨p, hp, h1, h2⟩
      refine ⟨j + (hi - lo), ?_, ?_⟩
      · simp only [sizeL, List.foldl_cons]; rw [foldl_shift]; omega
      · simp only [unitL]; split
        · omega
        · simpa using hu

theorem mem_unit (s : Flock.Draw.Stratum) (u : ℕ) (h : s.units.any (fun (lo, hi) => lo ≤ u && u < hi) = true) :
    ∃ j < s.size, s.unit j = u := by
  rw [size_eq]
  simp only [unit_eq]
  apply mem_unitL
  simp only [Array.any_eq_true, Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨i, hi, h1, h2⟩ := h
  exact ⟨s.units[i], Array.getElem_mem_toList hi, h1, h2⟩

theorem samplesInFull_of_checkFull {own : Array Flock.Registered.Port} {j : Lean.Json} {d : Flock.Draw.UnitSample}
    (hok : Flock.Registered.checkFull (some own) (some j) = .ok ()) (hd : Flock.Draw.UnitSample.ofJson j = .ok d)
    {p : Flock.Registered.Port} (hp : p ∈ own) (hc : Flock.Registered.roundCommitment p.value = true)
    {u x : ℕ} (hu : p.reads[u]? = some (some x)) : Flock.Registered.samplesInFull d u = true := by
  have hm : Flock.Registered.fullMiss own d = none := by
    unfold Flock.Registered.checkFull at hok
    simp only [hd, bind, Except.bind] at hok
    split at hok
    · assumption
    · simp [throw, throwThe, MonadExceptOf.throw] at hok
  unfold Flock.Registered.fullMiss at hm
  rw [Array.findSome?_eq_none_iff] at hm
  have h := hm p hp
  simp only [hc, ↓reduceIte, Option.map_eq_none_iff, List.find?_eq_none] at h
  have hlt : u < p.reads.size := by
    rcases Nat.lt_or_ge u p.reads.size with h | h
    · exact h
    · simp [Array.getElem?_eq_none h] at hu
  have := h u (List.mem_range.mpr hlt)
  rw [getElem!_pos p.reads u hlt] at this
  rw [Array.getElem?_eq_getElem hlt] at hu
  simp_all

theorem indexed_digits (pre : String) (n : ℕ) :
    ((pre ++ toString n).startsWith pre &&
      (!((pre ++ toString n).drop pre.length).toString.isEmpty &&
        ((pre ++ toString n).drop pre.length).toString.all Char.isDigit)) = true := by
  have hn : (toString n).toList = Nat.toDigits 10 n := Nat.toList_repr
  have hd : ((pre ++ toString n).drop pre.length).toString = toString n := by
    apply String.toList_injective
    rw [String.Slice.toString_eq, String.toList_copy_drop, String.toList_append]
    exact List.drop_left' String.length_toList
  have hpre : (pre ++ toString n).startsWith pre = true :=
    String.startsWith_string_iff.mpr (by rw [String.toList_append]; exact List.prefix_append _ _)
  have hne : (!(toString n).isEmpty) = true := by
    rw [Bool.not_eq_true', String.isEmpty_eq_false_iff, ne_eq, ← String.toList_eq_nil_iff, hn]
    exact Nat.toDigits_ne_nil
  have hall : (toString n).all Char.isDigit = true := by
    rw [String.all_bool_eq, hn, List.all_eq_true]
    exact fun c hc => Nat.isDigit_of_mem_toDigits (by decide) (by decide) hc
  rw [Bool.and_eq_true, Bool.and_eq_true, hd]
  exact ⟨hpre, hne, hall⟩

theorem roundCommitment_indexed {pre : String} (h : pre ∈ Flock.Registered.roundIndexed) (n : ℕ) :
    Flock.Registered.roundCommitment (pre ++ toString n) = true := by
  unfold Flock.Registered.roundCommitment
  exact Bool.or_eq_true_iff.mpr (Or.inr (List.any_eq_true.mpr ⟨pre, h, indexed_digits pre n⟩))

theorem roundCommitment_named {v : String} (h : v ∈ Flock.Registered.roundNamed) :
    Flock.Registered.roundCommitment v = true := by
  unfold Flock.Registered.roundCommitment
  exact Bool.or_eq_true_iff.mpr (Or.inl (List.contains_iff_mem.mpr h))

theorem samplesInFull_of_checkFull_round {own : Array Flock.Registered.Port} {j : Lean.Json} {d : Flock.Draw.UnitSample}
    (hok : Flock.Registered.checkFull (some own) (some j) = .ok ()) (hd : Flock.Draw.UnitSample.ofJson j = .ok d)
    {p : Flock.Registered.Port} (hp : p ∈ own)
    (hv : (∃ pre ∈ Flock.Registered.roundIndexed, ∃ n : ℕ, p.value = pre ++ toString n) ∨
      p.value ∈ Flock.Registered.roundNamed)
    {u x : ℕ} (hu : p.reads[u]? = some (some x)) : Flock.Registered.samplesInFull d u = true := by
  refine samplesInFull_of_checkFull hok hd hp ?_ hu
  rcases hv with ⟨pre, hpre, n, hn⟩ | hn
  · rw [hn]; exact roundCommitment_indexed hpre n
  · exact roundCommitment_named hn

theorem full_stratum {m : ℕ} {St : Fin m → Flock.Draw.Stratum} {d : Flock.Draw.UnitSample} {Kd u : ℕ}
    (hlaw : d.samplingDesign = .stratified Kd (Array.ofFn St)) (hu : Flock.Registered.samplesInFull d u = true) :
    ∃ t : Fin m, (St t).units.any (fun (lo, hi) => lo ≤ u && u < hi) = true ∧ (St t).k = (St t).size := by
  simp only [Flock.Registered.samplesInFull, hlaw] at hu
  rw [Array.any_eq_true] at hu
  obtain ⟨i, hi, h⟩ := hu
  simp only [Array.size_ofFn] at hi
  simp only [Array.getElem_ofFn, Bool.and_eq_true, beq_iff_eq] at h
  exact ⟨⟨i, hi⟩, h⟩

variable {S : ℕ} {Vs : Fin S → VStmt}

theorem vFull_of_whole (r : ℕ) (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (hio : ∀ i < r, ∀ j g, posK (Vs j).c (Vs j).hU (Vs j).nv g ∈ Sk i j →
      ∃ u, g ∈ ((Placed.keyProg (Vs j).c (Vs j).hU (Vs j).nv).partition (Layout.outsOf (Vs j).c (Vs j).hU (Vs j).nv)).io u) :
    VFull r Sk :=
  hio

end FlockVBridge.Full
