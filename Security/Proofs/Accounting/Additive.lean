import Definitions.Accounting.Guarantees.Additive
import Mathlib.Algebra.BigOperators.Intervals
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.Order.Ring.NNRat
import Mathlib.Tactic.GCongr
import Mathlib.Tactic.Ring

/-!
# Compute additivity's capacity from the machine model 𝓜: proofs

`Accounting.Guarantees.AccelRate` and `AccelAdditive`. The one step that reads 𝓜 is that rates add over runs back to
back (`rateAt_stream`): a window of the period splits at a run's end into a window of that run and a window of the runs
after it, each with its own backlog.
-/

open scoped NNRat

namespace Accounting.SecurityProofs.Additivity

open Finset Verity.Accel

variable {ι : Type} {m : Machine ι}

/-- The rate the guarantees state: over every window, at most `ρ_r` units of each resource `r` a tick plus `b` ticks'
worth. -/
def RateAt (iss : List (List m.Instr)) (b : ℕ) : Prop :=
  ∀ (w : Window) (r : m.Res), ∑ t ∈ w.ticks, Machine.usage iss t r ≤ m.ρ r * ((w.len + b : ℕ) : ℚ≥0)

/-- 𝓜's rate (G2) is the rate at a backlog of `slack` ticks. -/
theorem rateAt_of_rate {iss : List (List m.Instr)} (h : Machine.Rate iss) : RateAt iss m.slack := h

theorem usage_append_left (iss₁ iss₂ : List (List m.Instr)) {t : ℕ} (h : t < iss₁.length) (r : m.Res) :
    Machine.usage (iss₁ ++ iss₂) t r = Machine.usage iss₁ t r := by
  simp [Machine.usage, List.getD_eq_getElem?_getD, List.getElem?_append_left h]

theorem usage_append_right (iss₁ iss₂ : List (List m.Instr)) {t : ℕ} (h : iss₁.length ≤ t) (r : m.Res) :
    Machine.usage (iss₁ ++ iss₂) t r = Machine.usage iss₂ (t - iss₁.length) r := by
  simp [Machine.usage, List.getD_eq_getElem?_getD, List.getElem?_append_right h]

theorem rateAt_nil : RateAt ([] : List (List m.Instr)) 0 := by
  intro w r
  simp [Machine.usage]

/-- Rates add over two streams back to back: a window splits at the first one's end. -/
theorem rateAt_append {iss₁ iss₂ : List (List m.Instr)} {b₁ b₂ : ℕ} (h₁ : RateAt iss₁ b₁) (h₂ : RateAt iss₂ b₂) :
    RateAt (iss₁ ++ iss₂) (b₁ + b₂) := by
  intro w r
  set n := iss₁.length
  set l₁ := min (w.start + w.len) n - w.start with hl₁
  have hfirst : ∑ t ∈ Ico w.start (w.start + l₁), Machine.usage (iss₁ ++ iss₂) t r =
      ∑ t ∈ (⟨w.start, l₁⟩ : Window).ticks, Machine.usage iss₁ t r := by
    refine Finset.sum_congr rfl fun t ht => ?_
    simp only [Window.ticks, Finset.mem_Ico] at ht
    exact usage_append_left _ _ (by omega) r
  have hsecond : ∑ t ∈ Ico (w.start + l₁) (w.start + w.len), Machine.usage (iss₁ ++ iss₂) t r =
      ∑ t ∈ (⟨w.start + l₁ - n, w.len - l₁⟩ : Window).ticks, Machine.usage iss₂ t r := by
    simp only [Window.ticks]
    rw [Finset.sum_Ico_eq_sum_range, Finset.sum_Ico_eq_sum_range,
      show w.start + w.len - (w.start + l₁) = w.len - l₁ by omega,
      show w.start + l₁ - n + (w.len - l₁) - (w.start + l₁ - n) = w.len - l₁ by omega]
    refine Finset.sum_congr rfl fun k hk => ?_
    rw [Finset.mem_range] at hk
    rw [usage_append_right _ _ (by omega) r]
    congr 1
    omega
  calc ∑ t ∈ w.ticks, Machine.usage (iss₁ ++ iss₂) t r
      = ∑ t ∈ (⟨w.start, l₁⟩ : Window).ticks, Machine.usage iss₁ t r +
          ∑ t ∈ (⟨w.start + l₁ - n, w.len - l₁⟩ : Window).ticks, Machine.usage iss₂ t r := by
        rw [Window.ticks, ← Finset.sum_Ico_consecutive _ (by omega : w.start ≤ w.start + l₁)
          (by omega : w.start + l₁ ≤ w.start + w.len), hfirst, hsecond]
    _ ≤ m.ρ r * ((l₁ + b₁ : ℕ) : ℚ≥0) + m.ρ r * ((w.len - l₁ + b₂ : ℕ) : ℚ≥0) :=
        add_le_add (h₁ ⟨w.start, l₁⟩ r) (h₂ _ r)
    _ = m.ρ r * ((w.len + (b₁ + b₂) : ℕ) : ℚ≥0) := by
        rw [← mul_add, ← Nat.cast_add]
        congr 2
        omega

/-- **Rates add over a period's runs**: `n` runs that each meet the rate at `b` meet it together at `n·b`. -/
theorem rateAt_stream (es : List (AccelRun m)) (b : ℕ) (h : ∀ e ∈ es, RateAt e.instrs b) :
    RateAt (stream es) (es.length * b) := by
  induction es with
  | nil => simpa [stream] using (rateAt_nil (m := m))
  | cons e es ih =>
    have he : RateAt e.instrs b := h e (by simp)
    have hes := ih fun e' he' => h e' (by simp [he'])
    have hs : stream (e :: es) = e.instrs ++ stream es := by simp [stream]
    have hb : (e :: es).length * b = b + es.length * b := by
      rw [List.length_cons, Nat.add_mul, Nat.one_mul, Nat.add_comm]
    rw [hs, hb]
    exact rateAt_append he hes

/-- A period of 𝓜's runs, each meeting 𝓜's rate (G2), meets it at `n·slack`. -/
theorem rateAt_stream_of_rate (es : List (AccelRun m)) (h : ∀ e ∈ es, Machine.Rate e.instrs) :
    RateAt (stream es) (es.length * m.slack) :=
  rateAt_stream es m.slack fun e he => rateAt_of_rate (h e he)

theorem sum_range_getElem? {α : Type*} (l : List α) (f : α → ℚ≥0) :
    ∑ k ∈ range l.length, ((l[k]?).map f).getD 0 = (l.map f).sum := by
  induction l with
  | nil => simp
  | cons a l ih =>
    rw [List.length_cons, Finset.sum_range_succ']
    simp [ih, add_comm]

/-- The occurrences issued during `w`, summed, are the window's instructions, summed tick by tick. -/
theorem sum_issued (iss : List (List m.Instr)) (w : Window) (r : m.Res) :
    ∑ p ∈ issued iss w, occUse iss r p = ∑ t ∈ w.ticks, Machine.usage iss t r := by
  rw [issued, Finset.sum_biUnion]
  · refine Finset.sum_congr rfl fun t _ => ?_
    rw [Finset.sum_image fun a _ b _ hab => (Prod.mk.inj hab).2]
    exact sum_range_getElem? _ _
  · intro t₁ _ t₂ _ hne
    refine Finset.disjoint_left.2 fun p h₁ h₂ => hne ?_
    simp only [Finset.mem_image] at h₁ h₂
    obtain ⟨_, _, rfl⟩ := h₁
    obtain ⟨_, _, h⟩ := h₂
    exact ((Prod.mk.inj h).1).symm

end Accounting.SecurityProofs.Additivity

namespace Accounting.SecurityProofs

open Finset Verity.Accel Additivity

theorem AccelRate : Guarantees.AccelRate := by
  intro ι m es b h
  exact rateAt_stream es b h

theorem AccelAdditive : Guarantees.AccelAdditive := by
  intro ι m iss β hrate w k S hS hdisj r
  calc ∑ j, ∑ p ∈ S j, occUse iss r p
      = ∑ p ∈ univ.biUnion S, occUse iss r p :=
        (Finset.sum_biUnion fun i _ j _ hij => hdisj i j hij).symm
    _ ≤ ∑ p ∈ issued iss w, occUse iss r p :=
        Finset.sum_le_sum_of_subset (Finset.biUnion_subset.2 fun j _ => hS j)
    _ = ∑ t ∈ w.ticks, Machine.usage iss t r := sum_issued _ _ _
    _ ≤ _ := hrate w r

end Accounting.SecurityProofs
