import Mathlib.InformationTheory.Hamming
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

/-!
# What a run at fixed coins must write, whatever program produced it (scratch; not in a lock)

**Draft.** `DistinctWritesMH100On` bounds one straight-line program fixed before the coins. A prover that holds the
coins while it computes runs a different program at every coin value. This file bounds each such run on its own.

* **The model.** Memory is slots `A` of `w`-word chunks. A run is any list of steps. `write a p v` sets word `p` of
  slot `a` to `v`. It is the only priced step, and stands for whatever produced `v`: an arithmetic op, a load or a
  per-word copy. `copy s d` copies slot `s` to slot `d` whole, free at any address. Memory starts at `m₀`: the
  salt-free tables and the coin words. A run that branches or computes addresses is just another list.
* **Provenance** (`good_run`). Copies create no chunk value, and each write creates at most one, a word away from a
  chunk value already present. So the chunk values a run ever holds are the initial ones and `writes P` new ones,
  each within one word of an earlier one or an initial one (`Chained`).
* **Per position** (`run_pairs_le`). Each write brings at most one (position, value) pair no initial chunk has. So a
  run that leaves the target chunks in memory writes at least the number of such pairs among them.
* **Balls** (`run_ball_le`). Take radii `r y` for the targets, with each target further than `r y` from every initial
  chunk and any two further apart than the sum of their radii. Then the run writes at least `Σ_y (r y + 1)`: the
  chain of new values that reaches `y` passes through every distance `0, …, r y` from it, and the balls are disjoint.

With `w = 1` the per-position bound counts distinct target values outside the initial memory.
-/

namespace Pouw.Dimension.Pointwise

open Finset

variable {A V : Type} {w : ℕ}

/-- One step of a run: a priced word write, or a free whole-chunk copy. -/
inductive Step (A V : Type) (w : ℕ)
  | write (a : A) (p : Fin w) (v : V)
  | copy (s d : A)

variable [DecidableEq A]

/-- A step's effect on memory. -/
def Step.apply : Step A V w → (A → Fin w → V) → A → Fin w → V
  | .write a p v, m => Function.update m a (Function.update (m a) p v)
  | .copy s d, m => Function.update m d (m s)

/-- A step's price in word writes. -/
def Step.cost : Step A V w → ℕ
  | .write .. => 1
  | .copy .. => 0

/-- The memory after a run. -/
def run (m₀ : A → Fin w → V) (P : List (Step A V w)) : A → Fin w → V :=
  P.foldl (fun m st => st.apply m) m₀

/-- A run's word writes. -/
def writes (P : List (Step A V w)) : ℕ := (P.map Step.cost).sum

/-- `N` new chunk values `L 0, …, L (N−1)`, each within one word of an initial chunk value or an earlier new one. -/
def Chained [DecidableEq V] (I : Set (Fin w → V)) (N : ℕ) (L : ℕ → Fin w → V) : Prop :=
  ∀ k < N, ∃ x, (x ∈ I ∨ ∃ j < k, L j = x) ∧ hammingDist x (L k) ≤ 1

/-- Every slot of `m` holds an initial chunk value or one of the `N` new ones. -/
def Covered (I : Set (Fin w → V)) (N : ℕ) (L : ℕ → Fin w → V) (m : A → Fin w → V) : Prop :=
  ∀ a, m a ∈ I ∨ ∃ k < N, L k = m a

theorem hammingDist_update_le [DecidableEq V] (x : Fin w → V) (p : Fin w) (v : V) :
    hammingDist x (Function.update x p v) ≤ 1 := by
  unfold hammingDist
  calc (univ.filter fun i => x i ≠ Function.update x p v i).card ≤ ({p} : Finset (Fin w)).card := by
        refine card_le_card fun i hi => ?_
        rw [mem_filter] at hi
        rw [mem_singleton]
        by_contra hne
        exact hi.2 (by rw [Function.update_of_ne hne])
    _ = 1 := card_singleton p

theorem good_step [DecidableEq V] {I : Set (Fin w → V)} {N : ℕ} {L : ℕ → Fin w → V} {m : A → Fin w → V}
    (hL : Chained I N L) (hm : Covered I N L m) (st : Step A V w) :
    ∃ L' : ℕ → Fin w → V, Chained I (N + st.cost) L' ∧ Covered I (N + st.cost) L' (st.apply m) ∧
      ∀ k < N, L' k = L k := by
  cases st with
  | copy s d =>
    refine ⟨L, by simpa [Step.cost] using hL, fun a => ?_, fun _ _ => rfl⟩
    simp only [Step.cost, add_zero, Step.apply]
    by_cases had : a = d
    · subst had
      rw [Function.update_self]
      exact hm s
    · rw [Function.update_of_ne had]
      exact hm a
  | write a p v =>
    refine ⟨fun k => if k = N then Function.update (m a) p v else L k, ?_, ?_, fun k hk => by simp [hk.ne]⟩
    · intro k hk
      simp only [Step.cost] at hk
      by_cases hkN : k = N
      · subst hkN
        refine ⟨m a, ?_, by simpa using hammingDist_update_le (m a) p v⟩
        rcases hm a with h | ⟨j, hj, hjm⟩
        · exact Or.inl h
        · exact Or.inr ⟨j, hj, by simp [hj.ne, hjm]⟩
      · have hk' : k < N := by omega
        obtain ⟨x, hx, hd⟩ := hL k hk'
        refine ⟨x, ?_, by simpa [hkN] using hd⟩
        rcases hx with h | ⟨j, hj, hjx⟩
        · exact Or.inl h
        · exact Or.inr ⟨j, hj, by simp [show j ≠ N by omega, hjx]⟩
    · intro b
      simp only [Step.cost, Step.apply]
      by_cases hab : b = a
      · subst hab
        rw [Function.update_self]
        exact Or.inr ⟨N, by omega, by simp⟩
      · rw [Function.update_of_ne hab]
        rcases hm b with h | ⟨j, hj, hjm⟩
        · exact Or.inl h
        · exact Or.inr ⟨j, by omega, by simp [hj.ne, hjm]⟩

theorem good_foldl [DecidableEq V] {I : Set (Fin w → V)} (P : List (Step A V w)) :
    ∀ (N : ℕ) (L : ℕ → Fin w → V) (m : A → Fin w → V), Chained I N L → Covered I N L m →
      ∃ L' : ℕ → Fin w → V, Chained I (N + writes P) L' ∧
        Covered I (N + writes P) L' (P.foldl (fun m st => st.apply m) m) := by
  induction P with
  | nil => intro N L m hL hm; exact ⟨L, by simpa [writes] using hL, by simpa [writes] using hm⟩
  | cons st P ih =>
    intro N L m hL hm
    obtain ⟨L₁, hL₁, hm₁, -⟩ := good_step hL hm st
    obtain ⟨L', hL', hm'⟩ := ih (N + st.cost) L₁ (st.apply m) hL₁ hm₁
    refine ⟨L', ?_, ?_⟩
    · simpa [writes, add_assoc] using hL'
    · simpa [writes, add_assoc] using hm'

/-- **Provenance**: every slot after a run holds an initial chunk value or one of `writes P` new ones, each within
one word of an earlier or initial one. -/
theorem good_run [DecidableEq V] [Inhabited V] (m₀ : A → Fin w → V) (P : List (Step A V w)) :
    ∃ L : ℕ → Fin w → V, Chained (Set.range m₀) (writes P) L ∧ Covered (Set.range m₀) (writes P) L (run m₀ P) := by
  obtain ⟨L, hL, hm⟩ := good_foldl (I := Set.range m₀) P 0 (fun _ _ => default) m₀
    (fun k hk => absurd hk (Nat.not_lt_zero k)) (fun a => Or.inl ⟨a, rfl⟩)
  exact ⟨L, by simpa using hL, by simpa [run] using hm⟩

/-! ## Counting the new values -/

/-- A chunk's (position, value) pairs. -/
def pairs [DecidableEq V] (x : Fin w → V) : Finset (Fin w × V) := univ.image fun p => (p, x p)

theorem mem_pairs [DecidableEq V] {x : Fin w → V} {pv : Fin w × V} : pv ∈ pairs x ↔ x pv.1 = pv.2 := by
  constructor
  · intro h
    obtain ⟨p, -, rfl⟩ := mem_image.1 h
    rfl
  · intro h
    exact mem_image.2 ⟨pv.1, mem_univ _, by rw [h]⟩

theorem eq_of_two_diff [DecidableEq V] {x y : Fin w → V} (h : hammingDist x y ≤ 1) {p p' : Fin w}
    (hp : x p ≠ y p) (hp' : x p' ≠ y p') : p = p' := by
  unfold hammingDist at h
  exact Finset.card_le_one.1 h p (mem_filter.2 ⟨mem_univ _, hp⟩) p' (mem_filter.2 ⟨mem_univ _, hp'⟩)

open Classical in
/-- **Per position**: each new value brings at most one (position, value) pair that no initial value has, so the
targets' pairs outside the initial values number at most `N`. -/
theorem pairs_le [DecidableEq V] {I : Set (Fin w → V)} {N : ℕ} {L : ℕ → Fin w → V} (hL : Chained I N L)
    (Y : Finset (Fin w → V)) (hY : ∀ y ∈ Y, y ∈ I ∨ ∃ k < N, L k = y) :
    ((Y.biUnion pairs).filter fun pv : Fin w × V => ∀ x ∈ I, x pv.1 ≠ pv.2).card ≤ N := by
  classical
  have hex : ∀ pv ∈ (Y.biUnion pairs).filter (fun pv : Fin w × V => ∀ x ∈ I, x pv.1 ≠ pv.2), ∃ k, k < N ∧ L k pv.1 = pv.2 := by
    intro pv hpv
    obtain ⟨hpv, hnew⟩ := mem_filter.1 hpv
    obtain ⟨y, hy, hyp⟩ := mem_biUnion.1 hpv
    rw [mem_pairs] at hyp
    rcases hY y hy with hI | ⟨k, hk, rfl⟩
    · exact absurd hyp (hnew y hI)
    · exact ⟨k, hk, hyp⟩
  let f : Fin w × V → ℕ := fun pv => if h : ∃ k, k < N ∧ L k pv.1 = pv.2 then Nat.find h else 0
  have hf : ∀ pv ∈ (Y.biUnion pairs).filter (fun pv : Fin w × V => ∀ x ∈ I, x pv.1 ≠ pv.2),
      f pv < N ∧ L (f pv) pv.1 = pv.2 ∧ ∀ j < f pv, L j pv.1 ≠ pv.2 := by
    intro pv hpv
    have h := hex pv hpv
    simp only [f, dif_pos h]
    refine ⟨(Nat.find_spec h).1, (Nat.find_spec h).2, fun j hj hjv => ?_⟩
    exact Nat.find_min h hj ⟨by have := (Nat.find_spec h).1; omega, hjv⟩
  have hinj : Set.InjOn f ↑((Y.biUnion pairs).filter fun pv : Fin w × V => ∀ x ∈ I, x pv.1 ≠ pv.2) := by
    intro pv hpv pv' hpv' heq
    obtain ⟨hk, hkv, hmin⟩ := hf pv hpv
    obtain ⟨hk', hkv', hmin'⟩ := hf pv' hpv'
    rw [← heq] at hkv' hmin'
    obtain ⟨x, hx, hd⟩ := hL (f pv) hk
    have hnew : ∀ q : Fin w × V, q ∈ (Y.biUnion pairs).filter (fun pv : Fin w × V => ∀ x ∈ I, x pv.1 ≠ pv.2) →
        (∀ j < f pv, L j q.1 ≠ q.2) → x q.1 ≠ q.2 := by
      intro q hq hmq
      rcases hx with hxI | ⟨j, hj, rfl⟩
      · exact (mem_filter.1 hq).2 x hxI
      · exact hmq j hj
    have h1 : x pv.1 ≠ L (f pv) pv.1 := by rw [hkv]; exact hnew pv hpv hmin
    have h2 : x pv'.1 ≠ L (f pv) pv'.1 := by rw [hkv']; exact hnew pv' hpv' hmin'
    have hp := eq_of_two_diff hd h1 h2
    exact Prod.ext hp (by rw [← hkv, ← hkv', hp])
  calc _ ≤ (range N).card := card_le_card_of_injOn f (fun pv hpv => mem_range.2 (hf pv hpv).1) hinj
    _ = N := card_range N

/-- **A chain passes through every distance**: from a new value within `d ≤ R` of `y`, with every initial value
further than `R` from `y`, the chain back to the initial values meets a new value at distance exactly `d`. -/
theorem chain_hits [DecidableEq V] {I : Set (Fin w → V)} {N : ℕ} {L : ℕ → Fin w → V} (hL : Chained I N L)
    (y : Fin w → V) (R : ℕ) (hI : ∀ x ∈ I, R < hammingDist x y) :
    ∀ k < N, ∀ d ≤ R, hammingDist (L k) y ≤ d → ∃ j ≤ k, hammingDist (L j) y = d := by
  intro k
  induction k using Nat.strong_induction_on with
  | _ k ih =>
    intro hk d hdR hd
    rcases eq_or_lt_of_le hd with h | h
    · exact ⟨k, le_rfl, h⟩
    obtain ⟨x, hx, hxk⟩ := hL k hk
    have hxy : hammingDist x y ≤ d := by
      have := hammingDist_triangle x (L k) y
      omega
    rcases hx with hxI | ⟨j, hj, rfl⟩
    · exact absurd (hI x hxI) (by omega)
    · obtain ⟨j', hj', hd'⟩ := ih j hj (by omega) d hdR hxy
      exact ⟨j', by omega, hd'⟩

/-- At least `R + 1` new values lie within `R` of a target that every initial value is further than `R` from. -/
theorem ball_card [DecidableEq V] {I : Set (Fin w → V)} {N : ℕ} {L : ℕ → Fin w → V} (hL : Chained I N L)
    (y : Fin w → V) (R : ℕ) (hI : ∀ x ∈ I, R < hammingDist x y) (hy : y ∈ I ∨ ∃ k < N, L k = y) :
    R + 1 ≤ ((range N).filter fun k => hammingDist (L k) y ≤ R).card := by
  classical
  obtain ⟨k₀, hk₀, rfl⟩ : ∃ k < N, L k = y := by
    rcases hy with hyI | h
    · exact absurd (hI y hyI) (by simp)
    · exact h
  have hhit : ∀ d ∈ range (R + 1), ∃ j, j ≤ k₀ ∧ hammingDist (L j) (L k₀) = d := fun d hd =>
    chain_hits hL (L k₀) R hI k₀ hk₀ d (by have := mem_range.1 hd; omega) (by simp)
  let f : ℕ → ℕ := fun d => if h : ∃ j, j ≤ k₀ ∧ hammingDist (L j) (L k₀) = d then h.choose else 0
  have hf : ∀ d ∈ range (R + 1), f d ≤ k₀ ∧ hammingDist (L (f d)) (L k₀) = d := by
    intro d hd
    have h := hhit d hd
    simp only [f, dif_pos h]
    exact h.choose_spec
  calc R + 1 = (range (R + 1)).card := (card_range _).symm
    _ ≤ _ := by
      refine card_le_card_of_injOn f (fun d hd => ?_) (fun d hd d' hd' heq => ?_)
      · obtain ⟨h1, h2⟩ := hf d hd
        refine mem_filter.2 ⟨mem_range.2 (by omega), ?_⟩
        rw [h2]
        have := mem_range.1 hd
        omega
      · have h1 := (hf d hd).2
        have h2 := (hf d' hd').2
        rw [heq] at h1
        exact h1.symm.trans h2

/-- **Balls**: targets further than `r y` from every initial value, and pairwise further apart than the sum of their
radii, need `Σ_y (r y + 1)` new values. -/
theorem ball_sum_le [DecidableEq V] {I : Set (Fin w → V)} {N : ℕ} {L : ℕ → Fin w → V} (hL : Chained I N L)
    (Y : Finset (Fin w → V)) (r : (Fin w → V) → ℕ) (hY : ∀ y ∈ Y, y ∈ I ∨ ∃ k < N, L k = y)
    (hI : ∀ y ∈ Y, ∀ x ∈ I, r y < hammingDist x y)
    (hYY : ∀ y ∈ Y, ∀ y' ∈ Y, y ≠ y' → r y + r y' < hammingDist y y') :
    ∑ y ∈ Y, (r y + 1) ≤ N := by
  classical
  let K : (Fin w → V) → Finset ℕ := fun y => (range N).filter fun k => hammingDist (L k) y ≤ r y
  have hdisj : (Y : Set (Fin w → V)).PairwiseDisjoint K := by
    intro y hy y' hy' hne
    refine Finset.disjoint_left.2 fun k hk hk' => ?_
    have h1 := (mem_filter.1 hk).2
    have h2 := (mem_filter.1 hk').2
    have := hammingDist_triangle y (L k) y'
    rw [hammingDist_comm y (L k)] at this
    have := hYY y hy y' hy' hne
    omega
  calc ∑ y ∈ Y, (r y + 1) ≤ ∑ y ∈ Y, (K y).card := sum_le_sum fun y hy => ball_card hL y (r y) (hI y hy) (hY y hy)
    _ = (Y.biUnion K).card := (card_biUnion hdisj).symm
    _ ≤ (range N).card := card_le_card (biUnion_subset.2 fun y _ => filter_subset _ _)
    _ = N := card_range N

/-! ## What a run writes -/

open Classical in
/-- **A run writes at least its targets' new (position, value) pairs**, whatever the program: any branching, any
copy addresses, chosen after the coins. `Y` are chunk values the run leaves in memory. -/
theorem run_pairs_le [DecidableEq V] [Inhabited V] (m₀ : A → Fin w → V) (P : List (Step A V w))
    (Y : Finset (Fin w → V)) (hY : ∀ y ∈ Y, ∃ a, run m₀ P a = y) :
    ((Y.biUnion pairs).filter fun pv : Fin w × V => ∀ a, m₀ a pv.1 ≠ pv.2).card ≤ writes P := by
  obtain ⟨L, hL, hm⟩ := good_run m₀ P
  have := pairs_le hL Y fun y hy => by obtain ⟨a, rfl⟩ := hY y hy; exact hm a
  refine le_trans (card_le_card fun pv hpv => ?_) this
  obtain ⟨h1, h2⟩ := mem_filter.1 hpv
  exact mem_filter.2 ⟨h1, fun x ⟨a, hx⟩ => hx ▸ h2 a⟩

/-- **A run writes at least `Σ_y (r y + 1)`** for target chunk values `Y` it leaves in memory, when each is further
than `r y` from every initial chunk and any two are further apart than the sum of their radii. -/
theorem run_ball_le [DecidableEq V] [Inhabited V] (m₀ : A → Fin w → V) (P : List (Step A V w))
    (Y : Finset (Fin w → V)) (hY : ∀ y ∈ Y, ∃ a, run m₀ P a = y) (r : (Fin w → V) → ℕ)
    (hI : ∀ y ∈ Y, ∀ a, r y < hammingDist (m₀ a) y)
    (hYY : ∀ y ∈ Y, ∀ y' ∈ Y, y ≠ y' → r y + r y' < hammingDist y y') :
    ∑ y ∈ Y, (r y + 1) ≤ writes P := by
  obtain ⟨L, hL, hm⟩ := good_run m₀ P
  exact ball_sum_le hL Y r (fun y hy => by obtain ⟨a, rfl⟩ := hY y hy; exact hm a)
    (fun y hy x ⟨a, hx⟩ => hx ▸ hI y hy a) hYY

end Pouw.Dimension.Pointwise
