import Definitions.Pouw.Dimension.Pointwise

/-!
# What a run at fixed coins must write, whatever program produced it: proofs

* **Provenance** (`good_prefix`, `good_run`). Copies create no chunk value, and each write creates at most one, a word
  away from a chunk value already present. So the chunk values a run ever holds, at any point, are the initial ones
  and `writes P` new ones, each within one word of an earlier one or an initial one (`Chained`).
* **Per position** (`run_pairs_le`). Each write brings at most one (position, value) pair no initial chunk has. So a
  run that holds the target chunks at some point writes at least the number of such pairs among them. With `w = 1`
  that is the number of distinct target values outside the initial memory (`run_values_le`).
* **Balls** (`run_ball_le`). Take radii `r y` for the targets, with each target further than `r y` from every initial
  chunk and any two further apart than the sum of their radii. Then the run writes at least `Σ_y (r y + 1)`: the chain
  of new values that reaches `y` passes through every distance `0, …, r y` from it, and the balls are disjoint.
* **The path run** (`pathRun_attained`). One slot visits the targets in any order, patching each from the one before:
  it holds every target at some point and writes exactly the path's Hamming length (`pathLen`). So in this model a
  run's writes lie between `Σ_y (r y + 1)` and the shortest such path, and a target that agrees with an earlier one
  on a fraction of its positions saves that fraction.
* **Branching on the coins** (`adaptive_pairs_le`). Each bound reads only the targets a coin outcome realizes, so it
  holds outcome by outcome for a program chosen from the coins, and sums over the outcomes.
-/

namespace Pouw.Dimension.Pointwise

open Finset

variable {A V : Type} {w : ℕ}

/-- `N` new chunk values `L 0, …, L (N−1)`, each within one word of an initial chunk value or an earlier new one. -/
def Chained [DecidableEq V] (I : Set (Fin w → V)) (N : ℕ) (L : ℕ → Fin w → V) : Prop :=
  ∀ k < N, ∃ x, (x ∈ I ∨ ∃ j < k, L j = x) ∧ hammingDist x (L k) ≤ 1

/-- Every slot of `m` holds an initial chunk value or one of the `N` new ones. -/
def Covered (I : Set (Fin w → V)) (N : ℕ) (L : ℕ → Fin w → V) (m : A → Fin w → V) : Prop :=
  ∀ a, m a ∈ I ∨ ∃ k < N, L k = m a

theorem covered_mono {I : Set (Fin w → V)} {N N' : ℕ} {L L' : ℕ → Fin w → V} {m : A → Fin w → V}
    (h : Covered I N L m) (hN : N ≤ N') (hL : ∀ k < N, L' k = L k) : Covered I N' L' m := by
  intro a
  rcases h a with h | ⟨k, hk, hkm⟩
  · exact Or.inl h
  · exact Or.inr ⟨k, by omega, by rw [hL k hk, hkm]⟩

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

variable [DecidableEq A]

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

omit [DecidableEq A] in
theorem writes_cons (st : Step A V w) (P : List (Step A V w)) : writes (st :: P) = st.cost + writes P := by
  simp [writes]

/-- One chain covers the memory after every prefix of a run. -/
theorem good_all [DecidableEq V] {I : Set (Fin w → V)} (P : List (Step A V w)) :
    ∀ (N : ℕ) (L : ℕ → Fin w → V) (m : A → Fin w → V), Chained I N L → Covered I N L m →
      ∃ L' : ℕ → Fin w → V, Chained I (N + writes P) L' ∧ (∀ k < N, L' k = L k) ∧
        ∀ i, Covered I (N + writes P) L' ((P.take i).foldl (fun m st => st.apply m) m) := by
  induction P with
  | nil =>
    intro N L m hL hm
    refine ⟨L, by simpa [writes] using hL, fun _ _ => rfl, fun i => ?_⟩
    simpa [writes] using hm
  | cons st P ih =>
    intro N L m hL hm
    obtain ⟨L₁, hL₁, hm₁, hag₁⟩ := good_step hL hm st
    obtain ⟨L', hL', hag', hcov⟩ := ih (N + st.cost) L₁ (st.apply m) hL₁ hm₁
    have hw : N + st.cost + writes P = N + writes (st :: P) := by rw [writes_cons, add_assoc]
    have hag : ∀ k < N, L' k = L k := fun k hk => by rw [hag' k (by omega), hag₁ k hk]
    refine ⟨L', hw ▸ hL', hag, fun i => ?_⟩
    cases i with
    | zero =>
      simp only [List.take_zero, List.foldl_nil]
      exact covered_mono hm (by omega) hag
    | succ i =>
      simp only [List.take_succ_cons, List.foldl_cons]
      exact hw ▸ hcov i

/-- **Provenance at every point of a run**: after every prefix of `P`, every slot holds an initial chunk value or one
of `writes P` new ones, each within one word of an earlier or initial one. -/
theorem good_prefix [DecidableEq V] [Inhabited V] (m₀ : A → Fin w → V) (P : List (Step A V w)) :
    ∃ L : ℕ → Fin w → V, Chained (Set.range m₀) (writes P) L ∧
      ∀ i, Covered (Set.range m₀) (writes P) L (run m₀ (P.take i)) := by
  obtain ⟨L, hL, -, hm⟩ := good_all (I := Set.range m₀) P 0 (fun _ _ => default) m₀
    (fun k hk => absurd hk (Nat.not_lt_zero k)) (fun a => Or.inl ⟨a, rfl⟩)
  exact ⟨L, by simpa using hL, fun i => by simpa [run] using hm i⟩

/-- **Provenance**: every slot after a run holds an initial chunk value or one of `writes P` new ones, each within
one word of an earlier or initial one. -/
theorem good_run [DecidableEq V] [Inhabited V] (m₀ : A → Fin w → V) (P : List (Step A V w)) :
    ∃ L : ℕ → Fin w → V, Chained (Set.range m₀) (writes P) L ∧ Covered (Set.range m₀) (writes P) L (run m₀ P) := by
  obtain ⟨L, hL, hm⟩ := good_prefix m₀ P
  exact ⟨L, hL, by simpa using hm P.length⟩

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
  have hex : ∀ pv ∈ (Y.biUnion pairs).filter (fun pv : Fin w × V => ∀ x ∈ I, x pv.1 ≠ pv.2),
      ∃ k, k < N ∧ L k pv.1 = pv.2 := by
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
copy addresses, chosen after the coins. `Y` are chunk values the run holds at some point. -/
theorem run_pairs_le [DecidableEq V] [Inhabited V] (m₀ : A → Fin w → V) (P : List (Step A V w))
    (Y : Finset (Fin w → V)) (hY : ∀ y ∈ Y, ∃ i a, run m₀ (P.take i) a = y) :
    ((Y.biUnion pairs).filter fun pv : Fin w × V => ∀ a, m₀ a pv.1 ≠ pv.2).card ≤ writes P := by
  obtain ⟨L, hL, hm⟩ := good_prefix m₀ P
  have := pairs_le hL Y fun y hy => by obtain ⟨i, a, rfl⟩ := hY y hy; exact hm i a
  refine le_trans (card_le_card fun pv hpv => ?_) this
  obtain ⟨h1, h2⟩ := mem_filter.1 hpv
  exact mem_filter.2 ⟨h1, fun x ⟨a, hx⟩ => hx ▸ h2 a⟩

/-- **A run writes at least `Σ_y (r y + 1)`** for target chunk values `Y` it holds at some point, when each is further
than `r y` from every initial chunk and any two are further apart than the sum of their radii. -/
theorem run_ball_le [DecidableEq V] [Inhabited V] (m₀ : A → Fin w → V) (P : List (Step A V w))
    (Y : Finset (Fin w → V)) (hY : ∀ y ∈ Y, ∃ i a, run m₀ (P.take i) a = y) (r : (Fin w → V) → ℕ)
    (hI : ∀ y ∈ Y, ∀ a, r y < hammingDist (m₀ a) y)
    (hYY : ∀ y ∈ Y, ∀ y' ∈ Y, y ≠ y' → r y + r y' < hammingDist y y') :
    ∑ y ∈ Y, (r y + 1) ≤ writes P := by
  obtain ⟨L, hL, hm⟩ := good_prefix m₀ P
  exact ball_sum_le hL Y r (fun y hy => by obtain ⟨i, a, rfl⟩ := hY y hy; exact hm i a)
    (fun y hy x ⟨a, hx⟩ => hx ▸ hI y hy a) hYY

/-- **One-word chunks: a run writes at least its distinct new values.** With `w = 1` (every copy moves one word, free),
the distinct values a run holds at some point and no initial slot holds number at most its writes. -/
theorem run_values_le [DecidableEq V] [Inhabited V] (m₀ : A → Fin 1 → V) (P : List (Step A V 1)) (Y : Finset V)
    (hY : ∀ v ∈ Y, ∃ i a, run m₀ (P.take i) a 0 = v) (hnew : ∀ v ∈ Y, ∀ a, m₀ a 0 ≠ v) : Y.card ≤ writes P := by
  classical
  have h := run_pairs_le m₀ P (Y.image fun v _ => v) fun y hy => by
    obtain ⟨v, hv, rfl⟩ := mem_image.1 hy
    obtain ⟨i, a, ha⟩ := hY v hv
    exact ⟨i, a, funext fun p => (congrArg (run m₀ (P.take i) a) (Subsingleton.elim p 0)).trans ha⟩
  refine le_trans ?_ h
  calc Y.card = (Y.image fun v => ((0 : Fin 1), v)).card :=
        (card_image_of_injective _ fun v v' h => (Prod.ext_iff.1 h).2).symm
    _ ≤ _ := card_le_card fun pv hpv => by
      obtain ⟨v, hv, rfl⟩ := mem_image.1 hpv
      refine mem_filter.2 ⟨mem_biUnion.2 ⟨fun _ => v, mem_image.2 ⟨v, hv, rfl⟩, mem_pairs.2 rfl⟩, ?_⟩
      exact fun a => hnew v hv a

/-! ## The path run: one slot, patched from each target to the next -/

/-- Patch slot `a` from `x` to `y`: one write per differing position. -/
noncomputable def patch [DecidableEq V] (a : A) (x y : Fin w → V) : List (Step A V w) :=
  (univ.filter fun p => x p ≠ y p).toList.map fun p => .write a p (y p)

/-- Visit the chunk values `ys` in order in slot `a`, starting from `x`, patching each from the one before. -/
noncomputable def pathRun [DecidableEq V] (a : A) : (Fin w → V) → List (Fin w → V) → List (Step A V w)
  | _, [] => []
  | x, y :: ys => patch a x y ++ pathRun a y ys

/-- The path's length: the Hamming distances from `x` to the first target and between consecutive ones. -/
def pathLen [DecidableEq V] : (Fin w → V) → List (Fin w → V) → ℕ
  | _, [] => 0
  | x, y :: ys => hammingDist x y + pathLen y ys

theorem run_append (m : A → Fin w → V) (P Q : List (Step A V w)) : run m (P ++ Q) = run (run m P) Q := by
  simp [run, List.foldl_append]

omit [DecidableEq A] in
theorem writes_append (P Q : List (Step A V w)) : writes (P ++ Q) = writes P + writes Q := by
  simp [writes]

theorem run_writes (m : A → Fin w → V) (a : A) (y : Fin w → V) (L : List (Fin w)) :
    run m (L.map fun p => Step.write a p (y p)) = Function.update m a (fun p => if p ∈ L then y p else m a p) := by
  induction L generalizing m with
  | nil => simp [run]
  | cons p L ih =>
    change run (Step.apply (.write a p (y p)) m) _ = _
    rw [ih]
    simp only [Step.apply, Function.update_self, Function.update_idem, List.mem_cons]
    congr 1
    funext q
    by_cases hq : q ∈ L
    · simp [hq]
    · by_cases hqp : q = p
      · subst hqp; simp [hq]
      · simp [hq, hqp]

theorem run_patch [DecidableEq V] (m : A → Fin w → V) (a : A) (y : Fin w → V) :
    run m (patch a (m a) y) a = y := by
  unfold patch
  rw [run_writes, Function.update_self]
  funext p
  by_cases h : m a p = y p <;> simp [h]

omit [DecidableEq A] in
theorem writes_patch [DecidableEq V] (a : A) (x y : Fin w → V) : writes (patch a x y) = hammingDist x y := by
  simp [writes, patch, Step.cost, hammingDist, Function.comp_def]

/-- **The path run is attained**: from any memory, the run that visits `ys` in one slot holds each of them at some
point and writes exactly the path's length. So no bound in this model exceeds the shortest path through the targets
from an initial chunk. -/
theorem pathRun_attained [DecidableEq V] (m₀ : A → Fin w → V) (a : A) (ys : List (Fin w → V)) :
    (∀ y ∈ ys, ∃ i, run m₀ ((pathRun a (m₀ a) ys).take i) a = y) ∧
      writes (pathRun a (m₀ a) ys) = pathLen (m₀ a) ys := by
  induction ys generalizing m₀ with
  | nil => simp [pathRun, pathLen, writes]
  | cons y ys ih =>
    have hy : run m₀ (patch a (m₀ a) y) a = y := run_patch m₀ a y
    obtain ⟨ih1, ih2⟩ := ih (run m₀ (patch a (m₀ a) y))
    rw [hy] at ih1 ih2
    refine ⟨fun y' hy' => ?_, ?_⟩
    · rcases List.mem_cons.1 hy' with rfl | hy'
      · exact ⟨(patch a (m₀ a) y').length, by simp [pathRun, hy]⟩
      · obtain ⟨i, hi⟩ := ih1 y' hy'
        refine ⟨(patch a (m₀ a) y).length + i, ?_⟩
        simp only [pathRun]
        rw [List.take_length_add_append, run_append]
        exact hi
    · simp only [pathRun, pathLen, writes_append, writes_patch, ih2]

open Classical in
/-- **A program that branches on the coins** (`adaptive_pairs_le`): at each coin outcome `ω` it is a run `P ω` from the
memory `m₀ ω` holding the targets `Y ω`, so summed over any outcomes its writes are at least each outcome's new
(position, value) pairs. The bound reads only the targets each outcome realizes; a design whose targets repeat a
(position, value) pair at some outcomes loses those pairs there, whatever the program. -/
theorem adaptive_pairs_le [DecidableEq V] [Inhabited V] {Ω : Type} (O : Finset Ω) (m₀ : Ω → A → Fin w → V)
    (P : Ω → List (Step A V w)) (Y : Ω → Finset (Fin w → V))
    (hY : ∀ ω ∈ O, ∀ y ∈ Y ω, ∃ i a, run (m₀ ω) ((P ω).take i) a = y) :
    ∑ ω ∈ O, (((Y ω).biUnion pairs).filter fun pv : Fin w × V => ∀ a, m₀ ω a pv.1 ≠ pv.2).card ≤
      ∑ ω ∈ O, writes (P ω) :=
  sum_le_sum fun ω hω => run_pairs_le (m₀ ω) (P ω) (Y ω) (hY ω hω)

end Pouw.Dimension.Pointwise
