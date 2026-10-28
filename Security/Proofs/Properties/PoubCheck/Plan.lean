import Batteries.Tactic.OpenPrivate
import Properties.PoubCheck
import Proofs.Properties.PoubCheck.Bisect
import Proofs.Properties.PoubCheck.Sort

/-!
# A pass's plan reads distinct blocks (`PlanReadsDistinct`, `ChangedReadsDistinct`)

A plan's position `p` is block `runs[r].1 + (p − Σ_{<r} count)` with `r = bisectRight ends p` (`block_spec`): a block of
run `r`, inside the store. `Plan.make?` refuses runs outside the store and checks that, sorted, each run ends at or
before the next starts; so sorted they are pairwise disjoint (`chain_pairwise`), and disjointness, being symmetric,
survives the sort's permutation back to the runs as given. Two positions in one run at one block are one position;
two in different runs can't share a block (`readsDistinct_of`). `Plan.changed` returns `Plan.make?` of the runs it
builds, or `none` (`changed_make`).
-/

open PoubCheck

open private PoubCheck.endsOf PoubCheck.lexLt from PoubCheck.Plan

namespace Properties.PoubCheck

/-- Two runs that don't overlap. -/
def Apart (x y : Nat × Nat) : Prop := x.1 + x.2 ≤ y.1 ∨ y.1 + y.2 ≤ x.1

theorem endsOf_prefix (rs : Array (Nat × Nat)) : PrefixSums (PoubCheck.endsOf rs) (rs.toList.map (·.2)) := by
  unfold PoubCheck.endsOf
  exact prefixSums_fold (·.2) rs

theorem block_spec (blocks : Nat) (rs : Array (Nat × Nat)) (ends : Array Nat)
    (hP : PrefixSums ends (rs.toList.map (·.2))) (p : Nat) (hp : p < (rs.toList.map (·.2)).sum) :
    bisectRight ends p < rs.size ∧ ((rs.toList.map (·.2)).take (bisectRight ends p)).sum ≤ p ∧
      p < ((rs.toList.map (·.2)).take (bisectRight ends p)).sum + rs[bisectRight ends p]!.2 ∧
      Plan.block ⟨blocks, rs, ends⟩ p =
        rs[bisectRight ends p]!.1 + (p - ((rs.toList.map (·.2)).take (bisectRight ends p)).sum) := by
  obtain ⟨hr, hlo, hhi, hoff⟩ := locate hP p hp
  simp only [List.length_map, Array.length_toList] at hr
  have hl : (rs.toList.map (·.2))[bisectRight ends p]! = rs[bisectRight ends p]!.2 := by
    rw [getElem!_pos _ _ (by simpa using hr), getElem!_pos rs _ hr]
    simp
  rw [hl] at hhi
  refine ⟨hr, hlo, hhi, ?_⟩
  simp only [Plan.block]
  rw [hoff]
  omega

/-- **A plan whose runs are inside the store and pairwise apart reads distinct blocks.** -/
theorem readsDistinct_of (blocks blocks' : Nat) (rs : Array (Nat × Nat)) (ends : Array Nat)
    (hP : PrefixSums ends (rs.toList.map (·.2))) (hin : ∀ x ∈ rs.toList, x.1 + x.2 ≤ blocks)
    (hdisj : rs.toList.Pairwise Apart) : ReadsDistinct ⟨blocks', rs, ends⟩ blocks := by
  have hlen : Plan.length ⟨blocks', rs, ends⟩ = (rs.toList.map (·.2)).sum := back_getD_of_prefix hP
  refine ⟨fun p hp => ?_, fun p q hp hq hpq => ?_⟩
  · rw [hlen] at hp
    obtain ⟨hr, hlo, hhi, hb⟩ := block_spec blocks' rs ends hP p hp
    rw [hb]
    have := hin rs[bisectRight ends p]! (by rw [getElem!_pos rs _ hr]; exact Array.getElem_mem_toList hr)
    omega
  · rw [hlen] at hp hq
    obtain ⟨hr, hlo, hhi, hb⟩ := block_spec blocks' rs ends hP p hp
    obtain ⟨hr', hlo', hhi', hb'⟩ := block_spec blocks' rs ends hP q hq
    rw [hb, hb'] at hpq
    generalize bisectRight ends p = a at hr hlo hhi hpq
    generalize bisectRight ends q = b at hr' hlo' hhi' hpq
    rcases Nat.lt_trichotomy a b with hab | rfl | hab
    · exfalso
      have := List.pairwise_iff_getElem.1 hdisj a b (by simpa using hr) (by simpa using hr') hab
      simp only [Array.getElem_toList] at this
      rw [getElem!_pos rs a hr] at hpq hhi
      rw [getElem!_pos rs b hr'] at hpq hhi'
      unfold Apart at this
      omega
    · omega
    · exfalso
      have := List.pairwise_iff_getElem.1 hdisj b a (by simpa using hr') (by simpa using hr) hab
      simp only [Array.getElem_toList] at this
      rw [getElem!_pos rs a hr] at hpq hhi
      rw [getElem!_pos rs b hr'] at hpq hhi'
      unfold Apart at this
      omega

/-- Runs each ending at or before the next starts are pairwise apart. -/
theorem chain_pairwise (l : List (Nat × Nat)) (h : ∀ k (hk : k + 1 < l.length), l[k].1 + l[k].2 ≤ l[k + 1].1) :
    l.Pairwise Apart := by
  have step : ∀ d i (hj : i + d + 1 < l.length), l[i].1 + l[i].2 ≤ (l[i + d + 1]'hj).1 := by
    intro d
    induction d with
    | zero => intro i hj; exact h i hj
    | succ d ih =>
      intro i hj
      have h1 := ih i (by omega)
      have h2 := h (i + d + 1) (by omega)
      simp only [show i + d + 1 + 1 = i + (d + 1) + 1 by omega] at h2
      omega
  refine List.pairwise_iff_getElem.2 fun i j hi hj hij => Or.inl ?_
  have := step (j - i - 1) i (by omega)
  simp only [show i + (j - i - 1) + 1 = j by omega] at this
  exact this

/-- Runs inside the store whose sort by `lexLt` chains are pairwise apart, and a plan of them reads distinct blocks. -/
theorem readsDistinct_sorted (blocks blocks' : Nat) (rs : Array (Nat × Nat))
    (hin : ∀ x ∈ rs.toList, x.1 + x.2 ≤ blocks)
    (hord : ∀ k, k < (rs.qsort PoubCheck.lexLt).size - 1 →
      ¬ ((rs.qsort PoubCheck.lexLt)[k]!.1 + (rs.qsort PoubCheck.lexLt)[k]!.2 > (rs.qsort PoubCheck.lexLt)[k + 1]!.1)) :
    ReadsDistinct ⟨blocks', rs, PoubCheck.endsOf rs⟩ blocks := by
  refine readsDistinct_of blocks blocks' rs _ (endsOf_prefix rs) hin ?_
  have hp := qsort_toList_perm rs PoubCheck.lexLt
  refine (hp.pairwise_iff (fun {x y} hxy => Or.symm hxy)).1 (chain_pairwise _ fun k hk => ?_)
  simp only [Array.length_toList] at hk
  have := hord k (by omega)
  rw [getElem!_pos _ k (by omega), getElem!_pos _ (k + 1) hk] at this
  simp only [Array.getElem_toList]
  omega

theorem planReadsDistinct : PlanReadsDistinct := by
  intro runs blocks pl h
  unfold Plan.make? at h
  simp only [Bool.not_eq_true', Option.bind_eq_bind, Option.bind_none] at h
  split at h
  · cases h
  split at h
  · cases h
  split at h
  · cases h
  rename_i hin
  split at h
  · cases h
  split at h
  · cases h
  rename_i hord
  cases h
  refine readsDistinct_sorted blocks blocks _ (fun x hx => ?_) (fun k hk => ?_)
  · rw [Array.toList_map, List.mem_map] at hx
    obtain ⟨y, hy, rfl⟩ := hx
    have := Array.any_eq_false.1 (Bool.not_eq_true _ ▸ hin)
    obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem hy
    have h' := this i (by simpa using hi)
    simp only [Bool.or_eq_true, decide_eq_true_eq, not_or, Array.getElem_toList] at h' ⊢
    obtain ⟨⟨h1, h2⟩, h3⟩ := h'
    omega
  · have := Array.any_eq_false.1 (Bool.not_eq_true _ ▸ hord)
    have h' := this k (by simpa using hk)
    simpa using h'

/-! ## A change of the previous pass's plan -/

/-- A loop whose body keeps `Q` ends in a state with `Q`, and what follows it sees that state. -/
theorem forIn_bind_inv {α σ β : Type} (Q : σ → Prop) (R : β → Prop) (F : α → σ → Id (ForInStep σ))
    (POST : σ → Id β) (hF : ∀ x s, Q s → Q (F x s).run.value) (hP : ∀ s, Q s → R (POST s).run) :
    ∀ (l : List α) (s : σ), Q s → R (forIn l s F >>= POST).run := by
  intro l
  induction l with
  | nil => intro s hs; exact hP s hs
  | cons x l ih =>
    intro s hs
    rw [List.forIn_cons]
    have hx := hF x s hs
    revert hx
    generalize F x s = st
    intro hx
    cases st with
    | done b => exact hP b hx
    | yield b => exact ih b hx

theorem forIn_bind_inv_arr {α σ β : Type} (Q : σ → Prop) (R : β → Prop) (F : α → σ → Id (ForInStep σ))
    (POST : σ → Id β) (hF : ∀ x s, Q s → Q (F x s).run.value) (hP : ∀ s, Q s → R (POST s).run)
    (xs : Array α) (s : σ) (hs : Q s) : R (forIn xs s F >>= POST).run := by
  rw [← Array.forIn_toList]
  exact forIn_bind_inv Q R F POST hF hP xs.toList s hs

/-- A loop state whose early return, if taken, returned `none`. -/
def Quiet {τ : Type} (s : Option (Option Plan) × τ) : Prop := s.1 = none ∨ s.1 = some none

/-- What `Plan.changed` returns: `none`, or `Plan.make?` of some runs in the store. -/
def MakeOrNone (blocks : Nat) (r : Option Plan) : Prop := r = none ∨ ∃ runs, r = Plan.make? runs blocks

/-- **`Plan.changed` returns `none` or `Plan.make?` of the runs it builds.** -/
theorem changed_make (pl : Plan) (added removed : Array (Int × Int)) (blocks : Nat) :
    MakeOrNone blocks (pl.changed added removed blocks) := by
  unfold Plan.changed
  simp only [Id.run, Std.Legacy.Range.forIn_eq_forIn_range']
  split
  · exact Or.inl rfl
  split
  · split
    · refine forIn_bind_inv_arr Quiet (MakeOrNone blocks) _ _ ?_ ?_ _ _ (Or.inl rfl)
      · intro x s _
        refine forIn_bind_inv Quiet (fun st => Quiet (ForInStep.value st)) _ _ ?_ ?_ _ _ ?_
        · intro y s' _
          split
          · split
            · exact Or.inr rfl
            · split <;> exact Or.inl rfl
          · exact Or.inl rfl
        · intro s' hs'
          rcases hs' with h0 | h0 <;> rw [h0]
          · dsimp only
            split <;> exact Or.inl rfl
          · exact Or.inr rfl
        · exact Or.inl rfl
      · intro s hs
        rcases hs with h0 | h0 <;> rw [h0]
        · dsimp only
          split
          · exact Or.inl rfl
          · refine forIn_bind_inv_arr Quiet (MakeOrNone blocks) _ _ ?_ ?_ _ _ (Or.inl rfl)
            · intro x s' _
              split
              · split
                · exact Or.inr rfl
                · split <;> exact Or.inl rfl
              · exact Or.inl rfl
            · intro s' hs'
              rcases hs' with h1 | h1 <;> rw [h1]
              · exact Or.inr ⟨_, rfl⟩
              · exact Or.inl rfl
        · exact Or.inl rfl
    · exact Or.inl rfl
  · exact Or.inl rfl

theorem changedReadsDistinct : ChangedReadsDistinct := by
  intro pl added removed blocks pl' h
  rcases changed_make pl added removed blocks with h0 | ⟨runs, h1⟩
  · rw [h0] at h; cases h
  · rw [h1] at h
    exact planReadsDistinct runs blocks pl' h

end Properties.PoubCheck
