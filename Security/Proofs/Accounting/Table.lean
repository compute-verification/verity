import Proofs.Accounting.Gates
import Definitions.Accounting.Guarantees.Cut

/-!
# The table circuit's units, run-input bits and restriction, and its encode bound

`table w` charges 9 units and 8 run-input bits for every `w`: its first gather pays the token's 9 index bits and the
other `w − 1` share them, and its columns are registered, so no gather pays for a run-time column. Its window is proven
(each gather's hull is its column's int8 interval) and it has no budgeted gate but its gathers, so it is restricted at
every `B`, and `EncodeChainHardness` applies to it as it stands.
-/

namespace Accounting.SecurityProofs.Cut

open Accounting.Gates Accounting.Assumptions Accounting.SecurityProofs.Gates

variable (w : ℕ)

lemma table_length : (table w).length = 256 * w + 1 + w := by
  simp [table]
  omega

lemma table_get_word {i : ℕ} (h : i < 256 * w) : (table w)[i]? = some (tableWord i) := by
  rw [table, List.getElem?_append_left (by simpa using h)]
  simp [h]

lemma table_get_token : (table w)[256 * w]? = some (tableToken w) := by
  rw [table, List.getElem?_append_right (by simp)]
  simp

lemma table_get_gather {c : ℕ} (h : c < w) : (table w)[256 * w + 1 + c]? = some (tableGather w c) := by
  rw [table, List.getElem?_append_right (by simp only [List.length_map, List.length_range]; omega)]
  simp only [List.length_map, List.length_range]
  rw [show 256 * w + 1 + c - 256 * w = c + 1 by omega, List.getElem?_cons_succ]
  simp [h]

lemma getElem_of_get {C : Dag ℕ} {i : ℕ} {g : Gate ℕ} (h : C[i]? = some g) (hi : i < C.length) : C[i] = g := by
  rw [List.getElem?_eq_getElem hi] at h
  exact Option.some.inj h

lemma table_cases {i : ℕ} (hi : i < (table w).length) :
    (i < 256 * w ∧ (table w)[i] = tableWord i) ∨ (i = 256 * w ∧ (table w)[i] = tableToken w) ∨
      ∃ c < w, i = 256 * w + 1 + c ∧ (table w)[i] = tableGather w c := by
  have hl := table_length w
  rcases lt_trichotomy i (256 * w) with h | h | h
  · exact Or.inl ⟨h, getElem_of_get (table_get_word w h) hi⟩
  · subst h
    exact Or.inr (Or.inl ⟨rfl, getElem_of_get (table_get_token w) hi⟩)
  · obtain ⟨c, rfl⟩ : ∃ c, i = 256 * w + 1 + c := ⟨i - (256 * w + 1), by omega⟩
    have hc : c < w := by omega
    exact Or.inr (Or.inr ⟨c, hc, rfl, getElem_of_get (table_get_gather w hc) hi⟩)

lemma table_dyn_word {j : ℕ} (h : j < 256 * w) : dynAt (table w) j = false := by
  have hj : j < (table w).length := by rw [table_length]; omega
  rw [dynAt_eq _ j hj, getElem_of_get (table_get_word w h) hj]
  rfl

lemma table_dyn_token : dynAt (table w) (256 * w) = true := by
  have hj : 256 * w < (table w).length := by rw [table_length]; omega
  rw [dynAt_eq _ _ hj, getElem_of_get (table_get_token w) hj]
  rfl

lemma table_dyn_gather {c : ℕ} (h : c < w) : dynAt (table w) (256 * w + 1 + c) = true := by
  have hj : 256 * w + 1 + c < (table w).length := by rw [table_length]; omega
  rw [dynAt_eq _ _ hj, getElem_of_get (table_get_gather w h) hj]
  simp only [dynStep, tableGather, List.any_cons]
  rw [getD_dynamic_take _ _ _ hj.le]
  simp only [show 256 * w < 256 * w + 1 + c by omega, ite_true, table_dyn_token, Bool.true_or]

lemma table_units_low {i : ℕ} (h : i ≤ 256 * w) : units (table w) i = 0 := by
  rcases Nat.lt_or_eq_of_le h with h | rfl
  · exact units_of_budget_none _ i _ (table_get_word w h) rfl
  · exact units_of_budget_none _ _ _ (table_get_token w) rfl

lemma foldl_max_replicate (a : ℕ) : ∀ (n b : ℕ), (List.replicate n a).foldl max b = if n = 0 then b else max b a
  | 0, b => rfl
  | n + 1, b => by
    rw [List.replicate_succ, List.foldl_cons, foldl_max_replicate a n (max b a)]
    rcases n with _ | n <;> simp

lemma table_take_gather {c : ℕ} (h : c < w) :
    (table w).take (256 * w + 1 + c) =
      (List.range (256 * w)).map tableWord ++ tableToken w :: (List.range c).map (tableGather w) := by
  rw [table, List.take_append]
  simp only [List.length_map, List.length_range]
  rw [List.take_of_length_le (by simp; omega), show 256 * w + 1 + c - 256 * w = c + 1 by omega,
    List.take_succ_cons, ← List.map_take, List.take_range, min_eq_left h.le]

lemma table_units_gather {c : ℕ} (h : c < w) : units (table w) (256 * w + 1 + c) = if c = 0 then 9 else 0 := by
  have h9 : Nat.clog 2 257 = 9 := by decide
  have hcol : ((List.range 256).map (256 * c + ·)).any (fun j => (dynamic (table w)).getD j false) = false := by
    rw [List.any_eq_false]
    intro j hj
    simp only [List.mem_map, List.mem_range] at hj
    obtain ⟨t, ht, rfl⟩ := hj
    rw [show (dynamic (table w)).getD (256 * c + t) false = false from table_dyn_word w (by omega)]
    exact Bool.false_ne_true
  unfold units
  rw [table_get_gather w h]
  simp only [tableGather, show (dynamic (table w)).getD (256 * w) false = true from table_dyn_token w, hcol,
    table_take_gather w h, ite_true, Bool.false_eq_true, ite_false, add_zero]
  simp [List.filter_append, List.filter_map, Function.comp_def, tableWord, tableToken, tableGather, Op.gatherKind,
    Gate.indexBits, h9, List.map_const', foldl_max_replicate]
  split_ifs <;> rfl

lemma table_budgeted : budgeted (table w) = if w = 0 then 0 else 9 := by
  rw [budgeted, table_length, Finset.sum_range_add,
    Finset.sum_eq_zero fun i hi => table_units_low w (by rw [Finset.mem_range] at hi; omega), zero_add,
    Finset.sum_congr rfl fun c hc => table_units_gather w (Finset.mem_range.1 hc),
    Finset.sum_ite_eq' (Finset.range w) 0 fun _ => 9]
  simp only [Finset.mem_range]
  split_ifs <;> omega

lemma table_runBits : runBits (table w) = 8 := by
  have h8 : Nat.clog 2 256 = 8 := by decide
  simp [runBits, table, Function.comp_def, Gate.inputBits, tableWord, tableToken, tableGather, Gate.bits, Gate.size,
    h8]

lemma table_proven : Proven (table w) := by
  intro i hi
  dsimp only
  rcases table_cases w hi with ⟨_, hg⟩ | ⟨_, hg⟩ | ⟨c, hc, rfl, hg⟩
  · rw [hg]
    simp [tableWord, Op.IsDerived]
  · rw [hg]
    simp [tableToken, Op.IsDerived]
  · rw [hg]
    refine ⟨fun _ => ?_, by simp [tableGather], by simp [tableGather]⟩
    simp only [Gate.derive, tableGather, List.map_cons, List.tail_cons]
    apply foldl_join_within
    · exact ⟨by norm_num, by norm_num⟩
    · intro b hb
      simp only [List.map_map, List.mem_map, List.mem_range, Function.comp_def] at hb
      obtain ⟨t, ht, rfl⟩ := hb
      have hjw : 256 * c + t < 256 * w := by omega
      have hjl : 256 * c + t < (table w).length := by rw [table_length]; omega
      have hg' := getElem_of_get (table_get_word w hjw) hjl
      rw [getD_intervals_take _ _ _ (by omega) hi.le,
        getD_intervals_decl _ _ hjl (by rw [hg']; simp [tableWord, Op.IsDerived]), hg']
      simp [tableWord, Iv.Within]

lemma table_operandBits (B : ℕ) : OperandBits B (table w) := by
  intro g hg cls f hop
  simp only [table, List.mem_append, List.mem_map, List.mem_cons, List.mem_range] at hg
  rcases hg with ⟨r, _, rfl⟩ | rfl | ⟨c, _, rfl⟩ <;> simp [tableWord, tableToken, tableGather] at hop

/-- **The table circuit stacks at most its token's bits**, under `EncodeChainHardness`. -/
theorem Table : Accounting.Guarantees.Cut.Table := by
  refine ⟨fun w c hc => table_dyn_gather w hc, ?_⟩
  intro F k B E εnum εden w m x words out hE hε hk hp hp4 hc hnd hlt hW
  have h := hE hε hk hp hp4 (table w) m x words out ⟨table_proven w, table_operandBits w B⟩ hc hnd hlt hW
  rw [table_budgeted, table_runBits] at h
  refine h.trans (Nat.add_le_add_right (Nat.mul_le_mul_right _ ?_) _)
  split_ifs <;> omega

end Accounting.SecurityProofs.Cut
