import Proofs.Flock.Claims.Recursion.Deploy

/-!
# The compiled verifier's completeness error is below `1`

`deployRec`'s error (`δcRec`) is a union bound over the sessions: the inner session's plain `δc` and each V* session's
masked one at the recursive setup (`δcO`). Where the compiled verifier can accept (`readyRec`), at an inner session the
numbers are stated at (`FlockDeploy.PlainIn`: at most `200` tables at `m ≤ 35`), at V* statements of tables at
`m ≤ 35`, and at most `2^64` parts, it is at most `(44·n_I + 50·n_S)/2^128 < 1` (`δcRec_le`, `δcRec_lt_one`): at most
`44` outer coins per inner table (`ε_le`), and `50` coins at each V* session's one table (`εO_le`).
-/

namespace RecDeploy

open Flock
open FlockDeploy (Ins setupOf PlainIn)

theorem length_flatMap_range_le {β : Type} (n c : ℕ) (f : ℕ → List β) (h : ∀ q < n, (f q).length ≤ c) :
    ((List.range n).flatMap f).length ≤ n * c := by
  rw [List.length_flatMap]
  refine (List.sum_le_length_nsmul _ c ?_).trans (by simp)
  intro y hy
  obtain ⟨q, hq, rfl⟩ := List.mem_map.1 hy
  exact h q (List.mem_range.1 hq)

theorem m_sub_le {sts : Array Stmt} (hm : ∀ st ∈ sts.toList, st.m ≤ 35) {q : ℕ} (hq : q < 2 * sts.size) :
    sts[q / 2]!.m - 13 ≤ 22 := by
  have hi : q / 2 < sts.size := by omega
  have := hm sts[q / 2]! (by rw [getElem!_pos sts (q / 2) hi]; exact Array.getElem_mem_toList hi)
  omega

/-- **At most `22` outer coins per rep** of the inner session, two reps per table, at tables of `m ≤ 35`. -/
theorem outerSlots_length_le {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hs : setupOf false x = some (s, spec, sts)) (hm : ∀ st ∈ sts.toList, st.m ≤ 35) :
    (FlockComplete.outerSlots x).length ≤ 2 * sts.size * 22 := by
  unfold FlockComplete.outerSlots
  rw [hs]
  exact length_flatMap_range_le _ _ _ fun q hq => by simpa using m_sub_le hm hq

/-- **At most `22` outer coins per rep** of a V* session, at tables of `m ≤ 35`. -/
theorem outerSlotsO_length_le {x : Claims.Msg} {s : RecIns} {spec : SessionSpec} {sts : Array Stmt}
    (hs : setupO x = some (s, spec, sts)) (hm : ∀ st ∈ sts.toList, st.m ≤ 35) :
    (outerSlotsO x).length ≤ 2 * sts.size * 22 := by
  unfold outerSlotsO
  rw [hs]
  exact length_flatMap_range_le _ _ _ fun q hq => by simpa using m_sub_le hm hq

theorem rhoSlotsO_length_le {x : Claims.Msg} {s : RecIns} {spec : SessionSpec} {sts : Array Stmt}
    (hs : setupO x = some (s, spec, sts)) : (rhoSlotsO x).length ≤ 2 * sts.size * 1 := by
  unfold rhoSlotsO
  rw [hs]
  exact length_flatMap_range_le _ _ _ fun q _ => by split <;> simp

theorem betaSlotsO_length_le {x : Claims.Msg} {s : RecIns} {spec : SessionSpec} {sts : Array Stmt}
    (hs : setupO x = some (s, spec, sts)) : (betaSlotsO x).length ≤ 2 * sts.size * 1 := by
  unfold betaSlotsO
  rw [hs]
  exact length_flatMap_range_le _ _ _ fun q _ => by split <;> simp

/-- A V* statement the masked verifier can accept sets up at one table. -/
theorem setupO_of_readyO {H : Claims.Msg → Claims.Msg} {x : Claims.Msg} (h : readyO H x = true) :
    ∃ s spec sts, setupO x = some (s, spec, sts) ∧ sts.size = 1 := by
  unfold readyO at h
  split at h
  · rename_i s spec sts hs
    simp only [Bool.and_eq_true, beq_iff_eq] at h
    exact ⟨s, spec, sts, hs, h.1.1.1.1⟩
  · exact absurd h Bool.false_ne_true

/-- **The inner session's error**: at most `44` outer coins per table over `2^128`, at a session the numbers are
stated at. -/
theorem ε_le {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array Stmt} (hin : PlainIn x s spec sts) :
    FlockComplete.ε x ≤ 44 * sts.size / 2 ^ 128 := by
  have hlen := outerSlots_length_le hin.setup fun st hst => (hin.tab st hst).2.1
  rw [FlockComplete.ε]
  gcongr
  exact_mod_cast hlen.trans (le_of_eq (by ring))

/-- **A V\* session's error**: `44` outer coins, two `β`s and two pairs counted twice, over `2^128`, at a V* statement
the masked verifier can accept, of tables at `m ≤ 35`. -/
theorem εO_le {H : Claims.Msg → Claims.Msg} {x : Claims.Msg} (hr : readyO H x = true)
    (hm : ∀ s spec sts, setupO x = some (s, spec, sts) → ∀ st ∈ sts.toList, st.m ≤ 35) :
    εO x ≤ 50 / 2 ^ 128 := by
  obtain ⟨s, spec, sts, hs, h1⟩ := setupO_of_readyO hr
  have ho := outerSlotsO_length_le hs (hm s spec sts hs)
  have hρ := rhoSlotsO_length_le hs
  have hβ := betaSlotsO_length_le hs
  rw [h1] at ho hρ hβ
  unfold εO
  gcongr
  have ho' : ((outerSlotsO x).length : ℝ) ≤ 44 := by exact_mod_cast ho
  have hρ' : ((rhoSlotsO x).length : ℝ) ≤ 2 := by exact_mod_cast hρ
  have hβ' : ((betaSlotsO x).length : ℝ) ≤ 2 := by exact_mod_cast hβ
  linarith

/-- **The compiled verifier's error, bounded**: where it can accept, at an inner session the numbers are stated at and
V* statements of tables at `m ≤ 35`, at most `44` coins per inner table and `50` per V* session, over `2^128`. -/
theorem δcRec_le (st : Stage) (H : Claims.Msg → Claims.Msg) {x : Claims.Msg} {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} (hin : PlainIn (xIOf x) s spec sts) (hready : readyRec st H x = true)
    (hm : ∀ j < nS x, ∀ s' spec' sts', setupO (staged st x (pick st x j) j) = some (s', spec', sts') →
      ∀ t ∈ sts'.toList, t.m ≤ 35) :
    δcRec st H x ≤ (44 * sts.size + 50 * nS x) / 2 ^ 128 := by
  have hr := hready
  unfold readyRec at hr
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hr
  obtain ⟨hI, hO⟩ := hr
  simp only [δcRec, hready, FlockComplete.δc, hI, ↓reduceIte]
  have hsum : ∑ j ∈ Finset.range (nS x), δcO H (staged st x (pick st x j) j) ≤
      ∑ _j ∈ Finset.range (nS x), (50 / 2 ^ 128 : ℝ) := by
    refine Finset.sum_le_sum fun j hj => ?_
    have hj' := Finset.mem_range.1 hj
    simp only [δcO, hO j hj', ↓reduceIte]
    exact εO_le (hO j hj') (hm j hj')
  rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul] at hsum
  calc _ ≤ 44 * (sts.size : ℝ) / 2 ^ 128 + nS x * (50 / 2 ^ 128) := add_le_add (ε_le hin) hsum
    _ = _ := by ring

/-- **The compiled verifier's error is below `1`** where it can accept, at an inner session the numbers are stated at,
V* statements of tables at `m ≤ 35`, and at most `2^64` parts. -/
theorem δcRec_lt_one (st : Stage) (H : Claims.Msg → Claims.Msg) {x : Claims.Msg} {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} (hin : PlainIn (xIOf x) s spec sts) (hready : readyRec st H x = true)
    (hm : ∀ j < nS x, ∀ s' spec' sts', setupO (staged st x (pick st x j) j) = some (s', spec', sts') →
      ∀ t ∈ sts'.toList, t.m ≤ 35) (hn : nS x ≤ 2 ^ 64) :
    δcRec st H x < 1 := by
  refine lt_of_le_of_lt (δcRec_le st H hin hready hm) ?_
  have h200 : (sts.size : ℝ) ≤ 200 := by exact_mod_cast hin.size_le
  have hn' : (nS x : ℝ) ≤ 2 ^ 64 := by exact_mod_cast hn
  rw [div_lt_one (by positivity)]
  calc 44 * (sts.size : ℝ) + 50 * nS x ≤ 44 * 200 + 50 * 2 ^ 64 := by linarith
    _ < 2 ^ 128 := by norm_num

end RecDeploy
