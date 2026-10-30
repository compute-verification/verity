import Proofs.Flock.Soundness.ExecFlatCopies

/-!
# The unit net's rows at its input port bits and past its useful rows

Two facts about the unit net that the forced-zero rows and the input copies read, from the checks that built it:
- `parse_rows`: a net `Net.parse` read holds exactly its `useful` rows, so every row past them is empty. `Net.ofRows`
  checks nothing past them; the parser reads exactly `useful` row lines.
- `portBits_nodup`, `port_self`: a built net's input port bits are distinct columns (its groups ascend, `Net.inputsOk`,
  and each group's ports fit its words), and each is a self row `[x]·[x]` (`Net.inputsOk`).
-/

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Aliased

/-! ## Rows past `useful` -/

/-- A `for` loop in `Except` that pushes one entry to each of two arrays per step pushes one per item. -/
theorem forIn_push_size {α γ : Type} {f : γ → Array α × Array α → Except String (ForInStep (Array α × Array α))}
    (hf : ∀ i b s, f i b = .ok s → ∃ x y, s = .yield (b.1.push x, b.2.push y)) :
    ∀ (l : List γ) b v, forIn l b f = .ok v → v.1.size = b.1.size + l.length ∧ v.2.size = b.2.size + l.length
  | [], b, v, h => by rw [List.forIn_nil] at h; cases h; simp
  | a :: l, b, v, h => by
    rw [List.forIn_cons] at h
    obtain ⟨s, hs, h⟩ := except_bind_ok h
    obtain ⟨x, y, rfl⟩ := hf a b s hs
    have := forIn_push_size hf l _ v h
    simp only [Array.size_push, List.length_cons] at this ⊢
    omega

/-- **A net `Net.parse` read holds exactly its useful rows**: it reads one row line per useful row, so its `A` and `B`
have `useful` rows and every row past them is empty. -/
theorem parse_rows {lines : Array String} {net : Net} (h : Net.parse lines = .ok net) :
    net.a.rows = net.useful ∧ net.b.rows = net.useful := by
  unfold Net.parse at h
  simp only at h
  replace h := ite_throw_ok h
  obtain ⟨useful, -, h⟩ := except_bind_ok h
  obtain ⟨constPos, -, h⟩ := except_bind_ok h
  obtain ⟨ig, -, h⟩ := except_bind_ok h
  obtain ⟨og, -, h⟩ := except_bind_ok h
  replace h := ite_throw_ok (ite_throw_ok (ite_throw_ok (ite_throw_ok (ite_throw_ok h))))
  obtain ⟨s, hs, h⟩ := except_bind_ok h
  have ho : Net.ofRows ((tokens (lines.getD 0 "")).getD 1 "") useful constPos ig.1 og.1 s.1 s.2 = .ok net := by
    by_cases hL : (lines.getD (1 + useful) "").startsWith "LEAVES " = true
    · rw [ite_eq_left_of_eq_true _ _ (eq_true hL)] at h
      rcases hj : Flock.JsonParse.parse ((lines.getD (1 + useful) "").drop 7).toString with e | j
      · rw [hj] at h; exact absurd h throw_bind_ne
      · rw [hj] at h
        exact ite_throw_ok (ite_throw_ok h)
    · rw [ite_eq_right_of_eq_false _ _ (eq_false hL)] at h
      exact ite_throw_ok (ite_throw_ok h)
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hs
  have hsz := forIn_push_size (fun i b s h => by
    obtain ⟨v, -, h⟩ := except_bind_ok h
    split at h
    · exact absurd h throw_bind_ne
    · cases h; exact ⟨_, _, rfl⟩) _ _ s hs
  simp only [Array.size_empty, List.length_range', Std.Legacy.Range.size, Nat.zero_add] at hsz
  obtain ⟨-, ha, hb, hu, -⟩ := Net.ofRows_ok ho
  rw [ofRows_rows ha, ofRows_rows hb, hu]
  omega

/-! ## Input port bits -/

theorem portCols_one (g : Group) : (portCols #[g]).toList = ((forIn g.bits.toList (#[], g.col * WORD)
    (fun w s => pure (ForInStep.yield (s.1.push (s.2, w), s.2 + w))) : Id _).1.toList) := by
  unfold portCols
  simp only
  rw [← Array.forIn_toList]
  simp only [List.forIn_cons, List.forIn_nil]
  rw [← Array.forIn_toList]
  rfl

theorem inner_bits : ∀ (L : List ℕ) (a : ℕ),
    ((forIn L (#[], a) (fun w s => pure (ForInStep.yield (s.1.push (s.2, w), s.2 + w))) : Id _).1.toList).flatMap
      (fun q => List.range' q.1 q.2) = List.range' a L.sum
  | [], _ => rfl
  | w :: L, a => by
    rw [List.forIn_cons]
    show ((forIn L (#[].push (a, w), a + w) _ : Id _).1.toList).flatMap _ = _
    rw [inner_append L (#[].push (a, w)), List.flatMap_append, inner_bits L (a + w), List.sum_cons]
    simp [List.range'_append_1]

/-- A net's input port bits are each group's first `Σ bits` columns. -/
theorem portBits_eq (gs : Array Group) :
    portBits gs = gs.toList.flatMap fun g => List.range' (g.col * WORD) g.bits.toList.sum := by
  unfold portBits
  rw [portCols_toList, List.flatMap_assoc]
  congr 1
  funext g
  rw [portCols_one, inner_bits]

theorem groups_asc {ig : Array Group} (h : ∀ i (hi : i + 1 < ig.size), ig[i].col + ig[i].words ≤ ig[i + 1].col) :
    ∀ j (hj : j < ig.size) i (hi : i < j), ig[i].col + ig[i].words ≤ ig[j].col
  | 0, _, _, hi => absurd hi (Nat.not_lt_zero _)
  | j + 1, hj, i, hi => by
    have hk := h j hj
    rcases Nat.lt_succ_iff_lt_or_eq.1 hi with hi | rfl
    · have := groups_asc h j (by omega) i hi
      omega
    · exact hk

/-- **A built net's input port bits are distinct columns**: its groups ascend and are disjoint (`Net.inputsOk`), and each
group's ports fit its words. -/
theorem portBits_nodup {net : Net} (h : Built net) : (portBits net.inGroups).Nodup := by
  obtain ⟨_, _, _, ig, _, _, _, ho⟩ := h
  have hig := (Net.ofRows_ok ho).2.2.2.2.2.1
  have hasc := (Net.ofRows_inputs ho).1
  have hfit := (ofRows_groups ho).2
  rw [hig, portBits_eq, List.nodup_flatMap]
  refine ⟨fun g _ => List.nodup_range', ?_⟩
  rw [List.pairwise_iff_getElem]
  intro i j hi hj hij
  simp only [Array.length_toList] at hi hj
  simp only [Array.getElem_toList]
  have hle := groups_asc hasc j hj i hij
  have hw := (hfit ig[i] (Array.getElem_mem_toList hi)).1
  rw [← Array.foldl_toList, foldl_add_eq_sum] at hw
  have hm := Nat.mul_le_mul_right WORD hle
  rw [Nat.add_mul] at hm
  intro x hx hx'
  rw [List.mem_range'_1] at hx hx'
  omega

/-- **A built net's input port bits are self rows**, `[x]·[x]` (`Net.inputsOk`). -/
theorem port_self {net : Net} (h : Built net) (hlog : net.unitLog ≤ 32) {x : ℕ} (hx : x ∈ portBits net.inGroups)
    (j : ℕ) : sparseEntry net.a x j = (if j = x then 1 else 0) ∧ sparseEntry net.b x j = if j = x then 1 else 0 := by
  obtain ⟨_, _, _, ig, _, ra, rb, ho⟩ := h
  obtain ⟨-, ha, hb, -, -, hig, -⟩ := Net.ofRows_ok ho
  have hin := (Net.ofRows_inputs ho).2
  rw [hig, portBits_eq] at hx
  obtain ⟨g, hg, hxr⟩ := List.mem_flatMap.1 hx
  rw [List.mem_range'_1] at hxr
  have hlt : x - g.col * WORD < g.bits.foldl (· + ·) 0 := by
    rw [← Array.foldl_toList, foldl_add_eq_sum]; omega
  obtain ⟨hra, hrb⟩ := hin g hg _ hlt
  rw [show g.col * WORD + (x - g.col * WORD) = x by omega] at hra hrb
  have hw : 2 ^ net.unitLog ≤ 2 ^ 32 := Nat.pow_le_pow_right (by norm_num) hlog
  have hself : ((((#[x] : Array ℕ)).toList.count j : ℕ) : ZMod 2) = if j = x then 1 else 0 := by
    by_cases hj : j = x
    · subst hj; simp
    · simp [hj, Ne.symm hj]
  rw [sparseEntry_ofRows ha, sparseEntry_ofRows hb, ofRows_row ha hw, ofRows_row hb hw, hra, hrb]
  exact ⟨hself, hself⟩

/-- **A built net's input port bits' rows are `[x]`**, in `A` and in `B` (`port_self`'s rows). -/
theorem port_row {net : Net} (h : Built net) (hlog : net.unitLog ≤ 32) {x : ℕ} (hx : x ∈ portBits net.inGroups) :
    net.a.row x = #[x] ∧ net.b.row x = #[x] := by
  obtain ⟨_, _, _, ig, _, ra, rb, ho⟩ := h
  obtain ⟨-, ha, hb, -, -, hig, -⟩ := Net.ofRows_ok ho
  have hin := (Net.ofRows_inputs ho).2
  rw [hig, portBits_eq] at hx
  obtain ⟨g, hg, hxr⟩ := List.mem_flatMap.1 hx
  rw [List.mem_range'_1] at hxr
  have hlt : x - g.col * WORD < g.bits.foldl (· + ·) 0 := by
    rw [← Array.foldl_toList, foldl_add_eq_sum]; omega
  obtain ⟨hra, hrb⟩ := hin g hg _ hlt
  rw [show g.col * WORD + (x - g.col * WORD) = x by omega] at hra hrb
  have hw : 2 ^ net.unitLog ≤ 2 ^ 32 := Nat.pow_le_pow_right (by norm_num) hlog
  refine ⟨?_, ?_⟩
  · rw [ofRows_row ha hw, hra]
  · rw [ofRows_row hb hw, hrb]

/-- A port bit is bit `t` of some port `p`. -/
theorem mem_portBits {gs : Array Group} {x : ℕ} (hx : x ∈ portBits gs) :
    ∃ p t, p < (portCols gs).size ∧ t < (portCols gs)[p]!.2 ∧ x = (portCols gs)[p]!.1 + t := by
  obtain ⟨q, hq, hqr⟩ := List.mem_flatMap.1 hx
  obtain ⟨p, hp, rfl⟩ := List.getElem_of_mem hq
  rw [List.mem_range'_1] at hqr
  have hp' : p < (portCols gs).size := by simpa using hp
  have hq' : (portCols gs)[p]! = (portCols gs).toList[p] := by
    rw [getElem!_pos (portCols gs) p hp', Array.getElem_toList]
  exact ⟨p, x - (portCols gs)[p]!.1, hp', by rw [hq']; omega, by rw [hq']; omega⟩

end FlockSoundness.Discharge.Aliased
