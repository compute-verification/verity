import Proofs.Properties.NetworkScan.Check

/-!
# What the unit's reads and Bits' parameter check give

Bits reads a circuit file's unit against its template's Definition (`SampledProofs.Bits.readsWhy`, `paramsWhy`), each a
loop over the parameters' bits that stops at the first fault. Where `readsWhy` at the names `[nm]` finds nothing wrong
and the Definition's first parameter is `nm`, the unit reads each of its bits at its port's column (`readsWhy_first`);
where `paramsWhy` finds nothing wrong, instance 0's input port bit of each of the first parameter's bits is a message
bit (`paramsWhy_first`, through `fieldBit_first`). A loop each of whose steps yields `none` or stops at `some (some m)`
(`Step`) ends at `none` or `some (some m)` (`forIn_found`, `forIn_some`), and where it ends at `none`, each element's
step yielded (`forIn_yields`, `forIn_each`).

An input column the unit reads (`SampledProofs.Bits.unitReads`) is read by a computed row in its A or its B
(`unitReads_true`, through `fold_set_true` and `fold_rows_true`). Where the unit's input ports are in column order
(`Check.PortsOrdered`), the source of the column at bit `τ` of port `p` is instance 0's port `p` bit `τ`
(`srcKey_port`), and a column whose source is no zero has a key (`keyIdx_isSome`).
-/

open SampledProofs.Bits

namespace Properties.NetworkScan.Reads

/-- Each step of the loop body `f` yields `none` or stops at `some (some m)`. -/
def Step {α β γ : Type} (f : γ → Option (Option α) × β → Id (ForInStep (Option (Option α) × β))) : Prop :=
  ∀ y s, (∃ b, (f y s).run = .yield (none, b)) ∨ ∃ m b, (f y s).run = .done (some (some m), b)

theorem forIn_found {α β γ : Type} {f : γ → Option (Option α) × β → Id (ForInStep (Option (Option α) × β))}
    (hf : Step f) : ∀ (l : List γ) (b : β),
      (forIn l (none, b) f).run.1 = none ∨ ∃ m, (forIn l (none, b) f).run.1 = some (some m)
  | [], b => by rw [List.forIn_nil]; exact Or.inl rfl
  | x :: l, b => by
    rw [List.forIn_cons]
    rcases hf x (none, b) with ⟨b', hb⟩ | ⟨m, b', hm⟩
    · rw [Id.run_bind, hb]
      exact forIn_found hf l b'
    · rw [Id.run_bind, hm]
      exact Or.inr ⟨m, rfl⟩

theorem forIn_some {α β γ : Type} {f : γ → Option (Option α) × β → Id (ForInStep (Option (Option α) × β))}
    {l : List γ} {b : β} {r : Option α} (h : (forIn l (none, b) f).run.1 = some r) (hf : Step f) :
    ∃ m, r = some m := by
  rcases forIn_found hf l b with h' | ⟨m, h'⟩
  · rw [h'] at h; cases h
  · rw [h'] at h; exact ⟨m, (Option.some.inj h).symm⟩

theorem forIn_yields {α β γ : Type} {f : γ → Option (Option α) × β → Id (ForInStep (Option (Option α) × β))}
    {x : γ} {l : List γ} {b : β} (h : (forIn (x :: l) (none, b) f).run.1 = none) (hf : Step f) :
    ∃ b', (f x (none, b)).run = .yield (none, b') ∧ (forIn l (none, b') f).run.1 = none := by
  rcases hf x (none, b) with ⟨b', hb⟩ | ⟨m, b', hm⟩
  · refine ⟨b', hb, ?_⟩
    rw [List.forIn_cons, Id.run_bind, hb] at h
    exact h
  · exfalso
    rw [List.forIn_cons, Id.run_bind, hm] at h
    have h2 : (some (some m) : Option (Option α)) = none := h
    cases h2

theorem forIn_each {α γ : Type} {f : γ → Option (Option α) × Unit → Id (ForInStep (Option (Option α) × Unit))}
    (hf : Step f) : ∀ (l : List γ), (forIn l (none, ()) f).run.1 = none →
      ∀ x ∈ l, (f x (none, ())).run = .yield (none, ())
  | [], _, x, hx => by simp at hx
  | y :: l, h, x, hx => by
    obtain ⟨b', hb, h'⟩ := forIn_yields h hf
    rcases List.mem_cons.1 hx with rfl | hx'
    · exact hb
    · exact forIn_each hf l h' x hx'

theorem toList_cons {α : Type} [Inhabited α] {a : Array α} (h : 0 < a.size) : a.toList = a[0]! :: a.toList.tail := by
  rcases ha : a.toList with _ | ⟨x, rest⟩
  · have h1 : a.toList.length = a.size := Array.length_toList
    rw [ha, List.length_nil] at h1
    omega
  · have hx : a[0]! = x := by
      rw [Array.getElem!_eq_getD, Array.getD_eq_getD_getElem?, ← Array.getElem?_toList, ha]
      rfl
    rw [hx]
    try rfl

/-- **The unit reads every bit of a parameter it names first**: where `readsWhy` finds nothing wrong at the names
`[nm]` and `d`'s first parameter is `nm`, of type `t`, the reads are one a unit input column and read each of `t`'s
bits at its port's column. -/
theorem readsWhy_first {c : Flock.Circuit} {d : Flock.Program.Defn} {reads : Array Bool} {nm : String}
    {t : Flock.Program.Ty} {rest : List (String × Flock.Program.Ty)} (hd : d.params.toList = (nm, t) :: rest)
    (h : readsWhy c d reads [nm] = none) :
    reads.size = min (c.unit.inWords * Flock.WORD) c.unit.constPos ∧
      ∀ β < bitsOf t, reads.getD (((Flock.portCols c.unit.inGroups).getD (β / 16) (0, 0)).1 + β % 16) false = true := by
  unfold readsWhy at h
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', ← Array.forIn_toList] at h
  split at h
  · cases h
  rename_i hsz
  refine ⟨bne_eq_false_iff_eq.1 (Bool.eq_false_iff.2 hsz), ?_⟩
  try simp only [Id.run_bind] at h
  split at h
  · rename_i r hr
    obtain ⟨m, rfl⟩ := forIn_some hr (by
      intro y s
      try dsimp only
      repeat' split
      all_goals first | exact Or.inl ⟨_, rfl⟩ | exact Or.inr ⟨_, _, rfl⟩)
    cases h
  rename_i hr
  try simp only [Id.run_bind] at h
  split at h
  · rename_i r hr3
    obtain ⟨m, rfl⟩ := forIn_some hr3 (by
      intro y s
      try dsimp only
      split
      · try simp only [Id.run_bind]
        split
        · rename_i r' hr'
          obtain ⟨m', rfl⟩ := forIn_some hr' (by
            intro y s
            try dsimp only
            repeat' split
            all_goals first | exact Or.inl ⟨_, rfl⟩ | exact Or.inr ⟨_, _, rfl⟩)
          exact Or.inr ⟨_, _, rfl⟩
        · exact Or.inl ⟨_, rfl⟩
      · exact Or.inl ⟨_, rfl⟩)
    cases h
  rename_i hr2
  rw [hd] at hr2
  obtain ⟨s', hs', -⟩ := forIn_yields hr2 (by
    intro y s
    try dsimp only
    split
    · try simp only [Id.run_bind]
      split
      · rename_i r' hr'
        obtain ⟨m', rfl⟩ := forIn_some hr' (by
          intro y s
          try dsimp only
          repeat' split
          all_goals first | exact Or.inl ⟨_, rfl⟩ | exact Or.inr ⟨_, _, rfl⟩)
        exact Or.inr ⟨_, _, rfl⟩
      · exact Or.inl ⟨_, rfl⟩
    · exact Or.inl ⟨_, rfl⟩)
  try dsimp only at hs'
  have hc : [nm].contains nm = true := by simp
  rw [ite_eq_left hc] at hs'
  try simp only [Id.run_bind] at hs'
  split at hs'
  · cases hs'
  rename_i hr3
  intro β hβ
  have hb := forIn_each (by
    intro y s
    try dsimp only
    repeat' split
    all_goals first | exact Or.inl ⟨_, rfl⟩ | exact Or.inr ⟨_, _, rfl⟩) _ hr3 β (by
    simp only [List.mem_range'_1, Std.Legacy.Range.size]
    omega)
  try dsimp only at hb
  split at hb
  · rename_i hc'
    rw [Nat.zero_add] at hc'
    exact hc'
  · cases hb

theorem fieldBit_first {c : Flock.Circuit} {nm : String} {w : ℕ} {ws : List (String × ℕ)} {β : ℕ} (hu : 0 < c.upv)
    (hβ : β / 16 < w) : fieldBit c ((nm, w) :: ws) nm β = some (0, β / 16, β % 16) := by
  have h1 : w ≠ 0 := by omega
  have h2 : ¬ c.upv * w ≤ β / 16 := by
    have := Nat.le_mul_of_pos_left w hu
    omega
  simp [fieldBit, List.findIdx?_cons, h1, h2, Nat.div_eq_of_lt hβ, Nat.mod_eq_of_lt hβ]

/-- **Bits' parameter check makes each bit of the first parameter a message bit**: where `paramsWhy` finds nothing
wrong and `d`'s first parameter is `nm`, of type `t`, instance 0's input port `β / 16`'s bit `β % 16` is a message bit
for each of `t`'s bits `β`. -/
theorem paramsWhy_first {c : Flock.Circuit} {d : Flock.Program.Defn} {nm : String} {t : Flock.Program.Ty}
    {rest : List (String × Flock.Program.Ty)} (hd : d.params.toList = (nm, t) :: rest) (hu : 0 < c.upv)
    (h : paramsWhy c d = none) {β : ℕ} (hβ : β < bitsOf t) :
    ∃ q byte bit, srcAt c (Index.of c) 0 (β / 16) (β % 16) = .msg q byte bit := by
  have hp0 : d.params[0]! = (nm, t) := by
    rw [Array.getElem!_eq_getD, Array.getD_eq_getD_getElem?, ← Array.getElem?_toList, hd]; try rfl
  have hsz : 0 < d.params.size := by
    have h1 : d.params.toList.length = d.params.size := Array.length_toList
    rw [hd, List.length_cons] at h1; omega
  have hβw : β / 16 < wordsOf t := by unfold wordsOf; omega
  have hlt : 16 * (β / 16 % wordsOf t) + β % 16 < bitsOf t := by rw [Nat.mod_eq_of_lt hβw]; omega
  have hmul : β < 16 * (c.upv * wordsOf t) := by
    have h1 := Nat.le_mul_of_pos_left (wordsOf t) hu
    have h2 : bitsOf t ≤ 16 * wordsOf t := by unfold wordsOf; omega
    omega
  unfold paramsWhy at h
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', Id.run_bind] at h
  split at h
  · rename_i r hr
    obtain ⟨m, rfl⟩ := forIn_some hr (by
      intro j s
      try dsimp only
      try simp only [Id.run_bind]
      split
      · rename_i r' hr'
        obtain ⟨m', rfl⟩ := forIn_some hr' (by
          intro y s
          try dsimp only
          repeat' split
          all_goals first | exact Or.inl ⟨_, rfl⟩ | exact Or.inr ⟨_, _, rfl⟩)
        exact Or.inr ⟨_, _, rfl⟩
      · exact Or.inl ⟨_, rfl⟩)
    cases h
  rename_i hr
  have hj := forIn_each (by
    intro j s
    try dsimp only
    try simp only [Id.run_bind]
    split
    · rename_i r' hr'
      obtain ⟨m', rfl⟩ := forIn_some hr' (by
        intro y s
        try dsimp only
        repeat' split
        all_goals first | exact Or.inl ⟨_, rfl⟩ | exact Or.inr ⟨_, _, rfl⟩)
      exact Or.inr ⟨_, _, rfl⟩
    · exact Or.inl ⟨_, rfl⟩) _ hr 0 (by
    simp only [List.mem_range'_1, Std.Legacy.Range.size]
    omega)
  try dsimp only at hj
  rw [hp0] at hj
  try dsimp only at hj
  try simp only [Id.run_bind] at hj
  split at hj
  · cases hj
  rename_i hr'
  have hb := forIn_each (by
    intro y s
    try dsimp only
    repeat' split
    all_goals first | exact Or.inl ⟨_, rfl⟩ | exact Or.inr ⟨_, _, rfl⟩) _ hr' β (by
    simp only [List.mem_range'_1, Std.Legacy.Range.size]
    omega)
  try dsimp only at hb
  rw [hd, List.map_cons] at hb
  try dsimp only at hb
  rw [fieldBit_first hu hβw] at hb
  try dsimp only at hb
  simp only [hlt, ↓reduceIte] at hb
  rcases hS : srcAt c (Index.of c) 0 (β / 16) (β % 16) with _ | ⟨q, byte, bit⟩ | ⟨net, slot, col⟩ | ⟨col⟩ | ⟨k, off⟩
  all_goals first
    | exact ⟨_, _, _, rfl⟩
    | exfalso
      rw [hS] at hb
      try dsimp only at hb
      try simp only [Bool.false_eq_true, ↓reduceIte] at hb
      repeat' split at hb
      all_goals cases hb

theorem fold_set_true {N : ℕ} : ∀ (r : List ℕ) (m : Array Bool) {y : ℕ},
    (r.foldl (fun m x => if x < N then m.set! x true else m) m).getD y false = true →
      m.getD y false = true ∨ (y ∈ r ∧ y < N)
  | [], m, y, h => Or.inl h
  | x :: r, m, y, h => by
    rw [List.foldl_cons] at h
    rcases fold_set_true r _ h with h1 | ⟨hy, hyN⟩
    · try dsimp only at h1
      by_cases hx : x < N
      · rw [ite_eq_left hx, Array.set!_eq_setIfInBounds, Array.getD_eq_getD_getElem?,
          Array.getElem?_setIfInBounds] at h1
        by_cases hxy : x = y
        · subst hxy; exact Or.inr ⟨List.mem_cons.2 (Or.inl rfl), hx⟩
        · rw [ite_eq_right hxy, ← Array.getD_eq_getD_getElem?] at h1; exact Or.inl h1
      · rw [ite_eq_right hx] at h1; exact Or.inl h1
    · exact Or.inr ⟨List.mem_cons_of_mem _ hy, hyN⟩

theorem fold_rows_true {N : ℕ} : ∀ (rs : List (List ℕ)) (m : Array Bool) {y : ℕ},
    (rs.foldl (fun m r => r.foldl (fun m x => if x < N then m.set! x true else m) m) m).getD y false = true →
      m.getD y false = true ∨ ∃ r ∈ rs, y ∈ r ∧ y < N
  | [], m, y, h => Or.inl h
  | r :: rs, m, y, h => by
    rw [List.foldl_cons] at h
    rcases fold_rows_true rs _ h with h1 | ⟨r', hr', hy⟩
    · try dsimp only at h1
      rcases fold_set_true r m h1 with h2 | hy
      · exact Or.inl h2
      · exact Or.inr ⟨r, List.mem_cons.2 (Or.inl rfl), hy⟩
    · exact Or.inr ⟨r', List.mem_cons_of_mem _ hr', hy⟩

/-- **A read input column is read by a computed row**: `unitReads` marks an input column only where a computed row of
the unit net reads it in its A or its B. -/
theorem unitReads_true {c : Flock.Circuit} {x : ℕ} (h : (unitReads c).getD x false = true) :
    ∃ i < (Rows.ofNet c.unit).nComp, x ∈ (Rows.ofNet c.unit).a.getD i [] ++ (Rows.ofNet c.unit).b.getD i [] ∧
      x < (Rows.ofNet c.unit).nIn := by
  unfold unitReads at h
  try dsimp only at h
  rw [← Array.foldl_toList] at h
  rcases fold_rows_true _ _ h with h0 | ⟨r, hr, hx, hxN⟩
  · exfalso
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_replicate] at h0
    split at h0 <;> simp at h0
  · have hsa : (Rows.ofNet c.unit).a.size = (Rows.ofNet c.unit).nComp := by simp [Rows.ofNet]
    have hsb : (Rows.ofNet c.unit).b.size = (Rows.ofNet c.unit).nComp := by simp [Rows.ofNet]
    rcases Array.mem_append.1 (Array.mem_def.2 hr) with ha | hb
    · obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 ha
      have hget : (Rows.ofNet c.unit).a.getD i [] = (Rows.ofNet c.unit).a[i] := by
        rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hi, Option.getD_some]
      exact ⟨i, by omega, List.mem_append_left _ (by rw [hget]; exact hx), hxN⟩
    · obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hb
      have hget : (Rows.ofNet c.unit).b.getD i [] = (Rows.ofNet c.unit).b[i] := by
        rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hi, Option.getD_some]
      exact ⟨i, by omega, List.mem_append_right _ (by rw [hget]; exact hx), hxN⟩

/-- **A port bit's source is its port's**: where the ports are in column order, input column `a + τ` of port `p` at
column `a`, inside the port and the unit's input words, is instance 0's port `p` bit `τ`. -/
theorem srcKey_port {c : Flock.Circuit} {ix : Index} (hO : Check.PortsOrdered c) {p τ : ℕ}
    (hp : p < (Flock.portCols c.unit.inGroups).size) (hτ : τ < ((Flock.portCols c.unit.inGroups).getD p (0, 0)).2)
    (hW : ((Flock.portCols c.unit.inGroups).getD p (0, 0)).1 + τ < c.unit.inWords * Flock.WORD) :
    srcKey c ix (((Flock.portCols c.unit.inGroups).getD p (0, 0)).1 + τ) = srcAt c ix 0 p τ := by
  have hfind : (List.range (Flock.portCols c.unit.inGroups).size).find? (fun q =>
      decide (((Flock.portCols c.unit.inGroups).getD q (0, 0)).1 ≤ ((Flock.portCols c.unit.inGroups).getD p (0, 0)).1 + τ ∧
        ((Flock.portCols c.unit.inGroups).getD p (0, 0)).1 + τ <
          ((Flock.portCols c.unit.inGroups).getD q (0, 0)).1 + ((Flock.portCols c.unit.inGroups).getD q (0, 0)).2)) =
      some p := by
    rw [List.find?_range_eq_some]
    refine ⟨?_, List.mem_range.2 hp, fun j hj => ?_⟩
    · simp only [decide_eq_true_eq]; omega
    · have h1 := Check.ordered_le hO (p := j) p hj hp
      simp only [Bool.not_eq_true', decide_eq_false_iff_not, not_and, Nat.not_lt]
      omega
  simp only [srcKey]
  rw [Nat.mod_eq_of_lt hW, Nat.div_eq_of_lt hW, hfind]
  dsimp only
  rw [Nat.add_sub_cancel_left]

theorem keyIdx_isSome {c : Flock.Circuit} {n x : ℕ} (hx : x < n) (hz : srcKey c (Index.of c) x ≠ .zero) :
    ∃ b, keyIdx c n x = some b := by
  simp only [keyIdx, hz, ↓reduceIte]
  exact Option.isSome_iff_exists.1 (List.find?_isSome.2 ⟨x, List.mem_range.2 hx, by simp⟩)

end Properties.NetworkScan.Reads
