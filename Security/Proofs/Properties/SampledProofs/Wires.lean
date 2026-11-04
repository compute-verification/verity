import SampledProofs.Bits

/-! # Where a source bit's wire lands (`SampledProofs.Bits.wiresOf`)

A source bit's wire is either a writer's output bit or another source bit (`wiresOf_mem`). A writer's output bit is the
AND of a returned output's copy row, since `outGate` names a computed row and a row lowers to two XORs and their AND
(`writers_gate`). A source bit is an input gate the constants leave free, which lowers to `.input` (`sources_gate`). So
in the audit's members' lowered programs side by side (`auditGates`, member `m`'s from `base ms m`), a wire's gate is
an `.and` or an `.input` (`wiresOf_gate`).

The positions are read off the lowered program through `prog_getElem?`: gate `g`'s lowered gates start at `off kp g`,
the lowered sizes of the gates before it (`off_eq`), and an input gate sits at its own index (`off_input`).
-/

open SampledProofs.Bits

namespace Properties.SampledProofs.Wires

/-- A member's lowered program: its key program, for its units, at its constants. -/
def memberGates (ci : Flock.Circuit × Array Nat) : List LGate :=
  (prog (keyProg ci.1 ci.2.size) (consts ci.1 ci.2.size)).1

/-- The audit's members' lowered programs side by side, member `m`'s from `base ms m` (`auditGates_getElem?`). -/
def auditGates (ms : Members) : List LGate := ms.flatMap memberGates

/-! ## Lists -/

theorem lt_of_getElem? {α : Type} {l : List α} {i : Nat} {a : α} (h : l[i]? = some a) : i < l.length := by
  rcases Nat.lt_or_ge i l.length with hi | hi
  · exact hi
  · rw [List.getElem?_eq_none hi] at h
    cases h

theorem getElem?_flatMap {α β : Type} [Inhabited α] (F : α → List β) : ∀ (l : List α) (m y : Nat), m < l.length →
    y < (F (l.getD m default)).length →
    (l.flatMap F)[((l.take m).map fun a => (F a).length).sum + y]? = (F (l.getD m default))[y]?
  | [], _, _, h, _ => absurd h (Nat.not_lt_zero _)
  | a :: l, 0, y, _, hy => by
    simp only [List.getD_cons_zero] at hy ⊢
    simp only [List.take_zero, List.map_nil, List.sum_nil, Nat.zero_add, List.flatMap_cons]
    exact List.getElem?_append_left hy
  | a :: l, m + 1, y, hm, hy => by
    simp only [List.getD_cons_succ] at hy ⊢
    simp only [List.take_succ_cons, List.map_cons, List.sum_cons, List.flatMap_cons]
    rw [Nat.add_assoc, List.getElem?_append_right (Nat.le_add_right _ _), Nat.add_sub_cancel_left]
    exact getElem?_flatMap F l m y (Nat.lt_of_succ_lt_succ hm) hy

theorem filter_range_lt {g : Nat} : ∀ {N : Nat}, g ≤ N →
    (List.range N).filter (fun x => decide (x < g)) = List.range g
  | 0, h => by simp [Nat.le_zero.1 h]
  | N + 1, h => by
    rcases Nat.lt_or_eq_of_le h with h | rfl
    · rw [List.range_succ, List.filter_append, filter_range_lt (Nat.le_of_lt_succ h)]
      simp [Nat.not_lt.2 (Nat.le_of_lt_succ h)]
    · exact List.filter_eq_self.2 fun x hx => by simpa using List.mem_range.1 hx

theorem sum_map_const_range (n c : Nat) : ((List.range n).map fun _ => c).sum = n * c := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [List.range_succ, List.map_append, List.sum_append_nat, ih, Nat.add_mul]
    simp

/-! ## The lowered program -/

theorem lower_length (kp : KeyProg) (κ : Nat → Option Bool) (g : Nat) :
    (lower kp κ g).length = gateSize kp g := by
  unfold lower gateSize
  cases kp.row? g with
  | some ab =>
    obtain ⟨a, b⟩ := ab
    rfl
  | none => cases κ g <;> rfl

theorem off_eq {kp : KeyProg} {g : Nat} (hg : g ≤ kp.size) :
    off kp g = ((List.range g).map (gateSize kp)).sum := by
  unfold off
  rw [filter_range_lt hg]

theorem prog_length (kp : KeyProg) (κ : Nat → Option Bool) : (prog kp κ).1.length = off kp kp.size := by
  rw [off_eq (Nat.le_refl _)]
  simp only [prog, List.length_flatMap, lower_length]

/-- **Gate `g`'s lowered gates sit from `off kp g`** in the lowered program. -/
theorem prog_getElem? {kp : KeyProg} {κ : Nat → Option Bool} {g i : Nat} (hg : g < kp.size)
    (hi : i < gateSize kp g) : (prog kp κ).1[off kp g + i]? = (lower kp κ g)[i]? := by
  have hd : (List.range kp.size).getD g default = g := by
    rw [List.getD_eq_getElem?_getD, List.getElem?_range hg] <;> rfl
  have h := getElem?_flatMap (lower kp κ) (List.range kp.size) g i (by rw [List.length_range]; exact hg)
    (by rw [hd, lower_length]; exact hi)
  rw [hd, List.take_range, Nat.min_eq_left (Nat.le_of_lt hg)] at h
  simp only [lower_length] at h
  rw [off_eq (Nat.le_of_lt hg)]
  exact h

theorem row?_input {kp : KeyProg} {g : Nat} (h : g < kp.inputs) : kp.row? g = none := by
  simp [KeyProg.row?, h]

theorem gateSize_input {kp : KeyProg} {g : Nat} (h : g < kp.inputs) : gateSize kp g = 1 := by
  simp [gateSize, row?_input h]

theorem sum_gateSize_input {kp : KeyProg} : ∀ {g : Nat}, g ≤ kp.inputs →
    ((List.range g).map (gateSize kp)).sum = g
  | 0, _ => by simp
  | g + 1, h => by
    rw [List.range_succ, List.map_append, List.sum_append_nat, sum_gateSize_input (Nat.le_of_succ_le h)]
    simp [gateSize_input (Nat.lt_of_succ_le h)]

/-- **An input gate sits at its own index** in the lowered program. -/
theorem off_input {kp : KeyProg} {g : Nat} (h : g ≤ kp.inputs) : off kp g = g := by
  rw [off_eq (kp := kp) (Nat.le_trans h (Nat.le_add_right _ _)), sum_gateSize_input h]

theorem lower_input {kp : KeyProg} {κ : Nat → Option Bool} {g : Nat} (h : g < kp.inputs) (hκ : κ g = none) :
    lower kp κ g = [.input] := by
  simp [lower, row?_input h, hκ]

theorem lower_row {kp : KeyProg} {κ : Nat → Option Bool} {g : Nat} (h : (kp.row? g).isSome = true) :
    gateSize kp g = 3 ∧ (lower kp κ g)[2]? = some (.and [off kp g, off kp g + 1]) := by
  obtain ⟨⟨a, b⟩, hab⟩ := Option.isSome_iff_exists.1 h
  refine ⟨by simp [gateSize, h], ?_⟩
  unfold lower
  rw [hab] <;> rfl

/-! ## Output gates are rows -/

theorem row_lt {I Y X u k n upv e : Nat} (hu : u < n) (hk : k < upv) (he : e < X) (hY : Y = upv * X) :
    I ≤ I + u * Y + (k * X + e) ∧ I + u * Y + (k * X + e) - I < n * Y := by
  have h1 : (k + 1) * X ≤ upv * X := Nat.mul_le_mul_right X (Nat.succ_le_of_lt hk)
  have h2 : (u + 1) * Y ≤ n * Y := Nat.mul_le_mul_right Y (Nat.succ_le_of_lt hu)
  rw [Nat.add_mul] at h1 h2
  constructor <;> omega

theorem row_isSome {kp : KeyProg} {g : Nat} (h1 : kp.inputs ≤ g) (h2 : g - kp.inputs < kp.rows.size) :
    g < kp.size ∧ (kp.row? g).isSome = true := by
  refine ⟨by unfold KeyProg.size; omega, ?_⟩
  unfold KeyProg.row?
  rw [ite_eq_right (Nat.not_lt.2 h1), Array.getElem?_eq_getElem h2] <;> rfl

theorem keyProg_rows_size (c : Flock.Circuit) (n : Nat) :
    (keyProg c n).rows.size = n * ((Rows.ofNet c.unit).stack c.upv).nComp := by
  simp only [keyProg, List.size_toArray, List.length_flatMap, List.length_map, List.length_range]
  exact sum_map_const_range _ _

/-- **An output bit's gate is a row** of the key program. -/
theorem outGate_row {c : Flock.Circuit} {n u k j t g : Nat} (h : outGate c n u k j t = some g) :
    g < (keyProg c n).size ∧ ((keyProg c n).row? g).isSome = true := by
  unfold outGate at h
  dsimp only at h
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
    obtain ⟨⟨⟨⟨⟨hu, hk⟩, -⟩, -⟩, hlo⟩, hhi⟩ := hc
    cases h
    obtain ⟨h1, h2⟩ := row_lt (I := 2 + n * ((Rows.ofNet c.unit).stack c.upv).nIn)
      (Y := ((Rows.ofNet c.unit).stack c.upv).nComp) (X := (Rows.ofNet c.unit).nComp + 1)
      (e := 1 + (outCol c j t - (Rows.ofNet c.unit).nIn)) hu hk (by omega) rfl
    exact row_isSome (kp := keyProg c n) h1 (by rw [keyProg_rows_size]; exact h2)
  · cases h

/-! ## Members -/

theorem base_eq (ms : Members) (m : Nat) :
    base ms m = ((ms.take m).map fun ci => (memberGates ci).length).sum := by
  simp only [base, memberGates, prog_length]

theorem auditGates_getElem? {ms : Members} {m y : Nat} (hm : m < ms.length)
    (hy : y < (memberGates (ms.getD m default)).length) :
    (auditGates ms)[base ms m + y]? = (memberGates (ms.getD m default))[y]? := by
  rw [base_eq]
  exact getElem?_flatMap memberGates ms m y hm hy

theorem member_writer {c : Flock.Circuit} {n u k j t g : Nat} (hg : outGate c n u k j t = some g) :
    ∃ ws, (prog (keyProg c n) (consts c n)).1[posOf (keyProg c n) g]? = some (.and ws) := by
  obtain ⟨hlt, hrow⟩ := outGate_row hg
  obtain ⟨h3, hand⟩ := lower_row (κ := consts c n) hrow
  have hp : posOf (keyProg c n) g = off (keyProg c n) g + 2 := by
    unfold posOf
    omega
  refine ⟨[off (keyProg c n) g, off (keyProg c n) g + 1], ?_⟩
  rw [hp, prog_getElem? (i := 2) hlt (by omega)]
  exact hand

theorem member_source {c : Flock.Circuit} {n g : Nat} (hg : g < (keyProg c n).inputs) (hκ : consts c n g = none) :
    (prog (keyProg c n) (consts c n)).1[g]? = some .input := by
  have hs := gateSize_input hg
  have h := prog_getElem? (kp := keyProg c n) (κ := consts c n) (g := g) (i := 0)
    (Nat.lt_of_lt_of_le hg (Nat.le_add_right _ _)) (by omega)
  rw [off_input (Nat.le_of_lt hg), Nat.add_zero, lower_input hg hκ] at h
  exact h

theorem mem_writers {ms : Members} {w : Nat} {r : RowPos} (h : (w, r) ∈ writers ms) :
    ∃ m < ms.length, ∃ u k j t g, outGate (ms.getD m default).1 (ms.getD m default).2.size u k j t = some g ∧
      w = base ms m + posOf (keyProg (ms.getD m default).1 (ms.getD m default).2.size) g := by
  simp only [writers, List.mem_flatMap, List.mem_range, List.mem_filterMap] at h
  obtain ⟨m, hm, u, -, k, -, j, -, t, -, he⟩ := h
  refine ⟨m, hm, u, k, j, t, ?_⟩
  revert he
  cases hg : outGate (ms.getD m default).1 (ms.getD m default).2.size u k j t with
  | none => simp
  | some g =>
    cases outRowBit (ms.getD m default).1 u k j t with
    | none => simp
    | some r' =>
      intro he
      simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq] at he
      exact ⟨g, rfl, he.1.symm⟩

theorem mem_sources {ms : Members} {w : Nat} {r : RowPos} (h : (w, r) ∈ sources ms) :
    ∃ m < ms.length, ∃ g < (keyProg (ms.getD m default).1 (ms.getD m default).2.size).inputs,
      consts (ms.getD m default).1 (ms.getD m default).2.size g = none ∧ w = base ms m + g := by
  simp only [sources, List.mem_flatMap, List.mem_range, List.mem_filterMap] at h
  obtain ⟨m, hm, g, hg, he⟩ := h
  refine ⟨m, hm, g, hg, ?_⟩
  revert he
  cases consts (ms.getD m default).1 (ms.getD m default).2.size g with
  | some _ => simp
  | none =>
    cases rowBit (ms.getD m default).1 (ms.getD m default).2.size g with
    | none => simp
    | some r' =>
      intro he
      simp at he
      obtain ⟨h1, -⟩ := he
      exact ⟨rfl, by omega⟩

/-! ## Wires -/

/-- **A writer's output bit is an AND** in the audit's lowered programs: its copy row's. -/
theorem writers_gate {ms : Members} {w : Nat} {r : RowPos} (h : (w, r) ∈ writers ms) :
    ∃ ws, (auditGates ms)[w]? = some (.and ws) := by
  obtain ⟨m, hm, u, k, j, t, g, hg, rfl⟩ := mem_writers h
  obtain ⟨ws, hws⟩ := member_writer hg
  exact ⟨ws, by rw [auditGates_getElem? hm (lt_of_getElem? hws)]; exact hws⟩

/-- **A source bit is an input** in the audit's lowered programs: an input gate the constants leave free. -/
theorem sources_gate {ms : Members} {w : Nat} {r : RowPos} (h : (w, r) ∈ sources ms) :
    (auditGates ms)[w]? = some .input := by
  obtain ⟨m, hm, g, hg, hκ, rfl⟩ := mem_sources h
  have hs := member_source hg hκ
  rw [auditGates_getElem? hm (lt_of_getElem? hs)]
  exact hs

/-- **A source bit's wire is a writer's output bit or a source bit.** -/
theorem wiresOf_mem {ms : Members} {x w : Nat} (h : wiresOf ms x = some w) :
    (∃ r, (w, r) ∈ writers ms) ∨ ∃ r, (w, r) ∈ sources ms := by
  simp only [wiresOf] at h
  split at h
  · cases h
  · split at h
    · rename_i hw
      cases h
      exact Or.inl ⟨_, List.mem_of_find?_eq_some hw⟩
    · simp only [Option.filter_eq_some_iff, Option.map_eq_some_iff] at h
      obtain ⟨⟨p, hp, rfl⟩, -⟩ := h
      exact Or.inr ⟨p.2, List.mem_of_find?_eq_some hp⟩

/-- **A source bit's wire is an AND or an input** in the audit's members' lowered programs side by side: a writer's
copy-row AND, or a source gate the constants leave free. -/
theorem wiresOf_gate {ms : Members} {x w : Nat} (h : wiresOf ms x = some w) :
    (∃ ws, (auditGates ms)[w]? = some (.and ws)) ∨ (auditGates ms)[w]? = some .input := by
  rcases wiresOf_mem h with ⟨r, hr⟩ | ⟨r, hr⟩
  · exact Or.inl (writers_gate hr)
  · exact Or.inr (sources_gate hr)

end Properties.SampledProofs.Wires
