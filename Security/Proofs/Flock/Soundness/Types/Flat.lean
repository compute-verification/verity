import Flock.Derive
import Proofs.Flock.Soundness.Audit.Circuit

/-!
# Derived rows compute their circuit type (flat types)

`Flock.Derive.derive` is the function the verifier runs to lay a circuit type out as rows (Project doc
`verified-lowering-design.md` §1.2). For every flat type it accepts (AND items over linear forms):

* **`compose_sound`**: column values that satisfy the derived rows, with the constant at 1, carry on each output's column
  the type's output (`eval`) on the inputs' columns.
* **`compose_complete`**: on every input, the honest column values (`honest`) satisfy the derived rows, with the
  constant at 1, and carry the inputs and the outputs on their columns. There are no assertion rows, so the rows ask
  nothing an honest evaluation doesn't meet.

Both are about the executed `derive`. The layout's geometry comes from `derive`'s own check (`geoOk`), so nothing here is
about how `place` arranges groups. Calls, exported inputs and reads come later.
-/

namespace FlockSoundness.Types

open Flock Flock.Derive Audit

/-- A form's value: the XOR of its wires' values, and its constant. -/
def formVal (v : ℕ → Bool) (f : Form) : Bool := Bool.xor (xorSum f.ws.toList v) (decide (f.c = 1))

/-- An item's value on the wires before it. A flat type's items are ANDs. -/
def itemVal (v : ℕ → Bool) : Item → Bool
  | .and a b => formVal v a && formVal v b
  | _ => false

/-- The wire values once the items from wire `w` on have run, each on the values before it. -/
def run (v : ℕ → Bool) (w : ℕ) : List Item → ℕ → Bool
  | [] => v
  | it :: rest => run (fun j => if j = w then itemVal v it else v j) (w + 1) rest

/-- **What a flat type computes** on inputs `x`: its output forms on its wire values. -/
def eval (t : CircuitType) (x : ℕ → Bool) : List Bool :=
  t.outputs.toList.map (formVal (run x t.inBits t.items.toList))

/-- Rows satisfied by column values `z`: row `r` is `z r = (A_r · z)(B_r · z)`. -/
def Sat (R : List (List ℕ × List ℕ)) (z : ℕ → Bool) : Prop :=
  ∀ r p, R[r]? = some p → z r = (xorSum p.1 z && xorSum p.2 z)

/-- **The honest column values** on inputs `x`: each input bit on its column, each AND item's value on its row, each
output on its copy row, 1 on the constant, and 0 on every forced-zero row. -/
def honest (t : CircuitType) (G : Geo) (x : ℕ → Bool) (r : ℕ) : Bool :=
  if r = G.const then true
  else if r < G.andBase then
    match bitAt G.ins r with
    | some i => x i
    | none => false
  else if r < G.andBase + G.nAnd then run x t.inBits t.items.toList (t.inBits + (r - G.andBase))
  else
    match bitAt G.outs r with
    | some o => formVal (run x t.inBits t.items.toList) (t.outputs[o]?.getD default)
    | none => false

/-! ## Forms and wires -/

theorem xorSum_append {ι : Type} (l m : List ι) (v : ι → Bool) :
    xorSum (l ++ m) v = Bool.xor (xorSum l v) (xorSum m v) := by
  induction l with
  | nil => simp [xorSum]
  | cons a l ih =>
    simp only [xorSum, List.cons_append, List.map_cons, List.foldr_cons] at ih ⊢
    rw [ih, Bool.xor_assoc]

theorem formVal_congr {v v' : ℕ → Bool} {f : Form} (h : ∀ x ∈ f.ws.toList, v x = v' x) :
    formVal v f = formVal v' f := by
  unfold formVal; rw [xorSum_congr h]

theorem itemVal_congr {v v' : ℕ → Bool} {it : Item}
    (h : ∀ f ∈ (CircuitType.itemForms it).toList, ∀ x ∈ f.ws.toList, v x = v' x) : itemVal v it = itemVal v' it := by
  cases it with
  | and a b =>
    simp only [itemVal]
    rw [formVal_congr (h a (by simp [CircuitType.itemForms])), formVal_congr (h b (by simp [CircuitType.itemForms]))]
  | call _ _ => rfl
  | read _ _ => rfl
  | bad _ => rfl

/-- A form's row entries, evaluated: its value on the columns' values, the constant at 1. -/
theorem xorSum_formCols (G : Geo) (nIn : ℕ) (f : Form) (z : ℕ → Bool) (h1 : z G.const = true) :
    xorSum (G.formCols nIn f) z = formVal (fun w => z (G.col nIn w)) f := by
  unfold Geo.formCols formVal
  rw [xorSum_append, xorSum_map]
  by_cases hc : f.c = 1
  · simp [hc, xorSum, h1]; rfl
  · simp [hc, xorSum]; rfl

theorem run_lt : ∀ (items : List Item) (v : ℕ → Bool) (w j : ℕ), j < w → run v w items j = v j
  | [], _, _, _, _ => rfl
  | _ :: rest, v, w, j, h => by
    simp only [run]
    rw [run_lt rest _ (w + 1) j (by omega)]
    simp [show j ≠ w by omega]

/-- Every item reads only earlier wires: item `k`'s forms read wires below `w + k`. -/
def ReadsEarlier (w : ℕ) (items : List Item) : Prop :=
  ∀ k (hk : k < items.length), ∀ f ∈ (CircuitType.itemForms items[k]).toList, ∀ x ∈ f.ws.toList, x < w + k

/-- Item `k`'s wire holds its value on the final wire values. -/
theorem run_item : ∀ (items : List Item) (v : ℕ → Bool) (w : ℕ), ReadsEarlier w items →
    ∀ k (hk : k < items.length), run v w items (w + k) = itemVal (run v w items) items[k]
  | [], _, _, _, k, hk => absurd hk (by simp)
  | it :: rest, v, w, h, k, hk => by
    cases k with
    | zero =>
      simp only [run, Nat.add_zero, List.getElem_cons_zero]
      rw [run_lt rest _ (w + 1) w (by omega)]
      simp only [↓reduceIte]
      apply itemVal_congr
      intro f hf x hx
      have hx' := h 0 (by simp) f (by simpa using hf) x hx
      rw [run_lt rest _ (w + 1) x (by omega)]
      simp [show x ≠ w by omega]
    | succ k =>
      simp only [run, List.getElem_cons_succ]
      rw [show w + (k + 1) = (w + 1) + k by omega]
      exact run_item rest _ (w + 1) (fun k' hk' f hf x hx => by
        have := h (k' + 1) (by simp; omega) f (by simpa using hf) x hx; omega) k (by simpa using hk)

/-! ## What `derive` checked -/

theorem derive_ok {t : CircuitType} {inSplit outSplit : List ℕ} {d : Derived}
    (hd : derive t inSplit outSplit = .ok d) :
    flatOk t = true ∧ geoOk t (geo t inSplit outSplit) = true ∧
      d = ⟨geo t inSplit outSplit, t.inBits, t.outBits, (List.range ((geo t inSplit outSplit).const + 1)).map
        ((geo t inSplit outSplit).rowAt t)⟩ := by
  unfold derive at hd
  split_ifs at hd with h1 h2
  dsimp only at hd
  split_ifs at hd with h3
  simp only [Bool.not_eq_true', Bool.not_eq_false] at h1 h3
  exact ⟨by simpa using h1, by simpa using h3, (Except.ok.inj hd).symm⟩

theorem formOk_lt {n : ℕ} {f : Form} (h : CircuitType.formOk n f = true) : ∀ x ∈ f.ws.toList, x < n := by
  unfold CircuitType.formOk at h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  intro x hx
  have := Array.all_eq_true.mp h.1.2
  obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem hx
  simpa using this i (by simpa using hi)

theorem flatOk_items {t : CircuitType} (h : flatOk t = true) (k : ℕ) (hk : k < t.items.size) :
    ∃ a b, t.items[k] = .and a b ∧ (∀ x ∈ a.ws.toList, x < t.inBits + k) ∧ (∀ x ∈ b.ws.toList, x < t.inBits + k) := by
  unfold flatOk at h
  simp only [Bool.and_eq_true, List.all_eq_true, List.mem_range] at h
  have := h.1.1 k hk
  rw [Array.getElem?_eq_getElem hk] at this
  revert this
  cases hi : t.items[k] with
  | and a b =>
    intro hh
    simp only [Bool.and_eq_true] at hh
    exact ⟨a, b, rfl, formOk_lt hh.1, formOk_lt hh.2⟩
  | call _ _ => simp
  | read _ _ => simp
  | bad _ => simp

theorem flatOk_outputs {t : CircuitType} (h : flatOk t = true) :
    t.outputs.size = t.outBits ∧ ∀ o (ho : o < t.outputs.size), ∀ x ∈ t.outputs[o].ws.toList, x < t.inBits + t.items.size := by
  unfold flatOk at h
  simp only [Bool.and_eq_true, beq_iff_eq] at h
  refine ⟨h.1.2, fun o ho x hx => formOk_lt ?_ x hx⟩
  exact Array.all_eq_true.mp h.2 o ho

theorem flatOk_reads {t : CircuitType} (h : flatOk t = true) : ReadsEarlier t.inBits t.items.toList := by
  intro k hk f hf x hx
  have hk' : k < t.items.size := by simpa using hk
  obtain ⟨a, b, hab, ha, hb⟩ := flatOk_items h k hk'
  simp only [Array.getElem_toList, hab, CircuitType.itemForms, List.mem_cons, List.not_mem_nil, or_false] at hf
  rcases hf with rfl | rfl
  · exact ha x hx
  · exact hb x hx

/-- The geometry `geoOk` checks, as facts. -/
theorem geoOk_facts {t : CircuitType} {G : Geo} (h : geoOk t G = true) :
    G.andBase + G.nAnd ≤ G.const ∧
    (∀ i < t.inBits, ∃ c, colOf G.ins i = some c ∧ c < G.andBase ∧ bitAt G.ins c = some i) ∧
    (∀ r < G.andBase, ∀ i, bitAt G.ins r = some i → i < t.inBits ∧ colOf G.ins i = some r) ∧
    (∀ o < t.outBits, ∃ c, colOf G.outs o = some c ∧ G.andBase + G.nAnd ≤ c ∧ c < G.const ∧
      bitAt G.outs c = some o) ∧
    (∀ r, G.andBase + G.nAnd ≤ r → r < G.const → ∀ o, bitAt G.outs r = some o →
      o < t.outBits ∧ colOf G.outs o = some r) := by
  unfold geoOk at h
  simp only [Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, List.mem_range] at h
  obtain ⟨⟨⟨⟨h0, h1⟩, h2⟩, h3⟩, h4⟩ := h
  refine ⟨h0, fun i hi => ?_, fun r hr i hb => ?_, fun o ho => ?_, fun r hr1 hr2 o hb => ?_⟩
  · have := h1 i hi
    revert this
    cases hc : colOf G.ins i with
    | none => simp
    | some c =>
      intro hh
      simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hh
      exact ⟨c, rfl, hh.1, hh.2⟩
  · have := h2 r hr
    rw [hb] at this
    simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at this
    exact this
  · have := h3 o ho
    revert this
    cases hc : colOf G.outs o with
    | none => simp
    | some c =>
      intro hh
      simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hh
      exact ⟨c, rfl, hh.1.1, hh.1.2, hh.2⟩
  · have := h4 (r - (G.andBase + G.nAnd)) (by omega)
    rw [show G.andBase + G.nAnd + (r - (G.andBase + G.nAnd)) = r by omega, hb] at this
    simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at this
    exact this

theorem geo_nAnd (t : CircuitType) (inSplit outSplit : List ℕ) : (geo t inSplit outSplit).nAnd = t.items.size := rfl

/-! ## The rows, by kind -/

section rows

variable {t : CircuitType} {G : Geo}

theorem rowAt_const : G.rowAt t G.const = ([G.const], [G.const]) := by simp [Geo.rowAt]

theorem rowAt_in_some {r i : ℕ} (hr : r < G.andBase) (hc : G.andBase + G.nAnd ≤ G.const)
    (hb : bitAt G.ins r = some i) : G.rowAt t r = ([r], [r]) := by
  unfold Geo.rowAt
  rw [ite_eq_right (show ¬ r = G.const by omega), ite_eq_left hr, hb]

theorem rowAt_in_none {r : ℕ} (hr : r < G.andBase) (hc : G.andBase + G.nAnd ≤ G.const)
    (hb : bitAt G.ins r = none) : G.rowAt t r = ([], []) := by
  unfold Geo.rowAt
  rw [ite_eq_right (show ¬ r = G.const by omega), ite_eq_left hr, hb]

theorem rowAt_and {k : ℕ} {a b : Form} (hk : k < G.nAnd) (hc : G.andBase + G.nAnd ≤ G.const)
    (hi : t.items[k]? = some (.and a b)) :
    G.rowAt t (G.andBase + k) = (G.formCols t.inBits a, G.formCols t.inBits b) := by
  unfold Geo.rowAt
  rw [ite_eq_right (show ¬ G.andBase + k = G.const by omega), ite_eq_right (show ¬ G.andBase + k < G.andBase by omega),
    ite_eq_left (show G.andBase + k < G.andBase + G.nAnd by omega), Nat.add_sub_cancel_left, hi]

theorem rowAt_out_some {r o : ℕ} (hr1 : G.andBase + G.nAnd ≤ r) (hr2 : r < G.const) (hb : bitAt G.outs r = some o) :
    G.rowAt t r = (G.formCols t.inBits (t.outputs[o]?.getD default), [G.const]) := by
  unfold Geo.rowAt
  rw [ite_eq_right (show ¬ r = G.const by omega), ite_eq_right (show ¬ r < G.andBase by omega),
    ite_eq_right (show ¬ r < G.andBase + G.nAnd by omega), hb]

theorem rowAt_out_none {r : ℕ} (hr1 : G.andBase + G.nAnd ≤ r) (hr2 : r < G.const) (hb : bitAt G.outs r = none) :
    G.rowAt t r = ([], []) := by
  unfold Geo.rowAt
  rw [ite_eq_right (show ¬ r = G.const by omega), ite_eq_right (show ¬ r < G.andBase by omega),
    ite_eq_right (show ¬ r < G.andBase + G.nAnd by omega), hb]

end rows

theorem rows_get {G : Geo} {t : CircuitType} {r : ℕ} (hr : r ≤ G.const) :
    ((List.range (G.const + 1)).map (G.rowAt t))[r]? = some (G.rowAt t r) := by
  rw [List.getElem?_map, List.getElem?_range (by omega)]; rfl

theorem eval_getD (t : CircuitType) (x : ℕ → Bool) {o : ℕ} (ho : o < t.outputs.size) :
    (eval t x).getD o false = formVal (run x t.inBits t.items.toList) t.outputs[o] := by
  simp [eval, List.getD_eq_getElem?_getD, ho]

/-! ## Soundness -/

/-- **Derived rows compute their type.** If `derive` lays the flat type `t` out, column values that satisfy its rows with
the constant at 1 carry on output `o`'s column the type's output `o` on the inputs' columns. -/
theorem compose_sound {t : CircuitType} {inSplit outSplit : List ℕ} {d : Derived}
    (hd : derive t inSplit outSplit = .ok d) (z : ℕ → Bool) (hz : Sat d.rows z) (h1 : z d.geo.const = true)
    (o : ℕ) (ho : o < t.outputs.size) :
    z (d.outCol o) = (eval t fun i => z (d.inCol i)).getD o false := by
  obtain ⟨hflat, hgeo, rfl⟩ := derive_ok hd
  obtain ⟨hc, -, -, hout, -⟩ := geoOk_facts hgeo
  have hn := geo_nAnd t inSplit outSplit
  simp only [Derived.inCol, Derived.outCol] at h1 ⊢
  generalize hG : geo t inSplit outSplit = G at *
  obtain ⟨x, hx⟩ : ∃ x : ℕ → Bool, x = fun i => z ((colOf G.ins i).getD 0) := ⟨_, rfl⟩
  obtain ⟨V, hV⟩ : ∃ V, V = run x t.inBits t.items.toList := ⟨_, rfl⟩
  have hsat : ∀ r, r ≤ G.const → z r = (xorSum (G.rowAt t r).1 z && xorSum (G.rowAt t r).2 z) :=
    fun r hr => hz r _ (rows_get hr)
  -- every wire's column holds its value
  have hw : ∀ w, w < t.inBits + t.items.size → z (G.col t.inBits w) = V w := by
    intro w
    induction w using Nat.strong_induction_on with
    | _ w ih =>
      intro hwlt
      by_cases hwi : w < t.inBits
      · rw [hV, run_lt _ _ _ _ hwi, hx]
        simp [Geo.col, hwi]
      · obtain ⟨k, rfl⟩ : ∃ k, w = t.inBits + k := ⟨w - t.inBits, by omega⟩
        have hk : k < t.items.size := by omega
        obtain ⟨a, b, hab, ha, hb⟩ := flatOk_items hflat k hk
        have hcol : G.col t.inBits (t.inBits + k) = G.andBase + k := by simp [Geo.col]
        rw [hcol, hsat _ (by omega), rowAt_and (by omega) hc (by rw [Array.getElem?_eq_getElem hk, hab])]
        simp only
        rw [xorSum_formCols _ _ _ _ h1, xorSum_formCols _ _ _ _ h1,
          formVal_congr (v' := V) (fun y hy => ih y (by have := ha y hy; omega) (by have := ha y hy; omega)),
          formVal_congr (v' := V) (fun y hy => ih y (by have := hb y hy; omega) (by have := hb y hy; omega))]
        have hi := run_item t.items.toList x t.inBits (flatOk_reads hflat) k (by simpa using hk)
        rw [← hV] at hi
        rw [hi]
        simp [hab, itemVal]
  -- the output's copy row
  obtain ⟨hosz, hof⟩ := flatOk_outputs hflat
  obtain ⟨c, hcO, hc1, hc2, hbc⟩ := hout o (by omega)
  have hrow := hsat c (by omega)
  rw [rowAt_out_some hc1 hc2 hbc] at hrow
  rw [hcO, Option.getD_some, hrow, xorSum_formCols _ _ _ _ h1, Array.getElem?_eq_getElem ho, Option.getD_some]
  simp only [xorSum, List.map_cons, List.map_nil, List.foldr_cons, List.foldr_nil, h1, Bool.xor_false, Bool.and_true]
  rw [formVal_congr (v' := V) (fun y hy => hw y (hof o ho y hy)), eval_getD t _ ho, hV, hx]

/-! ## Completeness -/

/-- **The honest evaluation satisfies the derived rows.** On every input `x`, the honest column values satisfy the rows
`derive` lays the flat type `t` out as, with the constant at 1, and carry each input bit and each output on its column. -/
theorem compose_complete {t : CircuitType} {inSplit outSplit : List ℕ} {d : Derived}
    (hd : derive t inSplit outSplit = .ok d) (x : ℕ → Bool) :
    Sat d.rows (honest t d.geo x) ∧ honest t d.geo x d.geo.const = true ∧
      (∀ i < t.inBits, honest t d.geo x (d.inCol i) = x i) ∧
      ∀ o < t.outputs.size, honest t d.geo x (d.outCol o) = (eval t x).getD o false := by
  obtain ⟨hflat, hgeo, rfl⟩ := derive_ok hd
  obtain ⟨hc, hin, hin', hout, hout'⟩ := geoOk_facts hgeo
  have hn := geo_nAnd t inSplit outSplit
  simp only [Derived.inCol, Derived.outCol]
  generalize hG : geo t inSplit outSplit = G at *
  obtain ⟨V, hV⟩ : ∃ V, V = run x t.inBits t.items.toList := ⟨_, rfl⟩
  have hconst : honest t G x G.const = true := by simp [honest]
  have hinput : ∀ i < t.inBits, ∀ c, colOf G.ins i = some c → c < G.andBase → bitAt G.ins c = some i →
      honest t G x c = x i := by
    intro i hi c _ hclt hbc
    unfold honest
    rw [ite_eq_right (show ¬ c = G.const by omega), ite_eq_left hclt, hbc]
  -- the honest value on each wire's column is the wire's value
  have hw : ∀ w, w < t.inBits + t.items.size → honest t G x (G.col t.inBits w) = V w := by
    intro w hwlt
    by_cases hwi : w < t.inBits
    · obtain ⟨c, hcol, hclt, hbc⟩ := hin w hwi
      have : G.col t.inBits w = c := by simp [Geo.col, hwi, hcol]
      rw [this, hinput w hwi c hcol hclt hbc, hV, run_lt _ _ _ _ hwi]
    · obtain ⟨k, rfl⟩ : ∃ k, w = t.inBits + k := ⟨w - t.inBits, by omega⟩
      have : G.col t.inBits (t.inBits + k) = G.andBase + k := by simp [Geo.col]
      rw [this]
      unfold honest
      rw [ite_eq_right (show ¬ G.andBase + k = G.const by omega), ite_eq_right (show ¬ G.andBase + k < G.andBase by omega),
        ite_eq_left (show G.andBase + k < G.andBase + G.nAnd by omega), Nat.add_sub_cancel_left, hV]
  have hform : ∀ f : Form, (∀ y ∈ f.ws.toList, y < t.inBits + t.items.size) →
      xorSum (G.formCols t.inBits f) (honest t G x) = formVal V f := fun f hf => by
    rw [xorSum_formCols _ _ _ _ hconst]
    exact formVal_congr fun y hy => hw y (hf y hy)
  refine ⟨fun r p hp => ?_, hconst, fun i hi => ?_, fun o ho => ?_⟩
  · -- each row, by kind
    rw [List.getElem?_map] at hp
    have hr : r < G.const + 1 := by
      by_contra hr
      rw [List.getElem?_eq_none (by simp only [List.length_range]; omega)] at hp
      cases hp
    rw [List.getElem?_range hr] at hp
    simp only [Option.map_some, Option.some.injEq] at hp
    subst hp
    by_cases h0 : r = G.const
    · subst h0; rw [rowAt_const]; simp [xorSum, hconst]
    by_cases h2 : r < G.andBase
    · cases hb : bitAt G.ins r with
      | none =>
        rw [rowAt_in_none h2 hc hb]
        unfold honest; rw [ite_eq_right h0, ite_eq_left h2, hb]; simp [xorSum]
      | some i =>
        rw [rowAt_in_some h2 hc hb]
        simp [xorSum]
    by_cases h3 : r < G.andBase + G.nAnd
    · obtain ⟨k, rfl⟩ : ∃ k, r = G.andBase + k := ⟨r - G.andBase, by omega⟩
      have hk : k < t.items.size := by omega
      obtain ⟨a, b, hab, ha, hb⟩ := flatOk_items hflat k hk
      rw [rowAt_and (by omega) hc (by rw [Array.getElem?_eq_getElem hk, hab])]
      simp only
      rw [hform a (fun y hy => by have := ha y hy; omega), hform b (fun y hy => by have := hb y hy; omega)]
      have hi := run_item t.items.toList x t.inBits (flatOk_reads hflat) k (by simpa using hk)
      rw [← hV] at hi
      unfold honest
      rw [ite_eq_right h0, ite_eq_right h2, ite_eq_left h3, Nat.add_sub_cancel_left, ← hV, hi]
      simp [hab, itemVal]
    · have hr2 : r < G.const := by omega
      have hr1 : G.andBase + G.nAnd ≤ r := by omega
      cases hb : bitAt G.outs r with
      | none =>
        rw [rowAt_out_none hr1 hr2 hb]
        unfold honest; rw [ite_eq_right h0, ite_eq_right h2, ite_eq_right h3, hb]; simp [xorSum]
      | some o =>
        have ho := (hout' r hr1 hr2 o hb).1
        obtain ⟨hosz, hof⟩ := flatOk_outputs hflat
        have ho' : o < t.outputs.size := by omega
        rw [rowAt_out_some hr1 hr2 hb, Array.getElem?_eq_getElem ho', Option.getD_some]
        simp only
        rw [hform _ (hof o ho')]
        unfold honest
        rw [ite_eq_right h0, ite_eq_right h2, ite_eq_right h3, hb]
        simp only [Array.getElem?_eq_getElem ho', Option.getD_some, ← hV]
        simp [xorSum]
  · obtain ⟨c, hcol, hclt, hbc⟩ := hin i hi
    rw [hcol, Option.getD_some, hinput i hi c hcol hclt hbc]
  · obtain ⟨hosz, hof⟩ := flatOk_outputs hflat
    obtain ⟨c, hcol, hc1, hc2, hbc⟩ := hout o (by omega)
    rw [hcol, Option.getD_some, eval_getD t x ho]
    unfold honest
    rw [ite_eq_right (show ¬ c = G.const by omega), ite_eq_right (show ¬ c < G.andBase by omega),
      ite_eq_right (show ¬ c < G.andBase + G.nAnd by omega), hbc]
    simp only [Array.getElem?_eq_getElem ho, Option.getD_some]

/-! ## The logical order -/

theorem nodup_map_of_inj {l : List ℕ} {f : ℕ → ℕ} (hl : l.Nodup) (h : ∀ i ∈ l, ∀ j ∈ l, f i = f j → i = j) :
    (l.map f).Nodup := List.Nodup.map_on h hl

theorem disjoint_of_lt {l m : List ℕ} {B : ℕ} (hl : ∀ x ∈ l, x < B) (hm : ∀ y ∈ m, B ≤ y) : l.Disjoint m :=
  fun x hx hx' => by have := hl x hx; have := hm x hx'; omega

/-- **`order` has no repeats**: the input bits, the constant, the AND rows and the output copy rows sit at distinct
columns. -/
theorem order_nodup {t : CircuitType} {inSplit outSplit : List ℕ} {d : Derived}
    (hd : derive t inSplit outSplit = .ok d) : d.order.Nodup := by
  obtain ⟨hflat, hgeo, rfl⟩ := derive_ok hd
  obtain ⟨hc, hin, -, hout, -⟩ := geoOk_facts hgeo
  simp only [Derived.order]
  generalize hG : geo t inSplit outSplit = G at *
  have hic : ∀ i < t.inBits, (colOf G.ins i).getD 0 < G.andBase ∧ bitAt G.ins ((colOf G.ins i).getD 0) = some i :=
    fun i hi => by obtain ⟨c, h1, h2, h3⟩ := hin i hi; simp [h1, h2, h3]
  have hoc : ∀ o < t.outBits, G.andBase + G.nAnd ≤ (colOf G.outs o).getD 0 ∧ (colOf G.outs o).getD 0 < G.const ∧
      bitAt G.outs ((colOf G.outs o).getD 0) = some o :=
    fun o ho => by obtain ⟨c, h1, h2, h3, h4⟩ := hout o ho; simp [h1, h2, h3, h4]
  have n1 : ((List.range t.inBits).map fun i => (colOf G.ins i).getD 0).Nodup :=
    nodup_map_of_inj List.nodup_range fun i hi j hj h => by
      simp only [List.mem_range] at hi hj
      have := (hic i hi).2; rw [h, (hic j hj).2] at this; exact (Option.some.inj this).symm
  have n2 : ((List.range G.nAnd).map (G.andBase + ·)).Nodup :=
    nodup_map_of_inj List.nodup_range fun i _ j _ h => by simpa using h
  have n3 : ((List.range t.outBits).map fun o => (colOf G.outs o).getD 0).Nodup :=
    nodup_map_of_inj List.nodup_range fun i hi j hj h => by
      simp only [List.mem_range] at hi hj
      have := (hoc i hi).2.2; rw [h, (hoc j hj).2.2] at this; exact (Option.some.inj this).symm
  have b1 : ∀ x ∈ (List.range t.inBits).map (fun i => (colOf G.ins i).getD 0), x < G.andBase := by
    intro x hx; obtain ⟨i, hi, rfl⟩ := List.mem_map.mp hx; exact (hic i (List.mem_range.mp hi)).1
  have b2 : ∀ x ∈ (List.range G.nAnd).map (G.andBase + ·), G.andBase ≤ x ∧ x < G.andBase + G.nAnd := by
    intro x hx; obtain ⟨k, hk, rfl⟩ := List.mem_map.mp hx; have := List.mem_range.mp hk; omega
  have b3 : ∀ x ∈ (List.range t.outBits).map (fun o => (colOf G.outs o).getD 0),
      G.andBase + G.nAnd ≤ x ∧ x < G.const := by
    intro x hx; obtain ⟨o, ho, rfl⟩ := List.mem_map.mp hx
    exact ⟨(hoc o (List.mem_range.mp ho)).1, (hoc o (List.mem_range.mp ho)).2.1⟩
  have n12 : (((List.range t.inBits).map fun i => (colOf G.ins i).getD 0) ++ [G.const]).Nodup :=
    List.nodup_append.mpr ⟨n1, List.nodup_singleton _, fun x hx y hy hxy => by
      rw [List.mem_singleton] at hy; have := b1 x hx; omega⟩
  have n123 : ((((List.range t.inBits).map fun i => (colOf G.ins i).getD 0) ++ [G.const]) ++
      (List.range G.nAnd).map (G.andBase + ·)).Nodup :=
    List.nodup_append.mpr ⟨n12, n2, fun x hx y hy hxy => by
      have := b2 y hy
      rcases List.mem_append.mp hx with hx | hx
      · have := b1 x hx; omega
      · rw [List.mem_singleton] at hx; omega⟩
  exact List.nodup_append.mpr ⟨n123, n3, fun x hx y hy hxy => by
    have := b3 y hy
    rcases List.mem_append.mp hx with hx | hx
    · rcases List.mem_append.mp hx with hx | hx
      · have := b1 x hx; omega
      · rw [List.mem_singleton] at hx; omega
    · have := b2 x hx; omega⟩

/-- **Every row that isn't forced zero sits at a column in `order`.** -/
theorem order_covers {t : CircuitType} {inSplit outSplit : List ℕ} {d : Derived}
    (hd : derive t inSplit outSplit = .ok d) (r : ℕ) (p : List ℕ × List ℕ) (hp : d.rows[r]? = some p)
    (hne : p ≠ ([], [])) : r ∈ d.order := by
  obtain ⟨hflat, hgeo, rfl⟩ := derive_ok hd
  obtain ⟨hc, hin, hin', hout, hout'⟩ := geoOk_facts hgeo
  simp only [Derived.order] at hp ⊢
  generalize hG : geo t inSplit outSplit = G at *
  rw [List.getElem?_map] at hp
  have hr : r < G.const + 1 := by
    by_contra hr
    rw [List.getElem?_eq_none (by simp only [List.length_range]; omega)] at hp
    cases hp
  rw [List.getElem?_range hr] at hp
  simp only [Option.map_some, Option.some.injEq] at hp
  subst hp
  simp only [List.mem_append, List.mem_map, List.mem_range, List.mem_singleton]
  by_cases h0 : r = G.const
  · exact Or.inl (Or.inl (Or.inr h0))
  by_cases h2 : r < G.andBase
  · cases hb : bitAt G.ins r with
    | none => exact absurd (rowAt_in_none h2 hc hb) hne
    | some i =>
      obtain ⟨hi, hci⟩ := hin' r h2 i hb
      exact Or.inl (Or.inl (Or.inl ⟨i, hi, by simp [Derived.inCol, hci]⟩))
  by_cases h3 : r < G.andBase + G.nAnd
  · exact Or.inl (Or.inr ⟨r - G.andBase, by omega, by omega⟩)
  · cases hb : bitAt G.outs r with
    | none => exact absurd (rowAt_out_none (by omega) (by omega) hb) hne
    | some o =>
      obtain ⟨ho, hco⟩ := hout' r (by omega) (by omega) o hb
      exact Or.inr ⟨o, ho, by simp [Derived.outCol, hco]⟩

end FlockSoundness.Types
