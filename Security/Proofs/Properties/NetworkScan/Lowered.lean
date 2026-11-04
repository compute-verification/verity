import Properties.NetworkScan.Flattens

/-!
# The lowered program's facts

What the scan's and the wired rows' proofs read about Bits' key program lowered to the Boolean profile
(`SampledProofs.Bits.prog`) and an audit's members side by side (`Properties.NetworkScan.jointOf`):

* positions: gate `g`'s lowered gates sit from `off` (`getElem?_prog`); `posOf` is strictly increasing on the program's
  gates (`posOf_lt_posOf`), so injective (`posOf_inj`); an input gate sits at itself (`posOf_input`), a row at its AND
  (`posOf_row`); member `m`'s positions start at `base ms m` (`getElem?_jointOf`), and two members' never meet
  (`joint_unique`);
* lengths: the program's is its gates' lowered sizes' sum (`length_prog`), the members' the programs' sum
  (`length_jointOf`, `base_succ`);
* inputs: a lowered gate is `.input` exactly at an input gate the constants don't fix (`prog_input_iff`);
* outputs: the key program returns every instance's output copy bits (`mem_outs_keyProg`), each a row
  (`row?_outGate`) whose position holds its AND (`prog_and`), and the members return theirs from `base`
  (`mem_outs_jointOf_iff`);
* reads: a key program's row reads the constants and its own unit's gates (`row_unit`, through `wireOf`), so a lowered
  gate reads the constants and its own unit's positions (`reads_unit`), and in the members' program a gate that depends
  on a unit's position is that unit's (`depends_unit`).
-/

open SampledProofs.Bits

namespace Properties.NetworkScan.Lowered

/-! ## Sums over a range -/

theorem range_add_one (n : ℕ) : List.range (n + 1) = List.range n ++ [n] := List.range_succ

theorem sum_range_add_one (f : ℕ → ℕ) (n : ℕ) :
    ((List.range (n + 1)).map f).sum = ((List.range n).map f).sum + f n := by
  rw [range_add_one, List.map_append, List.sum_append]
  simp

theorem sum_range_mono (f : ℕ → ℕ) {m n : ℕ} (h : m ≤ n) :
    ((List.range m).map f).sum ≤ ((List.range n).map f).sum := by
  induction n with
  | zero => obtain rfl := Nat.le_zero.1 h; exact Nat.le_refl _
  | succ n ih =>
    rcases Nat.lt_or_ge m (n + 1) with hm | hm
    · rw [sum_range_add_one]; exact Nat.le_trans (ih (by omega)) (Nat.le_add_right _ _)
    · obtain rfl := Nat.le_antisymm h hm; exact Nat.le_refl _

theorem sum_range_const (n a : ℕ) : ((List.range n).map fun _ => a).sum = n * a := by
  induction n with
  | zero => simp
  | succ n ih => rw [sum_range_add_one, ih, Nat.succ_mul]

/-- A point below a range's sum sits in one term's segment. -/
theorem exists_seg (f : ℕ → ℕ) {n x : ℕ} (hx : x < ((List.range n).map f).sum) :
    ∃ g < n, ∃ j < f g, x = ((List.range g).map f).sum + j := by
  induction n with
  | zero => simp at hx
  | succ n ih =>
    rw [sum_range_add_one] at hx
    by_cases h : x < ((List.range n).map f).sum
    · obtain ⟨g, hg, j, hj, rfl⟩ := ih h
      exact ⟨g, by omega, j, hj, rfl⟩
    · exact ⟨n, by omega, x - ((List.range n).map f).sum, by omega, by omega⟩

/-- Entry `j` of term `g` of a flattened range sits after the terms before it. -/
theorem getElem?_flatMap_range {α : Type} (f : ℕ → List α) {n g j : ℕ} (hg : g < n) (hj : j < (f g).length) :
    ((List.range n).flatMap f)[((List.range g).map fun i => (f i).length).sum + j]? = (f g)[j]? := by
  induction n with
  | zero => omega
  | succ n ih =>
    rw [range_add_one, List.flatMap_append, List.flatMap_singleton]
    rcases Nat.lt_or_ge g n with h | h
    · have hle := sum_range_mono (fun i => (f i).length) (show g + 1 ≤ n from h)
      rw [sum_range_add_one] at hle
      beta_reduce at hle
      have hlt : ((List.range g).map fun i => (f i).length).sum + j < ((List.range n).flatMap f).length := by
        rw [List.length_flatMap]; omega
      rw [List.getElem?_append_left hlt, ih h]
    · obtain rfl : g = n := by omega
      have hge : ((List.range g).flatMap f).length ≤ ((List.range g).map fun i => (f i).length).sum + j := by
        rw [List.length_flatMap]; omega
      rw [List.getElem?_append_right hge, List.length_flatMap, Nat.add_sub_cancel_left]

theorem mul_add_le {B u n i : ℕ} (hu : u < n) (hi : i ≤ B) : u * B + i ≤ n * B := by
  have := Nat.mul_le_mul_right B (show u + 1 ≤ n from hu)
  rw [Nat.add_mul, Nat.one_mul] at this
  omega

theorem mul_add_unique {B u u' x : ℕ} (h1 : u * B ≤ x) (h2 : x < u * B + B) (h1' : u' * B ≤ x)
    (h2' : x < u' * B + B) : u = u' := by
  rcases Nat.lt_trichotomy u u' with h | h | h
  · have := mul_add_le (B := B) (i := B) h (Nat.le_refl _); omega
  · exact h
  · have := mul_add_le (B := B) (i := B) h (Nat.le_refl _); omega

/-! ## A key program's positions -/

section Positions

variable (kp : KeyProg) (κ : ℕ → Option Bool)

theorem length_lower (g : ℕ) : (lower kp κ g).length = gateSize kp g := by
  unfold lower gateSize
  cases kp.row? g with
  | none => cases κ g <;> rfl
  | some ab => obtain ⟨a, b⟩ := ab; rfl

theorem gateSize_pos (g : ℕ) : 0 < gateSize kp g := by
  unfold gateSize; split <;> omega

theorem range_filter_lt (n g : ℕ) :
    (List.range n).filter (fun x => decide (x < g)) = List.range (min n g) := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [range_add_one, List.filter_append, ih]
    by_cases h : n < g
    · rw [Nat.min_eq_left (Nat.le_of_lt h), Nat.min_eq_left (show n + 1 ≤ g from h), range_add_one]
      simp [List.filter_cons, h]
    · rw [Nat.min_eq_right (show g ≤ n by omega), Nat.min_eq_right (show g ≤ n + 1 by omega)]
      simp [List.filter_cons, h]

theorem off_eq_min (g : ℕ) : off kp g = ((List.range (min kp.size g)).map (gateSize kp)).sum := by
  unfold off; rw [range_filter_lt]

theorem off_eq {g : ℕ} (hg : g ≤ kp.size) : off kp g = ((List.range g).map (gateSize kp)).sum := by
  rw [off_eq_min, Nat.min_eq_right hg]

theorem off_mono {g h : ℕ} (hgh : g ≤ h) : off kp g ≤ off kp h := by
  rw [off_eq_min, off_eq_min]
  exact sum_range_mono _ (by omega)

theorem off_succ {g : ℕ} (hg : g < kp.size) : off kp (g + 1) = off kp g + gateSize kp g := by
  rw [off_eq kp (g := g + 1) hg, off_eq kp (Nat.le_of_lt hg), sum_range_add_one]

theorem posOf_lt_off_succ {g : ℕ} (hg : g < kp.size) : posOf kp g < off kp (g + 1) := by
  have := gateSize_pos kp g
  have := off_succ kp hg
  unfold posOf; omega

theorem off_le_posOf (g : ℕ) : off kp g ≤ posOf kp g := by
  have := gateSize_pos kp g
  unfold posOf; omega

/-- **Positions increase with gates** on the program's gates. -/
theorem posOf_lt_posOf {g h : ℕ} (hg : g < kp.size) (hgh : g < h) : posOf kp g < posOf kp h :=
  Nat.lt_of_lt_of_le (posOf_lt_off_succ kp hg)
    (Nat.le_trans (off_mono kp (show g + 1 ≤ h from hgh)) (off_le_posOf kp h))

/-- **Positions are injective** on the program's gates. -/
theorem posOf_inj {g h : ℕ} (hg : g < kp.size) (hh : h < kp.size) (he : posOf kp g = posOf kp h) : g = h := by
  rcases Nat.lt_trichotomy g h with hlt | heq | hgt
  · exact absurd he (Nat.ne_of_lt (posOf_lt_posOf kp hg hlt))
  · exact heq
  · exact absurd he.symm (Nat.ne_of_lt (posOf_lt_posOf kp hh hgt))

/-- Two gates' lowered gates never meet. -/
theorem seg_unique {g g' y : ℕ} (hg : g < kp.size) (hg' : g' < kp.size) (h1 : off kp g ≤ y)
    (h2 : y < off kp g + gateSize kp g) (h1' : off kp g' ≤ y) (h2' : y < off kp g' + gateSize kp g') : g = g' := by
  rcases Nat.lt_trichotomy g g' with h | h | h
  · have := off_mono kp (show g + 1 ≤ g' from h); rw [off_succ kp hg] at this; omega
  · exact h
  · have := off_mono kp (show g' + 1 ≤ g from h); rw [off_succ kp hg'] at this; omega

/-- **The lowered program's length** is its gates' lowered sizes' sum. -/
theorem length_prog : (prog kp κ).1.length = off kp kp.size := by
  rw [off_eq kp (Nat.le_refl _)]
  show ((List.range kp.size).flatMap (lower kp κ)).length = _
  rw [List.length_flatMap]
  simp only [length_lower]

/-- **Gate `g`'s lowered gates sit from `off kp g`.** -/
theorem getElem?_prog {g j : ℕ} (hg : g < kp.size) (hj : j < gateSize kp g) :
    (prog kp κ).1[off kp g + j]? = (lower kp κ g)[j]? := by
  have h := getElem?_flatMap_range (lower kp κ) hg (j := j) (by rw [length_lower]; exact hj)
  simp only [length_lower] at h
  rw [off_eq kp (Nat.le_of_lt hg)]
  exact h

theorem exists_off {x : ℕ} (hx : x < (prog kp κ).1.length) :
    ∃ g < kp.size, ∃ j < gateSize kp g, x = off kp g + j := by
  rw [length_prog, off_eq kp (Nat.le_refl _)] at hx
  obtain ⟨g, hg, j, hj, rfl⟩ := exists_seg (gateSize kp) hx
  exact ⟨g, hg, j, hj, by rw [off_eq kp (Nat.le_of_lt hg)]⟩

theorem lt_length_of_getElem? {x : ℕ} {G : LGate} (h : (prog kp κ).1[x]? = some G) :
    x < (prog kp κ).1.length := by
  by_contra hc
  have hle : (prog kp κ).1.length ≤ x := by omega
  rw [List.getElem?_eq_none hle] at h
  cases h

theorem row?_input {g : ℕ} (hg : g < kp.inputs) : kp.row? g = none := by
  simp [KeyProg.row?, hg]

theorem row?_some {g : ℕ} (h1 : kp.inputs ≤ g) (h2 : g < kp.size) : ∃ ab, kp.row? g = some ab := by
  unfold KeyProg.size at h2
  have hr : g - kp.inputs < kp.rows.size := by omega
  exact ⟨kp.rows[g - kp.inputs], by rw [KeyProg.row?, if_neg (Nat.not_lt.2 h1), Array.getElem?_eq_getElem hr]⟩

theorem lt_size_of_row? {g : ℕ} {ab : List ℕ × List ℕ} (h : kp.row? g = some ab) : g < kp.size := by
  unfold KeyProg.row? at h
  split at h
  · cases h
  · unfold KeyProg.size
    by_contra hc
    have hle : kp.rows.size ≤ g - kp.inputs := by omega
    rw [Array.getElem?_eq_none hle] at h
    cases h

theorem inputs_le_of_row? {g : ℕ} {ab : List ℕ × List ℕ} (h : kp.row? g = some ab) : kp.inputs ≤ g := by
  unfold KeyProg.row? at h
  split at h
  · cases h
  · omega

theorem gateSize_input {g : ℕ} (hg : g < kp.inputs) : gateSize kp g = 1 := by
  simp [gateSize, row?_input kp hg]

theorem gateSize_row {g : ℕ} {ab : List ℕ × List ℕ} (h : kp.row? g = some ab) : gateSize kp g = 3 := by
  simp [gateSize, h]

theorem off_input {g : ℕ} (hg : g ≤ kp.inputs) : off kp g = g := by
  rw [off_eq kp (g := g) (by unfold KeyProg.size; omega)]
  induction g with
  | zero => simp
  | succ g ih => rw [sum_range_add_one, ih (by omega), gateSize_input kp (show g < kp.inputs by omega)]

/-- **An input gate sits at itself.** -/
theorem posOf_input {g : ℕ} (hg : g < kp.inputs) : posOf kp g = g := by
  have h1 := gateSize_input kp hg
  have h2 := off_input kp (Nat.le_of_lt hg)
  unfold posOf; omega

/-- **A row sits at its AND**, its third lowered gate. -/
theorem posOf_row {g : ℕ} {ab : List ℕ × List ℕ} (h : kp.row? g = some ab) : posOf kp g = off kp g + 2 := by
  have h3 := gateSize_row kp h
  unfold posOf; omega

/-- **A row's position holds the AND** of its two XORs. -/
theorem prog_and {g : ℕ} {a b : List ℕ} (h : kp.row? g = some (a, b)) :
    (prog kp κ).1[posOf kp g]? = some (.and [off kp g, off kp g + 1]) := by
  have h3 := gateSize_row kp h
  rw [posOf_row kp h, getElem?_prog kp κ (j := 2) (lt_size_of_row? kp h) (by omega)]
  simp [lower, h]

/-- **The input gates**: a lowered gate is `.input` exactly at an input gate the constants `κ` don't fix. -/
theorem prog_input_iff (x : ℕ) : (prog kp κ).1[x]? = some .input ↔ x < kp.inputs ∧ κ x = none := by
  constructor
  · intro h
    obtain ⟨g, hg, j, hj, rfl⟩ := exists_off kp κ (lt_length_of_getElem? kp κ h)
    rw [getElem?_prog kp κ hg hj] at h
    by_cases hi : g < kp.inputs
    · have h1 := gateSize_input kp hi
      obtain rfl : j = 0 := by omega
      rw [Nat.add_zero, off_input kp (Nat.le_of_lt hi)]
      rcases hκ : κ g with _ | b
      · exact ⟨hi, by simp [hκ]⟩
      · simp [lower, row?_input kp hi, hκ] at h
    · obtain ⟨⟨a, b⟩, hr⟩ := row?_some kp (Nat.not_lt.1 hi) hg
      obtain _ | _ | _ | j := j <;> simp [lower, hr] at h
  · rintro ⟨hi, hκ⟩
    have h1 := gateSize_input kp hi
    have hs : x < kp.size := by unfold KeyProg.size; omega
    have := getElem?_prog kp κ hs (j := 0) (by omega)
    rw [Nat.add_zero, off_input kp (Nat.le_of_lt hi)] at this
    rw [this]
    simp [lower, row?_input kp hi, hκ]

end Positions

/-! ## The key program -/

/-- The stacked rows' inputs: a unit's source gates. -/
def srcs (c : Flock.Circuit) : ℕ := ((Rows.ofNet c.unit).stack c.upv).nIn

theorem keyProg_inputs (c : Flock.Circuit) (n : ℕ) : (keyProg c n).inputs = 2 + n * srcs c := rfl

theorem keyProg_perUnit (c : Flock.Circuit) (n : ℕ) :
    (keyProg c n).perUnit = ((Rows.ofNet c.unit).stack c.upv).nComp := rfl

theorem keyProg_size_rows (c : Flock.Circuit) (n : ℕ) : (keyProg c n).rows.size = n * (keyProg c n).perUnit := by
  simp only [keyProg, List.size_toArray, List.length_flatMap, List.length_map, List.length_range]
  exact sum_range_const n _

theorem two_le_length (c : Flock.Circuit) (n : ℕ) (κ : ℕ → Option Bool) : 2 ≤ (prog (keyProg c n) κ).1.length := by
  have h2 : 2 ≤ (keyProg c n).inputs := by rw [keyProg_inputs]; omega
  have hs : 2 ≤ (keyProg c n).size := by unfold KeyProg.size; omega
  have := off_mono (keyProg c n) hs
  rw [off_input _ h2] at this
  rw [length_prog]; exact this

/-- `keyProg`'s wiring of unit `u`'s stacked column `col`, of `n` units: an input column on its key's source gate (the
zero's where it has none), a computed column on the unit's own row, the constant on gate 0. -/
def wireOf (c : Flock.Circuit) (n u col : ℕ) : ℕ :=
  if col < ((Rows.ofNet c.unit).stack c.upv).nIn then
    (match ((Array.range ((Rows.ofNet c.unit).stack c.upv).nIn).map
        (keyIdx c ((Rows.ofNet c.unit).stack c.upv).nIn)).getD col none with
      | some b => 2 + u * ((Rows.ofNet c.unit).stack c.upv).nIn + b
      | none => 1)
  else if col < ((Rows.ofNet c.unit).stack c.upv).nIn + ((Rows.ofNet c.unit).stack c.upv).nComp then
    2 + n * ((Rows.ofNet c.unit).stack c.upv).nIn + u * ((Rows.ofNet c.unit).stack c.upv).nComp +
      (col - ((Rows.ofNet c.unit).stack c.upv).nIn)
  else 0

theorem keyProg_rows (c : Flock.Circuit) (n : ℕ) :
    (keyProg c n).rows = ((List.range n).flatMap fun u =>
      (List.range ((Rows.ofNet c.unit).stack c.upv).nComp).map fun i =>
        ((((Rows.ofNet c.unit).stack c.upv).a.getD i []).map (wireOf c n u),
          (((Rows.ofNet c.unit).stack c.upv).b.getD i []).map (wireOf c n u))).toArray := rfl

/-- **Unit `u`'s row `i`** is the stacked rows' row `i` on `wireOf`. -/
theorem keyProg_row? (c : Flock.Circuit) {n u i : ℕ} (hu : u < n) (hi : i < (keyProg c n).perUnit) :
    (keyProg c n).row? ((keyProg c n).inputs + (u * (keyProg c n).perUnit + i)) =
      some ((((Rows.ofNet c.unit).stack c.upv).a.getD i []).map (wireOf c n u),
        (((Rows.ofNet c.unit).stack c.upv).b.getD i []).map (wireOf c n u)) := by
  have hc : ¬ ((keyProg c n).inputs + (u * (keyProg c n).perUnit + i) < (keyProg c n).inputs) := by omega
  have hi' : i < ((Rows.ofNet c.unit).stack c.upv).nComp := hi
  rw [KeyProg.row?, if_neg hc, Nat.add_sub_cancel_left, keyProg_rows, List.getElem?_toArray, keyProg_perUnit]
  have h := getElem?_flatMap_range (fun u => (List.range ((Rows.ofNet c.unit).stack c.upv).nComp).map fun i =>
      ((((Rows.ofNet c.unit).stack c.upv).a.getD i []).map (wireOf c n u),
        (((Rows.ofNet c.unit).stack c.upv).b.getD i []).map (wireOf c n u))) hu (j := i)
    (by simp only [List.length_map, List.length_range]; exact hi')
  simp only [List.length_map, List.length_range, sum_range_const] at h
  rw [h, List.getElem?_map, List.getElem?_range hi']
  rfl

theorem keyIdx_lt {c : Flock.Circuit} {n x b : ℕ} (h : keyIdx c n x = some b) : b < n := by
  unfold keyIdx at h
  dsimp only at h
  split at h
  · cases h
  · exact List.mem_range.1 (List.mem_of_find?_eq_some h)

theorem keys_getD {c : Flock.Circuit} {B col b : ℕ}
    (h : ((Array.range B).map (keyIdx c B)).getD col none = some b) : b < B := by
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_map, Array.getElem?_range] at h
  split at h
  · exact keyIdx_lt (c := c) (x := col) (by simpa using h)
  · simp at h

/-- **A unit's column is the constants or the unit's own gate**: a source gate or a row. -/
theorem wireOf_cases (c : Flock.Circuit) (n u col : ℕ) :
    wireOf c n u col ≤ 1 ∨ (∃ b < srcs c, wireOf c n u col = 2 + u * srcs c + b) ∨
      (∃ i < (keyProg c n).perUnit, wireOf c n u col = (keyProg c n).inputs + (u * (keyProg c n).perUnit + i)) := by
  have hI := keyProg_inputs c n
  have hP := keyProg_perUnit c n
  by_cases h1 : col < ((Rows.ofNet c.unit).stack c.upv).nIn
  · rcases hk : ((Array.range ((Rows.ofNet c.unit).stack c.upv).nIn).map
        (keyIdx c ((Rows.ofNet c.unit).stack c.upv).nIn)).getD col none with _ | b
    · have hw : wireOf c n u col = 1 := by simp only [wireOf, h1, ↓reduceIte, hk]
      exact .inl (by omega)
    · have hw : wireOf c n u col = 2 + u * srcs c + b := by simp only [wireOf, h1, ↓reduceIte, hk, srcs]
      exact .inr (.inl ⟨b, keys_getD hk, hw⟩)
  · by_cases h2 : col < ((Rows.ofNet c.unit).stack c.upv).nIn + ((Rows.ofNet c.unit).stack c.upv).nComp
    · have hw : wireOf c n u col = 2 + n * ((Rows.ofNet c.unit).stack c.upv).nIn +
          u * ((Rows.ofNet c.unit).stack c.upv).nComp + (col - ((Rows.ofNet c.unit).stack c.upv).nIn) := by
        simp only [wireOf, h1, h2, ↓reduceIte]
      refine .inr (.inr ⟨col - ((Rows.ofNet c.unit).stack c.upv).nIn, by omega, ?_⟩)
      rw [hw, hI, hP]; unfold srcs; omega
    · have hw : wireOf c n u col = 0 := by simp only [wireOf, h1, h2, ↓reduceIte]
      exact .inl (by omega)

/-- **Gate `g` of `keyProg c n` is unit `u`'s**: one of its source gates or one of its rows. -/
def UnitGate (c : Flock.Circuit) (n u g : ℕ) : Prop :=
  u < n ∧ ((2 + u * srcs c ≤ g ∧ g < 2 + u * srcs c + srcs c) ∨
    ((keyProg c n).inputs + u * (keyProg c n).perUnit ≤ g ∧
      g < (keyProg c n).inputs + u * (keyProg c n).perUnit + (keyProg c n).perUnit))

theorem unitGate_lt_size {c : Flock.Circuit} {n u g : ℕ} (h : UnitGate c n u g) : g < (keyProg c n).size := by
  obtain ⟨hu, h⟩ := h
  have h1 := mul_add_le (B := srcs c) (i := srcs c) hu (Nat.le_refl _)
  have h2 := mul_add_le (B := (keyProg c n).perUnit) (i := (keyProg c n).perUnit) hu (Nat.le_refl _)
  have := keyProg_size_rows c n
  have := keyProg_inputs c n
  unfold KeyProg.size; omega

theorem two_le_of_unitGate {c : Flock.Circuit} {n u g : ℕ} (h : UnitGate c n u g) : 2 ≤ g := by
  obtain ⟨-, h⟩ := h
  have := keyProg_inputs c n
  omega

/-- **A gate is at most one unit's.** -/
theorem unitGate_unique {c : Flock.Circuit} {n u u' g : ℕ} (h : UnitGate c n u g) (h' : UnitGate c n u' g) :
    u = u' := by
  obtain ⟨hu, h⟩ := h
  obtain ⟨hu', h'⟩ := h'
  have e1 := mul_add_le (B := srcs c) (i := srcs c) hu (Nat.le_refl _)
  have e2 := mul_add_le (B := srcs c) (i := srcs c) hu' (Nat.le_refl _)
  have := keyProg_inputs c n
  rcases h with ⟨h1, h2⟩ | ⟨h1, h2⟩ <;> rcases h' with ⟨h1', h2'⟩ | ⟨h1', h2'⟩
  · exact mul_add_unique (B := srcs c) (x := g - 2) (by omega) (by omega) (by omega) (by omega)
  · omega
  · omega
  · exact mul_add_unique (B := (keyProg c n).perUnit) (x := g - (keyProg c n).inputs) (by omega) (by omega)
      (by omega) (by omega)

/-- **A row is its unit's, and reads the constants and its unit's gates.** -/
theorem row_unit (c : Flock.Circuit) {n g : ℕ} {a b : List ℕ} (h : (keyProg c n).row? g = some (a, b)) :
    UnitGate c n ((g - (keyProg c n).inputs) / (keyProg c n).perUnit) g ∧
      ∀ x ∈ a ++ b, x ≤ 1 ∨ UnitGate c n ((g - (keyProg c n).inputs) / (keyProg c n).perUnit) x := by
  have h1 := inputs_le_of_row? _ h
  have h2 := lt_size_of_row? _ h
  have hs := keyProg_size_rows c n
  unfold KeyProg.size at h2
  have hr : g - (keyProg c n).inputs < n * (keyProg c n).perUnit := by omega
  have hP : 0 < (keyProg c n).perUnit :=
    Nat.pos_of_ne_zero fun h0 => by rw [h0, Nat.mul_zero] at hr; omega
  have hu : (g - (keyProg c n).inputs) / (keyProg c n).perUnit < n := (Nat.div_lt_iff_lt_mul hP).2 hr
  have hi := Nat.mod_lt (g - (keyProg c n).inputs) hP
  have hdm : (g - (keyProg c n).inputs) / (keyProg c n).perUnit * (keyProg c n).perUnit +
      (g - (keyProg c n).inputs) % (keyProg c n).perUnit = g - (keyProg c n).inputs := by
    rw [Nat.mul_comm]; exact Nat.div_add_mod _ _
  have hrow := keyProg_row? c hu hi
  rw [show (keyProg c n).inputs + ((g - (keyProg c n).inputs) / (keyProg c n).perUnit * (keyProg c n).perUnit +
      (g - (keyProg c n).inputs) % (keyProg c n).perUnit) = g by omega, h] at hrow
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj hrow)
  refine ⟨⟨hu, .inr ⟨by omega, by omega⟩⟩, fun x hx => ?_⟩
  rw [List.mem_append, List.mem_map, List.mem_map] at hx
  obtain ⟨col, -, rfl⟩ | ⟨col, -, rfl⟩ := hx
  all_goals
    rcases wireOf_cases c n ((g - (keyProg c n).inputs) / (keyProg c n).perUnit) col with
      hw | ⟨b, hb, hw⟩ | ⟨i, hi, hw⟩
    · exact .inl hw
    · exact .inr ⟨hu, .inl ⟨by omega, by omega⟩⟩
    · exact .inr ⟨hu, .inr ⟨by omega, by omega⟩⟩

/-! ## Unit positions -/

/-- **Position `y` of `prog (keyProg c n) κ` is unit `u`'s**: a lowered gate of one of its gates. -/
def UnitPos (c : Flock.Circuit) (n u y : ℕ) : Prop :=
  ∃ g, UnitGate c n u g ∧ off (keyProg c n) g ≤ y ∧ y < off (keyProg c n) g + gateSize (keyProg c n) g

theorem unitPos_posOf {c : Flock.Circuit} {n u g : ℕ} (h : UnitGate c n u g) :
    UnitPos c n u (posOf (keyProg c n) g) :=
  ⟨g, h, off_le_posOf _ g, by have := gateSize_pos (keyProg c n) g; unfold posOf; omega⟩

/-- **A position is at most one unit's.** -/
theorem unitPos_unique {c : Flock.Circuit} {n u u' y : ℕ} (h : UnitPos c n u y) (h' : UnitPos c n u' y) :
    u = u' := by
  obtain ⟨g, hg, h1, h2⟩ := h
  obtain ⟨g', hg', h1', h2'⟩ := h'
  obtain rfl := seg_unique _ (unitGate_lt_size hg) (unitGate_lt_size hg') h1 h2 h1' h2'
  exact unitGate_unique hg hg'

/-- A unit's positions are past the constants. -/
theorem two_le_of_unitPos {c : Flock.Circuit} {n u y : ℕ} (h : UnitPos c n u y) : 2 ≤ y := by
  obtain ⟨g, hg, h1, -⟩ := h
  have h2 := two_le_of_unitGate hg
  have h22 : 2 ≤ (keyProg c n).inputs := by rw [keyProg_inputs]; omega
  have := off_mono (keyProg c n) h2
  rw [off_input _ h22] at this
  omega

theorem unitPos_lt_length {c : Flock.Circuit} {n u y : ℕ} (κ : ℕ → Option Bool) (h : UnitPos c n u y) :
    y < (prog (keyProg c n) κ).1.length := by
  obtain ⟨g, hg, -, h2⟩ := h
  have hs := unitGate_lt_size hg
  rw [length_prog]
  have := off_mono (keyProg c n) (show g + 1 ≤ (keyProg c n).size from hs)
  rw [off_succ _ hs] at this
  omega

/-- **A lowered gate reads the constants and its own unit's positions.** -/
theorem reads_unit (c : Flock.Circuit) (n : ℕ) (κ : ℕ → Option Bool) {y z : ℕ} {G : LGate}
    (h : (prog (keyProg c n) κ).1[y]? = some G) (hz : z ∈ G.reads) :
    z ≤ 1 ∨ ∃ u, UnitPos c n u y ∧ UnitPos c n u z := by
  obtain ⟨g, hg, j, hj, rfl⟩ := exists_off _ κ (lt_length_of_getElem? _ κ h)
  rw [getElem?_prog _ κ hg hj] at h
  by_cases hi : g < (keyProg c n).inputs
  · have h1 := gateSize_input _ hi
    obtain rfl : j = 0 := by omega
    have hG : G.reads = [] := by
      rcases hκ : κ g with _ | v <;> simp [lower, row?_input _ hi, hκ] at h <;> subst h <;> rfl
    rw [hG] at hz; cases hz
  · obtain ⟨⟨a, b⟩, hr⟩ := row?_some _ (Nat.not_lt.1 hi) hg
    obtain ⟨hgu, hab⟩ := row_unit c hr
    have h3 := gateSize_row _ hr
    have hy : UnitPos c n ((g - (keyProg c n).inputs) / (keyProg c n).perUnit) (off (keyProg c n) g + j) :=
      ⟨g, hgu, by omega, by omega⟩
    have hread : ∀ x ∈ a ++ b, z = posOf (keyProg c n) x → z ≤ 1 ∨
        ∃ u, UnitPos c n u (off (keyProg c n) g + j) ∧ UnitPos c n u z := by
      intro x hx hzx
      rcases hab x hx with hx1 | hxu
      · have hx2 : x < (keyProg c n).inputs := by rw [keyProg_inputs]; omega
        left; rw [hzx, posOf_input _ hx2]; exact hx1
      · right; refine ⟨_, hy, ?_⟩; rw [hzx]; exact unitPos_posOf hxu
    obtain _ | _ | _ | j := j
    · simp [lower, hr] at h
      subst h
      simp [LGate.reads] at hz
      obtain ⟨x, hx, rfl⟩ := hz
      exact hread x (List.mem_append_left _ hx) rfl
    · simp [lower, hr] at h
      subst h
      simp [LGate.reads] at hz
      obtain ⟨x, hx, rfl⟩ := hz
      exact hread x (List.mem_append_right _ hx) rfl
    · simp [lower, hr] at h
      subst h
      simp [LGate.reads] at hz
      exact .inr ⟨_, hy, g, hgu, by omega, by omega⟩
    · omega

/-! ## An audit's members side by side -/

theorem length_memberProg (ci : Flock.Circuit × Array ℕ) :
    (memberProg ci).1.length = off (keyProg ci.1 ci.2.size) (keyProg ci.1 ci.2.size).size :=
  length_prog _ _

theorem sum_take_map {α : Type} [Inhabited α] (F : α → ℕ) (l : List α) {m : ℕ} (hm : m ≤ l.length) :
    ((l.take m).map F).sum = ((List.range m).map fun i => F (l.getD i default)).sum := by
  induction m with
  | zero => simp
  | succ m ih =>
    rw [List.take_succ, List.map_append, List.sum_append, ih (by omega), sum_range_add_one]
    simp [List.getElem?_eq_getElem (show m < l.length by omega), List.getD_eq_getElem?_getD]

theorem base_eq (ms : Members) {m : ℕ} (hm : m ≤ ms.length) :
    base ms m = ((List.range m).map fun i => (memberProg (ms.getD i default)).1.length).sum := by
  unfold base
  rw [sum_take_map _ _ hm]
  simp only [length_memberProg]

/-- **Member `m + 1`'s positions start past member `m`'s.** -/
theorem base_succ (ms : Members) {m : ℕ} (hm : m < ms.length) :
    base ms (m + 1) = base ms m + (memberProg (ms.getD m default)).1.length := by
  rw [base_eq ms (m := m + 1) hm, base_eq ms (Nat.le_of_lt hm), sum_range_add_one]

theorem base_mono (ms : Members) {m m' : ℕ} (h : m ≤ m') (hm' : m' ≤ ms.length) : base ms m ≤ base ms m' := by
  rw [base_eq ms (Nat.le_trans h hm'), base_eq ms hm']
  exact sum_range_mono _ h

/-- **The members' program's length** is their programs' sum. -/
theorem length_jointOf (ms : Members) : (jointOf ms).gates.length = base ms ms.length := by
  rw [base_eq ms (Nat.le_refl _)]
  show ((List.range ms.length).flatMap fun m =>
    (memberProg (ms.getD m default)).1.map (bgateAt (base ms m))).length = _
  rw [List.length_flatMap]
  simp only [List.length_map]

/-- **Member `m`'s gates sit from `base ms m`**, their reads shifted there. -/
theorem getElem?_jointOf (ms : Members) {m x : ℕ} (hm : m < ms.length)
    (hx : x < (memberProg (ms.getD m default)).1.length) :
    (jointOf ms).gates[base ms m + x]? = ((memberProg (ms.getD m default)).1[x]?).map (bgateAt (base ms m)) := by
  have h := getElem?_flatMap_range (fun m => (memberProg (ms.getD m default)).1.map (bgateAt (base ms m))) hm
    (j := x) (by rw [List.length_map]; exact hx)
  simp only [List.length_map] at h
  rw [List.getElem?_map, ← base_eq ms (Nat.le_of_lt hm)] at h
  exact h

theorem exists_member {ms : Members} {y : ℕ} (hy : y < (jointOf ms).gates.length) :
    ∃ m < ms.length, ∃ x < (memberProg (ms.getD m default)).1.length, y = base ms m + x := by
  rw [length_jointOf, base_eq ms (Nat.le_refl _)] at hy
  obtain ⟨m, hm, x, hx, rfl⟩ := exists_seg _ hy
  exact ⟨m, hm, x, hx, by rw [base_eq ms (Nat.le_of_lt hm)]⟩

/-- **Two members' positions never meet.** -/
theorem joint_unique {ms : Members} {m m' x x' : ℕ} (hm : m < ms.length) (hm' : m' < ms.length)
    (hx : x < (memberProg (ms.getD m default)).1.length) (hx' : x' < (memberProg (ms.getD m' default)).1.length)
    (he : base ms m + x = base ms m' + x') : m = m' ∧ x = x' := by
  rcases Nat.lt_trichotomy m m' with h | rfl | h
  · have := base_mono ms (show m + 1 ≤ m' from h) (Nat.le_of_lt hm')
    rw [base_succ ms hm] at this; omega
  · exact ⟨rfl, by omega⟩
  · have := base_mono ms (show m' + 1 ≤ m from h) (Nat.le_of_lt hm)
    rw [base_succ ms hm'] at this; omega

theorem bgateAt_reads (b : ℕ) (G : LGate) : (bgateAt b G).reads = G.reads.map (b + ·) := by
  cases G <;> rfl

theorem bgateAt_input {b : ℕ} {G : LGate} : bgateAt b G = .input ↔ G = .input := by
  cases G <;> simp [bgateAt]

/-- **The members' outputs** are each member's, from its `base`. -/
theorem mem_outs_jointOf (ms : Members) (y : ℕ) :
    y ∈ (jointOf ms).outs ↔ ∃ m < ms.length, ∃ o ∈ (memberProg (ms.getD m default)).2, y = base ms m + o := by
  show y ∈ (List.range ms.length).flatMap (fun m => (memberProg (ms.getD m default)).2.map (base ms m + ·)) ↔ _
  simp only [List.mem_flatMap, List.mem_range, List.mem_map]
  constructor
  · rintro ⟨m, hm, o, ho, rfl⟩; exact ⟨m, hm, o, ho, rfl⟩
  · rintro ⟨m, hm, o, ho, rfl⟩; exact ⟨m, hm, o, ho, rfl⟩

/-! ## Outputs -/

theorem outGate_some {c : Flock.Circuit} {n u k j t g : ℕ} (h : outGate c n u k j t = some g) :
    u < n ∧ k < c.upv ∧ j < (c.leavesOut.getD k #[]).size ∧ t < 16 ∧
      (Rows.ofNet c.unit).nIn ≤ outCol c j t ∧ outCol c j t < (Rows.ofNet c.unit).nIn + (Rows.ofNet c.unit).nComp ∧
      g = rowGate (Rows.ofNet c.unit) (2 + n * ((Rows.ofNet c.unit).stack c.upv).nIn)
        ((Rows.ofNet c.unit).stack c.upv).nComp u k (outCol c j t) := by
  unfold outGate at h
  dsimp only at h
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, decide_eq_true_eq, and_assoc] at hc
    obtain ⟨hu, hk, hj, ht, h1, h2⟩ := hc
    exact ⟨hu, hk, hj, ht, h1, h2, (Option.some.inj h).symm⟩
  · cases h

/-- **The key program returns every instance's output copy bits.** -/
theorem mem_outs_keyProg (c : Flock.Circuit) (n g : ℕ) :
    g ∈ (keyProg c n).outs ↔ ∃ u k j t, outGate c n u k j t = some g := by
  simp only [keyProg, List.mem_eraseDups, List.mem_mergeSort, List.mem_flatMap, List.mem_range, List.mem_filterMap]
  constructor
  · rintro ⟨u, -, k, -, j, -, t, -, h⟩
    exact ⟨u, k, j, t, h⟩
  · rintro ⟨u, k, j, t, h⟩
    obtain ⟨hu, hk, hj, ht, -⟩ := outGate_some h
    exact ⟨u, hu, k, hk, j, hj, t, ht, h⟩

/-- **An output copy bit is a row of its instance's unit.** -/
theorem row?_outGate {c : Flock.Circuit} {n u k j t g : ℕ} (h : outGate c n u k j t = some g) :
    ∃ ab, (keyProg c n).row? g = some ab := by
  obtain ⟨hu, hk, -, -, h1, h2, rfl⟩ := outGate_some h
  have hNC : ((Rows.ofNet c.unit).stack c.upv).nComp = c.upv * ((Rows.ofNet c.unit).nComp + 1) := rfl
  have h3 : k * ((Rows.ofNet c.unit).nComp + 1) + (1 + (outCol c j t - (Rows.ofNet c.unit).nIn)) <
      ((Rows.ofNet c.unit).stack c.upv).nComp := by
    have := mul_add_le (B := (Rows.ofNet c.unit).nComp + 1) (i := (Rows.ofNet c.unit).nComp + 1) hk (Nat.le_refl _)
    rw [hNC]; omega
  have h4 := mul_add_le (B := ((Rows.ofNet c.unit).stack c.upv).nComp)
    (i := ((Rows.ofNet c.unit).stack c.upv).nComp) hu (Nat.le_refl _)
  have hs := keyProg_size_rows c n
  rw [keyProg_perUnit] at hs
  apply row?_some
  · rw [keyProg_inputs]; unfold rowGate srcs; omega
  · unfold KeyProg.size rowGate; rw [keyProg_inputs, hs]; unfold srcs; omega

/-- **An output copy bit's position holds its row's AND**, never an input or a constant. -/
theorem prog_outGate (κ : ℕ → Option Bool) {c : Flock.Circuit} {n u k j t g : ℕ} (h : outGate c n u k j t = some g) :
    ∃ a b, (prog (keyProg c n) κ).1[posOf (keyProg c n) g]? = some (.and [a, b]) := by
  obtain ⟨⟨a, b⟩, hr⟩ := row?_outGate h
  exact ⟨_, _, prog_and _ κ hr⟩

/-- **The members return every member's output copy bits**, `outBit`, from its `base`. -/
theorem mem_outs_jointOf_iff (ms : Members) (y : ℕ) :
    y ∈ (jointOf ms).outs ↔ ∃ m < ms.length, ∃ u k j t x,
      outBit (ms.getD m default).1 (ms.getD m default).2.size u k j t = some x ∧ y = base ms m + x := by
  rw [mem_outs_jointOf]
  constructor
  · rintro ⟨m, hm, o, ho, rfl⟩
    simp only [memberProg, prog, List.mem_map] at ho
    obtain ⟨g, hg, rfl⟩ := ho
    obtain ⟨u, k, j, t, h⟩ := (mem_outs_keyProg _ _ g).1 hg
    exact ⟨m, hm, u, k, j, t, posOf (keyProg (ms.getD m default).1 (ms.getD m default).2.size) g,
      by simp [outBit, h], rfl⟩
  · rintro ⟨m, hm, u, k, j, t, x, hx, rfl⟩
    simp only [outBit, Option.map_eq_some_iff] at hx
    obtain ⟨g, hg, rfl⟩ := hx
    refine ⟨m, hm, _, ?_, rfl⟩
    simp only [memberProg, prog]
    exact List.mem_map_of_mem ((mem_outs_keyProg _ _ g).2 ⟨u, k, j, t, hg⟩)

/-! ## Reads in the members' program -/

/-- **Position `y` of `jointOf ms` is member `m`'s unit `u`'s.** -/
def JointUnit (ms : Members) (m u y : ℕ) : Prop :=
  m < ms.length ∧ ∃ x, y = base ms m + x ∧ UnitPos (ms.getD m default).1 (ms.getD m default).2.size u x

theorem jointUnit_unique {ms : Members} {m m' u u' y : ℕ} (h : JointUnit ms m u y) (h' : JointUnit ms m' u' y) :
    m = m' ∧ u = u' := by
  obtain ⟨hm, x, rfl, hx⟩ := h
  obtain ⟨hm', x', he, hx'⟩ := h'
  obtain ⟨rfl, rfl⟩ := joint_unique hm hm' (unitPos_lt_length consts hx) (unitPos_lt_length consts hx') he
  exact ⟨rfl, unitPos_unique hx hx'⟩

/-- **A gate of the members' program that reads a unit's position is that unit's.** -/
theorem reads_jointUnit {ms : Members} {m u y z : ℕ} {G : Claims.Sampling.BGate}
    (h : (jointOf ms).gates[y]? = some G) (hz : z ∈ G.reads) (hzu : JointUnit ms m u z) : JointUnit ms m u y := by
  have hy : y < (jointOf ms).gates.length := by
    by_contra hc
    have hle : (jointOf ms).gates.length ≤ y := by omega
    rw [List.getElem?_eq_none hle] at h; cases h
  obtain ⟨m', hm', x, hx, rfl⟩ := exists_member hy
  rw [getElem?_jointOf ms hm' hx, Option.map_eq_some_iff] at h
  obtain ⟨G', hG', rfl⟩ := h
  rw [bgateAt_reads, List.mem_map] at hz
  obtain ⟨z', hz', rfl⟩ := hz
  obtain ⟨hm, w, hw, hwu⟩ := hzu
  have hw2 := two_le_of_unitPos hwu
  have hwl := unitPos_lt_length consts hwu
  rcases reads_unit _ _ consts hG' hz' with h1 | ⟨u', hxu, hzu'⟩
  · have hl := two_le_length (ms.getD m' default).1 (ms.getD m' default).2.size consts
    have hz'l : z' < (memberProg (ms.getD m' default)).1.length := by
      show z' < (prog (keyProg (ms.getD m' default).1 (ms.getD m' default).2.size) consts).1.length
      omega
    obtain ⟨-, rfl⟩ := joint_unique hm' hm hz'l hwl hw
    omega
  · obtain ⟨rfl, rfl⟩ := joint_unique hm' hm (unitPos_lt_length consts hzu') hwl hw
    obtain rfl := unitPos_unique hwu hzu'
    exact ⟨hm', x, rfl, hxu⟩

/-- **A gate that depends on a unit's position is that unit's**: dependence never leaves a unit of a member. -/
theorem depends_unit {ms : Members} {m u z y : ℕ} (hd : NetTiming.Depends (jointOf ms) z y)
    (hz : JointUnit ms m u z) : JointUnit ms m u y := by
  unfold NetTiming.Depends at hd
  induction hd with
  | refl => exact hz
  | @tail b c _ hbc ih =>
    have hbc' : b ∈ (((jointOf ms).gates[c]?).map Claims.Sampling.BGate.reads).getD [] := hbc
    rcases hG : (jointOf ms).gates[c]? with _ | G
    · rw [hG] at hbc'; simp at hbc'
    · rw [hG] at hbc'
      exact reads_jointUnit hG (by simpa using hbc') ih

end Properties.NetworkScan.Lowered
