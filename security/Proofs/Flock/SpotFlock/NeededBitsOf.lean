module

public import Proofs.Flock.SpotFlock.NeededBits
public import Proofs.Flock.SpotFlock.AddressesAt
public import Proofs.Flock.Bridge.BitsRowMap
public import Proofs.Flock.Bridge.Members

/-!
# Every gate a checked unit needs is one of its own unit's bits, where each member's bits are rowed

`NeededBits h` asks, at every accepted run, that each gate its checked unit's proof block needs be a bit of that unit
(`BitOf`): a free source gate with a commitment position, or a returned gate with one. This file proves it from one
hypothesis on the members' circuit files, `MembersRowed h`: each member's free source gates have a commitment position
(`Bits.rowBit`), and so does each returned output leaf bit (`Bits.outRowBit`).

* `blocksClosed_bitsProgs`, `sideBySide_of_held`: the members' proof blocks stand side by side (`SideBySide h`), with no
  hypothesis. A unit's rows read its own source gates (`2 + u·B + b`, whose owner is the unit), the constant and the
  zero (both fixed), and its own rows; a proof block holds only XORs, ANDs and free inputs.
* So a needed gate is a gate of the block (`mem_of_needs`): a free source gate the unit owns, or one of its rows' gates,
  which is then returned (`SpotFlock.needs_committed_of_closed`).
* `neededBits_of_rowed`: a free source gate is a source bit (`sourceBits`) of the unit, and a returned row gate a written
  bit (`writtenBits`) of the unit, each at its member's own positions (`pos_eq`).

`MembersRowed h` fails at a member whose unit reads a source no row port holds (a wire into the unit, a literal, a
padding row): its source gate is free, its block needs it, and it has no commitment position.
-/

@[expose] public section

namespace Flock.Bridge.NeededBitsOf

open Claims.Sampling Properties.Spot

/-! ## Arithmetic -/

theorem div_units {N u i : ℕ} (hi : i < N) : (u * N + i) / N = u := by
  rw [Nat.mul_comm, Nat.mul_add_div (by omega), Nat.div_eq_of_lt hi, Nat.add_zero]

theorem mul_add_lt {N n u i : ℕ} (hu : u < n) (hi : i < N) : u * N + i < n * N := by
  have := Nat.mul_le_mul_right N (show u + 1 ≤ n by omega)
  rw [Nat.add_mul, Nat.one_mul] at this
  omega

theorem eq_units {N u u' i i' : ℕ} (hi : i < N) (hi' : i' < N) (h : u * N + i = u' * N + i') : u = u' :=
  calc u = (u * N + i) / N := (div_units hi).symm
    _ = (u' * N + i') / N := by rw [h]
    _ = u' := div_units hi'

theorem split_units {k m n g : ℕ} (h1 : k ≤ g) (h2 : g < k + n * m) :
    ∃ u i, u < n ∧ i < m ∧ g = k + (u * m + i) := by
  have hm : 0 < m := Nat.pos_of_ne_zero fun h => by simp [h] at h2; omega
  refine ⟨(g - k) / m, (g - k) % m, (Nat.div_lt_iff_lt_mul hm).2 (by omega), Nat.mod_lt _ hm, ?_⟩
  have := Nat.div_add_mod (g - k) m
  rw [Nat.mul_comm] at this
  omega

theorem src_lt {B n u b : ℕ} (hu : u < n) (hb : b < B) : 2 + u * B + b < 2 + n * B := by
  have := mul_add_lt hu hb
  omega

/-! ## A key program's gates, lowered -/

section kp

variable {kp : Spot.Bits.KeyProg}

theorem row?_of_lt {g : ℕ} (hg : g < kp.inputs) : kp.row? g = none := by
  simp [Spot.Bits.KeyProg.row?, hg]

theorem lt_size_of_lt {g : ℕ} (hg : g < kp.inputs) : g < kp.size := by
  unfold Spot.Bits.KeyProg.size
  omega

theorem row?_some {g : ℕ} (h1 : kp.inputs ≤ g) (h2 : g < kp.size) : ∃ a b, kp.row? g = some (a, b) := by
  have h3 : g - kp.inputs < kp.rows.size := by
    unfold Spot.Bits.KeyProg.size at h2
    omega
  refine ⟨kp.rows[g - kp.inputs].1, kp.rows[g - kp.inputs].2, ?_⟩
  simp [Spot.Bits.KeyProg.row?, Nat.not_lt.2 h1, h3]

theorem lt_inputs_of_row_none {g : ℕ} (hs : g < kp.size) (hr : kp.row? g = none) : g < kp.inputs := by
  refine Nat.lt_of_not_le fun h => ?_
  obtain ⟨a, b, hab⟩ := row?_some h hs
  rw [hab] at hr
  cases hr

theorem unitOf_eq_some {g u : ℕ} (h : Spot.Bits.unitOf kp g = some u) :
    kp.inputs ≤ g ∧ g < kp.size ∧ 0 < kp.perUnit ∧ (g - kp.inputs) / kp.perUnit = u := by
  unfold Spot.Bits.unitOf at h
  split at h
  · cases h
  · rename_i hc
    simp only [Bool.or_eq_true, decide_eq_true_eq, beq_iff_eq] at hc
    exact ⟨by omega, by omega, by omega, Option.some.inj h⟩

theorem row?_of_unitOf {g u : ℕ} (h : Spot.Bits.unitOf kp g = some u) : ∃ a b, kp.row? g = some (a, b) :=
  row?_some (unitOf_eq_some h).1 (unitOf_eq_some h).2.1

theorem gateSize_of_some {g : ℕ} {a b : List ℕ} (h : kp.row? g = some (a, b)) : Spot.Bits.gateSize kp g = 3 := by
  simp [Spot.Bits.gateSize, h]

theorem gateSize_of_none {g : ℕ} (h : kp.row? g = none) : Spot.Bits.gateSize kp g = 1 := by
  simp [Spot.Bits.gateSize, h]

theorem one_le_gateSize {g : ℕ} : 1 ≤ Spot.Bits.gateSize kp g := by
  unfold Spot.Bits.gateSize
  split <;> omega

theorem posOf_of_some {g : ℕ} {a b : List ℕ} (h : kp.row? g = some (a, b)) :
    Spot.Bits.posOf kp g = Spot.Bits.off kp g + 2 := by
  have := gateSize_of_some h
  unfold Spot.Bits.posOf
  omega

theorem posOf_of_none {g : ℕ} (h : kp.row? g = none) : Spot.Bits.posOf kp g = Spot.Bits.off kp g := by
  have := gateSize_of_none h
  unfold Spot.Bits.posOf
  omega

theorem off_succ {g : ℕ} (hg : g < kp.size) :
    Spot.Bits.off kp (g + 1) = Spot.Bits.off kp g + Spot.Bits.gateSize kp g := by
  rw [SecurityProofs.off_of_le kp (show g + 1 ≤ kp.size by omega), SecurityProofs.off_of_le kp (Nat.le_of_lt hg),
    List.range_succ, List.map_append, List.sum_append]
  simp

theorem off_mono {a : ℕ} : ∀ {b : ℕ}, a ≤ b → b ≤ kp.size → Spot.Bits.off kp a ≤ Spot.Bits.off kp b
  | 0, hab, _ => by
    obtain rfl : a = 0 := by omega
    exact Nat.le_refl _
  | b + 1, hab, hb => by
    rcases Nat.lt_or_ge a (b + 1) with hlt | hge
    · have := off_mono (a := a) (b := b) (by omega) (by omega)
      rw [off_succ (show b < kp.size by omega)]
      omega
    · obtain rfl : a = b + 1 := by omega
      exact Nat.le_refl _

theorem le_off : ∀ {g : ℕ}, g ≤ kp.size → g ≤ Spot.Bits.off kp g
  | 0, _ => Nat.zero_le _
  | g + 1, hg => by
    have h1 := le_off (g := g) (by omega)
    have h2 := one_le_gateSize (kp := kp) (g := g)
    rw [off_succ (show g < kp.size by omega)]
    omega

theorem off_eq_of_le_inputs : ∀ {g : ℕ}, g ≤ kp.inputs → Spot.Bits.off kp g = g
  | 0, _ => by simp [SecurityProofs.off_of_le kp (Nat.zero_le _)]
  | g + 1, hg => by
    rw [off_succ (lt_size_of_lt (show g < kp.inputs by omega)), off_eq_of_le_inputs (show g ≤ kp.inputs by omega),
      gateSize_of_none (row?_of_lt (show g < kp.inputs by omega))]

theorem off_add_lt {g j : ℕ} (hg : g < kp.size) (hj : j < Spot.Bits.gateSize kp g) :
    Spot.Bits.off kp g + j < Spot.Bits.off kp kp.size := by
  have h1 := off_succ hg
  have h2 := off_mono (show g + 1 ≤ kp.size by omega) (Nat.le_refl kp.size)
  omega

theorem posOf_lt_off {g : ℕ} (hg : g < kp.size) : Spot.Bits.posOf kp g < Spot.Bits.off kp kp.size := by
  have h1 := one_le_gateSize (kp := kp) (g := g)
  have h2 := off_add_lt hg (j := Spot.Bits.gateSize kp g - 1) (by omega)
  unfold Spot.Bits.posOf
  omega

/-- **Distinct gates' lowered gates are apart.** -/
theorem eq_of_off {g g' j j' : ℕ} (hg : g < kp.size) (hg' : g' < kp.size) (hj : j < Spot.Bits.gateSize kp g)
    (hj' : j' < Spot.Bits.gateSize kp g') (h : Spot.Bits.off kp g + j = Spot.Bits.off kp g' + j') : g = g' := by
  refine Nat.le_antisymm (Nat.le_of_not_lt fun hlt => ?_) (Nat.le_of_not_lt fun hlt => ?_)
  · have := off_mono (show g' + 1 ≤ g by omega) (Nat.le_of_lt hg)
    rw [off_succ hg'] at this
    omega
  · have := off_mono (show g + 1 ≤ g' by omega) (Nat.le_of_lt hg')
    rw [off_succ hg] at this
    omega

theorem mem_unitGates {u q : ℕ} (hq : q ∈ Spot.Bits.unitGates kp u) :
    ∃ g < kp.size, Spot.Bits.unitOf kp g = some u ∧ ∃ j < Spot.Bits.gateSize kp g, q = Spot.Bits.off kp g + j := by
  unfold Spot.Bits.unitGates at hq
  obtain ⟨g, hg, hq⟩ := List.mem_flatMap.1 hq
  split at hq
  · rename_i hu
    obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hq
    exact ⟨g, List.mem_range.1 hg, by simpa using hu, j, List.mem_range.1 hj, rfl⟩
  · simp at hq

theorem mem_unitGates_of {g u j : ℕ} (hg : g < kp.size) (hu : Spot.Bits.unitOf kp g = some u)
    (hj : j < Spot.Bits.gateSize kp g) : Spot.Bits.off kp g + j ∈ Spot.Bits.unitGates kp u := by
  unfold Spot.Bits.unitGates
  refine List.mem_flatMap.2 ⟨g, List.mem_range.2 hg, ?_⟩
  simp [hu, hj]

theorem mem_insOf {κ : ℕ → Option Bool} {u g : ℕ} :
    g ∈ Spot.Bits.insOf kp κ u ↔
      g < kp.size ∧ kp.row? g = none ∧ κ g = none ∧ Spot.Bits.owner kp g = some u := by
  simp [Spot.Bits.insOf, List.mem_filter, Option.isNone_iff_eq_none, and_assoc]

theorem exists_of_owner {w k : ℕ} (h : Spot.Bits.owner kp w = some k) :
    ∃ g, Spot.Bits.unitOf kp g = some k ∧ w ∈ Spot.Bits.args kp g := by
  unfold Spot.Bits.owner at h
  have hp := List.find?_some h
  obtain ⟨g, -, hgp⟩ := List.any_eq_true.1 hp
  simp only [Bool.and_eq_true, beq_iff_eq, List.contains_iff_mem] at hgp
  exact ⟨g, hgp⟩

theorem lower_row (κ : ℕ → Option Bool) {g : ℕ} {a b : List ℕ} (h : kp.row? g = some (a, b)) :
    Spot.Bits.lower kp κ g = [.xor (a.map (Spot.Bits.posOf kp)), .xor (b.map (Spot.Bits.posOf kp)),
      .and [Spot.Bits.off kp g, Spot.Bits.off kp g + 1]] := by
  simp only [Spot.Bits.lower, h]

/-- **Gate `g`'s lowered gate `j`** sits at `off kp g + j`. -/
theorem bitsProg_gate_at (κ : ℕ → Option Bool) {g j : ℕ} (hg : g < kp.size) (hj : j < Spot.Bits.gateSize kp g) :
    (bitsProg kp κ).gates[Spot.Bits.off kp g + j]? = ((Spot.Bits.lower kp κ g)[j]?).map bitsGate := by
  have hoff : Spot.Bits.off kp g = ((List.range g).map fun x => (Spot.Bits.lower kp κ x).length).sum := by
    rw [SecurityProofs.off_of_le kp (Nat.le_of_lt hg)]
    simp only [SecurityProofs.lower_length]
  simp only [bitsProg, Spot.Bits.prog, List.getElem?_map]
  rw [hoff, Members.getElem?_flatMap_range _ hg (by rw [SecurityProofs.lower_length]; exact hj)]

theorem gate_fixed (κ : ℕ → Option Bool) {g : ℕ} {b : Bool} (hs : g < kp.size) (hr : kp.row? g = none)
    (hb : κ g = some b) : (bitsProg kp κ).gates[Spot.Bits.posOf kp g]? = some (.const b) := by
  rw [SecurityProofs.bitsProg_gate kp κ hs, gateSize_of_none hr]
  simp [Spot.Bits.lower, hr, hb, bitsGate]

theorem gate_free (κ : ℕ → Option Bool) {g : ℕ} (hs : g < kp.size) (hr : kp.row? g = none) (hb : κ g = none) :
    (bitsProg kp κ).gates[Spot.Bits.posOf kp g]? = some .input := by
  rw [SecurityProofs.bitsProg_gate kp κ hs, gateSize_of_none hr]
  simp [Spot.Bits.lower, hr, hb, bitsGate]

theorem gate_input_free (κ : ℕ → Option Bool) {g : ℕ} (hg : g < kp.inputs) (hb : κ g = none) :
    (bitsProg kp κ).gates[g]? = some .input := by
  have h := gate_free κ (lt_size_of_lt hg) (row?_of_lt hg) hb
  rwa [posOf_of_none (row?_of_lt hg), off_eq_of_le_inputs (Nat.le_of_lt hg)] at h

theorem row_gate_not_input (κ : ℕ → Option Bool) {g j : ℕ} {a b : List ℕ} (hg : g < kp.size)
    (hr : kp.row? g = some (a, b)) (hj : j < Spot.Bits.gateSize kp g) :
    (bitsProg kp κ).gates[Spot.Bits.off kp g + j]? ≠ some .input := by
  rw [bitsProg_gate_at κ hg hj, lower_row κ hr]
  rw [gateSize_of_some hr] at hj
  rcases (show j = 0 ∨ j = 1 ∨ j = 2 by omega) with rfl | rfl | rfl <;> simp [bitsGate]

theorem lt_len_of_input (κ : ℕ → Option Bool) {g : ℕ} (hg : g < kp.inputs) : g < (bitsProg kp κ).gates.length := by
  rw [SecurityProofs.bitsProg_length]
  have := off_add_lt (lt_size_of_lt hg) (j := 0) (by rw [gateSize_of_none (row?_of_lt hg)]; omega)
  rw [off_eq_of_le_inputs (Nat.le_of_lt hg)] at this
  omega

theorem lt_length_of_mem_blockOf {κ : ℕ → Option Bool} {u q : ℕ} (hq : q ∈ Spot.Bits.blockOf kp κ u) :
    q < (bitsProg kp κ).gates.length := by
  rw [SecurityProofs.bitsProg_length]
  unfold Spot.Bits.blockOf at hq
  rcases List.mem_append.1 hq with hq | hq
  · obtain ⟨g, hgs, -, j, hj, rfl⟩ := mem_unitGates hq
    exact off_add_lt hgs hj
  · obtain ⟨g, hg, rfl⟩ := List.mem_map.1 hq
    exact posOf_lt_off (mem_insOf.1 hg).1

end kp

/-! ## A circuit file's key program -/

section key

variable (c : Flock.Circuit) (n : ℕ)

theorem memberProg_eq : memberProg c n = bitsProg (Spot.Bits.keyProg c n) (Spot.Bits.consts c n) := rfl

theorem memberProg_outs :
    (memberProg c n).outs = (Spot.Bits.keyProg c n).outs.map (Spot.Bits.posOf (Spot.Bits.keyProg c n)) := rfl

theorem consts_zero : Spot.Bits.consts c n 0 = some true := by
  simp [Spot.Bits.consts]

theorem consts_one : Spot.Bits.consts c n 1 = some false := by
  simp [Spot.Bits.consts]

/-- **Unit `u`'s column `col`** is the constant, the zero, a source gate `2 + u·B + b` of its own, or one of its own
rows. -/
theorem keyWire_cases (u col : ℕ) :
    keyWire c n u col = 0 ∨ keyWire c n u col = 1 ∨
      (∃ b, Spot.Bits.keyIdx c (bitsRows c).nIn col = some b ∧
        keyWire c n u col = 2 + u * (bitsRows c).nIn + b) ∨
      ∃ i < (bitsRows c).nComp,
        keyWire c n u col = 2 + n * (bitsRows c).nIn + (u * (bitsRows c).nComp + i) := by
  unfold keyWire
  by_cases h1 : col < (bitsRows c).nIn
  · rw [if_pos h1]
    have hk : ((Array.range (bitsRows c).nIn).map (Spot.Bits.keyIdx c (bitsRows c).nIn)).getD col none =
        Spot.Bits.keyIdx c (bitsRows c).nIn col := by
      simp [Array.getD_eq_getD_getElem?, h1]
    rw [hk]
    cases hb : Spot.Bits.keyIdx c (bitsRows c).nIn col with
    | none => exact Or.inr (Or.inl rfl)
    | some b => exact Or.inr (Or.inr (Or.inl ⟨b, rfl, rfl⟩))
  · rw [if_neg h1]
    by_cases h2 : col < (bitsRows c).nIn + (bitsRows c).nComp
    · rw [if_pos h2]
      exact Or.inr (Or.inr (Or.inr ⟨col - (bitsRows c).nIn, by omega, by omega⟩))
    · rw [if_neg h2]
      exact Or.inl rfl

theorem unitOf_some {g u : ℕ} (hg : Spot.Bits.unitOf (Spot.Bits.keyProg c n) g = some u) :
    u < n ∧ ∃ i < (bitsRows c).nComp, g = 2 + n * (bitsRows c).nIn + (u * (bitsRows c).nComp + i) := by
  obtain ⟨h1, h2, -, h4⟩ := unitOf_eq_some hg
  rw [keyProg_inputs] at h1 h4
  rw [keyProg_size] at h2
  rw [keyProg_perUnit] at h4
  obtain ⟨u', i, hu', hi, rfl⟩ := split_units h1 h2
  rw [Nat.add_sub_cancel_left, div_units hi] at h4
  subst h4
  exact ⟨hu', i, hi, rfl⟩

theorem unitOf_row {u i : ℕ} (hu : u < n) (hi : i < (bitsRows c).nComp) :
    Spot.Bits.unitOf (Spot.Bits.keyProg c n) (2 + n * (bitsRows c).nIn + (u * (bitsRows c).nComp + i)) =
      some u := by
  have h := mul_add_lt hu hi
  have hs := keyProg_size c n
  have hin : (Spot.Bits.keyProg c n).inputs = 2 + n * (bitsRows c).nIn := keyProg_inputs c n
  have hpu : (Spot.Bits.keyProg c n).perUnit = (bitsRows c).nComp := keyProg_perUnit c n
  unfold Spot.Bits.unitOf
  rw [if_neg]
  · rw [hin, hpu, Nat.add_sub_cancel_left, div_units hi]
  · simp only [Bool.or_eq_true, decide_eq_true_eq, beq_iff_eq, not_or]
    omega

/-- **What a unit's row reads**: the constant, the zero, one of the unit's own source gates, or one of its own rows. -/
theorem args_of_unit {g u w : ℕ} (hg : Spot.Bits.unitOf (Spot.Bits.keyProg c n) g = some u)
    (hw : w ∈ Spot.Bits.args (Spot.Bits.keyProg c n) g) :
    u < n ∧ (w = 0 ∨ w = 1 ∨
      (∃ col b, Spot.Bits.keyIdx c (bitsRows c).nIn col = some b ∧ w = 2 + u * (bitsRows c).nIn + b) ∨
      ∃ i < (bitsRows c).nComp, w = 2 + n * (bitsRows c).nIn + (u * (bitsRows c).nComp + i)) := by
  obtain ⟨hu, i, hi, rfl⟩ := unitOf_some c n hg
  refine ⟨hu, ?_⟩
  simp only [Spot.Bits.args, keyProg_row?_row c n hu hi, List.mem_append, List.mem_map] at hw
  rcases hw with ⟨col, -, rfl⟩ | ⟨col, -, rfl⟩
  all_goals
    rcases keyWire_cases c n u col with h | h | ⟨b, hb, h⟩ | ⟨j, hj, h⟩
    · exact Or.inl h
    · exact Or.inr (Or.inl h)
    · exact Or.inr (Or.inr (Or.inl ⟨col, b, hb, h⟩))
    · exact Or.inr (Or.inr (Or.inr ⟨j, hj, h⟩))

/-- **A unit owns the source gates its rows read**: no earlier unit reads them. -/
theorem owner_src {g u b : ℕ} (hg : Spot.Bits.unitOf (Spot.Bits.keyProg c n) g = some u)
    (hb : b < (bitsRows c).nIn) (hw : 2 + u * (bitsRows c).nIn + b ∈ Spot.Bits.args (Spot.Bits.keyProg c n) g) :
    Spot.Bits.owner (Spot.Bits.keyProg c n) (2 + u * (bitsRows c).nIn + b) = some u := by
  have hu := (unitOf_some c n hg).1
  have hgs := (unitOf_eq_some hg).2.1
  have hlt := src_lt hu hb
  unfold Spot.Bits.owner
  rw [List.find?_range_eq_some]
  refine ⟨List.any_eq_true.2 ⟨g, List.mem_range.2 hgs, ?_⟩, List.mem_range.2 (by rw [keyProg_units]; exact hu),
    fun j hj => ?_⟩
  · simp only [hg, beq_self_eq_true, Bool.true_and]
    exact List.contains_iff_mem.2 hw
  · have hf : ((List.range (Spot.Bits.keyProg c n).size).any fun h =>
        Spot.Bits.unitOf (Spot.Bits.keyProg c n) h == some j &&
          (Spot.Bits.args (Spot.Bits.keyProg c n) h).contains (2 + u * (bitsRows c).nIn + b)) = false := by
      rw [List.any_eq_false]
      intro h' _ hh
      simp only [Bool.and_eq_true, beq_iff_eq, List.contains_iff_mem] at hh
      rcases (args_of_unit c n hh.1 hh.2).2 with h0 | h0 | ⟨col, b', hb', h0⟩ | ⟨i, hi, h0⟩
      · omega
      · omega
      · have := eq_units hb (keyIdx_lt hb') (show u * (bitsRows c).nIn + b = j * (bitsRows c).nIn + b' by omega)
        omega
      · omega
    simp only [hf, Bool.not_false]

theorem outGate_some {u k j t x : ℕ} (h : Spot.Bits.outGate c n u k j t = some x) :
    u < n ∧ k < c.upv ∧ j < (c.leavesOut.getD k #[]).size ∧ t < 16 ∧
      ∃ i < (bitsRows c).nComp, x = 2 + n * (bitsRows c).nIn + (u * (bitsRows c).nComp + i) := by
  unfold Spot.Bits.outGate at h
  dsimp only at h
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
    obtain ⟨⟨⟨⟨⟨hu, hk⟩, hj⟩, ht⟩, h1⟩, h2⟩ := hc
    cases h
    refine ⟨hu, hk, hj, ht, k * ((Spot.Bits.Rows.ofNet c.unit).nComp + 1) +
      (1 + (Spot.Bits.outCol c j t - (Spot.Bits.Rows.ofNet c.unit).nIn)), ?_, ?_⟩
    · have hle := Nat.mul_le_mul_right ((Spot.Bits.Rows.ofNet c.unit).nComp + 1) (show k + 1 ≤ c.upv by omega)
      rw [Nat.add_mul, Nat.one_mul] at hle
      show _ < c.upv * ((Spot.Bits.Rows.ofNet c.unit).nComp + 1)
      omega
    · unfold Spot.Bits.rowGate bitsRows
      omega
  · cases h

theorem of_mem_outs {o : ℕ} (ho : o ∈ (Spot.Bits.keyProg c n).outs) :
    ∃ u k j t, Spot.Bits.outGate c n u k j t = some o := by
  rw [keyProg_outs, List.mem_eraseDups, List.mem_mergeSort] at ho
  simp only [List.mem_flatMap, List.mem_range, List.mem_filterMap] at ho
  obtain ⟨u, -, k, -, j, -, t, -, h⟩ := ho
  exact ⟨u, k, j, t, h⟩

end key

/-! ## Commitment positions -/

theorem srcPos_leaf {c : Flock.Circuit} {u : ℕ} {s : Spot.Bits.Src} {rp : Spot.Bits.RowPos}
    (h : Spot.Bits.srcPos c u s = some rp) : rp.leaf = u := by
  unfold Spot.Bits.srcPos at h
  split at h
  · cases h
    rfl
  · cases h

theorem rowBit_leaf {c : Flock.Circuit} {n g : ℕ} {rp : Spot.Bits.RowPos} (h : Spot.Bits.rowBit c n g = some rp) :
    rp.leaf = (g - 2) / (bitsRows c).nIn := by
  unfold Spot.Bits.rowBit at h
  dsimp only at h
  split at h
  · cases h
  · exact srcPos_leaf h

theorem outRowBit_leaf {c : Flock.Circuit} {u k j t : ℕ} {rp : Spot.Bits.RowPos}
    (h : Spot.Bits.outRowBit c u k j t = some rp) : rp.leaf = u := by
  unfold Spot.Bits.outRowBit at h
  obtain ⟨x, -, hx⟩ := Option.bind_eq_some_iff.1 h
  unfold Spot.Bits.outPos at hx
  dsimp only at hx
  obtain ⟨o, -, rfl⟩ := Option.map_eq_some_iff.1 hx
  rfl

/-! ## Programs side by side -/

theorem gates_sideBySide (ps : List Claims.Sampling.Prog) {m p : ℕ} (hm : m < ps.length)
    (hp : p < ps[m].gates.length) :
    (sideBySide ps).gates[offsetOf ps m + p]? = ps[m].gates[p]?.map (shiftGate (offsetOf ps m)) := by
  induction ps generalizing m with
  | nil => exact absurd hm (Nat.not_lt_zero _)
  | cons q ps ih =>
    cases m with
    | zero =>
      have hp' : p < q.gates.length := by simpa using hp
      simp [Members.offsetOf_zero, Members.sideBySide_gates, List.getElem?_append_left hp', Members.shiftGate_zero]
    | succ m =>
      have hm' : m < ps.length := by simpa using hm
      have hp' : p < ps[m].gates.length := by simpa using hp
      rw [Members.offsetOf_succ, Members.sideBySide_gates,
        List.getElem?_append_right (show q.gates.length ≤ q.gates.length + offsetOf ps m + p by omega),
        List.getElem?_map, show q.gates.length + offsetOf ps m + p - q.gates.length = offsetOf ps m + p by omega,
        ih hm' hp', Option.map_map, Members.shiftGate_comp]
      simp

theorem exists_of_mem_outs (ps : List Claims.Sampling.Prog) {i : ℕ} (hi : i ∈ (sideBySide ps).outs) :
    ∃ m, ∃ hm : m < ps.length, ∃ p ∈ ps[m].outs, i = offsetOf ps m + p := by
  induction ps generalizing i with
  | nil => simp [sideBySide] at hi
  | cons q ps ih =>
    rw [Members.sideBySide_outs, List.mem_append] at hi
    rcases hi with hi | hi
    · exact ⟨0, by simp, i, by simpa using hi, by simp [Members.offsetOf_zero]⟩
    · obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hi
      obtain ⟨m, hm, p, hp, rfl⟩ := ih hj
      exact ⟨m + 1, by simpa using hm, p, by simpa using hp, by rw [Members.offsetOf_succ]; omega⟩

section members

variable (ms : Spot.Bits.Members)

theorem gates_bitsProgs {m p : ℕ} (hm : m < ms.length) (hp : p < (memberProg ms[m].1 ms[m].2.size).gates.length) :
    (bitsProgs ms).gates[Spot.Bits.base ms m + p]? =
      (memberProg ms[m].1 ms[m].2.size).gates[p]?.map (shiftGate (Spot.Bits.base ms m)) := by
  have h := gates_sideBySide (ms.map fun ci => memberProg ci.1 ci.2.size) (m := m) (p := p) (by simpa using hm)
    (by simpa using hp)
  rw [Members.offsetOf_bitsProgs, List.getElem_map] at h
  exact h

theorem exists_of_mem_outs_bitsProgs {i : ℕ} (hi : i ∈ (bitsProgs ms).outs) :
    ∃ m, ∃ hm : m < ms.length, ∃ p ∈ (memberProg ms[m].1 ms[m].2.size).outs, i = Spot.Bits.base ms m + p := by
  unfold bitsProgs at hi
  obtain ⟨m, hm, p, hp, rfl⟩ := exists_of_mem_outs _ hi
  have hm' : m < ms.length := by simpa using hm
  refine ⟨m, hm', p, by simpa using hp, ?_⟩
  rw [Members.offsetOf_bitsProgs]

/-- **Positions of two members are apart**: a position of member `m`'s program is one of member `t`'s only where
`m = t`, at the same gate. -/
theorem pos_eq {m t p q : ℕ} (hm : m < ms.length) (ht : t < ms.length)
    (hp : p < (memberProg ms[m].1 ms[m].2.size).gates.length) (hq : q < (memberProg ms[t].1 ms[t].2.size).gates.length)
    (h : Spot.Bits.base ms m + p = Spot.Bits.base ms t + q) : m = t ∧ p = q := by
  have hmt : m = t := by
    refine Nat.le_antisymm (Nat.le_of_not_lt fun hlt => ?_) (Nat.le_of_not_lt fun hlt => ?_)
    · have h1 := Members.base_succ ms ht
      have h2 := Members.base_mono ms (show t + 1 ≤ m by omega)
      omega
    · have h1 := Members.base_succ ms hm
      have h2 := Members.base_mono ms (show m + 1 ≤ t by omega)
      omega
  subst hmt
  exact ⟨rfl, by omega⟩

theorem mem_blocks_membersPart {bl : List ℕ} (hbl : bl ∈ (membersPart ms).blocks) :
    ∃ m, ∃ hm : m < ms.length, ∃ u < ms[m].2.size,
      bl = (Spot.Bits.blockOf (Spot.Bits.keyProg ms[m].1 ms[m].2.size) (Spot.Bits.consts ms[m].1 ms[m].2.size)
        u).map (Spot.Bits.base ms m + ·) := by
  rw [Members.blocks_membersPart] at hbl
  obtain ⟨m, hm, hbl⟩ := List.mem_flatMap.1 hbl
  have hm' := List.mem_range.1 hm
  simp only [memberBlocks, List.getElem?_eq_getElem hm', List.mem_map, List.mem_range] at hbl
  obtain ⟨u, hu, rfl⟩ := hbl
  exact ⟨m, hm', u, hu, rfl⟩

theorem exists_member {u : ℕ} (hu : u < unitBase ms ms.length) :
    ∃ t, ∃ ht : t < ms.length, ∃ k < ms[t].2.size, u = unitBase ms t + k := by
  suffices H : ∀ m ≤ ms.length, u < unitBase ms m →
      ∃ t, ∃ ht : t < ms.length, ∃ k < ms[t].2.size, u = unitBase ms t + k from H _ (Nat.le_refl _) hu
  intro m
  induction m with
  | zero =>
    intro _ h0
    simp [unitBase] at h0
  | succ m ih =>
    intro hm hlt
    by_cases h' : u < unitBase ms m
    · exact ih (by omega) h'
    · have hm' : m < ms.length := by omega
      have := Members.unitBase_succ ms hm'
      exact ⟨m, hm', u - unitBase ms m, by omega, by omega⟩

theorem getD_ms {m : ℕ} (hm : m < ms.length) : ms.getD m default = ms[m] := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hm, Option.getD_some]

end members

/-! ## The members' proof blocks stand side by side -/

section closed

variable (c : Flock.Circuit) (n : ℕ)

/-- **A member's proof block is closed**: each of its gates reads only gates of the block and constants, and none is a
constant. -/
theorem member_closed {u q : ℕ} (hq : q ∈ Spot.Bits.blockOf (Spot.Bits.keyProg c n) (Spot.Bits.consts c n) u)
    {x : BGate} (hx : (memberProg c n).gates[q]? = some x) :
    (∀ b, x ≠ .const b) ∧ ∀ w ∈ x.reads, w ∈ Spot.Bits.blockOf (Spot.Bits.keyProg c n) (Spot.Bits.consts c n) u ∨
      ∃ b, (memberProg c n).gates[w]? = some (.const b) := by
  rw [memberProg_eq] at hx ⊢
  unfold Spot.Bits.blockOf at hq ⊢
  rcases List.mem_append.1 hq with hq | hq
  · obtain ⟨g, hgs, hgu, j, hj, rfl⟩ := mem_unitGates hq
    obtain ⟨a, b, hab⟩ := row?_of_unitOf hgu
    have hu := (unitOf_some c n hgu).1
    have harg : ∀ w0 ∈ Spot.Bits.args (Spot.Bits.keyProg c n) g,
        Spot.Bits.posOf (Spot.Bits.keyProg c n) w0 ∈ Spot.Bits.unitGates (Spot.Bits.keyProg c n) u ++
            (Spot.Bits.insOf (Spot.Bits.keyProg c n) (Spot.Bits.consts c n) u).map
              (Spot.Bits.posOf (Spot.Bits.keyProg c n)) ∨
          ∃ b, (bitsProg (Spot.Bits.keyProg c n) (Spot.Bits.consts c n)).gates[
            Spot.Bits.posOf (Spot.Bits.keyProg c n) w0]? = some (.const b) := by
      intro w0 hw0
      have hin1 : 1 < (Spot.Bits.keyProg c n).inputs := by
        rw [keyProg_inputs]
        omega
      rcases (args_of_unit c n hgu hw0).2 with rfl | rfl | ⟨col, b', hb', rfl⟩ | ⟨i, hi, rfl⟩
      · exact Or.inr ⟨true, gate_fixed _ (lt_size_of_lt (by omega)) (row?_of_lt (by omega)) (consts_zero c n)⟩
      · exact Or.inr ⟨false, gate_fixed _ (lt_size_of_lt hin1) (row?_of_lt hin1) (consts_one c n)⟩
      · have hbB := keyIdx_lt hb'
        have hlt : 2 + u * (bitsRows c).nIn + b' < (Spot.Bits.keyProg c n).inputs := by
          rw [keyProg_inputs]
          exact src_lt hu hbB
        cases hκ : Spot.Bits.consts c n (2 + u * (bitsRows c).nIn + b') with
        | some b'' => exact Or.inr ⟨b'', gate_fixed _ (lt_size_of_lt hlt) (row?_of_lt hlt) hκ⟩
        | none =>
          exact Or.inl (List.mem_append_right _ (List.mem_map.2 ⟨_, mem_insOf.2
            ⟨lt_size_of_lt hlt, row?_of_lt hlt, hκ, owner_src c n hgu hbB hw0⟩, rfl⟩))
      · have hwu := unitOf_row c n hu hi
        obtain ⟨a', b'', hab'⟩ := row?_of_unitOf hwu
        rw [posOf_of_some hab']
        exact Or.inl (List.mem_append_left _ (mem_unitGates_of (unitOf_eq_some hwu).2.1 hwu
          (by rw [gateSize_of_some hab']; omega)))
    have h3 := gateSize_of_some hab
    rw [bitsProg_gate_at _ hgs hj, lower_row _ hab] at hx
    have hj3 : j < 3 := h3 ▸ hj
    rcases (show j = 0 ∨ j = 1 ∨ j = 2 by omega) with rfl | rfl | rfl
    · obtain rfl : x = .xor (a.map (Spot.Bits.posOf (Spot.Bits.keyProg c n))) := by
        simpa [bitsGate] using hx.symm
      refine ⟨fun b h => (by cases h), fun w hw => ?_⟩
      obtain ⟨w0, hw0, rfl⟩ := List.mem_map.1 hw
      exact harg w0 (by simp [Spot.Bits.args, hab, hw0])
    · obtain rfl : x = .xor (b.map (Spot.Bits.posOf (Spot.Bits.keyProg c n))) := by
        simpa [bitsGate] using hx.symm
      refine ⟨fun b h => (by cases h), fun w hw => ?_⟩
      obtain ⟨w0, hw0, rfl⟩ := List.mem_map.1 hw
      exact harg w0 (by simp [Spot.Bits.args, hab, hw0])
    · obtain rfl : x = .and [Spot.Bits.off (Spot.Bits.keyProg c n) g, Spot.Bits.off (Spot.Bits.keyProg c n) g + 1] := by
        simpa [bitsGate] using hx.symm
      refine ⟨fun b h => (by cases h), fun w hw => ?_⟩
      have hw' : w = Spot.Bits.off (Spot.Bits.keyProg c n) g + 0 ∨ w = Spot.Bits.off (Spot.Bits.keyProg c n) g + 1 := by
        simpa [BGate.reads] using hw
      rcases hw' with rfl | rfl
      · exact Or.inl (List.mem_append_left _ (mem_unitGates_of hgs hgu (by omega)))
      · exact Or.inl (List.mem_append_left _ (mem_unitGates_of hgs hgu (by omega)))
  · obtain ⟨g, hg, rfl⟩ := List.mem_map.1 hq
    obtain ⟨hgs, hr, hκ, -⟩ := mem_insOf.1 hg
    rw [gate_free _ hgs hr hκ] at hx
    cases hx
    exact ⟨fun b h => (by cases h), fun w hw => absurd hw (by simp [BGate.reads])⟩

end closed

/-- **The members' proof blocks stand side by side**, with no hypothesis: `BlocksClosed` at the members' programs and
units. -/
theorem blocksClosed_bitsProgs (ms : Spot.Bits.Members) :
    _root_.SpotFlock.BlocksClosed (bitsProgs ms) (membersPart ms) := by
  have key : ∀ bl ∈ (membersPart ms).blocks, ∀ i ∈ bl, ∀ x : BGate, (bitsProgs ms).gates[i]? = some x →
      (∀ b, x ≠ .const b) ∧ ∀ w ∈ x.reads, w ∈ bl ∨ ∃ b, (bitsProgs ms).gates[w]? = some (.const b) := by
    intro bl hbl i hi x hx
    obtain ⟨m, hm, u, hu, rfl⟩ := mem_blocks_membersPart ms hbl
    obtain ⟨q, hq, rfl⟩ := List.mem_map.1 hi
    have hql : q < (memberProg ms[m].1 ms[m].2.size).gates.length := lt_length_of_mem_blockOf hq
    rw [gates_bitsProgs ms hm hql] at hx
    obtain ⟨y, hy, rfl⟩ := Option.map_eq_some_iff.1 hx
    obtain ⟨hc, hr⟩ := member_closed _ _ hq hy
    refine ⟨fun b h => hc b (Members.shiftGate_eq_const.1 h), fun w hw => ?_⟩
    rw [Members.reads_shiftGate] at hw
    obtain ⟨w', hw', rfl⟩ := List.mem_map.1 hw
    rcases hr w' hw' with h1 | ⟨b, hb⟩
    · exact Or.inl (List.mem_map.2 ⟨w', h1, rfl⟩)
    · have hwl : w' < (memberProg ms[m].1 ms[m].2.size).gates.length := (List.getElem?_eq_some_iff.1 hb).1
      exact Or.inr ⟨b, by simp [gates_bitsProgs ms hm hwl, hb, shiftGate]⟩
  exact And.intro (fun bl hbl i hi x hx => (key bl hbl i hi x hx).2)
    (fun bl hbl i hi b hb => (key bl hbl i hi _ hb).1 b rfl)

/-- **A gate a closed proof block needs is one of its gates.** -/
theorem mem_of_needs {C : Claims.Sampling.Prog} {Pt : Partition} (hcl : _root_.SpotFlock.BlocksClosed C Pt)
    {bu : List ℕ} (hbu : bu ∈ Pt.blocks) {g : ℕ} (hg : C.Needs bu g) : g ∈ bu := by
  obtain ⟨hc, -⟩ := hcl
  unfold Prog.Needs at hg
  rcases hg with hg | ⟨⟨g', hg', x, hx, hi⟩, hgb, hncg⟩
  · exact hg.1
  · rcases hc bu hbu g' hg' x hx g hi with hin | ⟨c, hcc⟩
    · exact absurd hin hgb
    · exact absurd hcc (hncg c)

/-! ## Source bits and written bits -/

theorem find?_of_all {α : Type} {p : α → Bool} {l : List α} {Q : α → Prop} (hex : ∃ a ∈ l, p a = true)
    (hQ : ∀ b ∈ l, p b = true → Q b) : ∃ a, l.find? p = some a ∧ Q a := by
  cases hf : l.find? p with
  | none =>
    obtain ⟨a, ha, hp⟩ := hex
    exact absurd hp (List.find?_eq_none.1 hf a ha)
  | some b => exact ⟨b, rfl, hQ b (List.mem_of_find?_eq_some hf) (List.find?_some hf)⟩

section bits

variable {ms : Spot.Bits.Members}

theorem of_mem_sourceBits {e : ℕ × ℕ × Spot.Bits.RowPos} (he : e ∈ _root_.SpotFlock.sourceBits ms) :
    ∃ m, ∃ hm : m < ms.length, ∃ g < (Spot.Bits.keyProg ms[m].1 ms[m].2.size).inputs,
      Spot.Bits.consts ms[m].1 ms[m].2.size g = none ∧ ∃ r, Spot.Bits.rowBit ms[m].1 ms[m].2.size g = some r ∧
        e = (Spot.Bits.base ms m + g, m, r) := by
  simp only [_root_.SpotFlock.sourceBits, List.mem_flatMap, List.mem_range, List.mem_filterMap] at he
  obtain ⟨m, hm, g, hg, he⟩ := he
  rw [getD_ms ms hm] at hg he
  refine ⟨m, hm, g, hg, ?_⟩
  cases hκ : Spot.Bits.consts ms[m].1 ms[m].2.size g with
  | some b =>
    rw [hκ] at he
    simp at he
  | none =>
    rw [hκ] at he
    cases hr : Spot.Bits.rowBit ms[m].1 ms[m].2.size g with
    | none =>
      rw [hr] at he
      simp at he
    | some r =>
      rw [hr] at he
      exact ⟨rfl, r, rfl, by simpa using he.symm⟩

theorem mem_sourceBits_of {t g : ℕ} {rp : Spot.Bits.RowPos} (ht : t < ms.length)
    (hg : g < (Spot.Bits.keyProg ms[t].1 ms[t].2.size).inputs) (hκ : Spot.Bits.consts ms[t].1 ms[t].2.size g = none)
    (hr : Spot.Bits.rowBit ms[t].1 ms[t].2.size g = some rp) :
    (Spot.Bits.base ms t + g, t, rp) ∈ _root_.SpotFlock.sourceBits ms := by
  simp only [_root_.SpotFlock.sourceBits, List.mem_flatMap, List.mem_range, List.mem_filterMap]
  refine ⟨t, ht, ?_⟩
  rw [getD_ms ms ht]
  exact ⟨g, hg, by simp [hκ, hr]⟩

theorem of_mem_writtenBits {e : ℕ × ℕ × Spot.Bits.RowPos} (he : e ∈ _root_.SpotFlock.writtenBits ms) :
    ∃ m, ∃ hm : m < ms.length, ∃ u k j t x r, Spot.Bits.outGate ms[m].1 ms[m].2.size u k j t = some x ∧
      Spot.Bits.outRowBit ms[m].1 u k j t = some r ∧
      e = (Spot.Bits.base ms m + Spot.Bits.posOf (Spot.Bits.keyProg ms[m].1 ms[m].2.size) x, m, r) := by
  simp only [_root_.SpotFlock.writtenBits, List.mem_flatMap, List.mem_range, List.mem_filterMap] at he
  obtain ⟨m, hm, u, -, k, -, j, -, t, -, he⟩ := he
  rw [getD_ms ms hm] at he
  refine ⟨m, hm, u, k, j, t, ?_⟩
  cases hx : Spot.Bits.outGate ms[m].1 ms[m].2.size u k j t with
  | none =>
    rw [hx] at he
    simp at he
  | some x =>
    cases hr : Spot.Bits.outRowBit ms[m].1 u k j t with
    | none =>
      rw [hx, hr] at he
      simp at he
    | some r =>
      rw [hx, hr] at he
      exact ⟨x, r, rfl, rfl, by simpa using he.symm⟩

theorem mem_writtenBits_of {m u k j t x : ℕ} {r : Spot.Bits.RowPos} (hm : m < ms.length)
    (hx : Spot.Bits.outGate ms[m].1 ms[m].2.size u k j t = some x) (hr : Spot.Bits.outRowBit ms[m].1 u k j t = some r) :
    (Spot.Bits.base ms m + Spot.Bits.posOf (Spot.Bits.keyProg ms[m].1 ms[m].2.size) x, m, r) ∈
      _root_.SpotFlock.writtenBits ms := by
  obtain ⟨hu, hk, hj, ht, -⟩ := outGate_some _ _ hx
  simp only [_root_.SpotFlock.writtenBits, List.mem_flatMap, List.mem_range, List.mem_filterMap]
  refine ⟨m, hm, ?_⟩
  rw [getD_ms ms hm]
  exact ⟨u, hu, k, hk, j, hj, t, ht, by simp [hx, hr]⟩

/-- **A written bit at a row gate of member `t`'s unit `k`** is member `t`'s, of unit `k`. -/
theorem written_eq {t m g k j u k' j' t' x : ℕ} (ht : t < ms.length) (hm : m < ms.length)
    (hgu : Spot.Bits.unitOf (Spot.Bits.keyProg ms[t].1 ms[t].2.size) g = some k)
    (hj : j < Spot.Bits.gateSize (Spot.Bits.keyProg ms[t].1 ms[t].2.size) g)
    (hx : Spot.Bits.outGate ms[m].1 ms[m].2.size u k' j' t' = some x)
    (h : Spot.Bits.base ms m + Spot.Bits.posOf (Spot.Bits.keyProg ms[m].1 ms[m].2.size) x =
      Spot.Bits.base ms t + (Spot.Bits.off (Spot.Bits.keyProg ms[t].1 ms[t].2.size) g + j)) :
    m = t ∧ u = k := by
  obtain ⟨hu, -, -, -, i, hi, rfl⟩ := outGate_some _ _ hx
  have hxu := unitOf_row _ _ hu hi
  obtain ⟨a, b, hab⟩ := row?_of_unitOf hxu
  have hxs := (unitOf_eq_some hxu).2.1
  have hgs := (unitOf_eq_some hgu).2.1
  have hp : Spot.Bits.posOf (Spot.Bits.keyProg ms[m].1 ms[m].2.size)
      (2 + ms[m].2.size * (bitsRows ms[m].1).nIn + (u * (bitsRows ms[m].1).nComp + i)) <
      (memberProg ms[m].1 ms[m].2.size).gates.length := by
    rw [memberProg_eq, SecurityProofs.bitsProg_length]
    exact posOf_lt_off hxs
  have hq : Spot.Bits.off (Spot.Bits.keyProg ms[t].1 ms[t].2.size) g + j <
      (memberProg ms[t].1 ms[t].2.size).gates.length := by
    rw [memberProg_eq, SecurityProofs.bitsProg_length]
    exact off_add_lt hgs hj
  obtain ⟨rfl, hpq⟩ := pos_eq ms hm ht hp hq h
  rw [posOf_of_some hab] at hpq
  have hxg := eq_of_off hxs hgs (show 2 < Spot.Bits.gateSize _ _ by rw [gateSize_of_some hab]; omega) hj hpq
  rw [hxg, hgu] at hxu
  exact ⟨rfl, (Option.some.inj hxu).symm⟩

/-- **No source bit sits at a row gate.** -/
theorem src_ne_row {t m g k j g' : ℕ} (ht : t < ms.length) (hm : m < ms.length)
    (hgu : Spot.Bits.unitOf (Spot.Bits.keyProg ms[t].1 ms[t].2.size) g = some k)
    (hj : j < Spot.Bits.gateSize (Spot.Bits.keyProg ms[t].1 ms[t].2.size) g)
    (hg' : g' < (Spot.Bits.keyProg ms[m].1 ms[m].2.size).inputs)
    (h : Spot.Bits.base ms m + g' =
      Spot.Bits.base ms t + (Spot.Bits.off (Spot.Bits.keyProg ms[t].1 ms[t].2.size) g + j)) : False := by
  have hgs := unitOf_eq_some hgu
  have hp : g' < (memberProg ms[m].1 ms[m].2.size).gates.length := lt_len_of_input _ hg'
  have hq : Spot.Bits.off (Spot.Bits.keyProg ms[t].1 ms[t].2.size) g + j <
      (memberProg ms[t].1 ms[t].2.size).gates.length := by
    rw [memberProg_eq, SecurityProofs.bitsProg_length]
    exact off_add_lt hgs.2.1 hj
  obtain ⟨rfl, hpq⟩ := pos_eq ms hm ht hp hq h
  have h1 := le_off (kp := Spot.Bits.keyProg ms[m].1 ms[m].2.size) (Nat.le_of_lt hgs.2.1)
  have h2 := hgs.1
  omega

end bits

end Flock.Bridge.NeededBitsOf

namespace SpotFlock

open Properties.Spot (bitsProgs membersPart memberProg unitBase)

/-- **A member's bits are rowed**: each source gate its constants leave free has a commitment position
(`Bits.rowBit`), and so does each returned output leaf bit (`Bits.outRowBit`). -/
def MemberRowed (c : _root_.Flock.Circuit) (n : ℕ) : Prop :=
  (∀ g < (Spot.Bits.keyProg c n).inputs, Spot.Bits.consts c n g = none → (Spot.Bits.rowBit c n g).isSome) ∧
    ∀ u k j t x, Spot.Bits.outGate c n u k j t = some x → (Spot.Bits.outRowBit c u k j t).isSome

/-- **Every member's bits are rowed** (`MemberRowed`). -/
def MembersRowed (h : Held) : Prop := ∀ ci ∈ h.ms, MemberRowed ci.1 ci.2.size

/-- **The code's proof blocks stand side by side**, at every held code. -/
theorem sideBySide_of_held (h : Held) : SideBySide h :=
  _root_.Flock.Bridge.NeededBitsOf.blocksClosed_bitsProgs h.ms

end SpotFlock

namespace Flock.Bridge.NeededBitsOf

open Claims.Sampling Properties.Spot

theorem neededBits_of_membersRowed {h : _root_.SpotFlock.Held} (hR : _root_.SpotFlock.MembersRowed h) :
    _root_.SpotFlock.NeededBits h := by
  unfold _root_.SpotFlock.NeededBits
  intro r hr i hi
  obtain ⟨Pt, d, spec, hx, hP, -, hin, -, -, -, hb, -⟩ := hr
  have hprog : r.s.prog = bitsProgs h.ms := _root_.SpotFlock.parse_prog h hx
  have hPt' : some (membersPart h.ms) = some Pt := hP
  have hPt : membersPart h.ms = Pt := Option.some.inj hPt'
  subst hPt
  have hpart : (membersPart h.ms).Partitions (bitsProgs h.ms) := by
    rw [← hprog]
    exact hin.2.1
  rw [hprog] at hi
  have hcl := blocksClosed_bitsProgs h.ms
  have hcom := _root_.SpotFlock.needs_committed_of_closed hpart hcl hb hi
  have hib := mem_of_needs hcl (List.mem_of_getElem? hb) hi
  have hul : r.u < unitBase h.ms h.ms.length := by
    rw [← Members.length_blocks_membersPart]
    exact (List.getElem?_eq_some_iff.1 hb).1
  obtain ⟨t, ht, k, hk, hu⟩ := exists_member h.ms hul
  rw [hu, Members.getElem?_blocks_membersPart h.ms ht hk] at hb
  rw [← Option.some.inj hb] at hib
  rw [hu, _root_.SpotFlock.templateOf_unitBase ht hk, Nat.add_sub_cancel_left]
  obtain ⟨q, hq, rfl⟩ := List.mem_map.1 hib
  obtain ⟨hR1, -⟩ := hR _ (List.getElem_mem ht)
  unfold Spot.Bits.blockOf at hq
  rcases List.mem_append.1 hq with hq | hq
  · obtain ⟨g, hgs, hgu, j, hj, rfl⟩ := mem_unitGates hq
    obtain ⟨a, b, hab⟩ := row?_of_unitOf hgu
    have hlen : Spot.Bits.off (Spot.Bits.keyProg h.ms[t].1 h.ms[t].2.size) g + j <
        (memberProg h.ms[t].1 h.ms[t].2.size).gates.length := by
      rw [memberProg_eq, SecurityProofs.bitsProg_length]
      exact off_add_lt hgs hj
    have hgate : (bitsProgs h.ms).gates[Spot.Bits.base h.ms t +
        (Spot.Bits.off (Spot.Bits.keyProg h.ms[t].1 h.ms[t].2.size) g + j)]? ≠ some .input := by
      rw [gates_bitsProgs h.ms ht hlen]
      intro hc
      obtain ⟨y, hy, hy'⟩ := Option.map_eq_some_iff.1 hc
      rw [Members.shiftGate_eq_input] at hy'
      subst hy'
      exact row_gate_not_input _ hgs hab hj hy
    have hout := hcom.resolve_right hgate
    have hall : ∀ e ∈ _root_.SpotFlock.writtenBits h.ms,
        (e.1 == Spot.Bits.base h.ms t + (Spot.Bits.off (Spot.Bits.keyProg h.ms[t].1 h.ms[t].2.size) g + j)) =
          true →
        e.1 = Spot.Bits.base h.ms t + (Spot.Bits.off (Spot.Bits.keyProg h.ms[t].1 h.ms[t].2.size) g + j) ∧
          e.2.1 = t ∧ e.2.2.leaf = k := by
      intro e he hpe
      obtain ⟨m, hm, u, k', j', t', x, r', hx', hr', rfl⟩ := of_mem_writtenBits he
      simp only [beq_iff_eq] at hpe
      obtain ⟨rfl, rfl⟩ := written_eq ht hm hgu hj hx' hpe
      exact ⟨hpe, rfl, outRowBit_leaf hr'⟩
    have hex : ∃ e ∈ _root_.SpotFlock.writtenBits h.ms,
        (e.1 == Spot.Bits.base h.ms t + (Spot.Bits.off (Spot.Bits.keyProg h.ms[t].1 h.ms[t].2.size) g + j)) =
          true := by
      obtain ⟨m, hm, p, hp, hpe⟩ := exists_of_mem_outs_bitsProgs h.ms hout
      rw [memberProg_outs] at hp
      obtain ⟨o, ho, rfl⟩ := List.mem_map.1 hp
      obtain ⟨u, k', j', t', hx'⟩ := of_mem_outs _ _ ho
      obtain ⟨-, hR2⟩ := hR _ (List.getElem_mem hm)
      obtain ⟨r', hr'⟩ := Option.isSome_iff_exists.1 (hR2 _ _ _ _ _ hx')
      exact ⟨_, mem_writtenBits_of hm hx' hr', by simp [hpe]⟩
    obtain ⟨e, hfe, he1, he2, he3⟩ := find?_of_all hex hall
    refine ⟨e.2.2, ?_⟩
    unfold _root_.SpotFlock.BitOf
    refine ⟨he3, Or.inr ⟨?_, ?_, hout, hgate⟩⟩
    · rw [List.find?_eq_none]
      intro e' he' hpe'
      obtain ⟨m, hm, g', hg', hκ', r', hr', hee⟩ := of_mem_sourceBits he'
      rw [hee] at hpe'
      exact src_ne_row ht hm hgu hj hg' (beq_iff_eq.1 hpe')
    · rw [hfe]
      exact congrArg some (Prod.ext he1 (Prod.ext he2 rfl))
  · obtain ⟨g, hg, rfl⟩ := List.mem_map.1 hq
    obtain ⟨hgs, hrn, hκ, hown⟩ := mem_insOf.1 hg
    have hgi := lt_inputs_of_row_none hgs hrn
    rw [posOf_of_none hrn, off_eq_of_le_inputs (Nat.le_of_lt hgi)]
    obtain ⟨hh, hhu, hgh⟩ := exists_of_owner hown
    obtain ⟨bb, hbB, rfl⟩ : ∃ bb < (bitsRows h.ms[t].1).nIn, g = 2 + k * (bitsRows h.ms[t].1).nIn + bb := by
      rcases (args_of_unit _ _ hhu hgh).2 with rfl | rfl | ⟨col, b, hb', rfl⟩ | ⟨i0, hi0, rfl⟩
      · rw [consts_zero] at hκ
        cases hκ
      · rw [consts_one] at hκ
        cases hκ
      · exact ⟨b, keyIdx_lt hb', rfl⟩
      · rw [keyProg_inputs] at hgi
        omega
    obtain ⟨rp, hrp⟩ := Option.isSome_iff_exists.1 (hR1 _ hgi hκ)
    refine ⟨rp, ?_⟩
    unfold _root_.SpotFlock.BitOf
    refine ⟨?_, Or.inl ⟨?_, ?_⟩⟩
    · rw [rowBit_leaf hrp,
        show 2 + k * (bitsRows h.ms[t].1).nIn + bb - 2 = k * (bitsRows h.ms[t].1).nIn + bb by omega, div_units hbB]
    · have hall : ∀ e ∈ _root_.SpotFlock.sourceBits h.ms,
          (e.1 == Spot.Bits.base h.ms t + (2 + k * (bitsRows h.ms[t].1).nIn + bb)) = true →
          e = (Spot.Bits.base h.ms t + (2 + k * (bitsRows h.ms[t].1).nIn + bb), t, rp) := by
        intro e he hpe
        obtain ⟨m, hm, g', hg', -, r', hr', rfl⟩ := of_mem_sourceBits he
        simp only [beq_iff_eq] at hpe
        obtain ⟨rfl, rfl⟩ := pos_eq h.ms hm ht (lt_len_of_input _ hg') (lt_len_of_input _ hgi) hpe
        rw [hr'] at hrp
        cases hrp
        rfl
      obtain ⟨e, hfe, rfl⟩ := find?_of_all ⟨_, mem_sourceBits_of ht hgi hκ hrp, by simp⟩ hall
      exact hfe
    · have hlen := lt_len_of_input (Spot.Bits.consts h.ms[t].1 h.ms[t].2.size) hgi
      simp only [gates_bitsProgs h.ms ht hlen, memberProg_eq, gate_input_free _ hgi hκ, Option.map_some, shiftGate]

end Flock.Bridge.NeededBitsOf

namespace SpotFlock

/-- **`NeededBits` where every member's bits are rowed.** -/
theorem neededBits_of_rowed {h : Held} (hR : MembersRowed h) : NeededBits h :=
  _root_.Flock.Bridge.NeededBitsOf.neededBits_of_membersRowed hR

end SpotFlock
