module

public import Proofs.Flock.SpotFlock.WireMapRewires

/-!
# Source bits and written bits, one at a position

The wiring map (`wireTarget`, `WireMap`) finds a source bit by its position in the members' programs and its writer's
gate by the member and the commitment position. This file proves what those two searches return, from the layout of
a key program alone (`Spot.Bits.outGate`, `rowBit`, `outRowBit`), with no hypothesis on a member.

* `outGate_spec`, `outGate_inj`, `outGate_unit`: a gate on a returned output's bit is a row of its unit's instance;
  two bits of a key program on one gate are one bit; and a bit that has a gate in one unit has one in every unit.
* `posOf_row`, `outGate_row`: a row's position in the lowered program, three gates a row after the inputs, and a
  gate on a returned output's bit is a row.
* `outRowBit_leaf`: a returned output's bit is committed at the same row and bit in every unit, at the unit's leaf.
* `mem_sourceBits_rowBit`, `sourceBits_inj`, `find?_sourceBits`: a source bit is its member's source gate with its
  commitment position, the one source bit at its position, which the search by position returns.
* `mem_writtenBits_iff`, `writtenBits_inj`: a written bit is its member's gate on a returned output's bit with its
  commitment position, and the one written bit at its position: no gate is written at two commitment positions.
* `rowBit_spec`: a source bit's commitment position is a unit's leaf of a row port of the circuit file, at a bit
  under the port's `bits`.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling
open Properties.Spot (bitsProgs membersPart memberProg)
open Flock.Bridge.Members

/-! ## A key program's gates on returned outputs' bits -/

/-- **Rows of two units, instances or columns are two gates**: `I` input gates, then `K·(M + 1)` rows a unit,
`M + 1` an instance, the instance's constant row first. -/
theorem rowGate_inj {I K M u u' k k' e e' : ℕ} (hk : k < K) (hk' : k' < K) (he : e < M) (he' : e' < M)
    (h : I + u * (K * (M + 1)) + (k * (M + 1) + (1 + e)) = I + u' * (K * (M + 1)) + (k' * (M + 1) + (1 + e'))) :
    u = u' ∧ k = k' ∧ e = e' := by
  have hK := Nat.mul_le_mul_right (M + 1) (show k + 1 ≤ K from hk)
  have hK' := Nat.mul_le_mul_right (M + 1) (show k' + 1 ≤ K from hk')
  rw [Nat.add_one_mul] at hK hK'
  obtain ⟨hu, hin⟩ := FlockSoundness.Discharge.Inst.mul_add_inj (k := K * (M + 1)) (a := u) (a' := u')
    (b := k * (M + 1) + (1 + e)) (b' := k' * (M + 1) + (1 + e')) (by omega) (by omega) (by omega)
  obtain ⟨hkk, hee⟩ := FlockSoundness.Discharge.Inst.mul_add_inj (k := M + 1) (a := k) (a' := k') (b := 1 + e)
    (b' := 1 + e') (by omega) (by omega) hin
  exact ⟨hu, hkk, by omega⟩

/-- **What a gate on a returned output's bit is**: bit `t < 16` of leaf `j` of unit `u`'s instance `k`, its column
(`Bits.outCol`) a computed row of the unit net, the gate that row of the instance in the unit's rows. -/
theorem outGate_spec {c : _root_.Flock.Circuit} {n u k j t a : ℕ} (h : Spot.Bits.outGate c n u k j t = some a) :
    u < n ∧ k < c.upv ∧ j < (c.leavesOut.getD k #[]).size ∧ t < 16 ∧
      (Spot.Bits.Rows.ofNet c.unit).nIn ≤ Spot.Bits.outCol c j t ∧
      Spot.Bits.outCol c j t < (Spot.Bits.Rows.ofNet c.unit).nIn + (Spot.Bits.Rows.ofNet c.unit).nComp ∧
      a = 2 + n * (c.upv * (Spot.Bits.Rows.ofNet c.unit).nIn) +
        u * (c.upv * ((Spot.Bits.Rows.ofNet c.unit).nComp + 1)) +
        (k * ((Spot.Bits.Rows.ofNet c.unit).nComp + 1) +
          (1 + (Spot.Bits.outCol c j t - (Spot.Bits.Rows.ofNet c.unit).nIn))) := by
  simp only [Spot.Bits.outGate, Spot.Bits.rowGate] at h
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, decide_eq_true_eq, and_assoc] at hc
    obtain ⟨hu, hk, hj, ht, h1, h2⟩ := hc
    exact ⟨hu, hk, hj, ht, h1, h2, (Option.some.inj h).symm⟩
  · simp at h

/-- **Two bits of a key program on one gate are one bit**: the unit, the instance, the leaf and the bit. -/
theorem outGate_inj {c : _root_.Flock.Circuit} {n u u' k k' j j' t t' a : ℕ}
    (h : Spot.Bits.outGate c n u k j t = some a) (h' : Spot.Bits.outGate c n u' k' j' t' = some a) :
    u = u' ∧ k = k' ∧ j = j' ∧ t = t' := by
  obtain ⟨-, hk, -, ht, h1, h2, ha⟩ := outGate_spec h
  obtain ⟨-, hk', -, ht', h1', h2', ha'⟩ := outGate_spec h'
  obtain ⟨hu, hkk, he⟩ := rowGate_inj hk hk' (show Spot.Bits.outCol c j t - (Spot.Bits.Rows.ofNet c.unit).nIn <
    (Spot.Bits.Rows.ofNet c.unit).nComp by omega) (show Spot.Bits.outCol c j' t' - (Spot.Bits.Rows.ofNet c.unit).nIn <
    (Spot.Bits.Rows.ofNet c.unit).nComp by omega) (ha.symm.trans ha')
  have hoc : Spot.Bits.outCol c j t = Spot.Bits.outCol c j' t' := by omega
  unfold Spot.Bits.outCol at hoc
  exact ⟨hu, hkk, by omega, by omega⟩

/-- **A bit with a gate in one unit has one in every unit**: the conditions on the instance, the leaf, the bit and
its column are the same in each. -/
theorem outGate_unit {c : _root_.Flock.Circuit} {n u k j t : ℕ} (h : (Spot.Bits.outGate c 1 0 k j t).isSome)
    (hu : u < n) : ∃ a, Spot.Bits.outGate c n u k j t = some a := by
  obtain ⟨a0, ha0⟩ := Option.isSome_iff_exists.1 h
  obtain ⟨-, hk, hj, ht, h1, h2, -⟩ := outGate_spec ha0
  have hc : (decide (u < n) && decide (k < c.upv) && decide (j < (c.leavesOut.getD k #[]).size) && decide (t < 16) &&
      decide ((Spot.Bits.Rows.ofNet c.unit).nIn ≤ Spot.Bits.outCol c j t) &&
      decide (Spot.Bits.outCol c j t <
        (Spot.Bits.Rows.ofNet c.unit).nIn + (Spot.Bits.Rows.ofNet c.unit).nComp)) = true := by
    simp only [Bool.and_eq_true, decide_eq_true_eq]
    exact ⟨⟨⟨⟨⟨hu, hk⟩, hj⟩, ht⟩, h1⟩, h2⟩
  refine ⟨Spot.Bits.rowGate (Spot.Bits.Rows.ofNet c.unit)
    (2 + n * ((Spot.Bits.Rows.ofNet c.unit).stack c.upv).nIn) ((Spot.Bits.Rows.ofNet c.unit).stack c.upv).nComp u k
    (Spot.Bits.outCol c j t), ?_⟩
  unfold Spot.Bits.outGate
  exact ite_eq_left hc

/-- **A row's position in the lowered program**: the inputs, one gate each, then three gates a row, the row's own
last. -/
theorem posOf_row (kp : Spot.Bits.KeyProg) {i : ℕ} (hi : i < kp.rows.size) :
    Spot.Bits.posOf kp (kp.inputs + i) = kp.inputs + 3 * i + 2 := by
  have hsz : kp.inputs + i ≤ kp.size := by
    unfold Spot.Bits.KeyProg.size
    omega
  have h1 : Spot.Bits.off kp (kp.inputs + i) = kp.inputs + 3 * i :=
    (Properties.Spot.SecurityProofs.off_of_le kp hsz).trans (sum_gateSize kp i (Nat.le_of_lt hi))
  have h2 : Spot.Bits.gateSize kp (kp.inputs + i) = 3 := gateSize_of_row hi
  unfold Spot.Bits.posOf
  omega

/-- **A gate on a returned output's bit is a row** of the key program. -/
theorem outGate_row {c : _root_.Flock.Circuit} {n u k j t a : ℕ} (h : Spot.Bits.outGate c n u k j t = some a) :
    ∃ i, i < (Spot.Bits.keyProg c n).rows.size ∧ a = (Spot.Bits.keyProg c n).inputs + i := by
  obtain ⟨hu, hk, hj, ht, -⟩ := outGate_spec h
  have hrow := Flock.Bridge.isSome_row?_of_mem_outs (outGate_mem_outs hu hk hj ht h)
  have hlt := lt_size_of_isSome_row? hrow
  unfold Spot.Bits.KeyProg.size at hlt
  rcases Nat.lt_or_ge a (Spot.Bits.keyProg c n).inputs with hin | hge
  · rw [row?_of_lt_inputs hin] at hrow
    simp at hrow
  · exact ⟨a - (Spot.Bits.keyProg c n).inputs, by omega, by omega⟩

/-- A member's program's gates: its key program's inputs, and three gates a row. -/
theorem length_memberProg_rows (c : _root_.Flock.Circuit) (n : ℕ) :
    (memberProg c n).gates.length = (Spot.Bits.keyProg c n).inputs + 3 * (Spot.Bits.keyProg c n).rows.size := by
  have hlen := length_memberProg c n
  have hin := Flock.Bridge.keyProg_inputs c n
  have hrs := keyProg_rows_size c n
  rw [nIn_bitsRows] at hin
  rw [nComp_bitsRows] at hrs
  unfold memberSize at hlen
  omega

/-- **A returned output's bit is committed at the same row and bit in every unit**, at the unit's leaf. -/
theorem outRowBit_leaf (c : _root_.Flock.Circuit) (u k j t : ℕ) :
    Spot.Bits.outRowBit c u k j t =
      (Spot.Bits.outRowBit c 0 k j t).map fun r => Spot.Bits.RowPos.mk r.row u r.bit := by
  unfold Spot.Bits.outRowBit
  cases (c.leavesOut.getD k #[])[j]? with
  | none => rfl
  | some x =>
    show Spot.Bits.outPos c u x t = (Spot.Bits.outPos c 0 x t).map fun r => Spot.Bits.RowPos.mk r.row u r.bit
    unfold Spot.Bits.outPos
    rw [Option.map_map]
    rfl

/-! ## Source bits -/

/-- **A source bit is its member's source gate with its commitment position** (`Bits.rowBit`), at the member's
positions. -/
theorem mem_sourceBits_rowBit {ms : Spot.Bits.Members} {x t : ℕ} {r : Spot.Bits.RowPos}
    (hs : (x, t, r) ∈ sourceBits ms) :
    ∃ ht : t < ms.length, ∃ g, g < (Spot.Bits.keyProg ms[t].1 ms[t].2.size).inputs ∧
      Spot.Bits.rowBit ms[t].1 ms[t].2.size g = some r ∧ x = Spot.Bits.base ms t + g := by
  unfold sourceBits at hs
  obtain ⟨m, hm, hx⟩ := List.mem_flatMap.1 hs
  have hm' := List.mem_range.1 hm
  obtain ⟨g, hg, hgx⟩ := List.mem_filterMap.1 hx
  have hci : ms.getD m default = ms[m] := by
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hm', Option.getD_some]
  rw [hci] at hg hgx
  cases hκ : Spot.Bits.consts ms[m].1 ms[m].2.size g with
  | some b =>
    rw [hκ] at hgx
    simp at hgx
  | none =>
    rw [hκ] at hgx
    have hgx' : (Spot.Bits.rowBit ms[m].1 ms[m].2.size g).map (fun r => (Spot.Bits.base ms m + g, m, r)) =
        some (x, t, r) := hgx
    obtain ⟨r', hr', he⟩ := Option.map_eq_some_iff.1 hgx'
    have h1 : Spot.Bits.base ms m + g = x := congrArg Prod.fst he
    have h2 : m = t := congrArg (fun p => p.2.1) he
    have h3 : r' = r := congrArg (fun p => p.2.2) he
    subst h2
    subst h3
    exact ⟨hm', g, List.mem_range.1 hg, hr', h1.symm⟩

/-- **One source bit at a position**: two source bits at one position of the members' programs are one member's,
at one commitment position. -/
theorem sourceBits_inj {ms : Spot.Bits.Members} {x t t' : ℕ} {r r' : Spot.Bits.RowPos}
    (hs : (x, t, r) ∈ sourceBits ms) (hs' : (x, t', r') ∈ sourceBits ms) : t = t' ∧ r = r' := by
  obtain ⟨ht, g, hg, hr, hx⟩ := mem_sourceBits_rowBit hs
  obtain ⟨ht', g', hg', hr', hx'⟩ := mem_sourceBits_rowBit hs'
  have hlen := length_memberProg_rows ms[t].1 ms[t].2.size
  have hlen' := length_memberProg_rows ms[t'].1 ms[t'].2.size
  have hm := gateMember_base_add ms ht (p := g) (by omega)
  have hm' := gateMember_base_add ms ht' (p := g') (by omega)
  rw [← hx] at hm
  rw [← hx'] at hm'
  have htt : t = t' := hm.symm.trans hm'
  subst htt
  have hgg : g = g' := Nat.add_left_cancel (hx.symm.trans hx')
  subst hgg
  exact ⟨rfl, Option.some.inj (hr.symm.trans hr')⟩

/-- **The search by position returns the source bit there.** -/
theorem find?_sourceBits {ms : Spot.Bits.Members} {x t : ℕ} {r : Spot.Bits.RowPos}
    (hs : (x, t, r) ∈ sourceBits ms) : (sourceBits ms).find? (·.1 == x) = some (x, t, r) := by
  cases hf : (sourceBits ms).find? (·.1 == x) with
  | none =>
    have hno : ¬ (x == x) = true := List.find?_eq_none.1 hf (x, t, r) hs
    exact absurd (beq_self_eq_true x) hno
  | some s =>
    obtain ⟨sx, st, sr⟩ := s
    have hsp := List.find?_some hf
    have hsx : sx = x := eq_of_beq hsp
    have hm : (sx, st, sr) ∈ sourceBits ms := List.mem_of_find?_eq_some hf
    subst hsx
    obtain ⟨h1, h2⟩ := sourceBits_inj hm hs
    rw [h1, h2]

/-! ## Written bits -/

/-- **A written bit is its member's gate on a returned output's bit with its commitment position**: bit `b` of leaf
`j` of unit `u`'s instance `k`, where the key program has a gate there (`Bits.outGate`) and the bit a commitment
position (`Bits.outRowBit`), at the gate's position among the member's. -/
theorem mem_writtenBits_iff {ms : Spot.Bits.Members} {y tw : ℕ} {r : Spot.Bits.RowPos} :
    (y, tw, r) ∈ writtenBits ms ↔ ∃ ht : tw < ms.length, ∃ u k j b a, u < ms[tw].2.size ∧ k < ms[tw].1.upv ∧
      j < (ms[tw].1.leavesOut.getD k #[]).size ∧ b < 16 ∧
      Spot.Bits.outGate ms[tw].1 ms[tw].2.size u k j b = some a ∧ Spot.Bits.outRowBit ms[tw].1 u k j b = some r ∧
      y = Spot.Bits.base ms tw + Spot.Bits.posOf (Spot.Bits.keyProg ms[tw].1 ms[tw].2.size) a := by
  constructor
  · intro hs
    unfold writtenBits at hs
    obtain ⟨m, hm, hx⟩ := List.mem_flatMap.1 hs
    have hm' := List.mem_range.1 hm
    obtain ⟨u, hu, hx⟩ := List.mem_flatMap.1 hx
    obtain ⟨k, hk, hx⟩ := List.mem_flatMap.1 hx
    obtain ⟨j, hj, hx⟩ := List.mem_flatMap.1 hx
    obtain ⟨b, hb, hx⟩ := List.mem_filterMap.1 hx
    have hci : ms.getD m default = ms[m] := by
      rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hm', Option.getD_some]
    rw [hci] at hu hk hj hx
    cases hg : Spot.Bits.outGate ms[m].1 ms[m].2.size u k j b with
    | none =>
      rw [hg] at hx
      cases hr : Spot.Bits.outRowBit ms[m].1 u k j b with
      | none =>
        rw [hr] at hx
        simp at hx
      | some r' =>
        rw [hr] at hx
        simp at hx
    | some a =>
      rw [hg] at hx
      cases hr : Spot.Bits.outRowBit ms[m].1 u k j b with
      | none =>
        rw [hr] at hx
        simp at hx
      | some r' =>
        rw [hr] at hx
        have he : (Spot.Bits.base ms m + Spot.Bits.posOf (Spot.Bits.keyProg ms[m].1 ms[m].2.size) a, m, r') =
            (y, tw, r) := Option.some.inj hx
        have h1 : Spot.Bits.base ms m + Spot.Bits.posOf (Spot.Bits.keyProg ms[m].1 ms[m].2.size) a = y :=
          congrArg Prod.fst he
        have h2 : m = tw := congrArg (fun p => p.2.1) he
        have h3 : r' = r := congrArg (fun p => p.2.2) he
        subst h2
        subst h3
        exact ⟨hm', u, k, j, b, a, List.mem_range.1 hu, List.mem_range.1 hk, List.mem_range.1 hj,
          List.mem_range.1 hb, hg, hr, h1.symm⟩
  · rintro ⟨ht, u, k, j, b, a, hu, hk, hj, hb, hg, hr, hy⟩
    subst hy
    have hci : ms.getD tw default = ms[tw] := by
      rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem ht, Option.getD_some]
    unfold writtenBits
    refine List.mem_flatMap.2 ⟨tw, List.mem_range.2 ht, ?_⟩
    rw [hci]
    refine List.mem_flatMap.2 ⟨u, List.mem_range.2 hu, ?_⟩
    refine List.mem_flatMap.2 ⟨k, List.mem_range.2 hk, ?_⟩
    refine List.mem_flatMap.2 ⟨j, List.mem_range.2 hj, ?_⟩
    refine List.mem_filterMap.2 ⟨b, List.mem_range.2 hb, ?_⟩
    rw [hg, hr]
    rfl

/-- **One written bit at a position**: two written bits at one position of the members' programs are one member's,
at one commitment position. No gate is written at two commitment positions. -/
theorem writtenBits_inj {ms : Spot.Bits.Members} {y t t' : ℕ} {r r' : Spot.Bits.RowPos}
    (hs : (y, t, r) ∈ writtenBits ms) (hs' : (y, t', r') ∈ writtenBits ms) : t = t' ∧ r = r' := by
  obtain ⟨ht, u, k, j, b, a, -, -, -, -, hg, hr, hy⟩ := mem_writtenBits_iff.1 hs
  obtain ⟨ht', u', k', j', b', a', -, -, -, -, hg', hr', hy'⟩ := mem_writtenBits_iff.1 hs'
  obtain ⟨i, hi, ha⟩ := outGate_row hg
  obtain ⟨i', hi', ha'⟩ := outGate_row hg'
  have hp := posOf_row (Spot.Bits.keyProg ms[t].1 ms[t].2.size) hi
  have hp' := posOf_row (Spot.Bits.keyProg ms[t'].1 ms[t'].2.size) hi'
  rw [← ha] at hp
  rw [← ha'] at hp'
  have hlen := length_memberProg_rows ms[t].1 ms[t].2.size
  have hlen' := length_memberProg_rows ms[t'].1 ms[t'].2.size
  have hm := gateMember_base_add ms ht (p := Spot.Bits.posOf (Spot.Bits.keyProg ms[t].1 ms[t].2.size) a) (by omega)
  have hm' := gateMember_base_add ms ht' (p := Spot.Bits.posOf (Spot.Bits.keyProg ms[t'].1 ms[t'].2.size) a')
    (by omega)
  rw [← hy] at hm
  rw [← hy'] at hm'
  have htt : t = t' := hm.symm.trans hm'
  subst htt
  have hpp : Spot.Bits.posOf (Spot.Bits.keyProg ms[t].1 ms[t].2.size) a =
      Spot.Bits.posOf (Spot.Bits.keyProg ms[t].1 ms[t].2.size) a' := Nat.add_left_cancel (hy.symm.trans hy')
  have haa : a = a' := by omega
  subst haa
  obtain ⟨rfl, rfl, rfl, rfl⟩ := outGate_inj hg hg'
  exact ⟨rfl, Option.some.inj (hr.symm.trans hr')⟩

/-! ## A source bit's commitment position -/

theorem msgAt_bits {c : _root_.Flock.Circuit} {ix : Spot.Bits.Index} {x t : ℕ} {wide : Bool} {q byte bit : ℕ}
    (h : Spot.Bits.msgAt c ix x t wide = .msg q byte bit) : 8 * byte + bit < (c.ports.getD q default).bits := by
  unfold Spot.Bits.msgAt at h
  dsimp only at h
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
    injection h with h1 h2 h3
    subst h1
    subst h2
    subst h3
    omega
  · cases h

theorem srcAt_bits {c : _root_.Flock.Circuit} {ix : Spot.Bits.Index} {k p t q byte bit : ℕ}
    (h : Spot.Bits.srcAt c ix k p t = .msg q byte bit) : 8 * byte + bit < (c.ports.getD q default).bits := by
  unfold Spot.Bits.srcAt at h
  split at h
  · dsimp only at h
    split at h
    · cases h
    · exact msgAt_bits h
  · split at h
    · exact msgAt_bits h
    · split at h
      · cases h
      · cases h

/-- **A source that is a row port's message bit is a bit of the port**: under its `bits`. -/
theorem srcKey_bits {c : _root_.Flock.Circuit} {ix : Spot.Bits.Index} {x q byte bit : ℕ}
    (h : Spot.Bits.srcKey c ix x = .msg q byte bit) : 8 * byte + bit < (c.ports.getD q default).bits := by
  unfold Spot.Bits.srcKey at h
  dsimp only at h
  split at h
  · exact srcAt_bits h
  · cases h

/-- **A source bit's commitment position**: a unit's leaf, the unit under the units, of the row a port of the
circuit file names, at a bit under the port's `bits`. -/
theorem rowBit_spec {c : _root_.Flock.Circuit} {n g : ℕ} {r : Spot.Bits.RowPos}
    (h : Spot.Bits.rowBit c n g = some r) :
    r.leaf < n ∧ ∃ q, r.row = (c.ports.getD q default).name ∧ r.bit < (c.ports.getD q default).bits := by
  unfold Spot.Bits.rowBit at h
  dsimp only at h
  split at h
  · cases h
  · rename_i hc
    have hc' := Bool.eq_false_iff.2 hc
    simp only [Bool.or_eq_false_iff, decide_eq_false_iff_not, beq_eq_false_iff_ne, Nat.not_lt, Nat.not_le] at hc'
    obtain ⟨⟨⟨h2, hB⟩, hn⟩, -⟩ := hc'
    generalize hk : Spot.Bits.srcKey _ _ _ = sk at h
    cases sk with
    | msg q byte bit =>
      have hr : Spot.Bits.RowPos.mk (c.ports.getD q default).name
          ((g - 2) / ((Spot.Bits.Rows.ofNet c.unit).stack c.upv).nIn) (8 * byte + bit) = r := Option.some.inj h
      subst hr
      exact ⟨(Nat.div_lt_iff_lt_mul (Nat.pos_of_ne_zero hB)).2 (by omega), q, rfl, srcKey_bits hk⟩
    | zero => simp [Spot.Bits.srcPos] at h
    | wire net slot col => simp [Spot.Bits.srcPos] at h
    | lit col => simp [Spot.Bits.srcPos] at h
    | pad k off => simp [Spot.Bits.srcPos] at h

/-- A row port with a bit is a port of the circuit file: past its ports there is none. -/
theorem port_mem_of_bits {c : _root_.Flock.Circuit} {q β : ℕ} (h : β < (c.ports.getD q default).bits) :
    c.ports.getD q default ∈ c.ports := by
  rcases Nat.lt_or_ge q c.ports.size with hq | hq
  · rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hq, Option.getD_some]
    exact Array.getElem_mem hq
  · rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none hq, Option.getD_none] at h
    exact absurd h (Nat.not_lt_zero β)

end SpotFlock
