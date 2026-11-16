module

public import Properties.NetworkScan.Members
public import Proofs.Flock.Bridge.Members

/-!
# The scan's layout fits its members side by side

Of a member's own program (`Properties.Spot.memberProg c n`, the key program `Spot.Bits.keyProg c n` lowered at its
constants):

* `outBit_spec`: a returned output leaf's bit (`Spot.Bits.outBit`) is a gate the program returns, and an AND: it is
  the last of its copy row's three lowered gates (`outGate_row`), and the key program's outputs list every copy row
  (`outGate_mem_outs`).
* `memberProg_inGate`: an input port's bit (`Spot.Bits.inGate`) is an input gate wherever a unit of the program reads
  its source gate (`Spot.Bits.owner`): a source gate is past the constant and the zero, below the rows, and the
  constants (`Spot.Bits.consts`) fix it only when no unit reads it. It is its own position (`posOf_input`).

At SPOT[Flock]'s statement, `Properties.Spot.bitsProgs ms`, the members sit side by side from `Spot.Bits.base ms m`,
and three facts of it, proofs' area's, are taken as hypotheses: an output bit's position is a returned AND
(`OutBitAt`), a read input bit's is an input (`InBitAt`), and a member's gate depends only on its member's
(`DependsMember`). Under them (`membersLayout_fits`), where the members name every position the layout reads: every
egress gate is a gate the egress member returns, an AND, and no input; every ingress gate is one of the ingress
member's input gates; and no egress gate depends on an ingress gate of any step, as the ingress member's gates are all
below the egress member's first. `scanScopeAccMembers_of`: the restated scope.
-/

@[expose] public section

namespace Properties.NetworkScan.MembersProof

open NetworkCheck.Period

theorem sum_range_const (c : ℕ) : ∀ m, ((List.range m).map fun _ => c).sum = m * c
  | 0 => by simp
  | m + 1 => by
    rw [List.range_succ, List.map_append, List.sum_append, sum_range_const c m]
    simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero, Nat.add_one_mul]

theorem keyProg_inputs (c : Flock.Circuit) (n : ℕ) :
    (Spot.Bits.keyProg c n).inputs = 2 + n * ((Spot.Bits.Rows.ofNet c.unit).stack c.upv).nIn := rfl

theorem keyProg_rows_size (c : Flock.Circuit) (n : ℕ) :
    (Spot.Bits.keyProg c n).rows.size = n * ((Spot.Bits.Rows.ofNet c.unit).stack c.upv).nComp := by
  simp only [Spot.Bits.keyProg, List.size_toArray, List.length_flatMap, List.length_map, List.length_range]
  exact sum_range_const _ n

/-! ## A returned output leaf's bit -/

theorem outGate_spec {c : Flock.Circuit} {n u k j t g : ℕ} (h : Spot.Bits.outGate c n u k j t = some g) :
    u < n ∧ k < c.upv ∧ j < (c.leavesOut.getD k #[]).size ∧ t < 16 ∧
      (Spot.Bits.Rows.ofNet c.unit).nIn ≤ Spot.Bits.outCol c j t ∧
      Spot.Bits.outCol c j t < (Spot.Bits.Rows.ofNet c.unit).nIn + (Spot.Bits.Rows.ofNet c.unit).nComp ∧
      g = Spot.Bits.rowGate (Spot.Bits.Rows.ofNet c.unit) (2 + n * ((Spot.Bits.Rows.ofNet c.unit).stack c.upv).nIn)
        ((Spot.Bits.Rows.ofNet c.unit).stack c.upv).nComp u k (Spot.Bits.outCol c j t) := by
  unfold Spot.Bits.outGate at h
  try simp only at h
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, decide_eq_true_eq, and_assoc] at hc
    obtain ⟨h1, h2, h3, h4, h5, h6⟩ := hc
    exact ⟨h1, h2, h3, h4, h5, h6, (Option.some.inj h).symm⟩
  · cases h

/-- **The key program's outputs list every copy row.** -/
theorem outGate_mem_outs {c : Flock.Circuit} {n u k j t g : ℕ} (h : Spot.Bits.outGate c n u k j t = some g) :
    g ∈ (Spot.Bits.keyProg c n).outs := by
  obtain ⟨hu, hk, hj, ht, -⟩ := outGate_spec h
  simp only [Spot.Bits.keyProg, List.mem_eraseDups, List.mem_mergeSort, List.mem_flatMap, List.mem_range,
    List.mem_filterMap]
  exact ⟨u, hu, k, hk, j, hj, t, ht, h⟩

/-- **A copy row is a row of the key program.** -/
theorem outGate_row {c : Flock.Circuit} {n u k j t g : ℕ} (h : Spot.Bits.outGate c n u k j t = some g) :
    g < (Spot.Bits.keyProg c n).size ∧ ∃ ab, (Spot.Bits.keyProg c n).row? g = some ab := by
  obtain ⟨hu, hk, -, -, h5, h6, rfl⟩ := outGate_spec h
  have hsz := keyProg_rows_size c n
  have hin := keyProg_inputs c n
  set R0 := Spot.Bits.Rows.ofNet c.unit
  have hN : (R0.stack c.upv).nComp = c.upv * (R0.nComp + 1) := rfl
  have hw : k * (R0.nComp + 1) + (1 + (Spot.Bits.outCol c j t - R0.nIn)) < (R0.stack c.upv).nComp := by
    rw [hN]
    have h1 : (k + 1) * (R0.nComp + 1) ≤ c.upv * (R0.nComp + 1) := Nat.mul_le_mul_right _ hk
    rw [Nat.add_one_mul] at h1
    omega
  have hr : u * (R0.stack c.upv).nComp + (k * (R0.nComp + 1) + (1 + (Spot.Bits.outCol c j t - R0.nIn))) <
      n * (R0.stack c.upv).nComp := by
    have h1 : (u + 1) * (R0.stack c.upv).nComp ≤ n * (R0.stack c.upv).nComp := Nat.mul_le_mul_right _ hu
    rw [Nat.add_one_mul] at h1
    omega
  unfold Spot.Bits.rowGate
  refine ⟨?_, ?_⟩
  · unfold Spot.Bits.KeyProg.size
    omega
  · unfold Spot.Bits.KeyProg.row?
    split
    · exfalso
      omega
    · exact ⟨_, Array.getElem?_eq_getElem (by omega)⟩

/-- **A row's position is its AND**: the last of its three lowered gates. -/
theorem memberProg_row {c : Flock.Circuit} {n g : ℕ} (hg : g < (Spot.Bits.keyProg c n).size)
    {ab : List ℕ × List ℕ} (hr : (Spot.Bits.keyProg c n).row? g = some ab) :
    (Properties.Spot.memberProg c n).gates[Spot.Bits.posOf (Spot.Bits.keyProg c n) g]? =
      some (.and [Spot.Bits.off (Spot.Bits.keyProg c n) g, Spot.Bits.off (Spot.Bits.keyProg c n) g + 1]) := by
  obtain ⟨a, b⟩ := ab
  unfold Properties.Spot.memberProg
  rw [Properties.Spot.SecurityProofs.bitsProg_gate _ _ hg]
  simp [Spot.Bits.lower, Spot.Bits.gateSize, hr, Properties.Spot.bitsGate]

/-- **A returned output leaf's bit is a returned AND** of its member's program. -/
theorem outBit_spec {c : Flock.Circuit} {n u k j t p : ℕ} (h : Spot.Bits.outBit c n u k j t = some p) :
    p ∈ (Properties.Spot.memberProg c n).outs ∧
      ∃ ws, (Properties.Spot.memberProg c n).gates[p]? = some (.and ws) := by
  unfold Spot.Bits.outBit at h
  obtain ⟨g, hg, rfl⟩ := Option.map_eq_some_iff.1 h
  refine ⟨?_, ?_⟩
  · unfold Properties.Spot.memberProg
    rw [Properties.Spot.SecurityProofs.bitsProg_outs]
    exact List.mem_map.2 ⟨g, outGate_mem_outs hg, rfl⟩
  · obtain ⟨hlt, ab, hr⟩ := outGate_row hg
    exact ⟨_, memberProg_row hlt hr⟩

/-! ## An input port's bit -/

theorem inGate_spec {c : Flock.Circuit} {n u k p t g : ℕ} (h : Spot.Bits.inGate c n u k p t = some g) :
    u < n ∧ ∃ b, b < ((Spot.Bits.Rows.ofNet c.unit).stack c.upv).nIn ∧
      g = 2 + u * ((Spot.Bits.Rows.ofNet c.unit).stack c.upv).nIn + b := by
  unfold Spot.Bits.inGate at h
  try simp only at h
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, decide_eq_true_eq, and_assoc] at hc
    obtain ⟨b, hb, rfl⟩ := Option.map_eq_some_iff.1 h
    refine ⟨hc.1, b, ?_, rfl⟩
    unfold Spot.Bits.keyIdx at hb
    try simp only at hb
    split at hb
    · cases hb
    · exact List.mem_range.1 (List.mem_of_find?_eq_some hb)
  · cases h

theorem inGate_lt_inputs {c : Flock.Circuit} {n u k p t g : ℕ} (h : Spot.Bits.inGate c n u k p t = some g) :
    g < (Spot.Bits.keyProg c n).inputs := by
  obtain ⟨hu, b, hb, hg⟩ := inGate_spec h
  have hin := keyProg_inputs c n
  have h1 : (u + 1) * ((Spot.Bits.Rows.ofNet c.unit).stack c.upv).nIn ≤
      n * ((Spot.Bits.Rows.ofNet c.unit).stack c.upv).nIn := Nat.mul_le_mul_right _ hu
  rw [Nat.add_one_mul] at h1
  omega

theorem gateSize_input {kp : Spot.Bits.KeyProg} {h : ℕ} (hh : h < kp.inputs) : Spot.Bits.gateSize kp h = 1 := by
  simp [Spot.Bits.gateSize, Spot.Bits.KeyProg.row?, hh]

/-- **An input gate is its own position**: every gate before it is an input, one lowered gate each. -/
theorem posOf_input {kp : Spot.Bits.KeyProg} {g : ℕ} (hg : g < kp.inputs) : Spot.Bits.posOf kp g = g := by
  have hle : g ≤ kp.size := by
    unfold Spot.Bits.KeyProg.size
    omega
  have hoff : Spot.Bits.off kp g = g := by
    rw [Properties.Spot.SecurityProofs.off_of_le kp hle,
      List.map_congr_left (g := fun _ => 1) fun h hh => gateSize_input (by have := List.mem_range.1 hh; omega),
      sum_range_const, Nat.mul_one]
  simp only [Spot.Bits.posOf, hoff, gateSize_input hg, Nat.add_sub_cancel]

/-- **An input port's bit is an input gate where a unit reads it**: of its member's program, the bit's source gate
lowers to an input, as the constants fix only the constant, the zero and the source gates no unit reads. -/
theorem memberProg_inGate {c : Flock.Circuit} {n u k p t g : ℕ} (h : Spot.Bits.inGate c n u k p t = some g)
    (ho : Spot.Bits.owner (Spot.Bits.keyProg c n) g ≠ none) :
    (Properties.Spot.memberProg c n).gates[Spot.Bits.posOf (Spot.Bits.keyProg c n) g]? = some .input := by
  have hlt := inGate_lt_inputs h
  obtain ⟨-, b, -, hg⟩ := inGate_spec h
  have hsz : g < (Spot.Bits.keyProg c n).size := by
    unfold Spot.Bits.KeyProg.size
    omega
  have hr : (Spot.Bits.keyProg c n).row? g = none := by
    unfold Spot.Bits.KeyProg.row?
    simp only [hlt, ↓reduceIte]
  have hκ : Spot.Bits.consts c n g = none := by
    obtain ⟨w, hw⟩ := Option.ne_none_iff_exists'.1 ho
    have h0 : g ≠ 0 := by omega
    have h1 : g ≠ 1 := by omega
    simp [Spot.Bits.consts, hw, h0, h1]
  unfold Properties.Spot.memberProg
  rw [Properties.Spot.SecurityProofs.bitsProg_gate _ _ hsz]
  simp [Spot.Bits.lower, Spot.Bits.gateSize, hr, hκ, Properties.Spot.bitsGate]

/-! ## What proofs' area gives: three of its four facts at `bitsProgs`, as statements -/

/-- Fact (2), `Flock.Bridge.Members.bitsProgs_outBit`: a returned output bit's position, at its member's base, is a
gate the statement returns, and an AND. -/
def OutBitAt : Prop :=
  ∀ (ms : Spot.Bits.Members) {m u k j t x : ℕ} (hm : m < ms.length),
    Spot.Bits.outBit ms[m].1 ms[m].2.size u k j t = some x →
      Spot.Bits.base ms m + x ∈ (Properties.Spot.bitsProgs ms).outs ∧
        ∃ ws, (Properties.Spot.bitsProgs ms).gates[Spot.Bits.base ms m + x]? = some (.and ws)

/-- Fact (3), `Flock.Bridge.Members.bitsProgs_inBit`: an input bit's position that a unit reads, at its member's base,
is an input gate of the statement. -/
def InBitAt : Prop :=
  ∀ (ms : Spot.Bits.Members) {m u k p t x : ℕ} (hm : m < ms.length),
    Spot.Bits.inBit ms[m].1 ms[m].2.size u k p t = some x →
    Spot.Bits.owner (Spot.Bits.keyProg ms[m].1 ms[m].2.size) x ≠ none →
      (Properties.Spot.bitsProgs ms).gates[Spot.Bits.base ms m + x]? = some .input

/-- Fact (4), `Flock.Bridge.Members.bitsProgs_depends_member`: a gate of member `m` depends only on gates of member
`m`. -/
def DependsMember : Prop :=
  ∀ (ms : Spot.Bits.Members) {m i g : ℕ}, m < ms.length →
    Spot.Bits.base ms m ≤ g → g < Spot.Bits.base ms (m + 1) →
    NetTiming.Depends (Properties.Spot.bitsProgs ms) i g →
      Spot.Bits.base ms m ≤ i ∧ i < Spot.Bits.base ms (m + 1)

/-! ## The members' layout fits -/

theorem getD_eq {ms : Spot.Bits.Members} {m : ℕ} (hm : m < ms.length) : ms.getD m default = ms[m] := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hm, Option.getD_some]

/-- **The members' layout fits their programs side by side** over the window, where the members name every position
it reads. -/
theorem membersLayout_fits (h2 : OutBitAt) (h3 : InBitAt) (h4 : DependsMember) {ms : Spot.Bits.Members}
    {s : Schedule} (hm : 2 < ms.length) (hd : NamesDefined ms s) :
    (membersLayout ms s).Fits (Properties.Spot.bitsProgs ms) s.window := by
  have h1 : 1 < ms.length := by omega
  have hg2 := getD_eq hm
  have hg1 := getD_eq h1
  have hb : Spot.Bits.base ms 2 =
      Spot.Bits.base ms 1 + (Properties.Spot.memberProg ms[1].1 ms[1].2.size).gates.length :=
    Flock.Bridge.Members.base_succ ms h1
  have hb3 : Spot.Bits.base ms (2 + 1) =
      Spot.Bits.base ms 2 + (Properties.Spot.memberProg ms[2].1 ms[2].2.size).gates.length :=
    Flock.Bridge.Members.base_succ ms hm
  have hout : ∀ t < s.window, ∀ leaf < outBits + s.allocation * (8 * s.cellBytes),
      outAt ms 2 t leaf ∈ (Properties.Spot.bitsProgs ms).outs ∧
        (Properties.Spot.bitsProgs ms).gates[outAt ms 2 t leaf]? ≠ some .input ∧
        Spot.Bits.base ms 2 ≤ outAt ms 2 t leaf ∧ outAt ms 2 t leaf < Spot.Bits.base ms (2 + 1) := by
    intro t ht leaf hl
    obtain ⟨x, hx⟩ := (hd t ht).1 leaf hl
    rw [hg2] at hx
    obtain ⟨ho, ws, hw⟩ := h2 ms hm hx
    obtain ⟨-, ws', hw'⟩ := outBit_spec hx
    have hlen := (List.getElem?_eq_some_iff.1 hw').1
    have he : outAt ms 2 t leaf = Spot.Bits.base ms 2 + x := by
      simp only [outAt, hg2, hx, Option.getD_some]
    rw [he, hw]
    exact ⟨ho, by simp, by omega, by omega⟩
  have hin : ∀ t < s.window, ∀ j < s.allocation * (8 * s.cellBytes),
      (Properties.Spot.bitsProgs ms).gates[inAt ms 1 t j]? = some .input ∧ inAt ms 1 t j < Spot.Bits.base ms 2 := by
    intro t ht j hj
    obtain ⟨x, hx, ho⟩ := (hd t ht).2 j hj
    rw [hg1] at hx ho
    have hq := h3 ms h1 hx ho
    have hx' := hx
    unfold Spot.Bits.inBit at hx'
    obtain ⟨g, hg, rfl⟩ := Option.map_eq_some_iff.1 hx'
    have hlt := inGate_lt_inputs hg
    rw [posOf_input hlt] at ho hq
    have hm1 := memberProg_inGate hg ho
    rw [posOf_input hlt] at hm1
    have hlen := (List.getElem?_eq_some_iff.1 hm1).1
    have he : inAt ms 1 t j = Spot.Bits.base ms 1 + g := by
      simp only [inAt, hg1, hx, Option.getD_some, posOf_input hlt]
    rw [he]
    exact ⟨hq, by omega⟩
  have hegress : ∀ t < s.window, ∀ ℓ σ, ∀ g ∈ NetTiming.slotGates ((membersLayout ms s).egress t ℓ σ),
      g ∈ (Properties.Spot.bitsProgs ms).outs ∧ (Properties.Spot.bitsProgs ms).gates[g]? ≠ some .input ∧
        Spot.Bits.base ms 2 ≤ g ∧ g < Spot.Bits.base ms (2 + 1) := by
    intro t ht ℓ σ g hg
    simp only [membersLayout, NetTiming.slotGates, List.mem_cons, List.mem_map, List.mem_range] at hg
    rcases hg with rfl | ⟨b, hb, rfl⟩
    · exact hout t ht 0 (by simp only [outBits]; omega)
    · have hσ := σ.isLt
      refine hout t ht _ ?_
      have h8 : (σ.val + 1) * (8 * s.cellBytes) ≤ s.allocation * (8 * s.cellBytes) := Nat.mul_le_mul_right _ hσ
      rw [Nat.add_one_mul] at h8
      omega
  refine ⟨fun t ht ℓ σ g hg => ⟨(hegress t ht ℓ σ g hg).1, (hegress t ht ℓ σ g hg).2.1⟩, ?_, ?_⟩
  · intro t ht g hg
    simp only [membersLayout, List.mem_map, List.mem_range] at hg
    obtain ⟨j, hj, rfl⟩ := hg
    exact (hin t ht j hj).1
  · intro t ht t' ht' _ ℓ σ g hg i hi hd'
    simp only [membersLayout, List.mem_map, List.mem_range] at hi
    obtain ⟨j, hj, rfl⟩ := hi
    obtain ⟨-, -, hlo, hhi⟩ := hegress t ht ℓ σ g hg
    have := (h4 ms hm hlo hhi hd').1
    have := (hin t' ht' j hj).2
    omega

/-- **The restated scope from proofs' facts (2), (3) and (4).** -/
theorem scanScopeAccMembers_of (h2 : OutBitAt) (h3 : InBitAt) (h4 : DependsMember) : ScanScopeAccMembers := by
  intro j program s sc c _ _ Coin _ _ _ A D msOf x hmem hS _ hdef
  refine ⟨hS, ?_⟩
  obtain ⟨st, _, hx, _⟩ := hS
  obtain ⟨P, -, hprog, hM⟩ := hmem x st hx
  refine ⟨st, hx, ?_⟩
  rw [hprog]
  exact membersLayout_fits h2 h3 h4 (by have := hM.1; omega) (hdef st hx)

end Properties.NetworkScan.MembersProof
