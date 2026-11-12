module

public import Proofs.Flock.SpotFlock.WireMap
public import Proofs.Flock.SpotFlock.WiredAt
public import Proofs.Flock.SpotFlock.ScopeDesc

/-!
# The code's wiring map rewires the members' programs

`Prog.Rewires` (`SamplingGeneric.WiredScope`) is the one condition the wired program's scope and a proof block's
values there ask of a map. This file proves it of the map the code's own wires give (`wireMap h`, `WireMap`), over
the members' programs side by side (`bitsProgs h.ms`) and their proof blocks (`membersPart h.ms`), returning the gates
the boundary record holds (`returned h`), through `Prog.Rewires.of_earlier`:

* `bitsProg_input`, `bitsProg_row`: in a key program lowered, a source gate the constants leave free is an input
  gate at its own index, and a row's position holds an AND gate. `exists_row_blockOf`: a unit's proof block holds a
  row's position, where the unit has a row.
* `gateMember ms i`: the member whose gates hold position `i`, counted as the members whose gates end at or before
  it. It rises with the position (`lt_of_gateMember_lt`), and member `m`'s positions have it `m`
  (`gateMember_base_add`).
* `bitsProgs_input`, `bitsProgs_written`: in the members' programs, a member's source gate the constants leave free
  is an input gate, and its gate on a returned output's bit is an AND gate the members' programs return.
* `lt_length_blockOf_keyProg`: a gate of a unit's proof block is a gate of its member's program.
* `gateMember_touches`: a proof block of the members holds a gate that is no input, and every gate it holds or reads
  is one member's.
* `mem_sourceBits`, `mem_writtenBits`: a source bit is a member's source gate the constants leave free, and a written
  bit a member's gate on a returned output's bit.
* `writersFirst h`: each wire's writer's part is before its reader's, decided, a part's place being the first
  template of its name (`heldPos`). `writer_lt_of_wireUnit`: there, the template a unit's wire names is before the
  unit's own.
* `rewires_wireMap`: `(bitsProgs h.ms).Rewires (membersPart h.ms) (wireMap h) (returned h)`, where each member's
  unit net is built (`Built`) with at most `2^32` rows, each of its units has a computed row, and each wire's
  writer's part is before its reader's. It asks nothing of a member's size.

The map may send two source bits of one unit to one gate: a unit that reads one output of its writer at two of its
ports. `Prog.Rewires` does not exclude it; a proof block's values at the wired program are then the block's own where
they agree at the two (`Prog.Rewires.exists_correct`).
-/

@[expose] public section

namespace Flock.Bridge.Members

open Claims.Sampling Properties.Spot

/-! ## A key program's source gates and rows, lowered -/

theorem lt_size_of_isSome_row? {kp : Spot.Bits.KeyProg} {g : ℕ} (h : (kp.row? g).isSome) : g < kp.size := by
  unfold Spot.Bits.KeyProg.row? at h
  unfold Spot.Bits.KeyProg.size
  by_cases hg : g < kp.inputs
  · omega
  · rw [ite_eq_right hg] at h
    obtain ⟨ab, hab⟩ := Option.isSome_iff_exists.1 h
    have hlt : g - kp.inputs < kp.rows.size := (Array.getElem?_eq_some_iff.1 hab).1
    omega

theorem row?_of_lt_inputs {kp : Spot.Bits.KeyProg} {g : ℕ} (hg : g < kp.inputs) : kp.row? g = none := by
  unfold Spot.Bits.KeyProg.row?
  exact ite_eq_left hg

/-- **An input gate's position is its index**: a key program's inputs come first, one lowered gate each. -/
theorem posOf_of_lt_inputs (kp : Spot.Bits.KeyProg) {g : ℕ} (hg : g < kp.inputs) : Spot.Bits.posOf kp g = g := by
  have hsz : g ≤ kp.size := by
    unfold Spot.Bits.KeyProg.size
    omega
  have h1 : Spot.Bits.off kp g = g :=
    (Properties.Spot.SecurityProofs.off_of_le kp hsz).trans (sum_gateSize_inputs kp g (Nat.le_of_lt hg))
  have h2 : Spot.Bits.gateSize kp g = 1 := gateSize_of_lt hg
  unfold Spot.Bits.posOf
  omega

/-- **A source gate the constants leave free is an input gate of the lowered program**, at its own index. -/
theorem bitsProg_input (kp : Spot.Bits.KeyProg) (κ : ℕ → Option Bool) {g : ℕ} (hg : g < kp.inputs)
    (hκ : κ g = none) : (bitsProg kp κ).gates[g]? = some .input := by
  have hsz : g < kp.size := by
    unfold Spot.Bits.KeyProg.size
    omega
  have hl : Spot.Bits.lower kp κ g = [.input] := by
    simp only [Spot.Bits.lower, row?_of_lt_inputs hg, hκ]
  have h := Properties.Spot.SecurityProofs.bitsProg_gate kp κ hsz
  rw [posOf_of_lt_inputs kp hg, hl, gateSize_of_lt hg] at h
  exact h.trans rfl

/-- **A row is lowered to an AND gate at its position.** -/
theorem bitsProg_row (kp : Spot.Bits.KeyProg) (κ : ℕ → Option Bool) {g : ℕ} (hg : (kp.row? g).isSome) :
    (bitsProg kp κ).gates[Spot.Bits.posOf kp g]? = some (.and [Spot.Bits.off kp g, Spot.Bits.off kp g + 1]) := by
  obtain ⟨⟨a, b⟩, hr⟩ := Option.isSome_iff_exists.1 hg
  have hl : Spot.Bits.lower kp κ g = [.xor (a.map (Spot.Bits.posOf kp)), .xor (b.map (Spot.Bits.posOf kp)),
      .and [Spot.Bits.off kp g, Spot.Bits.off kp g + 1]] := by
    simp only [Spot.Bits.lower, hr]
  have hs : Spot.Bits.gateSize kp g = 3 := by
    unfold Spot.Bits.gateSize
    exact ite_eq_left hg
  have h := Properties.Spot.SecurityProofs.bitsProg_gate kp κ (lt_size_of_isSome_row? hg)
  rw [hl, hs] at h
  exact h.trans rfl

/-- The first row of unit `u` is unit `u`'s. -/
theorem unitOf_first (kp : Spot.Bits.KeyProg) {u : ℕ} (hp : 0 < kp.perUnit)
    (hs : kp.inputs + u * kp.perUnit < kp.size) :
    Spot.Bits.unitOf kp (kp.inputs + u * kp.perUnit) = some u := by
  have h1 : decide (kp.inputs + u * kp.perUnit < kp.inputs) = false := decide_eq_false (by omega)
  have h2 : decide (kp.size ≤ kp.inputs + u * kp.perUnit) = false := decide_eq_false (by omega)
  have h3 : (kp.perUnit == 0) = false := beq_eq_false_iff_ne.2 (Nat.ne_of_gt hp)
  have hne : ¬ (decide (kp.inputs + u * kp.perUnit < kp.inputs) || decide (kp.size ≤ kp.inputs + u * kp.perUnit) ||
      kp.perUnit == 0) = true := by
    rw [h1, h2, h3]
    exact Bool.false_ne_true
  unfold Spot.Bits.unitOf
  rw [ite_eq_right hne, Nat.add_sub_cancel_left, Nat.mul_div_cancel _ hp]

/-- A gate's position is among its unit's lowered gates. -/
theorem bitsPosOf_mem_unitGates (kp : Spot.Bits.KeyProg) {g u : ℕ} (hg : g < kp.size)
    (hu : Spot.Bits.unitOf kp g = some u) : Spot.Bits.posOf kp g ∈ Spot.Bits.unitGates kp u := by
  have hpos : 0 < Spot.Bits.gateSize kp g := by
    unfold Spot.Bits.gateSize
    split <;> omega
  have hlast : Spot.Bits.off kp g + (Spot.Bits.gateSize kp g - 1) = Spot.Bits.posOf kp g := by
    unfold Spot.Bits.posOf
    omega
  have hmem : Spot.Bits.posOf kp g ∈ (List.range (Spot.Bits.gateSize kp g)).map (Spot.Bits.off kp g + ·) :=
    List.mem_map.2 ⟨Spot.Bits.gateSize kp g - 1, List.mem_range.2 (by omega), hlast⟩
  have hb : (Spot.Bits.unitOf kp g == some u) = true := by
    simp [hu]
  unfold Spot.Bits.unitGates
  refine List.mem_flatMap.2 ⟨g, List.mem_range.2 hg, ?_⟩
  show Spot.Bits.posOf kp g ∈ (if Spot.Bits.unitOf kp g == some u then
    (List.range (Spot.Bits.gateSize kp g)).map (Spot.Bits.off kp g + ·) else [])
  rw [ite_eq_left hb]
  exact hmem

/-- **A unit's proof block holds a row's position**, where the unit has a row. -/
theorem exists_row_blockOf (kp : Spot.Bits.KeyProg) (κ : ℕ → Option Bool) {u : ℕ} (hp : 0 < kp.perUnit)
    (hs : kp.inputs + u * kp.perUnit < kp.size) :
    ∃ g, (kp.row? g).isSome ∧ Spot.Bits.posOf kp g ∈ Spot.Bits.blockOf kp κ u := by
  refine ⟨kp.inputs + u * kp.perUnit, Flock.Bridge.isSome_row? (Nat.le_add_right _ _) hs, ?_⟩
  unfold Spot.Bits.blockOf
  exact List.mem_append_left _ (bitsPosOf_mem_unitGates kp hs (unitOf_first kp hp hs))

/-- **The gate on a returned output's bit is one of the key program's outputs.** -/
theorem outGate_mem_outs {c : _root_.Flock.Circuit} {n u k j t a : ℕ} (hu : u < n) (hk : k < c.upv)
    (hj : j < (c.leavesOut.getD k #[]).size) (ht : t < 16) (h : Spot.Bits.outGate c n u k j t = some a) :
    a ∈ (Spot.Bits.keyProg c n).outs := by
  rw [Flock.Bridge.keyProg_outs, List.mem_eraseDups, List.mem_mergeSort]
  simp only [List.mem_flatMap, List.mem_range, List.mem_filterMap]
  exact ⟨u, hu, k, hk, j, hj, t, ht, h⟩

/-! ## A gate's member -/

/-- **The member whose gates hold position `i`**: the number of members whose gates end at or before it. -/
def gateMember (ms : Spot.Bits.Members) (i : ℕ) : ℕ :=
  (List.range ms.length).countP fun m => decide (Spot.Bits.base ms (m + 1) ≤ i)

private theorem countP_lt_range {n t : ℕ} (ht : t ≤ n) : (List.range n).countP (fun m => decide (m < t)) = t := by
  induction n with
  | zero =>
    obtain rfl : t = 0 := by omega
    simp
  | succ n ih =>
    rw [List.range_succ, List.countP_append, List.countP_singleton]
    rcases Nat.lt_or_ge t (n + 1) with h | h
    · have hn : ¬ decide (n < t) = true := fun hd => absurd (of_decide_eq_true hd) (by omega)
      rw [ih (by omega), ite_eq_right hn, Nat.add_zero]
    · obtain rfl : t = n + 1 := by omega
      have hall : (List.range n).countP (fun m => decide (m < n + 1)) = (List.range n).length :=
        List.countP_eq_length.2 fun a ha => decide_eq_true (Nat.lt_succ_of_lt (List.mem_range.1 ha))
      have hn : decide (n < n + 1) = true := decide_eq_true (Nat.lt_succ_self n)
      rw [hall, List.length_range, ite_eq_left hn]

variable (ms : Spot.Bits.Members)

theorem gateMember_mono {i j : ℕ} (hij : i ≤ j) : gateMember ms i ≤ gateMember ms j := by
  unfold gateMember
  exact List.countP_mono_left fun _ _ hm => decide_eq_true (Nat.le_trans (of_decide_eq_true hm) hij)

/-- **A member's gates are before every later member's.** -/
theorem lt_of_gateMember_lt {i j : ℕ} (h : gateMember ms i < gateMember ms j) : i < j := by
  rcases Nat.lt_or_ge i j with hlt | hge
  · exact hlt
  · exact absurd (gateMember_mono ms hge) (Nat.not_le.2 h)

theorem gateMember_eq {t i : ℕ} (ht : t < ms.length) (h1 : Spot.Bits.base ms t ≤ i)
    (h2 : i < Spot.Bits.base ms (t + 1)) : gateMember ms i = t := by
  have hc : (List.range ms.length).countP (fun m => decide (Spot.Bits.base ms (m + 1) ≤ i)) =
      (List.range ms.length).countP (fun m => decide (m < t)) :=
    List.countP_congr fun m _ => by
      simp only [decide_eq_true_eq]
      constructor
      · intro hm
        by_contra hge
        have hb := base_mono ms (show t + 1 ≤ m + 1 by omega)
        omega
      · intro hm
        have hb := base_mono ms (show m + 1 ≤ t by omega)
        omega
  unfold gateMember
  rw [hc, countP_lt_range (Nat.le_of_lt ht)]

/-- **Member `m`'s positions are member `m`'s.** -/
theorem gateMember_base_add {m p : ℕ} (hm : m < ms.length)
    (hp : p < (memberProg ms[m].1 ms[m].2.size).gates.length) :
    gateMember ms (Spot.Bits.base ms m + p) = m := by
  have hb := base_succ ms hm
  exact gateMember_eq ms hm (Nat.le_add_right _ _) (by omega)

/-! ## A member's source gates and written gates, in the members' programs -/

/-- **A member's source gate the constants leave free is an input gate of the members' programs**, and a position
of the member's. -/
theorem bitsProgs_input (htop : ∀ ci ∈ ms, (memberProg ci.1 ci.2.size).Topological) {m g : ℕ} (hm : m < ms.length)
    (hg : g < (Spot.Bits.keyProg ms[m].1 ms[m].2.size).inputs)
    (hκ : Spot.Bits.consts ms[m].1 ms[m].2.size g = none) :
    (bitsProgs ms).gates[Spot.Bits.base ms m + g]? = some .input ∧
      g < (memberProg ms[m].1 ms[m].2.size).gates.length := by
  have hin : (memberProg ms[m].1 ms[m].2.size).gates[g]? = some .input :=
    bitsProg_input (Spot.Bits.keyProg ms[m].1 ms[m].2.size) (Spot.Bits.consts ms[m].1 ms[m].2.size) hg hκ
  have hlt : g < (memberProg ms[m].1 ms[m].2.size).gates.length := (List.getElem?_eq_some_iff.1 hin).1
  have hq := (sitsAt_bitsProgs ms htop m hm).gates g hlt
  rw [hin] at hq
  exact ⟨hq.trans rfl, hlt⟩

/-- **A member's gate on a returned output's bit is an AND gate the members' programs return**, and a position of
the member's. -/
theorem bitsProgs_written (htop : ∀ ci ∈ ms, (memberProg ci.1 ci.2.size).Topological) {m a : ℕ} (hm : m < ms.length)
    (ha : a ∈ (Spot.Bits.keyProg ms[m].1 ms[m].2.size).outs) :
    (∃ ws, (bitsProgs ms).gates[Spot.Bits.base ms m +
        Spot.Bits.posOf (Spot.Bits.keyProg ms[m].1 ms[m].2.size) a]? = some (.and ws)) ∧
      Spot.Bits.base ms m + Spot.Bits.posOf (Spot.Bits.keyProg ms[m].1 ms[m].2.size) a ∈ (bitsProgs ms).outs ∧
      Spot.Bits.posOf (Spot.Bits.keyProg ms[m].1 ms[m].2.size) a <
        (memberProg ms[m].1 ms[m].2.size).gates.length := by
  have hA := sitsAt_bitsProgs ms htop m hm
  have hand : (memberProg ms[m].1 ms[m].2.size).gates[
      Spot.Bits.posOf (Spot.Bits.keyProg ms[m].1 ms[m].2.size) a]? =
        some (.and [Spot.Bits.off (Spot.Bits.keyProg ms[m].1 ms[m].2.size) a,
          Spot.Bits.off (Spot.Bits.keyProg ms[m].1 ms[m].2.size) a + 1]) :=
    bitsProg_row (Spot.Bits.keyProg ms[m].1 ms[m].2.size) (Spot.Bits.consts ms[m].1 ms[m].2.size)
      (Flock.Bridge.isSome_row?_of_mem_outs ha)
  have hlt : Spot.Bits.posOf (Spot.Bits.keyProg ms[m].1 ms[m].2.size) a <
      (memberProg ms[m].1 ms[m].2.size).gates.length := (List.getElem?_eq_some_iff.1 hand).1
  have hout : Spot.Bits.posOf (Spot.Bits.keyProg ms[m].1 ms[m].2.size) a ∈
      (bitsProg (Spot.Bits.keyProg ms[m].1 ms[m].2.size) (Spot.Bits.consts ms[m].1 ms[m].2.size)).outs := by
    rw [Properties.Spot.SecurityProofs.bitsProg_outs]
    exact List.mem_map.2 ⟨a, ha, rfl⟩
  have hq := hA.gates _ hlt
  rw [hand] at hq
  exact ⟨⟨[Spot.Bits.off (Spot.Bits.keyProg ms[m].1 ms[m].2.size) a,
      Spot.Bits.off (Spot.Bits.keyProg ms[m].1 ms[m].2.size) a + 1].map (Spot.Bits.base ms m + ·), hq.trans rfl⟩,
    (hA.outs _ hlt).2 hout, hlt⟩

/-! ## A proof block of the members -/

section keyProg

open FlockSoundness FlockSoundness.Discharge

/-- **A gate of a unit's proof block is a gate of its member's program**, where the member's unit net is built
(`Built`) with at most `2^32` rows: what `partitions_keyProg` says of a block's gates, with nothing asked of the
program's size. -/
theorem lt_length_blockOf_keyProg (c : _root_.Flock.Circuit) (n : ℕ) {hU : Built c.unit} (hu : c.unit.unitLog ≤ 32)
    (u : Fin n) {p : ℕ} (hp : p ∈ Spot.Bits.blockOf (Spot.Bits.keyProg c n) (Spot.Bits.consts c n) u.val) :
    p < (memberProg c n).gates.length := by
  rw [← blockOf_bitsKeyProg (hU := hU) hu u] at hp
  have hlt := lt_length_blockOf hp
  rw [progOf_keyProg hu] at hlt
  exact hlt

end keyProg

/-- **A proof block of the members holds a gate that is no input, and the gates it holds or reads are one
member's**, where each member's unit net is built (`Built`) with at most `2^32` rows and each of its units has a
computed row. Nothing is asked of a member's size. -/
theorem gateMember_touches (hbuilt : ∀ ci ∈ ms, FlockSoundness.Built ci.1.unit)
    (hlog : ∀ ci ∈ ms, ci.1.unit.unitLog ≤ 32) (hrow : ∀ ci ∈ ms, 0 < ci.2.size → 0 < (bitsRows ci.1).nComp)
    {u : List ℕ} (hu : u ∈ (membersPart ms).blocks) :
    (∃ g ∈ u, (bitsProgs ms).gates[g]? ≠ some .input) ∧
      ∃ t, ∀ i, (bitsProgs ms).Touches u i → gateMember ms i = t := by
  have htop : ∀ ci ∈ ms, (memberProg ci.1 ci.2.size).Topological := fun ci hci =>
    topological_memberProg ci.1 (hU := hbuilt ci hci) (hlog ci hci) ci.2.size
  rw [blocks_membersPart, List.mem_flatMap] at hu
  obtain ⟨m, hm, hx⟩ := hu
  have hm' := List.mem_range.1 hm
  rw [memberBlocks_eq ms hm'] at hx
  obtain ⟨y, hy, rfl⟩ := List.mem_map.1 hx
  obtain ⟨w, hw, rfl⟩ := List.mem_map.1 hy
  have hw' : w < ms[m].2.size := List.mem_range.1 hw
  have hmem : ms[m] ∈ ms := List.getElem_mem hm'
  have hA := sitsAt_bitsProgs ms htop m hm'
  have hlt : ∀ p ∈ Spot.Bits.blockOf (Spot.Bits.keyProg ms[m].1 ms[m].2.size)
      (Spot.Bits.consts ms[m].1 ms[m].2.size) w, p < (memberProg ms[m].1 ms[m].2.size).gates.length :=
    fun p hp => lt_length_blockOf_keyProg ms[m].1 ms[m].2.size (hU := hbuilt ms[m] hmem) (hlog ms[m] hmem)
      ⟨w, hw'⟩ hp
  constructor
  · have hc : 0 < (bitsRows ms[m].1).nComp := hrow ms[m] hmem (Nat.lt_of_le_of_lt (Nat.zero_le w) hw')
    have hpu : 0 < (Spot.Bits.keyProg ms[m].1 ms[m].2.size).perUnit := by
      rw [Flock.Bridge.keyProg_perUnit]
      exact hc
    have hs : (Spot.Bits.keyProg ms[m].1 ms[m].2.size).inputs +
        w * (Spot.Bits.keyProg ms[m].1 ms[m].2.size).perUnit < (Spot.Bits.keyProg ms[m].1 ms[m].2.size).size := by
      rw [Flock.Bridge.keyProg_size, Flock.Bridge.keyProg_inputs, Flock.Bridge.keyProg_perUnit]
      have hmul := Nat.mul_le_mul_right (bitsRows ms[m].1).nComp (show w + 1 ≤ ms[m].2.size from hw')
      rw [Nat.add_one_mul] at hmul
      omega
    obtain ⟨g, hg, hgb⟩ := exists_row_blockOf (Spot.Bits.keyProg ms[m].1 ms[m].2.size)
      (Spot.Bits.consts ms[m].1 ms[m].2.size) (u := w) hpu hs
    have hand : (memberProg ms[m].1 ms[m].2.size).gates[
        Spot.Bits.posOf (Spot.Bits.keyProg ms[m].1 ms[m].2.size) g]? =
          some (.and [Spot.Bits.off (Spot.Bits.keyProg ms[m].1 ms[m].2.size) g,
            Spot.Bits.off (Spot.Bits.keyProg ms[m].1 ms[m].2.size) g + 1]) :=
      bitsProg_row (Spot.Bits.keyProg ms[m].1 ms[m].2.size) (Spot.Bits.consts ms[m].1 ms[m].2.size) hg
    refine ⟨Spot.Bits.base ms m + Spot.Bits.posOf (Spot.Bits.keyProg ms[m].1 ms[m].2.size) g,
      List.mem_map.2 ⟨_, hgb, rfl⟩, fun hne => ?_⟩
    rw [hA.gates _ (hlt _ hgb), hand, Option.map_some] at hne
    have hgate := shiftGate_eq_input.1 (Option.some.inj hne)
    cases hgate
  · refine ⟨m, fun i hi => ?_⟩
    rcases hi with hi | ⟨g, hg, x, hx, hi⟩
    · obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hi
      exact gateMember_base_add ms hm' (hlt p hp)
    · obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hg
      have hpl := hlt p hp
      rw [hA.gates_eq hpl] at hx
      obtain rfl := Option.some.inj hx
      rw [reads_shiftGate] at hi
      obtain ⟨q, hq, rfl⟩ := List.mem_map.1 hi
      have hqp : q < p := (htop ms[m] (List.getElem_mem hm')).1 p _ (List.getElem?_eq_getElem hpl) q hq
      exact gateMember_base_add ms hm' (Nat.lt_trans hqp hpl)

end Flock.Bridge.Members

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy
open Properties.Spot (bitsProgs membersPart memberProg)

/-! ## Source bits and written bits -/

/-- **A source bit is a member's source gate the constants leave free**, at the member's positions. -/
theorem mem_sourceBits {ms : Spot.Bits.Members} {x t : ℕ} {r : Spot.Bits.RowPos}
    (hs : (x, t, r) ∈ sourceBits ms) :
    ∃ ht : t < ms.length, ∃ g, g < (Spot.Bits.keyProg ms[t].1 ms[t].2.size).inputs ∧
      Spot.Bits.consts ms[t].1 ms[t].2.size g = none ∧ x = Spot.Bits.base ms t + g := by
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
    obtain ⟨r', -, he⟩ := Option.map_eq_some_iff.1 hgx'
    have h1 : Spot.Bits.base ms m + g = x := congrArg Prod.fst he
    have h2 : m = t := congrArg (fun p => p.2.1) he
    subst h2
    exact ⟨hm', g, List.mem_range.1 hg, hκ, h1.symm⟩

/-- **A written bit is a member's gate on a returned output's bit**, one of its key program's outputs, at the
member's positions. -/
theorem mem_writtenBits {ms : Spot.Bits.Members} {y tw : ℕ} {r : Spot.Bits.RowPos}
    (hs : (y, tw, r) ∈ writtenBits ms) :
    ∃ ht : tw < ms.length, ∃ a, a ∈ (Spot.Bits.keyProg ms[tw].1 ms[tw].2.size).outs ∧
      y = Spot.Bits.base ms tw + Spot.Bits.posOf (Spot.Bits.keyProg ms[tw].1 ms[tw].2.size) a := by
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
      subst h2
      exact ⟨hm', a, Flock.Bridge.Members.outGate_mem_outs (List.mem_range.1 hu) (List.mem_range.1 hk)
        (List.mem_range.1 hj) (List.mem_range.1 hb) hg, h1.symm⟩

variable (h : Held)

/-! ## Each wire's writer's part is before its reader's -/

/-- **A template name's place among round 2's parts**: the first template of the name (`heldWriter`), the members'
count where none has it. -/
def heldPos (name : String) : ℕ := (heldWriter h name).getD h.ms.length

/-- **Each wire's writer's part is before its reader's**, decided: the first template of the writer's name comes
before the first of the reader's. No unit is then wired to its own template or to a later one. -/
def writersFirst : Bool :=
  (h.wires.getD #[]).all fun w => decide (heldPos h w.writer < heldPos h w.template)

/-- The first template of a name is at or before every template of that name. -/
theorem heldWriter_le {name : String} {t : ℕ} (ht : t < h.ms.length) (hn : h.names.getD t "" = name) :
    ∃ t', heldWriter h name = some t' ∧ t' ≤ t := by
  unfold heldWriter
  cases hf : (List.range h.ms.length).find? (fun t => h.names.getD t "" == name) with
  | none =>
    have hnone : ¬ (h.names.getD t "" == name) = true := List.find?_eq_none.1 hf t (List.mem_range.2 ht)
    rw [hn] at hnone
    exact absurd (beq_self_eq_true name) hnone
  | some t' =>
    refine ⟨t', rfl, ?_⟩
    have hbefore := (List.find?_range_eq_some.1 hf).2.2
    by_contra hlt
    have hnot : (!(h.names.getD t "" == name)) = true := hbefore t (by omega)
    rw [hn, beq_self_eq_true name] at hnot
    exact Bool.false_ne_true hnot

/-- **Where each wire's writer's part is before its reader's, the template a unit's wire names is before the
unit's own.** -/
theorem writer_lt_of_wireUnit (hc : writersFirst h = true) {t : ℕ} (ht : t < h.ms.length) {port : String} {u : ℕ}
    {y : ℕ × String × ℕ} (hy : wireUnit h t port u = some y) : y.1 < t := by
  unfold wireUnit at hy
  obtain ⟨z, hz, rfl⟩ := Option.map_eq_some_iff.1 hy
  obtain ⟨w, r, -, hw, -, htw, -, -⟩ := wireCell_wire h hz
  have hwm : w ∈ h.wires.getD #[] := Array.mem_of_find?_eq_some hw
  have hwp := Array.find?_some hw
  have htmpl : w.template = h.names.getD t "" := eq_of_beq (Bool.and_eq_true_iff.1 hwp).1
  have hlt : heldPos h w.writer < heldPos h w.template :=
    of_decide_eq_true (Array.all_eq_true_iff_forall_mem.1 hc w hwm)
  obtain ⟨t', ht', hle⟩ := heldWriter_le h ht htmpl.symm
  unfold heldPos at hlt
  rw [htw, ht', Option.getD_some, Option.getD_some] at hlt
  exact Nat.lt_of_lt_of_le hlt hle

/-! ## The wiring map rewires the members' programs -/

/-- **The code's wiring map rewires the members' programs** (`Prog.Rewires`), returning the gates the boundary
record holds, where each member's unit net is built (`Built`) with at most `2^32` rows, each of its units has a
computed row, and each wire's writer's part is before its reader's (`writersFirst`): a gate the map moves is a
source bit, an input gate; the gate it goes to is a written bit, an AND gate the members' programs return, of an
earlier member; and a proof block's gates and reads are one member's. Nothing is asked of a member's size. -/
theorem rewires_wireMap (hbuilt : ∀ ci ∈ h.ms, FlockSoundness.Built ci.1.unit)
    (hlog : ∀ ci ∈ h.ms, ci.1.unit.unitLog ≤ 32)
    (hrow : ∀ ci ∈ h.ms, 0 < ci.2.size → 0 < (Flock.Bridge.bitsRows ci.1).nComp)
    (hfirst : writersFirst h = true) :
    (bitsProgs h.ms).Rewires (membersPart h.ms) (wireMap h) (returned h) := by
  have htop : ∀ ci ∈ h.ms, (memberProg ci.1 ci.2.size).Topological := fun ci hci =>
    Flock.Bridge.Members.topological_memberProg ci.1 (hU := hbuilt ci hci) (hlog ci hci) ci.2.size
  have hmoved : ∀ i, wireMap h i ≠ i → ∃ t r tw wport u, (i, t, r) ∈ sourceBits h.ms ∧
      wireUnit h t r.row r.leaf = some (tw, wport, u) ∧
      (wireMap h i, tw, Spot.Bits.RowPos.mk wport u r.bit) ∈ writtenBits h.ms := fun i hi => by
    cases hv : wireTarget h i with
    | none => exact absurd (wireMap_of_no_target h hv) hi
    | some y =>
      obtain ⟨t, r, tw, wport, u, hs, hu, -, hw, -⟩ := wireTarget_spec h hv
      rw [wireMap_of_target h hv]
      exact ⟨t, r, tw, wport, u, hs, hu, hw⟩
  have hin : ∀ i, wireMap h i ≠ i → (bitsProgs h.ms).gates[i]? = some .input := fun i hi => by
    obtain ⟨t, r, tw, wport, u, hs, -, -⟩ := hmoved i hi
    obtain ⟨ht, g, hg, hκ, hx⟩ := mem_sourceBits hs
    rw [hx]
    exact (Flock.Bridge.Members.bitsProgs_input h.ms htop ht hg hκ).1
  refine Prog.Rewires.of_earlier (mem := Flock.Bridge.Members.gateMember h.ms) hin ?_ ?_ ?_ ?_ ?_ ?_
  · intro i hi
    obtain ⟨t, r, tw, wport, u, -, -, hw⟩ := hmoved i hi
    obtain ⟨ht, a, ha, he⟩ := mem_writtenBits hw
    obtain ⟨⟨ws, hand⟩, -, -⟩ := Flock.Bridge.Members.bitsProgs_written h.ms htop ht ha
    rw [he, hand]
    refine ⟨fun hne => ?_, fun c hne => ?_⟩
    · cases hne
    · cases hne
  · intro i hi
    obtain ⟨t, r, tw, wport, u, -, -, hw⟩ := hmoved i hi
    obtain ⟨ht, a, ha, he⟩ := mem_writtenBits hw
    rw [he]
    exact (Flock.Bridge.Members.bitsProgs_written h.ms htop ht ha).2.1
  · exact fun i j hij => Flock.Bridge.Members.lt_of_gateMember_lt h.ms hij
  · intro i hi
    obtain ⟨t, r, tw, wport, u, hs, hu, hw⟩ := hmoved i hi
    obtain ⟨ht, g, hg, hκ, hx⟩ := mem_sourceBits hs
    obtain ⟨htw, a, ha, he⟩ := mem_writtenBits hw
    have hlt : tw < t := writer_lt_of_wireUnit h hfirst ht hu
    have h1 := Flock.Bridge.Members.gateMember_base_add h.ms htw
      (Flock.Bridge.Members.bitsProgs_written h.ms htop htw ha).2.2
    have h2 := Flock.Bridge.Members.gateMember_base_add h.ms ht
      (Flock.Bridge.Members.bitsProgs_input h.ms htop ht hg hκ).2
    rw [← he] at h1
    rw [← hx] at h2
    rw [h1, h2]
    exact hlt
  · intro i hi
    cases hr : returned h i with
    | false => rfl
    | true =>
      exfalso
      unfold returned at hr
      obtain ⟨⟨yx, yt, yr⟩, hy, hyi⟩ := List.any_eq_true.1 hr
      have hyx : yx = i := eq_of_beq (Bool.and_eq_true_iff.1 hyi).1
      obtain ⟨ht, a, ha, he⟩ := mem_writtenBits hy
      obtain ⟨⟨ws, hand⟩, -, -⟩ := Flock.Bridge.Members.bitsProgs_written h.ms htop ht ha
      rw [← he, hyx, hin i hi] at hand
      cases hand
  · exact fun u hu => Flock.Bridge.Members.gateMember_touches h.ms hbuilt hlog hrow hu

end SpotFlock
