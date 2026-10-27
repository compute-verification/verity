import Proofs.Flock.Bridge.PublicWitnesses
import Proofs.Flock.VBridge.Unit
import Flock.Test.ToyPubFiles

/-!
# The toy session's circuit, through the key program

`Flock.Test.ToyPubFiles.circuit` is one unit whose returned bit (`y`'s bit 0) is a row reading `x`'s bit 0 alone. In
the key program of one unit (`Placed.keyProg circuit hU 1`) that is a returned gate `gOut` of unit 0 whose operation is a
row on the one input gate `xIn` (`op_gOut`), which no constant fixes (`consts_xIn`); `xIn` is bit `k` of unit 0's row of
port 0 (`rowOf_xIn`), and every row bit `gOut` is on is of a later port (`rowOf_gOut`). The kernel reads the unit's row
(`row1024`) and its first input's key (`srcKey0`) off the written fields of `circuit`.
-/

namespace Flock.Bridge.ToyPub

open FlockSoundness FlockSoundness.Discharge
open Flock.Test.ToyPubFiles (circuit)

theorem row1024 : (circuit.unit.a.row 1024).toList = [0] := by decide +kernel

theorem srcKey0 : Aliased.srcKey circuit 0 = .msg 0 0 0 := by decide +kernel

theorem hmNIn : HmOut.nIn circuit = 1 := by decide +kernel

/-- **An input whose key is no zero is its own first key.** -/
theorem keyIdx_zero {B : ℕ} (hB : 0 < B) (h : Aliased.srcKey circuit 0 ≠ .zero) :
    Aliased.keyIdx circuit B ⟨0, hB⟩ = some ⟨0, hB⟩ := by
  unfold Aliased.keyIdx
  rw [if_neg h]
  obtain ⟨B', rfl⟩ : ∃ B', B = B' + 1 := ⟨B - 1, by omega⟩
  rw [List.finRange_succ, List.find?_cons_of_pos]
  · rfl
  · simp

section

variable (hU : Built circuit.unit)

theorem nIn_ofNet : (Rows.ofNet hU).nIn = 1024 := by
  rw [Rows.ofNet_nIn]
  decide +kernel

theorem nComp_ofNet : (Rows.ofNet hU).nComp = 128 := by
  have h : (Rows.ofNet hU).nComp =
      circuit.unit.constPos - min (circuit.unit.inWords * Flock.WORD) circuit.unit.constPos := rfl
  rw [h]
  decide +kernel

/-- The key program's circuit. -/
noncomputable abbrev Cc := (Placed.keyProg circuit hU 1).circuit (Layout.outsOf circuit hU 1)

/-- The key program's partition. -/
noncomputable abbrev P := (Placed.keyProg circuit hU 1).partition (Layout.outsOf circuit hU 1)

/-- Unit 0's rows. -/
noncomputable abbrev R := (Placed.keyProg circuit hU 1).rows 0

/-- Unit 0's instance in the key program. -/
noncomputable abbrev inst := Const.Prog.instC (Placed.keyProg circuit hU 1) (Layout.outsOf circuit hU 1) 0

theorem eR : R hU = (Rows.ofNet hU).stack 1 :=
  Placed.rows_circuitProg circuit hU 1 (Layout.keyB circuit hU) (Aliased.keyIdx circuit (Layout.keyB circuit hU)) 0

theorem nIn_R : (R hU).nIn = 1024 := by
  rw [congrArg Rows.nIn (eR hU)]
  show 1 * (Rows.ofNet hU).nIn = 1024
  rw [nIn_ofNet]

theorem nComp_R : (R hU).nComp = 129 := by
  rw [congrArg Rows.nComp (eR hU)]
  show 1 * ((Rows.ofNet hU).nComp + 1) = 129
  rw [nComp_ofNet]

theorem keyB_eq : Layout.keyB circuit hU = 1024 := by
  show circuit.upv * (Rows.ofNet hU).nIn = 1024
  rw [nIn_ofNet]
  rfl

theorem h1R : 1 < (R hU).nComp := by
  rw [nComp_R]
  omega

/-- **Unit 0's computed row 1 reads one column, input column 0** (the copy's row 1024 of the net). -/
theorem a_copy (h1 : 1 < (R hU).nComp) : ∃ z, (R hU).a ⟨1, h1⟩ = [z] ∧ z.val = 0 := by
  have hA := Rows.ofNet_a hU (by decide +kernel) ⟨0, by rw [nComp_ofNet hU]; omega⟩
  have hrow : (circuit.unit.a.row ((Rows.ofNet hU).nIn + 0)).toList = [0] := by
    rw [nIn_ofNet hU]
    exact row1024
  obtain ⟨y, hy, hy0⟩ := List.map_eq_singleton_iff.1 (hA.trans hrow)
  revert h1
  rw [eR hU]
  intro h1
  have hs := FlockVBridge.stack_a (Rows.ofNet hU) 1 (u := 0) Nat.zero_lt_one ⟨0, by rw [nComp_ofNet hU]; omega⟩
  have hs' : ((Rows.ofNet hU).stack 1).a ⟨1, h1⟩ = _ :=
    (congrArg ((Rows.ofNet hU).stack 1).a (Fin.ext (by simp))).trans hs
  rw [hs', hy]
  refine ⟨_, rfl, ?_⟩
  have hlt : y.val < (Rows.ofNet hU).nIn := by rw [hy0, nIn_ofNet hU]; omega
  unfold Rows.stackCol
  rw [dif_pos hlt]
  simp [hy0]

/-- **The returned gate**: unit 0's computed column 1, the copy of `x`'s bit 0. -/
noncomputable def gOut : Fin (Cc hU).N := (inst hU).wire ((R hU).comp ⟨1, h1R hU⟩)

/-- **The input gate it reads**: unit 0's input column 0. -/
noncomputable def xIn : Fin (Cc hU).N := (inst hU).wire (Classical.choose (a_copy hU (h1R hU)))

theorem op_gOut :
    (Cc hU).op (gOut hU) = .row [xIn hU] (((R hU).b ⟨1, h1R hU⟩).map (inst hU).wire) := by
  obtain ⟨hz, -⟩ := Classical.choose_spec (a_copy hU (h1R hU))
  show (Cc hU).op ((inst hU).wire ((R hU).comp ⟨1, h1R hU⟩)) = _
  rw [(inst hU).op ⟨1, h1R hU⟩, hz]
  rfl

theorem unit_gOut : (P hU).unit (gOut hU) = some 0 := (inst hU).mem ⟨1, h1R hU⟩

theorem xIn_val : (xIn hU).val = 2 := by
  obtain ⟨-, hz0⟩ := Classical.choose_spec (a_copy hU (h1R hU))
  have hlt : (Classical.choose (a_copy hU (h1R hU))).val < (R hU).nIn := by rw [hz0, nIn_R]; omega
  have hlt0 : 0 < (Placed.circuitRows circuit hU).nIn := by
    show 0 < 1 * (Rows.ofNet hU).nIn
    rw [nIn_ofNet]
    omega
  unfold xIn
  rw [Const.Prog.instC_wire_input (hc := hlt), hz0]
  calc (Const.Prog.insAt (Placed.keyProg circuit hU 1) 0 0).val
      = (Placed.insOf 1 (Layout.keyB circuit hU) (Aliased.keyIdx circuit (Layout.keyB circuit hU)) 0
          ⟨0, hlt0⟩).val := Placed.insAt_uniform 1 0 0 hlt0
    _ = 2 := by
      rw [Placed.insOf_of_lt (hu := Nat.zero_lt_one),
        keyIdx_zero (B := Layout.keyB circuit hU) hlt0 (by rw [srcKey0]; simp)]
      simp [Placed.srcGate]

theorem consts_xIn : constsOf circuit hU 1 (xIn hU) = none := by
  unfold constsOf
  have h0 : (Layout.oneG (c := circuit) (hU := hU) (n := 1)).val = 0 := rfl
  have h1 : (Layout.zeroG (c := circuit) (hU := hU) (n := 1)).val = 1 := rfl
  have hx := xIn_val hU
  have hn1 : xIn hU ≠ Layout.oneG (c := circuit) (hU := hU) (n := 1) := fun h => by
    have := congrArg Fin.val h
    omega
  have hn0 : xIn hU ≠ Layout.zeroG (c := circuit) (hU := hU) (n := 1) := fun h => by
    have := congrArg Fin.val h
    omega
  first
    | (split_ifs with ha hb <;> first | rfl | exact (hn1 ha).elim | exact (hn0 hb).elim)
    | simp [hn1, hn0]

theorem op_xIn : (Cc hU).op (xIn hU) = .input := by
  obtain ⟨-, hz0⟩ := Classical.choose_spec (a_copy hU (h1R hU))
  have hne : (P hU).unit (xIn hU) ≠ some 0 :=
    (inst hU).outside _ fun i h => by
      have := congrArg Fin.val h
      change _ = (R hU).nIn + i.val at this
      rw [hz0, nIn_R] at this
      omega
  have hn : (P hU).unit (xIn hU) = none := by
    rcases h : (P hU).unit (xIn hU) with _ | u
    · rfl
    · rw [Subsingleton.elim u 0] at h
      exact absurd h hne
  have hi := ((P hU).input_iff (xIn hU)).1 hn
  generalize (Cc hU).op (xIn hU) = o at hi ⊢
  cases o <;> simp_all [Audit.Op.isInput]

/-- **`xIn` is a bit of unit 0's row of port 0**: its key is message bit 0 of byte 0 of port 0. -/
theorem keyRow_xIn : ∃ u k, Layout.keyRowOf circuit hU 1 (Layout.portRowW circuit) (xIn hU) =
    some (u, ⟨⟨0, by decide⟩, k⟩) := by
  have hx := xIn_val hU
  have hB := keyB_eq hU
  have h2 : 2 ≤ (xIn hU).val ∧ (xIn hU).val < 2 + 1 * Layout.keyB circuit hU := by rw [hx, hB]; omega
  have hs : Aliased.srcKey circuit (((xIn hU).val - 2) % Layout.keyB circuit hU) = .msg 0 0 0 := by
    rw [hx, Nat.sub_self, Nat.zero_mod, srcKey0]
  unfold Layout.keyRowOf
  rw [dif_pos h2, hs]
  exact ⟨_, _, rfl⟩

theorem rowOf_xIn : ∃ u k, HiddenExec.rowOfH circuit hU 1 (Layout.outsOf circuit hU 1) (xIn hU) =
    some (u, ⟨⟨0, by decide⟩, k⟩) := by
  obtain ⟨u, k, h⟩ := keyRow_xIn hU
  exact ⟨u, k, HiddenExec.rowOfH_of_key _ h⟩

/-- **Every row bit `gOut` is on is of a port after the input**: an output row's (`outBitRow_spec`), never a key's. -/
theorem rowOf_gOut {u : Fin 1} {q : Fin circuit.ports.size} {k : Fin (Layout.portRowW circuit q)}
    (h : HiddenExec.rowOfH circuit hU 1 (Layout.outsOf circuit hU 1) (gOut hU) = some (u, ⟨q, k⟩)) : q.val ≠ 0 := by
  unfold gOut at h
  rcases hk : Layout.keyRowOf circuit hU 1 (Layout.portRowW circuit)
      ((inst hU).wire ((R hU).comp ⟨1, h1R hU⟩)) with _ | ⟨u', q', k'⟩
  · rw [HiddenExec.rowOfH_of_none _ hk] at h
    have := (HiddenExec.outBitRow_spec (HiddenExec.outRowOf_wire_some (h := h)).2.2).2.1
    rw [hmNIn] at this
    omega
  · have := (Layout.keyRowOf_wire (h := hk)).2.1
    change (R hU).nIn + 1 < (R hU).nIn at this
    omega

theorem gOut_mem : gOut hU ∈ (Cc hU).outputs := by
  show _ ∈ Layout.outsOf circuit hU 1
  refine (@Finset.mem_filter _ _ (Classical.decPred _) _ _).2
    ⟨Finset.mem_univ _, 0, (R hU).comp ⟨1, h1R hU⟩, rfl, 0, 0, 0, ?_⟩
  have hoc : Layout.outCol circuit 0 0 = 1024 := by
    unfold Layout.outCol
    decide
  have e1 := nIn_R hU
  have e2 := nIn_ofNet hU
  have e3 := nComp_ofNet hU
  have hup : circuit.upv = 1 := rfl
  refine ⟨by decide, by decide, by decide, ?_, ?_, ?_⟩
  · omega
  · omega
  · show (R hU).nIn + 1 = _
    rw [hup]
    omega

theorem ioSum_le : ioSum (P hU) ≤ 2 ^ 40 := by
  have hN : (Cc hU).N = 1155 := by
    show 2 + 1 * (circuit.upv * (Rows.ofNet hU).nIn) + circuit.upv * ((Rows.ofNet hU).nComp + 1) = 1155
    rw [nIn_ofNet, nComp_ofNet]
    rfl
  unfold ioSum
  calc ∑ u, ((P hU).io u).card ≤ ∑ _u : Fin 1, (Cc hU).N :=
        Finset.sum_le_sum fun u _ => (Finset.card_le_univ _).trans_eq (Fintype.card_fin _)
    _ ≤ 2 ^ 40 := by simp [hN]

end

end Flock.Bridge.ToyPub
