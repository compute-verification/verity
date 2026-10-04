import FlockSoundness.Discharge.ZkPubIn.SiteP
import FlockSoundness.Discharge.HiddenExec.KeyH

set_option Elab.async false

/-!
# `keyProgP`'s wiring beside `keyProg`'s

`keyProgP` and `keyProg` are `Placed.circuitProg` at one circuit's rows with different input sources, so they have one
partition into units (`unit_keyProgP`) and put each unit's computed columns and constant on one gate
(`wire_keyProgP`); they differ only at the input columns. So HiddenExec's output-row bits (`outRowOf`, `rowOfH`) read
`keyProgP`'s instances as `keyProg`'s (`outRowOf_wireP`, `rowOfH_wire_unitP`), and `OutsOut` transfers
(`outsOut_keyProgP`). At an input column, a row bit's key is a message bit, so the input is no public leaf's bit
(`srcKey_of_pubLeaf`, `keyRowOf_wireP`).
-/

namespace FlockSoundness.Discharge.ZkPubIn

open Flock FlockLevel3 Model Audit Binding Placed HiddenExec
open FlockSoundness.Discharge.Layout (keyN keyB keyRowOf portRowW OutColAt outDecode OutsOut)

/-! ## `uniform` at two input wirings -/

section uniform

variable {k : ℕ} {R : FlockSoundness.Rows}

/-- `uniform`'s partition into units does not read the input wiring. -/
theorem unit_uniform_eq {ins ins' : ℕ → Fin R.nIn → Fin k} {one : Fin k} {hone : ∀ u c, ins u c ≠ one}
    {hone' : ∀ u c, ins' u c ≠ one} :
    ∀ n, (Placed.uniform k R ins one hone n).unit = (Placed.uniform k R ins' one hone' n).unit
  | 0 => rfl
  | n + 1 => by
    funext g
    change (FlockSoundness.Prog.snoc (Placed.uniform k R ins one hone n) R _ _ _).unit g =
      (FlockSoundness.Prog.snoc (Placed.uniform k R ins' one hone' n) R _ _ _).unit g
    simp only [FlockSoundness.Prog.unit]
    rw [unit_uniform_eq (ins' := ins') (hone' := hone') n]

section snoc

variable {N₀ n₀ : ℕ} (p : FlockSoundness.Prog N₀ n₀) (R : FlockSoundness.Rows) (ins : Fin R.nIn → Fin N₀)
  (one : Fin N₀) (hone : ∀ c, ins c ≠ one)

theorem rowsWireC_last_val (x : Fin ((FlockSoundness.Prog.snoc p R ins one hone).rows (Fin.last n₀)).w)
    (hx : ¬ x.val < R.nIn) :
    ((Const.Prog.rowsWireC (FlockSoundness.Prog.snoc p R ins one hone) (Fin.last n₀)).wire x).val =
      if x.val < R.nIn + R.nComp then N₀ + (x.val - R.nIn) else one.val := by
  rw [Const.Prog.rowsWireC_last, Const.Prog.castW_wire]
  change (FlockSoundness.Prog.colWire R ins one _).val = _
  rw [FlockSoundness.Prog.colWire_val, Fin.val_cast, dif_neg hx]

theorem rowsWireC_castSucc_val (v : Fin n₀)
    (x : Fin ((FlockSoundness.Prog.snoc p R ins one hone).rows v.castSucc).w) :
    ((Const.Prog.rowsWireC (FlockSoundness.Prog.snoc p R ins one hone) v.castSucc).wire x).val =
      ((Const.Prog.rowsWireC p v).wire (Fin.cast (by rw [Const.Prog.rows_castSucc]) x)).val := by
  rw [Const.Prog.rowsWireC_castSucc, Const.Prog.castW_wire]
  rfl

end snoc

/-- **`uniform`'s instance puts a unit's computed columns and constant on fixed gates**: computed column `x` on gate
`gates k R u + (x - nIn)`, the constant on `one`. -/
theorem rowsWireC_uniform_val {ins : ℕ → Fin R.nIn → Fin k} {one : Fin k} {hone : ∀ u c, ins u c ≠ one} :
    ∀ (n : ℕ) (u : Fin n) (x : Fin ((Placed.uniform k R ins one hone n).rows u).w), ¬ x.val < R.nIn →
      ((Const.Prog.rowsWireC (Placed.uniform k R ins one hone n) u).wire x).val =
        if x.val < R.nIn + R.nComp then Placed.gates k R u.val + (x.val - R.nIn) else one.val
  | 0, u, _, _ => u.elim0
  | n + 1, u, x, hx => by
    revert x
    induction u using Fin.lastCases with
    | last =>
      intro x hx
      exact rowsWireC_last_val (Placed.uniform k R ins one hone n) R _ _ _ x hx
    | cast v =>
      intro x hx
      refine (rowsWireC_castSucc_val (Placed.uniform k R ins one hone n) R _ _ _ v x).trans ?_
      rw [rowsWireC_uniform_val n v _ hx]
      rfl

end uniform

/-! ## `keyProgP` beside `keyProg` -/

variable {c : Flock.Circuit} {hU : Built c.unit} {n : ℕ} {pi : HmIn.Ports}

theorem rows_keyProgP (u : Fin n) : (keyProgP c hU n pi).rows u = (keyProg c hU n).rows u :=
  (Placed.rows_circuitProg c hU n _ _ u).trans (Placed.rows_circuitProg c hU n _ _ u).symm

theorem nIn_keyProgP (u : Fin n) : ((keyProgP c hU n pi).rows u).nIn = keyB c hU :=
  congrArg FlockSoundness.Rows.nIn (Placed.rows_circuitProg c hU n _ _ u)

/-- `keyProgP` and `keyProg` have one partition into units. -/
theorem unit_keyProgP : (keyProgP c hU n pi).unit = (keyProg c hU n).unit :=
  unit_uniform_eq n

/-- **`keyProgP` and `keyProg` put a unit's computed columns and constant on one gate.** -/
theorem wire_keyProgP (outs outs' : Finset (Fin (keyN c hU n))) (u : Fin n) (x : Fin ((keyProgP c hU n pi).rows u).w)
    (hx : ¬ x.val < ((keyProgP c hU n pi).rows u).nIn) :
    (Const.Prog.instC (keyProgP c hU n pi) outs u).wire x =
      (Const.Prog.instC (keyProg c hU n) outs' u).wire (Fin.cast (congrArg FlockSoundness.Rows.w (rows_keyProgP u)) x) := by
  have hx' : ¬ x.val < (Placed.circuitRows c hU).nIn := by rw [nIn_keyProgP] at hx; exact hx
  apply Fin.ext
  change ((Const.Prog.rowsWireC (keyProgP c hU n pi) u).wire x).val =
    ((Const.Prog.rowsWireC (keyProg c hU n) u).wire _).val
  exact (rowsWireC_uniform_val (k := Placed.inGates n (keyB c hU)) (R := Placed.circuitRows c hU)
      (ins := Placed.insOf n (keyB c hU) (keyIdxP c pi (keyB c hU))) (one := Placed.oneGate n (keyB c hU))
      (hone := Placed.insOf_ne_one n (keyB c hU) (keyIdxP c pi (keyB c hU))) n u x hx').trans
    (rowsWireC_uniform_val (k := Placed.inGates n (keyB c hU)) (R := Placed.circuitRows c hU)
      (ins := Placed.insOf n (keyB c hU) (Aliased.keyIdx c (keyB c hU))) (one := Placed.oneGate n (keyB c hU))
      (hone := Placed.insOf_ne_one n (keyB c hU) (Aliased.keyIdx c (keyB c hU))) n u
      (Fin.cast (congrArg FlockSoundness.Rows.w (rows_keyProgP u)) x) hx').symm

/-- **`keyProgP`'s units read only input gates besides their own** (`Layout.keyProg_reads`). -/
theorem keyProgP_reads (outs : Finset (Fin (keyN c hU n))) (u : Fin n) (y : Fin ((keyProgP c hU n pi).rows u).w)
    (hy : ∀ i, y ≠ ((keyProgP c hU n pi).rows u).comp i) :
    ((keyProgP c hU n pi).partition outs).unit ((Const.Prog.instC (keyProgP c hU n pi) outs u).wire y) = none := by
  show (keyProgP c hU n pi).unit _ = none
  refine Layout.unit_uniform_input (k := Placed.inGates n (keyB c hU)) (R := Placed.circuitRows c hU)
    (ins := Placed.insOf n (keyB c hU) (keyIdxP c pi (keyB c hU))) (one := Placed.oneGate n (keyB c hU))
    (hone := Placed.insOf_ne_one n (keyB c hU) (keyIdxP c pi (keyB c hU))) n _ ?_
  by_cases hx : y.val < ((keyProgP c hU n pi).rows u).nIn
  · rw [Const.Prog.instC_wire_input _ _ _ y hx]
    have hxB : y.val < keyB c hU := lt_of_lt_of_eq hx (nIn_keyProgP u)
    have hv : (Const.Prog.insAt (keyProgP c hU n pi) u y.val).val =
        (Placed.insOf n (keyB c hU) (keyIdxP c pi (keyB c hU)) u.val ⟨y.val, hxB⟩).val :=
      Placed.insAt_uniform n u y.val hxB
    rw [hv]
    exact Fin.isLt _
  · have hone : y = ((keyProgP c hU n pi).rows u).one := by
      apply Fin.ext
      have hw : y.val < ((keyProgP c hU n pi).rows u).nIn + ((keyProgP c hU n pi).rows u).nComp + 1 := y.isLt
      by_contra hne
      have hi : y.val - ((keyProgP c hU n pi).rows u).nIn < ((keyProgP c hU n pi).rows u).nComp := by
        simp only [FlockSoundness.Rows.one] at hne
        omega
      exact hy ⟨_, hi⟩ (Fin.ext (by simp only [FlockSoundness.Rows.comp]; omega))
    rw [hone, Const.Prog.instC_wire_one]
    have h0 : (Const.Prog.oneAt (keyProgP c hU n pi) u).val = (Placed.oneGate n (keyB c hU)).val :=
      Layout.oneAt_uniform_val (k := Placed.inGates n (keyB c hU)) (R := Placed.circuitRows c hU)
        (ins := Placed.insOf n (keyB c hU) (keyIdxP c pi (keyB c hU))) (one := Placed.oneGate n (keyB c hU))
        (hone := Placed.insOf_ne_one n (keyB c hU) (keyIdxP c pi (keyB c hU))) n u
    rw [h0]
    exact Fin.isLt _

/-- **A unit's own column on a gate it computes** is a computed column, of that unit (`unit_wire`). -/
theorem unit_wireP (outs : Finset (Fin (keyN c hU n))) {u u' : Fin n} (x : Fin ((keyProgP c hU n pi).rows u).w)
    (h : (keyProgP c hU n pi).unit ((Const.Prog.instC (keyProgP c hU n pi) outs u).wire x) = some u') :
    u' = u ∧ ((keyProgP c hU n pi).rows u).nIn ≤ x.val := by
  by_cases hcomp : ∃ i, x = ((keyProgP c hU n pi).rows u).comp i
  · obtain ⟨i, rfl⟩ := hcomp
    have hm : (keyProgP c hU n pi).unit ((Const.Prog.instC (keyProgP c hU n pi) outs u).wire
        (((keyProgP c hU n pi).rows u).comp i)) = some u := (Const.Prog.instC (keyProgP c hU n pi) outs u).mem i
    rw [hm] at h
    refine ⟨(Option.some.inj h).symm, ?_⟩
    simp only [FlockSoundness.Rows.comp]
    omega
  · have hn : (keyProgP c hU n pi).unit ((Const.Prog.instC (keyProgP c hU n pi) outs u).wire x) = none :=
      keyProgP_reads outs u x fun i e => hcomp ⟨i, e⟩
    rw [hn] at h
    cases h

/-- **The output-row bit of a computed gate of `keyProgP`'s instance** is `keyProg`'s (`outRowOf_wire`). -/
theorem outRowOf_wireP (outs : Finset (Fin (keyN c hU n))) {u : Fin n} (x : Fin ((keyProgP c hU n pi).rows u).w)
    (hgu : (keyProgP c hU n pi).unit ((Const.Prog.instC (keyProgP c hU n pi) outs u).wire x) = some u)
    (hx : ((keyProgP c hU n pi).rows u).nIn ≤ x.val) :
    outRowOf c hU n outs ((Const.Prog.instC (keyProgP c hU n pi) outs u).wire x) =
      outBitRow c u (outDecode c hU x.val) := by
  have hx' : ¬ x.val < ((keyProgP c hU n pi).rows u).nIn := by omega
  rw [wire_keyProgP outs outs u x hx'] at hgu ⊢
  rw [unit_keyProgP] at hgu
  have hn : ((keyProg c hU n).rows u).nIn ≤
      (Fin.cast (congrArg FlockSoundness.Rows.w (rows_keyProgP (pi := pi) u)) x).val := by
    rw [← congrArg FlockSoundness.Rows.nIn (rows_keyProgP (pi := pi) u)]; exact hx
  exact outRowOf_wire outs _ hgu hn

theorem outRowOf_wire_someP (outs : Finset (Fin (keyN c hU n))) {u u' : Fin n}
    (x : Fin ((keyProgP c hU n pi).rows u).w) {q : Fin c.ports.size} {k : Fin (portRowW c q)}
    (h : outRowOf c hU n outs ((Const.Prog.instC (keyProgP c hU n pi) outs u).wire x) = some (u', ⟨q, k⟩)) :
    (keyProgP c hU n pi).unit ((Const.Prog.instC (keyProgP c hU n pi) outs u).wire x) = some u ∧
      ((keyProgP c hU n pi).rows u).nIn ≤ x.val ∧ outBitRow c u (outDecode c hU x.val) = some (u', ⟨q, k⟩) := by
  rcases hgu : (keyProgP c hU n pi).unit ((Const.Prog.instC (keyProgP c hU n pi) outs u).wire x) with _ | u''
  · rw [unit_keyProgP] at hgu
    unfold outRowOf at h
    rw [hgu] at h
    exact absurd h (by simp)
  · obtain ⟨rfl, hx⟩ := unit_wireP outs x hgu
    exact ⟨rfl, hx, (outRowOf_wireP outs x hgu hx).symm.trans h⟩

/-- **A public leaf's bit has the zero key**: its leaf of `leaves_in` is `-1` (`LeavesPublic.maps`), which Δ reads as
the zero (`Aliased.srcAt`). -/
theorem srcKey_of_pubLeaf (lp : LeavesPublic c pi) {x : ℕ} {v : ℕ × ℕ} (h : pubLeaf c pi x = some v) :
    Aliased.srcKey c x = .zero := by
  unfold pubLeaf at h
  dsimp only at h
  split at h
  · rename_i hb
    split at h
    · rename_i hq
      unfold Aliased.srcKey Aliased.srcAt
      rw [if_pos hb.1, if_pos hb.2]
      by_cases hu : x / (c.unit.inWords * WORD) < c.upv
      · obtain ⟨hsz, hall⟩ := lp.maps _ hu
        have hj : FlockSoundness.portOf (Flock.portCols c.unit.inGroups) (x % (c.unit.inWords * WORD)) <
            (pi.leaves.getD (x / (c.unit.inWords * WORD)) #[]).size := by
          by_contra hj
          rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none (by omega)] at hq
          simp at hq
        have h1 := (hall _ hj).2.2 hq
        rw [hsz] at hj
        rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hj, Option.getD_some] at h1
        rw [if_pos (by rw [getElem!_pos (c.leavesIn.getD (x / (c.unit.inWords * WORD)) #[]) _ hj, h1]; decide)]
      · exfalso
        rw [Array.getD_eq_getD_getElem? (xs := pi.leaves), Array.getElem?_eq_none (by rw [lp.size]; omega)] at hq
        simp at hq
    · cases h
  · cases h

/-- The first input with an input's key that is no public leaf's bit has that key; a public leaf's bit is its own. -/
theorem keyIdxP_some {m : ℕ} {x b : Fin m} (h : keyIdxP c pi m x = some b) :
    Aliased.srcKey c b.val = Aliased.srcKey c x.val := by
  cases hx : pubLeaf c pi x.val with
  | none => exact ((keyIdxP_none c pi m x hx).2 b h).2
  | some v =>
    have : keyIdxP c pi m x = some x := by unfold keyIdxP; simp [hx]
    rw [this] at h
    cases h
    rfl

/-- **A column of `keyProgP`'s own instance on a row bit** is an input column of that unit, whose Δ key is the row
bit's message bit (`Layout.keyRowOf_wire`). -/
theorem keyRowOf_wireP (outs : Finset (Fin (keyN c hU n))) {rowW : Fin c.ports.size → ℕ} {u u' : Fin n}
    (x : Fin ((keyProgP c hU n pi).rows u).w) {q : Fin c.ports.size} {k : Fin (rowW q)}
    (h : keyRowOf c hU n rowW ((Const.Prog.instC (keyProgP c hU n pi) outs u).wire x) = some (u', ⟨q, k⟩)) :
    u' = u ∧ x.val < ((keyProgP c hU n pi).rows u).nIn ∧
      Aliased.srcKey c x.val = .msg q.val (k.val / 8) (k.val % 8) := by
  by_cases hx : x.val < ((keyProgP c hU n pi).rows u).nIn
  · obtain ⟨h2, hlt, hu, hkey⟩ := Layout.keyRowOf_spec h
    have hxB : x.val < keyB c hU := lt_of_lt_of_eq hx (nIn_keyProgP u)
    refine ⟨?_, hx, ?_⟩ <;>
    · rw [Const.Prog.instC_wire_input _ _ _ x hx] at h2 hu hkey
      have hv : (Const.Prog.insAt (keyProgP c hU n pi) u x.val).val =
          (Placed.insOf n (keyB c hU) (keyIdxP c pi (keyB c hU)) u.val ⟨x.val, hxB⟩).val :=
        Placed.insAt_uniform n u x.val hxB
      rw [hv, Placed.insOf_of_lt (nU := n) (B := keyB c hU) (src := keyIdxP c pi (keyB c hU)) u.isLt] at h2 hu hkey
      rcases hb : keyIdxP c pi (keyB c hU) ⟨x.val, hxB⟩ with _ | b
      · rw [hb] at h2
        exact absurd h2 (by simp [Placed.srcGate, Placed.zeroGate])
      · rw [hb] at hu hkey
        simp only [Placed.srcGate] at hu hkey
        have hB : 0 < keyB c hU := lt_of_le_of_lt (Nat.zero_le _) hxB
        have hd : (2 + u.val * keyB c hU + b.val - 2) / keyB c hU = u.val := by
          rw [show 2 + u.val * keyB c hU + b.val - 2 = b.val + u.val * keyB c hU by omega,
            Nat.add_mul_div_right _ _ hB, Nat.div_eq_of_lt b.isLt, Nat.zero_add]
        have hm : (2 + u.val * keyB c hU + b.val - 2) % keyB c hU = b.val := by
          rw [show 2 + u.val * keyB c hU + b.val - 2 = b.val + u.val * keyB c hU by omega,
            Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt b.isLt]
        first
          | exact Fin.ext (hu.trans hd)
          | rw [hm, keyIdxP_some hb] at hkey; exact hkey
  · rw [wire_keyProgP outs outs u x hx] at h
    have := (Layout.keyRowOf_wire outs _ h).2.1
    rw [← congrArg FlockSoundness.Rows.nIn (rows_keyProgP (pi := pi) u)] at this
    exact absurd this hx

/-- **A row bit of a unit's own instance of `keyProgP` is that unit's row** (`rowOfH_wire_unit`). -/
theorem rowOfH_wire_unitP (outs : Finset (Fin (keyN c hU n))) {u u' : Fin n}
    (x : Fin ((keyProgP c hU n pi).rows u).w) {q : Fin c.ports.size} {k : Fin (portRowW c q)}
    (h : rowOfH c hU n outs ((Const.Prog.instC (keyProgP c hU n pi) outs u).wire x) = some (u', ⟨q, k⟩)) :
    u' = u := by
  rcases hk : keyRowOf c hU n (portRowW c) ((Const.Prog.instC (keyProgP c hU n pi) outs u).wire x) with _ | r
  · rw [rowOfH_of_none outs hk] at h
    exact (outBitRow_spec (outRowOf_wire_someP outs x h).2.2).1
  · rw [rowOfH_of_key outs hk] at h
    rw [h] at hk
    exact (keyRowOf_wireP outs x hk).1


theorem mem_io_of_unit {C : Audit.Circuit} {m : ℕ} {P : Audit.Partition C m} {u : Fin m} {g : Fin C.N}
    (hc : g ∈ P.committed) (hu : P.unit g = some u) : g ∈ P.io u := by
  classical
  simp only [Audit.Partition.io, Finset.mem_filter]
  exact ⟨hc, Or.inl hu⟩

/-- **`OutsOut` at `keyProgP`**: a committed gate a unit of `keyProgP` computes is an output
(`Layout.computed_committed_output`, `keyProgP_reads`), so a committed gate `keyProg`'s unit computes on the same
computed column (`wire_keyProgP`), which `OutsOut` puts on a returned output's column. -/
theorem outsOut_keyProgP {outs : Finset (Fin (keyN c hU n))} (houts : OutsOut c hU n outs) (u : Fin n)
    (g : Fin (keyN c hU n)) (hg : g ∈ ((keyProgP c hU n pi).partition outs).io u)
    (hgu : (keyProgP c hU n pi).unit g = some u) (x : Fin ((keyProgP c hU n pi).rows u).w)
    (hx : (Const.Prog.instC (keyProgP c hU n pi) outs u).wire x = g) :
    ∃ u' j t, OutColAt c hU x.val u' j t := by
  have hout : g ∈ outs :=
    Layout.computed_committed_output (fun u => Const.Prog.instC (keyProgP c hU n pi) outs u)
      (fun u y hy => keyProgP_reads outs u y hy) hg hgu
  have hgu' : (keyProg c hU n).unit g = some u := by rw [← unit_keyProgP (pi := pi)]; exact hgu
  have hio : g ∈ ((keyProg c hU n).partition outs).io u :=
    mem_io_of_unit (Audit.Partition.mem_committed_of_output _ hout) hgu'
  subst hx
  have hxn := (unit_wireP outs x hgu).2
  rw [wire_keyProgP outs outs u x (by omega)] at hio hgu'
  exact houts u _ hio hgu' _ rfl

/-- **A returned output's column holds its output row's bit** at an accepted statement with public inputs
(`out_copy_key` with `out_copyHP`). -/
theorem out_copy_keyP {T : Model.Statement} (a : AcceptedHP T) (hce : a.st.c = c) (ho : c.outNet = c.unitNet)
    (hv1 : ∀ p ∈ c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks)
    {z : Witness T.m} (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl)) {g : ℕ} (hg : g < a.st.c.g) (x : Fin a.rows.w)
    {u' j t : ℕ} (hoc : OutColAt c hU x.val u' j t) {u v : Fin n} {q : Fin c.ports.size} {k : Fin (portRowW c q)}
    (hb : outBitRow c u (u', j, t) = some (v, ⟨q, k⟩)) :
    T.block z o (a.col hg x) = z (siteAddrHP a o (Typed.msgCol a.st.c g q.val (k.val / 8) (k.val % 8))) := by
  subst hce
  obtain ⟨-, hq, hk⟩ := outBitRow_spec hb
  obtain ⟨hu', hj, ht, h1, h2, hx⟩ := hoc
  have P := parse_facts a.parse
  obtain ⟨ls, hls⟩ := P.check
  obtain ⟨-, -, hlo⟩ := Layout.check_out P.typed ho hls
  have hget : a.st.c.leavesOut.getD u' #[] = a.st.c.leavesOut[u']! := by
    rw [Hidden.getElem!_eq_getD']; rfl
  have hwj : outWordOf a.st.c (u', j, t) = (a.st.c.leavesOut.getD u' #[])[j] := getElem!_pos _ j hj
  have hwlt : outWordOf a.st.c (u', j, t) < a.st.c.outWords := by
    rw [hwj]
    refine (hlo u' hu').2 _ ?_
    rw [← hget]
    exact Array.getElem_mem_toList _
  obtain ⟨hjo, hle, hkk⟩ := outPortOf_spec a.st.c hwlt
  have hw : Hidden.outAt a.st.c (outPortOf a.st.c (outWordOf a.st.c (u', j, t))) +
      (outWordOf a.st.c (u', j, t) - Hidden.outAt a.st.c (outPortOf a.st.c (outWordOf a.st.c (u', j, t)))) =
      (a.st.c.leavesOut.getD u' #[])[j] := by
    rw [← hwj]
    omega
  have := out_copyHP a ho hv1 hz o hg x hu' hj ht h1 h2 hx hjo hkk hw
  rw [hq, hk]
  exact this

end FlockSoundness.Discharge.ZkPubIn
