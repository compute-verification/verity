module

public import Proofs.Flock.SpotFlock.RowsOpen
public import Proofs.Flock.SpotFlock.UnitRows

@[expose] public section

/-!
# A placed unit, read off its session's block circuit

`valueAt_unitPlace`: at a column of a unit placed in table `j` of a parsed statement at block `o` (`unitPlace`), under
a witness of the deployed relation, the block circuit's value at that column's input gate (`valueAt_relation`) is the
value the unit's block decodes at the column's wire.

`embeds_unitPlace`: so at any wire map that sends each wire the unit computes or reads to the input gate of one of its
columns, the block circuit's values hold each of the unit's gates (`UnitPlace.apply_decode`) and carry the lowering's
constants (`decode_consts`). That is what `Embeds` asks of a checked unit (`unitVals`), where the layout's wires are
such a map.

`shape_of_checked`: a checked proof block, replayed in request `ρ`, is member `t`'s unit `k` (`member_of_replayOf`),
and in Bridge's terms it is the key program's block `k`, `Bits.base` later, whose lowering sits in the statement's
program (`sitsAt_member`): the shape half of `Embeds`. `parsed_of_stmtOkAt`: a checked session's statement is the
readable form of a parsed statement that passes its check with hidden outputs, the session `embeds_unitPlace` reads.
-/

namespace SpotFlock

open FlockSoundness FlockSoundness.Discharge Exec Flock Flock.Readable

variable {c : _root_.Flock.Circuit} (hU : Built c.unit) {n : ℕ}
  (outs : Finset (Fin (Placed.gates (Placed.inGates n (Placed.circuitRows c hU).nIn) (Placed.circuitRows c hU) n)))
  {T : Model.Statement} (a : HiddenExec.AcceptedH T) (hparse : HmOut.unrolled a.st.c = c)

include hparse in
/-- **The session's value at a placed unit's column is the unit's decoded value**: for a unit placed in table `j` of
a parsed statement at block `o` (`unitPlace`), under a witness of the deployed relation whose table-`j` share the
accepted statement's block R1CS holds, the block circuit's value at the input gate of the unit's column `cc` is the
value the unit's block decodes at `cc`'s wire. -/
theorem valueAt_unitPlace {p : Parsed} {j : ℕ} (hj : j < p.sts.size) (hk : T.kLog = p.sts[j].c.kLog) (u : Fin n)
    {g : ℕ} (hg : g < a.st.c.g) (o : Fin (2 ^ T.nbl)) {w : Claims.Flock.Wit}
    (hR : Claims.Flock.RelS (toReadable relation p) w) (hz : T.Satisfies (zOf p.sts.size w j T.m))
    (cc : Fin (unitPlace hU outs a hparse u hg o).R.w) :
    valueAt (toReadable relation p) o.val w
        (j * 2 ^ (toReadable relation p).kLog + ((unitPlace hU outs a hparse u hg o).col cc).val) =
      (unitPlace hU outs a hparse u hg o).decode (zOf p.sts.size w j T.m)
        ((unitPlace hU outs a hparse u hg o).inst.wire cc) := by
  obtain ⟨hfit, -⟩ := (relS_relation p w).1 hR j hj
  have hk0 : p.sts[j]!.c.kLog = (toReadable relation p).kLog := hfit.2.1
  have hk' : T.kLog = p.sts[j]!.c.kLog := by rw [hk, getElem!_pos p.sts j hj]
  have hcol : ((unitPlace hU outs a hparse u hg o).col cc).val < 2 ^ (toReadable relation p).kLog :=
    lt_of_lt_of_eq ((unitPlace hU outs a hparse u hg o).col cc).isLt (by rw [hk', hk0])
  rw [valueAt_relation o.val w hj hcol]
  show _ = (unitPlace hU outs a hparse u hg o).inst.decode _ ((unitPlace hU outs a hparse u hg o).inst.wire cc)
  rw [(unitPlace hU outs a hparse u hg o).inst.decode_wire ((unitPlace hU outs a hparse u hg o).agree hz) cc]
  have hb := bit_block_zOf p w j hk' o ((unitPlace hU outs a hparse u hg o).col cc)
  unfold Flock.InnerRows.bitAt at hb
  rw [hk0] at hb
  exact hb.symm

include hparse in
/-- **A placed unit's gates hold at its session's values**: for a unit placed in table `j` of a parsed statement at
block `o`, under a witness of the deployed relation whose table-`j` share the accepted statement's block R1CS holds,
and a wire map `W` that sends each wire the unit computes or reads to the input gate of one of its columns, the block
circuit's values at `W` hold each of the unit's gates and carry the lowering's constants (`Bridge.constsOf`) wherever
the unit reads them. -/
theorem embeds_unitPlace {p : Parsed} {j : ℕ} (hj : j < p.sts.size) (hk : T.kLog = p.sts[j].c.kLog) (u : Fin n)
    {g : ℕ} (hg : g < a.st.c.g) (o : Fin (2 ^ T.nbl)) {w : Claims.Flock.Wit}
    (hR : Claims.Flock.RelS (toReadable relation p) w) (hz : T.Satisfies (zOf p.sts.size w j T.m))
    (W : Fin (Discharge.Layout.keyN c hU n) → ℕ)
    (hW : ∀ x, (((Placed.keyProg c hU n).partition (Discharge.Layout.outsOf c hU n)).unit x = some u ∨
        ∃ g', ((Placed.keyProg c hU n).partition (Discharge.Layout.outsOf c hU n)).unit g' = some u ∧
          x ∈ (((Placed.keyProg c hU n).circuit (Discharge.Layout.outsOf c hU n)).op g').args) →
      ∃ cc, (unitPlace hU (Discharge.Layout.outsOf c hU n) a hparse u hg o).inst.wire cc = x ∧
        W x = j * 2 ^ (toReadable relation p).kLog +
          ((unitPlace hU (Discharge.Layout.outsOf c hU n) a hparse u hg o).col cc).val) :
    (∀ x, ((Placed.keyProg c hU n).partition (Discharge.Layout.outsOf c hU n)).unit x = some u →
      valueAt (toReadable relation p) o.val w (W x) =
        (((Placed.keyProg c hU n).circuit (Discharge.Layout.outsOf c hU n)).op x).apply
          fun y => valueAt (toReadable relation p) o.val w (W y)) ∧
    ∀ x b, Bridge.constsOf c hU n x = some b →
      (∃ g', ((Placed.keyProg c hU n).partition (Discharge.Layout.outsOf c hU n)).unit g' = some u ∧
        x ∈ (((Placed.keyProg c hU n).circuit (Discharge.Layout.outsOf c hU n)).op g').args) →
      valueAt (toReadable relation p) o.val w (W x) = b := by
  have hval : ∀ x, (((Placed.keyProg c hU n).partition (Discharge.Layout.outsOf c hU n)).unit x = some u ∨
      ∃ g', ((Placed.keyProg c hU n).partition (Discharge.Layout.outsOf c hU n)).unit g' = some u ∧
        x ∈ (((Placed.keyProg c hU n).circuit (Discharge.Layout.outsOf c hU n)).op g').args) →
      valueAt (toReadable relation p) o.val w (W x) =
        (unitPlace hU (Discharge.Layout.outsOf c hU n) a hparse u hg o).decode (zOf p.sts.size w j T.m) x := by
    intro x hx
    obtain ⟨cc, rfl, hWx⟩ := hW x hx
    rw [hWx]
    exact valueAt_unitPlace hU _ a hparse hj hk u hg o hR hz cc
  refine ⟨fun x hu => ?_, fun x b hb hr => ?_⟩
  · rw [hval x (Or.inl hu), (unitPlace hU (Discharge.Layout.outsOf c hU n) a hparse u hg o).apply_decode hz x hu]
    exact Verity.Boolean.Op.apply_congr _ fun y hy => (hval y (Or.inr ⟨x, hu, hy⟩)).symm
  · rw [hval x (Or.inr hr)]
    exact decode_consts hU a hparse u hg o hz x b hb hr

/-! ## A checked proof block is a member's unit -/

/-- **Member `t`'s unit `k`, in Bridge's terms, sits in the members' program**: its proof block in the members'
partition is its circuit file's proof block `k` of the key program's partition at the circuit's constants,
`Bits.base ms t` later, and that key program's lowering sits in `bitsProgs ms` from `Bits.base ms t`. -/
theorem sitsAt_member (ms : Spot.Bits.Members)
    (htop : ∀ ci ∈ ms, (Properties.Spot.memberProg ci.1 ci.2.size).Topological) {t : ℕ} (ht : t < ms.length)
    (hU : Built ms[t].1.unit) (hu : ms[t].1.unit.unitLog ≤ 32) (k : Fin ms[t].2.size) {b : List ℕ}
    (hb : (Properties.Spot.membersPart ms).blocks[Properties.Spot.unitBase ms t + k]? = some b) :
    _root_.Flock.Bridge.Members.SitsAt (Properties.Spot.bitsProgs ms) (Spot.Bits.base ms t)
        (_root_.Flock.Bridge.progOf
          ((Placed.keyProg ms[t].1 hU ms[t].2.size).circuit (Discharge.Layout.outsOf ms[t].1 hU ms[t].2.size))
          (Bridge.constsOf ms[t].1 hU ms[t].2.size)) ∧
      b = (_root_.Flock.Bridge.blockOf
          ((Placed.keyProg ms[t].1 hU ms[t].2.size).partition (Discharge.Layout.outsOf ms[t].1 hU ms[t].2.size))
          (Bridge.constsOf ms[t].1 hU ms[t].2.size) k).map (Spot.Bits.base ms t + ·) := by
  refine ⟨?_, ?_⟩
  · rw [_root_.Flock.Bridge.progOf_keyProg hu]
    exact _root_.Flock.Bridge.Members.sitsAt_bitsProgs ms htop t ht
  · rw [_root_.Flock.Bridge.Members.getElem?_blocks_membersPart_bridge ms ht (hU := hU) hu k] at hb
    exact (Option.some.inj hb).symm

variable (h : Held)

/-- **A checked proof block is a member's unit**: at a statement whose partition and two-stage design evaluate, in
the two-stage scope, proof block `u`, replayed in request `ρ`, is member `t`'s unit `ρ·n_t + p` for some `p < n_t`,
which member `t` has (`filesOk`). -/
theorem member_of_replayOf {s : Claims.Sampling.AuditStatement} {Pt : Claims.Sampling.Partition}
    {d : Claims.Sampling.Design.Two} {u ρ : ℕ} (hP : SpotFlock.partition h s = some Pt)
    (hd : SpotFlock.two h s = some d) (hin : Claims.Sampling.InScopeTwo Pt d s) (hu : u < Pt.blocks.length)
    (hρ : Claims.Sampling.replayOf Pt d.part u = some ρ) :
    ∃ t, ∃ _ : t < h.ms.length, ∃ p, p < h.reg.ns.getD t 0 ∧ ρ * h.reg.ns.getD t 0 + p < h.ms[t].2.size ∧
      u = Properties.Spot.unitBase h.ms t + (ρ * h.reg.ns.getD t 0 + p) := by
  obtain rfl := SpotFlock.partition_eq h hP
  obtain ⟨l, -, -, hdd, -, hfo⟩ := SpotFlock.two_eq h hd
  obtain ⟨-, -, hfit, -⟩ := SpotFlock.filesOk_sound hfo
  have hpart : d.part = Properties.Spot.filesReplay h.ms h.reg.ns l := congrArg Claims.Sampling.Design.Two.part hdd
  have hρl : ρ < l.blocks.size := by
    have hf := hρ
    unfold Claims.Sampling.replayOf at hf
    have := List.mem_range.1 (List.mem_of_find?_eq_some hf)
    rwa [hpart, _root_.Flock.Bridge.Members.length_blocks_filesReplay] at this
  rw [hpart] at hρ
  have hw := (_root_.Flock.Bridge.Members.replayOf_filesReplay h.ms hin.2.1.1 hin.2.1.2.2.2.1 hfit hu hρl).1 hρ
  obtain ⟨hroom, t, ht, p, hp, rfl⟩ := _root_.Flock.Bridge.Members.exists_of_mem_slotFiles h.ms hfit hw
  exact ⟨t, ht, p, hp,
    lt_of_lt_of_le (_root_.Flock.Bridge.Members.rank_mul_add_lt (Nat.lt_add_one ρ) hp) (hroom t ht), rfl⟩

/-- **A checked proof block, in Bridge's terms**: at a statement whose partition and two-stage design evaluate, in the
two-stage scope, where each member's unit net is built with `unitLog ≤ 32`, a checked proof block `b`, replayed in
request `ρ`, is member `t`'s unit `k`: Bridge's proof block `k` of the member's key program, `Bits.base h.ms t` later,
whose lowering sits in the statement's program from `Bits.base h.ms t`. -/
theorem shape_of_checked
    (hmem : ∀ t (ht : t < h.ms.length), Built h.ms[t].1.unit ∧ h.ms[t].1.unit.unitLog ≤ 32) {x}
    {s : Claims.Sampling.AuditStatement} {Pt : Claims.Sampling.Partition} {d : Claims.Sampling.Design.Two}
    {u ρ : ℕ} {b : List ℕ} (hx : SpotFlock.parse h x = some s) (hP : SpotFlock.partition h s = some Pt)
    (hd : SpotFlock.two h s = some d) (hin : Claims.Sampling.InScopeTwo Pt d s) (hb : Pt.blocks[u]? = some b)
    (hρ : Claims.Sampling.replayOf Pt d.part u = some ρ) :
    ∃ t, ∃ ht : t < h.ms.length, ∃ k : Fin h.ms[t].2.size, u = Properties.Spot.unitBase h.ms t + k ∧
      _root_.Flock.Bridge.Members.SitsAt s.prog (Spot.Bits.base h.ms t)
        (_root_.Flock.Bridge.progOf ((Placed.keyProg h.ms[t].1 (hmem t ht).1 h.ms[t].2.size).circuit
          (Discharge.Layout.outsOf h.ms[t].1 (hmem t ht).1 h.ms[t].2.size))
          (Bridge.constsOf h.ms[t].1 (hmem t ht).1 h.ms[t].2.size)) ∧
      b = (_root_.Flock.Bridge.blockOf ((Placed.keyProg h.ms[t].1 (hmem t ht).1 h.ms[t].2.size).partition
          (Discharge.Layout.outsOf h.ms[t].1 (hmem t ht).1 h.ms[t].2.size))
          (Bridge.constsOf h.ms[t].1 (hmem t ht).1 h.ms[t].2.size) k).map (Spot.Bits.base h.ms t + ·) := by
  have htop : ∀ ci ∈ h.ms, (Properties.Spot.memberProg ci.1 ci.2.size).Topological := by
    intro ci hci
    obtain ⟨t, ht, rfl⟩ := List.mem_iff_getElem.1 hci
    exact _root_.Flock.Bridge.Members.topological_memberProg h.ms[t].1 (hU := (hmem t ht).1) (hmem t ht).2 _
  obtain ⟨t, ht, p, -, hk, rfl⟩ := member_of_replayOf h hP hd hin (List.getElem?_eq_some_iff.1 hb).1 hρ
  obtain rfl := SpotFlock.partition_eq h hP
  obtain ⟨hsit, hbeq⟩ := sitsAt_member h.ms htop ht (hmem t ht).1 (hmem t ht).2 ⟨_, hk⟩ hb
  rw [← SpotFlock.parse_prog h hx] at hsit
  exact ⟨t, ht, ⟨_, hk⟩, rfl, hsit, hbeq⟩

/-- **A checked session statement, parsed**: at a run that passed round 2, the session statement the claims' parser
reads is the readable form of the statement the verifier's parser returns (`parsed_setupOf`), whose statement check
passes and whose outputs are hidden (`sessionScoped`), as `accepted_check` reads it. -/
theorem parsed_of_stmtOkAt {x m₁ xS c₀ c₁ t} {S : Claims.Flock.Stmt}
    (hok : SpotFlock.stmtOkAt h x m₁ xS c₀ c₁ t = true) (hS : FlockDeploy.parse true relation xS = some S) :
    ∃ p : Parsed, parseBytes (FlockDeploy.readOf true) xS = some p ∧ S = toReadable relation p ∧ Parses p ∧
      p.check = .ok () ∧ p.tags.hiddenOutputs = true := by
  unfold FlockDeploy.parse parseOf at hS
  obtain ⟨p, hp, rfl⟩ := Option.map_eq_some_iff.1 hS
  have hs := sessionOk_of_stmtOkAt h hok
  simp only [sessionOk, Bool.and_eq_true] at hs
  obtain ⟨ins, spec, sts, hset, -⟩ := zkOk_sound hs.1.1.1
  obtain ⟨rfl, hpp⟩ := parsed_setupOf hset hp
  obtain ⟨hc, hh⟩ := sessionScoped h x m₁ xS c₀ c₁ t hok ins spec sts hset
  exact ⟨_, hp, rfl, hpp, hc, hh⟩

end SpotFlock
