module

public import Proofs.Flock.SpotFlock.WiringOk
public import Proofs.Flock.SpotFlock.ReadsDesign
public import Proofs.Flock.SpotFlock.Placement

/-!
# A needed gate the wiring sends inside its replay block stays inside

`inside_of`: the fact `Inside` that the entry and one tree a gate ask of the held code (`Wiring.inside`), from three
facts of the members' programs and the accepted run's own checks.

A needed gate the wiring sends to a gate `y` the boundary record doesn't hold, no input, is sent to a written bit
of a unit of the run's replay block: its own unit's, or the unit its wire names. The run's design check holds each
wire's rows a replay block to its writer's units a block (`SecondStage.wiresOk`), so a wire's writer is a unit of
its reader's replay block (`wireUnit_slot`). Every gate that reads `y` through the wiring is then a gate of that
replay block the wiring leaves alone (`readers_inside`): a gate that reads `y` itself is a gate of `y`'s unit, the
members' proof blocks standing side by side, and a gate that reads a source bit wired to `y` is a gate of that
source bit's unit, of the same replay block as `y`'s. So the connected program doesn't commit `y` as the replay
block sees it (`Prog.boundary_wired_eq_false`).

The facts of the members' programs it takes:

* `SideBySide`: each unit's gates read only that unit's gates and constants;
* `BitsInUnits`: a source bit's gate and a written bit's gate are gates of their unit's proof block;
* `WrittenOnce`: a gate is at one written position at most.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy
open Properties.Spot (bitsProgs membersPart memberProg unitBase filesReplay)

variable (h : Held)

/-- **Each bit is a gate of its unit's proof block**: a source bit's gate and a written bit's gate, at a commitment
position of leaf `u` of member `t`, are gates of the proof block of member `t`'s unit `u`, one of the member's
units. -/
def BitsInUnits : Prop :=
  ∀ (x t : ℕ) (rp : Spot.Bits.RowPos), (x, t, rp) ∈ sourceBits h.ms ∨ (x, t, rp) ∈ writtenBits h.ms →
    ∃ ht : t < h.ms.length, rp.leaf < h.ms[t].2.size ∧
      ∃ b, (membersPart h.ms).blocks[unitBase h.ms t + rp.leaf]? = some b ∧ x ∈ b

variable {h}

/-- **A member's unit is a position of a replay block's cell**: where the partition and the design evaluate in the
two-stage scope, unit `u` of member `t` is `ρ · n_t + p` for a replay block `ρ` and a position `p < n_t`. -/
theorem unit_split {s : AuditStatement} {Pt : Partition} {d : Design.Two} (hP : partition h s = some Pt)
    (hd : two h s = some d) (hin : InScopeTwo Pt d s) {t u : ℕ} (ht : t < h.ms.length)
    (hu : u < h.ms[t].2.size) : ∃ ρ p, p < h.reg.ns.getD t 0 ∧ u = ρ * h.reg.ns.getD t 0 + p := by
  have hlen : unitBase h.ms t + u < Pt.blocks.length := by
    rw [partition_eq h hP]
    exact _root_.Flock.Bridge.Members.unitBase_add_lt h.ms ht hu
  obtain ⟨ρ, hρ⟩ := Option.isSome_iff_exists.1 (hin.2.2.2.1 _ hlen)
  obtain ⟨t₂, ht₂, p, hp, hk, he⟩ := member_of_replayOf h hP hd hin hlen hρ
  have e : t₂ = t := by
    have h1 := templateOf_unitBase ht₂ hk
    rw [← he, templateOf_unitBase ht hu] at h1
    exact h1.symm
  subst e
  exact ⟨ρ, p, hp, Nat.add_left_cancel he⟩

/-- **The wires' check gives each wire its writer's part**: where the wires are derived over parts `ps`
(`SecondStage.wiresOk`), each wire's writer is the template of one of the parts, whose units a block are the rows a
block of the wire's reader's port. -/
theorem wiresOk_writer {ps : Array Spot.SecondStage.Part} {own : Spot.SecondStage.Own}
    (hok : Spot.SecondStage.wiresOk ps own = true) {w : Spot.SecondStage.Wire} (hw : w ∈ own.wires.getD #[]) :
    ∃ p ∈ ps, (p.template == w.writer) = true ∧
      ((Spot.SecondStage.rowsOf own w.template w.port).any (·.per == p.n)) = true := by
  unfold Spot.SecondStage.wiresOk at hok
  cases hws : own.wires with
  | none =>
    rw [hws] at hok
    cases hok
  | some ws =>
    simp only [hws, Option.getD_some] at hok hw
    have h1 := Array.all_eq_true_iff_forall_mem.1 hok w hw
    simp only [Bool.and_eq_true] at h1
    obtain ⟨p, hp, hpw⟩ := Array.any_eq_true'.1 h1.1.1.2
    exact ⟨p, hp, Bool.and_eq_true_iff.1 hpw⟩

namespace CheckedRun

variable {r : CheckedRun h.stage}

/-- **A unit of the run's replay block has its gates in the block**: at an accepted run, the gates of the proof
block of member `t`'s unit at position `p < n_t` of the run's replay block `ρ` are gates of that replay block. -/
theorem mem_rb (hr : r.Ok h) {t p : ℕ} (ht : t < h.ms.length) (hp : p < h.reg.ns.getD t 0) {b : List ℕ}
    (hb : (membersPart h.ms).blocks[unitBase h.ms t + (r.ρ * h.reg.ns.getD t 0 + p)]? = some b) {g : ℕ}
    (hg : g ∈ b) : g ∈ r.rb := by
  obtain ⟨Pt, d, spec, hx, hP, h2, hin, hok, hu, hck, hb0, hρ, hrb, hset⟩ := hr
  obtain rfl := partition_eq h hP
  obtain ⟨l, -, -, hdd, -, hfo⟩ := two_eq h h2
  obtain ⟨-, -, hfit, -⟩ := filesOk_sound hfo
  have hpart : d.part = filesReplay h.ms h.reg.ns l := congrArg Design.Two.part hdd
  rw [hpart] at hρ hrb
  have hρlen : r.ρ < (filesReplay h.ms h.reg.ns l).blocks.length := (List.getElem?_eq_some_iff.1 hrb).1
  have hρl : r.ρ < l.blocks.size := by
    rw [_root_.Flock.Bridge.Members.length_blocks_filesReplay] at hρlen
    exact hρlen
  have hul : r.u < (membersPart h.ms).blocks.length := (List.getElem?_eq_some_iff.1 hb0).1
  have hw := (_root_.Flock.Bridge.Members.replayOf_filesReplay h.ms hin.2.1.1 hin.2.1.2.2.2.1 hfit hul hρl).1 hρ
  have hrb' := _root_.Flock.Bridge.Members.getElem?_blocks_filesReplay h.ms h.reg.ns l hρlen
  rw [hrb] at hrb'
  rw [Option.some.inj hrb']
  refine List.mem_flatMap.2 ⟨unitBase h.ms t + (r.ρ * h.reg.ns.getD t 0 + p), ?_, ?_⟩
  · unfold _root_.Flock.Bridge.Members.slotFiles at hw ⊢
    rcases hlb : l.blocks[r.ρ]? with _ | sb
    · simp [hlb] at hw
    · rcases hsu : sb.unit with _ | u0
      · simp [hlb, hsu] at hw
      · simp only [hsu, Option.bind_some]
        rw [(hfit r.ρ sb hlb u0 hsu).1]
        exact (_root_.Flock.Bridge.Members.mem_filesUnits h.ms).2 ⟨t, ht, p, hp, rfl⟩
  · show g ∈ (membersPart h.ms).blocks.getD (unitBase h.ms t + (r.ρ * h.reg.ns.getD t 0 + p)) []
    rw [List.getD_eq_getElem?_getD, hb, Option.getD_some]
    exact hg

/-- **A wire's rows a replay block are its writer's units a replay block**, at an accepted run: the design's check
(`SecondStage.wiresOk`) holds the rows of the wire's reader's port to the units a block of a part of the writer's
name, and the templates' names are distinct (`namesOnce`), so that part is the writer's. -/
theorem per_of_wire (hr : r.Ok h) {name port : String} {w : Spot.SecondStage.Wire} {rr : Spot.SecondStage.Rows}
    {tw : ℕ} (hw : heldWire h name port = some w) (hrr : heldRows h name port = some rr)
    (htw : heldWriter h w.writer = some tw) : rr.per = h.reg.ns.getD tw 0 := by
  obtain ⟨Pt, d, spec, hx, hP, h2, hin, hok, hu, hck, hb0, hρ, hrb, hset⟩ := hr
  have hat := stmtOkAt_unit h hx hok hu
  unfold stmtOkAt at hat
  rw [hx] at hat
  simp only [Bool.and_eq_true] at hat
  obtain ⟨st, rec, draw, -, -, -, hdes⟩ := design_of_readsOk hat.2
  simp only [Spot.SecondStage.designOk, Bool.and_eq_true] at hdes
  have hwm : w ∈ h.wires.getD #[] := Array.mem_of_find?_eq_some hw
  have hwp := Array.find?_some hw
  have hwt : w.template = name := eq_of_beq (Bool.and_eq_true_iff.1 hwp).1
  have hwq : w.port = port := eq_of_beq (Bool.and_eq_true_iff.1 hwp).2
  obtain ⟨p, hp, hpt, hpr⟩ := wiresOk_writer hdes.2 hwm
  obtain ⟨t₂, ht₂, hpe⟩ :
      ∃ t₂, t₂ < h.ms.length ∧ partAt h (ownOf h st rec draw r.c₀).sample.size t₂ = p := by
    have hp' : p ∈ (List.range h.ms.length).map (partAt h (ownOf h st rec draw r.c₀).sample.size) :=
      Array.mem_toList_iff.2 hp
    obtain ⟨t₂, ht₂m, hpe⟩ := List.mem_map.1 hp'
    exact ⟨t₂, List.mem_range.1 ht₂m, hpe⟩
  subst hpe
  have hname : h.names.getD t₂ "" = w.writer := eq_of_beq hpt
  have hany : ((heldRows h name port).any (·.per == h.reg.ns.getD t₂ 0)) = true := by
    rw [← hwt, ← hwq]
    exact hpr
  simp only [hrr, Option.any_some, beq_iff_eq] at hany
  have htw' : (List.range h.ms.length).find? (fun t => h.names.getD t "" == w.writer) = some tw := htw
  have htwlt : tw < h.ms.length := List.mem_range.1 (List.mem_of_find?_eq_some htw')
  have h0 := List.find?_some htw'
  have h1 : (h.names.getD tw "" == w.writer) = true := h0
  have htwn : h.names.getD tw "" = w.writer := eq_of_beq h1
  have he : t₂ = tw :=
    List.inj_on_of_nodup_map (names_nodup_of_two h h2) (List.mem_range.2 ht₂) (List.mem_range.2 htwlt)
      (hname.trans htwn.symm)
  rw [he] at hany
  exact hany

/-- **A wired read's writer is a unit of its reader's replay block**: at an accepted run, where the unit at
position `p < n_t` of replay block `ρ`'s cell of member `t` reads, at a wired port, unit `u'` of template `tw`
(`wireUnit`), `u'` is a position of replay block `ρ`'s cell of `tw`. -/
theorem wireUnit_slot (hr : r.Ok h) {t : ℕ} {port : String} {ρ p : ℕ} (hp : p < h.reg.ns.getD t 0)
    {y : ℕ × String × ℕ} (hy : wireUnit h t port (ρ * h.reg.ns.getD t 0 + p) = some y) :
    ∃ c, c < h.reg.ns.getD y.1 0 ∧ y.2.2 = ρ * h.reg.ns.getD y.1 0 + c := by
  rw [Nat.add_comm (ρ * h.reg.ns.getD t 0) p, wireUnit_block h port ρ hp] at hy
  obtain ⟨z, hz, rfl⟩ := Option.map_eq_some_iff.1 hy
  obtain ⟨w, rr, -, hw, hrows, htw, -, hcell⟩ := wireCell_wire h hz
  have hper := per_of_wire hr hw hrows htw
  unfold cellRow at hcell
  obtain ⟨x, -, hx⟩ := Option.bind_eq_some_iff.1 hcell
  by_cases hlt : x < rr.per
  · rw [if_pos hlt] at hx
    refine ⟨z.2.2, ?_, rfl⟩
    show z.2.2 < h.reg.ns.getD z.1 0
    rw [← hper, ← Option.some.inj hx]
    exact hlt
  · rw [if_neg hlt] at hx
    cases hx

/-- **Every gate that reads, through the wiring, a written bit of a unit of the run's replay block is a gate of
that replay block the wiring leaves alone**, and so is the written bit's gate: at an accepted run, where the
members' proof blocks stand side by side, each bit is a gate of its unit's proof block and a gate is at one written
position at most. -/
theorem readers_inside (hcl : SideBySide h) (hU : BitsInUnits h) (hO : WrittenOnce h) (hr : r.Ok h) {y tw c : ℕ}
    {rq : Spot.Bits.RowPos} (hy : (y, tw, rq) ∈ writtenBits h.ms) (hc : c < h.reg.ns.getD tw 0)
    (hleaf : rq.leaf = r.ρ * h.reg.ns.getD tw 0 + c) :
    y ∈ cutBlock (wireMap h) r.rb ∧
      ∀ g' (x : BGate) (j : ℕ), (bitsProgs h.ms).gates[g']? = some x → j ∈ x.reads → wireMap h j = y →
        g' ∈ cutBlock (wireMap h) r.rb := by
  have hr' := hr
  obtain ⟨Pt, d, spec, hx, hP, h2, hin, hok, hu, hck, hb0, hρ, hrb, hset⟩ := hr'
  have htop := topological_of_two h2
  have hPt := partition_eq h hP
  have hprog := parse_prog h hx
  have hpart : (membersPart h.ms).Partitions (bitsProgs h.ms) := by
    rw [← hPt, ← hprog]
    exact hin.2.1
  obtain ⟨htw, -, bw, hbw, hyb⟩ := hU y tw rq (Or.inr hy)
  rw [hleaf] at hbw
  have hyfix : wireMap h y = y := wireMap_of_not_source h (source_none_of_written htop hy)
  refine ⟨mem_cutBlock.2 ⟨mem_rb hr htw hc hbw hyb, hyfix⟩, fun g' x j hg' hj hjy => ?_⟩
  -- a gate that reads is no input, so the wiring leaves it alone
  have hg'fix : wireMap h g' = g' := by
    cases hv : wireTarget h g' with
    | none => exact wireMap_of_no_target h hv
    | some z =>
      exfalso
      obtain ⟨t, rs, tw0, wport0, u0, hs, -, -, -, -⟩ := wireTarget_spec h hv
      obtain ⟨hst, g, hg, hκ, hxe⟩ := mem_sourceBits hs
      have hinp := (_root_.Flock.Bridge.Members.bitsProgs_input h.ms htop hst hg hκ).1
      rw [← hxe, hg'] at hinp
      have hxi : x = .input := Option.some.inj hinp
      rw [hxi] at hj
      simp [BGate.reads] at hj
  refine mem_cutBlock.2 ⟨?_, hg'fix⟩
  -- it is a gate of a proof block
  have hg'len : g' < (bitsProgs h.ms).gates.length := (List.getElem?_eq_some_iff.1 hg').1
  have hg'mem : g' ∈ (membersPart h.ms).blocks.flatten := by
    refine hpart.2.2.1 g' hg'len (fun b hb => ?_) (fun hun => ?_)
    · rw [hg'] at hb
      have hxc : x = .const b := Option.some.inj hb
      rw [hxc] at hj
      simp [BGate.reads] at hj
    · have hxi : x = .input := Option.some.inj (hg'.symm.trans hun.1)
      rw [hxi] at hj
      simp [BGate.reads] at hj
  obtain ⟨b', hb', hg'b⟩ := List.mem_flatten.1 hg'mem
  obtain ⟨v', hv'lt, hv'eq⟩ := List.getElem_of_mem hb'
  have hb'v : (membersPart h.ms).blocks[v']? = some b' := by
    rw [List.getElem?_eq_getElem hv'lt, hv'eq]
  -- and of the proof block of the gate it reads
  have hjb : ∀ {bj : List ℕ} {vj : ℕ}, (membersPart h.ms).blocks[vj]? = some bj → j ∈ bj → g' ∈ bj := by
    intro bj vj hbj hjbj
    have hjin : j ∈ b' := by
      rcases hcl.1 b' hb' g' hg'b x hg' j hj with hin' | ⟨cst, hcst⟩
      · exact hin'
      · exact absurd hcst (hcl.2 bj (List.mem_of_getElem? hbj) j hjbj cst)
    by_cases hvv : v' = vj
    · rw [hvv, hbj] at hb'v
      rw [Option.some.inj hb'v]
      exact hg'b
    · exact (Partition.disjoint_blocks hpart.1 hvv hb'v hbj hjin hjbj).elim
  cases hv : wireTarget h j with
  | none =>
    have hje : j = y := by
      rw [wireMap_of_no_target h hv] at hjy
      exact hjy
    rw [hje] at hjb
    exact mem_rb hr htw hc hbw (hjb hbw hyb)
  | some z =>
    have hzy : z = y := by
      rw [wireMap_of_target h hv] at hjy
      exact hjy
    rw [hzy] at hv
    obtain ⟨t, rs, tw', wport, u', hs, hwu, -, hym, -⟩ := wireTarget_spec h hv
    have heq := hO _ hym _ hy rfl
    have htw'eq : tw' = tw := congrArg (fun e : ℕ × ℕ × Spot.Bits.RowPos => e.2.1) heq
    have hu'eq : u' = rq.leaf := congrArg (fun e : ℕ × ℕ × Spot.Bits.RowPos => e.2.2.leaf) heq
    obtain ⟨hts, hszs, bj, hbj, hjbj⟩ := hU j t rs (Or.inl hs)
    obtain ⟨ρ', p', hp', hsplit⟩ := unit_split hP h2 hin hts hszs
    rw [hsplit] at hwu hbj
    obtain ⟨c', hc', hu'⟩ : ∃ c', c' < h.reg.ns.getD tw' 0 ∧ u' = ρ' * h.reg.ns.getD tw' 0 + c' :=
      wireUnit_slot hr hp' hwu
    rw [htw'eq] at hc' hu'
    have hdiv : ∀ {a b N : ℕ}, b < N → (a * N + b) / N = a := by
      intro a b N hb
      have hpos : 0 < N := Nat.lt_of_le_of_lt (Nat.zero_le b) hb
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ hpos, Nat.div_eq_of_lt hb, Nat.zero_add]
    have hρ' : ρ' = r.ρ := by
      have e1 := hdiv (a := ρ') hc'
      rw [← hu', hu'eq, hleaf, hdiv hc] at e1
      exact e1.symm
    rw [hρ'] at hbj
    exact mem_rb hr hts hp' hbj (hjb hbj hjbj)

end CheckedRun

/-- **A needed gate the wiring sends inside its replay block stays inside** (`Inside`), where every gate a checked
unit needs is one of its unit's bits (`NeededBits`), the members' proof blocks stand side by side (`SideBySide`),
each bit is a gate of its unit's proof block (`BitsInUnits`) and a gate is at one written position at most
(`WrittenOnce`). -/
theorem inside_of (hN : NeededBits h) (hcl : SideBySide h) (hU : BitsInUnits h) (hO : WrittenOnce h) :
    Inside h := by
  intro r hr i hi hret hnin
  obtain ⟨rp, hb⟩ := hN r hr i hi
  have hr' := hr
  obtain ⟨Pt, d, spec, hx, hP, h2, hin, hok, hu, hck, hb0, hρ, hrb, hset⟩ := hr'
  have hprog := parse_prog h hx
  have hul : r.u < Pt.blocks.length := (List.getElem?_eq_some_iff.1 hb0).1
  obtain ⟨T, hT, p, hp, hk, hue⟩ := member_of_replayOf h hP h2 hin hul hρ
  have hTe : templateOf h.ms r.u = T := by
    rw [hue]
    exact templateOf_unitBase hT hk
  have hke : r.u - unitBase h.ms T = r.ρ * h.reg.ns.getD T 0 + p := by
    rw [hue]
    omega
  rw [hTe, hke] at hb
  suffices H : ∃ y tw c, ∃ rq : Spot.Bits.RowPos, (y, tw, rq) ∈ writtenBits h.ms ∧ c < h.reg.ns.getD tw 0 ∧
      rq.leaf = r.ρ * h.reg.ns.getD tw 0 + c ∧ wireMap h i = y by
    obtain ⟨y, tw, c, rq, hy, hc, hleaf, hmap⟩ := H
    obtain ⟨hycut, hrd⟩ := CheckedRun.readers_inside hcl hU hO hr hy hc hleaf
    rw [hmap] at hret hnin ⊢
    refine ⟨hycut, ?_⟩
    rw [hprog]
    exact Prog.boundary_wired_eq_false hret hnin hrd
  rcases hb.2 with ⟨hs, hinp⟩ | ⟨hs, hw, -, -⟩
  · cases hv : wireTarget h i with
    | none =>
      exfalso
      rw [wireMap_of_no_target h hv] at hnin
      exact hnin hinp
    | some y =>
      obtain ⟨tw, wport, u', hwu, -, hym, -⟩ := wireTarget_at hs hv
      rw [hb.1] at hwu
      obtain ⟨c, hc, hu'⟩ : ∃ c, c < h.reg.ns.getD tw 0 ∧ u' = r.ρ * h.reg.ns.getD tw 0 + c :=
        CheckedRun.wireUnit_slot hr hp hwu
      exact ⟨y, tw, c, ⟨wport, u', rp.bit⟩, hym, hc, hu', wireMap_of_target h hv⟩
  · have hwm : (i, T, rp) ∈ writtenBits h.ms := List.mem_of_find?_eq_some hw
    exact ⟨i, T, p, rp, hwm, hp, hb.1, wireMap_of_not_source h hs⟩

end SpotFlock
