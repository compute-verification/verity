module

public import Proofs.Flock.SpotFlock.WiringOk
public import Proofs.Flock.SpotFlock.ReadWays

/-! Two of the five address facts, `EntryFact` and `OneTreeFact`, at the address map, from what they ask of the held
wiring (`Wiring`). -/

@[expose] public section

namespace Claims.Sampling

theorem Prog.wired_input {C : Prog} {ω : ℕ → ℕ} {keep : ℕ → Bool} {g : ℕ}
    (hg : C.gates[g]? = some .input) :
    (C.wired ω keep).gates[g]? = some .input := by
  show (C.gates.map (BGate.rewire ω))[g]? = some .input
  rw [List.getElem?_map]
  exact congrArg (Option.map (BGate.rewire ω)) hg

theorem Prog.wired_out {C : Prog} {ω : ℕ → ℕ} {keep : ℕ → Bool} {g : ℕ} (ho : g ∈ C.outs)
    (hk : keep g = true) :
    g ∈ (C.wired ω keep).outs :=
  List.mem_filter.2 ⟨ho, hk⟩

end Claims.Sampling

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy
open Properties.Spot (bitsProgs membersPart memberProg)
open Lean (Json)

structure Wiring (h : Held) : Prop where
  needed : NeededBits h
  inside : Inside h
  once : WrittenOnce h
  rows : RowsArePorts h
  ports : PortsMember h
  own : OwnReads h
  named : sourcesNamed h = true
  move : wiresMove h = true
  first : WiresFirst h
  alike : wiresAlike h = true
  whole : returnedWhole h = true
  unwired : writtenUnwired h = true

variable {h : Held}

theorem slotAt_linked {t : ℕ} {rp : Spot.Bits.RowPos} {l : Spot.SecondStage.Link}
    (hl : heldLink h (h.names.getD t "") rp.row = some l) :
    (slotAt h t rp).key = .port l.value ∧
      (slotAt h t rp).schema = _root_.Flock.PortRoots.schemaOf (memberCircuit h t) (portNamed h t rp.row) := by
  unfold slotAt
  rw [hl]
  exact ⟨rfl, rfl⟩

theorem slotAt_unlinked {t : ℕ} {rp : Spot.Bits.RowPos} (hl : heldLink h (h.names.getD t "") rp.row = none) :
    (slotAt h t rp).key = .interior (h.names.getD t "") rp.row ∧
      (slotAt h t rp).schema = _root_.Flock.PortRoots.schemaOf (memberCircuit h t) (portNamed h t rp.row) := by
  unfold slotAt
  rw [hl]
  exact ⟨rfl, rfl⟩

theorem wireTarget_of_linked {x t : ℕ} {rp : Spot.Bits.RowPos}
    (hf : (sourceBits h.ms).find? (·.1 == x) = some (x, t, rp)) {l : Spot.SecondStage.Link}
    (hl : heldLink h (h.names.getD t "") rp.row = some l) : wireTarget h x = none := by
  cases hxy : wireTarget h x with
  | none => rfl
  | some y =>
    exfalso
    obtain ⟨tw, wport, u, hwu, -⟩ := wireTarget_at hf hxy
    obtain ⟨w, hln, -⟩ := wireUnit_wire hwu
    rw [hl] at hln
    cases hln

theorem inputs_of_stmtOk {x m₁ m₂ : Msg} {s} {c₀ : FlockDeploy.Coin} {c₁ : Stage h} {u : ℕ}
    (hx : parse h x = some s) (hok : stmtOk h x m₁ m₂ c₀ c₁ = true)
    (hu : u ∈ h.reg.checked (units (base h) x) (src c₀) (stageSrc c₁)) :
    ∃ st rec draw, decode x = some st ∧ jsonMsg m₁ = some rec ∧ secondDraw h c₀ c₁ = some draw ∧
      Spot.SecondStage.inputsFailure (ownOf h st rec draw c₀) = none := by
  have hat := stmtOkAt_unit h hx hok hu
  unfold stmtOkAt at hat
  rw [hx] at hat
  simp only [Bool.and_eq_true] at hat
  exact inputs_of_readsOk hat.2

namespace CheckedRun

variable {r : CheckedRun h.stage}

structure Reads (r : CheckedRun h.stage) (st : Roots) (rec draw : Json) : Prop where
  dec : decode r.x = some st
  roots : r.s.roots = st.roots
  msg : jsonMsg r.m₁ = some rec
  sample : secondDraw h r.c₀ r.c₁ = some draw
  inputs : Spot.SecondStage.inputsFailure (ownOf h st rec draw r.c₀) = none
  prog : r.s.prog = bitsProgs h.ms
  lt : templateOf h.ms r.u < h.ms.length
  design : ∃ d, two h r.s = some d
  reads : ∀ own reg, r.ins.registered = some own → reg ∈ own →
    ∃ e, Spot.getD e "value" = .str reg.value ∧
      Spot.SecondStage.readOk (ownOf h st rec draw r.c₀) (Spot.SecondStage.byTemplate rec)
        (partAt h (ownOf h st rec draw r.c₀).sample.size (templateOf h.ms r.u)) reg.name e = true
  schema : ∀ own reg, r.ins.registered = some own → reg ∈ own → ∃ q,
    (List.range r.sts[0]!.c.ports.size).find? (fun q => r.sts[0]!.c.ports[q]!.name == reg.name) = some q ∧
      _root_.Flock.Registered.portSchema r.sts[0]!.c.ports[q]! = reg.schema

theorem reads_of_ok (hr : r.Ok h) : ∃ st rec draw, r.Reads st rec draw := by
  have hpos := sts_pos hr
  obtain ⟨Pt, d, spec, hx, hP, h2, hin, hok, hu, hck, hb, hρ, hrb, hset⟩ := hr
  obtain ⟨st, rec, draw, hst, hrec, hdraw, hinp⟩ := inputs_of_stmtOk hx hok hu
  obtain ⟨st', hst', hroots⟩ := parse_roots h hx
  have e0 : st' = st := Option.some.inj (hst'.symm.trans hst)
  rw [e0] at hroots
  have hmem : r.sts[0]! ∈ r.sts := by
    rw [getElem!_pos r.sts 0 hpos]
    exact Array.getElem_mem hpos
  refine ⟨st, rec, draw, hst, hroots, hrec, hdraw, hinp, parse_prog h hx, templateOf_lt_of_checked h hx hu,
    ⟨d, h2⟩, fun own reg hown hreg => ?_, fun own reg hown hreg => schema_of_setup hset hown _ hmem reg hreg⟩
  obtain ⟨st2, rec2, draw2, e, hst2, hrec2, hdraw2, hv, hrd, -⟩ := read_of_reg h hx hok hu hset hown hreg
  have e1 : st2 = st := Option.some.inj (hst2.symm.trans hst)
  have e2 : rec2 = rec := Option.some.inj (hrec2.symm.trans hrec)
  have e3 : draw2 = draw := Option.some.inj (hdraw2.symm.trans hdraw)
  rw [e1, e2, e3] at hrd
  exact ⟨e, hv, hrd⟩

theorem port_at (hR : RowsArePorts h) (hP : PortsMember h) (hr : r.Ok h) {k i : ℕ} {rp : Spot.Bits.RowPos}
    (hb : BitOf h (templateOf h.ms r.u) k i rp) :
    (List.range r.sts[0]!.c.ports.size).find? (fun q => r.sts[0]!.c.ports[q]!.name == rp.row) =
        some (portNamed h (templateOf h.ms r.u) rp.row) ∧
      portNamed h (templateOf h.ms r.u) rp.row < r.sts[0]!.c.ports.size ∧
      (r.sts[0]!.c.ports[portNamed h (templateOf h.ms r.u) rp.row]!).name = rp.row := by
  have hq : ∃ q, q < (memberCircuit h (templateOf h.ms r.u)).ports.size ∧
      (memberCircuit h (templateOf h.ms r.u)).ports[q]!.name = rp.row := by
    rcases hb.2 with ⟨hs, -⟩ | ⟨-, hw, -⟩
    · exact hR.1 (i, templateOf h.ms r.u, rp) (List.mem_of_find?_eq_some hs)
    · exact hR.2 (i, templateOf h.ms r.u, rp) (List.mem_of_find?_eq_some hw)
  rw [hP r hr]
  exact portNamed_spec hq

theorem holds_linked (W : Wiring h) (hr : r.Ok h) {st : Roots} {rec draw : Json} (hR : r.Reads st rec draw)
    {k i : ℕ} {rp : Spot.Bits.RowPos} (hb : BitOf h (templateOf h.ms r.u) k i rp)
    (hω : ∀ j, wireMap h (wireMap h j) = wireMap h j) (hfix : wireMap h i = i) {l : Spot.SecondStage.Link}
    (hl : heldLink h (h.names.getD (templateOf h.ms r.u) "") rp.row = some l)
    (hB : (r.s.prog.wired (wireMap h) (returned h)).Boundary (cutBlock (wireMap h) r.rb) i = true) :
    r.EntryHolds h (wireMap h) (returned h) (addressOf h i) (wireMap h i) ∧
      r.RootsHold h (wireMap h) (returned h) (addressOf h i) (wireMap h i) := by
  have haddr : addressOf h i = ⟨fun _ => portNamed h (templateOf h.ms r.u) rp.row,
      placeAt h (templateOf h.ms r.u) rp, slotAt h (templateOf h.ms r.u) rp⟩ :=
    (addressOf_eq h hω i).trans (addressAt_of_fixed hb hfix)
  obtain ⟨hfind, hlt, hname⟩ := port_at W.rows W.ports hr hb
  have hroots : r.rootsWired (wireMap h) (returned h) (wireMap h i) = r.s.roots := by
    rw [hfix]
    exact rootsWired_boundary hB
  obtain ⟨own0, reg0, hown0, hmem0, hn0⟩ := reg_of_named hr (Or.inr (link_named h hl))
  have hsome : regNamed r.ins r.sts (portNamed h (templateOf h.ms r.u) rp.row) ≠ none :=
    regNamed_ne_none hname hown0 hmem0 hn0
  obtain ⟨hkey, hsl⟩ := slotAt_linked hl
  have key : ∀ reg, regNamed r.ins r.sts (portNamed h (templateOf h.ms r.u) rp.row) = some reg →
      reg.name = rp.row ∧
        readRoots h r.x r.m₁ r.c₀ r.c₁ r.ρ (templateOf h.ms r.u) reg.name reg.value = r.s.roots ∧
        readKeyAt h r.x r.m₁ r.c₀ r.c₁ (templateOf h.ms r.u) reg.name reg.value = .port l.value ∧
        _root_.Flock.PortRoots.schemaOf (memberCircuit h (templateOf h.ms r.u))
          (portNamed h (templateOf h.ms r.u) rp.row) = reg.schema := by
    intro reg hreg
    obtain ⟨own, hown, hmem, hn⟩ := regNamed_spec hname hreg
    have hl' : heldLink h (h.names.getD (templateOf h.ms r.u) "") reg.name = some l := by
      rw [hn]
      exact hl
    obtain ⟨hri, hk⟩ := readsInterior_linked h hR.inputs hl' reg.value
    have hrr : readRoots h r.x r.m₁ r.c₀ r.c₁ r.ρ (templateOf h.ms r.u) reg.name reg.value = r.s.roots := by
      rw [hR.roots]
      exact readRoots_of_not_readsInterior h hR.dec hR.msg hR.sample r.ρ _ reg.name reg.value hri
    have hrk : readKeyAt h r.x r.m₁ r.c₀ r.c₁ (templateOf h.ms r.u) reg.name reg.value = .port l.value := by
      rw [readKeyAt_eq h hR.dec hR.msg hR.sample]
      exact hk
    obtain ⟨q, hq, hsch⟩ := hR.schema own reg hown hmem
    rw [hn] at hq
    have hqp : q = portNamed h (templateOf h.ms r.u) rp.row := Option.some.inj (hq.symm.trans hfind)
    rw [hqp] at hsch
    refine ⟨hn, hrr, hrk, ?_⟩
    rw [← schemaOf_ports (W.ports r hr), schemaOf_lt hlt]
    exact hsch
  unfold EntryHolds RootsHold
  simp only [haddr]
  refine ⟨⟨fun reg hreg => ?_, fun hnone => absurd hnone hsome⟩,
    ⟨fun reg hreg => ?_, fun hnone => absurd hnone hsome⟩⟩
  · obtain ⟨hn, hrr, hrk, hsch⟩ := key reg hreg
    refine ⟨?_, ?_, ?_⟩
    · rw [hn]
      exact hfind
    · rw [hroots, hrk, entryAt, hkey]
    · rw [hsl]
      exact hsch
  · obtain ⟨hn, hrr, hrk, hsch⟩ := key reg hreg
    rw [hroots, hrr]

theorem holds_inside (W : Wiring h) (hr : r.Ok h) {st : Roots} {rec draw : Json} (hR : r.Reads st rec draw)
    {k i tw : ℕ} {rp rq : Spot.Bits.RowPos} (hb : BitOf h (templateOf h.ms r.u) k i rp)
    (hω : ∀ j, wireMap h (wireMap h j) = wireMap h j) (hw : bitAtGate h (wireMap h i) = some (tw, rq))
    (hlw : heldLink h (h.names.getD tw "") rq.row = none)
    (hin : wireMap h i ∈ cutBlock (wireMap h) r.rb ∧
      (r.s.prog.wired (wireMap h) (returned h)).Boundary (cutBlock (wireMap h) r.rb) (wireMap h i) = false)
    (hschema : _root_.Flock.PortRoots.schemaOf (memberCircuit h tw) (portNamed h tw rq.row) =
      _root_.Flock.PortRoots.schemaOf (memberCircuit h (templateOf h.ms r.u))
        (portNamed h (templateOf h.ms r.u) rp.row))
    (hread : ∀ own reg, r.ins.registered = some own → reg ∈ own → reg.name = rp.row →
      heldLink h (h.names.getD (templateOf h.ms r.u) "") rp.row = none ∧
        ((Spot.Record.byPort rec).find? (·.1 == reg.value)).isSome = true ∧
        reg.value = Spot.SecondStage.portOf (Spot.SecondStage.byTemplate rec) (h.names.getD tw "") rq.row)
    (hnone : regNamed r.ins r.sts (portNamed h (templateOf h.ms r.u) rp.row) = none →
      h.names.getD tw "" = h.names.getD (templateOf h.ms r.u) "" ∧ rq.row = rp.row) :
    r.EntryHolds h (wireMap h) (returned h) (addressOf h i) (wireMap h i) ∧
      r.RootsHold h (wireMap h) (returned h) (addressOf h i) (wireMap h i) := by
  have haddr : addressOf h i = ⟨fun _ => portNamed h (templateOf h.ms r.u) rp.row, placeAt h tw rq,
      slotAt h tw rq⟩ :=
    (addressOf_eq h hω i).trans (addressAt_of_bitOf hb hw)
  obtain ⟨hfind, hlt, hname⟩ := port_at W.rows W.ports hr hb
  have hroots : r.rootsWired (wireMap h) (returned h) (wireMap h i) = irootsOf r.m₁ r.ρ := rootsWired_inside hin
  have hir : iroots r.x r.seen r.ρ = irootsOf r.m₁ r.ρ := rfl
  obtain ⟨hkey, hsl⟩ := slotAt_unlinked hlw
  have hentry : entryAt (irootsOf r.m₁ r.ρ) (slotAt h tw rq) =
      keyAt (irootsOf r.m₁ r.ρ)
        (.port (Spot.SecondStage.portOf (Spot.SecondStage.byTemplate rec) (h.names.getD tw "") rq.row)) := by
    rw [entryAt_interior_port _ _ _ _ hkey, namedOf_irootsOf hR.msg]
  have key : ∀ reg, regNamed r.ins r.sts (portNamed h (templateOf h.ms r.u) rp.row) = some reg →
      reg.name = rp.row ∧
        readRoots h r.x r.m₁ r.c₀ r.c₁ r.ρ (templateOf h.ms r.u) reg.name reg.value = irootsOf r.m₁ r.ρ ∧
        readKeyAt h r.x r.m₁ r.c₀ r.c₁ (templateOf h.ms r.u) reg.name reg.value =
          .port (Spot.SecondStage.portOf (Spot.SecondStage.byTemplate rec) (h.names.getD tw "") rq.row) ∧
        _root_.Flock.PortRoots.schemaOf (memberCircuit h (templateOf h.ms r.u))
          (portNamed h (templateOf h.ms r.u) rp.row) = reg.schema := by
    intro reg hreg
    obtain ⟨own, hown, hmem, hn⟩ := regNamed_spec hname hreg
    obtain ⟨hl, hport, hval⟩ := hread own reg hown hmem hn
    have hl' : heldLink h (h.names.getD (templateOf h.ms r.u) "") reg.name = none := by
      rw [hn]
      exact hl
    obtain ⟨hri, hk⟩ := readsInterior_port h st draw r.c₀ hl' hport
    have hrk : readKeyAt h r.x r.m₁ r.c₀ r.c₁ (templateOf h.ms r.u) reg.name reg.value =
        .port (Spot.SecondStage.portOf (Spot.SecondStage.byTemplate rec) (h.names.getD tw "") rq.row) := by
      rw [readKeyAt_eq h hR.dec hR.msg hR.sample, hk, hval]
    obtain ⟨q, hq, hsch⟩ := hR.schema own reg hown hmem
    rw [hn] at hq
    have hqp : q = portNamed h (templateOf h.ms r.u) rp.row := Option.some.inj (hq.symm.trans hfind)
    rw [hqp] at hsch
    refine ⟨hn, readRoots_of_readsInterior h hR.dec hR.msg hR.sample r.ρ _ reg.name reg.value hri, hrk, ?_⟩
    rw [← schemaOf_ports (W.ports r hr), schemaOf_lt hlt]
    exact hsch
  unfold EntryHolds RootsHold
  simp only [haddr]
  refine ⟨⟨fun reg hreg => ?_, fun hnone' => ?_⟩, ⟨fun reg hreg => ?_, fun _ => hin⟩⟩
  · obtain ⟨hn, hrr, hrk, hsch⟩ := key reg hreg
    refine ⟨?_, ?_, ?_⟩
    · rw [hn]
      exact hfind
    · rw [hroots, hrk, hentry]
    · rw [hsl, hschema]
      exact hsch
  · obtain ⟨hnm, hrow⟩ := hnone hnone'
    refine ⟨hlt, ?_, fun pop _ => ?_⟩
    · have e1 : entryAt (iroots r.x r.seen r.ρ) (slotAt h tw rq) =
          keyAt (iroots r.x r.seen r.ρ) (.interior (h.names.getD tw "") rq.row) := by
        rw [entryAt, hkey]
      rw [e1, hname, hnm, hrow]
    · rw [hsl, hschema]
      exact (schemaOf_ports (W.ports r hr) _).symm
  · obtain ⟨hn, hrr, hrk, hsch⟩ := key reg hreg
    rw [hroots, hrr]

theorem holds_of_wiring (W : Wiring h) (hr : r.Ok h) {i : ℕ} (hi : r.s.prog.Needs r.b i) :
    r.EntryHolds h (wireMap h) (returned h) (addressOf h i) (wireMap h i) ∧
      r.RootsHold h (wireMap h) (returned h) (addressOf h i) (wireMap h i) := by
  obtain ⟨st, rec, draw, hR⟩ := reads_of_ok hr
  obtain ⟨d, h2⟩ := hR.design
  have hω : ∀ j, wireMap h (wireMap h j) = wireMap h j := (rewires_of_two h2 W.first).idem
  have htop := topological_of_two h2
  obtain ⟨rp, hb⟩ := W.needed r hr i hi
  obtain ⟨-, -, hname⟩ := port_at W.rows W.ports hr hb
  rcases hb.2 with ⟨hs, hinp⟩ | ⟨hs, hw, hout, hnin⟩
  ·
    have hsm : (i, templateOf h.ms r.u, rp) ∈ sourceBits h.ms := List.mem_of_find?_eq_some hs
    cases hl : heldLink h (h.names.getD (templateOf h.ms r.u) "") rp.row with
    | some l =>
      have hfix : wireMap h i = i := wireMap_of_no_target h (wireTarget_of_linked hs hl)
      have hgate : (bitsProgs h.ms).gates[i]? = some .input := hinp
      rw [← hR.prog] at hgate
      exact holds_linked W hr hR hb hω hfix hl (Prog.boundary_of_input _ (Prog.wired_input hgate))
    | none =>
      obtain ⟨y, hxy⟩ := wireTarget_of_sourcesNamed h W.named W.move hsm hl
      obtain ⟨tw, wport, u', hwu, hlw, hym, hret⟩ := wireTarget_at hs hxy
      obtain ⟨w, -, hwire, htw, hwp⟩ := wireUnit_wire hwu
      have htw' : heldWriter h w.writer = some tw := htw
      have hwp' : wport = w.wport := hwp
      rw [hwp'] at hlw hym
      have hmap : wireMap h i = y := wireMap_of_target h hxy
      have hnm : h.names.getD tw "" = w.writer := by
        have h0 := List.find?_some htw'
        have h1 : (h.names.getD tw "" == w.writer) = true := h0
        exact eq_of_beq h1
      have hbit : bitAtGate h (wireMap h i) = some (tw, Spot.Bits.RowPos.mk w.wport u' rp.bit) := by
        rw [hmap]
        simp only [bitAtGate, source_none_of_written htop hym, written_find W.once hym, Option.map_some]
      have hin := W.inside r hr i hi (by rw [hmap]; exact hret)
        (by rw [hmap]; exact written_not_input htop hym)
      have hschema : _root_.Flock.PortRoots.schemaOf (memberCircuit h tw) (portNamed h tw w.wport) =
          _root_.Flock.PortRoots.schemaOf (memberCircuit h (templateOf h.ms r.u))
            (portNamed h (templateOf h.ms r.u) rp.row) :=
        (wiresAlike_spec W.alike hR.lt hwire htw').2.symm
      have hread : ∀ own reg, r.ins.registered = some own → reg ∈ own → reg.name = rp.row →
          heldLink h (h.names.getD (templateOf h.ms r.u) "") rp.row = none ∧
            ((Spot.Record.byPort rec).find? (·.1 == reg.value)).isSome = true ∧
            reg.value =
              Spot.SecondStage.portOf (Spot.SecondStage.byTemplate rec) (h.names.getD tw "") w.wport := by
        intro own reg hown hmem hn
        obtain ⟨e, hv, hok⟩ := hR.reads own reg hown hmem
        rw [hn] at hok
        obtain ⟨xe, pw, rr, hv', hmem', -⟩ :=
          Spot.SecondStage.readOk_wired (own := ownOf h st rec draw r.c₀)
            (named := Spot.SecondStage.byTemplate rec)
            (p := partAt h (ownOf h st rec draw r.c₀).sample.size (templateOf h.ms r.u)) (port := rp.row)
            (e := e) (w := w) hl hwire hok
        have hval : reg.value =
            Spot.SecondStage.portOf (Spot.SecondStage.byTemplate rec) w.writer w.wport :=
          Json.str.inj (hv.symm.trans hv')
        refine ⟨hl, ?_, ?_⟩
        · rw [hval]
          exact port_found hmem'
        · rw [hnm]
          exact hval
      obtain ⟨own0, reg0, hown0, hmem0, hn0⟩ := reg_of_named hr (Or.inl (wire_named h hwire))
      exact holds_inside W hr hR hb hω hbit hlw hin hschema hread
        (fun hn' => absurd hn' (regNamed_ne_none hname hown0 hmem0 hn0))
  ·
    have hwm : (i, templateOf h.ms r.u, rp) ∈ writtenBits h.ms := List.mem_of_find?_eq_some hw
    have hfix : wireMap h i = i := wireMap_of_not_source h hs
    cases hl : heldLink h (h.names.getD (templateOf h.ms r.u) "") rp.row with
    | some l =>
      have hret : returned h i = true := by
        unfold returned
        refine List.any_eq_true.2 ⟨(i, templateOf h.ms r.u, rp), hwm, ?_⟩
        show (i == i && (heldLink h (h.names.getD (templateOf h.ms r.u) "") rp.row).isSome) = true
        rw [hl]
        simp
      have hout' : i ∈ (bitsProgs h.ms).outs := hout
      rw [← hR.prog] at hout'
      exact holds_linked W hr hR hb hω hfix hl (Prog.boundary_of_out _ (Prog.wired_out hout' hret))
    | none =>
      have hret : returned h i = false := returned_of_whole W.whole hwm hl
      have huw : heldWire h (h.names.getD (templateOf h.ms r.u) "") rp.row = none :=
        unwired_of_written W.unwired hwm hl
      have hbit : bitAtGate h (wireMap h i) = some (templateOf h.ms r.u, rp) := by
        rw [hfix]
        exact bitAtGate_of_bitOf hb
      have hin := W.inside r hr i hi (by rw [hfix]; exact hret) (by rw [hfix]; exact hnin)
      have hread : ∀ own reg, r.ins.registered = some own → reg ∈ own → reg.name = rp.row →
          heldLink h (h.names.getD (templateOf h.ms r.u) "") rp.row = none ∧
            ((Spot.Record.byPort rec).find? (·.1 == reg.value)).isSome = true ∧
            reg.value = Spot.SecondStage.portOf (Spot.SecondStage.byTemplate rec)
              (h.names.getD (templateOf h.ms r.u) "") rp.row := by
        intro own reg hown hmem hn
        have hl' : heldLink h (h.names.getD (templateOf h.ms r.u) "") reg.name = none := by
          rw [hn]
          exact hl
        have huw' : heldWire h (h.names.getD (templateOf h.ms r.u) "") reg.name = none := by
          rw [hn]
          exact huw
        have hport := W.own r hr own reg hown hmem hl' huw' rec hR.msg
        obtain ⟨e, hv, hok⟩ := hR.reads own reg hown hmem
        have hval : reg.value = Spot.SecondStage.portOf (Spot.SecondStage.byTemplate rec)
            (h.names.getD (templateOf h.ms r.u) "") reg.name :=
          readOk_own (own := ownOf h st rec draw r.c₀) (named := Spot.SecondStage.byTemplate rec)
            (p := partAt h (ownOf h st rec draw r.c₀).sample.size (templateOf h.ms r.u)) (port := reg.name)
            (e := e) (v := reg.value) hl' huw' hv hport hok
        rw [hn] at hval
        exact ⟨hl, hport, hval⟩
      exact holds_inside W hr hR hb hω hbit hl hin rfl hread (fun _ => ⟨rfl, rfl⟩)

end CheckedRun

theorem entryFact_of_wiring (W : Wiring h) : EntryFact h (wireMap h) (returned h) (addressOf h) :=
  fun _ hr _ hi => (CheckedRun.holds_of_wiring W hr hi).1

theorem oneTreeFact_of_wiring (W : Wiring h) : OneTreeFact h (wireMap h) (returned h) (addressOf h) :=
  ⟨addressOf_shares h, fun _ hr _ hi => (CheckedRun.holds_of_wiring W hr hi).2⟩

end SpotFlock
