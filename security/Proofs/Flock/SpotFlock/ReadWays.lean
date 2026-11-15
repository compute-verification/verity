module

public import Proofs.Flock.SpotFlock.ReadsDesign
public import Proofs.Flock.SpotFlock.Reads
public import Proofs.Flock.SpotFlock.WiredCode
public import Proofs.Flock.SpotFlock.AddressFacts
public import Proofs.Flock.Verifier.Registered

/-!
# The ways round 2 reads a registered read

A registered read of an accepted run's checked unit is read one of three ways, by what the auditor's copies name its
port: at a port a link names, the boundary port the link names, under the statement's roots; at a port a wire names,
the writer's output port of the interior record, under the replay block's roots; at a port neither names, where the
interior record has a port of the read's value, the template's own port of the interior record, under the replay
block's roots. This file states each from round 2's check of the read (`SecondStage.readOk`), with what `readRoots`
and `readKeyAt` are there:

* `registeredEntry_none_of_inputs`: at what the model holds of the run (`ownOf`: no value pinned or supplied, no
  table read), round 2's `second-inputs` passes only where round 1's record registers no value;
* `readsInterior_linked`, `readsInterior_port`: which record round 2 reads the value off, and the entry;
* `readOk_own`: at a port no link or wire names, a read of a port of the interior record is of the template's own;
* `read_of_reg`: the entry round 2 passes for a registered read of a checked unit's session, with `second-inputs`;
* `schema_of_setup`: each registered read of a session that sets up has the schema of the circuit's first port of
  its name (`Registered.checkPort`);
* `regNamed_spec`, `regNamed_ne_none`, `reg_of_named`: the registered read named like a port, and that a port a link
  or a wire names has one.
-/

@[expose] public section

namespace Flock.Registered

/-- **A port `checkPort` accepts has the schema of the circuit's first port of its name.** -/
theorem checkPort_schema {c : Circuit} {pub : Public} {t : Tables} {indices : Array Nat} {reg : Port}
    (h : checkPort c pub t indices reg = .ok ()) :
    ∃ p, (List.range c.ports.size).find? (fun p => c.ports[p]!.name == reg.name) = some p ∧
      portSchema c.ports[p]! = reg.schema := by
  unfold checkPort at h
  simp only at h
  split at h
  · rename_i p hp
    split at h
    · exact absurd (except_bind_ok h) fun ⟨_, h1, _⟩ => throw_ne_ok _ _ h1
    · rename_i hsch
      exact ⟨p, hp, by simpa using hsch⟩
  · exact absurd h (throw_ne_ok _ _)

end Flock.Registered

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy
open Lean (Json)

/-- A non-empty array none of whose elements passes has not all of them passing. -/
theorem nonempty_all_false {α : Type} (xs : Array α) (p : α → Bool) (hp : ∀ x, p x = false) :
    (!xs.isEmpty && xs.all p) = false := by
  by_cases hsz : xs.size = 0
  · have he : xs.isEmpty = true := by
      exact Array.isEmpty_iff_size_eq_zero.mpr hsz
    rw [he]
    rfl
  · have hall : xs.all p = false := by
      apply Bool.eq_false_iff.2
      intro hc
      have hpos : 0 < xs.size := Nat.pos_of_ne_zero hsz
      have hx := Array.all_eq_true_iff_forall_mem.1 hc xs[0] (Array.getElem_mem hpos)
      rw [hp] at hx
      cases hx
    rw [hall, Bool.and_false]

theorem pass_of_false {a b c : Bool} (ha : a = false) (hb : b = false) : (!a && (!b || c)) = true := by
  subst ha
  subst hb
  rfl

/-- A port a circuit has, has the schema Flock frames its rows in. -/
theorem schemaOf_lt {c : _root_.Flock.Circuit} {q : ℕ} (hq : q < c.ports.size) :
    _root_.Flock.PortRoots.schemaOf c q = _root_.Flock.Registered.portSchema c.ports[q]! := by
  simp [_root_.Flock.PortRoots.schemaOf, hq]

/-- Circuits of the same ports frame each port's rows in the same schema. -/
theorem schemaOf_ports {c c' : _root_.Flock.Circuit} (hp : c.ports = c'.ports) (q : ℕ) :
    _root_.Flock.PortRoots.schemaOf c q = _root_.Flock.PortRoots.schemaOf c' q := by
  unfold _root_.Flock.PortRoots.schemaOf
  rw [hp]

variable (h : Held)

/-! ## What the model holds of the run -/

/-- **At what the model holds of the run, no link names an opened row**: the model records no table read. -/
theorem openedRow_ownOf (st : Roots) (rec draw : Json) (c₀ : FlockDeploy.Coin) (names : Array String)
    (mine : Array (String × Array (Array (String × String)))) (x : Spot.SecondStage.Link) :
    Spot.SecondStage.openedRow (ownOf h st rec draw c₀) names mine x = false := by
  have hsz : Spot.SecondStage.readSize (ownOf h st rec draw c₀) x.template = none := by
    have hreads : (ownOf h st rec draw c₀).reads = #[] := rfl
    unfold Spot.SecondStage.readSize
    rw [hreads]
    simp
  unfold Spot.SecondStage.openedRow
  simp only [hsz, Option.isSome_none, Bool.false_and]

/-- **At what the model holds of the run, `second-inputs` passes only where round 1's record registers no value**:
the model holds no value pinned or supplied and records no table read, so every value the interior registration
registers is one a replay block's input may not read (`SecondStage.inputsFailure`). -/
theorem registeredEntry_none_of_inputs {st : Roots} {rec draw : Json} {c₀ : FlockDeploy.Coin}
    (hin : Spot.SecondStage.inputsFailure (ownOf h st rec draw c₀) = none) (v : String) :
    Spot.SecondStage.registeredEntry rec v = none := by
  cases hre : Spot.SecondStage.registeredEntry rec v with
  | none => rfl
  | some e =>
    exfalso
    unfold Spot.SecondStage.registeredEntry at hre
    have he := Array.mem_of_find?_eq_some hre
    have hv := Array.find?_some hre
    cases hval : Spot.getD e "value" with
    | str w =>
      unfold Spot.SecondStage.inputsFailure at hin
      have hnone := Array.find?_eq_none.1 hin w (Array.mem_filterMap.2 ⟨e, he, by simp only [hval]⟩)
      apply hnone
      have h1 : ((ownOf h st rec draw c₀).fixed.contains w ||
          (Spot.SecondStage.ownValues (ownOf h st rec draw c₀)).any (·.1 == w)) = false := by
        have hf : (ownOf h st rec draw c₀).fixed = #[] := rfl
        have hs : Spot.SecondStage.ownValues (ownOf h st rec draw c₀) = #[] := by
          have hsup : (ownOf h st rec draw c₀).supplied = #[] := rfl
          unfold Spot.SecondStage.ownValues
          rw [hsup]
          simp
        rw [hf, hs]
        simp
      exact pass_of_false h1 (nonempty_all_false _ _ (openedRow_ownOf h st rec draw c₀ _ _))
    | _ => simp [Spot.jsonStrIs, hval] at hv

/-- **A linked read is read off the boundary record, at the port the link names**: at a port a link of the auditor's
copies names, in a run whose `second-inputs` passes, round 2 reads the value under the statement's roots, off the
boundary port the link names. -/
theorem readsInterior_linked {st : Roots} {rec draw : Json} {c₀ : FlockDeploy.Coin}
    (hin : Spot.SecondStage.inputsFailure (ownOf h st rec draw c₀) = none) {t port : String}
    {l : Spot.SecondStage.Link} (hl : heldLink h t port = some l) (v : String) :
    readsInterior (ownOf h st rec draw c₀) t port v = false ∧
      readKey (ownOf h st rec draw c₀) t port v = .port l.value := by
  have hlo : Spot.SecondStage.linkOf (ownOf h st rec draw c₀) t port = some l := hl
  have hre : Spot.SecondStage.registeredEntry (ownOf h st rec draw c₀).interior v = none :=
    registeredEntry_none_of_inputs h hin v
  unfold readsInterior readKey
  rw [hlo, hre]
  exact ⟨rfl, rfl⟩

/-- **A read of a port of the interior record at a port no link names is read off the interior record, at that
port.** -/
theorem readsInterior_port (st : Roots) {rec : Json} (draw : Json) (c₀ : FlockDeploy.Coin) {t port : String}
    (hl : heldLink h t port = none) {v : String}
    (hp : ((Spot.Record.byPort rec).find? (·.1 == v)).isSome = true) :
    readsInterior (ownOf h st rec draw c₀) t port v = true ∧
      readKey (ownOf h st rec draw c₀) t port v = .port v := by
  have hlo : Spot.SecondStage.linkOf (ownOf h st rec draw c₀) t port = none := hl
  have hp' : ((Spot.Record.byPort (ownOf h st rec draw c₀).interior).find? (·.1 == v)).isSome = true := hp
  unfold readsInterior readKey
  rw [hlo, hp']
  exact ⟨rfl, rfl⟩

variable {h} in
/-- **At a port no link or wire names, a read of a port of the interior record is of the template's own port**
(`SecondStage.readOk`, its last arm). -/
theorem readOk_own {own : Spot.SecondStage.Own} {named : Bool} {p : Spot.SecondStage.Part} {port : String}
    {e : Json} {v : String} (hl : Spot.SecondStage.linkOf own p.template port = none)
    (hw : Spot.SecondStage.wireOf own p.template port = none) (hv : Spot.getD e "value" = .str v)
    (hp : ((Spot.Record.byPort own.interior).find? (·.1 == v)).isSome = true)
    (hok : Spot.SecondStage.readOk own named p port e = true) :
    v = Spot.SecondStage.portOf named p.template port := by
  unfold Spot.SecondStage.readOk at hok
  rw [hv, hl, hw] at hok
  cases hf : (Spot.Record.byPort own.interior).find? (·.1 == v) with
  | none =>
    rw [hf] at hp
    cases hp
  | some kx =>
    obtain ⟨k, x⟩ := kx
    simp only [hf, Bool.and_eq_true, beq_iff_eq] at hok
    exact hok.1.1

/-- **A member of a record's ports is found by its name.** -/
theorem port_found {ps : Array (String × Json)} {v : String} {x : Json} (hm : (v, x) ∈ ps) :
    (ps.find? (·.1 == v)).isSome = true := by
  cases hf : ps.find? (·.1 == v) with
  | some y => rfl
  | none =>
    have hne := Array.find?_eq_none.1 hf (v, x) hm
    simp at hne

/-! ## An accepted run's registered reads -/

/-- **Round 2 passes each registered read of a checked unit's session**: in an accepted run, a registered read `reg`
of a checked unit's session has an entry of the session statement's registered reads, of `reg`'s value, that
`SecondStage.readOk` passes at `reg`'s port against the unit's template's part, at what the model holds of the run,
whose `second-inputs` passes. -/
theorem read_of_reg {x m₁ m₂ : Msg} {s} {c₀ : FlockDeploy.Coin} {c₁ : Stage h} {u : ℕ} {ins : FlockDeploy.Ins}
    {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hx : parse h x = some s) (hok : stmtOk h x m₁ m₂ c₀ c₁ = true)
    (hu : u ∈ h.reg.checked (units (base h) x) (src c₀) (stageSrc c₁))
    (hset : FlockDeploy.setupOf true (unitStmt h m₂ u) = some (ins, spec, sts))
    {own : Array _root_.Flock.Registered.Port} {reg : _root_.Flock.Registered.Port}
    (hown : ins.registered = some own) (hreg : reg ∈ own) :
    ∃ st rec draw e, decode x = some st ∧ jsonMsg m₁ = some rec ∧ secondDraw h c₀ c₁ = some draw ∧
      Spot.getD e "value" = .str reg.value ∧
      Spot.SecondStage.readOk (ownOf h st rec draw c₀) (Spot.SecondStage.byTemplate rec)
        (partAt h (ownOf h st rec draw c₀).sample.size (templateOf h.ms u)) reg.name e = true ∧
      Spot.SecondStage.inputsFailure (ownOf h st rec draw c₀) = none := by
  have hat := stmtOkAt_unit h hx hok hu
  unfold stmtOkAt at hat
  rw [hx] at hat
  simp only [Bool.and_eq_true] at hat
  obtain ⟨st, rec, draw, w, S, hst, hrec, hdraw, hw, hS, hc⟩ := readsOk_true h hat.2
  obtain ⟨st', rec', draw', hst', hrec', hdraw', hin⟩ := inputs_of_readsOk hat.2
  have e1 : st' = st := Option.some.inj (hst'.symm.trans hst)
  have e2 : rec' = rec := Option.some.inj (hrec'.symm.trans hrec)
  have e3 : draw' = draw := Option.some.inj (hdraw'.symm.trans hdraw)
  rw [e1, e2, e3] at hin
  obtain ⟨bR, jR, hbR, hjR, hoj⟩ := registered_of_ins (ins_of_setupOf hw hset) hown
  obtain ⟨-, -, hSr, -⟩ := stmtJsonOf_fields h hS
  have hSj : S.registered = jR := by
    have h4 := hSr bR hbR
    rw [hjR] at h4
    exact (Option.some.inj h4).symm
  obtain ⟨kvs, hports, hent, -⟩ := portsOf_entries (portsOf_of_ofJson hoj)
  obtain ⟨ej, hm, -, -, -, -, hv⟩ := hent reg hreg
  obtain ⟨-, -, -, -, -, -, hreads, -, -, -⟩ := Spot.SecondStage.stmtCodes_nil (List.isEmpty_iff.1 hc)
  have hents : Spot.entries (Spot.getD S.registered "ports") = kvs.toArray := by
    simp only [hSj, getD_of_ok hports, Spot.entries]
  rw [hents] at hreads
  have hrd : Spot.SecondStage.readOk (ownOf h st rec draw c₀) (Spot.SecondStage.byTemplate rec)
      (partAt h (ownOf h st rec draw c₀).sample.size (templateOf h.ms u)) reg.name ej = true :=
    Array.all_eq_true_iff_forall_mem.1 hreads _ hm
  exact ⟨st, rec, draw, ej, hst, hrec, hdraw, getD_of_ok hv, hrd, hin⟩

/-- **Each registered read of a session that sets up has the schema of the circuit's first port of its name**: the
session's verifier refuses a read whose rows are of another schema than the port's (`Registered.checkPort`). -/
theorem schema_of_setup {xS : Msg} {ins : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec}
    {sts : Array _root_.Flock.Stmt} {own} (hset : FlockDeploy.setupOf true xS = some (ins, spec, sts))
    (hown : ins.registered = some own) :
    ∀ st ∈ sts, ∀ reg ∈ own, ∃ p,
      (List.range st.c.ports.size).find? (fun p => st.c.ports[p]!.name == reg.name) = some p ∧
        _root_.Flock.Registered.portSchema st.c.ports[p]! = reg.schema := by
  intro st hst reg hreg
  obtain ⟨-, hsetup⟩ := FlockDeploy.setupOf_some hset
  have hck : _root_.Flock.checkRegistered st (some own) = .ok () := by
    rw [← hown]
    exact setup_checkRegistered hsetup st hst
  obtain ⟨tb, idx, -, -, hp⟩ := FlockSoundness.Discharge.ZkReg.check_tables
    (FlockSoundness.Discharge.ZkReg.checkRegistered_ok hck)
  exact _root_.Flock.Registered.checkPort_schema (hp reg hreg)

/-! ## The registered read named like a port -/

/-- **The registered read named like a port is one of the session's, of the port's name.** -/
theorem regNamed_spec {ins : FlockDeploy.Ins} {sts : Array _root_.Flock.Stmt} {q : ℕ} {name : String}
    (hname : (sts[0]!.c.ports[q]!).name = name) {reg : _root_.Flock.Registered.Port}
    (hreg : regNamed ins sts q = some reg) :
    ∃ own, ins.registered = some own ∧ reg ∈ own ∧ reg.name = name := by
  obtain ⟨own, hown, hfo⟩ := Option.bind_eq_some_iff.1 hreg
  have hp := Array.find?_some hfo
  rw [hname] at hp
  exact ⟨own, hown, Array.mem_of_find?_eq_some hfo, eq_of_beq hp⟩

/-- **A port one of the session's registered reads is named like has a registered read named like it.** -/
theorem regNamed_ne_none {ins : FlockDeploy.Ins} {sts : Array _root_.Flock.Stmt} {q : ℕ} {name : String}
    (hname : (sts[0]!.c.ports[q]!).name = name) {own : Array _root_.Flock.Registered.Port}
    {reg : _root_.Flock.Registered.Port} (hown : ins.registered = some own) (hmem : reg ∈ own)
    (hn : reg.name = name) : regNamed ins sts q ≠ none := by
  intro hnone
  have hnf : own.find? (fun reg => reg.name == (sts[0]!.c.ports[q]!).name) = none := by
    simpa [regNamed, hown] using hnone
  have hne : reg.name ≠ (sts[0]!.c.ports[q]!).name := by
    simpa using Array.find?_eq_none.1 hnf reg hmem
  exact hne (hn.trans hname.symm)

/-- A link the auditor's copies find for a template's port is one of the copies', of that template and port. -/
theorem link_named {t port : String} {l : Spot.SecondStage.Link} (hl : heldLink h t port = some l) :
    ∃ l ∈ h.links.getD #[], l.template = t ∧ l.port = port := by
  have hp := Array.find?_some hl
  simp only [Bool.and_eq_true, beq_iff_eq] at hp
  exact ⟨l, Array.mem_of_find?_eq_some hl, hp.1, hp.2⟩

/-- A wire the auditor's copies find for a template's port is one of the copies', of that template and port. -/
theorem wire_named {t port : String} {w : Spot.SecondStage.Wire} (hw : heldWire h t port = some w) :
    ∃ wr ∈ h.wires.getD #[], wr.template = t ∧ wr.port = port := by
  have hp := Array.find?_some hw
  simp only [Bool.and_eq_true, beq_iff_eq] at hp
  exact ⟨w, Array.mem_of_find?_eq_some hw, hp.1, hp.2⟩

variable {h} in
/-- **In an accepted run, a port a wire or a link names has a registered read of the checked unit's session named
like it** (`named_port_of_stmtOk`). -/
theorem CheckedRun.reg_of_named {r : CheckedRun h.stage} (hr : r.Ok h) {port : String}
    (hport : (∃ wr ∈ h.wires.getD #[], wr.template = h.names.getD (templateOf h.ms r.u) "" ∧ wr.port = port) ∨
      ∃ l ∈ h.links.getD #[], l.template = h.names.getD (templateOf h.ms r.u) "" ∧ l.port = port) :
    ∃ own reg, r.ins.registered = some own ∧ reg ∈ own ∧ reg.name = port := by
  obtain ⟨Pt, d, spec, hx, hP, h2, hin, hok, hu, hck, hb, hρ, hrb, hset⟩ := hr
  obtain ⟨ins, spec', sts, hset', hall⟩ := named_port_of_stmtOk h hok (templateOf_lt_of_checked h hx hu)
  have he : (ins, spec', sts) = (r.ins, spec, r.sts) := Option.some.inj (hset'.symm.trans hset)
  have hi : ins = r.ins := congrArg Prod.fst he
  obtain ⟨own, reg, hown, hmem, hn, -⟩ := hall port hport
  rw [hi] at hown
  exact ⟨own, reg, hown, hmem, hn⟩

/-! ## The roots a registered read is read under -/

/-- **Round 2 reads a value off the boundary record under the statement's roots.** -/
theorem readRoots_of_not_readsInterior {x m₁ : Msg} {c₀ : FlockDeploy.Coin} {c₁ : Stage h} {st : Roots}
    {rec smp : Json}
    (hx : decode x = some st) (hm : jsonMsg m₁ = some rec) (hd : secondDraw h c₀ c₁ = some smp) (ρ t : ℕ)
    (port v : String) (hri : readsInterior (ownOf h st rec smp c₀) (h.names.getD t "") port v = false) :
    readRoots h x m₁ c₀ c₁ ρ t port v = st.roots := by
  unfold readRoots
  rw [hx, hm, hd]
  dsimp only
  exact if_neg (by rw [hri]; simp)

/-- **The entry round 2 reads a registered read off**, where the run's statement decodes, its round-1 message is a
record and its second stage's sample evaluates. -/
theorem readKeyAt_eq {x m₁ : Msg} {c₀ : FlockDeploy.Coin} {c₁ : Stage h} {st : Roots} {rec smp : Json}
    (hx : decode x = some st) (hm : jsonMsg m₁ = some rec) (hd : secondDraw h c₀ c₁ = some smp) (t : ℕ)
    (port v : String) :
    readKeyAt h x m₁ c₀ c₁ t port v = readKey (ownOf h st rec smp c₀) (h.names.getD t "") port v := by
  unfold readKeyAt
  simp only [hx, hm, hd]

namespace CheckedRun

variable {h} {r : CheckedRun h.stage} {ω : ℕ → ℕ} {keep : ℕ → Bool}

/-- **Inside its replay block, the wired program opens a gate under the block's roots.** -/
theorem rootsWired_inside {g : ℕ}
    (hin : g ∈ cutBlock ω r.rb ∧ (r.s.prog.wired ω keep).Boundary (cutBlock ω r.rb) g = false) :
    r.rootsWired ω keep g = irootsOf r.m₁ r.ρ := by
  unfold rootsWired
  exact (if_pos hin).trans rfl

/-- **On its replay block's boundary, the wired program opens a gate under the statement's roots.** -/
theorem rootsWired_boundary {g : ℕ} (hB : (r.s.prog.wired ω keep).Boundary (cutBlock ω r.rb) g = true) :
    r.rootsWired ω keep g = r.s.roots := by
  unfold rootsWired
  exact if_neg fun hc => absurd (hB.symm.trans hc.2) (by decide)

end CheckedRun

end SpotFlock
