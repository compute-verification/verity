module

public import Proofs.Flock.SpotFlock.WiredKS
public import Proofs.Flock.SpotFlock.Cap
import all Init.Data.Array.Basic

/-! A one-template two-stage registration (`held`) where SPOT[Flock]'s two-stage premises are decided; no concrete
reference statement `Ref` with an honest session is defined, so a play its verifier accepts is not shown. -/

@[expose] public section

namespace SpotFlock.Honest

open Claims Claims.Sampling SamplingDeploy Properties.Spot
open Lean (Json)

noncomputable abbrev compiled (h : Held) :=
  SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (codeConnected h) SpotFlock.iroots
    (twoConnected h) (rsConnected h) (SpotFlock.cap h)

def Premises (h : Held) : Prop :=
  BuiltWired h (wireMap h) (returned h) ∧ AddressesConnected h

/-- The open premise of the compiled code's knowledge soundness (`CompiledTwo.Scope`), and with it `InScopeTwo`. -/
def InScope (h : Held) : Prop := ∃ x, (compiled h).Scope x

/-- Where no run is accepted with a checked unit, `AddressesConnected` holds vacuously. -/
def Accepts (h : Held) : Prop := ∃ r : CheckedRun h.stage, CheckedRun.Ok h r

/-- Where the verifier accepts no play, the two-stage guarantees bound an event that never happens. -/
def AcceptsPlay (h : Held) : Prop :=
  ∃ (Ω : Type) (P : Claims.Prover Ω) (x : Msg) (ω : Ω) (r : Fin ((V h).k x) → FlockDeploy.Coin),
    (Claims.run FlockDeploy.sha512 P (V h) x ω r).out = .acc

theorem compiled_V (h : Held) : (compiled h).V = V h := rfl

/-- It asks for a play the verifier accepts and not a witness (`Accepts`), because a witness in Flock's relation
follows from an accept only through knowledge soundness, with its error. -/
def HonestExists : Prop := ∃ h : Held, Premises h ∧ InScope h ∧ AcceptsPlay h

def oneLayout (gs : ℕ × ℕ) : Spot.Slots.SlotOrder := ⟨#[#[#[⟨some 0, gs⟩]], #[]]⟩

def design1 : Design := ⟨[.subset 1, .all]⟩

def ms2 (x : Msg) : Spot.Bits.Members := [(stmtOf x, #[0, 1])]

def part2 (x : Msg) : Partition := filesReplay (ms2 x) #[2] (oneLayout (0, 0))

theorem filesReplay_oneLayout (ms : Spot.Bits.Members) (ns : Array ℕ) (gs : ℕ × ℕ) :
    filesReplay ms ns (oneLayout gs) = filesReplay ms ns (oneLayout (0, 0)) := rfl

structure Bnd where
  g : Reg
  gs : ℕ × ℕ
  boundaryNames : Names
  interiorPorts : List String
  layout : Spot.Slots.layout g.program g.query = .ok (oneLayout gs)
  design : slotDesign g.rule g.program g.query g.table = .ok design1
  firstK : firstK g = some 1

structure Ref where
  x : Msg
  name : String
  ports : Array String
  reads : Json
  headers : Json
  zk : zkOk x = true
  tmpl : ∀ h : Held, h.xS₀ 0 = x → templateOk h 0 = true
  scope : scopeOk (bitsProgs (ms2 x)) (membersPart (ms2 x)) ⟨part2 x, design1, [1]⟩ = true
  named : (sourceRows (stmtOf x)).all (fun row => ports.contains row) = true

def Ref.links (R : Ref) : Array Spot.SecondStage.Link := R.ports.map fun p => ⟨R.name, p, p⟩

def Ref.rows (R : Ref) : Array Spot.SecondStage.Rows := R.ports.map fun p => ⟨R.name, p, 2, #[⟨0, 2, 0, 1⟩]⟩

def reg (B : Bnd) : RegTwo :=
  { first := B.g, ns := #[2], ks := #[1], blockOf := fun ρ _ p => ρ * 2 + p, always := fun _ => false }

def held₀ (R : Ref) (B : Bnd) (desc : Msg) (q : Query) (c₀ : Code) : Held where
  desc := desc
  vus := [#[0, 1]]
  query := q
  reg := reg B
  k := [1]
  names := [R.name]
  links := some R.links
  wires := some #[]
  rows := R.rows
  boundaryNames := B.boundaryNames
  interiorNames := ⟨B.interiorPorts, []⟩
  xS₀ := fun _ => R.x
  xS₀_ok := fun _ => R.zk
  refReads := fun _ => R.reads
  refHeaders := fun _ => R.headers
  frame := c₀
  more := 0

noncomputable def held (R : Ref) (B : Bnd) (desc : Msg) (q : Query) (c₀ : Code) : Held :=
  let h := held₀ R B desc q c₀
  { h with
    frame := frameAt (slotsOf (addressMapWired h (wireMap h) (returned h)))
      (revealCodeWired h (wireMap h) (returned h) (addressMapWired h (wireMap h) (returned h)) c₀) }

variable (R : Ref) (B : Bnd) (desc : Msg) (q : Query) (c₀ : Code)

theorem ms_held : (held R B desc q c₀).ms = ms2 R.x := rfl

theorem wiresFirst_held : WiresFirst (held R B desc q c₀) := by
  simp [WiresFirst, writersFirst, held, held₀]

theorem heldWire_held (t port : String) : heldWire (held R B desc q c₀) t port = none := by
  simp [heldWire, held, held₀]

theorem heldLink_held (p : String) (hp : R.ports.contains p = true) :
    (heldLink (held R B desc q c₀) R.name p).isSome = true := by
  have hmem : p ∈ R.ports := by simpa using hp
  show ((R.ports.map fun p => (⟨R.name, p, p⟩ : Spot.SecondStage.Link)).find?
    fun l => l.template == R.name && l.port == p).isSome = true
  rw [Array.find?_isSome]
  exact ⟨⟨R.name, p, p⟩, Array.mem_map.2 ⟨p, hmem, rfl⟩, by simp⟩

theorem wireMap_frame (h : Held) (f : Code) : wireMap { h with frame := f } = wireMap h := rfl

theorem returned_frame (h : Held) (f : Code) : returned { h with frame := f } = returned h := rfl

theorem builtWired_held :
    BuiltWired (held R B desc q c₀) (wireMap (held R B desc q c₀)) (returned (held R B desc q c₀)) := by
  have e₁ : wireMap (held R B desc q c₀) = wireMap (held₀ R B desc q c₀) := wireMap_frame _ _
  have e₂ : returned (held R B desc q c₀) = returned (held₀ R B desc q c₀) := returned_frame _ _
  rw [e₁, e₂]
  exact builtWired_spot (held₀ R B desc q c₀) (wireMap (held₀ R B desc q c₀)) (returned (held₀ R B desc q c₀)) c₀

theorem replayDesign_held : replayDesign (held R B desc q c₀) = some design1 := by
  show (slotDesign B.g.rule B.g.program B.g.query B.g.table).toOption = some design1
  rw [B.design]
  rfl

theorem layout_held :
    (Spot.Slots.layout (held R B desc q c₀).reg.first.program (held R B desc q c₀).reg.first.query).toOption =
      some (oneLayout B.gs) := by
  show (Spot.Slots.layout B.g.program B.g.query).toOption = some (oneLayout B.gs)
  rw [B.layout]
  rfl

theorem popOk_held :
    Spot.Slots.popOk (held R B desc q c₀).reg.first.program (held R B desc q c₀).reg.first.query 1 = true := by
  show Spot.Slots.popOk B.g.program B.g.query 1 = true
  unfold Spot.Slots.popOk
  rw [B.layout]
  rfl

theorem filesOk_held : filesOk (reg B) (ms2 R.x) (oneLayout B.gs) [1] = true := rfl

theorem templates_held :
    (List.range (held R B desc q c₀).ms.length).all (templateOk (held R B desc q c₀)) = true := by
  rw [ms_held]
  show [0].all (templateOk (held R B desc q c₀)) = true
  simp only [List.all_cons, List.all_nil, Bool.and_true]
  exact R.tmpl _ rfl

theorem two_held (hS : membersScoped (held R B desc q c₀) = true) (hW : wiringOk (held R B desc q c₀) = true)
    (s : AuditStatement) :
    two (held R B desc q c₀) s = some ⟨filesReplay (ms2 R.x) #[2] (oneLayout B.gs), design1, [1]⟩ := by
  have hc : (scopeOk (bitsProgs (held R B desc q c₀).ms) (membersPart (held R B desc q c₀).ms)
      ⟨filesReplay (held R B desc q c₀).ms (held R B desc q c₀).reg.ns (oneLayout B.gs), design1,
        (held R B desc q c₀).k⟩ &&
      filesOk (held R B desc q c₀).reg (held R B desc q c₀).ms (oneLayout B.gs) (held R B desc q c₀).k &&
      (List.range (held R B desc q c₀).ms.length).all (templateOk (held R B desc q c₀)) &&
      membersScoped (held R B desc q c₀) && wiringOk (held R B desc q c₀)) = true := by
    rw [hS, hW, templates_held, ms_held]
    simp only [Bool.and_true, Bool.and_eq_true]
    refine ⟨?_, filesOk_held R B⟩
    rw [filesReplay_oneLayout]
    exact R.scope
  unfold two
  rw [layout_held, replayDesign_held]
  dsimp only
  rw [ite_eq_left hc, ms_held]
  rfl

theorem wireCell_held (t : ℕ) (port : String) (c : ℕ) : wireCell (held R B desc q c₀) t port c = none := by
  unfold wireCell
  split <;> simp_all [heldWire_held]

theorem wireTarget_held (x : ℕ) : wireTarget (held R B desc q c₀) x = none := by
  unfold wireTarget wireUnit
  simp [wireCell_held]

theorem wireMap_held : wireMap (held R B desc q c₀) = id :=
  funext fun x => by simp [wireMap, wireTarget_held]

theorem wired_id_keep (C : Prog) (keep : ℕ → Bool) : C.wired id keep = ⟨C.gates, C.outs.filter keep⟩ := by
  cases C with
  | mk gates outs =>
    have hg : gates.map (BGate.rewire id) = gates :=
      (List.map_congr_left fun x _ => BGate.rewire_id x).trans (List.map_id _)
    show (⟨gates.map (BGate.rewire id), outs.filter keep⟩ : Prog) = ⟨gates, outs.filter keep⟩
    rw [hg]

theorem cut_id (P : Partition) : P.cut id = P := by
  cases P with
  | mk classes =>
    show (⟨classes.map (List.map (List.map (cutBlock id)))⟩ : Partition) = ⟨classes⟩
    rw [show cutBlock id = id from funext cutBlock_id]
    simp

theorem boundary_filter {C : Prog} {keep : ℕ → Bool} {u : List ℕ} {g : ℕ}
    (hb : Prog.Boundary ⟨C.gates, C.outs.filter keep⟩ u g = true) : C.Boundary u g = true := by
  unfold Prog.Boundary at hb ⊢
  simp only [Bool.or_eq_true] at hb ⊢
  rcases hb with (hb | hb) | hb
  · exact Or.inl (Or.inl (List.elem_eq_true_of_mem (List.mem_filter.1 (List.mem_of_elem_eq_true hb)).1))
  · exact Or.inl (Or.inr hb)
  · exact Or.inr hb

theorem committed_filter_le (C : Prog) (keep : ℕ → Bool) (u : List ℕ) :
    committed ⟨C.gates, C.outs.filter keep⟩ u ≤ committed C u := by
  unfold committed
  rw [← List.countP_eq_length_filter, ← List.countP_eq_length_filter]
  exact List.countP_mono_left fun _ _ hg => boundary_filter hg

theorem partitions_filter {P : Partition} {C : Prog} (keep : ℕ → Bool) (hP : P.Partitions C) :
    P.Partitions ⟨C.gates, C.outs.filter keep⟩ := by
  obtain ⟨hnd, hlt, hall, hne, hcls⟩ := hP
  refine ⟨hnd, hlt, fun g hg hc hu => hall g hg hc fun hU => hu ⟨hU.1, fun ho => hU.2.1 (List.mem_filter.1 ho).1,
    hU.2.2⟩, hne, fun u hu => ?_⟩
  obtain ⟨w, hw, hle⟩ := hcls u hu
  have hm : w ∈ sizeClasses := List.mem_of_find?_eq_some hw
  have hsome : (sizeClasses.find? fun w => decide (committed ⟨C.gates, C.outs.filter keep⟩ u ≤ w)).isSome :=
    List.find?_isSome.2 ⟨w, hm, decide_eq_true ((committed_filter_le C keep u).trans hle)⟩
  obtain ⟨w', hw'⟩ := Option.isSome_iff_exists.1 hsome
  have hp := List.find?_some hw'
  exact ⟨w', hw', by simpa using hp⟩

theorem inScopeTwo_filter {Pt : Partition} {d : Design.Two} {s : AuditStatement} {C : Prog} (keep : ℕ → Bool)
    (h : InScopeTwo Pt d s) (hs : s.prog = C) :
    InScopeTwo Pt d { s with prog := ⟨C.gates, C.outs.filter keep⟩ } := by
  subst hs
  obtain ⟨ht, hp, hn, hr, hw⟩ := h
  exact ⟨⟨ht.1, fun o ho => ht.2 o (List.mem_filter.1 ho).1⟩, partitions_filter keep hp, hn, hr, hw⟩

def wiredStmt (h : Held) (s : AuditStatement) : AuditStatement :=
  { s with prog := (bitsProgs h.ms).wired (wireMap h) (returned h) }

theorem parse_compiled (h : Held) :
    (compiled h).parse = parseWired h (wireMap h) (returned h) := rfl

theorem partition_compiled (h : Held) : (compiled h).partition = partitionWired h (wireMap h) := rfl

theorem two_compiled (h : Held) : (compiled h).two = twoConnected h := rfl

theorem scope_held (hS : membersScoped (held R B desc q c₀) = true) (hW : wiringOk (held R B desc q c₀) = true)
    (x : Msg) (st : Roots) (hx : decode x = some st) : (compiled (held R B desc q c₀)).Scope x := by
  obtain ⟨s₁, hs₁⟩ : ∃ s₁, parse (held R B desc q c₀) x = some s₁ := by simp [parse, hx]
  refine ⟨wiredStmt (held R B desc q c₀) s₁, membersPart (ms2 R.x),
    ⟨filesReplay (ms2 R.x) #[2] (oneLayout B.gs), design1, [1]⟩, ?_, ?_, ?_, ?_⟩
  · rw [parse_compiled]
    simp [parseWired, hs₁, wiredStmt]
  · rw [partition_compiled]
    show ((partition (held R B desc q c₀) _).map fun P => P.cut (wireMap (held R B desc q c₀))) = _
    rw [wireMap_held]
    simp only [partition, Option.map_some, cut_id]
    rfl
  · rw [two_compiled]
    show ((two (held R B desc q c₀) _).map fun d =>
      ({ d with part := d.part.cut (wireMap (held R B desc q c₀)) } : Design.Two)) = _
    rw [two_held R B desc q c₀ hS hW, wireMap_held]
    simp only [Option.map_some, cut_id]
  · unfold wiredStmt
    rw [wireMap_held, wired_id_keep]
    exact inScopeTwo_filter (s := { s₁ with prog := bitsProgs (held R B desc q c₀).ms }) _
      (scopeOk_sound R.scope _ rfl) rfl

theorem inScope_of_accepts (hA : AcceptsPlay (held R B desc q c₀)) : InScope (held R B desc q c₀) := by
  obtain ⟨Ω, P, x, ω, r, hacc⟩ := hA
  obtain ⟨s, Pt, d, hp, -, h2, -⟩ := refusesTwo (held R B desc q c₀) FlockDeploy.sha512 x Ω P ω r hacc
  have hp' : parse (held R B desc q c₀) x = some s := hp
  obtain ⟨hS, hW⟩ := scoped_wiring_of_two _ h2
  cases hx : decode x with
  | none => simp [parse, hx] at hp'
  | some st => exact ⟨x, scope_held R B desc q c₀ hS hW x st hx⟩

def Owed : Prop :=
  ∃ (R : Ref) (B : Bnd) (desc : Msg) (q : Query) (c₀ : Code),
    AddressesConnected (held R B desc q c₀) ∧ AcceptsPlay (held R B desc q c₀)

theorem honest_of_owed (hO : Owed) : HonestExists := by
  obtain ⟨R, B, desc, q, c₀, hA, hR⟩ := hO
  exact ⟨held R B desc q c₀, ⟨builtWired_held R B desc q c₀, hA⟩, inScope_of_accepts R B desc q c₀ hR, hR⟩

end SpotFlock.Honest
