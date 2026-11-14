module

public import Proofs.Flock.SpotFlock.WiredKS
public import Proofs.Flock.SpotFlock.Cap
import all Init.Data.Array.Basic

/-!
# A two-stage registration where SPOT[Flock]'s two-stage premises are decided

`SpotFlock.Honest.held R B desc q c₀`: the auditor's copies (`Held`) of a registration of one template whose one
request is two units, of which the second stage proves one, over the template's reference statement `R` (`Ref`) and a
boundary registration `B` (`Bnd`) of one request sampled `subset:1`, at the descriptor `desc` and the statement's query
`q`, with SPOT's own frame at the wired program (`frameAt` at the wired address map, `revealCodeWired`, from the code
`c₀`). Every port of the template's statement is linked to the boundary port of its own name (`Ref.links`), each of
the request's two units reading its own row of the port (`Ref.rows`), there are no wires (`some #[]`) and the second
stage reads the first stage's coins only (`more := 0`). This is the token select's shape: one template over every
port, its input and output each at the boundary record.

What `SpotKSTwo` and `SpotExfilTwo` take of a registration, and what it is for its verifier to accept, are stated here
as `Premises`, `InScope` and `AcceptsPlay`; `HonestExists` says one registration has all three. `Accepts` is the run
with a checked unit and its witness, which the knowledge-soundness proofs read. At the instance:
* `builtWired_held`: the frame is SPOT's own at the wired program;
* `wiresFirst_held`, `heldWire_held`: there are no wires;
* `heldLink_held`: each of `R`'s ports has a link, and every row its units read source bits at is one of them
  (`Ref.named`);
* `popOk_held`: the boundary registration passes the run's population check at one unit;
* `two_held`, `scope_held`: where every member's circuit is in the scope Flock's hidden-output proof covers and the
  registered wiring passes its check (`membersScoped`, `wiringOk`), the design evaluates, and every statement that
  decodes is in the two-stage scope at the compiled code;
* `inScope_of_accepts`: an accepted play's held code passes both, so a play the verifier accepts gives `InScope`.

## What this does not show

* `AddressesConnected` at the instance: it holds only where the first stage's record carries each linked port and the
  interior registers none (`CheckedRun.not_addresses_of_input`, `CheckedRun.not_addresses_of_readsInterior`).
  `Honest.Sits` proves it from `R`'s reference reads being well formed (`addressesConnected_held`);
* a play the verifier accepts (`AcceptsPlay`): it needs a concrete `Ref`, a reference statement with hidden outputs and
  `sha512/row/v2` ports in the two-stage scope at two units a request, and an honest session of it. No such statement is
  defined in Lean yet.
-/

@[expose] public section

namespace SpotFlock.Honest

open Claims Claims.Sampling SamplingDeploy Properties.Spot
open Lean (Json)

/-! ## What the two-stage guarantees take, and what they say something about -/

/-- **The compiled two-stage verifier at the connected program**, the one `SpotKSTwo` and `SpotExfilTwo` state their
guarantee of. -/
noncomputable abbrev compiled (h : Held) :=
  SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (codeConnected h) SpotFlock.iroots
    (twoConnected h) (rsConnected h) (SpotFlock.cap h)

/-- **The premises `SpotKSTwo` takes of a registration**, in its order. -/
def Premises (h : Held) : Prop :=
  BuiltWired h (wireMap h) (returned h) ∧ AddressesConnected h

/-- **Some statement is in the two-stage scope**, the open premise of the compiled code's knowledge soundness
(`CompiledTwo.Scope`), and with it `InScopeTwo`, which `SpotExfilTwo` reads. -/
def InScope (h : Held) : Prop := ∃ x, (compiled h).Scope x

/-- **Some run is accepted with a checked unit** (`CheckedRun.Ok`), which holds a witness in Flock's relation at the
unit's session statement. Where none is, `AddressesConnected` holds vacuously. -/
def Accepts (h : Held) : Prop := ∃ r : CheckedRun h.stage, CheckedRun.Ok h r

/-- **SPOT[Flock]'s verifier accepts a play**: at SHA-512, the hash the two-stage guarantees are stated at, the
verifier at the registration (`V h`, the compiled verifier's: `compiled_V`) accepts some prover at some statement on
some coins. Where it accepts none, the two-stage guarantees bound an event that never happens. -/
def AcceptsPlay (h : Held) : Prop :=
  ∃ (Ω : Type) (P : Claims.Prover Ω) (x : Msg) (ω : Ω) (r : Fin ((V h).k x) → FlockDeploy.Coin),
    (Claims.run FlockDeploy.sha512 P (V h) x ω r).out = .acc

/-- The compiled two-stage verifier at the connected program runs SPOT[Flock]'s verifier. -/
theorem compiled_V (h : Held) : (compiled h).V = V h := rfl

/-- **The two-stage guarantees are not vacuous**: one registration meets their premises, has a statement in scope and
its verifier accepts a play. It asks for a play the verifier accepts and not for a witness (`Accepts`), since a play is
what a verifier's accept gives of a run, while a witness in Flock's relation is the prover's and follows from an
accept only through knowledge soundness, with its error. -/
def HonestExists : Prop := ∃ h : Held, Premises h ∧ InScope h ∧ AcceptsPlay h

/-! ## The registration -/

/-- **One unit's boundary layout**: a population of one stratum of one proof block, unit `0` at the gates `gs`, and an
empty input class. -/
def oneLayout (gs : ℕ × ℕ) : Spot.Slots.SlotOrder := ⟨#[#[#[⟨some 0, gs⟩]], #[]]⟩

/-- **The first stage's design** `subset:1`: one unit of the population, and every input slot. -/
def design1 : Design := ⟨[.subset 1, .all]⟩

/-- **One template with two units**: the statement circuit of `x` over two VUs. -/
def ms2 (x : Msg) : Spot.Bits.Members := [(stmtOf x, #[0, 1])]

/-- **Its replay partition** over one request's boundary layout, two units a request. -/
def part2 (x : Msg) : Partition := filesReplay (ms2 x) #[2] (oneLayout (0, 0))

theorem filesReplay_oneLayout (ms : Spot.Bits.Members) (ns : Array ℕ) (gs : ℕ × ℕ) :
    filesReplay ms ns (oneLayout gs) = filesReplay ms ns (oneLayout (0, 0)) := rfl

/-- **A boundary registration of one unit, sampled `subset:1`**, with the facts the two-stage design reads of it: its
layout is one proof block and its first stage's design is `subset:1`, which Flock reads as a uniform `1`-subset with
no closure map; and the names the boundary and interior registrations' records must name (`boundaryNames`, and
`interiorPorts` for the interior record's ports, which registers no value). -/
structure Bnd where
  g : Reg
  gs : ℕ × ℕ
  boundaryNames : Names
  interiorPorts : List String
  layout : Spot.Slots.layout g.program g.query = .ok (oneLayout gs)
  design : slotDesign g.rule g.program g.query g.table = .ok design1
  firstK : firstK g = some 1

/-- **A template's reference statement**, named `name`, with its ports (`ports`), each linked to the boundary port of
its own name, its reference registered reads (`reads`, `.null` for none), its reference header (`headers`), and the
facts the registration reads of it:
it sets up in Flock's masked scope, reads as a member, with its two units is in the two-stage scope, and each row
its units read source bits at (`sourceRows`) is one of its ports. -/
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

/-- **The second stage's links**: each port of the template's statement reads the boundary port of its own name. -/
def Ref.links (R : Ref) : Array Spot.SecondStage.Link := R.ports.map fun p => ⟨R.name, p, p⟩

/-- **The rows each port's units read**: two a request, unit `p` its row `p`. -/
def Ref.rows (R : Ref) : Array Spot.SecondStage.Rows := R.ports.map fun p => ⟨R.name, p, 2, #[⟨0, 2, 0, 1⟩]⟩

/-- **The two-stage registration** over the boundary `B`: two units a request and a count of one (`ns`, `ks`), request
`ρ`'s unit `p` its block `2ρ + p`, and no slot always checked. -/
def reg (B : Bnd) : RegTwo :=
  { first := B.g, ns := #[2], ks := #[1], blockOf := fun ρ _ p => ρ * 2 + p, always := fun _ => false }

/-- **The auditor's copies at the frame `c₀`**: one template, `R`'s, with two VUs, the second stage's count `1`, `R`'s
links and rows, its reference registered reads and header, no wires, and the first stage's coins only. -/
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

/-- **The auditor's copies**: `held₀`'s, with SPOT's own frame at the wired program from `c₀`. -/
noncomputable def held (R : Ref) (B : Bnd) (desc : Msg) (q : Query) (c₀ : Code) : Held :=
  let h := held₀ R B desc q c₀
  { h with
    frame := frameAt (slotsOf (addressMapWired h (wireMap h) (returned h)))
      (revealCodeWired h (wireMap h) (returned h) (addressMapWired h (wireMap h) (returned h)) c₀) }

variable (R : Ref) (B : Bnd) (desc : Msg) (q : Query) (c₀ : Code)

theorem ms_held : (held R B desc q c₀).ms = ms2 R.x := rfl

/-! ## The design's premises -/

/-- **Each wire's writer is first**: there are no wires. -/
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

/-- The wiring map doesn't read the frame. -/
theorem wireMap_frame (h : Held) (f : Code) : wireMap { h with frame := f } = wireMap h := rfl

/-- The gates the boundary record holds don't read the frame. -/
theorem returned_frame (h : Held) (f : Code) : returned { h with frame := f } = returned h := rfl

/-- **The frame is SPOT's own at the wired program.** -/
theorem builtWired_held :
    BuiltWired (held R B desc q c₀) (wireMap (held R B desc q c₀)) (returned (held R B desc q c₀)) := by
  have e₁ : wireMap (held R B desc q c₀) = wireMap (held₀ R B desc q c₀) := wireMap_frame _ _
  have e₂ : returned (held R B desc q c₀) = returned (held₀ R B desc q c₀) := returned_frame _ _
  rw [e₁, e₂]
  exact builtWired_spot (held₀ R B desc q c₀) (wireMap (held₀ R B desc q c₀)) (returned (held₀ R B desc q c₀)) c₀

/-! ## The design evaluates -/

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

/-- **The instance passes the run's population check at one unit** (`Slots.popOk`): its layout's population is the
one proof block, unit `0`. -/
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

/-- **The design evaluates** where every member's circuit is in the scope Flock's hidden-output proof covers and the
registered wiring passes its check: the members' replay partition over one request's boundary layout, the design
`subset:1` and the count `1`. -/
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

/-! ## The wiring moves nothing -/

theorem wireCell_held (t : ℕ) (port : String) (c : ℕ) : wireCell (held R B desc q c₀) t port c = none := by
  unfold wireCell
  split <;> simp_all [heldWire_held]

theorem wireTarget_held (x : ℕ) : wireTarget (held R B desc q c₀) x = none := by
  unfold wireTarget wireUnit
  simp [wireCell_held]

/-- **The wiring map is the identity**: there are no wires. -/
theorem wireMap_held : wireMap (held R B desc q c₀) = id :=
  funext fun x => by simp [wireMap, wireTarget_held]

/-! ## Returning fewer gates keeps the two-stage scope -/

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

/-- **Returning fewer gates keeps the two-stage scope**: the gates are the same, so is every read, and a gate is
committed only where it was. -/
theorem inScopeTwo_filter {Pt : Partition} {d : Design.Two} {s : AuditStatement} {C : Prog} (keep : ℕ → Bool)
    (h : InScopeTwo Pt d s) (hs : s.prog = C) :
    InScopeTwo Pt d { s with prog := ⟨C.gates, C.outs.filter keep⟩ } := by
  subst hs
  obtain ⟨ht, hp, hn, hr, hw⟩ := h
  exact ⟨⟨ht.1, fun o ho => ht.2 o (List.mem_filter.1 ho).1⟩, partitions_filter keep hp, hn, hr, hw⟩

/-! ## Every statement that parses is in the two-stage scope -/

/-- **A statement at the wired program**: the members' programs wired by the code's wiring map, returning the gates
the boundary record holds. -/
def wiredStmt (h : Held) (s : AuditStatement) : AuditStatement :=
  { s with prog := (bitsProgs h.ms).wired (wireMap h) (returned h) }

theorem parse_compiled (h : Held) :
    (compiled h).parse = parseWired h (wireMap h) (returned h) := rfl

theorem partition_compiled (h : Held) : (compiled h).partition = partitionWired h (wireMap h) := rfl

theorem two_compiled (h : Held) : (compiled h).two = twoConnected h := rfl

/-- **Every statement that decodes is in the two-stage scope at the compiled code**, where the design evaluates
(`two_held`): the program is the members' returning the gates the boundary record holds, the partition and design
are uncut since the wiring moves nothing, and the two-stage scope holds of the members' (`Ref.scope`), which
returning fewer gates keeps (`inScopeTwo_filter`). -/
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

/-- **An accepted play's statement is in scope**: the verifier's accept passed the session statements' check, at
which the statement decodes and the design evaluates (`refusesTwo`), so the held code passes what the design's
evaluation decides (`scoped_wiring_of_two`). -/
theorem inScope_of_accepts (hA : AcceptsPlay (held R B desc q c₀)) : InScope (held R B desc q c₀) := by
  obtain ⟨Ω, P, x, ω, r, hacc⟩ := hA
  obtain ⟨s, Pt, d, hp, -, h2, -⟩ := refusesTwo (held R B desc q c₀) FlockDeploy.sha512 x Ω P ω r hacc
  have hp' : parse (held R B desc q c₀) x = some s := hp
  obtain ⟨hS, hW⟩ := scoped_wiring_of_two _ h2
  cases hx : decode x with
  | none => simp [parse, hx] at hp'
  | some st => exact ⟨x, scope_held R B desc q c₀ hS hW x st hx⟩

/-! ## The statements the instance owes -/

/-- **What the instance owes for `HonestExists`**: its address premise and a play its verifier accepts, at some
reference statement and boundary registration. -/
def Owed : Prop :=
  ∃ (R : Ref) (B : Bnd) (desc : Msg) (q : Query) (c₀ : Code),
    AddressesConnected (held R B desc q c₀) ∧ AcceptsPlay (held R B desc q c₀)

/-- **What the instance owes is all `HonestExists` lacks.** -/
theorem honest_of_owed (hO : Owed) : HonestExists := by
  obtain ⟨R, B, desc, q, c₀, hA, hR⟩ := hO
  exact ⟨held R B desc q c₀, ⟨builtWired_held R B desc q c₀, hA⟩, inScope_of_accepts R B desc q c₀ hR, hR⟩

end SpotFlock.Honest
