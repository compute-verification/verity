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
stage reads the coins of round 1 only (`more := 0`). This is the token select's shape: one template over every port,
its input and output each at the boundary record.

What `SpotKSTwo` and `SpotExfilTwo` take of a registration, and what an accepted run of it is, are stated here as
`Premises`, `InScope` and `Accepts`; `Honest` says one registration has all three. At the instance:
* `wiresFirst_held`, `wiresMove_held`: there are no wires;
* `sourcesNamed_held`: every source bit's port is one of `R`'s ports (`Ref.named`), each of which a link names;
* `builtWired_held`: the frame is SPOT's own at the wired program;
* `two_held`: the design evaluates.

## What this does not show

* `InScope` at the instance: the compiled code's design is `twoConnected`, the design at the wiring map, which
  `two_held` doesn't yet reach;
* `AddressesConnected` at the instance: it holds only where round 1's record carries each linked port and the interior
  registers none (`not_addresses_of_input`, `not_addresses_of_readsInterior`);
* an accepted run (`Accepts`): it needs a concrete `Ref`, a reference statement with hidden outputs and
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
  WiresFirst h ∧ WiresMove h ∧ SourcesNamed h ∧ BuiltWired h (wireMap h) (returned h) ∧ AddressesConnected h

/-- **Some statement is in the two-stage scope**, the open premise of the compiled code's knowledge soundness
(`CompiledTwo.Scope`), and with it `InScopeTwo`, which `SpotExfilTwo` reads. -/
def InScope (h : Held) : Prop := ∃ x, (compiled h).Scope x

/-- **Some run is accepted with a checked unit** (`CheckedRun.Ok`). Where none is, `AddressesConnected` holds
vacuously and the two-stage guarantees say nothing at the registration. -/
def Accepts (h : Held) : Prop := ∃ r : CheckedRun h.stage, CheckedRun.Ok h r

/-- **The two-stage guarantees are not vacuous**: one registration meets their premises, has a statement in scope and
accepts a run. -/
def Honest : Prop := ∃ h : Held, Premises h ∧ InScope h ∧ Accepts h

/-! ## The registration -/

/-- **One unit's boundary layout**: a population of one stratum of one proof block, unit `0` at the gates `gs`, and an
empty input class. -/
def oneLayout (gs : ℕ × ℕ) : Spot.Slots.Layout := ⟨#[#[#[⟨some 0, gs⟩]], #[]]⟩

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
no closure map. -/
structure Bnd where
  g : Reg
  gs : ℕ × ℕ
  layout : Spot.Slots.layout g.program g.query = .ok (oneLayout gs)
  design : slotDesign g.rule g.program g.query g.table = .ok design1
  firstK : firstK g = some 1

/-- **A template's reference statement**, named `name`, with its ports (`ports`), each linked to the boundary port of
its own name, its reference registered reads (`reads`, `.null` for none), and the facts the registration reads of it:
it sets up in Flock's masked scope, reads as a member, with its two units is in the two-stage scope, and each of its
source bits is at one of its ports. -/
structure Ref where
  x : Msg
  name : String
  ports : Array String
  reads : Json
  zk : zkOk x = true
  tmpl : ∀ h : Held, h.xS₀ 0 = x → templateOk h 0 = true
  scope : scopeOk (bitsProgs (ms2 x)) (membersPart (ms2 x)) ⟨part2 x, design1, [1]⟩ = true
  named : (sourceBits (ms2 x)).all (fun s => ports.contains s.2.2.row) = true

/-- **Round 2's links**: each port of the template's statement reads the boundary port of its own name. -/
def Ref.links (R : Ref) : Array Spot.SecondStage.Link := R.ports.map fun p => ⟨R.name, p, p⟩

/-- **The rows each port's units read**: two a request, unit `p` its row `p`. -/
def Ref.rows (R : Ref) : Array Spot.SecondStage.Rows := R.ports.map fun p => ⟨R.name, p, 2, #[⟨0, 2, 0, 1⟩]⟩

/-- **The two-stage registration** over the boundary `B`: two units a request and a count of one (`ns`, `ks`), request
`ρ`'s unit `p` its block `2ρ + p`, and no slot always checked. -/
def reg (B : Bnd) : RegTwo :=
  { first := B.g, ns := #[2], ks := #[1], blockOf := fun ρ _ p => ρ * 2 + p, always := fun _ => false }

/-- **The auditor's copies at the frame `c₀`**: one template, `R`'s, with two VUs, the second stage's count `1`, `R`'s
links and rows, its reference registered reads, no wires, and the coins of round 1 only. -/
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
  xS₀ := fun _ => R.x
  xS₀_ok := fun _ => R.zk
  refReads := fun _ => R.reads
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

/-- **Every wired source bit moves**: there are no wires. -/
theorem wiresMove_held : WiresMove (held R B desc q c₀) := by
  unfold WiresMove wiresMove
  simp [heldWire_held]

theorem heldLink_held (p : String) (hp : R.ports.contains p = true) :
    (heldLink (held R B desc q c₀) R.name p).isSome = true := by
  have hmem : p ∈ R.ports := by simpa using hp
  show ((R.ports.map fun p => (⟨R.name, p, p⟩ : Spot.SecondStage.Link)).find?
    fun l => l.template == R.name && l.port == p).isSome = true
  rw [Array.find?_isSome]
  exact ⟨⟨R.name, p, p⟩, Array.mem_map.2 ⟨p, hmem, rfl⟩, by simp⟩

/-- **Every source bit is at a port a link names**: each is at one of `R`'s ports (`Ref.named`). -/
theorem sourcesNamed_held : SourcesNamed (held R B desc q c₀) := by
  unfold SourcesNamed sourcesNamed
  rw [List.all_eq_true]
  intro s hs
  rw [ms_held] at hs
  have hn : R.ports.contains s.2.2.row = true := List.all_eq_true.1 R.named s hs
  have ht : s.2.1 = 0 := by
    unfold sourceBits at hs
    simp only [List.mem_flatMap, List.mem_filterMap, List.mem_range] at hs
    obtain ⟨m, hm, g, -, hg⟩ := hs
    have hm0 : m = 0 := by simpa [ms2] using hm
    subst hm0
    split at hg
    · simp at hg
    · obtain ⟨r, -, rfl⟩ := Option.map_eq_some_iff.1 hg
      rfl
  have e : [R.name].getD s.2.1 "" = R.name := by rw [ht]; rfl
  show ((heldLink (held R B desc q c₀) ([R.name].getD s.2.1 "") s.2.2.row).isSome ||
    (heldWire (held R B desc q c₀) ([R.name].getD s.2.1 "") s.2.2.row).isSome) = true
  rw [e, heldLink_held R B desc q c₀ _ hn, Bool.true_or]

/-- **The frame is SPOT's own at the wired program.** -/
theorem builtWired_held :
    BuiltWired (held R B desc q c₀) (wireMap (held R B desc q c₀)) (returned (held R B desc q c₀)) :=
  builtWired_spot (held₀ R B desc q c₀) (wireMap (held₀ R B desc q c₀)) (returned (held₀ R B desc q c₀)) c₀

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

theorem filesOk_held : filesOk (reg B) (ms2 R.x) (oneLayout B.gs) [1] = true := rfl

theorem templates_held :
    (List.range (held R B desc q c₀).ms.length).all (templateOk (held R B desc q c₀)) = true := by
  rw [ms_held]
  show [0].all (templateOk (held R B desc q c₀)) = true
  simp only [List.all_cons, List.all_nil, Bool.and_true]
  exact R.tmpl _ rfl

/-- **The design evaluates**: the members' replay partition over one request's boundary layout, the design `subset:1`
and the count `1`. -/
theorem two_held (s : AuditStatement) :
    two (held R B desc q c₀) s = some ⟨filesReplay (ms2 R.x) #[2] (oneLayout B.gs), design1, [1]⟩ := by
  have hc : (scopeOk (bitsProgs (held R B desc q c₀).ms) (membersPart (held R B desc q c₀).ms)
      ⟨filesReplay (held R B desc q c₀).ms (held R B desc q c₀).reg.ns (oneLayout B.gs), design1,
        (held R B desc q c₀).k⟩ &&
      filesOk (held R B desc q c₀).reg (held R B desc q c₀).ms (oneLayout B.gs) (held R B desc q c₀).k &&
      (List.range (held R B desc q c₀).ms.length).all (templateOk (held R B desc q c₀))) = true := by
    rw [templates_held, ms_held]
    simp only [Bool.and_true, Bool.and_eq_true]
    refine ⟨?_, filesOk_held R B⟩
    rw [filesReplay_oneLayout]
    exact R.scope
  unfold two
  rw [layout_held, replayDesign_held]
  dsimp only
  rw [ite_eq_left hc, ms_held]
  rfl

/-! ## The statements the instance owes -/

/-- **The instance's premises, short of the address premise**: what `wiresFirst_held`, `wiresMove_held`,
`sourcesNamed_held` and `builtWired_held` give. -/
theorem design_held :
    WiresFirst (held R B desc q c₀) ∧ WiresMove (held R B desc q c₀) ∧ SourcesNamed (held R B desc q c₀) ∧
      BuiltWired (held R B desc q c₀) (wireMap (held R B desc q c₀)) (returned (held R B desc q c₀)) :=
  ⟨wiresFirst_held R B desc q c₀, wiresMove_held R B desc q c₀, sourcesNamed_held R B desc q c₀,
    builtWired_held R B desc q c₀⟩

/-- **What the instance owes for `Honest`**: its address premise, a statement in scope and an accepted run, at some
reference statement and boundary registration. -/
def Owed : Prop :=
  ∃ (R : Ref) (B : Bnd) (desc : Msg) (q : Query) (c₀ : Code),
    AddressesConnected (held R B desc q c₀) ∧ InScope (held R B desc q c₀) ∧ Accepts (held R B desc q c₀)

/-- **What the instance owes is all `Honest` lacks.** -/
theorem honest_of_owed (hO : Owed) : Honest := by
  obtain ⟨R, B, desc, q, c₀, hA, hS, hR⟩ := hO
  obtain ⟨h1, h2, h3, h4⟩ := design_held R B desc q c₀
  exact ⟨held R B desc q c₀, ⟨h1, h2, h3, h4, hA⟩, hS, hR⟩

end SpotFlock.Honest
