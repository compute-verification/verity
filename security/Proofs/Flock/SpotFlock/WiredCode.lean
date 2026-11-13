module

public import Proofs.Flock.SpotFlock.RevealsWired
public import Proofs.Flock.SpotFlock.WireMapRewires
public import Proofs.Flock.SpotFlock.KS
import all Lean.Data.Json.Basic

/-!
# SPOT[Flock]'s two-stage code at the connected program

`code h` (`Code`) reads the members' programs side by side (`bitsProgs h.ms`), where no gate of one proof block's
circuit reads a gate of another. This file defines the same code at the members' programs wired by a map `ω`
(`Prog.wired`): in the statement's program every gate reads wire `ω i` where it read wire `i`, so a gate `ω` moves
stays in the program, an input gate that no gate reads and no proof block holds; each proof block and replay block
is without the gates `ω` moves; and a session reveals each opening at the gate `ω` sends it to, under the roots the
wired program opens that gate under (`partitionWired`, `twoWired`, `revealWired`, `rootsAtWired`, `RevealsWired`). The
verifier, the sessions, their rounds, the samples and the frame are `code h`'s.

* `parseWired`, `baseWired`, `codeWired`, `rsWired`: `parse h`, `base h`, `code h` and `rs h` at the wired program.
* `codeConnected h`, `twoConnected h`, `rsConnected h`: the code, the design and the replay sample at the code's own
  wiring map (`wireMap h`, `returned h`). The replay sample reads the registration and the first stage's coins only,
  so `rsConnected h` is `rs h` itself.
* `WiresFirst h`, `WiresMove h`, `SourcesNamed h`: three conditions on the code's own wiring, each decided on what the
  verifier holds and none decided by the verifier, as named premises.
* `rows_of_two`: where the design evaluates, every member with a unit has a computed row.
* `rewires_of_two`: where the design evaluates and each wire's writer's part is before its reader's, the code's
  wiring map rewires the members' programs (`Prog.Rewires`).
* `named_port_of_stmtOk`: in an accepted run, each port a wire or a link names for a template is a port of that
  template's session circuit, by that name. A run whose session statement's circuit has that port's rows under other
  names is refused; the lemma's docstring says what ties that circuit's ports to the member's.

The verifier is `V h` itself. Its check of the session statements (`stmtOk`) holds each session's reads to the
links and wires the verifier holds and runs round 2's check of the design on the run (`readsOk`,
`SecondStage.designOk`). A statement about the connected code reads the three conditions wherever the design
evaluates, at a statement no run accepted too (`rewires_of_two`), and the design's evaluation (`two h`) decides none
of them. So they are premises of each statement about the connected code, and such a statement says nothing at a held
code that fails one.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy Properties.Spot Finset

variable (h : Held) (ω : ℕ → ℕ) (keep : ℕ → Bool)

/-- **The statement at the wired program**: `parse h`'s, its program the members' wired by `ω`. -/
def parseWired (x : Msg) : Option AuditStatement :=
  (parse h x).map fun s => { s with prog := (bitsProgs h.ms).wired ω keep }

/-- **The code at the wired program, before its samples**: `base h` with the statement, the partition and the reveal
at the wired program. The reveal count is the side-by-side one (`length_revealWired_le`). -/
noncomputable def baseWired : Code :=
  { h.frame with
    V := V h
    parse := parseWired h ω keep
    scopeCheck := fun x => ((parseWired h ω keep x).bind (partitionWired h ω)).isSome
    partition := partitionWired h ω
    stage2 := fun _ => univ
    checked := fun _ _ => []
    rounds := fun _ u => univ.filter fun t =>
      off h (templateOf h.ms u) ≤ t.val ∧ t.val < off h (templateOf h.ms u) + k₀ h (templateOf h.ms u)
    session := fun _ u v => session h u v
    reveal := revealWired h ω keep h.frame
    reveals := Reveal.reveals (partition h) (parse h)
    sessions := fun _ => h.ms.length
    tables := fun _ => ((List.range (h.ms.length + 1)).map fun t => (shapeOf (h.xS₀ t)).length).foldr max 0
    unitN₀ := fun _ u => n₀ h (templateOf h.ms u)
    innerN₀ := fun _ => n₀ h 0 }

/-- The run's round order at the wired code: `order h`, the verifier being `V h`. -/
def orderWired (x : Msg) : Order ((baseWired h ω keep).V.k x) := order h x

/-- The run's stage order at the wired code: `stageOrder h`. -/
def stageOrderWired (x : Msg) : StageOrder ((baseWired h ω keep).V.k x) h.stage := stageOrder h x

/-- **The two-stage code at the wired program**: `baseWired` at its stage order, its checked units the two stages'
on the auditor's registration, the first stage's bytes round 0's coin's and the second stage's those of its rounds'
coins (`code h`'s shape). -/
noncomputable abbrev codeWired : Code :=
  codeTwoStage (baseWired h ω keep) (reg h) (fun _ => L) (fun _ => h.stage * L) (fun _ => coinByte)
    (fun _ => h.stage) (stageOrderWired h ω keep) fun _ => stageByte

/-- **The replay sample of record at the wired code**: the first stage's checked slots on the replay layout
(`rs h`'s shape). -/
noncomputable def rsWired : (x : Msg) → (Fin ((codeWired h ω keep).V.k x) → FlockDeploy.Coin) → List ℕ :=
  replayTwo ((baseWired h ω keep).atOrder (orderWired h ω keep)) (reg h) (fun _ => L)
    (bytesFirst (baseWired h ω keep) (fun _ => L) (orderWired h ω keep) fun _ => coinByte)

/-! ## The code's own wiring -/

/-- **SPOT[Flock]'s two-stage code at the connected program**: `codeWired` at the code's own wiring map and the
gates the boundary record holds. Its verifier is `V h`. -/
noncomputable abbrev codeConnected : Code := codeWired h (wireMap h) (returned h)

/-- **The two-stage design at the connected program**: `two h`'s with each replay block without the gates the wiring
moves. -/
def twoConnected : AuditStatement → Option Design.Two := twoWired h (wireMap h)

/-- **The replay sample of record at the connected program.** -/
noncomputable def rsConnected : (x : Msg) → (Fin ((codeConnected h).V.k x) → FlockDeploy.Coin) → List ℕ :=
  rsWired h (wireMap h) (returned h)

/-! ## The premises on the code's own wiring -/

/-- **Each wire's writer's part is before its reader's** (`writersFirst`), as a premise. A wire whose writer names no
template fails it. Round 2 decides it at a run's parts (`SecondStage.wiresOk`, fourth clause, refused as
`second-wires`), and `V h` runs that check on each run (`readsOk`); a statement about the connected code reads the
condition wherever the design evaluates, which `two h` does not decide. With no wires held it holds, where `wiresOk`
fails. -/
def WiresFirst : Prop := writersFirst h = true

/-- **Every wired source bit has a gate to go to** (`wiresMove`), as a premise. Nothing the verifier runs decides
it. It enters no step of the proof of a statement about the connected code: such a statement takes it so that the
connected program has no wired read left as a free input. Under it and `SourcesNamed`, a source bit at a port no
link names is sent to its writer's output (`wireTarget_of_sourcesNamed`), so only the source bits at a port a link
names stay inputs of the connected program. -/
def WiresMove : Prop := wiresMove h = true

/-- **Every source bit is at a port a link or a wire names** (`sourcesNamed`), as a premise. Nothing the verifier
runs decides it. It is what `AddressesConnected h` asks before it says where a gate's value sits, so a statement
that takes both says nothing of a held code with a source bit at a port neither names: an input the developer
supplies, or a row round 2 opens off the interior record. -/
def SourcesNamed : Prop := sourcesNamed h = true

/-! ## The wiring map rewires the members' programs -/

/-- **A unit of a key program whose units have no row has an empty proof block**: no gate is in a unit, so no
input has an owner. -/
theorem blockOf_of_perUnit_zero {kp : Spot.Bits.KeyProg} (h0 : kp.perUnit = 0) (κ : ℕ → Option Bool) (u : ℕ) :
    Spot.Bits.blockOf kp κ u = [] := by
  have hu : ∀ g, Spot.Bits.unitOf kp g = none := fun g => by simp [Spot.Bits.unitOf, h0]
  have ho : ∀ g, Spot.Bits.owner kp g = none := fun g => by simp [Spot.Bits.owner, hu]
  simp [Spot.Bits.blockOf, Spot.Bits.unitGates, Spot.Bits.insOf, hu, ho]

variable {h} in
/-- **Where the design evaluates, every member with a unit has a computed row**: the two-stage scope refuses an
empty proof block (`partitionsOk`), and a unit of a member with no computed row has one
(`blockOf_of_perUnit_zero`). -/
theorem rows_of_two {s : AuditStatement} {d : Design.Two} (hd : two h s = some d) :
    ∀ ci ∈ h.ms, 0 < ci.2.size → 0 < (Flock.Bridge.bitsRows ci.1).nComp := by
  intro ci hci hn
  obtain ⟨l, -, -, -, hsc, -⟩ := two_eq h hd
  obtain ⟨-, ⟨-, -, -, hne, -⟩, -⟩ := scopeOk_sound hsc { s with prog := bitsProgs h.ms } rfl
  obtain ⟨m, hm, rfl⟩ := List.getElem_of_mem hci
  refine Nat.pos_of_ne_zero fun h0 => ?_
  have h0' : (Spot.Bits.keyProg h.ms[m].1 h.ms[m].2.size).perUnit = 0 :=
    (Flock.Bridge.keyProg_perUnit h.ms[m].1 h.ms[m].2.size).trans h0
  have hb : (Spot.Bits.blockOf (Spot.Bits.keyProg h.ms[m].1 h.ms[m].2.size)
      (Spot.Bits.consts h.ms[m].1 h.ms[m].2.size) 0).map (Spot.Bits.base h.ms m + ·) ∈
        (membersPart h.ms).blocks := by
    rw [Flock.Bridge.Members.blocks_membersPart]
    refine List.mem_flatMap.2 ⟨m, List.mem_range.2 hm, ?_⟩
    rw [Flock.Bridge.Members.memberBlocks_eq h.ms hm]
    exact List.mem_map.2 ⟨_, List.mem_map.2 ⟨0, List.mem_range.2 hn, rfl⟩, rfl⟩
  exact hne _ hb (by rw [blockOf_of_perUnit_zero h0', List.map_nil])

variable {h} in
/-- **Where the design evaluates and each wire's writer's part is before its reader's, the code's wiring map
rewires the members' programs** (`rewires_wireMap`): each member's unit is built by its template's check
(`templateOk_facts`), which the design asks of every template, and each member with a unit has a computed row
(`rows_of_two`). -/
theorem rewires_of_two {s : AuditStatement} {d : Design.Two} (hd : two h s = some d) (hF : WiresFirst h) :
    (bitsProgs h.ms).Rewires (membersPart h.ms) (wireMap h) (returned h) := by
  have hT : ∀ ci ∈ h.ms, FlockSoundness.Built ci.1.unit ∧ ci.1.unit.unitLog ≤ 32 := fun ci hci => by
    obtain ⟨t, hlt, rfl⟩ := List.getElem_of_mem hci
    exact (templateOk_facts h (templateOk_of_two h hd hlt) hlt).2
  exact rewires_wireMap h (fun ci hci => (hT ci hci).1) (fun ci hci => (hT ci hci).2) (rows_of_two hd) hF

/-! ## What the verifier refuses of a design -/

/-- **In an accepted run, a port a wire or a link names is a port of its template's session circuit.** Template
`t`'s session statement sets up, and each port a wire or a link of the auditor's copies names for the template is the
name of one of the session's registered reads and of a port of each of its statement circuits. So a run whose session
statement's circuit has no port of that name is refused: its session's registered reads either leave the port out,
which round 2's statement check refuses (`SecondStage.wiredRead`, `linkedRead`, the code `second-read`), or name a
port the circuit doesn't have, which the session's verifier refuses (`Flock.Registered.checkPort`). The circuits here
are the session statement's, not the member's, and no lemma states that the member's circuit has their ports. The step
is made inside the proof of `CheckedRun.v2Row_port`, for the first table of an accepted run's checked unit's session:
the table's circuit unrolls, ports and all (`FlockSoundness.unrolled_ports`), to its template's reference statement's
statement circuit (`pin_of_stmtOkAt`, `stmt_of_pin`), which is the member's (`templateOk_facts`). The refusal is the
run's and not the registration's: the design still evaluates. -/
theorem named_port_of_stmtOk {x m₁ m₂ : Msg} {c₀ : FlockDeploy.Coin} {c₁ : Stage h}
    (hok : stmtOk h x m₁ m₂ c₀ c₁ = true) {t : ℕ} (ht : t < h.ms.length) :
    ∃ ins spec sts, FlockDeploy.setupOf true (stmtAt h m₂ t) = some (ins, spec, sts) ∧
      ∀ port : String,
        ((∃ wr ∈ h.wires.getD #[], wr.template = h.names.getD t "" ∧ wr.port = port) ∨
          ∃ l ∈ h.links.getD #[], l.template = h.names.getD t "" ∧ l.port = port) →
        ∃ own reg, ins.registered = some own ∧ reg ∈ own ∧ reg.name = port ∧
          ∀ sc ∈ sts, ∃ p, (List.range sc.c.ports.size).find? (fun p => sc.c.ports[p]!.name == port) = some p := by
  have hat := stmtOkAt_of h hok ht
  obtain ⟨ins, spec, sts, hset⟩ :
      ∃ ins spec sts, FlockDeploy.setupOf true (stmtAt h m₂ t) = some (ins, spec, sts) := by
    have hs := sessionOk_of_stmtOkAt h hat
    simp only [sessionOk, Bool.and_eq_true] at hs
    obtain ⟨ins, spec, sts, hset, -⟩ := zkOk_sound hs.1.1.1
    exact ⟨ins, spec, sts, hset⟩
  refine ⟨ins, spec, sts, hset, fun port hport => ?_⟩
  obtain ⟨s, hx⟩ := parse_of_stmtOkAt h hat
  unfold stmtOkAt at hat
  rw [hx] at hat
  simp only [Bool.and_eq_true] at hat
  obtain ⟨_, _, _, w, S, -, -, -, hw, hS, hc⟩ := readsOk_true h hat.2
  have hwi := ins_of_setupOf hw hset
  obtain ⟨-, -, hSr, hSn⟩ := stmtJsonOf_fields h hS
  obtain ⟨-, -, -, -, -, -, -, hlinked, hwired, -, -⟩ := Spot.SecondStage.stmtCodes_nil (List.isEmpty_iff.1 hc)
  have hany : (Spot.entries (Spot.getD S.registered "ports")).any (·.1 == port) = true := by
    rcases hport with ⟨wr, hwr, hwt, rfl⟩ | ⟨l, hl, hlt, rfl⟩
    · unfold Spot.SecondStage.wiredRead at hwired
      have h1 : (wr.template != h.names.getD t "" ||
          (Spot.entries (Spot.getD S.registered "ports")).any (·.1 == wr.port)) = true :=
        Array.all_eq_true_iff_forall_mem.1 hwired wr hwr
      rcases Bool.or_eq_true_iff.1 h1 with hne | hany
      · exact absurd hwt (bne_iff_ne.1 hne)
      · exact hany
    · unfold Spot.SecondStage.linkedRead at hlinked
      have h1 : (l.template != h.names.getD t "" ||
          (Spot.entries (Spot.getD S.registered "ports")).any (·.1 == l.port)) = true :=
        Array.all_eq_true_iff_forall_mem.1 hlinked l hl
      rcases Bool.or_eq_true_iff.1 h1 with hne | hany
      · exact absurd hlt (bne_iff_ne.1 hne)
      · exact hany
  obtain ⟨⟨nm, e⟩, hme, hnm⟩ := Array.any_eq_true'.1 hany
  obtain rfl : nm = port := by simpa using hnm
  cases hreg : ins.registered with
  | none =>
    rw [hSn (registered_none hwi hreg), show Spot.entries (Spot.getD Lean.Json.null "ports") = #[] from rfl] at hme
    exact absurd hme (Array.not_mem_empty _)
  | some own =>
    obtain ⟨bR, jR, hbR, hjR, hoj⟩ := registered_of_ins hwi hreg
    have hSj : S.registered = jR := by
      have h4 := hSr bR hbR
      rw [hjR] at h4
      exact (Option.some.inj h4).symm
    obtain ⟨kvs, hports, -, hback⟩ := portsOf_entries (portsOf_of_ofJson hoj)
    rw [hSj, getD_of_ok hports] at hme
    obtain ⟨reg, hmem, hrn⟩ := hback _ hme
    refine ⟨own, reg, rfl, hmem, hrn, fun sc hsc => ?_⟩
    obtain ⟨_, _, -, -, hregs⟩ := opens_of_setup hset hreg sc hsc
    obtain ⟨p, _, _, hp, -⟩ := hregs reg hmem
    rw [hrn] at hp
    exact ⟨p, hp⟩

end SpotFlock
