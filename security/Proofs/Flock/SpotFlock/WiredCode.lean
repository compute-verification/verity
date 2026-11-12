module

public import Proofs.Flock.SpotFlock.RevealsWired
public import Proofs.Flock.SpotFlock.WireMapRewires
public import Proofs.Flock.SpotFlock.KS

/-!
# SPOT[Flock]'s two-stage code at the connected program

`code h` (`Code`) reads the members' programs side by side (`bitsProgs h.ms`), where no gate of one proof block's
circuit reads a gate of another. This file defines the same code at the members' programs wired by a map `ω`
(`Prog.wired`): the statement's program has every gate `ω` moves replaced by the gate it is sent to, each proof block
and replay block is without the gates `ω` moves, and a session reveals each opening at the gate `ω` sends it to
(`partitionWired`, `twoWired`, `revealWired`, `RevealsWired`). The verifier, the sessions, their rounds, the samples
and the frame are `code h`'s.

* `parseWired`, `baseWired`, `codeWired`, `rsWired`: `parse h`, `base h`, `code h` and `rs h` at the wired program.
* `wiredOk h`: the code's own wiring is one the proof covers, decided on what the verifier holds.
* `codeConnected h`, `twoConnected h`, `rsConnected h`: the code, the design and the replay sample at the code's own
  wiring map (`wireMap h`, `returned h`).
* `rewires_of_wiredOk`: where `wiredOk h` holds, the code's wiring map rewires the members' programs
  (`Prog.Rewires`).

The verifier is `V h` itself, and its check of the session statements (`stmtOk`) reads no wire: no run is refused
for its wiring here. So `wiredOk h` is a hypothesis of each statement about the connected code, and such a statement
says nothing at a registration that fails it.
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

/-- **The code's own wiring is one the proof covers**, decided on what the verifier holds. Five conditions:

1. every template's session statement is in Flock's scope (`templateOk`, which `two h` asks too);
2. every member fits the two-stage scope (`memberOk`);
3. each wire's writer's part is before its reader's (`writersFirst`);
4. no wire's writer's port is one a link names (`noReturnedWired`);
5. every source bit at a port a wire names has a gate to go to (`wiresMove`).

The proof uses the first three: they are what `Prog.Rewires` needs of the map (`rewires_of_wiredOk`). The last two
enter no proof. They are what make the connected program mean what it reads: under them every wired read is sent to
its writer's output, so none stays an input of the connected program, which a bound on what the program's inputs
determine takes as given; and no wire reads an output the replay block returns, which the boundary record holds and
the interior record's port of it is not held to. -/
def wiredOk : Bool :=
  (List.range h.ms.length).all (templateOk h) && h.ms.all memberOk && writersFirst h && noReturnedWired h &&
    wiresMove h

/-- **SPOT[Flock]'s two-stage code at the connected program**: `codeWired` at the code's own wiring map and the
gates the boundary record holds. Its verifier is `V h`. -/
noncomputable abbrev codeConnected : Code := codeWired h (wireMap h) (returned h)

/-- **The two-stage design at the connected program**: `two h`'s with each replay block without the gates the wiring
moves. -/
def twoConnected : AuditStatement → Option Design.Two := twoWired h (wireMap h)

/-- **The replay sample of record at the connected program.** -/
noncomputable def rsConnected : (x : Msg) → (Fin ((codeConnected h).V.k x) → FlockDeploy.Coin) → List ℕ :=
  rsWired h (wireMap h) (returned h)

/-- **Where the code's wiring is one the proof covers, its wiring map rewires the members' programs**
(`rewires_wireMap`): each member fits, its unit built by its template's check. Of `wiredOk h` it reads the first
three conditions. -/
theorem rewires_of_wiredOk (hok : wiredOk h = true) :
    (bitsProgs h.ms).Rewires (membersPart h.ms) (wireMap h) (returned h) := by
  unfold wiredOk at hok
  simp only [Bool.and_eq_true] at hok
  obtain ⟨⟨⟨⟨ht, hm⟩, hfirst⟩, -⟩, -⟩ := hok
  refine rewires_wireMap h (fun ci hci => ?_) hfirst
  obtain ⟨t, hlt, rfl⟩ := List.getElem_of_mem hci
  have htt : templateOk h t = true := List.all_eq_true.1 ht t (List.mem_range.2 hlt)
  exact memberFits_of_memberOk (templateOk_facts h htt hlt).2.1 (List.all_eq_true.1 hm _ hci)

end SpotFlock
