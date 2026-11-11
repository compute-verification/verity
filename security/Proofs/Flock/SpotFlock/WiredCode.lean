module

public import Proofs.Flock.SpotFlock.WireMapRewires
public import Proofs.Flock.SpotFlock.KS

/-!
# SPOT[Flock]'s two-stage code at the connected program

`SpotFlock.code h` states its guarantee over the members' programs side by side (`bitsProgs`), where no gate of one
proof block reads another's. This file is the same code over the connected program (`wiredProg`): the members'
programs with every wired read sent to its writer's output (`wireMap`), returning the gates the boundary record holds
(`returned`).

* `wiredOk`: the wiring is one the proof covers, decided on what the verifier holds: every member fits (`memberOk`),
  each wire's writer's part is before its reader's (`writersFirst`), no returned output is wired inside
  (`noReturnedWired`) and every wired source bit has a gate to go to (`wiresMove`).
* `gated`: a verifier behind a check of what it holds, which refuses every run where the check fails. An input
  condition is the verifier's refusal and no hypothesis of the guarantee.
* `parseWired`, `partitionWired`, `twoWired`: the statement at the connected program, the members' partition with
  each proof block's moved gates removed (`Partition.cut`), and the two-stage design with its replay partition cut
  the same way, where `two h` evaluates and the wiring is covered.
* `revealWired`: what a session reveals at the connected program, one opening for each gate the unit's proof block
  needs side by side, at the gate the wiring sends it to and under that gate's roots there. The count of openings a
  unit is the side-by-side count.
* `baseWired`, `codeWired`, `rsWired`: `base h`, `code h` and `rs h` with those four and the gated verifier; the
  sessions, their rounds, the samples and the frame are `code h`'s.

The verifier's runs are `V h`'s wherever the wiring is covered (`gated`), so every fact of an accepted run of `V h`
is one of the connected code's.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy Properties.Spot Finset

variable (h : Held)

/-- **The wiring is one the proof covers**, decided on what the verifier holds: every member fits the two-stage scope
(`memberOk`), each wire's writer's part is before its reader's (`writersFirst`), no wire's writer's port is one a link
names (`noReturnedWired`), and every source bit at a port a wire names has a gate to go to (`wiresMove`). -/
def wiredOk : Bool :=
  h.ms.all memberOk && writersFirst h && noReturnedWired h && wiresMove h

/-- **A verifier behind a check of what it holds**: `V` where the check passes, and a verifier of `V`'s rounds that
refuses every run where it fails. -/
def gated (ok : Bool) (V : Verifier FlockDeploy.Coin) : Verifier FlockDeploy.Coin where
  k := V.k
  next := fun x as cs => if ok then V.next x as cs else .ret .rej

/-- **The statement at the connected program**: `parse h`'s, its program the members' read through the wiring map
(`wiredProg`). -/
def parseWired (x : Msg) : Option AuditStatement :=
  (parse h x).map fun s => { s with prog := wiredProg h }

/-- **The proof-block partition at the connected program**: the members' units, each without the gates the wiring
moves (`Partition.cut`). -/
def partitionWired (_ : AuditStatement) : Option Partition :=
  some ((membersPart h.ms).cut (wireMap h))

/-- **The two-stage design at the connected program**: `two h`'s with its replay partition cut the same way, where
`two h` evaluates and the wiring is one the proof covers (`wiredOk`). -/
def twoWired (s : AuditStatement) : Option Design.Two :=
  if wiredOk h then (two h s).map fun d => { d with part := d.part.cut (wireMap h) } else none

/-- **What a session's witness reveals for unit `u` at the connected program**: one opening for each gate its proof
block needs side by side (`Reveal.needs` at the members' programs), at the gate the wiring sends it to, under that
gate's roots at the connected program (`Reveal.rootsAt` at `partitionWired` and `twoWired`), holding the frame's
opening for the side-by-side gate under those roots. Two of a unit's gates sent to one gate are two openings at one
position under one roots. -/
def revealWired (s : AuditStatement) (u : ℕ) (v : Seen) (wF : Claims.Flock.Wit) : List (List Msg × Opening Bool) :=
  (Reveal.needs (partition h) { s with prog := bitsProgs h.ms } u).map fun i =>
    (Reveal.rootsAt (partitionWired h) (twoWired h) (iroots []) s u v (wireMap h i),
      { Reveal.frameAt h.frame { s with prog := bitsProgs h.ms } u v wF i
          (Reveal.rootsAt (partitionWired h) (twoWired h) (iroots []) s u v (wireMap h i)) with
        pos := wireMap h i })

/-- **The code at the connected program, before its samples**: `base h` with the gated verifier, the statement, the
partition and the reveal at the connected program. The reveal count is the side-by-side one. -/
noncomputable def baseWired : Code :=
  { h.frame with
    V := gated (wiredOk h) (V h)
    parse := parseWired h
    scopeCheck := fun x => ((parseWired h x).bind (partitionWired h)).isSome
    partition := partitionWired h
    stage2 := fun _ => univ
    checked := fun _ _ => []
    rounds := fun _ u => univ.filter fun t =>
      off h (templateOf h.ms u) ≤ t.val ∧ t.val < off h (templateOf h.ms u) + k₀ h (templateOf h.ms u)
    session := fun _ u v => session h u v
    reveal := revealWired h
    reveals := Reveal.reveals (partition h) (parse h)
    sessions := fun _ => h.ms.length
    tables := fun _ => ((List.range (h.ms.length + 1)).map fun t => (shapeOf (h.xS₀ t)).length).foldr max 0
    unitN₀ := fun _ u => n₀ h (templateOf h.ms u)
    innerN₀ := fun _ => n₀ h 0 }

/-- The run's round order at the connected code: `order h`, the gated verifier having `V h`'s rounds. -/
def orderWired (x : Msg) : Order ((baseWired h).V.k x) := order h x

/-- **SPOT[Flock]'s two-stage code at the connected program**: `baseWired h` at its round order, its checked units
the two stages' on the auditor's registration, each stage's bytes its own coin's. -/
noncomputable abbrev codeWired : Code :=
  codeTwoAt (baseWired h) (reg h) (fun _ => L) (fun _ => L) (orderWired h) (fun _ => coinByte) (fun _ => coinByte)

/-- **The replay sample of record at the connected code**: the first stage's checked slots on the replay layout. -/
noncomputable def rsWired : (x : Msg) → (Fin ((codeWired h).V.k x) → FlockDeploy.Coin) → List ℕ :=
  replayTwo ((baseWired h).atOrder (orderWired h)) (reg h) (fun _ => L)
    (bytesFirst (baseWired h) (fun _ => L) (orderWired h) fun _ => coinByte)

/-- Where the wiring is covered, the gated verifier is `V h`. -/
theorem gated_true (V : Verifier FlockDeploy.Coin) : gated true V = V := by
  cases V
  simp [gated]

/-- Where the wiring is not covered, the design does not evaluate: no statement is in the connected code's scope. -/
theorem twoWired_none (hw : wiredOk h = false) (s : AuditStatement) : twoWired h s = none := by
  unfold twoWired
  simp [hw]

/-- Where the wiring is covered, the design is `two h`'s with its replay partition cut. -/
theorem twoWired_eq (hw : wiredOk h = true) (s : AuditStatement) :
    twoWired h s = (two h s).map fun d => { d with part := d.part.cut (wireMap h) } := by
  unfold twoWired
  simp [hw]

end SpotFlock
