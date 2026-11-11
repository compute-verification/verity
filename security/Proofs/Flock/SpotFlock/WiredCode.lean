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
(`partitionWired`, `twoWired`, `revealWired`, `RevealsWired`). The sessions, their rounds, the samples and the frame
are `code h`'s.

* `gated ok V`: the verifier `V` behind a check's verdict: `V` where `ok` holds, and a verifier that refuses every
  run where it doesn't. `twoGated h ω ok`: the design at the wired program where `ok` holds, none where it doesn't,
  so no statement is in the code's scope there.
* `parseWired`, `baseWired`, `codeWired`, `rsWired`: `parse h`, `base h`, `code h` and `rs h` at the wired program,
  behind a verdict `ok`.
* `wiredOk h`: the check of the code's own wiring, decided on what the verifier holds. `codeConnected h`,
  `twoConnected h`, `rsConnected h`: the code, the design and the replay sample at the code's own wiring map
  (`wireMap h`, `returned h`), behind `wiredOk h`.
* `rewires_of_wiredOk`: where `wiredOk h` holds, the code's wiring map rewires the members' programs
  (`Prog.Rewires`).

The verifier's own check of the session statements (`stmtOk`) reads no wire, so every condition on the wiring is a
condition on what the verifier holds; `gated` makes it a refusal, where a hypothesis on `h` would leave the guarantee
silent at a registration that fails it.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy Properties.Spot Finset

variable (h : Held) (ω : ℕ → ℕ) (keep : ℕ → Bool)

/-- **A verifier behind a check's verdict**: `V` where the check passes, and a verifier of `V`'s rounds that refuses
every run where it fails. -/
def gated (ok : Bool) (V : Verifier FlockDeploy.Coin) : Verifier FlockDeploy.Coin where
  k := V.k
  next := fun x as cs => bif ok then V.next x as cs else .ret .rej

/-- **The statement at the wired program**: `parse h`'s, its program the members' wired by `ω`. -/
def parseWired (x : Msg) : Option AuditStatement :=
  (parse h x).map fun s => { s with prog := (bitsProgs h.ms).wired ω keep }

/-- **The two-stage design at the wired program, behind a check's verdict**: `twoWired h ω`'s where the check passes,
none where it fails. -/
def twoGated (ok : Bool) (s : AuditStatement) : Option Design.Two := bif ok then twoWired h ω s else none

/-- **The code at the wired program, before its samples, behind a check's verdict**: `base h` with the gated
verifier and the statement, the partition and the reveal at the wired program. The reveal count is the side-by-side
one (`length_revealWired_le`). -/
noncomputable def baseWired (ok : Bool) : Code :=
  { h.frame with
    V := gated ok (V h)
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

/-- The run's round order at the wired code: `order h`, the gated verifier having `V h`'s rounds. -/
def orderWired (ok : Bool) (x : Msg) : Order ((baseWired h ω keep ok).V.k x) := order h x

/-- **The two-stage code at the wired program, behind a check's verdict**: `baseWired` at its round order, its
checked units the two stages' on the auditor's registration, each stage's bytes its own coin's (`code h`'s shape). -/
noncomputable abbrev codeWired (ok : Bool) : Code :=
  codeTwoAt (baseWired h ω keep ok) (reg h) (fun _ => L) (fun _ => L) (orderWired h ω keep ok) (fun _ => coinByte)
    (fun _ => coinByte)

/-- **The replay sample of record at the wired code**: the first stage's checked slots on the replay layout
(`rs h`'s shape). -/
noncomputable def rsWired (ok : Bool) :
    (x : Msg) → (Fin ((codeWired h ω keep ok).V.k x) → FlockDeploy.Coin) → List ℕ :=
  replayTwo ((baseWired h ω keep ok).atOrder (orderWired h ω keep ok)) (reg h) (fun _ => L)
    (bytesFirst (baseWired h ω keep ok) (fun _ => L) (orderWired h ω keep ok) fun _ => coinByte)

/-- Where the check passes, the gated verifier is `V`. -/
theorem gated_true (V : Verifier FlockDeploy.Coin) : gated true V = V := rfl

/-- Where the check fails, the design does not evaluate: no statement is in the wired code's scope. -/
theorem twoGated_false (s : AuditStatement) : twoGated h ω false s = none := rfl

/-- Where the check passes, the design is `twoWired h ω`'s. -/
theorem twoGated_true (s : AuditStatement) : twoGated h ω true s = twoWired h ω s := rfl

/-! ## The code's own wiring -/

/-- **The code's own wiring is one the proof covers**, decided on what the verifier holds: every template's session
statement is in Flock's scope (`templateOk`, which `two h` asks too), every member fits the two-stage scope
(`memberOk`), each wire's writer's part is before its reader's (`writersFirst`), no wire's writer's port is one a link
names (`noReturnedWired`), and every source bit at a port a wire names has a gate to go to (`wiresMove`). The first
three are what `Prog.Rewires` needs of the map; the last two say that every wired read is sent to its writer's
output, so no wired read is an input of the connected program. -/
def wiredOk : Bool :=
  (List.range h.ms.length).all (templateOk h) && h.ms.all memberOk && writersFirst h && noReturnedWired h &&
    wiresMove h

/-- **SPOT[Flock]'s two-stage code at the connected program**: `codeWired` at the code's own wiring map and the
gates the boundary record holds, behind the check of that wiring. Its verifier refuses every run where the check
fails, and is `V h` where it passes. -/
noncomputable abbrev codeConnected : Code := codeWired h (wireMap h) (returned h) (wiredOk h)

/-- **The two-stage design at the connected program**: `two h`'s with each replay block without the gates the wiring
moves, where `two h` evaluates and the check of the wiring passes. -/
def twoConnected : AuditStatement → Option Design.Two := twoGated h (wireMap h) (wiredOk h)

/-- **The replay sample of record at the connected program.** -/
noncomputable def rsConnected : (x : Msg) → (Fin ((codeConnected h).V.k x) → FlockDeploy.Coin) → List ℕ :=
  rsWired h (wireMap h) (returned h) (wiredOk h)

/-- **Where the check of the wiring passes, the code's wiring map rewires the members' programs**
(`rewires_wireMap`): each member fits, its unit built by its template's check. -/
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
