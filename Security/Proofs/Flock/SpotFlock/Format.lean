module

public import Proofs.Flock.Bridge.Members
public import Proofs.Flock.SpotFlock.Opens

/-!
# Where the two-stage session holds each checked unit's values

`decodes_of_embeds`: `Decodes` (`SpotFlock.Opens`) at the decoder a session layout gives, from the session statement
holding each checked unit's circuit gates at the layout's wires (`Embeds`), through Bridge's lowering of the unit
(`Bridge.correct_unitGates_of`, `Bridge.correct_blockOf`) and its place in the statement's program
(`Bridge.Members.SitsAt.correct`).

**The layout** (`Layout`). The session statement's block circuit `C` (`Claims.Flock.Stmt`) runs once per Flock block.
A layout names the block that holds each checked unit (`blockOf`), and for each program gate, the gate of `C` that holds
its value for that unit (`wire`). Both read the session statement, whose sample places the units: a unit's block and
its template instance in the block follow from its rank in the sample.

**The decoder** (`decOf`). Unit `u`'s value at program gate `g` is `C`'s value at its wire in `u`'s block, on the
inputs Flock's relation gives that block at the witness's hidden values (`valueAt`, as `Claims.Flock.RelS` evaluates
`C`). It is per unit, as `Dec` is: two units that read one gate read it from their own blocks' inputs.

**What the statement must hold** (`Embeds`). Each checked unit's proof block is a unit `v` of a circuit `C'` as
Bridge lowers it at constants `κ` (`Bridge.blockOf`), and that lowering sits in the statement's program from some
position `o` (`SitsAt`). At every hidden-value list Flock's relation accepts, `C`'s values at the wires the layout gives
the positions of `C'`'s wires (`unitVals`) hold each of `v`'s gates (`Op.apply`) and carry `κ` wherever `v` reads
it. These are `C'`'s gates, not the lowered program's: no value is asked at a row's two XORs, which no wire of `C`
holds and `Prog.Correct` never reads. It holds of every accepted list, not of one witness, whatever the run's public
values: whether it holds turns only on the unit's session statement's circuit, which the verifier's check fixes to its
template's reference statement's (`xS₀ t`, `pin_of_stmtOkAt`), and the layout (`SpotFlock.Placement`). So it is a
property of the registration. The relation, not `C`'s
evaluation alone, has to carry it: where a wire is an input of `C`, only the outputs Flock's relation checks tie its
value to the others.

**Where the reveal's values sit** (`decOf_input`, `decOf_output`). At a wire that is an input of `C`, a decoded value
is the input's value in the unit's block (`S.inp β j`), and at an output, under a witness the check accepts, the
output's (`S.out β j`): a hidden value where that source is hidden. `Reveals` at `decOf L` is then the commitment's
side: for each value the unit needs, the reveal opens that hidden value, under the roots the opening's position reads.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

/-- **A session layout**: the Flock block holding each checked unit (`blockOf x m₁ xS u`, at the run's statement `x`,
the interior registration `m₁` and the session statement `xS`), and the gate of the session's block circuit holding
each program gate's value for that unit there (`wire x m₁ xS u g`). -/
structure Layout where
  blockOf : Msg → Msg → Msg → ℕ → Option ℕ
  wire : Msg → Msg → Msg → ℕ → ℕ → ℕ

/-- **The block circuit's value at gate `j` of block `β`**, on the inputs Flock's relation gives that block at the
hidden values `w` (`Claims.Flock.RelS`); `false` past the circuit's gates. -/
def valueAt (S : Claims.Flock.Stmt) (β : ℕ) (w : Claims.Flock.Wit) (j : ℕ) : Bool :=
  if hj : j < S.C.N then
    (S.C.toCircuit.eval (fun g => BitVec.ofBool (Claims.Flock.val S w (S.inp β g.val))) ⟨j, hj⟩).getLsbD 0
  else false

/-- **The decoder at a layout**: unit `u`'s value at program gate `g` is the block circuit's value at its wire in
`u`'s block, on the session witness's hidden values. -/
def decOf (L : Layout) : Dec := fun x m₁ xS wF u g =>
  match FlockDeploy.parse true _root_.Flock.Readable.relation xS, L.blockOf x m₁ xS u with
  | some S, some β => valueAt S β wF (L.wire x m₁ xS u g)
  | _, _ => false

theorem decOf_eq {L : Layout} {x m₁ xS : Msg} {S : Claims.Flock.Stmt} {u β : ℕ}
    (hS : FlockDeploy.parse true _root_.Flock.Readable.relation xS = some S) (hβ : L.blockOf x m₁ xS u = some β)
    (w : Claims.Flock.Wit) (g : ℕ) : decOf L x m₁ xS w u g = valueAt S β w (L.wire x m₁ xS u g) := by
  simp only [decOf, hS, hβ]

/-- **At an input gate** the block circuit's value is the input Flock's relation gives it. -/
theorem valueAt_input {S : Claims.Flock.Stmt} {β j : ℕ} (w : Claims.Flock.Wit) (hj : j < S.C.N)
    (hin : (S.C.op ⟨j, hj⟩).isInput = true) : valueAt S β w j = Claims.Flock.val S w (S.inp β j) := by
  have he := Verity.Circuit.eval_input S.C.toCircuit
    (fun g => BitVec.ofBool (Claims.Flock.val S w (S.inp β g.val))) (g := ⟨j, hj⟩) hin
  simp only [valueAt, hj, ↓reduceDIte]
  exact (congrArg (fun v => BitVec.getLsbD v 0) he).trans
    ((fun b => by cases b <;> rfl : ∀ b : Bool, (BitVec.ofBool b).getLsbD 0 = b) _)

/-- **At an output gate** of a block, under a witness Flock's relation accepts, the block circuit's value is the
output's value. -/
theorem valueAt_output {S : Claims.Flock.Stmt} {β j : ℕ} {w : Claims.Flock.Wit} (hR : Claims.Flock.RelS S w)
    (hβ : β < S.nBlocks) (hj : j < S.C.N) (ho : (⟨j, hj⟩ : Fin S.C.N) ∈ S.C.outputs) :
    valueAt S β w j = Claims.Flock.val S w (S.out β j) := by
  simp only [valueAt, hj, ↓reduceDIte]
  exact hR β hβ ⟨j, hj⟩ ho

/-- **A decoded value at an input wire** is the input's value: the hidden or public value the session statement
gives that input in the unit's block. -/
theorem decOf_input {L : Layout} {x m₁ xS : Msg} {S : Claims.Flock.Stmt} {u β g : ℕ}
    (hS : FlockDeploy.parse true _root_.Flock.Readable.relation xS = some S) (hβ : L.blockOf x m₁ xS u = some β)
    (w : Claims.Flock.Wit) (hj : L.wire x m₁ xS u g < S.C.N)
    (hin : (S.C.op ⟨L.wire x m₁ xS u g, hj⟩).isInput = true) :
    decOf L x m₁ xS w u g = Claims.Flock.val S w (S.inp β (L.wire x m₁ xS u g)) := by
  rw [decOf_eq hS hβ, valueAt_input w hj hin]

/-- **A decoded value at an output wire**, under a session witness the verifier's check accepts, is the output's value:
the hidden or public value the session statement gives that output in the unit's block. -/
theorem decOf_output {L : Layout} {x m₁ xS : Msg} {S : Claims.Flock.Stmt} {u β g : ℕ} {w : Claims.Flock.Wit}
    (hS : FlockDeploy.parse true _root_.Flock.Readable.relation xS = some S) (hβ : L.blockOf x m₁ xS u = some β)
    (hck : Claims.Flock.check (FlockDeploy.parse true _root_.Flock.Readable.relation) xS w = true)
    (hβS : β < S.nBlocks) (hj : L.wire x m₁ xS u g < S.C.N)
    (ho : (⟨L.wire x m₁ xS u g, hj⟩ : Fin S.C.N) ∈ S.C.outputs) :
    decOf L x m₁ xS w u g = Claims.Flock.val S w (S.out β (L.wire x m₁ xS u g)) := by
  have hR : Claims.Flock.RelS S w := by simpa [Claims.Flock.check, hS] using hck
  rw [decOf_eq hS hβ, valueAt_output hR hβS hj ho]

/-- **A checked unit's values at a layout**: at circuit `C'`'s wire `a`, the block circuit's value in block `β` at the
hidden values `w`, at the wire the layout gives unit `u`'s program position of `a` at the session statement `xS`,
`C'`'s lowering sitting from `o`. -/
def unitVals (L : Layout) (x m₁ xS : Msg) (S : Claims.Flock.Stmt) (β : ℕ) (w : Claims.Flock.Wit) (u o : ℕ)
    (C' : Verity.Boolean.Circuit) (a : Fin C'.N) : Bool :=
  valueAt S β w (L.wire x m₁ xS u (o + _root_.Flock.Bridge.posOf C' a))

variable (h : Held)

/-- **The session statements hold each checked unit's circuit gates at a layout's wires**: at a run's statement in the
two-stage scope and session statements `m₂` the verifier's check passes, each checked unit's proof block is unit `v` of
a circuit `C'` as Bridge lowers it at constants `κ`, from a position `o` of the statement's program where that lowering
sits; the unit has its block in its own session statement (`unitStmt`), and at every hidden-value list Flock's relation
accepts, the block circuit's values at the unit's wires (`unitVals`) hold each of `v`'s gates and carry `κ` wherever
`v` reads it. -/
def Embeds (L : Layout) : Prop :=
  ∀ x s Pt d (m₁ m₂ : Msg) (c₀ c₁ : FlockDeploy.Coin) u b ρ rb S,
    parse h x = some s → partition h s = some Pt → two h s = some d → InScopeTwo Pt d s →
    stmtOk h x m₁ m₂ c₀ c₁ = true → u ∈ h.reg.checked (units (base h) x) (src c₀) (src c₁) →
    FlockDeploy.parse true _root_.Flock.Readable.relation (unitStmt h m₂ u) = some S →
    Pt.blocks[u]? = some b → replayOf Pt d.part u = some ρ → d.part.blocks[ρ]? = some rb →
    ∃ (C' : Verity.Boolean.Circuit) (κ : Fin C'.N → Option Bool) (n : ℕ)
      (P : FlockSoundness.Audit.Partition C' n) (v : Fin n) (o : ℕ),
      _root_.Flock.Bridge.Members.SitsAt s.prog o (_root_.Flock.Bridge.progOf C' κ) ∧
      b = (_root_.Flock.Bridge.blockOf P κ v).map (o + ·) ∧
      ∃ β, L.blockOf x m₁ (unitStmt h m₂ u) u = some β ∧ ∀ w : Claims.Flock.Wit, Claims.Flock.RelS S w →
        (∀ a, P.unit a = some v → unitVals L x m₁ (unitStmt h m₂ u) S β w u o C' a =
          (C'.op a).apply (unitVals L x m₁ (unitStmt h m₂ u) S β w u o C')) ∧
        ∀ a c, κ a = some c → (∃ g, P.unit g = some v ∧ a ∈ (C'.op g).args) →
          unitVals L x m₁ (unitStmt h m₂ u) S β w u o C' a = c

/-- **`Decodes` at a layout's decoder** where the session statement holds each checked unit's circuit gates there: the
unit's lowered gates are correct on the decoded values (`Bridge.correct_unitGates_of`), so is its proof block
(`Bridge.correct_blockOf`), and so is that block where it sits in the statement's program (`SitsAt.correct`). -/
theorem decodes_of_embeds (L : Layout) (he : Embeds h L) : Decodes h (decOf L) := by
  intro x s Pt d m₁ m₂ c₀ c₁ wF u b ρ rb hx hP h2 hin hok hu hck hb hρ hrb
  cases hS : FlockDeploy.parse true _root_.Flock.Readable.relation (unitStmt h m₂ u) with
  | none => simp [Claims.Flock.check, hS] at hck
  | some S =>
    obtain ⟨C', κ, n, P, v, o, hsit, rfl, β, hβ, hw⟩ :=
      he x s Pt d m₁ m₂ c₀ c₁ u b ρ rb S hx hP h2 hin hok hu hS hb hρ hrb
    obtain ⟨hV, hκ⟩ := hw wF (by simpa [Claims.Flock.check, hS] using hck)
    have href : s.prog.refIn rb (fun g => some (decOf L x m₁ (unitStmt h m₂ u) wF u g))
        (decOf L x m₁ (unitStmt h m₂ u) wF u) = fun g => some (decOf L x m₁ (unitStmt h m₂ u) wF u g) :=
      funext fun g => by simp only [Prog.refIn, ite_self]
    rw [href]
    refine hsit.correct _root_.Flock.Bridge.Members.topological_progOf (fun p hp => ?_) ?_
    · rcases _root_.Flock.Bridge.mem_blockOf.1 hp with hp | ⟨a, -, rfl⟩
      · exact _root_.Flock.Bridge.Members.lt_length_of_mem_unitGates hp
      · exact _root_.Flock.Bridge.Members.posOf_lt_length a
    · refine (_root_.Flock.Bridge.correct_blockOf P v _).2
        ⟨_root_.Flock.Bridge.correct_unitGates_of P v _ hV hκ _ fun a _ => ?_, fun a _ => by simp⟩
      exact congrArg some (decOf_eq hS hβ wF _)

/-- **`SessionOpensTwo` at SPOT[Flock]'s two-stage code from a layout** where the session statement holds each
checked unit's circuit gates (`Embeds`) and whose values the reveal opens (`Reveals`). -/
theorem sessionOpensTwo_of_embeds (L : Layout) (he : Embeds h L) (hr : Reveals h (decOf L)) :
    Claims.SamplingGeneric.SessionOpensTwo FlockDeploy.sha512 (deploy _root_.Flock.Readable.relation (code h))
      (two h) iroots :=
  sessionOpensTwo_of_dec h (decOf L) (decodes_of_embeds h L he) hr

end SpotFlock
