module

public import Proofs.Flock.SpotFlock.Opens

/-!
# Where the two-stage session holds each checked unit's values

`evaluates_of_embeds`: `Evaluates` (`SpotFlock.Opens`) at the decoder a session layout gives, from the session
statement holding each checked unit's gates at the layout's wires (`Embeds`).

**The layout** (`Layout`). The session statement's block circuit `C` (`Claims.Flock.Stmt`) runs once per Flock block.
A layout names the block that holds each checked unit (`blockOf`, read off the session statement, whose sample selects
the units), for each program gate, the gate of `C` that holds its value for that unit (`wire`), and for each value the
unit reads from the run's commitment, the session's registered read that holds its row (`rowOf`).

**The decoder** (`decOf`). Unit `u`'s value at program gate `g` is `C`'s value at `wire u g` in `u`'s block, on the
inputs Flock's relation gives that block at the witness's hidden values (`valueAt`, as `Claims.Flock.RelS` evaluates
`C`). It is per unit, as `Dec` is: two units that read one gate read it from their own blocks' inputs.

**What the statement must hold** (`Embeds`). At every hidden-value list Flock's relation accepts, `C`'s values at a
checked unit's wires respect each of the unit's gates but its inputs, and every constant of the program. It holds of
every accepted list, not of one witness: it reads only the session statement's circuit, inputs and public values, which
the verifier's check fixes to `xS₀`'s but for the sample and the registered reads (`pin_of_stmtOk`), and the layout. So
it is a property of the registration. The relation, not `C`'s evaluation alone, has to carry it: where a wire is an
input of `C`, only the outputs Flock's relation checks tie its value to the others.

**Where the reveal's values sit** (`decOf_input`, `decOf_output`). At a wire that is an input of `C`, a decoded value
is the input's value in the unit's block (`S.inp β j`), and at an output, under a witness the check accepts, the
output's (`S.out β j`): a hidden value where that source is hidden. `Reveals` at `decOf L` is then the commitment's
side: for each value the unit needs, the reveal opens that hidden value, under the roots the opening's position reads.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

/-- **A session layout**: the Flock block holding each checked unit (`blockOf x m₁ xS u`, at the run's statement `x`,
the interior registration `m₁` and the session statement `xS`), the gate of the session's block circuit holding each
program gate's value for that unit (`wire u g`), and where the session reads the row of each value the unit reads from
the run's commitment (`rowOf x m₁ xS u g = some (j, i, p)`): table `j`'s template instance `i` in the unit's block, at
its circuit's port `p`, whose registered read is the row's. -/
structure Layout where
  blockOf : Msg → Msg → Msg → ℕ → Option ℕ
  wire : ℕ → ℕ → ℕ
  rowOf : Msg → Msg → Msg → ℕ → ℕ → Option (ℕ × ℕ × ℕ)

/-- **The block circuit's value at gate `j` of block `β`**, on the inputs Flock's relation gives that block at the
hidden values `w` (`Claims.Flock.RelS`); `false` past the circuit's gates. -/
def valueAt (S : Claims.Flock.Stmt) (β : ℕ) (w : Claims.Flock.Wit) (j : ℕ) : Bool :=
  if hj : j < S.C.N then
    (S.C.toCircuit.eval (fun g => BitVec.ofBool (Claims.Flock.val S w (S.inp β g.val))) ⟨j, hj⟩).getLsbD 0
  else false

/-- **The decoder at a layout**: unit `u`'s value at program gate `g` is the block circuit's value at `wire u g` in
`u`'s block, on the session witness's hidden values. -/
def decOf (L : Layout) : Dec := fun x m₁ xS wF u g =>
  match FlockDeploy.parse true _root_.Flock.Readable.relation xS, L.blockOf x m₁ xS u with
  | some S, some β => valueAt S β wF (L.wire u g)
  | _, _ => false

theorem decOf_eq {L : Layout} {x m₁ xS : Msg} {S : Claims.Flock.Stmt} {u β : ℕ}
    (hS : FlockDeploy.parse true _root_.Flock.Readable.relation xS = some S) (hβ : L.blockOf x m₁ xS u = some β)
    (w : Claims.Flock.Wit) (g : ℕ) : decOf L x m₁ xS w u g = valueAt S β w (L.wire u g) := by
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
    (w : Claims.Flock.Wit) (hj : L.wire u g < S.C.N) (hin : (S.C.op ⟨L.wire u g, hj⟩).isInput = true) :
    decOf L x m₁ xS w u g = Claims.Flock.val S w (S.inp β (L.wire u g)) := by
  rw [decOf_eq hS hβ, valueAt_input w hj hin]

/-- **A decoded value at an output wire**, under a session witness the verifier's check accepts, is the output's value:
the hidden or public value the session statement gives that output in the unit's block. -/
theorem decOf_output {L : Layout} {x m₁ xS : Msg} {S : Claims.Flock.Stmt} {u β g : ℕ} {w : Claims.Flock.Wit}
    (hS : FlockDeploy.parse true _root_.Flock.Readable.relation xS = some S) (hβ : L.blockOf x m₁ xS u = some β)
    (hck : Claims.Flock.check (FlockDeploy.parse true _root_.Flock.Readable.relation) xS w = true)
    (hβS : β < S.nBlocks) (hj : L.wire u g < S.C.N) (ho : (⟨L.wire u g, hj⟩ : Fin S.C.N) ∈ S.C.outputs) :
    decOf L x m₁ xS w u g = Claims.Flock.val S w (S.out β (L.wire u g)) := by
  have hR : Claims.Flock.RelS S w := by simpa [Claims.Flock.check, hS] using hck
  rw [decOf_eq hS hβ, valueAt_output hR hβS hj ho]

variable (h : Held)

/-- **The session statement holds each checked unit's gates at a layout's wires**: at a run's statement in the
two-stage scope and a session statement the verifier's check passes, each checked unit has its block, and at every
hidden-value list Flock's relation accepts, the block circuit's values at the unit's wires respect each of the unit's
gates but its inputs, and every constant of the program. -/
def Embeds (L : Layout) : Prop :=
  ∀ x s Pt d (m₁ xS : Msg) (c₀ c₁ : FlockDeploy.Coin) u b ρ rb S,
    parse h x = some s → partition h s = some Pt → two h s = some d → InScopeTwo Pt d s →
    stmtOk h x m₁ xS c₀ c₁ = true → u ∈ h.reg.checked (units (base h) x) (src c₀) (src c₁) →
    FlockDeploy.parse true _root_.Flock.Readable.relation xS = some S →
    Pt.blocks[u]? = some b → replayOf Pt d.part u = some ρ → d.part.blocks[ρ]? = some rb →
    ∃ β, L.blockOf x m₁ xS u = some β ∧ ∀ w : Claims.Flock.Wit, Claims.Flock.RelS S w →
      (∀ g ∈ b, ∀ y : BGate, s.prog.gates[g]? = some y → y ≠ .input →
        y.val (fun i => some (valueAt S β w (L.wire u i))) none = some (valueAt S β w (L.wire u g))) ∧
      ∀ g c, s.prog.gates[g]? = some (.const c) → valueAt S β w (L.wire u g) = c

/-- **`Evaluates` at a layout's decoder** where the session statement holds each checked unit's gates there. -/
theorem evaluates_of_embeds (L : Layout) (he : Embeds h L) : Evaluates h (decOf L) := by
  intro x s Pt d m₁ xS c₀ c₁ wF u b ρ rb hx hP h2 hin hok hu hck hb hρ hrb
  cases hS : FlockDeploy.parse true _root_.Flock.Readable.relation xS with
  | none => simp [Claims.Flock.check, hS] at hck
  | some S =>
    obtain ⟨β, hβ, hw⟩ := he x s Pt d m₁ xS c₀ c₁ u b ρ rb S hx hP h2 hin hok hu hS hb hρ hrb
    obtain ⟨hg, hc⟩ := hw wF (by simpa [Claims.Flock.check, hS] using hck)
    refine ⟨fun g hgb y hy hne => ?_, fun g c hgc => ?_⟩
    · simp only [decOf_eq hS hβ]
      exact hg g hgb y hy hne
    · rw [decOf_eq hS hβ]
      exact hc g c hgc

/-- **`SessionOpensTwo` at SPOT[Flock]'s two-stage code from a layout** where the session statement holds each
checked unit's gates (`Embeds`) and whose values the reveal opens (`Reveals`). -/
theorem sessionOpensTwo_of_embeds (L : Layout) (he : Embeds h L) (hr : Reveals h (decOf L)) :
    Claims.SamplingGeneric.SessionOpensTwo FlockDeploy.sha512 (deploy _root_.Flock.Readable.relation (code h))
      (two h) iroots :=
  sessionOpensTwo_of_eval h (decOf L) (evaluates_of_embeds h L he) hr

end SpotFlock
