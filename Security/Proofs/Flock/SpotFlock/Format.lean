module

public import Proofs.Flock.SpotFlock.Opens

/-!
# Where the two-stage session holds each checked unit's values

`evaluates_of_embeds`: `Evaluates` (`SpotFlock.Opens`) at the decoder a session layout gives, from the session circuit
computing each checked unit's gates at the layout's wires (`Embeds`).

**The layout** (`Layout`). The session statement's block circuit `C` (`Claims.Flock.Stmt`) runs once per Flock block.
A layout names the block that holds each checked unit (`blockOf`, read off the session statement, whose sample selects
the units) and, for each program gate, the gate of `C` that holds its value for that unit (`wire`).

**The decoder** (`decOf`). Unit `u`'s value at program gate `g` is `C`'s value at `wire u g` in `u`'s block, on the
inputs Flock's relation gives that block at the witness's hidden values (`valueAt`, as `Claims.Flock.RelS` evaluates
`C`). It is per unit, as `Dec` is: two units that read one gate read it from their own blocks' inputs.

**What the circuit must compute** (`Embeds`). At every hidden-value list, `C`'s values at a checked unit's wires respect
each of the unit's gates but its inputs, and every constant of the program. That reads no witness: only the session
statement's circuit, inputs and public values, which the verifier's check fixes to `xS₀`'s but for the sample and the
registered reads (`pin_of_stmtOk`), and the layout. So it is a property of the registration.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

/-- **A session layout**: the Flock block holding each checked unit (`blockOf x m₁ xS u`, at the run's statement `x`,
the interior registration `m₁` and the session statement `xS`), and the gate of the session's block circuit holding
each program gate's value for that unit (`wire u g`). -/
structure Layout where
  blockOf : Msg → Msg → Msg → ℕ → Option ℕ
  wire : ℕ → ℕ → ℕ

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

variable (h : Held)

/-- **The session circuit computes each checked unit's gates at a layout's wires**: at a run's statement in the
two-stage scope and a session statement the verifier's check passes, each checked unit has its block, and at every
hidden-value list the block circuit's values at the unit's wires respect each of the unit's gates but its inputs, and
every constant of the program. -/
def Embeds (L : Layout) : Prop :=
  ∀ x s Pt d (m₁ xS : Msg) (c₀ c₁ : FlockDeploy.Coin) u b ρ rb S,
    parse h x = some s → partition h s = some Pt → two h s = some d → InScopeTwo Pt d s →
    stmtOk h x m₁ xS c₀ c₁ = true → u ∈ h.reg.checked (units (base h) x) (src c₀) (src c₁) →
    FlockDeploy.parse true _root_.Flock.Readable.relation xS = some S →
    Pt.blocks[u]? = some b → replayOf Pt d.part u = some ρ → d.part.blocks[ρ]? = some rb →
    ∃ β, L.blockOf x m₁ xS u = some β ∧ ∀ w : Claims.Flock.Wit,
      (∀ g ∈ b, ∀ y : BGate, s.prog.gates[g]? = some y → y ≠ .input →
        y.val (fun i => some (valueAt S β w (L.wire u i))) none = some (valueAt S β w (L.wire u g))) ∧
      ∀ g c, s.prog.gates[g]? = some (.const c) → valueAt S β w (L.wire u g) = c

/-- **`Evaluates` at a layout's decoder** where the session circuit computes each checked unit's gates there. -/
theorem evaluates_of_embeds (L : Layout) (he : Embeds h L) : Evaluates h (decOf L) := by
  intro x s Pt d m₁ xS c₀ c₁ wF u b ρ rb hx hP h2 hin hok hu hck hb hρ hrb
  cases hS : FlockDeploy.parse true _root_.Flock.Readable.relation xS with
  | none => simp [Claims.Flock.check, hS] at hck
  | some S =>
    obtain ⟨β, hβ, hw⟩ := he x s Pt d m₁ xS c₀ c₁ u b ρ rb S hx hP h2 hin hok hu hS hb hρ hrb
    have hdec : ∀ g, decOf L x m₁ xS wF u g = valueAt S β wF (L.wire u g) := fun g => by
      simp only [decOf, hS, hβ]
    obtain ⟨hg, hc⟩ := hw wF
    refine ⟨fun g hgb y hy hne => ?_, fun g c hgc => ?_⟩
    · simp only [hdec]
      exact hg g hgb y hy hne
    · rw [hdec]
      exact hc g c hgc

/-- **`SessionOpensTwo` at SPOT[Flock]'s two-stage code from a layout** where the session circuit computes each
checked unit's gates (`Embeds`) and whose values the reveal opens (`Reveals`). -/
theorem sessionOpensTwo_of_embeds (L : Layout) (he : Embeds h L) (hr : Reveals h (decOf L)) :
    Claims.SamplingGeneric.SessionOpensTwo FlockDeploy.sha512 (deploy _root_.Flock.Readable.relation (code h))
      (two h) iroots :=
  sessionOpensTwo_of_eval h (decOf L) (evaluates_of_embeds h L he) hr

end SpotFlock
