import Definitions.NetworkCertifier.SampledCharge

/-!
# The registered inputs: the registration's values, and the finder that a disagreement with them gives

Definitions only. `RAcc` pins the circuit's inputs to the trace's committed values, so the leak bound's count at the
inputs pinned at the registered inputs (`AsReg`) is at whatever values the trace holds there. The registration commits
those values under the statement's roots: openings `ops` (`RegOpens`), its values `regVal ops`. An extracted trace that
opens a registered input to another value makes, with the registration's opening there, two passing openings of one
position to different values, so under `Commitment.Binds` the calls that check them collide. `Breg` is the finder that
looks for that collision: the extractor run on the judged run, and the first collision among its own calls and those
that check the registration's openings.
-/

namespace NetTiming

open Claims Claims.Sampling

/-- **The registration's value** at gate `g`: its first opening at `g`'s position, `false` where it has none. -/
def regVal (ops : List (Opening Bool)) (g : ℕ) : Bool :=
  ((ops.find? fun o => o.pos == g).map (·.val)).getD false

/-- **The registration opens the registered inputs** `reg` under the statement's roots: each of its openings `ops`
passes, and every gate of `reg` has one. -/
def RegOpens (H : Msg → Msg) (Cm : Commitment Bool) (s : Stmt) (reg : Finset ℕ) (ops : List (Opening Bool)) : Prop :=
  (∀ o ∈ ops, Cm.Opens H s.roots o = true) ∧ ∀ g ∈ reg, ∃ o ∈ ops, o.pos = g

/-- The `H` calls that check the openings `ops` under `roots`. -/
def regLog (H : Msg → Msg) (Cm : Commitment Bool) (roots : List Msg) (ops : List (Opening Bool)) : List (Msg × Msg) :=
  ops.flatMap fun o => ((Cm.opens roots o).eval H).2

section Finder

variable {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **`B_reg`**, the registration's collision finder: on the judged run's tape and coins and the extractor's coins, the
extractor run, and the first collision among its own `H` calls and the calls that check the registration's openings
`ops s` under the statement's roots. Its cost is the extraction's SHA-512 calls, its reruns' included, and the checks'. -/
noncomputable def Breg (H : Msg → Msg) (V : Verifier Coin) (E : Extractor Coin W V) (Cm : Commitment Bool)
    (parse : Msg → Option Stmt) (ops : Stmt → List (Opening Bool)) : FinderFamily :=
  fun Ω _ _ P x =>
    let rl := ((parse x).map fun s => regLog H Cm s.roots (ops s)).getD []
    { S := Ω × (Fin (V.k x) → Coin) × E.C x
      out := fun s => findColl (ownLog H (run H P V x s.1) (E.ext x s.2.1 s.2.2) ++ rl)
      cost := fun s => (E.eval H P x s.1 s.2.1 s.2.2).2.1 + rl.length }

end Finder

end NetTiming
