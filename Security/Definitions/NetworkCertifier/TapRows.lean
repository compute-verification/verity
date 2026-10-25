import Definitions.NetworkCertifier.SampledCharge

/-!
# The tap's rows: what the tap registers at the transfer gates, and the leak relation at them

Definitions only. The audit's trace holds each step's egress rows at the layout's egress gates and its ingress at the
ingress gates (`Layout`). The tap on the server's uplink (the warden and the certifier, one auditor device) registers what
it records there: openings `ops` under the statement's roots, one at each transfer gate before the horizon, whose values
(`tapVal ops`) are its records' egress rows and the operator's moves on them (`TapOpens`). Its registered root is its
signed root for the window, a function of what the server did, so these rows vary with the server's secret, unlike the
values the prover registers at `reg` (`AsReg`).

An extracted trace that opens a transfer gate to a value other than the tap's makes, with the tap's opening there, two
passing openings of one position to different values, so under `Commitment.Binds` the calls that check them collide;
`Btap` looks for that collision. Every transfer gate outside the faulty units is opened, so where the trace agrees with
the tap it carries the tap's records there; a faulty unit's committed gates are the advice the count already charges
(`2^β` over `faultySets`). The leak relation at the tap's rows (`RLeakTap`) is `RLeak` with the observations it counts
picked out by the tap's registration rather than by the trace's transfer gates.

The transfer gates the tap registers at are gates the trace commits: an egress gate is no constant, which is public and
committed by nobody (an ingress gate is an input, `Layout.Fits`).
-/

namespace NetTiming

open Claims Claims.Sampling

/-- **The tap's value** at gate `g`: its first opening at `g`'s position, `false` where it has none. -/
def tapVal (ops : List (Opening Bool)) (g : ℕ) : Bool :=
  ((ops.find? fun o => o.pos == g).map (·.val)).getD false

/-- **The tap registers the records `o`** at the transfer gates of the statement `s` over the horizon `k`, at the
operator `op`: each of its openings `ops` passes under the statement's roots; before `k`, every egress gate is one the
program commits (no constant) and every egress and ingress gate has an opening; and each step's record exists, its egress
rows are the rows the openings hold at the egress gates, and the operator's move on the records before it is the ingress
they hold at the ingress gates. -/
def TapOpens {L K S : ℕ} (H : Msg → Msg) (Cm : Commitment Bool) (lay : Layout L S) (k : ℕ)
    (op : ℕ → List (Obs L K S) → List Bool) (s : Stmt) (ops : List (Opening Bool)) (o : List (Obs L K S)) : Prop :=
  (∀ q ∈ ops, Cm.Opens H s.roots q = true) ∧
    ∀ t (_ : t < k),
      (∀ ℓ σ, ∀ g ∈ slotGates (lay.egress t ℓ σ), (∀ b, s.prog.gates[g]? ≠ some (.const b)) ∧ ∃ q ∈ ops, q.pos = g) ∧
      (∀ g ∈ lay.ingress t, ∃ q ∈ ops, q.pos = g) ∧
      ∃ h : t < o.length, (∀ ℓ, o[t].rows ℓ = rowOf (tapVal ops) (lay.egress t ℓ)) ∧
        op t (o.take t) = (lay.ingress t).map (tapVal ops)

/-- **The leak relation at the tap's rows** at `δ`, on the statement's bytes: the statement parses and has a partition,
and every observation sequence of length `k` with jitter below `Dj` that the tap registers (`TapOpens`, its openings
`tops s`) with every status complete is among the sequences `AccSeqsSampled` counts at the inputs `As s ρ`, `ρ` the
trace's committed values. -/
def RLeakTap (H : Msg → Msg) (Cm : Commitment Bool) (parse : Msg → Option Stmt) (partition : Stmt → Option Partition)
    (L K S : ℕ) (lay : Stmt → Layout L S) (k : Stmt → ℕ) (δ : ℝ) (As : Stmt → (ℕ → Bool) → Set (ℕ → Bool)) (Dj : ℕ)
    (tops : Stmt → List (Opening Bool)) : Rel Wit :=
  fun x w => ∃ s P, parse x = some s ∧ partition s = some P ∧
    ∀ (op : ℕ → List (Obs L K S) → List Bool) (o : List (Obs L K S)),
      TapOpens H Cm (lay s) (k s) op s (tops s) o → Covered (k s) o → o.length = k s → (∀ y ∈ o, y.jitter < Dj) →
        o ∈ AccSeqsSampled s.prog P s.design (lay s) (k s) (As s (traceVal s.prog (w.val H Cm s))) δ op Dj

/-- The `H` calls that check the openings `ops` under `roots`. -/
def tapLog (H : Msg → Msg) (Cm : Commitment Bool) (roots : List Msg) (ops : List (Opening Bool)) : List (Msg × Msg) :=
  ops.flatMap fun o => ((Cm.opens roots o).eval H).2

section Finder

variable {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **`B_tap`**, the tap's collision finder: on the judged run's tape and coins and the extractor's coins, the extractor
run, and the first collision among its own `H` calls and the calls that check the tap's openings `tops s` under the
statement's roots. Its cost is the extraction's SHA-512 calls, its reruns' included, and the checks'. -/
noncomputable def Btap (H : Msg → Msg) (V : Verifier Coin) (E : Extractor Coin W V) (Cm : Commitment Bool)
    (parse : Msg → Option Stmt) (tops : Stmt → List (Opening Bool)) : FinderFamily :=
  fun Ω _ _ P x =>
    let tl := ((parse x).map fun s => tapLog H Cm s.roots (tops s)).getD []
    { S := Ω × (Fin (V.k x) → Coin) × E.C x
      out := fun s => findColl (ownLog H (run H P V x s.1) (E.ext x s.2.1 s.2.2) ++ tl)
      cost := fun s => (E.eval H P x s.1 s.2.1 s.2.2).2.1 + tl.length }

end Finder

end NetTiming
