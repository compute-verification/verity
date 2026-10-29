import Definitions.NetworkCertifier.SampledCharge

/-!
# The network certifier's rows: its registration at the transfer gates, and the leak relation at them

Definitions only. The audit's trace holds each step's egress rows at the layout's egress gates and its ingress at the
ingress gates (`Layout`). The network certifier on the server's uplink, which functions as an active warden, registers
what it records there: openings `ops` under the statement's roots, one at each transfer gate before the horizon, whose
values (`tapVal ops`) are its records' egress rows and the operator's moves on them (`TapOpens`). Its registered root is
its signed root for the window, a function of what the server did, so these rows vary with the server's secret, unlike
the values the prover registers at `reg` (`AsReg`).

An extracted trace that opens a transfer gate to a value other than the network certifier's makes, with the network
certifier's opening there, two passing openings of one position to different values, so under `Commitment.Binds` the
calls that check them collide; `Btap` looks for that collision. Every transfer gate outside the faulty units is opened,
so where the trace agrees with the network certifier it carries the network certifier's records there; a faulty unit's
committed gates are the advice the count already charges (`2^β` over `faultySets`). The leak relation at the network
certifier's rows (`RLeakTap`) is `RLeak` with the observations it counts picked out by the network certifier's
registration rather than by the trace's transfer gates.

A constant gate is committed by nobody, and every trace holds the program's constant there, so at an egress gate the
program holds constant the network certifier's record must be that constant (an ingress gate is an input,
`Layout.Fits`).
-/

namespace NetTiming

open Claims Claims.Sampling

/-- **The network certifier's value** at gate `g`: its first opening at `g`'s position, `false` where it has none. -/
def tapVal (ops : List (Opening Bool)) (g : ℕ) : Bool :=
  ((ops.find? fun o => o.pos == g).map (·.val)).getD false

/-- **The network certifier registers the records `o`** at the transfer gates of the statement `s` over the horizon `k`,
at the operator `op`: each of its openings `ops` passes under the statement's roots; before `k`, every egress and
ingress gate has an opening, and at an egress gate the program holds constant the opening holds that constant; and each
step's record exists, its egress rows are the rows the openings hold at the egress gates, and the operator's move on the
records before it is the ingress they hold at the ingress gates. -/
def TapOpens {L K S : ℕ} (H : Msg → Msg) (Cm : Commitment Bool) (lay : Layout L S) (k : ℕ)
    (op : ℕ → List (Obs L K S) → List Bool) (s : AuditStatement) (ops : List (Opening Bool)) (o : List (Obs L K S)) : Prop :=
  (∀ q ∈ ops, Cm.Opens H s.roots q = true) ∧
    ∀ t (_ : t < k),
      (∀ ℓ σ, ∀ g ∈ slotGates (lay.egress t ℓ σ),
        (∀ b, s.prog.gates[g]? = some (.const b) → tapVal ops g = b) ∧ ∃ q ∈ ops, q.pos = g) ∧
      (∀ g ∈ lay.ingress t, ∃ q ∈ ops, q.pos = g) ∧
      ∃ h : t < o.length, (∀ ℓ, o[t].rows ℓ = rowOf (tapVal ops) (lay.egress t ℓ)) ∧
        op t (o.take t) = (lay.ingress t).map (tapVal ops)

/-- **The leak relation at the network certifier's rows** at `δ`, on the statement's bytes: the statement parses and has
a partition, and every observation sequence of length `k` with jitter below `Dj` that the network certifier registers
(`TapOpens`, its openings `tops s`) with every status complete is among the sequences `AccSeqsSampled` counts at the
inputs `As s ρ`, `ρ` the trace's committed values. -/
def RLeakTap (H : Msg → Msg) (Cm : Commitment Bool) (parse : Msg → Option AuditStatement) (partition : AuditStatement → Option Partition)
    (L K S : ℕ) (lay : AuditStatement → Layout L S) (k : AuditStatement → ℕ) (δ : ℝ) (As : AuditStatement → (ℕ → Bool) → Set (ℕ → Bool)) (Dj : ℕ)
    (tops : AuditStatement → List (Opening Bool)) : Rel Wit :=
  fun x w => ∃ s P, parse x = some s ∧ partition s = some P ∧
    ∀ (op : ℕ → List (Obs L K S) → List Bool) (o : List (Obs L K S)),
      TapOpens H Cm (lay s) (k s) op s (tops s) o → Covered (k s) o → o.length = k s → (∀ y ∈ o, y.jitter < Dj) →
        o ∈ AccSeqsSampled s.prog P s.design (lay s) (k s) (As s (traceVal s.prog (w.val H Cm s))) δ op Dj

/-- The `H` calls that check the openings `ops` under `roots`. -/
def tapLog (H : Msg → Msg) (Cm : Commitment Bool) (roots : List Msg) (ops : List (Opening Bool)) : List (Msg × Msg) :=
  ops.flatMap fun o => ((Cm.opens roots o).eval H).2

section Finder

variable {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **`B_tap`**, the collision finder against the network certifier's openings: on the judged run's tape and coins and
the extractor's coins, the extractor run, and the first collision among its own `H` calls and the calls that check the
network certifier's openings `tops s` under the statement's roots. Its cost is the extraction's SHA-512 calls, its
reruns' included, and the checks'. -/
noncomputable def Btap (H : Msg → Msg) (V : Verifier Coin) (E : Extractor Coin W V) (Cm : Commitment Bool)
    (parse : Msg → Option AuditStatement) (tops : AuditStatement → List (Opening Bool)) : FinderFamily :=
  fun Ω _ _ P x =>
    let tl := ((parse x).map fun s => tapLog H Cm s.roots (tops s)).getD []
    { S := Ω × (Fin (V.k x) → Coin) × E.C x
      out := fun s => findColl (ownLog H (run H P V x s.1) (E.ext x s.2.1 s.2.2) ++ tl)
      cost := fun s => (E.eval H P x s.1 s.2.1 s.2.2).2.1 + tl.length }

end Finder

end NetTiming
