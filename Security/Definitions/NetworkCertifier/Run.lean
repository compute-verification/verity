import Definitions.NetworkCertifier.Grid
import Mathlib.Data.List.Basic

/-!
# The interaction, the Program, and acceptance

Definitions only. `docs/network-transparency/timing-channel.md` §1–§2.

**Granularity.** A step is one bucket of the whole boundary ∂X. At step `k` the observation `Obs` holds every egress
link's row, one jitter class, and every boundary link's coverage status. The formalization is for egress: the secret is
inside X, and what the observations carry out of X is counted. Ingress is the operator's move, which the operator
chooses from what it has observed. The ingress direction is the mirror image: `NetTiming.Protocol.Ingress`
(`Protocol/Ingress.lean`) reads this interaction inward, and its guarantees are `Guarantees.EncardIngressObsDecodableLe`
and `Guarantees.EncardIngressDecodableLe`, with their `…Contents` versions.

**Emission and acceptance are separate.** The certifier's `obs` and `wire` say what is recorded and emitted, whatever
happens. The verifier's `Accepted` is a predicate on an observation prefix and a declared advice string, and it reads
only the rows and the statuses (W5's record). No violation policy is modelled: every theorem is about runs accepted up to
the horizon.
-/

namespace NetTiming

/-- Why a link's coverage was not established (draft §4, W5). -/
inductive Reason
  | monitorUnavailable
  | overflow
  | malformed
  | dropped
  | deliveryRefused
  | coverageNeverEstablished
  deriving DecidableEq

/-- A boundary link's coverage status at a step. -/
inductive Status
  | complete
  | notEstablished (r : Reason)
  deriving DecidableEq

/-- What the certifiers of ∂X emit at one step, as the model observes it: the row of each of the `L` egress links, the
step's jitter class, and the coverage status of each of the `K` boundary links (ingress links included, because a
device in X can set an ingress link's status from its inner side). -/
structure Obs (L K S : ℕ) where
  rows : Fin L → Row S (List Bool)
  jitter : ℕ
  status : Fin K → Status

/-- The certifiers of ∂X (X a spatial unit), after conditioning on their own noise. At step `k` what they record (`obs`)
and what they put on the wire (`wire`) are functions of the ingress and of the sender's actions so far (both lists of
length `k + 1`). `wire` is everything any receiver can see, the published record (grid and status) included. `fault` marks actions that are faults (overflow, malformed traffic,
pulling a link, refusing delivery). -/
structure Certifier (L K S : ℕ) (In Act Wire : Type) where
  obs : ℕ → List In → List Act → Obs L K S
  wire : ℕ → List In → List Act → Wire
  fault : Act → Prop

/-- An interaction on the boundary.
* `op`: the operator (and every other outside party), which reacts to what it has observed; Z-independent.
* `snd`: the sender inside X, which knows the secret and everything observed so far.
Both are deterministic: a randomized party is handled by conditioning on its coins. -/
structure Interaction (L K S : ℕ) (Z In Act Wire : Type) where
  certifier : Certifier L K S In Act Wire
  op : ℕ → List (Obs L K S) → In
  snd : Z → ℕ → List (Obs L K S) → Act

namespace Interaction

variable {L K S : ℕ} {Z In Act Wire : Type} (I : Interaction L K S Z In Act Wire)

/-- The first `k` steps of the run with secret `z`: observations, ingress, actions and wire, each a list of length `k`. -/
def trace (z : Z) : ℕ → List (Obs L K S) × List In × List Act × List Wire
  | 0 => ([], [], [], [])
  | k + 1 =>
    let p := trace z k
    let xs := p.2.1 ++ [I.op k p.1]
    let as := p.2.2.1 ++ [I.snd z k p.1]
    (p.1 ++ [I.certifier.obs k xs as], xs, as, p.2.2.2 ++ [I.certifier.wire k xs as])

/-- The observations of the run's first `n` steps. -/
def obsRun (z : Z) (n : ℕ) : List (Obs L K S) := (I.trace z n).1

/-- The sender's actions over the run's first `n` steps. -/
def actRun (z : Z) (n : ℕ) : List Act := (I.trace z n).2.2.1

/-- What the wire carries over the run's first `n` steps. -/
def wireRun (z : Z) (n : ℕ) : List Wire := (I.trace z n).2.2.2

end Interaction

/-- The operator's ingress at steps `0 … t`, recomputed from the observations: its move at step `i` reads the
observations before `i`, so this depends only on `o.take t`. -/
def ingressUpTo {L K S : ℕ} {In : Type} (op : ℕ → List (Obs L K S) → In) (o : List (Obs L K S)) (t : ℕ) : List In :=
  (List.range (t + 1)).map fun i => op i (o.take i)

/-- A committed Program's prediction of every egress row at step `t`, from the ingress it reads and the declared
advice. -/
structure Program (L S : ℕ) (In Adv : Type) where
  rows : ℕ → List In → Adv → Fin L → Row S (List Bool)

/-- **Acceptance** of the first `k` steps with declared advice `a` (the exact binding): at every step `t < k` the
recorded egress rows are the Program's, given the ingress up to `t`, and every boundary link's status is complete.
It reads only rows and statuses. -/
def Accepted {L K S : ℕ} {In Adv : Type} (P : Program L S In Adv) (op : ℕ → List (Obs L K S) → In) (k : ℕ)
    (o : List (Obs L K S)) (a : Adv) : Prop :=
  ∀ t (ht : t < k), ∃ h : t < o.length,
    o[t].rows = P.rows t (ingressUpTo op o t) a ∧ ∀ κ, o[t].status κ = Status.complete

/-- Accepted up to step `k` for some declared advice (acceptance by prefix). -/
def AcceptedUpTo {L K S : ℕ} {In Adv : Type} (P : Program L S In Adv) (op : ℕ → List (Obs L K S) → In) (k : ℕ)
    (o : List (Obs L K S)) : Prop :=
  ∃ a, Accepted P op k o a

/-- The accepted observation sequences of length `n` whose jitter classes are below `D`: what T3 counts. -/
def AccSeqs {L K S : ℕ} {In Adv : Type} (P : Program L S In Adv) (op : ℕ → List (Obs L K S) → In) (n D : ℕ) :
    Set (List (Obs L K S)) :=
  {o | o.length = n ∧ (∀ x ∈ o, x.jitter < D) ∧ AcceptedUpTo P op n o}

end NetTiming
