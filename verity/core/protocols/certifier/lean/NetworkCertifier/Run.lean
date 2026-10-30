import NetworkCertifier.Grid

/-!
# The certifier program: statuses, observations, the egress prediction and acceptance

The executable forms of `Security/Definitions/NetworkCertifier/Run.lean` that the difftest runs, under the same names in
`NetworkCertifier` but `EgressPrediction` (the reference's `Program`), written as there. The interaction and its runs
have no executable role and stay in the reference only.
-/

namespace NetworkCertifier

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

/-- What the certifiers of ∂X emit at one step: the row of each of the `L` egress links, the step's jitter class, and the
coverage status of each of the `K` boundary links. -/
structure Obs (L K S : Nat) where
  rows : Fin L → Row S (List Bool)
  jitter : Nat
  status : Fin K → Status

/-- The operator's ingress at steps `0 … t`, recomputed from the observations. -/
def ingressUpTo {L K S : Nat} {In : Type} (op : Nat → List (Obs L K S) → In) (o : List (Obs L K S)) (t : Nat) :
    List In :=
  (List.range (t + 1)).map fun i => op i (o.take i)

/-- A committed Program's prediction of every egress row at step `t`, from the ingress it reads and the declared
advice. -/
structure EgressPrediction (L S : Nat) (In Adv : Type) where
  rows : Nat → List In → Adv → Fin L → Row S (List Bool)

/-- **Acceptance** of the first `k` steps with declared advice `a`: at every step `t < k` the recorded egress rows are
the Program's, given the ingress up to `t`, and every boundary link's status is complete. -/
def Accepted {L K S : Nat} {In Adv : Type} (P : EgressPrediction L S In Adv) (op : Nat → List (Obs L K S) → In)
    (k : Nat) (o : List (Obs L K S)) (a : Adv) : Prop :=
  ∀ t (ht : t < k), ∃ h : t < o.length,
    o[t].rows = P.rows t (ingressUpTo op o t) a ∧ ∀ κ, o[t].status κ = Status.complete

end NetworkCertifier
