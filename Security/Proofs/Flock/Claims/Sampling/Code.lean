module

public import Proofs.Flock.Claims.Deploy
public import Proofs.Flock.Claims.Sampling.Reference.Prover

/-!
# The sampled verifier's code

`SamplingDeploy.Code`, the record of what a deployed sampled verifier fixes, with only its fields' types: the deployed
record `SamplingDeploy.deployOf` (`Claims.Sampling.Deployed`) and each deployment's code (`SpotFlock.code`) read it
without the proofs the deployed record is built on.
-/

@[expose] public section

namespace SamplingDeploy

open Claims Claims.Sampling

/-- The deepest registered port's levels. -/
def depth (own : Array _root_.Flock.Registered.Port) : ℕ :=
  own.toList.foldr (fun p m => max (_root_.Flock.log2ceil p.leaves) m) 0

/-- **The sampled verifier's code**, as the record reads it: the verifier in public-coin form (`V`), its parser, scope
check and partition evaluation; its sample layout (`stage2`, `checked`, `rounds`); each checked unit's session and what
its Flock witness reveals (`session`, `reveal`); the trace commitment's registered ports (`own`), the ports a position
reads under a run's roots (`ownAt`: `own` unless the code sets it, each no deeper than `own`), the root and leaf
index of each position under a run's roots (`at_`) and its published leaf layer (`layer`); where each committed bit
sits, its row's width and its place there (`place`), and how the reveal encodes an opening's data: the row's other bits,
the salt and the path (`bits`, `salt`, `π`); and its numbers: the sessions' `N₀` at a statement (`innerN₀`) and at each
unit (`unitN₀`: the statement's unless the code sets it), the most tables a session at a
statement sets up (`tables`), the most openings one session reveals (`reveals`) and the slots whose checked units'
sessions are all of a run's (`sessions`: `1` when one session proves every sampled proof block, the proof blocks' count at one
session per unit); and the reference prover's framing (`Reference.refP`): where a run is at a history (`step`), the
positions a session opens (`reads`) and the Flock witness that encodes what it reveals (`wit`), each by the session's
statement. -/
structure Code where
  V : Verifier FlockDeploy.Coin
  parse : Msg → Option AuditStatement
  scopeCheck : Msg → Bool
  partition : AuditStatement → Option Partition
  stage2 : (x : Msg) → Finset (Fin (V.k x))
  checked : (x : Msg) → (Fin (V.k x) → FlockDeploy.Coin) → List ℕ
  rounds : (x : Msg) → ℕ → Finset (Fin (V.k x))
  session : Msg → ℕ → Seen → Option (Msg × Seen)
  reveal : AuditStatement → ℕ → Seen → Claims.Flock.Wit → List (List Msg × Opening Bool)
  own : Array _root_.Flock.Registered.Port
  ownAt : List Msg → ℕ → Array _root_.Flock.Registered.Port := fun _ _ => own
  ownAt_depth : ∀ roots g, depth (ownAt roots g) ≤ depth own := by intros; exact le_rfl
  at_ : List Msg → ℕ → Option (Msg × ℕ)
  layer : (Msg → Msg) → List Msg → List (Opening Bool) → List Msg
  place : ℕ → (w : ℕ) × Fin w
  bits : Msg → ℕ → Bool
  salt : Opening Bool → Msg
  π : Opening Bool → Msg
  innerN₀ : Msg → ℝ
  unitN₀ : Msg → ℕ → ℝ := fun x _ => innerN₀ x
  tables : Msg → ℕ
  reveals : Msg → ℕ
  sessions : Msg → ℕ
  step : Msg → List (Msg × Msg) → Reference.Step
  reads : AuditStatement → Msg → List ℕ
  wit : AuditStatement → Msg → List (Opening Bool) → Claims.Flock.Wit

/-- The proof blocks at `x` under the code's parser and partition evaluation, `0` on a statement that doesn't parse. -/
def units (c : Code) (x : Msg) : ℕ :=
  match c.parse x with
  | some s => ((c.partition s).getD ⟨[]⟩).blocks.length
  | none => 0

/-- The code with every unit's `N₀` its statement's (`unitN₀ x _ = innerN₀ x`), as a one-stage record reads it. -/
def Code.atInnerN₀ (c : Code) : Code := { c with unitN₀ := fun x _ => c.innerN₀ x }

end SamplingDeploy
