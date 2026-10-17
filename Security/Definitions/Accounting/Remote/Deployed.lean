import Definitions.Accounting.Remote.Run
import Definitions.Accounting.RateAuditDeployed

/-!
# The deployed game's session as a run of the remote audit (trusted)

`Run.ofDeployed` reads a deployed session (`Accounting.Rate.DeployedSession`, `Definitions/Accounting/RateAuditDeployed.lean`)
as a run of the remote audit (`Accounting.Remote.Run`) on a node `Nd` with parameters `a`, over the session's coins: the
store and every window's coins, uniform. The audit's passes are the session's `n·P`; its declared capacity is meant to be
the session's `n·P·N·b` bits, which the guarantees over it take as a hypothesis on `Nd` and `a`.

Each field is the game's quantity, or its bound where the game proves only a bound:

* `accepts`: every window's pooled draw lands on correct receipts; `correct`: the correct receipts over the session;
  `committed`: every receipt, `n·P·N`.
* `read ω m`: the device's reads of tier `m`, summed over the session's passes.
* `state`: the on-chip state, `S` bits a pass (the game's `Bits S`).
* `readOut`: the side link's bits beyond the certified stream, at its bound `L − ρ` a pass. The side sources answer as
  any function of the store, the stream and what the chip sent: memory and compute outside the node, whose answers
  conservation at no location (`Run.paid`) charges as reads outside it.
* `rebuilt`: the bits the passes complete, at their bound `X` a pass (`b·F/(rounds·c_root)` under `WorkHardness` and
  `SquaringCeiling`).
* `stateOut`, `rebuiltOut`: `0`; the game's outside is the side sources, counted in `readOut`.

`Run.ofDeployedWindow` is the same at one window `w`: its `P` passes, its correct receipts and reads, accepting when
every window's draw passes.

The game does not model the rest (the node's priced work, the signals over its boundary, the bound `z`, the outside's
work, Mbw's independent data): each is `0`, and no route proved for this run reads one.
-/

namespace Accounting.Remote

open Accounting.Rate

variable {n P N b h ρ S L q r s d G τ : ℕ}

/-- A deployed session as a run of the remote audit on node `Nd` with parameters `a`, with `X` bits completed a pass. -/
def Run.ofDeployed (A : DeployedSession n P N b h ρ S L q r s d G) (tier : ℕ → Fin τ) (X : ℕ) (Nd : Node (Fin τ))
    (a : Audit) : Run Nd a (Store N b × (Fin n → WindowCoins P N b h ρ r s d)) where
  accepts x := ∀ w t, (x.2 w).2 t ∈ (A.win w x.2).correctAll x.1 (x.2 w).1
  committed _ := n * P * N
  correct x := ∑ w, ((A.win w x.2).correctAll x.1 (x.2 w).1).card
  read x m := ∑ w, ∑ j,
    ((A.win w x.2).dev j x.1 (x.2 w).1).tierReads tier m x.1 ((x.2 w).1 j).1 ((A.win w x.2).oracle j x.1 (x.2 w).1)
  readOut _ := n * P * (L - ρ)
  state _ := n * P * S
  stateOut _ := 0
  rebuilt _ := n * P * X
  rebuiltOut _ := 0
  rebuildWork _ := 0
  pouwWork _ := 0
  inBits _ := 0
  outBits _ := 0
  zOut _ := 0
  workOut _ := 0
  other _ := 0

/-- Window `w` of a deployed session as a run of the remote audit on node `Nd` with parameters `a`, over the session's
coins: `Run.ofDeployed`'s fields at window `w`'s `P` passes, accepting when every window's draw passes. -/
def Run.ofDeployedWindow (A : DeployedSession n P N b h ρ S L q r s d G) (w : Fin n) (tier : ℕ → Fin τ) (X : ℕ)
    (Nd : Node (Fin τ)) (a : Audit) : Run Nd a (Store N b × (Fin n → WindowCoins P N b h ρ r s d)) where
  accepts x := ∀ v t, (x.2 v).2 t ∈ (A.win v x.2).correctAll x.1 (x.2 v).1
  committed _ := P * N
  correct x := ((A.win w x.2).correctAll x.1 (x.2 w).1).card
  read x m := ∑ j,
    ((A.win w x.2).dev j x.1 (x.2 w).1).tierReads tier m x.1 ((x.2 w).1 j).1 ((A.win w x.2).oracle j x.1 (x.2 w).1)
  readOut _ := P * (L - ρ)
  state _ := P * S
  stateOut _ := 0
  rebuilt _ := P * X
  rebuiltOut _ := 0
  rebuildWork _ := 0
  pouwWork _ := 0
  inBits _ := 0
  outBits _ := 0
  zOut _ := 0
  workOut _ := 0
  other _ := 0

end Accounting.Remote
