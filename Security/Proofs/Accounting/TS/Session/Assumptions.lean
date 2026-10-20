import Proofs.Flock.SoundAtBridge

/-!
# The named premises a window of C-Flock sessions takes

The premises `Accounting.Guarantees.TS.SessionSoundAt` takes on the layer's side of #2909's bridge, as decided for the
lane (option A for both, 20 Oct): the window's law (`WindowLaw`), and in place of `Flock.SoundAtBridge.RelOfRows`, for
relations that read served operand and receipt leaves, its two halves: the served Definition's check at the drawn units
(`ServedChecksHold`) and C-Flock's rows link at the served roots (`ServedRootsLinked`).

They belong in `Security/Models/Accounting/Assumptions/TS`, but they read `RowsSession`, which is in the proofs package
(the `security` package can't import it), so they are here until C-Flock's spec is extracted, as C-Flock's own
assumptions are (`Proofs/Flock/Soundness/Assumptions`).
-/

namespace Accounting.Assumptions.TS

open Flock.SoundAtBridge FlockSoundness.Discharge.ZkReg
open Pouw.RelativeSecurity (Paths)

variable {Ω : Type} {S nπ : ℕ} {V : Type}

/-- `accounting-ts-window-law` (live coins at the window: the proof service's draws and audit coins are the window's).
**At each round and proof, the window's coins read through the session have the session game's law**: for every event
`E` of a session's outcome, its probability over the window's paths at `(w, i)` is at most the average over the paths
of its probability in the session's own game (`Flock.SoundAtBridge.SessionLaw`, at equality when, given what the path
fixes before the session, its setting and its prover, the session's coins are the path's next coins, fresh and
uniform). Read with every `E` it is equality of the two laws; `Flock.SoundAtBridge.WindowLaw` is the same `Prop`. It
stays a premise until a window is modelled as C-Flock games run on one coin tape. -/
def WindowLaw [Fintype Ω] (sess : Paths Ω S → Fin S → Fin nπ → RowsSession) (out : ∀ x w i, (sess x w i).Out) :
    Prop :=
  ∀ w i, SessionLaw (Out := fun s : RowsSession => s.Out) (fun s => s.G) (fun s => s.τ) (fun x => sess x w i)
    (fun x => out x w i)

/-- `accounting-ts-served-checks` (a fact of each served Definition's circuit, to be computed by a check). **A served
Definition's check holds at its session's value layer wherever its drawn units are right**: on every path, when
flock-verify accepts proof `i` of round `w`'s session and every drawn unit is right (`WrongAt` fails) with its rows
right at its own owned reads (`RowsAt`), the check `chk` (the Definition's output is 1, read at a layer value) holds at
`rows` of the session's value layer (`XplurZC`). Under a sampled draw it holds only for a `chk` that reads drawn units
alone. -/
def ServedChecksHold (sess : Paths Ω S → Fin S → Fin nπ → RowsSession) (out : ∀ x w i, (sess x w i).Out)
    (rows : ∀ s : RowsSession, (Fin s.Circ.N → Bool) → V) (chk : Paths Ω S → Fin S → Fin nπ → V → Prop) : Prop :=
  ∀ x w i, LiveAcceptsZKR (sess x w i).I (sess x w i).own (out x w i) →
    (∀ u ∈ (out x w i).2.1, ¬ (sess x w i).WrongAt u ∧ (sess x w i).RowsAt (out x w i) u) →
    chk x w i (rows (sess x w i) (sess x w i).Xv)

/-- `accounting-ts-served-roots-linked` (C-Flock's rows link at the served roots; owner @proofs, by extending
`Flock.Guarantees.EndToEndRegisteredRows` to them). **The extracted layer's served leaves are the committed trees'**:
at each round and proof, the probability over the window's paths that flock-verify accepts the session, every drawn
unit is right at its own owned reads, and yet `rows` of the session's value layer is not `served` there (its served
operand and receipt leaves are not those of the trees the path commits before and after the coin) is at most `δs`.
`EndToEndRegisteredRows` links only registered reads, and the window checks' operand and receipt trees are served
roots, hiding commitments that bind nothing on their own. `Flock.SecurityProofs.ServedRowsLinked` links them pointwise
where the service registers each served tree as a port of `own` (`RowsSession.ServedRegistered`); this premise is what
a window needs of that link at `served`. -/
def ServedRootsLinked [Fintype Ω] (sess : Paths Ω S → Fin S → Fin nπ → RowsSession) (out : ∀ x w i, (sess x w i).Out)
    (rows : ∀ s : RowsSession, (Fin s.Circ.N → Bool) → V) (served : Paths Ω S → Fin S → Fin nπ → V → Prop)
    (δs : ℝ) : Prop :=
  ∀ w i, Pouw.RelativeSecurity.prob (fun x =>
    LiveAcceptsZKR (sess x w i).I (sess x w i).own (out x w i) ∧
      (∀ u ∈ (out x w i).2.1, ¬ (sess x w i).WrongAt u ∧ (sess x w i).RowsAt (out x w i) u) ∧
      ¬ served x w i (rows (sess x w i) (sess x w i).Xv)) ≤ δs

end Accounting.Assumptions.TS
