import Proofs.Flock.SoundAtBridge

/-!
# The named premises a window of C-Flock sessions takes

The premises `Accounting.Guarantees.TS.SessionSoundAt` takes on #2855's served layer `pf`, besides #2909's own, as
decided for the lane (option A for both, 20 Oct): the proof service's acceptance rule (`AcceptedLive`, #2909's premise
of that name, as a named `Prop`), the window's law (`WindowLaw`), and in place of #2909's
`Flock.SoundAtBridge.RelOfRows` and `Flock.SoundAtBridge.OpenedPlurAt`, the same two facts on the accepted sessions
whose drawn units are right: there `pf`'s relation, the served Definition's check, holds at the session's value layer
(`ServedChecksHold`), and the value layer is `pf`'s served opening, outside `δs` (`ServedRootsLinked`).

They belong in `Security/Models/Accounting/Assumptions/TS`, but they read `RowsSession`, which is in the proofs package
(the `security` package can't import it), so they are here until C-Flock's spec is extracted, as C-Flock's own
assumptions are (`Proofs/Flock/Soundness/Assumptions`).
-/

namespace Accounting.Assumptions.TS

open Accounting.TS Flock.SoundAtBridge FlockSoundness.Discharge.ZkReg
open Pouw.RelativeSecurity (Paths)

variable {Ω : Type} {S nπ : ℕ} {V : Type}

/-- `accounting-ts-accepted-live` (the proof service's acceptance rule; owner @proofs). **A proof is credited only after
flock-verify `--zk --registered` accepts its session**: on every path, `pf` accepts proof `i` of round `w` only where
flock-verify accepts its session's outcome (`LiveAcceptsZKR`). `Flock.SoundAtBridge.AcceptedLive` is the same `Prop`. -/
def AcceptedLive (pf : ProofLayer Ω S nπ V) (sess : Paths Ω S → Fin S → Fin nπ → RowsSession)
    (out : ∀ x w i, (sess x w i).Out) : Prop :=
  ∀ x w i, pf.accepted x w i → LiveAcceptsZKR (sess x w i).I (sess x w i).own (out x w i)

/-- `accounting-ts-window-law` (live coins at the window: the proof service's draws and audit coins are the window's).
**At each round and proof, the window's coins read through the session have the session game's law**: for every event
`E` of a session's outcome, its probability over the window's paths at `(w, i)` is at most the average over the paths
of its probability in the session's own game (`Flock.SoundAtBridge.SessionLaw`). Read with every `E` it is equality of
the two laws; `Flock.SoundAtBridge.WindowLaw` is the same `Prop`. The equality rests on four facts of the proof service
and of how `sess` is built:
1. **No retry or redraw.** `out x w i` is the outcome of the session's one run on its draw: a refused session is a
   visible failure and is never rerun on a fresh draw. Otherwise the kept outcome leans toward acceptance, and `δ`
   grows with the attempts.
2. **The draw comes after the setting.** The draw's bytes are revealed only after the session's registration and
   prover are fixed (the challenger records the registration, then opens the draw), so `sess x w i` can't depend on
   them.
3. **No coin reuse.** The draw's bytes are no coin the setting reads: not σ_w, not derived from σ_w or from a salt the
   prover holds, and not another session's draw.
4. **The path carries the draw.** On `Paths Ω S` round `w`'s coin is σ_w, which the session's unit reads, so the draw
   is a later coin, or `Ω` carries each round's draw beside σ_w, revealed after the registration. Round `S − 1`'s
   sessions have no later coin, so there `Ω` carries the draw; whoever builds `sess` says which.

If 1 fails, `δ` weakens; if 2 or 3 fails, it breaks. 1–3 are the challenger's and the firewall's code, which a check can
test. Under full checking (`kd = n`) the draw is every unit, whatever its bytes, and the law says only that `out` is the
session's outcome at its setting. It stays a premise until a window is modelled as C-Flock games run on one coin
tape. -/
def WindowLaw [Fintype Ω] (sess : Paths Ω S → Fin S → Fin nπ → RowsSession) (out : ∀ x w i, (sess x w i).Out) :
    Prop :=
  ∀ w i, SessionLaw (Out := fun s : RowsSession => s.Out) (fun s => s.G) (fun s => s.τ) (fun x => sess x w i)
    (fun x => out x w i)

/-- `accounting-ts-served-checks` (a fact of each served Definition's circuit, to be computed by a check). **The served
Definition's check holds at its session's value layer wherever the drawn units are right**: on every path, when
flock-verify accepts proof `i` of round `w`'s session and every drawn unit is right (`WrongAt` fails) with its rows
right at its own owned reads (`RowsAt`), `pf`'s relation (`rel x w i`, the proof's Definition) holds at `rows` of the
session's value layer (`XplurZC`). It is #2909's `RelOfRows` on the accepted sessions. Under a sampled draw it holds
only for a relation that reads the drawn units alone: an undrawn unit can be wrong while every drawn one is right. -/
def ServedChecksHold (pf : ProofLayer Ω S nπ V) (sess : Paths Ω S → Fin S → Fin nπ → RowsSession)
    (out : ∀ x w i, (sess x w i).Out) (rows : ∀ s : RowsSession, (Fin s.Circ.N → Bool) → V) : Prop :=
  ∀ x w i, LiveAcceptsZKR (sess x w i).I (sess x w i).own (out x w i) →
    (∀ u ∈ (out x w i).2.1, ¬ (sess x w i).WrongAt u ∧ (sess x w i).RowsAt (out x w i) u) →
    pf.rel x w i (rows (sess x w i) (sess x w i).Xv)

/-- `accounting-ts-served-roots-linked` (C-Flock's rows link at the served roots; owner @proofs). **Where the drawn
units are right, the session's value layer is the served layer's opening**: at each round and proof, the probability
over the window's paths that flock-verify accepts the session, every drawn unit is right at its own owned reads, and
yet `rows` of the session's value layer is not `pf`'s opening (`pf.opened x w i`: the round's served leaves, committed
before σ_w) is at most `δs`. It is #2909's `OpenedPlurAt` on those sessions. `EndToEndRegisteredRows` links only
registered reads, and the served trees are hiding commitments that bind nothing on their own;
`Flock.SecurityProofs.ServedRowsLinked` links them pointwise where the service registers each served tree as a port of
`own` (`RowsSession.ServedRegistered`), which with a `rows` that reads the served reads gives this at `δs = 0`. -/
def ServedRootsLinked [Fintype Ω] (pf : ProofLayer Ω S nπ V) (sess : Paths Ω S → Fin S → Fin nπ → RowsSession)
    (out : ∀ x w i, (sess x w i).Out) (rows : ∀ s : RowsSession, (Fin s.Circ.N → Bool) → V) (δs : ℝ) : Prop :=
  ∀ w i, Pouw.RelativeSecurity.prob (fun x =>
    LiveAcceptsZKR (sess x w i).I (sess x w i).own (out x w i) ∧
      (∀ u ∈ (out x w i).2.1, ¬ (sess x w i).WrongAt u ∧ (sess x w i).RowsAt (out x w i) u) ∧
      rows (sess x w i) (sess x w i).Xv ≠ pf.opened x w i) ≤ δs

end Accounting.Assumptions.TS
