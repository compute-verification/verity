module

public import Proofs.Flock.SoundAtBridge
public import Proofs.Flock.SoundAtBridge.Session
public import Proofs.Flock.SoundAtBridge.Union
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init

/-!
# The named premises a window of Flock sessions takes

The premises `Accounting.Guarantees.TS.SessionSoundAt` takes on the served proof layer `pf`
(`Accounting.TS.ProofLayer`), besides `Flock.SoundAtBridge`'s own: the proof system's acceptance rule (`AcceptedLive`,
the bridge's premise of that name, as a named `Prop`), the window's sampling design (`WindowDesign`), and in place of the bridge's
`Flock.SoundAtBridge.RelOfRows` and `Flock.SoundAtBridge.OpenedPlurAt`, the same two facts on the accepted sessions
whose sampled proof blocks are right: there `pf`'s relation, the served Definition's check, holds at the session's value layer
(`ServedChecksHold`), and the value layer is `pf`'s served opening, outside `δs` (`ServedRootsLinked`).

They belong in `Security/Definitions/Accounting/Assumptions/TS`, but they read `RowsSession`, which is in the proofs package
(the `security` package can't import it), so they are here until Flock's spec is extracted, as Flock's own
assumptions are (`Proofs/Flock/Soundness/Assumptions`).
-/

@[expose] public section

namespace Accounting.Assumptions.TS

open Accounting.TS Flock.SoundAtBridge FlockSoundness.Discharge.ZkReg
open Pouw.RelativeSecurity (Paths)

variable {Ω : Type} {S nπ : ℕ} {V : Type}

/-- `accounting-ts-accepted-live` (the proof system's acceptance rule). **A proof is credited only after
flock-verify `--zk --registered` accepts its session**: on every path, `pf` accepts proof `i` of round `w` only where
flock-verify accepts its session's outcome (`LiveAcceptsZKR`). `Flock.SoundAtBridge.AcceptedLive` is the same `Prop`.

Every served path today verifies with plain `flock-verify --zk` and binds its served values through the public file
(served_zk's Lean checks, `served_commit`'s verdict, `h1_prove`); of the audit's runners only `one_stage/a0.py` passes
`--registered`. So no served path meets it until the served trees are registered as ports of `own`
(`RowsSession.ServedRegistered`), or a bridge for public-file binding is stated. -/
def AcceptedLive (pf : ProofLayer Ω S nπ V) (sess : Paths Ω S → Fin S → Fin nπ → RowsSession)
    (out : ∀ x w i, (sess x w i).Out) : Prop :=
  ∀ x w i, pf.accepted x w i → LiveAcceptsZKR (sess x w i).I (sess x w i).own (out x w i)

/-- `accounting-ts-window-law` (live coins at the window: the proof system's samples and audit coins are the window's).
**At each round and proof, the window's coins read through the session have the session game's law**: for every event
`E` of a session's outcome, its probability over the window's paths at `(w, i)` is at most the average over the paths
of its probability in the session's own game (`Flock.SoundAtBridge.SessionLaw`). Read with every `E` it is equality of
the two laws; `Flock.SoundAtBridge.WindowLaw` is the same `Prop`. The equality rests on five facts of the proof system
and of how `sess` is built:
1. **No retry or resample.** `out x w i` is the outcome of the session's one run on its sample: a refused session is a
   visible failure and is never rerun on a fresh sample. Otherwise the kept outcome leans toward acceptance, and `δ`
   grows with the attempts.
2. **The sample comes after the setting.** The sample's bytes are revealed only after the session's registration and
   prover are fixed (the challenger records the registration, then opens the sample), so `sess x w i` can't depend on
   them.
3. **No coin reuse.** The sample's bytes are no coin the setting reads: not σ_w, not derived from σ_w or from a salt the
   prover holds, and not another session's sample.
4. **The path carries the sample.** On `Paths Ω S` round `w`'s coin is σ_w, which the session's unit reads, so the sample
   is a later coin, or `Ω` carries each round's sample beside σ_w, revealed after the registration. Round `S − 1`'s
   sessions have no later coin, so there `Ω` carries the sample; whoever builds `sess` says which.
5. **The audit coins are the auditor's.** The session's game also randomizes Flock's own live audit coins, and
   `EndToEndRegisteredRows`' `bound` is small only because they are fresh and uniform given the prover. Under `--zk`
   they are the verifier's keyed coin tree, committed at `Hello` after the prover's nonce and opened round by round.
   So the session's audit coins come from the auditor's verifier, on a machine the prover doesn't control, fresh for
   each session, under the named assumption `trusted-host`. Neither `CoinTree.checkFresh`, which refuses only a
   key reused within a run, nor the coin tree's commitment, which protects the prover, gives this.

If 1 fails, `δ` weakens; if 2, 3 or 5 fails, it breaks. 1–3 are the challenger's and the ZK server's code, which a check
can test; 5 is where the verifier runs. Under full checking (`kd = n`) the sample is every unit, whatever its bytes, so
2–4 constrain nothing, but the sampling design still says that `out` is the session's outcome at its setting on audit coins fresh
and uniform given the prover (5). served_zk's per-tile sessions are that case (one instance, `--draw subset:1`), and
today they flip their coin tree from a `serve --zk --operator loopback` that the job starts beside `prove --zk` on the
prover's node (`hidden_zk/job.sh`'s `session`), which doesn't meet 5. It stays a premise until a window is modelled as
Flock games run on one coin tape. -/
def WindowDesign [Fintype Ω] (sess : Paths Ω S → Fin S → Fin nπ → RowsSession) (out : ∀ x w i, (sess x w i).Out) :
    Prop :=
  ∀ w i, SessionLaw (Out := fun s : RowsSession => s.Out) (fun s => s.G) (fun s => s.τ) (fun x => sess x w i)
    (fun x => out x w i)

/-- `accounting-ts-served-checks` (a fact of each served Definition's circuit, to be computed by a check). **The served
Definition's check holds at its session's value layer wherever the sampled proof blocks are right**: on every path, when
flock-verify accepts proof `i` of round `w`'s session and every sampled proof block is right (`WrongAt` fails) with its rows
right at its own owned reads (`RowsAt`), `pf`'s relation (`rel x w i`, the proof's Definition) holds at `rows` of the
session's value layer (`XplurZC`). It is `Flock.SoundAtBridge.RelOfRows` on the accepted sessions. Under a sampled
sample it holds only for a relation that reads the sampled proof blocks alone: an unsampled unit can be wrong while every sampled one
is right. -/
def ServedChecksHold (pf : ProofLayer Ω S nπ V) (sess : Paths Ω S → Fin S → Fin nπ → RowsSession)
    (out : ∀ x w i, (sess x w i).Out) (rows : ∀ s : RowsSession, (Fin s.Circ.N → Bool) → V) : Prop :=
  ∀ x w i, LiveAcceptsZKR (sess x w i).I (sess x w i).own (out x w i) →
    (∀ u ∈ (out x w i).2.1, ¬ (sess x w i).WrongAt u ∧ (sess x w i).RowsAt (out x w i) u) →
    pf.rel x w i (rows (sess x w i) (sess x w i).Xv)

/-- `accounting-ts-served-roots-linked` (Flock's rows link at the served roots). **Where the sampled
proof blocks are right, the session's value layer is the served layer's opening**: at each round and proof, the probability
over the window's paths that flock-verify accepts the session, every sampled proof block is right at its own owned reads, and
yet `rows` of the session's value layer is not `pf`'s opening (`pf.opened x w i`: the round's served leaves, committed
before σ_w) is at most `δs`. It is `Flock.SoundAtBridge.OpenedPlurAt` on those sessions. `EndToEndRegisteredRows`
links only registered reads, and the served trees are hiding commitments that bind nothing on their own;
`Flock.SecurityProofs.ServedRowsLinked` links them pointwise, but only at sampled proof blocks, and where the service registers
each served tree as a port of `own` (`RowsSession.ServedRegistered`). With a `rows` that reads the served reads it gives
this at `δs = 0` only under full checking (`kd = n`) and `ServedRegistered` at every unit: under a sampled sample nothing proved ties an unsampled unit's value layer to the served
leaves, and `rows` can't see the sample. -/
def ServedRootsLinked [Fintype Ω] (pf : ProofLayer Ω S nπ V) (sess : Paths Ω S → Fin S → Fin nπ → RowsSession)
    (out : ∀ x w i, (sess x w i).Out) (rows : ∀ s : RowsSession, (Fin s.Circ.N → Bool) → V) (δs : ℝ) : Prop :=
  ∀ w i, Pouw.RelativeSecurity.prob (fun x =>
    LiveAcceptsZKR (sess x w i).I (sess x w i).own (out x w i) ∧
      (∀ u ∈ (out x w i).2.1, ¬ (sess x w i).WrongAt u ∧ (sess x w i).RowsAt (out x w i) u) ∧
      rows (sess x w i) (sess x w i).Xv ≠ pf.opened x w i) ≤ δs

end Accounting.Assumptions.TS
