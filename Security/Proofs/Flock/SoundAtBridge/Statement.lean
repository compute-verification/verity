import Proofs.Flock.SoundAtBridge.Union
import Proofs.Flock.SoundAtBridge.Session

/-!
# PoUW's proof premise from C-Flock's rows link (trusted)

`Flock.Guarantees.SoundAtBridge` concludes the proof layer's soundness (`Accounting.TS.ProofLayer.SoundAt`, the
layer whose relation reads the path's coins) from C-Flock's `Flock.Guarantees.EndToEndRegisteredRows`, applied once
per unit, when proof `i` of round `w` on the window's path `x` is a C-Flock session `sess x w i` (a `RowsSession`)
with outcome `out x w i`. It closes the three gaps between the two as follows:

1. **the probability bridge** is three named premises: the window's law at each `(w, i)` is its sessions'
   (`WindowLaw`), the layer accepts a proof only where flock-verify accepts its session (`AcceptedLive`), and the
   relation holds at the value layer's rows wherever every drawn unit of the session is right (`RelOfRows`);
2. **the opening against the plurality** is one named premise, `OpenedPlurAt`: an accepted proof's opening is the
   layer value `rows` of the plurality value layer `XplurZC`, outside `δo`; it holds at `δo = 0` when the layer's
   opening is defined as that value (`openedPlur_zero`);
3. **the per-unit link** is proved: each session carries one registrant's opening per unit, and a union bound over its
   `n` units gives `n` times `EndToEndRegisteredRows`' bound (`RowsSession.prob_badU_le`).

So `δ_ZK = δo + δ`, where `δ` bounds `n · (ε_ks + δ_link + δ_rows + q_R²/2^513)` at every session a path runs: the
whole of `EndToEndRegisteredRows`' bound, its rewinding terms `1/(e·M) + k/(e·Rw)` included, times the session's `n`.

Importing this module makes `X` a token (ArkLib's global `notation3 "X"`, `ArkLib.Data.Polynomial.Trivariate`), so a
module whose declarations bind `X` can't import it. `RowsSession` reads C-Flock's definitions, whose modules all reach
`Trivariate` through one edge, `Discharge.ZkSession.InnerSound` (BCIKS20's unique decoding), under
`ZkSession.Compose`; the edge leaves this closure only when C-Flock's definitions leave its proof modules. The
composition over any family of sessions, `Proofs.Flock.SoundAtBridge.Union`, imports no ArkLib module.
-/

namespace Flock.SoundAtBridge

open FlockSoundness FlockSoundness.Discharge.ZkReg Accounting.TS
open Pouw.RelativeSecurity (Paths)
open scoped ENNReal

variable {Ω : Type} [Fintype Ω] {S nπ : ℕ} {V : Type}

/-- **Gap 1, the window's law** (open): at each round `w` and
proof `i`, the window's coins, read through the session proof `i` runs and its outcome, have the sessions' law
(`SessionLaw`): the session's coins are the path's, fresh and uniform given its setting and its prover. -/
def WindowLaw (sess : Paths Ω S → Fin S → Fin nπ → RowsSession) (out : ∀ x w i, (sess x w i).Out) : Prop :=
  ∀ w i, SessionLaw (Out := fun s : RowsSession => s.Out) (fun s => s.G) (fun s => s.τ) (fun x => sess x w i)
    (fun x => out x w i)

/-- **Gap 1, acceptance** (open): the layer accepts proof `i` of round `w` only
where flock-verify accepts its session's outcome (`LiveAcceptsZKR`). -/
def AcceptedLive (pf : ProofLayer Ω S nπ V) (sess : Paths Ω S → Fin S → Fin nπ → RowsSession)
    (out : ∀ x w i, (sess x w i).Out) : Prop :=
  ∀ x w i, pf.accepted x w i → LiveAcceptsZKR (sess x w i).I (sess x w i).own (out x w i)

/-- **Gap 2, the opening is the plurality** (open): at each
`(w, i)`, an accepted proof's opening is other than `rows` of its session's value layer (`XplurZC`) with probability
at most `δo`; `δo = 0` when `opened` is defined as that value (`openedPlur_zero`). -/
def OpenedPlurAt (pf : ProofLayer Ω S nπ V) (sess : Paths Ω S → Fin S → Fin nπ → RowsSession)
    (rows : ∀ s : RowsSession, (Fin s.Circ.N → Bool) → V) (δo : ℝ) : Prop :=
  ∀ w i, Pouw.RelativeSecurity.prob
    (fun x => pf.accepted x w i ∧ pf.opened x w i ≠ rows (sess x w i) (sess x w i).Xv) ≤ δo

/-- **Gap 1, the relation is what the drawn units certify** (open): on every path, when no drawn unit of proof `i`'s
session is wrong (`WrongAt`) and every drawn unit's rows are right at its own owned reads (`RowsAt`), proof `i`'s
relation holds at `rows` of the value layer; it can hold only for a relation that reads no undrawn unit and no input
row at a read the verifier doesn't own. -/
def RelOfRows (pf : ProofLayer Ω S nπ V) (sess : Paths Ω S → Fin S → Fin nπ → RowsSession)
    (out : ∀ x w i, (sess x w i).Out) (rows : ∀ s : RowsSession, (Fin s.Circ.N → Bool) → V) : Prop :=
  ∀ x w i, (∀ u ∈ (out x w i).2.1, ¬ (sess x w i).WrongAt u ∧ (sess x w i).RowsAt (out x w i) u) →
    pf.rel x w i (rows (sess x w i) (sess x w i).Xv)

end Flock.SoundAtBridge

namespace Flock.Guarantees

open Flock.SoundAtBridge Accounting.TS
open Pouw.RelativeSecurity (Paths)
open scoped ENNReal

/-- **PoUW's proof premise from C-Flock's rows link, at `δ_ZK = δo + δ`.** For any window of proofs
(`pf : Accounting.TS.ProofLayer Ω S nπ V`, its relation on the path) whose proof `i` of round `w` on the
path `x` is the C-Flock session `sess x w i` (`EndToEndRegisteredRows`' setting with one registrant's opening per
unit, `RowsSession`) with outcome `out x w i`, and a reading `rows s` of a session's value layer as a layer value:
given `EndToEndRegisteredRows`' named assumptions at every session and every unit's opening (`RowsSession.CR`), the
window's law (`WindowLaw`), acceptance (`AcceptedLive`), the opening's agreement with the plurality outside `δo`
(`OpenedPlurAt`) and the relation's reading of the drawn units (`RelOfRows`), and `δ ≥ 0` with
$$n \cdot (\varepsilon_{ks} + \delta_{link} + \delta_{rows} + q_R^2/2^{513}) \le \delta$$
at every session a path runs (`RowsSession.bound`, `n` its units), each proof is accepted with an opening that fails
its relation on the path with probability at most `δo + δ` (`pf.SoundAt (δo + δ)`). -/
def SoundAtBridge : Prop :=
  ∀ {Ω : Type} [Fintype Ω] {S nπ : ℕ} {V : Type} (pf : ProofLayer Ω S nπ V)
    (sess : Paths Ω S → Fin S → Fin nπ → RowsSession) (out : ∀ x w i, (sess x w i).Out)
    (rows : ∀ s : RowsSession, (Fin s.Circ.N → Bool) → V) (δo δ : ℝ),
    -- cr/sha-512 and ecr/sha-512 as EndToEndRegisteredRows takes them, at every session and every unit's opening
    (∀ x w i, (sess x w i).CR) →
    WindowLaw sess out →
    AcceptedLive pf sess out →
    OpenedPlurAt pf sess rows δo →
    RelOfRows pf sess out rows →
    (∀ x w i, ((sess x w i).n : ℝ≥0∞) * (sess x w i).bound ≤ ENNReal.ofReal δ) → 0 ≤ δ →
    pf.SoundAt (δo + δ)

end Flock.Guarantees
