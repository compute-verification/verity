import Proofs.Accounting.TS.Session.Assumptions

/-!
# The served layer is sound through its C-Flock sessions (the statement)

`Accounting.Guarantees.TS.SessionSoundAt`, the bridge `Flock.SoundAtBridge` on the proof layer `pf` itself, with the
bridge's `AcceptedLive`, `WindowLaw`, `RelOfRows` and `OpenedPlurAt` taken from named premises
(`Accounting.Assumptions.TS`). Its conclusion is `pf.SoundAt`, so `BindingAt`, `Of` and `RightFixed` keep reading `pf`'s
served openings. It sits in the proofs package because the sessions do (`reads_exempt`). -/

namespace Accounting.Guarantees.TS

open Accounting.TS Flock.SoundAtBridge
open Pouw.RelativeSecurity (Paths)
open scoped ENNReal

/-- **A window of proofs is sound at `δs + δ` through its C-Flock sessions.** For any window of proofs
(`pf : ProofLayer Ω S nπ V`) whose proof `i` of round `w` on the path `x` is the C-Flock session `sess x w i`
(a `RowsSession`) with outcome `out x w i`, and a reading `rows s` of a session's value layer as a layer value:
given `Flock.Guarantees.EndToEndRegisteredRows`' named assumptions at every session and every unit's opening
(`RowsSession.CR`; the rows link itself is `Flock.SecurityProofs.EndToEndRegisteredRows`), the proof service's
acceptance rule (`Accounting.Assumptions.TS.AcceptedLive`), the window's law (`WindowLaw`), `pf`'s relation at the
value layer of an accepted session whose drawn units are right (`ServedChecksHold`), that value layer being `pf`'s
opening outside `δs` (`ServedRootsLinked`), and `δ ≥ 0` with
$$n \cdot (\varepsilon_{ks} + \delta_{link} + \delta_{rows} + q_R^2/2^{513}) \le \delta$$
at every session a path runs, each proof of `pf` is accepted with an opening that fails its relation on the path with
probability at most `δs + δ`.

Under a sampled draw `ServedChecksHold` holds only for a relation that reads the drawn units alone, so there this
bounds the drawn units' unsoundness; the bound on every unit (ε at δ) is the sampled-proofs statement's, not this
one's. Under full checking (`kd = n`) the relation may read every unit.

Every served path today verifies with plain `flock-verify --zk` and binds its served values through the public file
(served_zk's Lean checks, `served_commit`'s verdict, `h1_prove`); of the audit's runners only `one_stage/a0.py` passes
`--registered`. So no served path meets `AcceptedLive` until the served trees are registered as ports of `own`
(`RowsSession.ServedRegistered`), or a bridge for public-file binding is stated. -/
def SessionSoundAt : Prop :=
  ∀ {Ω : Type} [Fintype Ω] {S nπ : ℕ} {V : Type} (pf : ProofLayer Ω S nπ V)
    (sess : Paths Ω S → Fin S → Fin nπ → RowsSession) (out : ∀ x w i, (sess x w i).Out)
    (rows : ∀ s : RowsSession, (Fin s.Circ.N → Bool) → V) (δs δ : ℝ),
    (∀ x w i, (sess x w i).CR) →
    Accounting.Assumptions.TS.AcceptedLive pf sess out →
    Accounting.Assumptions.TS.WindowLaw sess out →
    Accounting.Assumptions.TS.ServedChecksHold pf sess out rows →
    Accounting.Assumptions.TS.ServedRootsLinked pf sess out rows δs →
    (∀ x w i, ((sess x w i).n : ℝ≥0∞) * (sess x w i).bound ≤ ENNReal.ofReal δ) → 0 ≤ δ →
    pf.SoundAt (δs + δ)

end Accounting.Guarantees.TS
