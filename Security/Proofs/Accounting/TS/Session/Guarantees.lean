import Proofs.Accounting.TS.Session.Layer
import Proofs.Accounting.TS.Session.Assumptions

/-!
# A window read through its C-Flock sessions is sound (the statement)

`Accounting.Guarantees.TS.SessionSoundAt`, #2909's bridge on #2855's proof layer read through its sessions
(`Accounting.TS.onSessions`), with the bridge's `WindowLaw` and `RelOfRows` taken from named premises
(`Accounting.Assumptions.TS`). It sits in the proofs package because the sessions do (`reads_exempt`).
-/

namespace Accounting.Guarantees.TS

open Accounting.TS Flock.SoundAtBridge
open Pouw.RelativeSecurity (Paths)
open scoped ENNReal

/-- **A window of proofs read through its C-Flock sessions is sound at `δs + δ`.** For any window of proofs
(`pf : ProofLayer Ω S nπ V`, #2855's) whose proof `i` of round `w` on the path `x` is the C-Flock session `sess x w i`
(#2909's `RowsSession`) with outcome `out x w i`, a reading `rows s` of a session's value layer as a layer value, a
check `chk` (the served Definition's output is 1 at a value) and `served` (a value's served operand and receipt leaves
are the trees the path commits):
given `Flock.Guarantees.EndToEndRegisteredRows`' named assumptions at every session and every unit's opening
(`RowsSession.CR`; the rows link itself is `Flock.SecurityProofs.EndToEndRegisteredRows`), the window's law
(`Accounting.Assumptions.TS.WindowLaw`), the check at the drawn units (`ServedChecksHold`), the rows link at the served
roots outside `δs` (`ServedRootsLinked`), a relation that holds at any value whose check holds and whose served leaves
are the committed trees' (the relation's Definition), and `δ ≥ 0` with
$$n \cdot (\varepsilon_{ks} + \delta_{link} + \delta_{rows} + q_R^2/2^{513}) \le \delta$$
at every session a path runs, each proof of `onSessions pf sess out rows` is accepted with an opening that fails its
relation on the path with probability at most `δs + δ`. -/
def SessionSoundAt : Prop :=
  ∀ {Ω : Type} [Fintype Ω] {S nπ : ℕ} {V : Type} (pf : ProofLayer Ω S nπ V)
    (sess : Paths Ω S → Fin S → Fin nπ → RowsSession) (out : ∀ x w i, (sess x w i).Out)
    (rows : ∀ s : RowsSession, (Fin s.Circ.N → Bool) → V) (chk served : Paths Ω S → Fin S → Fin nπ → V → Prop)
    (δs δ : ℝ),
    (∀ x w i, (sess x w i).CR) →
    Accounting.Assumptions.TS.WindowLaw sess out →
    Accounting.Assumptions.TS.ServedChecksHold sess out rows chk →
    Accounting.Assumptions.TS.ServedRootsLinked sess out rows served δs →
    (∀ x w i v, chk x w i v → served x w i v → pf.rel x w i v) →
    (∀ x w i, ((sess x w i).n : ℝ≥0∞) * (sess x w i).bound ≤ ENNReal.ofReal δ) → 0 ≤ δ →
    (onSessions pf sess out rows).SoundAt (δs + δ)

end Accounting.Guarantees.TS
