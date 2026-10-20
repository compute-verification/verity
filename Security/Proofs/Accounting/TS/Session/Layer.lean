import Proofs.Flock.SoundAtBridge

/-!
# A window's proofs read through their C-Flock sessions (trusted)

`onSessions pf sess out rows` is #2855's proof layer `pf` with proof `i` of round `w` on the path `x` read through the
C-Flock session `sess x w i` (#2909's `RowsSession`) and its outcome `out x w i`:

* it is accepted where `pf` accepts it and flock-verify `--zk --registered` accepts the session's outcome
  (`LiveAcceptsZKR`);
* its opening is the session's value layer (`XplurZC`, the plurality the rows extraction finds at the registration) read
  as a layer value (`rows`), the only value a hiding leaf has;
* its relation and rightness are `pf`'s.

So `Flock.SoundAtBridge.AcceptedLive` holds on it, and `OpenedPlurAt` at `δo = 0`, by definition. Binding
(`ProofLayer.BindingAt`, `ProofLayer.Forked`) reads `opened`, so on this layer it is about the extracted values.

It sits in the proofs package because `RowsSession` does; `Security/lean-audit.json`'s `reads_exempt` says so.
-/

namespace Accounting.TS

open Flock.SoundAtBridge FlockSoundness.Discharge.ZkReg
open Pouw.RelativeSecurity (Paths)

/-- **`pf` read through its sessions**: accepted where `pf` accepts and flock-verify accepts the session's outcome,
opened to the session's value layer read by `rows`, with `pf`'s relation and rightness. -/
noncomputable def onSessions {Ω : Type} {S nπ : ℕ} {V : Type} (pf : ProofLayer Ω S nπ V)
    (sess : Paths Ω S → Fin S → Fin nπ → RowsSession) (out : ∀ x w i, (sess x w i).Out)
    (rows : ∀ s : RowsSession, (Fin s.Circ.N → Bool) → V) : ProofLayer Ω S nπ V where
  accepted x w i := pf.accepted x w i ∧ LiveAcceptsZKR (sess x w i).I (sess x w i).own (out x w i)
  opened x w i := rows (sess x w i) (sess x w i).Xv
  rel := pf.rel
  right := pf.right

end Accounting.TS
