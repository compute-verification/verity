import Proofs.Flock.Recursive.Raw

/-!
# The registrant's record: the recursive audit's premise on its openings

A `--zk` transcript holds a registered row's `b ‖ c` and path only, and `ReadsX.committed` asks for the row and salt
under it, so no accepted run carries the registrant's openings. `RegistrantCovers` is the premise that its record
holds them at every row of a port the session's reads file names, standing in for their one-run extraction.
`FlockC3.Compose.registrantCovers_attained` (`Proofs.Flock.C3.Compose.RowsReg`) meets it at every family of V*'s
sessions whose registrant opens the rows the verifier reads, together with the guarantee's other hypotheses on them.
-/

namespace Flock.Assumptions

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR
  DecodeZL Composed ZkReg FailClosed

/-- Premise, standing in for the one-run extraction of the registrant's openings. **The registrant's record covers the
rows its reads file names.** At each typing of V*'s session `x` as the prover supplies it, every row of a port the
session's reads file `own` names is a registered read of the session's record `rd`, at every unit. The records are the
inner session's prover's: it committed each round's rows and salts behind the firewall (each row's `b ‖ c` is the
round's commitment the coin server records, `InnerRecordCustody`), and the registered roots and positions are the
verifier's own (`own`, `--registered`). -/
def RegistrantCovers {mPts nv J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit}
    {hscope : Layout.scopeOk c = true} {Lv : Law nv} (x : ZkOuterR mPts nv J Reg' c hU hscope Lv) : Prop :=
  ∀ ht S₀ d₀ y₀ hc hpub (u : Fin nv) (q : Fin c.ports.size), (∃ pt ∈ x.own, pt.name = c.ports[q].name) →
    (x.rd ht S₀ d₀ y₀ hc hpub).isRead (.inl (u, q))

end Flock.Assumptions
