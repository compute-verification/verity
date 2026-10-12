import Proofs.Flock.C3.PubRows.Honest

/-!
# n1c: the verifier's record of the inner session is the live history's

`HeldIs` (`PubRows/Honest.lean`, the record–history link) asks that the rows the verifier holds be the history's
commitments at every history where a session accepts. The verifier holds one record of the inner session, the coin
server's (`--rec-inner-transcript` and `--rec-inner-session`), while the history ranges over the recursive game's. No
check of the verifier's ties the two, the event constrains the record (`AcceptsC3`) and not the history, and
`Flock.Premises.RecordCustody` is V*'s sessions' records only. What ties them is custody of the inner session's records,
`live-verifier` at the inner session, which the guarantee takes as `_hIC` (`Flock.Premises.InnerRecordCustody`,
`Soundness/Assumptions/Inner.lean`) at the coin server's writer of them, `wr`:

* `heldIs_of_custody`: `HeldIs` at a family whose rows are the ones the verifier holds of the record written at the
  history, from the custody's commitment half at the map `m` and the commit strings' bytes being the commitment's at the
  port's value and row (`hbc`).
-/

namespace FlockC3.Compose

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockC3.PubRows

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

/-- **(n1c) The rows the verifier holds are the history's commitments**, at a family whose rows are the ones it holds of
the record written at the history, from the records' custody. -/
theorem heldIs_of_custody (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂)
    (own : Fin (vstarCount F.ish) → Array Flock.Registered.Port)
    (bc : (j : Fin (vstarCount F.ish)) → Cm → PosK (F.Vs j).c (F.Vs j).hU (F.Vs j).nv → List UInt8)
    (wr : List (Cm × (holoInnerAt T CT st hp).Coin) → Lean.Json × String) (reg : Lean.Json)
    (m : FlockSoundness.Assumptions.InnerRecordMap Cm (holoInnerAt T CT st hp).Coin)
    (hrow : ∀ v x t, F.row v x t = heldRow (innerCommitsAt (wr t).1 (wr t).2 st reg) F.ish.sparse v x)
    (hcust : Flock.Premises.InnerRecordCustody wr (fun w => innerCommitsAt w.1 w.2 st reg) F.ish.sparse m)
    (hbc : ∀ j c (u : Fin (F.Vs j).nv) (q : Fin (F.Vs j).c.ports.size), ∀ p ∈ own j,
      p.name = (F.Vs j).c.ports[q].name → ∀ x, p.reads[u.val]? = some (some x) →
        bc j c (.inl (u, q)) = m.bc c p.value x) :
    HeldIs F own m.sel bc := by
  intro i R ω t e he R₂ j _ u q p hpo hname x hx hsel
  exact (hrow p.value x t).trans ((((hcust t).1 i e he).1 p.value x hsel).trans
    (congrArg some (hbc j e.1 u q p hpo hname x hx).symm))

end FlockC3.Compose
