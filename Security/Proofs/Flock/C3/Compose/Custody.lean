import Proofs.Flock.C3.PubRows.Honest

/-!
# n1c: the verifier's record of the inner session is the live history's

`HeldIs` (`PubRows/Honest.lean`, the record–history link) asks that the rows the verifier holds be the history's
commitments at every history where a session accepts. The verifier holds one record of the inner session, the coin
server's (`--rec-inner-transcript` and `--rec-inner-session`, the guarantee's `tr` and `record`), while the history
ranges over the recursive game's. No check of the verifier's ties the two, the event constrains the record
(`AcceptsC3`) and not the history, and `Flock.Premises.RecordCustody` is V*'s sessions' records only. What ties them is
custody of the inner session's record, `live-verifier` at the inner session, stated here at the coin server's writer of
it:

* `InnerRecordCustody`: at every history, the record the coin server writes there holds, at every row of a value that
  round `i` fixes (`sel`), the `b ‖ c` of the history's round-`i` commitment (`bcv`);
* `heldIs_of_custody`: `HeldIs` at a family whose rows are the ones the verifier holds of the history's record, from the
  custody and the commit strings' bytes being the commitment's at the port's value and row (`hbc`).

The guarantee would read the record at the history (`AcceptsC3 ish st (wr t).1 (wr t).2 reg` in the event, the writer
`wr` in place of `tr` and `record`) and take the custody as a premise; no premise of it states the custody yet.
-/

namespace FlockC3.Compose

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockC3.PubRows

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

/-- **The coin server's record of the inner session is the live history's** (`live-verifier` at the inner session):
at every history `t`, the inner transcript and session record the coin server writes there (`wr t`), as the verifier
holds them against R's registration `reg` (`innerCommitsAt`), hold at every row `x` of a value `v` that round `i` fixes
(`sel i v x`) the `b ‖ c` of `t`'s round-`i` commitment (`bcv`). -/
def InnerRecordCustody (wr : List (Cm × (holoInnerAt T CT st hp).Coin) → Lean.Json × String) (reg : Lean.Json)
    (sparse : Option RecSparse.Shape) (sel : ℕ → String → ℕ → Prop) (bcv : Cm → String → ℕ → List UInt8) : Prop :=
  ∀ i t (e : Cm × (holoInnerAt T CT st hp).Coin), t[i]? = some e → ∀ v x, sel i v x →
    heldRow (innerCommitsAt (wr t).1 (wr t).2 st reg) sparse v x = some (bcv e.1 v x)

/-- **(n1c) The rows the verifier holds are the history's commitments**, at a family whose rows are the ones it holds of
the history's record, from the record's custody. -/
theorem heldIs_of_custody (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂)
    (own : Fin (vstarCount F.ish) → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop)
    (bc : (j : Fin (vstarCount F.ish)) → Cm → PosK (F.Vs j).c (F.Vs j).hU (F.Vs j).nv → List UInt8)
    (wr : List (Cm × (holoInnerAt T CT st hp).Coin) → Lean.Json × String) (reg : Lean.Json)
    (bcv : Cm → String → ℕ → List UInt8)
    (hrow : ∀ v x t, F.row v x t = heldRow (innerCommitsAt (wr t).1 (wr t).2 st reg) F.ish.sparse v x)
    (hcust : InnerRecordCustody wr reg F.ish.sparse sel bcv)
    (hbc : ∀ j c (u : Fin (F.Vs j).nv) (q : Fin (F.Vs j).c.ports.size), ∀ p ∈ own j,
      p.name = (F.Vs j).c.ports[q].name → ∀ x, p.reads[u.val]? = some (some x) →
        bc j c (.inl (u, q)) = bcv c p.value x) :
    HeldIs F own sel bc := by
  intro i R ω t e he R₂ j _ u q p hpo hname x hx hsel
  rw [hrow, hcust i t e he p.value x hsel, hbc j e.1 u q p hpo hname x hx]

end FlockC3.Compose
