import Proofs.Flock.C3.Compose.Family

/-!
# Piece (g): the public rows per value prefix, and the tape

`HPubRows`: G's (`VStarFamily.HPubRows`) at every value prefix `RecVStar.fixedRow` knows, not only `rec-c<k>`: where a
session's port registers a value its unit reads at row `x`, the public file's row there is the `b ‖ c` the inner
transcript holds at that value and row (`row`). The check that establishes it is `checkPubRows` (each `pub.bin` row
compared with `fixedRow`'s), so the obligation is its reading, as `pubCommits_of_pubRows` reads G's.

`HTape`: at every accepted outcome, the inner transcript's steps (V*'s layers zipped with the rounds' coins) are
`holoSteps` of the decoded run: the stages in the chain's order, which is the recorded order the model plays
(`holoTableCL`), so no reordering lemma sits between them.
-/

namespace FlockC3.PubRows

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockC3.Compose

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

/-- **(g) The public file's rows are the transcript's**, at every registered value `fixedRow` knows, in every session
the verifier accepts (where `VCommits` reads them; the check is the verifier's). -/
def HPubRows (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) : Prop :=
  ∀ R ω t R₂ j, (∃ tr, AcceptsZKR (F.zs R ω t R₂ j).I (F.zs R ω t R₂ j).own tr) → ∀ reg ∈ (F.zs R ω t R₂ j).own,
    ∀ (u : Fin (F.Vs j).nv) (q : Fin (F.Vs j).c.ports.size) x, (F.Vs j).c.ports[q].name = reg.name →
      reg.reads[u.val]? = some (some x) → ∀ b, F.row reg.value x t = some b →
        (F.zs R ω t R₂ j).pub.digest u.val q.val (F.Vs j).c.ports.size = ⟨b.toArray⟩

/-- **(g) The inner transcript's steps are the decoded run's, in the chain's order.** -/
def HTape (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ) : Prop :=
  F.AtAccepted dg N k Rw fun R ω t Lb =>
    let u := F.run R ω t Lb
    schedSteps T CT 0 ((F.wmsg Lb).zip (t.map Prod.snd)) =
      holoSteps u.cap₀ u.pts (u.ts 0) (u.ts 1) (u.sp 0) (u.sp 1) u.op u.lg u.os

end FlockC3.PubRows
