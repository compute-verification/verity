import Proofs.Flock.Claims.Complete.ZkTables.Honest
import Proofs.Flock.Claims.Complete.Zk.Coins
import Proofs.Flock.Claims.Complete.Runs

/-!
# The masked verifier's error at any number of tables

`ready` and `δc` (`Complete.Ready`, `deployZk`'s error): `readyZk` with its one-table condition dropped, at the honest
masked prover on each table's share (`RecordFitsAt`); `δc H` is `εZk` where `ready` holds and `1` elsewhere. `εZk`
already counts every table's reps (`outerSlotsZk`, `betaSlotsZk`, `rhoSlotsZk` run over `2·|tables|` reps), so `δc H` is
the masked verifier's union bound at every number of tables.

* **The regions** (`RegionsFit`, the plain `ready`'s): where some region's bytes at some block are not its free
  bits, the extra claims refuse (`Stmt.regionValue`), the honest prover's as the verifier's, so the honest masked
  session cannot run; where they fit, the extra claims return (`JsonRT.extraClaims_ok`).
* **The record** (`RecordFitsAt`): at some witness, tape and coins the honest masked session's printed record may be
  `2^64` bytes or more, past the last message's 64-bit word for its length.
* `complete_of`: if the honest masked prover (`honestP`) is accepted at every coin outside `BadZk`, on every statement
  in scope, ready, with a witness in the relation, it is complete with error `δc H`: `Claims.Flock.Complete H
  (deployZk rel)`, given the mask slots' rank bound (`MaskRank.MaskRankBound`, open).
* **Against `readyZk`**: a setup with several tables is never `readyZk` (`readyZk_tables`), so `δcZk H` is `1` there;
  where `readyZk` holds, `ready` holds (`ready_of_readyZk`, `Complete.Ready`: at one table the honest record is
  `honestPZk`'s), and there `δc H` is `δcZk H` (`δc_eq_δcZk`).
-/

namespace FlockComplete.ZkTables

open Flock FlockDeploy

/-- **Completeness from acceptance at the good coins**, at any number of tables: if, on every statement in scope that
the masked verifier can accept and every witness in the relation, the honest masked prover is accepted at every tape and
every coin outside `BadZk`, then Flock's masked verifier is complete for `honestP` with error `δc H`. -/
theorem complete_of (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (hMR : MaskRank.MaskRankBound)
    (hacc : ∀ x w, Claims.Flock.Scope (deployZk rel) x → Claims.Flock.R (deployZk rel).parse x w →
      ready H x = true → ∀ (ω : TapeZk) (r : Fin ((V true).k x) → Coin), ¬ BadZk x r →
        (Claims.run H (honestP w) (V true) x ω r).out = .acc) :
    Claims.Flock.Complete H (deployZk rel) := by
  intro _ x w hs hr ω
  show 1 - δc H x ≤ _
  unfold δc
  split
  · rename_i hready
    calc 1 - εZk x ≤ 1 - Claims.Pr (fun r : Fin ((V true).k x) → Coin => BadZk x r) := by
          linarith [pr_badZk_le hMR x (rankSlotsZk_layout hs) (k := (V true).k x)]
      _ = Claims.Pr (fun r : Fin ((V true).k x) → Coin => ¬ BadZk x r) := (Pr_not _).symm
      _ ≤ Claims.Pr (fun r : Fin ((V true).k x) → Coin =>
            (Claims.run H (honestP w) (V true) x ω r).out = .acc) :=
          Claims.Pr_mono fun r hr' => hacc x w hs hr hready ω r hr'
  · rw [sub_self]
    exact Claims.Pr_nonneg _

/-- **A setup with several tables is never `readyZk`**, so `δcZk H` is `1` there. -/
theorem readyZk_tables {H : Claims.Msg → Claims.Msg} {x : Claims.Msg} {s : Ins} {spec : SessionSpec}
    {sts : Array Flock.Stmt} (hs : setupOf true x = some (s, spec, sts)) (h : 1 < sts.size) : readyZk H x = false := by
  unfold readyZk
  rw [hs]
  have : (sts.size == 1) = false := by simp; omega
  simp [this]

end FlockComplete.ZkTables
