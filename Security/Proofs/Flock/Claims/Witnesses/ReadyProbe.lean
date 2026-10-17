import Proofs.Flock.Claims.Complete.Coins
import Proofs.Flock.Claims.Prob

/-!
# Probe: where the masked record's completeness says nothing

`deployZk`'s `δc` is `FlockComplete.δcZk`, which is `1` wherever `readyZk` fails, and `readyZk` asks for one table
(`sts.size == 1`). So at every masked statement whose setup has other than one table, `Claims.Flock.Complete` at
`deployZk` asks `Pr[acc] ≥ 0`, which every prover meets: there the record says nothing of the masked honest prover.
`Complete_meets_zk` is met at one table only.

* `δcZk_tables`: `(deployZk rel).δc H x = 1` at a masked statement whose setup has other than one table.
* `Complete_zk_tables_any`: there `1 - δc` is below every probability.
-/

namespace FlockComplete

open Flock FlockDeploy

theorem δcZk_tables (rel : Readable.Parsed → Readable.Relation) (H : Claims.Msg → Claims.Msg) {x : Claims.Msg}
    {s : Ins} {spec : SessionSpec} {sts : Array _root_.Flock.Stmt} (hs : setupOf true x = some (s, spec, sts))
    (h : sts.size ≠ 1) : (deployZk rel).δc H x = 1 := by
  have hr : readyZk H x ≠ true := by
    intro h'
    unfold readyZk at h'
    rw [hs] at h'
    simp only [Bool.and_eq_true] at h'
    exact h (beq_iff_eq.1 h'.1.1.1.1.1.1)
  show δcZk H x = 1
  unfold δcZk
  rw [if_neg hr]

theorem Complete_zk_tables_any (rel : Readable.Parsed → Readable.Relation) (H : Claims.Msg → Claims.Msg)
    {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hs : setupOf true x = some (s, spec, sts)) (h : sts.size ≠ 1) {α : Type} [Fintype α] (p : α → Prop) :
    1 - (deployZk rel).δc H x ≤ Claims.Pr p := by
  rw [δcZk_tables rel H hs h, sub_self]
  exact Claims.Pr_nonneg p

end FlockComplete
