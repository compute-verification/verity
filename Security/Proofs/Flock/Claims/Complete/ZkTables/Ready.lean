import Proofs.Flock.Claims.Complete.ZkTables.Honest
import Proofs.Flock.Claims.Complete.Zk.Coins
import Proofs.Flock.Claims.Complete.Runs

/-!
# The masked verifier's error at any number of tables

`readyZk` with its one-table condition dropped and the tables' regions asked to fit (`ready`), and `δc H`, `εZk` where
`ready` holds and `1` elsewhere. `εZk` already counts every table's reps (`outerSlotsZk`, `betaSlotsZk`, `rhoSlotsZk`
run over `2·|tables|` reps), so `δc H` is the masked verifier's union bound at every number of tables.

* **The regions** (`JsonRT.RegionsFit`, the plain `ready`'s): where some region's bytes at some block are not its free
  bits, the extra claims refuse (`Stmt.regionValue`), the honest prover's as the verifier's, so the honest masked
  session cannot run; where they fit, the extra claims return (`JsonRT.extraClaims_ok`).
* `complete_of`: if the honest masked prover (`honestP`) is accepted at every coin outside `BadZk`, on every statement
  in scope, ready, with a witness in the relation, it is complete with error `δc H`.
* **Against `readyZk`**: a setup with several tables is never `readyZk` (`readyZk_tables`), so `δcZk H` is `1` there;
  where `readyZk` holds and the regions fit, `ready` holds (`ready_of_readyZk`) and `δc H` is `δcZk H`
  (`δc_eq_δcZk`).
-/

namespace FlockComplete.ZkTables

open Flock FlockDeploy

open Classical in
/-- **The statements the masked verifier can accept** against `H`, at any number of tables: the session draws a unit,
the statement is under `2^56` bytes, every table's regions fit, its setup against `H` is the one at SHA-512, and its
`Hello` names the keyed coin tree. -/
noncomputable def ready (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) : Bool :=
  match setupOf true x with
  | some (s, spec, sts) => s.read.draw.isSome && decide (x.length < 2 ^ 56) &&
      decide (∀ st ∈ sts, JsonRT.RegionsFit st) && decide (setupWith (Claims.hOf H) s.read = .ok (spec, sts)) &&
      spec.coinTree.isSome
  | none => false

/-- **Flock's `δc` at the masked verifier**, at any number of tables: `εZk` where the verifier can accept, else `1`. -/
noncomputable def δc (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) : ℝ := if ready H x then εZk x else 1

open Classical in
/-- **At SHA-512, `ready` asks only for a drawn session, the statement's length, the regions and the coin tree.** -/
theorem ready_sha512 (x : Claims.Msg) :
    ready sha512 x = match setupOf true x with
      | some (s, spec, sts) => s.read.draw.isSome && decide (x.length < 2 ^ 56) &&
          decide (∀ st ∈ sts, JsonRT.RegionsFit st) && spec.coinTree.isSome
      | none => false := by
  unfold ready
  split
  · rename_i s spec sts hs
    rw [hOf_sha512, setupWith_sha512, Ins.setup_read (setupOf_some hs).2]
    simp
  · rfl

/-- **Completeness from acceptance at the good coins**, at any number of tables: if, on every statement in scope that
the masked verifier can accept and every witness in the relation, the honest masked prover is accepted at every tape and
every coin outside `BadZk`, then Flock's masked verifier is complete for `honestP` with error `δc H`. -/
theorem complete_of (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (hacc : ∀ x w, Claims.Flock.Scope (deployZk rel) x → Claims.Flock.R (deployZk rel).parse x w →
      ready H x = true → ∀ (ω : TapeZk) (r : Fin ((V true).k x) → Coin), ¬ BadZk x r →
        (Claims.run H (honestP w) (V true) x ω r).out = .acc) :
    Claims.Flock.Complete H (deployZk rel) honestP (δc H) := by
  intro x w hs hr ω
  unfold δc
  split
  · rename_i hready
    calc 1 - εZk x ≤ 1 - Claims.Pr (fun r : Fin ((V true).k x) → Coin => BadZk x r) := by
          linarith [pr_badZk_le x (k := (V true).k x)]
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

/-- **Where `readyZk` holds and the regions fit, `ready` holds.** -/
theorem ready_of_readyZk {H : Claims.Msg → Claims.Msg} {x : Claims.Msg} (h : readyZk H x = true)
    (hfit : ∀ s spec sts, setupOf true x = some (s, spec, sts) → ∀ st ∈ sts, JsonRT.RegionsFit st) :
    ready H x = true := by
  unfold readyZk at h
  unfold ready
  split at h
  · rename_i s spec sts hs
    rw [hs]
    have := hfit s spec sts hs
    simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at h
    simp only [Bool.and_eq_true, decide_eq_true_eq]
    exact ⟨⟨⟨⟨h.1.1.1.2, h.1.1.2⟩, this⟩, h.1.2⟩, h.2⟩
  · cases h

/-- **Where `readyZk` holds and the regions fit, `δc H` is `δcZk H`.** -/
theorem δc_eq_δcZk {H : Claims.Msg → Claims.Msg} {x : Claims.Msg} (h : readyZk H x = true)
    (hfit : ∀ s spec sts, setupOf true x = some (s, spec, sts) → ∀ st ∈ sts, JsonRT.RegionsFit st) :
    δc H x = δcZk H x := by
  simp [δc, δcZk, h, ready_of_readyZk h hfit]

end FlockComplete.ZkTables
