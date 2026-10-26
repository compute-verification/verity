import Proofs.Flock.Claims.Recursion.VerdictRows
import Proofs.Flock.Claims.Recursion.DecodeTiesW
import Proofs.Flock.Soundness.Discharge.DecodeZL.Sizes

/-!
# Row C at a decision: the decoded proof's ties

`RowC` asks, at table `i`'s rep `r`, that the rep's file decode to its proof and that every field the rep's frames
absorb but `final_c` be the one its `w` holds (`decoded`). Both follow from the setup's row (`RowG`), the session's
points being the link's (`pointsOf` at the first coin), and the proof being shaped for its file (the `.shaped` check):

* **The shape** (`shapeAt_eq`): the shape the file is written at (`shapeAt`) is the one V\*'s algebra reads it at
  (`DecAt.sh`), so the file decodes to the proof (`decode_rep`, `decode_filesAt`) and the proof has the shape's sizes
  (`decodeProof_sizes`).
* **The rep's context is the table's** (`ctx_st`, `ctx_sch`, `ctx_n`): the setup, schedule and claim count the frames
  are read at are table `i`'s.
* **w starts with the frames' values** (`w_prefix`): `DecAt.w` is the chunks' values, then the accumulators, and the
  chunks' values are the frames' (`chunks_vals`). So w holds the rounds' values (`wHolds_of_frames`), and the
  fields are the decoded ones (`zcProofOf_of`, …). The messages, out-of-domain values and `yr` need two levels or
  more, which a shaped proof's schedule has (`ProofShaped`'s `1 ≤ sh.sched.levels.size - 1`).
-/

namespace RecDeploy.Rows.DecAt

open Flock
open FlockDeploy (FinalP Coin tabsOf pointsOf filesAt chunksOf vals shapeAt decode_filesAt repCtx)
open FlockVBridge.Family (ProofShaped)
open FlockSoundness.Discharge.Exec (repFrames)
open FlockSoundness.Discharge.Recursion (HClass)
open FlockSoundness.Discharge.DecodeZL (ProofSizes decodeProof_sizes)

theorem rep_div_two (i : ℕ) (r : Fin 2) : (2 * i + r) / 2 = i := by
  have := r.isLt
  omega

variable {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {ms : List Claims.Msg} {cs : List Coin}
  (D : DecAt rd x ms cs) (cls : HClass) (acc : ℕ → Array F128) (i : ℕ) (r : Fin 2)

/-- **Table `i`'s file shape is V\*'s**, at extra claims the link's points give. -/
theorem shapeAt_eq (hext : (tabsOf D.sts)[i]!.extra (pointsOf D.spec D.d.c₁) = .ok (D.files cls i).extras) :
    shapeAt (tabsOf D.sts) (pointsOf D.spec D.d.c₁) (2 * i + r) = D.sh cls i := by
  simp only [shapeAt, rep_div_two i r, hext] <;> rfl

/-- **Rep `r`'s file decodes to its proof**, where the proof is shaped for it. -/
theorem decode_rep (hext : (tabsOf D.sts)[i]!.extra (pointsOf D.spec D.d.c₁) = .ok (D.files cls i).extras)
    (hk : 2 * i + r < D.d.f.ps.size)
    (hsh : ProofShaped (shapeAt (tabsOf D.sts) (pointsOf D.spec D.d.c₁) (2 * i + r)) D.d.f.ps[2 * i + r]) :
    decodeProof (D.sh cls i) ((D.files cls i).bytes r) = .ok ((D.files cls i).proof r) := by
  have h := decode_filesAt hk hsh
  rw [D.shapeAt_eq cls i r hext] at h
  show decodeProof (D.sh cls i) (filesAt (tabsOf D.sts) (pointsOf D.spec D.d.c₁) D.d.f.ps)[2 * i + r]! =
    .ok D.d.f.ps[2 * i + r]!
  rw [getElem!_pos D.d.f.ps _ hk, getElem!_pos (filesAt (tabsOf D.sts) (pointsOf D.spec D.d.c₁) D.d.f.ps) _
    (by simpa only [filesAt, Array.size_mapIdx] using hk)]
  exact h

theorem ctx_st (hG : D.RowG cls acc i) : (D.ctx (2 * i + r)).st = (tabsOf D.sts)[i]! := by
  simp only [DecAt.ctx, repCtx, rep_div_two i r, ← hG.sts] <;> rfl

theorem ctx_sch (hG : D.RowG cls acc i) : (D.ctx (2 * i + r)).sch = (D.sh cls i).sched := by
  simp only [DecAt.ctx, repCtx, rep_div_two i r, ← hG.sts] <;> rfl

theorem ctx_n (hG : D.RowG cls acc i)
    (hext : (tabsOf D.sts)[i]!.extra (pointsOf D.spec D.d.c₁) = .ok (D.files cls i).extras) :
    (D.ctx (2 * i + r)).n = (D.sh cls i).claims := by
  simp only [DecAt.ctx, repCtx, rep_div_two i r, ← hG.sts, ← hG.spec, hext] <;> rfl

/-- **Rep `k`'s w starts with its frames' values**: the chunks' values, in order, are the frames'. -/
theorem w_prefix (k : ℕ) (a : Array F128) :
    vals (repFrames (D.ctx k).st (D.ctx k).sch (D.ctx k).dom (D.ctx k).n D.d.f.ps[k]! (D.T k)) <+: (D.w k a).toList :=
  ⟨a.toList, by rw [DecAt.w, Array.toList_append, List.toList_toArray, chunksOf, FlockDecodeTies.chunks_vals]⟩

/-- **Row C**: at the setup's row, where the session's points are the link's and rep `r`'s proof is shaped for its
file, the file decodes to the proof and the frames absorb the decoded proof's fields. -/
theorem rowC_of (hG : D.RowG cls acc i) (hpts : D.sess.points = pointsOf D.spec D.d.c₁)
    (hk : 2 * i + r < D.d.f.ps.size)
    (hsh : ProofShaped (shapeAt (tabsOf D.sts) (pointsOf D.spec D.d.c₁) (2 * i + r)) D.d.f.ps[2 * i + r]) :
    D.RowC cls acc i r := by
  have hext : (tabsOf D.sts)[i]!.extra (pointsOf D.spec D.d.c₁) = .ok (D.files cls i).extras := by
    rw [← hpts]
    exact hG.extra
  have hdec := D.decode_rep cls i r hext hk hsh
  have hp : ProofSizes (D.sh cls i) ((D.files cls i).proof r) := decodeProof_sizes _ _ _ hdec
  have hlv : 1 ≤ (D.sh cls i).sched.levels.size - 1 := by
    have h := hsh.2.1
    rwa [D.shapeAt_eq cls i r hext] at h
  obtain ⟨n, hR⟩ : ∃ n, (D.sh cls i).sched.levels.size - 1 = n + 1 := ⟨_, (Nat.sub_add_cancel hlv).symm⟩
  have hw : FlockDecodeTies.WHolds (D.sh cls i) (D.rep cls acc i r).w ((D.files cls i).proof r) (D.T (2 * i + r)) :=
    FlockDecodeTies.wHolds_of_frames (st := (D.ctx (2 * i + r)).st) (sch := (D.ctx (2 * i + r)).sch)
      (n := (D.ctx (2 * i + r)).n) (dom := (D.ctx (2 * i + r)).dom)
      (by rw [D.ctx_st cls acc i r hG] <;> rfl) (by rw [D.ctx_st cls acc i r hG] <;> exact hG.foldKLog)
      (D.ctx_n cls acc i r hG hext) (D.ctx_sch cls acc i r hG) rfl hp
      (D.w_prefix (2 * i + r) (acc (2 * i + r)))
  have hzc := FlockDecodeTies.zcProofOf_of (comb := (D.rep cls acc i r).comb) (cons := (D.rep cls acc i r).cons)
    hw hp rfl
  exact
    { decode := hdec
      ab := hzc.ab
      c := hzc.c
      rounds := hzc.rounds
      fa := hzc.fa
      fb := hzc.fb
      lincheck := FlockDecodeTies.lcProofOf_of hw hp
      ringSwitches := FlockDecodeTies.rs_of hw hp
      messages := FlockDecodeTies.msgs_of hw hp hR
      oodValues := FlockDecodeTies.ood_of hw hp hR
      yr := FlockDecodeTies.yr_of hw hp hR }

end RecDeploy.Rows.DecAt
