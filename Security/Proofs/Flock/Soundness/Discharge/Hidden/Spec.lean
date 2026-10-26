import Proofs.Flock.Soundness.Discharge.Hidden.Defs
import Proofs.Flock.Soundness.Discharge.Exec.Setup
import Proofs.Flock.Soundness.Discharge.Exec.Merkle

/-!
# What `Stmt.setupHidden` checked and produced (`Spec`)

`setupHidden_spec` walks `Stmt.setupHidden` as `setupH_spec` walks `Stmt.setupH`: every return is the statement its
checks built. `Facts.lean` reads off it what [wave2-hidden-exec] reads.
-/

open Flock FlockLevel3
open Lean (Json)

namespace FlockSoundness.Discharge.Hidden

open Refine

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Json} {program : Option ByteArray} {st : Stmt}

theorem klog_of {k m : ℕ} (h : ¬ (decide (k > 27) || decide (m > 35)) = true) : k ≤ 27 ∧ m ≤ 35 := by
  simp only [Bool.or_eq_true, decide_eq_true_eq, not_or] at h
  omega

/-! ## A statement without the verifier's public inputs names no public input port -/

theorem checkOwn_none {pi : HmIn.Ports} {pop : Public} (h : HmIn.checkOwn pi pop none = .ok ()) :
    pi.ports.isEmpty = true := by
  unfold HmIn.checkOwn at h
  cases he : pi.ports.isEmpty
  · simp only [he] at h; cases h
  · rfl

theorem checkReads_noPorts {c : Circuit} {pi : HmIn.Ports} {n : ℕ} {t : Option Tables}
    (he : pi.ports.isEmpty = true) : HmIn.checkReads c pi n t = .ok () := by
  unfold HmIn.checkReads
  simp only [he, ↓reduceIte]
  rfl

theorem loadPublic_noPorts {c : Circuit} {pi : HmIn.Ports} {f : ByteArray} {pop : Public}
    (he : pi.ports.isEmpty = true) (h : HmIn.loadPublic tags c pi f = .ok pop) : HmOut.loadPublic tags c f = .ok pop := by
  unfold HmIn.loadPublic at h
  unfold HmOut.loadPublic
  simp only [he, ↓reduceIte, checkReads_noPorts he] at h
  cases hL : HmRow.loadPublic tags { c with outPorts := #[] } f with
  | error e => rw [hL] at h; cases h
  | ok pub =>
    rw [hL] at h
    simp only [bind, Except.bind] at h ⊢
    revert h
    cases pub.header.getObjVal? "public_inputs" <;>
      simp only [pure, Except.pure, throw, throwThe, MonadExceptOf.throw, Except.bind, imp_self, reduceCtorEq,
        false_implies]

theorem drawn_noPorts {c : Circuit} {pi : HmIn.Ports} {pop : Public} {d : Json} (he : pi.ports.isEmpty = true) :
    HmIn.drawn c pi pop d = HmOut.drawn c pop d := by
  simp only [HmIn.drawn, HmOut.drawn, he, ↓reduceIte]

theorem delta_noPorts {c : Circuit} {pi : HmIn.Ports} {pin : ℕ} (he : pi.ports.isEmpty = true) :
    HmIn.delta c pi pin = HmOut.delta c pin := by
  simp only [HmIn.delta, he, ↓reduceIte]

theorem regions_noPorts {c : Circuit} {pi : HmIn.Ports} {pub : Public} {r : Array Region}
    (he : pi.ports.isEmpty = true) (h : HmIn.regions c pi pub = .ok r) : r = #[] := by
  unfold HmIn.regions at h
  simp only [he, ↓reduceIte, pure, Except.pure, Except.ok.injEq] at h
  exact h.symm

set_option maxHeartbeats 4000000 in
/-- **What `Stmt.setupHidden` accepted** is its `Spec`. -/
theorem setupHidden_spec' (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    Spec tags cf pf coins tables draw st := by
  cases draw <;> unfold Stmt.setupHidden at h <;> dsimp only at h <;> simp only [throw_bind', pure_bind] at h <;>
    repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    have hk := klog_of ‹¬ (_ || _) = true›
    have he := checkOwn_none ‹HmIn.checkOwn _ _ none = _›
    have := loadPublic_noPorts he ‹HmIn.loadPublic _ _ _ _ = _›
    obtain rfl := regions_noPorts he ‹HmIn.regions _ _ _ = _›
    exact {
      tagsEq := rfl
      leaf := rfl
      parse := by assumption
      merkle := by assumption
      pub := by
        refine ⟨_, this, fun hd => ?_, fun d' hd => ?_⟩
        · first | exact absurd hd (Option.some_ne_none _) | rfl
        · first | exact absurd hd.symm (Option.some_ne_none _) | (cases hd; rw [← drawn_noPorts he]; assumption)
      checkPublic := by assumption
      n := rfl
      blocksReal := rfl
      nbl := rfl
      m := rfl
      kLog := hk.1
      m35 := hk.2
      mPts := rfl
      pin := by assumption
      da := congrArg Prod.fst (delta_noPorts he)
      db := congrArg Prod.snd (delta_noPorts he)
      regions := by rw [Array.append_empty]; assumption
      inRange := by assumption
      sigma := rfl
      hello := by assumption }

end FlockSoundness.Discharge.Hidden
