import Proofs.Flock.Soundness.Refine.RepConverse
import Proofs.Flock.Claims.Recursion.NextShaDec
import Flock.With.Eq

/-!
# A rep's verdict on its decoded proof, from its program's run

`hpieces` (`VerdictParts`) asks `verifyJPWith` to accept at `std`, the hashes `ofSha512 (hOf sha512)` are. Each table's
reps run through `verifyRepDWith`, which reads no proof file. These are the per-rep steps of that verdict, at a table
whose Merkle scheme is the registry's (`Named`):

* `verifyRepPWith_std_of_prog`: `verifyRepPWith` returns the proof's cap where its PCS parameters are the pinned ones,
  its program (`Refine.repProg`) returns from the stream's replay transcript having consumed every recorded round, and
  the cap is the committed `root_B`: `Refine.verifyRep_of_checks` with no decode, through `With/Eq`'s `_std` lemmas.
* `verifyRepDWith_std_of_prog`: the same, from the record's rep stream (S11) and the statement's extra claims.

`FlockVBridge.Family.runs_of_files` gives the program's run from the pieces (`RunsFrom`, `AlgebraHolds`, `AccHolds`).
-/

namespace RecDeploy

open Flock

theorem verifyRepPWith_std_of_prog (spec : SessionSpec) (st : Setup) (hN : st.merkle.Named) (sch : Flock.Schedule)
    (sess : Session) (i r : ℕ) (stream : StreamRec) (extras : Array Claim) (p : Proof) (t₄ : Transcript)
    (hpar : (p.params != pcsParams st.merkle sch) = false)
    (hprog : FlockSoundness.Refine.repProg st sch p extras
      (Transcript.init stream.rounds (spec.domain spec.tables[i]! r).toUTF8) = .ok ((), t₄))
    (hk : t₄.k = stream.rounds.size)
    (hcap : (p.cap.foldl (· ++ ·) ByteArray.empty == sess.rootB[i]!) = true) :
    verifyRepPWith With.std spec st sch sess i r stream extras p = .ok (p.cap.foldl (· ++ ·) ByteArray.empty) := by
  have ebind : ∀ {α β : Type} {x : Except String α} {f : α → Except String β} {a : α}, x = .ok a →
      (x >>= f) = f a := by
    intro α β x f a h; rw [h]; rfl
  unfold verifyRepPWith
  simp only [With.bindAndZerocheckWith_std, With.lincheckWith_std, With.ringSwitchWith_std, With.ligeritoWith_std _ hN]
  try dsimp only
  rw [ite_eq_right (by rw [hpar]; exact Bool.false_ne_true)]
  refine Eq.trans (ebind hprog) ?_
  change (if (t₄.k != stream.rounds.size) = true then _ else _) = _
  rw [ite_eq_right (by simp [hk])]
  rw [ite_eq_right (by simp [bne, hcap])]
  rfl

theorem verifyRepDWith_std_of_prog (spec : SessionSpec) (st : Setup) (hN : st.merkle.Named) (sch : Flock.Schedule)
    (sess : Session) (i r : ℕ) (stream : StreamRec) (extras : Array Claim) (p : Proof) (t₄ : Transcript)
    (hs : sess.streams.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString r) =
      some stream)
    (hex : st.extra sess.points = .ok extras)
    (hpar : (p.params != pcsParams st.merkle sch) = false)
    (hprog : FlockSoundness.Refine.repProg st sch p extras
      (Transcript.init stream.rounds (spec.domain spec.tables[i]! r).toUTF8) = .ok ((), t₄))
    (hk : t₄.k = stream.rounds.size)
    (hcap : (p.cap.foldl (· ++ ·) ByteArray.empty == sess.rootB[i]!) = true) :
    FlockDeploy.verifyRepDWith With.std spec st sch sess i r p = .ok (p.cap.foldl (· ++ ·) ByteArray.empty) := by
  have ebind : ∀ {α β : Type} {x : Except String α} {f : α → Except String β} {a : α}, x = .ok a →
      (x >>= f) = f a := by
    intro α β x f a h; rw [h]; rfl
  unfold FlockDeploy.verifyRepDWith
  try simp only [monadLift_self]
  try dsimp only
  rw [hs]
  refine Eq.trans (ebind hex) ?_
  exact verifyRepPWith_std_of_prog spec st hN sch sess i r stream extras p t₄ hpar hprog hk hcap

end RecDeploy
