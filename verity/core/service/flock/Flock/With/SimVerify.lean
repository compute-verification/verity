/-
The `…With` functions of `Flock.With.Hashes`, `Flock.With.Transcript` and `Flock.With.Verify` under a simulation
(`Flock.With.Sim`): run with two oracles that `S` relates hash by hash, each function's two runs are related by `S` (or by
its lift to the transcript and the Ligerito state).
-/
import Flock.With.SimWalk
import Flock.With.Verify

namespace Flock.With

open Transcript

variable {M N : Type → Type} [Monad M] [Monad N] [MonadExceptOf String M] [MonadExceptOf String N] [MonadLiftT V M]
  [MonadLiftT V N]

theorem leafWith_sim (S : Sim M N) {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs')
    (x salt : ByteArray) : S.R (Hm96.Default512.leafWith hs x salt) (Hm96.Default512.leafWith hs' x salt) := by
  unfold Hm96.Default512.leafWith
  sim_walk

theorem opsWith_sim (S : Sim M N) {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs')
    (ms : MerkleScheme) : S.Ops (ms.opsWith hs) (ms.opsWith hs') := by
  unfold MerkleScheme.opsWith
  split
  · split
    · exact ⟨fun l r => hh.sha256 _, fun row _ => hh.sha256 _⟩
    · split
      · exact ⟨fun l r => hh.sha512 _, fun row _ => hh.sha512 _⟩
      · split
        · exact ⟨fun l r => hh.sha512 _, fun row salt => S.bind (hh.sha512 _) fun x => leafWith_sim S hh x salt⟩
        · exact ⟨fun l r => S.pure _, fun row salt => S.pure _⟩
  · exact ⟨fun l r => S.pure _, fun row salt => S.pure _⟩

theorem climbWith_sim (S : Sim M N) {ops : MerkleOps M} {ops' : MerkleOps N} (ho : S.Ops ops ops') (acc : ByteArray)
    (idx : Nat) (sibs : List ByteArray) : S.R (climbWith ops acc idx sibs) (climbWith ops' acc idx sibs) := by
  induction sibs generalizing acc idx with
  | nil => exact S.pure acc
  | cons s rest ih =>
    simp only [climbWith]
    sim_walk [ho.node, ih]

theorem opensWith_sim (S : Sim M N) {ops : MerkleOps M} {ops' : MerkleOps N} (ho : S.Ops ops ops') (ms : MerkleScheme)
    (cap : Array ByteArray) (c d lanes pos : Nat) (row : Array F128) (salt : ByteArray) (sibs : List ByteArray) :
    S.R (opensWith ops ms cap c d lanes pos row salt sibs) (opensWith ops' ms cap c d lanes pos row salt sibs) := by
  unfold opensWith
  sim_walk [ho.leaf, climbWith_sim S ho]

theorem merkleCheckWith_sim (S : Sim M N) {ops : MerkleOps M} {ops' : MerkleOps N} (ho : S.Ops ops ops')
    (ms : MerkleScheme) (cap : Array ByteArray) (c d : Nat) (positions : Array Nat) (rows : Array (Array F128))
    (siblings salts : Array ByteArray) (lanes : Nat) :
    S.R (merkleCheckWith ops ms cap c d positions rows siblings salts lanes)
      (merkleCheckWith ops' ms cap c d positions rows siblings salts lanes) := by
  unfold merkleCheckWith
  sim_walk [opensWith_sim S ho]

theorem answersWith_sim (S : Sim M N) {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs') (r : Round)
    (c : ByteArray) : S.R (r.answersWith hs c) (r.answersWith hs' c) := by
  unfold Round.answersWith
  sim_walk

theorem squeezeWith_sim (S : Sim M N) {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs')
    (kind : UInt8) (n : Nat) : (S.st Transcript).R (squeezeWith hs kind n) (squeezeWith hs' kind n) := by
  intro t
  unfold squeezeWith
  sim_walk [answersWith_sim S hh]

theorem sampleWith_sim (S : Sim M N) {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs') :
    (S.st Transcript).R (sampleWith hs) (sampleWith hs') := by
  unfold sampleWith
  sim_walk [squeezeWith_sim S hh]

theorem sampleVecWith_sim (S : Sim M N) {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs') (n : Nat) :
    (S.st Transcript).R (sampleVecWith hs n) (sampleVecWith hs' n) :=
  squeezeWith_sim S hh 2 n

theorem sampleKWith_sim (S : Sim M N) {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs') :
    (S.st Transcript).R (sampleKWith hs) (sampleKWith hs') := by
  unfold sampleKWith
  sim_walk [squeezeWith_sim S hh]

theorem dgWith_sim (S : Sim M N) {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs') (spec : SessionSpec)
    (b : ByteArray) : S.R (spec.dgWith hs b) (spec.dgWith hs' b) := by
  unfold SessionSpec.dgWith
  sim_walk

theorem decodeWith_sim (S : Sim M N) [LawfulMonad M] [LawfulMonad N] {hs : Flock.Hashes M} {hs' : Flock.Hashes N}
    (hh : S.Hashes hs hs') (spec : SessionSpec) (text : String) :
    S.R (Record.decodeWith hs spec text) (Record.decodeWith hs' spec text) := by
  unfold Record.decodeWith
  sim_walk [dgWith_sim S hh]

theorem bindAndZerocheckWith_sim (S : Sim M N) {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs')
    (m : Nat) (digest cap : ByteArray) (p : ZerocheckProof) :
    (S.st Transcript).R (bindAndZerocheckWith hs m digest cap p) (bindAndZerocheckWith hs' m digest cap p) := by
  unfold bindAndZerocheckWith
  sim_walk [sampleWith_sim S hh, sampleVecWith_sim S hh]

theorem lincheckWith_sim (S : Sim M N) {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs')
    (c : CircuitFold) (zc : ZcOut) (p : LincheckProof) :
    (S.st Transcript).R (lincheckWith hs c zc p) (lincheckWith hs' c zc p) := by
  unfold lincheckWith
  sim_walk [sampleWith_sim S hh]

theorem ringSwitchWith_sim (S : Sim M N) {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs')
    (m : Nat) (claims : Array Claim) (rs : Array (Array F128)) :
    (S.st Transcript).R (ringSwitchWith hs m claims rs) (ringSwitchWith hs' m claims rs) := by
  unfold ringSwitchWith
  sim_walk [sampleVecWith_sim S hh]

theorem liftW_sim (S : Sim M N) {α : Type} {x : TMW M α} {y : TMW N α} (h : (S.st Transcript).R x y) :
    ((S.st Transcript).st LState).R (liftW x) (liftW y) :=
  (S.st Transcript).lift_st h

theorem liftLM_sim (S : Sim M N) {α : Type} (x : LM α) : ((S.st Transcript).st LState).R (liftLM x) (liftLM x) :=
  fun s t => S.lift (x s t)

theorem challengeWith_sim (S : Sim M N) {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs')
    (p : LigeritoProof) : ((S.st Transcript).st LState).R (challengeWith hs p) (challengeWith hs' p) := by
  unfold challengeWith
  sim_walk [liftW_sim, liftLM_sim, sampleKWith_sim S hh]

theorem oodIntroWith_sim (S : Sim M N) {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs')
    (p : LigeritoProof) (dim split : Nat) :
    ((S.st Transcript).st LState).R (oodIntroWith hs p dim split) (oodIntroWith hs' p dim split) := by
  unfold oodIntroWith
  sim_walk [liftW_sim, liftLM_sim, sampleWith_sim S hh, sampleVecWith_sim S hh]

theorem queryLevelWith_sim (S : Sim M N) {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs')
    (ms : MerkleScheme) (p : LigeritoProof) (lv : Level) (cap : Array ByteArray) (rows : Array (Array F128))
    (sibs salts : Array ByteArray) (fold : Array F256) :
    ((S.st Transcript).st LState).R (queryLevelWith hs ms p lv cap rows sibs salts fold)
      (queryLevelWith hs' ms p lv cap rows sibs salts fold) := by
  unfold queryLevelWith
  sim_walk [liftW_sim, liftLM_sim, sampleVecWith_sim S hh, merkleCheckWith_sim S (opsWith_sim S hh ms)]

theorem ligeritoWith_sim (S : Sim M N) {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs')
    (ms : MerkleScheme) (sch : Schedule) (T : F128) (cap : Array ByteArray) (p : LigeritoProof)
    (basisAt : Array F256 → Nat → Array F256) :
    (S.st Transcript).R (ligeritoWith hs ms sch T cap p basisAt) (ligeritoWith hs' ms sch T cap p basisAt) := by
  unfold ligeritoWith
  sim_walk [liftW_sim, liftLM_sim, sampleWith_sim S hh, sampleVecWith_sim S hh, challengeWith_sim S hh,
    oodIntroWith_sim S hh, queryLevelWith_sim S hh]

theorem verifyRepWith_sim (S : Sim M N) {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs')
    (spec : SessionSpec) (st : Setup) (sch : Schedule) (sess : Session) (i r : Nat) (bytes : ByteArray) :
    S.R (verifyRepWith hs spec st sch sess i r bytes) (verifyRepWith hs' spec st sch sess i r bytes) := by
  unfold verifyRepWith
  sim_walk [bindAndZerocheckWith_sim S hh, lincheckWith_sim S hh, ringSwitchWith_sim S hh, ligeritoWith_sim S hh]

theorem verifyWithM_sim (S : Sim M N) [LawfulMonad M] [LawfulMonad N] {hs : Flock.Hashes M} {hs' : Flock.Hashes N}
    (hh : S.Hashes hs hs') (spec : SessionSpec) (tabs : Array Setup) (record : String) (proofs : Array ByteArray) :
    S.R (verifyWithM hs spec tabs record proofs) (verifyWithM hs' spec tabs record proofs) := by
  unfold verifyWithM
  sim_walk [decodeWith_sim S hh, dgWith_sim S hh, verifyRepWith_sim S hh]

end Flock.With
