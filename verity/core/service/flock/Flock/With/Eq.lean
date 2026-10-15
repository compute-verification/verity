/-
The oracle copies under the deployed hashes: run in `V` with `Hashes.ofSha512 Sha512.hash`, each `…With` is the compiled
verifier's function it copies, so `verifyWith Sha512.hash` is `verify` for every session whose tables' Merkle schemes are the
registry's (`MerkleScheme.Named`).
-/
import Flock.With.Verify

namespace Flock

open Transcript

namespace With

/-- The deployed hashes. -/
abbrev std : Hashes V := Hashes.ofSha512 Sha512.hash

theorem liftTM_V {α : Type} (x : TM α) : liftTM (M := V) x = x := rfl

theorem ofScheme_node {M : Type → Type} [Monad M] (ms : MerkleScheme) (l r : ByteArray) :
    (MerkleOps.ofScheme (M := M) ms).node l r = pure (ms.node l r) := rfl

theorem ofScheme_leaf {M : Type → Type} [Monad M] (ms : MerkleScheme) (row : Array F128) (salt : ByteArray) :
    (MerkleOps.ofScheme (M := M) ms).leaf row salt = pure (ms.leaf row salt) := rfl

theorem ofScheme_climb {M : Type → Type} [Monad M] [LawfulMonad M] (ms : MerkleScheme) (acc : ByteArray) (idx : Nat)
    (sibs : List ByteArray) : climbWith (MerkleOps.ofScheme (M := M) ms) acc idx sibs = pure (climb ms acc idx sibs) := by
  induction sibs generalizing acc idx with
  | nil => rfl
  | cons s rest ih =>
    simp only [climbWith, climb]
    split <;> simp only [ofScheme_node, pure_bind, ih]

theorem ofScheme_opens {M : Type → Type} [Monad M] [LawfulMonad M] (ms : MerkleScheme) (cap : Array ByteArray)
    (c d lanes pos : Nat) (row : Array F128) (salt : ByteArray) (sibs : List ByteArray) :
    opensWith (MerkleOps.ofScheme (M := M) ms) ms cap c d lanes pos row salt sibs
      = pure (opens ms cap c d lanes pos row salt sibs) := by
  unfold opensWith opens
  split
  · next h => simp only [ofScheme_leaf, pure_bind, ofScheme_climb, h, Bool.true_and]
  · next h => simp only [Bool.not_eq_true] at h; simp only [h, Bool.false_and]

theorem ofScheme_merkleCheck (ms : MerkleScheme) (cap : Array ByteArray) (c d : Nat) (positions : Array Nat)
    (rows : Array (Array F128)) (siblings : Array ByteArray) (salts : Array ByteArray) (lanes : Nat) :
    merkleCheckWith (MerkleOps.ofScheme (M := V) ms) ms cap c d positions rows siblings salts lanes
      = merkleCheck ms cap c d positions rows siblings salts lanes := by
  unfold merkleCheckWith merkleCheck
  simp only [ofScheme_opens, pure_bind, List.findM?_pure]
  rfl

theorem named_default : MerkleScheme.Named default := by
  intro s h
  have e : MerkleScheme.find? (default : MerkleScheme).id = none := by decide
  rw [e] at h
  cases h

theorem opsWith_std (ms : MerkleScheme) (h : ms.Named) : ms.opsWith std = MerkleOps.ofScheme ms := by
  unfold MerkleScheme.opsWith
  cases hf : MerkleScheme.find? ms.id with
  | none => rfl
  | some s =>
    obtain rfl := h s hf
    have hm := List.mem_of_find?_eq_some hf
    simp only [MerkleScheme.all, List.mem_cons, List.not_mem_nil, or_false] at hm
    rcases hm with rfl | rfl | rfl | rfl <;> rfl

theorem std_sha512 (b : ByteArray) : std.sha512 b = pure (Sha512.hash b) := rfl

theorem std_sha256 (b : ByteArray) : std.sha256 b = pure (Sha256.hash b) := rfl

theorem answersWith_std (r : Round) (c : ByteArray) : r.answersWith std c = pure (decide (r.answers c)) := by
  unfold Round.answersWith
  split
  · next b h => exact congrArg pure (decide_eq_decide.mpr (by simp only [Round.answers, h]))
  · next h =>
    dsimp only
    split
    · next h64 =>
      simp only [std_sha512, pure_bind]
      exact congrArg pure (decide_eq_decide.mpr (by simp only [Round.answers, h, h64, ↓reduceIte]))
    · next h64 =>
      simp only [std_sha256, pure_bind]
      exact congrArg pure (decide_eq_decide.mpr (by simp only [Round.answers, h, h64, ↓reduceIte]))

theorem squeezeWith_std (kind : UInt8) (n : Nat) : squeezeWith std kind n = squeeze kind n := by
  funext t
  unfold squeezeWith squeeze
  dsimp only
  by_cases h0 : n = 0
  · simp only [h0, ↓reduceIte]; rfl
  · simp only [h0, ↓reduceIte]
    cases t.rounds[t.k]? with
    | none => rfl
    | some r =>
      dsimp only
      split
      · rfl
      · rw [answersWith_std, pure_bind]
        by_cases ha : r.answers (t.framed kind n)
        · simp only [ha, decide_true, Bool.not_true, Bool.false_eq_true, not_true_eq_false, ↓reduceIte]; rfl
        · simp only [ha, decide_false, Bool.not_false, not_false_eq_true, ↓reduceIte]; rfl

theorem sampleWith_std : sampleWith std = sample := by
  unfold sampleWith sample; rw [squeezeWith_std]

theorem sampleVecWith_std (n : Nat) : sampleVecWith std n = sampleVec n := by
  unfold sampleVecWith sampleVec; rw [squeezeWith_std]

theorem sampleKWith_std : sampleKWith std = sampleK := by
  unfold sampleKWith sampleK; rw [squeezeWith_std]

theorem dgWith_std (spec : SessionSpec) (b : ByteArray) : spec.dgWith std b = pure (spec.dg b) := by
  unfold SessionSpec.dgWith SessionSpec.dg
  split <;> rfl

theorem decodeWith_std (spec : SessionSpec) (text : String) : Record.decodeWith std spec text = Record.decode spec text := by
  unfold Record.decodeWith Record.decode
  simp only [monadLift_self, dgWith_std, pure_bind]
  rfl

theorem liftW_V {α : Type} (x : TMW V α) : liftW (M := V) x = lift x := rfl
theorem liftLM_V {α : Type} (x : LM α) : liftLM (M := V) x = x := rfl

theorem bindAndZerocheckWith_std (m : Nat) (digest cap : ByteArray) (p : ZerocheckProof) :
    bindAndZerocheckWith std m digest cap p = bindAndZerocheck m digest cap p := by
  unfold bindAndZerocheckWith bindAndZerocheck
  simp only [liftTM_V, sampleWith_std, sampleVecWith_std]

theorem lincheckWith_std (c : CircuitFold) (zc : ZcOut) (p : LincheckProof) :
    lincheckWith std c zc p = lincheck c zc p := by
  unfold lincheckWith lincheck
  simp only [liftTM_V, sampleWith_std]
  rfl

theorem ringSwitchWith_std (m : Nat) (claims : Array Claim) (rs : Array (Array F128)) :
    ringSwitchWith std m claims rs = ringSwitch m claims rs := by
  unfold ringSwitchWith ringSwitch
  simp only [liftTM_V, sampleVecWith_std]

theorem challengeWith_std (p : LigeritoProof) : challengeWith std p = challenge p := by
  unfold challengeWith challenge
  simp only [liftLM_V, liftW_V, sampleKWith_std]

theorem oodIntroWith_std (p : LigeritoProof) (dim split : Nat) : oodIntroWith std p dim split = oodIntro p dim split := by
  unfold oodIntroWith oodIntro
  simp only [liftLM_V, liftW_V, sampleWith_std, sampleVecWith_std]

theorem monadLift_LMW_bind {β : Type} (x : V Unit) (f : Unit → LMW V β) :
    ((monadLift x : LMW V Unit) >>= f) = (match x with | .ok () => f () | .error e => (throw e : LMW V Unit) >>= f) := by
  cases x <;> rfl

theorem queryLevelWith_std (ms : MerkleScheme) (h : ms.Named) (p : LigeritoProof) (lv : Level) (cap : Array ByteArray)
    (rows : Array (Array F128)) (sibs salts : Array ByteArray) (fold : Array F256) :
    queryLevelWith std ms p lv cap rows sibs salts fold = queryLevel ms p lv cap rows sibs salts fold := by
  unfold queryLevelWith queryLevel
  simp only [liftLM_V, liftW_V, sampleVecWith_std, opsWith_std ms h, ofScheme_merkleCheck, monadLift_LMW_bind]
  rfl

theorem ligeritoWith_std (ms : MerkleScheme) (h : ms.Named) (sch : Schedule) (T : F128) (cap : Array ByteArray)
    (p : LigeritoProof) (basisAt : Array F256 → Nat → Array F256) :
    ligeritoWith std ms sch T cap p basisAt = ligerito ms sch T cap p basisAt := by
  unfold ligeritoWith ligerito
  simp only [liftTM_V, liftLM_V, liftW_V, sampleWith_std, sampleVecWith_std, challengeWith_std, oodIntroWith_std,
    queryLevelWith_std ms h]
  rfl

theorem verifyRepWith_std (spec : SessionSpec) (st : Setup) (h : st.merkle.Named) (sch : Schedule) (sess : Session)
    (i r : Nat) (bytes : ByteArray) : verifyRepWith std spec st sch sess i r bytes = verifyRep spec st sch sess i r bytes := by
  unfold verifyRepWith verifyRep
  simp only [monadLift_self, bindAndZerocheckWith_std, lincheckWith_std, ringSwitchWith_std, ligeritoWith_std _ h]
  rfl

theorem named_getElem! (tabs : Array Setup) (hN : ∀ st ∈ tabs, st.merkle.Named) (i : Nat) : (tabs[i]!).merkle.Named := by
  by_cases hi : i < tabs.size
  · rw [getElem!_pos tabs i hi]; exact hN _ (Array.getElem_mem hi)
  · rw [getElem!_neg tabs i hi]; exact named_default

end With

/-- `verify` is `verifyWith Sha512.hash` when every table's Merkle scheme is the registry's of its id. -/
theorem verify_eq_verifyWith (spec : SessionSpec) (tabs : Array Setup) (record : String) (proofs : Array ByteArray)
    (hN : ∀ st ∈ tabs, st.merkle.Named) : verify spec tabs record proofs = verifyWith Sha512.hash spec tabs record proofs := by
  have hN' := With.named_getElem! tabs hN
  unfold verifyWith verifyWithM verify
  simp only [monadLift_self, With.decodeWith_std, With.dgWith_std, pure_bind, With.verifyRepWith_std _ _ (hN' _)]
  rfl

end Flock
