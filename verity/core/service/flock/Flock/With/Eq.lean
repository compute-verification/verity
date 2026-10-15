/-
The oracle copies under the deployed hashes: run in `V` with `Hashes.ofSha512 Sha512.hash`, each `…With` is the compiled
verifier's function it copies, so `verifyWith Sha512.hash` is `verify` for every session whose tables' Merkle schemes are the
registry's (`MerkleScheme.Named`).
-/
import Flock.With.Transcript

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

end With

end Flock
