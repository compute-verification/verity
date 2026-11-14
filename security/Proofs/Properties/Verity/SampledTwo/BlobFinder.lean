module

public import Proofs.Properties.Verity.SampledTwo.RefsOfSent
public import Proofs.Properties.Verity.SampledTwo.Cut

/-!
# The finder for a play's blobs against given files

`HeldOfAtOr A C B₀` (`RefsOfSent.lean`) concludes, at each accepted play, the held copies or a SHA-512 collision
between the play's blobs and the files `B₀` (`BlobsCollide`). `blobFinder` outputs that collision: on a uniform tape
of the play it hashes the play's blobs and `B₀`'s, and outputs the first pair that differs and has one digest
(`blobPair`). `prob_collide_le` puts the sampled verdict accepting on a play whose blobs collide with `B₀` under its
collision probability, and `cutBound_avoid` leaves those plays out of a cut at that term, the one the join's bound adds
(`joinBound`) for the plays `HeldOfAtOr` leaves out; under `ecr/sha-512` at the finder it is at most `blobBound`.

The vocabulary names no such finder: `fileFinder` and `publicFinder` compare the committed files with the registered
ones, once a strategy; this one compares each play's blobs with `B₀`. Its premise is `ecr/sha-512` at it, at its
calls: the strategy's, the play's blobs' and `B₀`'s.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims
open scoped ENNReal

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

open Classical in
/-- **The first colliding pair between the blobs `F` and the files `F₀`**: the first blob of `F` with some file of
`F₀` that differs from it and has its SHA-512, and the first such file. -/
noncomputable def blobPair (F F₀ : Array ByteArray) : Option (ByteArray × ByteArray) :=
  F.toList.findSome? fun b =>
    (F₀.toList.find? fun b' => decide (b ≠ b' ∧ Flock.Sha512.hash b = Flock.Sha512.hash b')).map fun b' => (b, b')

theorem blobPair_of_collide {F F₀ : Array ByteArray} (h : BlobsCollide F F₀) :
    ∃ p, blobPair F F₀ = some p ∧ p.1 ≠ p.2 ∧ Flock.Sha512.hash p.1 = Flock.Sha512.hash p.2 := by
  classical
  obtain ⟨b, hb, b', hb', hne, hh⟩ := h
  cases hf : blobPair F F₀ with
  | some p =>
    refine ⟨p, rfl, ?_⟩
    unfold blobPair at hf
    obtain ⟨a, -, ha⟩ := List.exists_of_findSome?_eq_some hf
    obtain ⟨a', ha', rfl⟩ := Option.map_eq_some_iff.1 ha
    simpa using List.find?_some ha'
  | none =>
    exfalso
    unfold blobPair at hf
    have h1 := List.findSome?_eq_none_iff.1 hf b (Array.mem_toList_iff.2 hb)
    rw [Option.map_eq_none_iff, List.find?_eq_none] at h1
    exact h1 b' (Array.mem_toList_iff.2 hb') (by simp [hne, hh])

/-- **The blob finder at a strategy and files `F₀`**: on a uniform tape of the play, the first pair of a blob the play
sent and a file of `F₀` that differ and have one SHA-512 (`blobPair`). Its calls are the strategy's, one a blob and one
a file. -/
noncomputable def blobFinder (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (F₀ : Array ByteArray) : Claims.Finder where
  S := Verity.Game.Tape (V.game θ F c) σ
  out t := (blobPair (Verity.Game.Tape.out _ σ t).sent.blobs F₀).map fun p => (p.1.data.toList, p.2.data.toList)
  cost t := V.cost θ F c σ + (Verity.Game.Tape.out _ σ t).sent.blobs.size + F₀.size

/-- **The sampled verdict accepting on a play whose blobs collide with `F₀` is at most the blob finder's collision
probability.** -/
theorem prob_collide_le (V : Vocab d α) (k : Verity.Registration.Composite) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (F₀ : Array ByteArray) :
    Verity.Game.prob (fun o => V.Verdict k .sampled o ∧ BlobsCollide o.sent.blobs F₀) (V.game θ F c) σ ≤
      ENNReal.ofReal ((blobFinder V θ F c σ F₀).Pr Verity.sha512Msg) := by
  rw [Verity.Game.prob_eq_prCoin]
  calc _ ≤ Verity.Game.prCoin (fun t : Verity.Game.Tape (V.game θ F c) σ =>
          ∃ u v, (blobFinder V θ F c σ F₀).out t = some (u, v) ∧ Claims.Collision Verity.sha512Msg u v) := by
        refine Verity.Game.prCoin_mono fun t ht => ?_
        obtain ⟨p, hp, hne, hh⟩ := blobPair_of_collide ht.2
        refine ⟨p.1.data.toList, p.2.data.toList, ?_, Verity.Sampled.Binding.collision_of_valid ⟨hne, hh⟩⟩
        show Option.map _ (blobPair _ F₀) = _
        rw [hp]
        rfl
    _ = _ := Verity.Game.prCoin_eq_ofReal_pr _

/-- **The cut with the plays whose blobs collide with `F₀` left out**: `Good` and no such collision at the cut's terms
plus the blob finder's collision probability. -/
theorem cutBound_avoid {V : Vocab d α} {k : Verity.Registration.Composite} {θ : ByteArray} {F : Array ByteArray}
    {c : ByteArray} {σ : Verity.Game.Strategy (V.game θ F c)} {Good : Verity.Exchange → Prop} {terms : ℝ≥0∞}
    (F₀ : Array ByteArray) (h : CutBound V k θ F c σ Good terms) :
    CutBound V k θ F c σ (fun o => Good o ∧ ¬ BlobsCollide o.sent.blobs F₀)
      (terms + ENNReal.ofReal ((blobFinder V θ F c σ F₀).Pr Verity.sha512Msg)) := by
  unfold CutBound at h ⊢
  calc Verity.Game.prob (fun o => V.Verdict k .sampled o ∧ ¬ (Good o ∧ ¬ BlobsCollide o.sent.blobs F₀))
        (V.game θ F c) σ
      ≤ Verity.Game.prob (fun o => (V.Verdict k .sampled o ∧ ¬ Good o) ∨
          (V.Verdict k .sampled o ∧ BlobsCollide o.sent.blobs F₀)) (V.game θ F c) σ :=
        Verity.Game.prob_mono (fun o ho => (Classical.em (Good o)).elim
          (fun hg => .inr ⟨ho.1, Classical.not_not.1 fun hn => ho.2 ⟨hg, hn⟩⟩)
          (fun hg => .inl ⟨ho.1, hg⟩)) _ σ
    _ ≤ Verity.Game.prob (fun o => V.Verdict k .sampled o ∧ ¬ Good o) (V.game θ F c) σ +
          Verity.Game.prob (fun o => V.Verdict k .sampled o ∧ BlobsCollide o.sent.blobs F₀) (V.game θ F c) σ :=
        Verity.Game.prob_or_le (fun o => V.Verdict k .sampled o ∧ ¬ Good o)
          (fun o => V.Verdict k .sampled o ∧ BlobsCollide o.sent.blobs F₀) (V.game θ F c) σ
    _ ≤ _ := add_le_add h (prob_collide_le V k θ F c σ F₀)

/-- **`ecr/sha-512` at the blob finder as a number**: its expected calls over `2^256`. -/
noncomputable abbrev blobBound (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (F₀ : Array ByteArray) : ℝ≥0∞ :=
  ENNReal.ofReal (Ex (fun t => ((blobFinder V θ F c σ F₀).cost t : ℝ)) / 2 ^ (256 : ℝ))

/-- **The blob finder's collision probability under `ecr/sha-512` at it.** -/
theorem blobPr_le {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray}
    {σ : Verity.Game.Strategy (V.game θ F c)} {F₀ : Array ByteArray}
    (h : Claims.ExpectedCR FlockDeploy.sha512 (blobFinder V θ F c σ F₀)) :
    ENNReal.ofReal ((blobFinder V θ F c σ F₀).Pr Verity.sha512Msg) ≤ blobBound V θ F c σ F₀ := by
  have e : Verity.sha512Msg = FlockDeploy.sha512 := funext fun _ => rfl
  rw [e]
  exact ENNReal.ofReal_le_ofReal h

end Verity.SampledTwo
