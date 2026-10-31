import Definitions.Accounting.Assumptions.RemoteOutside
import Definitions.Accounting.Assumptions.RemoteSketch
import Proofs.Accounting.RemoteHeld

/-!
# The certifiers' observation against the receipts' sketch bound and its derivation

Where nothing but the verifier's messages entered the node over the credited passes of an accepting outcome, the
certifiers' coverage of its uplinks (`Covered`) leaves no device outside the node computing for it then, so a block on
the node or computed on outside it for the node at a tick of a pass was on the node (`onNode_of_covered`). So:

* `SketchBoundAt` and `Covered` give the receipts' sketch bound where the certifiers observed nothing else entering
  (`Quiet` at `Ingress.counted`), at `δ + δ_net` with their count (`IngressCounted`, `sketchBound_of_covered`), and
  `SketchBound` gives `SketchBoundAt` at every record of outside computation (`sketchBoundAt_of_sketchBound`);
* the sketch model's `Present` is `PresentAt`, the same with outside computation allowed, under `Covered` where nothing
  else entered over the model's passes (`present_of_covered`, `presentAt_of_present`).
-/

namespace Accounting.SecurityProofs.Remote

open Accounting.Remote.Held Accounting.Remote.Held.Assumptions

section Trace

variable {Ω : Type} {T : Trace Ω} {O : Outside T} {I : Ingress Ω}

/-- Under `Covered`, a block on the node or computed on outside it for the node at a tick of a credited pass of an
accepting outcome where nothing but the verifier's messages entered was on the node then. -/
theorem onNode_of_covered (hC : Covered T O I) {ω : Ω} (ha : T.accepts ω) (hq : Quiet T I.entered ω) {p t : ℕ}
    (hp : p < T.passes) (h1 : T.tick ω p ≤ t) (h2 : t < T.tick ω (p + 1)) {j : Fin T.blocks}
    (hj : j ∈ T.present ω t ∪ O.blocks ω t) : j ∈ T.present ω t := by
  rcases Finset.mem_union.1 hj with h | h
  · exact h
  · rw [(hC ω ha hq p hp t h1 h2).1] at h
    simp at h

variable [Fintype Ω]

/-- `SketchBoundAt` at `δ`, `Covered` and the certifiers' count at `δnet` give the receipts' sketch bound where the
certifiers observed nothing but the verifier's messages entering, at `δ + δnet`. -/
theorem sketchBound_of_covered {δ δnet : ℝ} (hk : SketchBoundAt T O δ) (hC : Covered T O I)
    (hI : IngressCounted T I δnet) :
    T.Holds (δ + δnet) fun ω => Quiet T I.counted ω →
      ∀ p < T.passes, ∀ j ∈ T.correct ω p, ∃ t, T.tick ω p ≤ t ∧ t < T.tick ω (p + 1) ∧ j ∈ T.present ω t :=
  holdsMono (fun ω ha h hq p hp j hj => by
    obtain ⟨t, h1, h2, ht⟩ := h.1 p hp j hj
    exact ⟨t, h1, h2, onNode_of_covered hC ha (h.2 hq) hp h1 h2 ht⟩) (holdsAnd hk hI)

/-- `SketchBound` gives `SketchBoundAt` at every record of outside computation. -/
theorem sketchBoundAt_of_sketchBound {δ : ℝ} (hk : SketchBound T δ) (O : Outside T) : SketchBoundAt T O δ :=
  holdsMono (fun _ _ h p hp j hj => by
    obtain ⟨t, h1, h2, ht⟩ := h p hp j hj
    exact ⟨t, h1, h2, Finset.mem_union_left _ ht⟩) hk

end Trace

section Sketch

open Accounting.Remote.Sketch Accounting.Remote.Sketch.Assumptions

variable {β Hs : Type} {P N m h S L q ks : ℕ} {G : Fin P → ℕ}

/-- `Present` with outside computation allowed: on every accepting outcome, a block the kept candidate's device of pass
`j` obtained whole was in node memory or on a chip, or computed on by a device outside the node for it, at some tick of
the pass. -/
def PresentAt (A : Session β Hs P N m h S L q ks G) (T : Trace (Outcome β Hs N m)) (O : Outside T) : Prop :=
  ∀ ω, T.accepts ω → ∀ j : Fin P, ∀ t : Fin N, A.Whole ω j (A.kept ω j) t →
    ∀ b : Fin T.blocks, (b : ℕ) = t →
      ∃ τ, T.tick ω (j : ℕ) ≤ τ ∧ τ < T.tick ω ((j : ℕ) + 1) ∧ b ∈ T.present ω τ ∪ O.blocks ω τ

variable {A : Session β Hs P N m h S L q ks G} {T : Trace (Outcome β Hs N m)}

/-- `Present` gives `PresentAt` at every record of outside computation. -/
theorem presentAt_of_present (hP : Present A T) (O : Outside T) : PresentAt A T O :=
  fun ω ha j t hw b hb => by
    obtain ⟨τ, h1, h2, h3⟩ := hP ω ha j t hw b hb
    exact ⟨τ, h1, h2, Finset.mem_union_left _ h3⟩

/-- `PresentAt` and `Covered` give `Present` where nothing but the verifier's messages entered the node over the
credited passes of any accepting outcome, and the model's passes are among the trace's credited ones. -/
theorem present_of_covered {O : Outside T} {I : Ingress (Outcome β Hs N m)} (hle : P ≤ T.passes)
    (hC : Covered T O I) (hq : ∀ ω, T.accepts ω → Quiet T I.entered ω) (hP : PresentAt A T O) : Present A T :=
  fun ω ha j t hw b hb => by
    obtain ⟨τ, h1, h2, h3⟩ := hP ω ha j t hw b hb
    exact ⟨τ, h1, h2, onNode_of_covered hC ha (hq ω ha) (lt_of_lt_of_le j.2 hle) h1 h2 h3⟩

end Sketch

end Accounting.SecurityProofs.Remote
