module

public import Proofs.Properties.Nci.LeafWidths
public import Proofs.Properties.Nci.BodiesOk.Decoded

/-!
# The decoder ran the width check

The decoder's invariant holds, of every body it builds, that each strided part of its references repeats a pattern
of positive length and that the width check passed it (`BodiesOk.BodyFacts`). `decoded_inner` reads both off a
Program `Flock.Program.decode` returns, for every Definition: a primitive's body is the default one, which has no
reference.
-/

@[expose] public section

namespace NciCheck.Properties.SecurityProofs.LeafWidths
open Flock.Program

theorem partInner_of {q : Part} (h : BodiesOk.PartInner q) : PartInner q := by
  cases q <;> exact h

theorem bodyInner_of {B : Body} (h : BodiesOk.BodyInner B) : BodyInner B :=
  ⟨fun k hk a ha q hq => partInner_of (h.1 k hk a ha q hq), fun q hq => partInner_of (h.2 q hq)⟩

/-- **Every body of a decoded Program is `BodyInner`, and the width check passed it.** -/
theorem decoded_inner (bytes : ByteArray) (P : Prog) (h : decode bytes = .ok P) :
    ∀ d < P.defs.size, BodyInner (P.defs[d]!).body ∧
      ∃ defs' key w w', (Widths.bodyBad defs' key (P.defs[d]!).body).run w = .ok none w' := by
  obtain ⟨st, hP, hinv, -⟩ := BodiesOk.decode_spec bytes P h
  have hok : ∀ i < P.defs.size, BodiesOk.DefOk P.defs i P.defs[i]! := by rw [hP]; exact hinv.1
  intro d hd
  obtain ⟨-, -, -, ps, -, -, hI, hrun⟩ := hok d hd
  exact ⟨bodyInner_of hI, hrun⟩

end NciCheck.Properties.SecurityProofs.LeafWidths
