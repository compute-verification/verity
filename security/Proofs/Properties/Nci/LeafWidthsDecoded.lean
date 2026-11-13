module

public import Proofs.Properties.Nci.DecodeKeys
public import Proofs.Properties.Nci.LeafWidthsBody

/-!
# The decoder ran the width check

The decoder's invariant holds, of every body it builds, that each strided part of its references repeats a pattern
of positive length and that the width check passed it (`BodiesOk.BodyFacts`). `decoded_inner` reads both off a
Program `Flock.Program.decode` returns, for every Definition: a primitive's body is the default one, which has no
reference. `decoded_checked` adds, for a composite, that its body returns the Definition's type
(`DecodeKeys.decode_ret`), and `decoded_wide` is what the width check's soundness (`bodyBad_sound`) makes of the
three: every composite body of a decoded Program reads every leaf at its width.
-/

@[expose] public section

namespace NciCheck.Properties.SecurityProofs.LeafWidths
open Flock.Program

/-- **Every body of a decoded Program is `BodyInner`, and the width check passed it.** -/
theorem decoded_inner (bytes : ByteArray) (P : Prog) (h : decode bytes = .ok P) :
    ∀ d < P.defs.size, BodyInner (P.defs[d]!).body ∧
      ∃ defs' key w w', (Widths.bodyBad defs' key (P.defs[d]!).body).run w = .ok none w' := by
  obtain ⟨st, hP, hinv, -⟩ := BodiesOk.decode_spec bytes P h
  have hok : ∀ i < P.defs.size, BodiesOk.DefOk P.defs i P.defs[i]! := by rw [hP]; exact hinv.1
  intro d hd
  obtain ⟨-, -, -, ps, -, -, hI, hrun⟩ := hok d hd
  exact ⟨hI, hrun⟩

/-- **The decoder ran the width check**: a composite Definition of a decoded Program returns its body's return type,
its body is `BodyInner`, and the width check passed it. -/
theorem decoded_checked (bytes : ByteArray) (P : Prog) (h : decode bytes = .ok P) :
    ∀ d < P.defs.size, (P.defs[d]!).prim = false →
      (P.defs[d]!).body.retType = (P.defs[d]!).ret ∧ BodyInner (P.defs[d]!).body ∧
        ∃ defs' key w w', (Widths.bodyBad defs' key (P.defs[d]!).body).run w = .ok none w' :=
  fun d hd hp => ⟨DecodeKeys.decode_ret bytes P h d hd hp, decoded_inner bytes P h d hd⟩

/-- **Every composite body of a decoded Program reads every leaf at its width**, and returns the Definition's
type. -/
theorem decoded_wide (bytes : ByteArray) (P : Prog) (h : decode bytes = .ok P) :
    ∀ d < P.defs.size, (P.defs[d]!).prim = false →
      (P.defs[d]!).body.retType = (P.defs[d]!).ret ∧ BodyWide (P.defs[d]!).body := by
  obtain ⟨st, hP, hinv, -⟩ := BodiesOk.decode_spec bytes P h
  have hok : ∀ i < P.defs.size, BodiesOk.DefOk P.defs i P.defs[i]! := by rw [hP]; exact hinv.1
  intro d hd hp
  obtain ⟨hrt, hI, defs', key, w, w', hrun⟩ := decoded_checked bytes P h d hd hp
  obtain ⟨hc, hB⟩ := hok d hd
  refine ⟨hrt, bodyBad_sound P.defs defs' d key _ w w' hB hI ?_ hrun⟩
  rw [hrt]
  exact (hc.2 hp).2.1

end NciCheck.Properties.SecurityProofs.LeafWidths
