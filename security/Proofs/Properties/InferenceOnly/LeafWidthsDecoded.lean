module

public import Proofs.Properties.InferenceOnly.DecodeKeys
public import Proofs.Properties.InferenceOnly.LeafWidthsBody

/-! Every body of a decoded Program passed the width check, so every composite body reads every leaf at its width. -/

@[expose] public section

namespace InferenceOnlyCheck.Properties.SecurityProofs.LeafWidths
open Flock.Program

theorem decoded_inner (bytes : ByteArray) (P : Prog) (h : decode bytes = .ok P) :
    ∀ d < P.defs.size, BodyInner (P.defs[d]!).body ∧
      ∃ defs' key w w', (Widths.bodyBad defs' key (P.defs[d]!).body).run w = .ok none w' := by
  obtain ⟨st, hP, hinv, -⟩ := BodiesOk.decode_spec bytes P h
  have hok : ∀ i < P.defs.size, BodiesOk.DefOk P.defs i P.defs[i]! := by rw [hP]; exact hinv.1
  intro d hd
  obtain ⟨-, -, -, ps, -, -, hI, hrun⟩ := hok d hd
  exact ⟨hI, hrun⟩

theorem decoded_checked (bytes : ByteArray) (P : Prog) (h : decode bytes = .ok P) :
    ∀ d < P.defs.size, (P.defs[d]!).prim = false →
      (P.defs[d]!).body.retType = (P.defs[d]!).ret ∧ BodyInner (P.defs[d]!).body ∧
        ∃ defs' key w w', (Widths.bodyBad defs' key (P.defs[d]!).body).run w = .ok none w' :=
  fun d hd hp => ⟨DecodeKeys.decode_ret bytes P h d hd hp, decoded_inner bytes P h d hd⟩

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

end InferenceOnlyCheck.Properties.SecurityProofs.LeafWidths
