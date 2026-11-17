module

public import Proofs.Flock.Soundness.Binding.Rows
import Mathlib.Combinatorics.Matroid.Init

/-! `registered_weights`: the extracted rows are the registered weights, or a SHA-512 collision. -/

@[expose] public section

namespace FlockSoundness.Binding

open Game Model Accounting Refine Audit

variable {C : Verity.Boolean.Circuit} {n : ℕ} {P : Audit.Partition C n} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {decode : Audit.Partition.DecoderB (C := C) (Reg := Reg) plan tab}

namespace Layout

variable (lay : Layout P plan tab decode) (scheme : lay.Pos → ExplicitLeaf lay.Val)

theorem registered_weights (hreg : lay.Registered scheme) {S : Finset (Fin n)} {R : Reg} {u : Fin n} (hu : u ∈ S)
    (M : Msg plan tab S R u) (hM : Sat plan tab S R u M) {g : Fin C.N} (hg : g ∈ P.io u) (wv : lay.Val)
    (ws : List UInt8) (hw : (scheme (lay.pos g)).leaf wv ws = lay.regLeaf R (lay.pos g)) :
    lay.row S R u M (lay.pos g) = wv ∨
      Collides (lay.collide scheme (lay.openingAt S R u M (lay.pos g)) (lay.pos g, wv, ws)) := by
  by_cases hv : lay.row S R u M (lay.pos g) = wv
  · exact Or.inl hv
  · exact Or.inr (lay.collide_spec scheme _ _
      (congrArg (Prod.mk (lay.pos g)) ((hreg S R u hu M hM g hg).trans hw.symm)) hv)

end Layout

namespace HmRows

variable {lay : Layout P plan tab decode} (rs : HmRows lay)

theorem registered_weights_hm96 (hHm : rs.Computes) {S : Finset (Fin n)} {R : Reg} {u : Fin n} (hu : u ∈ S)
    (M : Msg plan tab S R u) (hM : Sat plan tab S R u M) {g : Fin C.N} (hg : g ∈ P.io u) (hrow : rs.isRow (lay.pos g))
    (wv : lay.Val) (ws : List UInt8)
    (hw : (ExplicitLeaf.hm96 rs.enc rs.enc_inj).leaf wv ws = lay.regLeaf R (lay.pos g)) :
    lay.row S R u M (lay.pos g) = wv ∨
      Collides (hm96Pair (rs.enc (lay.row S R u M (lay.pos g))) (lay.salt S R u M (lay.pos g)) (rs.enc wv) ws) := by
  by_cases hv : lay.row S R u M (lay.pos g) = wv
  · exact Or.inl hv
  · refine Or.inr (hm96Pair_spec _ _ _ _ (fun e => hv (rs.enc_inj e)) ?_)
    have hs : rs.scheme (lay.pos g) = ExplicitLeaf.hm96 rs.enc rs.enc_inj := by
      simp only [scheme, hrow, ↓reduceIte]
    have hr := rs.registered hHm S R u hu M hM g hg
    rw [hs] at hr
    exact hr.trans hw.symm

end HmRows

open Audit.Partition in
theorem registered_weights_extracted {D : Type} [Inhabited D] (H : List UInt8 → D)
    (E : Merkle.Enc (List FlockLevel3.GF128) D) {L : SamplingDesign n} (R : Reg) (τ : Cont L Reg (batchedSession H E plan) R)
    (lay : Layout P plan tab decode) (scheme : lay.Pos → ExplicitLeaf lay.Val) (hreg : lay.Registered scheme)
    {ω : L.Ω} {u : Fin n} (hu : u ∈ L.draw ω) {bs : List (List (OutC FlockLevel3.GF128 D))}
    {M : Msg plan tab (L.draw ω) R u} (hM : cand H E plan tab R τ ω u bs = some M) {g : Fin C.N} (hg : g ∈ P.io u)
    (wv : lay.Val) (ws : List UInt8) (hw : (scheme (lay.pos g)).leaf wv ws = lay.regLeaf R (lay.pos g)) :
    lay.row (L.draw ω) R u M (lay.pos g) = wv ∨
      Collides (lay.collide scheme (lay.openingAt (L.draw ω) R u M (lay.pos g)) (lay.pos g, wv, ws)) :=
  lay.registered_weights scheme hreg hu M (cand_sat H E plan tab R τ hM) hg wv ws hw

end FlockSoundness.Binding
