import Proofs.Flock.Soundness.Binding.Rows

/-!
# `registered_weights`: the extracted rows are the registered weights, or a SHA-512 collision

Knowledge soundness extracts, for each drawn unit, a message of its table that packs a satisfying witness
(`Partition.cand`). The weights the registrant committed to are an opening of each registered leaf: a row and a salt whose
leaf is the registered one. `registered_weights`: at each of a drawn unit's wires, the row a satisfying witness holds at
the wire's commit string is the registered row, or the binding's `collide` of the two openings is an explicit SHA-512
collision. It is the value binding's `collide_spec`, applied through the registration fact. `registered_weights_hm96` is it
for `hm96-sha512/row/v1` rows, from `HmRowComputes`, and `registered_weights_extracted` for the message the audit's
extractor settles on.
-/

namespace FlockSoundness.Binding

open Game Model Accounting Refine Audit

variable {C : Circuit} {n : ℕ} {P : Audit.Partition C n} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {decode : Audit.Partition.DecoderB (C := C) (Reg := Reg) plan tab}

namespace Layout

variable (lay : Layout P plan tab decode) (scheme : lay.Pos → ExplicitLeaf lay.Val)

/-- **The rows are the registered weights, or a SHA-512 collision.** Let `(wv, ws)` open the registered leaf at the
commit string of a drawn unit's wire `g`: a value and salt whose leaf is the registered one, as the weights and salt the
registrant committed to are. In a satisfying witness of the unit's table, the value the witness holds there is `wv`, or
`collide` of the witness's opening and `(wv, ws)` is a SHA-512 collision. -/
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

/-- **`registered_weights` for `hm96-sha512/row/v1` rows**, from `HmRowComputes`. Let `(wv, ws)` be a row and salt whose
hm96 leaf is the registered one at a row holding a drawn unit's wire `g`: the weights and salt the registrant committed
to. In a satisfying witness of the unit's table, the row the witness holds there is `wv`, or `hm96Pair` of the two rows'
bytes and salts is a SHA-512 collision: of the outer hash, of the salt digest, or of the rows' own digests. -/
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
/-- **`registered_weights` for the audit's extraction**: when the extractor settles, for a drawn unit `u` of the draw `ω`,
on a message of its table (`cand … = some M`), the value that message holds at the commit string of each of `u`'s wires is
the registered one, or `collide` of the two openings is a SHA-512 collision. -/
theorem registered_weights_extracted {D : Type} [Inhabited D] (H : List UInt8 → D)
    (E : Merkle.Enc (List FlockLevel3.GF128) D) {L : Law n} (R : Reg) (τ : Cont L Reg (batchedSession H E plan) R)
    (lay : Layout P plan tab decode) (scheme : lay.Pos → ExplicitLeaf lay.Val) (hreg : lay.Registered scheme)
    {ω : L.Ω} {u : Fin n} (hu : u ∈ L.draw ω) {bs : List (List (OutC FlockLevel3.GF128 D))}
    {M : Msg plan tab (L.draw ω) R u} (hM : cand H E plan tab R τ ω u bs = some M) {g : Fin C.N} (hg : g ∈ P.io u)
    (wv : lay.Val) (ws : List UInt8) (hw : (scheme (lay.pos g)).leaf wv ws = lay.regLeaf R (lay.pos g)) :
    lay.row (L.draw ω) R u M (lay.pos g) = wv ∨
      Collides (lay.collide scheme (lay.openingAt (L.draw ω) R u M (lay.pos g)) (lay.pos g, wv, ws)) :=
  lay.registered_weights scheme hreg hu M (cand_sat H E plan tab R τ hM) hg wv ws hw

end FlockSoundness.Binding
