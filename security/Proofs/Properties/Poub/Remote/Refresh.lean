module

public import Proofs.Properties.Poub.Remote.Accept
public import Properties.Poub.Remote.Refresh

/-!
# Proofs of refresh's code-side properties

`Properties.Poub.Remote.Refresh`'s `OpensOrigin`: an accepted window passes `checkOpening` on each sampled opening
(`checkWindow_accept`), and `checkOpening_origin` reads the origin's checks off its early returns.
-/

@[expose] public section

open Poub

namespace Properties.Poub

/-- An opening `checkOpening` passes, of a block `j` of an append `a` with an origin leaf `(q, oroot, odom)`, names
`q`, and its block decodes under `a`'s coin to a payload whose leaf is leaf `q` under `oroot`. -/
theorem checkOpening_origin {reg : Registration} {digest : ByteArray} {appends : Array Append} {w : WindowRecord}
    {r : Receipter} {i p : Nat} {o : Opening} {a : Append} {q : Nat} {oroot : ByteArray}
    {odom : Flock.FV3.Sha512.Domain}
    (h : checkOpening reg digest appends w r i p o = none) (hj : reg.blocks ≤ w.passes[i]!.plan.block p)
    (ha : appendOf appends (w.passes[i]!.plan.block p) = some a)
    (hq : originLeaf reg digest appends a (w.passes[i]!.plan.block p) = some (q, oroot, odom)) :
    o.originPosition = some q ∧ ∃ payload,
      Codec.decode o.block (w.passes[i]!.plan.block p) reg.key (some (o.digest, some a.coin)) = .ok (some payload) ∧
      odom.verifyOpening oroot odom.count q (hidden payload o.payloadSalt) SCHEMA o.payloadPath = true := by
  unfold checkOpening at h
  simp only [Id.run] at h
  generalize w.passes[i]!.plan.block p = j at h hj ha hq ⊢
  have hb : ¬ j < reg.blocks := Nat.not_lt.2 hj
  split at h
  · simp at h
  simp only [hb, ↓reduceIte, ha, hq, Option.bind, Option.map] at h
  split at h
  · simp at h
  split at h
  · simp at h
  rename_i hpos
  refine ⟨by simpa using hpos, ?_⟩
  split at h
  · split at h
    · simp at h
    split at h
    · rename_i payload hdec
      split at h
      · simp at h
      rename_i h4
      exact ⟨payload, hdec, by simpa using h4⟩
    · simp at h
  · simp at h

theorem opensOrigin : OpensOrigin := by
  intro reg appends w coin samples openings h
  obtain ⟨sampled, hd, hc⟩ := verifyWindow_accept h
  obtain ⟨hsize, hk⟩ := checkWindow_accept hc
  refine ⟨sampled, hd, hsize, ?_⟩
  intro k hk' a q oroot odom
  dsimp only
  intro hj ha hq
  exact checkOpening_origin (hk k hk') hj ha hq

end Properties.Poub
