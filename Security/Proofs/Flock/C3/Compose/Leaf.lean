import Proofs.Flock.VBridge.Family.Commits

/-!
# The decoder's last round, read from σ's leaf

Candidate 3's last round (the openings) is held by no round commitment, so `VBridgeC` asks nothing of its commit
strings there, and the decoder reads its message from the outer proof's layers, σ's leaf, in full. `skLeaf` is the
decoder's mask: stage `i`'s commit strings (`skOf`) at the committed rounds `i + 1 < r`, and every position at the
last. At the committed rounds its mask is `skOf`'s (`maskOf_skLeaf`), so `MsgsReadC3` reads those rounds from their
commit strings and the last from every read wire of the layers; `vDecodesR_wmsgLeaf` is `VBridgeC`'s decoding at it.
-/

namespace FlockC3.Compose

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open FlockVBridge.Family (skOf wmsgOf msgOf maskOf layerOf)

variable {S : ℕ} (Vs : Fin S → VStmt)

/-- **The decoder's mask at candidate 3**: stage `i`'s commit strings (`skOf`) at the committed rounds `i + 1 < r`,
and every position at the last, whose message no round commitment holds. -/
noncomputable def skLeaf (own : Fin S → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop) (r i : ℕ)
    (j : Fin S) : Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv) :=
  if i + 1 < r then skOf Vs own sel i j else Finset.univ

variable {Vs} {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}

theorem skLeaf_of_lt {r i : ℕ} (hi : i + 1 < r) (j : Fin S) : skLeaf Vs own sel r i j = skOf Vs own sel i j :=
  ite_eq_left hi

/-- **At the committed rounds the decoder's mask is the commit strings'.** -/
theorem maskOf_skLeaf {r i : ℕ} (hi : i + 1 < r) :
    maskOf Vs (skLeaf Vs own sel r) i = maskOf Vs (skOf Vs own sel) i :=
  funext fun _ => funext fun j => funext fun _ => by simp only [maskOf, skLeaf_of_lt hi j]

/-- **`VBridgeC`'s decoding at the leaf decoder**: messages `wmsgOf`'s and `msgOf`'s at `skLeaf` and one decoder `dec`
decode each committed round's message from the read wires of its commit strings, and hold one at the last round. -/
theorem vDecodesR_wmsgLeaf {I : Recursion.Inner} (dec : ℕ → ((j : Fin S) → (Vs j).Bits) → I.Msg) :
    VDecodesR (wmsgOf Vs (skLeaf Vs own sel I.r) I.r dec) (fun i _ a b => i + 1 < I.r → a = b)
      (fun i j => if i + 1 < I.r then skOf Vs own sel i j else ∅) (msgOf Vs (skLeaf Vs own sel I.r) dec) := by
  intro i hi _ Xt v hX
  simp only [wmsgOf, List.getElem?_map, List.getElem?_range hi, Option.map_some]
  refine Option.Rel.some ?_
  intro hc
  show dec i (maskOf Vs (skLeaf Vs own sel I.r) i Xt) = dec i (layerOf Vs (skLeaf Vs own sel I.r) i v)
  refine congrArg (dec i) (funext fun j => funext fun g => ?_)
  simp only [maskOf, layerOf, skLeaf_of_lt hc j]
  split_ifs with h
  · exact hX j g (by simpa [hc] using h.1) h.2
  · rfl

end FlockC3.Compose
