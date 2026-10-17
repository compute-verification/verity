import Proofs.Flock.Claims.Sampling.Path
import Proofs.Flock.Soundness.Binding.Leaf

/-!
# The hm96 leaf binds (hm96-sha512/row/v1)

The trace commitment's leaf is `hm96-sha512/row/v1`'s, as Flock's soundness proofs define it
(`FlockSoundness.Binding.ExplicitLeaf.hm96`, at the executable's key `Refine.K512`): the row's digest `x = H(row)`, the
salt's `c = H(salt_prefix ‖ y)`, and the leaf `H(leaf_prefix ‖ (x ⊕ M·y) ‖ c)`. Here it is as an oracle program:

* `hm96Leaf row salt`: at an opening `o`, the three queries in that order, on its row's `sha512/row/v1` input `row o`
  and its salt `salt o`, refusing when `x` or `c` isn't 64 bytes, which SHA-512's never are.
* `RowPlaces row`: openings at one position of different values have different rows (the row places the value).
* `hm96Leaf_binds`: **`LeafBinds H (hm96Leaf row salt)` for every `H`**, at a row that places the value. One leaf of
  two openings is one answer to the outer queries; where they differ that is the collision, and otherwise (both digests
  64 bytes, so `b` and `c` split) the salt digests agree at two salts, or else (one salt, so one mask) the row digests
  agree at two rows.
* `hm96Leaf_sha512`: at SHA-512 the program never refuses and its leaf is Flock's, `hm96Input`'s digest.
* `Binding_of_hm96`: **the claims' `Binding`** for a commitment of hm96 leaves under the registered path check.
-/

namespace Claims.Sampling

open FlockSoundness.Refine FlockSoundness.Binding

/-- The outer input `leaf_prefix ‖ (x ⊕ M·y) ‖ c` at the answers `H` gives. -/
def hm96Outer (H : Msg → Msg) (row salt : Opening Bool → Msg) (o : Opening Bool) : Msg :=
  K512.leafPrefix ++ (FlockSoundness.Merkle.xorBytes (H (row o)) (K512.mask (salt o)) ++ H (K512.saltPrefix ++ salt o))

/-- **hm96's leaf as an oracle program**: the row's digest, the salt's, then the outer hash, refusing digests that
aren't 64 bytes. -/
def hm96Leaf (row salt : Opening Bool → Msg) (o : Opening Bool) : OC (Option Msg) :=
  .query (row o) fun x =>
    .query (K512.saltPrefix ++ salt o) fun c =>
      .query (K512.leafPrefix ++ (FlockSoundness.Merkle.xorBytes x (K512.mask (salt o)) ++ c)) fun l =>
        .ret (if x.length = 64 ∧ c.length = 64 then some l else none)

/-- **The row places the value**: openings at one position of different values have different rows. -/
def RowPlaces (row : Opening Bool → Msg) : Prop :=
  ∀ o o' : Opening Bool, o.pos = o'.pos → o.val ≠ o'.val → row o ≠ row o'

theorem hm96Leaf_fst (H : Msg → Msg) (row salt : Opening Bool → Msg) (o : Opening Bool) :
    ((hm96Leaf row salt o).eval H).1 =
      if (H (row o)).length = 64 ∧ (H (K512.saltPrefix ++ salt o)).length = 64 then some (H (hm96Outer H row salt o))
      else none :=
  rfl

theorem hm96Leaf_log (H : Msg → Msg) (row salt : Opening Bool → Msg) (o : Opening Bool) :
    ((hm96Leaf row salt o).eval H).2 =
      [(row o, H (row o)), (K512.saltPrefix ++ salt o, H (K512.saltPrefix ++ salt o)),
        (hm96Outer H row salt o, H (hm96Outer H row salt o))] :=
  rfl

/-- **The hm96 leaf binds, for every `H`**: two openings at one position, of different values, with one leaf make two
different queries with one answer: the outer ones, else the salt digests', else the row digests'. -/
theorem hm96Leaf_binds (H : Msg → Msg) {row : Opening Bool → Msg} (hrow : RowPlaces row) (salt : Opening Bool → Msg) :
    LeafBinds H (hm96Leaf row salt) := by
  intro o o' l hpos hne h h'
  rw [hm96Leaf_fst] at h h'
  split_ifs at h with hg
  split_ifs at h' with hg'
  obtain ⟨hx, hc⟩ := hg
  obtain ⟨hx', hc'⟩ := hg'
  rw [hm96Leaf_log, hm96Leaf_log]
  by_cases hq : hm96Outer H row salt o = hm96Outer H row salt o'
  · have hcs := List.append_cancel_left hq
    have hm : ∀ y, (K512.mask y).length = 64 := K512.mask_len
    have hlen : (FlockSoundness.Merkle.xorBytes (H (row o)) (K512.mask (salt o))).length =
        (FlockSoundness.Merkle.xorBytes (H (row o')) (K512.mask (salt o'))).length := by
      simp only [FlockSoundness.Merkle.xorBytes, List.length_zipWith, hx, hx', hm]
    obtain ⟨hb, hcc⟩ := List.append_inj hcs hlen
    by_cases hy : salt o = salt o'
    · rw [hy] at hb
      have hxx := FlockSoundness.Merkle.xorBytes_inj (hx.trans (hm _).symm) (hx'.trans (hm _).symm) hb
      exact ⟨(row o, H (row o)), by simp, (row o', H (row o')), by simp, hrow o o' hpos hne, hxx⟩
    · exact ⟨(K512.saltPrefix ++ salt o, _), by simp, (K512.saltPrefix ++ salt o', _), by simp,
        fun e => hy (List.append_cancel_left e), hcc⟩
  · exact ⟨(hm96Outer H row salt o, _), by simp, (hm96Outer H row salt o', _), by simp, hq,
      (Option.some.inj h).trans (Option.some.inj h').symm⟩

/-- **At SHA-512 the program is Flock's hm96 leaf**: it never refuses, and its leaf is the digest of `hm96Input` at the
row and salt (`ExplicitLeaf.hm96`'s, the executable's `Hm96.Default512.treeLeaf` by `hm96_treeLeaf`). -/
theorem hm96Leaf_sha512 (row salt : Opening Bool → Msg) (o : Opening Bool) :
    ((hm96Leaf row salt o).eval sha512Msg).1 = some (digBytes (H512 (hm96Input (row o) (salt o)))) := by
  have e : ∀ m, sha512Msg m = digBytes (H512 m) := fun m => by
    unfold sha512Msg bytes
    rw [digBytes, dig_eq, H512_val]
    rfl
  have hlen : ∀ m, (sha512Msg m).length = 64 := fun m => by
    rw [e]
    exact hL512 _
  rw [hm96Leaf_fst, ite_eq_left ⟨hlen _, hlen _⟩]
  simp only [hm96Outer, e, hm96Input, commitString]

/-- **Sampling: binding, at hm96 leaves under the registered path check**: a deployment whose trace commitment checks
an opening's hm96 leaf (`hm96Leaf`, its row placing the value) with the verifier's own registered path check binds. -/
theorem Binding_of_hm96 {Coin Col T : Type} (D : Deploy Coin Col T) (own : Array _root_.Flock.Registered.Port)
    (at_ : List Msg → ℕ → Option (Msg × ℕ)) (row salt π : Opening Bool → Msg)
    (layer : (Msg → Msg) → List Msg → List (Opening Bool) → List Msg)
    (hD : D.commit = pathCommit at_ (hm96Leaf row salt) π (regPath own) layer) (hrow : RowPlaces row) : Binding D :=
  Binding_of_regPath D own at_ (hm96Leaf row salt) π layer hD fun H => hm96Leaf_binds H hrow salt

end Claims.Sampling
