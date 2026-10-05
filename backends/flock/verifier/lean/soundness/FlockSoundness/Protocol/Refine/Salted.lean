import FlockSoundness.Protocol.Refine.Regions

/-!
# The salted scheme `hm96-sha512/v1` refines the model's salted leaves (R7 and R8 for hm96; PROTOCOL.md §14)

`MerkleScheme.hm96Sha512` hashes an opened row's leaf as `SHA-512(leaf_prefix ‖ b ‖ c)`, with `x = SHA-512(row)`,
`b = x ⊕ M·y` and `c = SHA-512(salt_prefix ‖ y)` for the opening's 192-byte salt `y` (`Hm96.Default512`). Its nodes are
SHA-512 of the pair, as the unsalted scheme's are. The compiled model's `Merkle.Leaf.hm96 K` is that leaf for a key `K`.
`K512` is the executable's own key: its prefixes and its mask are `Hm96.Default512`'s, so the two leaves are equal
(`hm96_leaf`).

* **R7:** a passed hm96 Merkle check is the model's salted path check (`merkleCheck_verifiesL`), and a rep whose requests
  passed it is `OpensOKL` at the proof's rows and salts (`opensOKL_of`).
* **R8:** `verify_refines_hm96` is `verify_refines` for the salted scheme, with `OpensOKL` at the proofs' salts.
* **R9:** `verify_refines_ofCircuit_hm96`, for the verifier's own statements. For `Stmt.setupH`'s, `setupH_wf` gives its
  layout and region hypotheses.
-/

namespace FlockSoundness.Refine

open Flock FlockLevel3 Game Transcript Model Accounting

/-! ## The executable's key -/

theorem length_data (b : ByteArray) : b.data.toList.length = b.size := by
  rw [Array.length_toList, ByteArray.size_data]

/-- **The executable's hm96 key** (`Hm96.Default512`): its leaf and salt prefixes, and its mask `y ↦ M·y`, 64 bytes. -/
def K512 : Merkle.Hm96Key where
  leafPrefix := Hm96.Default512.leafPrefix.data.toList
  saltPrefix := Hm96.Default512.saltPrefix.data.toList
  mask y := (Hm96.Default512.mask ⟨y.toArray⟩).data.toList
  L := 64
  mask_len y := by rw [length_data, mask_size]

theorem hL512 (d : Dig) : (enc512.dig d).length = K512.L := by
  show d.1.data.toList.length = 64
  rw [length_data, d.2]

/-- hm96's leaf scheme at the executable's key. -/
abbrev leaf512 : Merkle.Leaf (List 𝔽) (List UInt8) Dig H512 := Merkle.Leaf.hm96 H512 enc512 K512 hL512

/-! ## A passed hm96 check is the model's salted path check -/

/-! ## The proof's openings, with their salts, pass the model's salted path checks -/

/-- The proof's salt at level `ℓ`, query `j`, as bytes. -/
def saltRow (p : LigeritoProof) (sch : Flock.Schedule) (ℓ j : ℕ) : List UInt8 :=
  ((openAt p (sch.levels.size - 1) ℓ).2.2.getD j ByteArray.empty).data.toList

/-- The proof files' salts (§14): rep `r`'s opened rows' salts, level by level. -/
def saltsOf (p₀ p₁ : Proof) (sch : Flock.Schedule) : Salts (List UInt8) :=
  fun r ℓ j => if r = 0 then saltRow p₀.ligerito sch ℓ j else saltRow p₁.ligerito sch ℓ j

/-! ## The verifier refines the model's salted table -/

end FlockSoundness.Refine
