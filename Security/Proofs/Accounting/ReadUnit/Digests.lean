module

public import Definitions.Accounting.ReadUnit
public import Properties.Verity.Game

/-!
# What the counterexample read's checks hash has no SHA-512 collision

`TableReadBinding_nontrivial`'s read (`Witness.R₀`) opens its one-row table at index 0. Its own opening holds no row
bits and a salt of ones (`op₀`), its table's opening 16 zero bits and the same salt (`o₀`), both under the row-leaf
verifier of one row at its own rank, the domain id of ones and the served hm96 key (`V₀`). The messages those two checks
hash under SHA-512 (`L₀`, what `Reads.hashedAt` lists there) have pairwise different digests, which the kernel computes
(`digests₀`), so the reads' collision finder finds no collision there (`no_collision₀`). The shared salt is the
expensive part of each hm96 leaf (its mask), so the kernel computes it once.
-/

@[expose] public section

namespace Accounting.ReadUnit.Witness

/-- The domain id the read's gates hold: every bit set. -/
def d₀ : Verity.Digest := digestOf (List.ofFn fun _ : Fin 512 => true)

/-- The read's opening: no row bits, a salt of ones, no siblings. -/
def op₀ : Hash.RowLeaf.Opening := ⟨[], List.replicate 192 255, []⟩

/-- The table's one row: 16 zero bits, a salt of ones, no siblings. -/
def o₀ : Hash.RowLeaf.Opening := ⟨List.replicate 16 false, List.replicate 192 255, []⟩

/-- The read's row-leaf verifier: one row at its own rank, the domain id `d₀`, the served hm96 key. -/
def V₀ : Hash.RowLeaf.Verifier :=
  Hash.EmbeddingRead.verifier ⟨d₀, 1, fun r => r⟩ Flock.Hm96.Default512.key.data.toList

/-- What the read's check and its table's check hash at index 0 (`Reads.hashedAt` with `H = G = SHA-512`). -/
def L₀ : List Hash.Msg :=
  Hash.RowLeaf.outerHashed Verity.sha512 Verity.sha512 V₀ 0 op₀ ++
      Hash.RowLeaf.outerHashed Verity.sha512 Verity.sha512 V₀ 0 o₀ ++
    (Hash.RowLeaf.innerHashed Verity.sha512 V₀ op₀ ++ Hash.RowLeaf.innerHashed Verity.sha512 V₀ o₀)

theorem digests₀ : (L₀.dedup.map fun m => (Verity.sha512 m).bytes).Nodup := by
  decide +kernel

theorem no_collision₀ : ¬ ∃ x ∈ L₀, ∃ y ∈ L₀, Hash.Collision (fun m => (Verity.sha512 m).bytes) x y := by
  rintro ⟨x, hx, y, hy, hxy, he⟩
  exact hxy (List.inj_on_of_nodup_map digests₀ (List.mem_dedup.2 hx) (List.mem_dedup.2 hy) he)

end Accounting.ReadUnit.Witness
