module

public import Definitions.Accounting.ReadUnit
public import Properties.Verity.Game
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.Data.BitVec.Basic
import all Init.Data.ByteArray.Basic
import all Init.Data.Fin.Fold
import all Init.Data.List.Basic
import all Init.Data.List.BasicAux
import all Init.Data.List.Control
import all Init.Data.List.OfFn
import all Init.Data.List.Pairwise
import all Init.Data.Nat.Basic
import all Init.Data.Option.Basic
import all Init.Data.Range.Basic
import all Init.Data.String.Basic
import all Init.Data.String.Bootstrap
import all Init.Data.String.Defs
import all Init.Data.String.Extra
import all Init.Data.UInt.Basic
import all Init.Data.UInt.BasicAux

/-!
# What the counterexample read's checks hash has no SHA-512 collision

`TableReadBinding_nontrivial`'s read (`Witness.R₀`) opens its one-row table at index 0. Its own opening holds no row
bits and a salt of ones (`op₀`), its table's opening 16 zero bits and the same salt (`o₀`), both under the row-leaf
verifier of one row at its own rank, the domain id of ones and the served hm96 key (`V₀`). The messages those two checks
hash under SHA-512 (`L₀`, what `Reads.hashedAt` lists there) have pairwise different digests, which the kernel computes
(`digests₀`), so the reads' collision finder finds no collision there (`no_collision₀`). The shared salt is the
expensive part of each hm96 leaf (its mask), so the kernel computes it once.

The `import all` lines let the kernel unfold the core definitions SHA-512 and the key's derivation run on (`ByteArray`,
`Std.Legacy.Range`'s loops, `String.toUTF8`, the `UInt` operations), which a module otherwise sees only by signature.
`digests₀` reads each mask bit through `xorFold` (`maskBit_eq`), which forces the fold's accumulator at each step.
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

/-- `Hm96.maskBit`'s fold, its accumulator forced at each step, so that the kernel never builds the xor of 1536 terms
that a salt of ones would leave it to evaluate at the end. -/
def xorFold (key : Hash.Msg) (i : ℕ) : List ℕ → Bool → Bool
  | [], acc => acc
  | j :: js, acc =>
    match xor acc (Hash.Hm96.bit key (i + j)) with
    | true => xorFold key i js true
    | false => xorFold key i js false

theorem foldl_xorFold (key : Hash.Msg) (i : ℕ) :
    ∀ (l : List ℕ) (acc : Bool), l.foldl (fun acc j => xor acc (Hash.Hm96.bit key (i + j))) acc = xorFold key i l acc
  | [], _ => rfl
  | j :: js, acc => by
    rw [List.foldl_cons, xorFold]
    cases xor acc (Hash.Hm96.bit key (i + j)) <;> exact foldl_xorFold key i js _

theorem maskBit_eq (key y : Hash.Msg) (i : ℕ) :
    Hash.Hm96.maskBit key y i = xorFold key i ((List.range (8 * Hash.Hm96.SALT_BYTES)).filter (Hash.Hm96.bit y)) false :=
  foldl_xorFold key i _ false

theorem digests₀ : (L₀.dedup.map fun m => (Verity.sha512 m).bytes).Nodup := by
  simp only [L₀, Hash.RowLeaf.outerHashed, Hash.RowLeaf.leafMsg, Hash.Hm96.hashed, Hash.Hm96.leaf,
    Hash.Hm96.commitString, Hash.Hm96.mask, maskBit_eq]
  decide +kernel

theorem no_collision₀ : ¬ ∃ x ∈ L₀, ∃ y ∈ L₀, Hash.Collision (fun m => (Verity.sha512 m).bytes) x y := by
  rintro ⟨x, hx, y, hy, hxy, he⟩
  exact hxy (List.inj_on_of_nodup_map digests₀ (List.mem_dedup.2 hx) (List.mem_dedup.2 hy) he)

end Accounting.ReadUnit.Witness
