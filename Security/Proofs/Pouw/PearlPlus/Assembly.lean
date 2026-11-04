import Properties.Pouw.PearlPlus
import Definitions.Pouw.PearlC.TileCheck
import Proofs.Pouw.PearlPlus.Words

/-!
# Pearl+'s bridge, part (d): a passed tile is good in the game

`TileAcceptMeans` says what `checkTile` establishes about a passed tile in the code's terms: its openings, the words its
replay forms, its leaf and its rows' rules. This part reads it in the game's: a tile the check passes, whose openings are
its unit's commitment, has `TileWordsGood` at the deployed semantics, or SHA-512 collides (the leaf binds the tile's words
through `sha512/row-seg/v1`). `TileGood` adds the cap and every weight row's rules, which wait on the restated credit set
and acceptance event (`Rules.lean`).
-/

namespace Pouw.PearlPlus.Bridge

open Pouw.PearlC Pouw.Fp8Atom PouwCheck.PearlPlus

/-- Tile `tid` of pass `p` is audit tile `g` of `U`'s layout: the same unit, shape and weight, rows and columns. -/
def TileOf (p : Pass) (tid : ℕ) (U : Workload) (g : ℕ) : Prop :=
  ∃ pos rows cols, p.tile? tid = some (pos, rows, cols) ∧
    g < (auditTiling 64 64 U.layout).NT ∧
    (auditTiling 64 64 U.layout).unit g = p.records[pos]!.matmul.index ∧
    (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).m = p.records[pos]!.matmul.m ∧
    (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).k = p.records[pos]!.matmul.k ∧
    (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).n = p.records[pos]!.matmul.n ∧
    U.layout.wid ((auditTiling 64 64 U.layout).unit g) = p.records[pos]!.matmul.weight ∧
    (auditTiling 64 64 U.layout).rows g = rows.toFinset ∧ (auditTiling 64 64 U.layout).cols g = cols.toFinset

/-- The tile's openings are unit `u`'s commitment `c`: its A rows are `c`'s activation rows and its B rows the weight's
(each word below `2^32`), and its leaf is the digest of `c`'s ticket and useful words on the tile. -/
def OpensCommitted (p : Pass) (tid : ℕ) (o : TileOpening) (U : Workload) (u : ℕ) (c : Commit) : Prop :=
  ∀ pos rows cols, p.tile? tid = some (pos, rows, cols) →
    o.a.map (fun x => openedValue x.value) = rows.map (rowBytesOf c.act (U.layout.shape u).k) ∧
    o.b.map (fun x => openedValue x.value) = cols.map (rowBytesOf (U.weight (U.layout.wid u)) (U.layout.shape u).k) ∧
    openedValue o.leaf.value = Words.leaf ⟨(rows.map fun i => (cols.map fun j => c.ticket i j).toArray).toArray,
      (rows.map fun i => (cols.map fun j => c.useful i j).toArray).toArray⟩ ∧
    (∀ i ∈ rows, ∀ l < (U.layout.shape u).k, c.act i l < 2 ^ 32) ∧
    (∀ j ∈ cols, ∀ l < (U.layout.shape u).k, U.weight (U.layout.wid u) j l < 2 ^ 32) ∧
    ∀ i ∈ rows, ∀ j ∈ cols, c.ticket i j < 2 ^ 32 ∧ c.useful i j < 2 ^ 32

/-- **(d) A passed tile's words are good**: a tile of pass `p` that is audit tile `g` of `U`, under the semantics' salt
and weight root, whose openings are its unit's commitment `c`, and that `checkTile` passes on its weight's noise, has
every word of its credited rows at the reference values, or SHA-512 collides. -/
def TilePassGivesTheWords : Prop :=
  ∀ {Q R S : Type} (saltOf : S → List ℕ) (rootOf : ℕ → List ℕ) (ρ : ℚ) (H : Q → R) (s : S) (p : Pass) (tid : ℕ)
    (o : TileOpening) (U : Workload) (g : ℕ) (c : Commit),
    TileOf p tid U g → bytesNat p.salt = saltOf s →
    bytesNat ((p.weightRoots.lookup (U.layout.wid ((auditTiling 64 64 U.layout).unit g))).getD ByteArray.empty) =
      rootOf (U.layout.wid ((auditTiling 64 64 U.layout).unit g)) →
    OpensCommitted p tid o U ((auditTiling 64 64 U.layout).unit g) c →
    checkTile p (Properties.Pouw.PearlPlus.noiseOf p tid) tid o = none →
      TileWordsGood (protocolH3 saltOf rootOf ρ) (tilesH3 saltOf rootOf ρ) U H s g c ∨
        ∃ x y : ByteArray, x ≠ y ∧ Flock.Sha512.hash x = Flock.Sha512.hash y

end Pouw.PearlPlus.Bridge
