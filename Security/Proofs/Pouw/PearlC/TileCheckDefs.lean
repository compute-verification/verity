import Definitions.Pouw.PearlC.TileCheck

/-! The proof-only part of `Definitions.Pouw.PearlC.TileCheck`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Pouw.PearlC

open Finset Pouw.Fp8Atom

variable {Q R S : Type}

/-- The registered values of unit `u` in `τ`: its commitment's, and zero for a blank unit, whose tiles are never
checked. -/
def TileCheck.valueAt (TC : TileCheck Q R S) (U : Workload) (τ : Transcript) (u : ℕ) :
    (TC.C (U.layout.shape u)).Transcript :=
  match τ u with
  | some c => TC.rowsOf U u c
  | none => fun _ => 0

/-- **L3: the tile check computes the good-tile predicate.** On every in-domain workload, for every oracle, salt, tile
`g` and commitment `c` of `g`'s unit, a witness that `g`'s check accepts makes `g` good under `c`, unless the
registered values fail to bind the circuit's words (`Collides`). On a transcript `τ` whose unit `(TR.tiling U.layout).unit
g` is `c`, `g` is then in `goodTiles`; a tile of a blank unit is in `blankTiles` whatever the witness. -/
def TileCheck.Computes (TC : TileCheck Q R S) (P : Protocol Q R S) (TR : TileRules Q R S) (D : Layout → Prop) : Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ (H : Q → R) (s : S) (g : ℕ), g < (TR.tiling U.layout).NT → ∀ (c : Commit) X,
    TC.Accepts P TR H s U g c X →
      TileGood P TR U H s g c ∨
        TC.Collides U ((TR.tiling U.layout).unit g) ((TR.tiling U.layout).rows g) ((TR.tiling U.layout).cols g) c X

/-- **L3's cap**: an accepted tile's committed activations keep its replayed debit within the cap. The Program has no
debit or cap yet (the replay shortcut, `PROTOCOL.md`), so its check leaves the activations' debit free: this holds of it
only once a cap unit returns the cap's verdict on a public wire, as under hiding it must, since the verifier never sees
the activations. -/
def TileCheck.ComputesCap (TC : TileCheck Q R S) (P : Protocol Q R S) (TR : TileRules Q R S) (D : Layout → Prop) :
    Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ (H : Q → R) (s : S) (g : ℕ), g < (TR.tiling U.layout).NT → ∀ (c : Commit) X,
    TC.Accepts P TR H s U g c X → TR.capOK H s U g c.act

/-- **L3's words**: an accepted tile's checked words have the reference values on its credited rows, unless the
registered values fail to bind the circuit's words. -/
def TileCheck.ComputesWords (TC : TileCheck Q R S) (P : Protocol Q R S) (TR : TileRules Q R S) (D : Layout → Prop) :
    Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ (H : Q → R) (s : S) (g : ℕ), g < (TR.tiling U.layout).NT → ∀ (c : Commit) X,
    TC.Accepts P TR H s U g c X →
      TileWordsGood P TR U H s g c ∨
        TC.Collides U ((TR.tiling U.layout).unit g) ((TR.tiling U.layout).rows g) ((TR.tiling U.layout).cols g) c X

end Pouw.PearlC
