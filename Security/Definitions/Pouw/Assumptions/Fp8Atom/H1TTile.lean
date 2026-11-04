import Mathlib.Data.Set.Card
import Definitions.Pouw.Assumptions.Fp8Atom.H1TGamma

/-!
# H-1T's tiles (trusted statements)

The per-tile count statement `TTH1T` (`H1TTileAssumptions`)
reads a tiling of the units' outputs and a program's fixed output slots, defined here.

* **`Tiling`**: every output `(i, j)` of unit `u` lies in one tile; every tile holds an output, lies in one unit, and
  is a block of that unit's rows and columns. A credited word's tile is its output's (`Tiling.of`). `AtLeast r c`, the
  domain `TTH1T` reads: every tile spans at least `r` rows and `c` columns.
* **`TileRight`**: at the joint salt `G`, every credited word of the tile equals its fixed output slot, a register of
  the program (`slotVal`; `0` past the last register). `rightWords` counts the credited words of the right tiles, so
  `32·rightWords` is `Σ_{c right} W c` with `W c = 32·|credited words of c|`. The useful output `R_(2T−1)` and
  `R_(2T−2)` are not credited (`Pos`), so "right" does not read them.
* **`TTH1TAllRight`** (M5's sanity statement): a program right on every tile at every salt is never in `TTH1T`'s event
  at `γ = 0`. It is `H1TGamma` for such a program, so it holds under `H1TRunningWords` (`tth1tAllRight`).
-/

namespace Pouw.Fp8Atom.H1T

open Pouw.Dimension.H100

/-- A tiling of units `Us`' outputs by the tiles `C`: output `(i, j)` of unit `u` lies in tile `tile u i j`. Every
tile holds an output (`onto`) and lies in one unit (`unit`), and it is a block of that unit's rows and columns: if it
holds outputs `(i, j)` and `(i', j')`, it holds `(i, j')` (`block`). -/
structure Tiling {N : ℕ} (Us : Fin N → Unit) (C : Type) where
  tile : (u : Fin N) → Fin (Us u).m → Fin (Us u).n → C
  onto : ∀ c, ∃ (u : Fin N) (i : Fin (Us u).m) (j : Fin (Us u).n), tile u i j = c
  unit : ∀ (u u' : Fin N) (i : Fin (Us u).m) (i' : Fin (Us u').m) (j : Fin (Us u).n) (j' : Fin (Us u').n),
    tile u i j = tile u' i' j' → u = u'
  block : ∀ (u : Fin N) (i i' : Fin (Us u).m) (j j' : Fin (Us u).n), tile u i j = tile u i' j' →
    tile u i j' = tile u i j

/-- The tile of credited word `x`: its output's. -/
def Tiling.of {N : ℕ} {Us : Fin N → Unit} {C : Type} (tl : Tiling Us C) (x : Idx Us) : C :=
  tl.tile x.1 x.2.1 x.2.2.1

/-- Every tile holds an `r × c` block of its unit's outputs: `r` rows and `c` columns, all of whose outputs lie in the
tile. Sampled proofs' layout A tiles by `16 × 16` blocks, so its tilings have `AtLeast 16 16`. -/
def Tiling.AtLeast {N : ℕ} {Us : Fin N → Unit} {C : Type} (tl : Tiling Us C) (r c : ℕ) : Prop :=
  ∀ t : C, ∃ (u : Fin N) (rows : Finset (Fin (Us u).m)) (cols : Finset (Fin (Us u).n)),
    r ≤ rows.card ∧ c ≤ cols.card ∧ ∀ i ∈ rows, ∀ j ∈ cols, tl.tile u i j = t

/-- Credited word `x`'s fixed output slot: register `slot x` of the program's registers `rs`, or `0` if there is
none. -/
def slotVal {N : ℕ} {Us : Fin N → Unit} (rs : List (MSalt Us → ℚ)) (slot : Idx Us → ℕ) (x : Idx Us) :
    MSalt Us → ℚ :=
  rs.getD (slot x) fun _ => 0

/-- Tile `c` is right at the joint salt `G`: each of its credited words equals its output slot at `G`. -/
def TileRight {N : ℕ} {Us : Fin N → Unit} {C : Type} (rs : List (MSalt Us → ℚ)) (slot : Idx Us → ℕ)
    (tl : Tiling Us C) (c : C) (G : MSalt Us) : Prop :=
  ∀ x : Idx Us, tl.of x = c → slotVal rs slot x G = mword Us x G

/-- The credited words of the tiles right at `G`: the right tiles, counted by their credited words. -/
noncomputable def rightWords {N : ℕ} {Us : Fin N → Unit} {C : Type} (rs : List (MSalt Us → ℚ)) (slot : Idx Us → ℕ)
    (tl : Tiling Us C) (G : MSalt Us) : ℕ :=
  {x : Idx Us | TileRight rs slot tl (tl.of x) G}.ncard

/-- **`TTH1T`'s event is consistent with `H1TGamma`** (M5's sanity statement): under `H32`, a well-formed `MH100w`
program (draft semantics) on free inputs that is right on every tile at every salt costs at least 32 per credited word
of the right tiles, at every salt. So at `γ = 0` it is never in `TTH1T`'s event. -/
def TTH1TAllRight : Prop :=
  ∀ pr : Prices100, H32 pr → ∀ (N : ℕ) (Us : Fin N → Unit), (∀ u, (Us u).Legal) →
    ∀ (C : Type) (tl : Tiling Us C) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (MSalt Us → ℚ))
      (P : List (Op100 (MSalt Us))) (slot : Idx Us → ℕ),
      (∀ f ∈ L, f ∈ mfree Us) → (∀ o ∈ P, o.WFw pr (mfree Us)) →
      (∀ G c, TileRight (regs100 fpSem P L) slot tl c G) →
      ∀ G, ¬ (cost100 pr P : ℚ) < 32 * rightWords (regs100 fpSem P L) slot tl G

end Pouw.Fp8Atom.H1T
