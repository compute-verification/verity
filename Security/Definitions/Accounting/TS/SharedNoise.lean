import Definitions.Accounting.TS.Receipt

/-!
# One noise seed a call: the call's coin and a tile's own part of it (trusted)

J10: H1 + fx1's noise derived from one seed a call, where
`h1tile` derives it once a tile. Banded, a call row block's row streams are shared along a band of at most 64 column
tiles, its column streams carry the row block, and each tile's fold keys stay its own (σ_R of its position, decision
121).

* `CallCoin m n Rw Cl Ky`: an `m × n` grid of tiles under one seed a call: each row block's noise `Rw` (its rows'
  dither bits and digit coins), each column block's `Cl` (its columns' coin bits and `U`), and each tile's fold keys
  `Ky`.
* `CallRest` and `CallCoin.keySplit t`: the call's coin as tile `t`'s keys and the rest (every block's noise and every
  other tile's keys).
* `ReceiptGame.CondReceiptLuck e pr`: `ReceiptLuck pr` with the coin split by `e` into a rest and an own part, at every
  value of the rest, over the own part alone.

**No new premise.** The composed statements read three luck premises per tile, each stated over the
stage's whole coin `C`, so their Lean form already allows tiles whose coins correlate. Under one seed a call they are
read at `C = CallCoin`, at the values they have today:

* `PerTile … OperandHMu` at `μ = ¾` (`accounting-ts-operand-h-mu`). No part of the noise is a tile's own, so it is the
  point mass over the call's coin, its view free to read every neighbour's noise and every word the cheater computed.
  The conditional form (given the rest, over a tile's own part) would be false: a tile's own part is its keys, which
  its skipped values don't read.
* `PerTile … ReceiptLuckAt` at `r_B = ½` (`accounting-ts-receipt-luck`), the marginal `ReceiptLuck` over the call's
  coin. The argument for it is the keys': fx1's word step is a bijection of the state and the keys are σ_R's stream at
  the tile's own position, so a tile's receipt keeps its luck given every block's noise and every other tile's keys,
  over its keys alone (`CondReceiptLuck` at `keySplit t`), and `Guarantees.TS.CondReceiptLuckMarginal` averages that
  into the marginal. The conditional form is strictly stronger, so it is the argument, not the premise.
* `PerTile … GuessLuckAt` at `p0 = ½` (`accounting-compose-guess-luck`), word for word: the fold state starts from the
  tile's keys.

What changes is the scope of those three claims (one seed a call, rows shared along a band), not their statements.
That scope is approved, with the same values, and served only once the gain is measured.

**The bits.** `CallCoin` fixes what is shared, not how it is derived. `h1tile.banded_noise`
(`verity/ml/protocols/pouw/circuit/h1tile.py`) is the bit-exact derivation. Bands are g ≤ 64 column tiles, cut from
column tile 0. Tile `(I, J)` reads its rows' noise from the seed at the position of its band's first tile, and its
columns' noise and its fold keys from its own position. A call is then one `CallCoin 1 g` per row block and band, each
from its own seed.
-/

namespace Accounting.TS

open Pouw.RelativeSecurity

/-- **The call's coin under one noise seed a call**: `m` row blocks' noise, `n` column blocks' noise, and each of the
`m × n` tiles' fold keys. -/
abbrev CallCoin (m n : ℕ) (Rw Cl Ky : Type) : Type := (Fin m → Rw) × (Fin n → Cl) × (Fin m × Fin n → Ky)

/-- Everything of the call's coin but tile `t`'s keys: the blocks' noise and the other tiles' keys. -/
abbrev CallRest (m n : ℕ) (Rw Cl Ky : Type) (t : Fin m × Fin n) : Type :=
  ((Fin m → Rw) × (Fin n → Cl)) × ({s : Fin m × Fin n // s ≠ t} → Ky)

/-- The call's coin as the rest and tile `t`'s keys. -/
def CallCoin.keySplit {m n : ℕ} {Rw Cl Ky : Type} (t : Fin m × Fin n) :
    CallCoin m n Rw Cl Ky ≃ CallRest m n Rw Cl Ky t × Ky :=
  ((Equiv.prodAssoc _ _ _).symm.trans
      (Equiv.prodCongr (Equiv.refl _) ((Equiv.piSplitAt t fun _ => Ky).trans (Equiv.prodComm _ _)))).trans
    (Equiv.prodAssoc _ _ _).symm

namespace ReceiptGame

variable {Ω : Type}

/-- **The receipt luck at `pr` given the rest**: at every round, whatever the past, and at every value `z` of the
rest, a deficient round whose deficient words are not all right keeps the honest committed value with probability at
most `pr` over the own part `o` alone (`ω = e.symm (z, o)`). At `e = CallCoin.keySplit t` the own part is tile `t`'s
fold keys, every block's noise and every other tile's keys fixed. -/
def CondReceiptLuck {Z O : Type} [Fintype O] (e : Ω ≃ Z × O) (pr : ℝ) : {n : ℕ} → ReceiptGame Ω n → Prop
  | _, .done => True
  | _, .round d _ right same next =>
      (∀ z, prob (fun o => same (e.symm (z, o)) ∧ ¬right (e.symm (z, o)) ∧ 0 < d (e.symm (z, o))) ≤
          pr * prob (fun o => ¬right (e.symm (z, o)) ∧ 0 < d (e.symm (z, o)))) ∧
        ∀ ω, CondReceiptLuck e pr (next ω)

end ReceiptGame

end Accounting.TS
