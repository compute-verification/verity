import Definitions.Accounting.TS.Receipt

/-!
# One noise seed a call: the call's coin and a tile's own part of it (trusted)

J10 (note:compute-accounting/20261019T0825Z-draft-j10-shared-noise, a draft for the red team): H1 + fx1's noise derived
once a call row and once a call column from one seed a call, where `h1tile` derives it once a tile, with each tile's
fold keys still its own (σ_R of its position, decision 121). A tile then shares its rows' noise with its call row's
other tiles and its columns' noise with its call column's, so no part of the call's noise is a tile's own; its keys
are.

* `CallCoin m n Rw Cl Ky`: an `m × n` grid of tiles under one seed a call: each call row block's noise `Rw` (its rows'
  dither bits and digit coins), each call column block's `Cl` (its columns' coin bits and `U`), and each tile's fold
  keys `Ky`.
* `CallRest` and `CallCoin.keySplit t`: the call's coin as tile `t`'s keys and the rest (every block's noise and every
  other tile's keys).
* `ReceiptGame.CondReceiptLuck e pr`: `ReceiptLuck pr` with the coin split by `e` into a rest and an own part, at every
  value of the rest, over the own part alone. `CondOperandLuck` is `OperandLuck` so.
-/

namespace Accounting.TS

open Pouw.RelativeSecurity

/-- **The call's coin under one noise seed a call**: `m` call row blocks' noise, `n` call column blocks' noise, and
each of the `m × n` tiles' fold keys. -/
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

/-- **The operand luck at `p` given the rest**: every round's share lies in `[0, 1]`, and at every round, whatever the
past, and at every value `z` of the rest, a deficient round's deficient words are all right with probability at most
`p` over the own part alone. Under one seed a tile it is (H-μ) given the other tiles' coins, the tile's own coin its
own part; under one seed a call no part of the noise is a tile's own, and (H-μ) is taken over the call's whole coin. -/
def CondOperandLuck {Z O : Type} [Fintype O] (e : Ω ≃ Z × O) (p : ℝ) : {n : ℕ} → ReceiptGame Ω n → Prop
  | _, .done => True
  | _, .round d _ right _ next =>
      (∀ ω, 0 ≤ d ω ∧ d ω ≤ 1) ∧
        (∀ z, prob (fun o => right (e.symm (z, o)) ∧ 0 < d (e.symm (z, o))) ≤
          p * prob (fun o => 0 < d (e.symm (z, o)))) ∧
        ∀ ω, CondOperandLuck e p (next ω)

end ReceiptGame

end Accounting.TS
