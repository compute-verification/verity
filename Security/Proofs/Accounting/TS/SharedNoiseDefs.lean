import Definitions.Accounting.TS.SharedNoise

/-! The proof-only part of `Definitions.Accounting.TS.SharedNoise`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

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

end Accounting.TS
