import Definitions.Pouw.PearlC.PeelFp4Vectors
import Proofs.Pouw.PearlC.PeelFp4Proofs

/-! # `PeelFp4Checks`' rows, operands, bases and noise lines, which its parts read -/

namespace Pouw.PearlC.PeelFp4Checks

open Pouw.PearlC Pouw.PearlC.Fp4 Pouw.PearlC.Fp4Dev Pouw.PearlC.PeelExact Pouw.PearlC.PeelFp4Proofs
  Pouw.PearlC.PeelFp4Vectors

/-- Selected row `i` of `A′`: its `k` codes and `k / block` scale bytes. -/
def rowA (p : Fp4Atom) (K : PeelCase) (i : Nat) : Row4 :=
  (nibs K.k (K.a.getD i (0, 0)).1, bytes (K.k / p.block) (K.a.getD i (0, 0)).2)

/-- Selected row `j` of `B̃`. -/
def rowB (p : Fp4Atom) (K : PeelCase) (j : Nat) : Row4 :=
  (nibs K.k (K.b.getD j (0, 0)).1, bytes (K.k / p.block) (K.b.getD j (0, 0)).2)

/-- The four selected rows of `A′` as an operand. -/
def opA (p : Fp4Atom) (K : PeelCase) : Op4 4 := fun i => some (rowA p K i.val)

/-- The four selected rows of `B̃` as an operand. -/
def opB (p : Fp4Atom) (K : PeelCase) : Op4 4 := fun j => some (rowB p K j.val)

/-- A basis from its 32 columns, `k` nibbles each. -/
def basis (k : Nat) (cols : List Nat) : Matrix (Fin k) (Fin 32) Nib :=
  fun t q => ⟨(cols.getD q 0 >>> (4 * t.val)) % 16, Nat.mod_lt _ (by decide)⟩

/-- A byte. -/
def byte (s : Nat) : Fin 256 := ⟨s % 256, Nat.mod_lt _ (by decide)⟩

/-- The selected rows' `Ẽ_A = eTilde E_A a_E`. -/
def beA (K : PeelCase) : Matrix (Fin 4) (Fin 32) Nat :=
  fun i => eTilde (line (K.ea.getD i (0, 0)).1) (byte (K.ea.getD i (0, 0)).2)

/-- The selected rows' `Ẽ_B`. -/
def beB (K : PeelCase) : Matrix (Fin 4) (Fin 32) Nat :=
  fun j => eTilde (line (K.eb.getD j (0, 0)).1) (byte (K.eb.getD j (0, 0)).2)

/-- Entry `(i, j)` of a list of rows. -/
def cell (M : List (List Nat)) (i j : Nat) : Nat := (M.getD i []).getD j 0

end Pouw.PearlC.PeelFp4Checks
