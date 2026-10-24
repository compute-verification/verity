import Definitions.Pouw.PearlC.PeelExactP

/-! The proof-only part of `Definitions.Pouw.PearlC.PeelExactP`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Pouw.PearlC.PeelExactP

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.PeelExact Matrix

variable {m n k : Nat}

/-- **The pure chain on words**: `p.step` over all the atoms from `+0`, each word finite, with no promotion. -/
def pureWordP? (p : Pipeline) (atoms : List AtomIn) : Option Nat :=
  atoms.foldlM (fun w x => finite? (p.step w (List.ofFn x.1) (List.ofFn x.2))) 0

/-- **U with pure chains on the step `p`**: `peelExactP`'s order with `pureWordP? p` for its four E4M3 chains. It is
`peelExactP p 0` (`AccumProofs.peelExactP_zero_eq_pure`). -/
def peelExactPureP (p : Pipeline) (Aq : Matrix (Fin m) (Fin k) Code) (Bq : Matrix (Fin n) (Fin k) Code)
    (FA FB : Matrix (Fin k) (Fin 32) Code) (beA : Matrix (Fin m) (Fin 32) Nat) (beB : Matrix (Fin n) (Fin 32) Nat) :
    Matrix (Fin m) (Fin n) (Option Nat) := fun i j => do
  let c ← pureWordP? p (rowAtoms (Aq i) (Bq j))
  let af ← (List.ofFn fun q => pureWordP? p (rowAtoms (Aq i) (FBᵀ q))).mapM id
  let bf ← (List.ofFn fun q => pureWordP? p (rowAtoms (Bq j) (FAᵀ q))).mapM id
  let g ← (List.ofFn fun q => (List.ofFn fun q' => pureWordP? p (rowAtoms (FBᵀ q) (FAᵀ q'))).mapM id).mapM id
  let pa ← paRow? (List.ofFn (beA i)) af
  let pb ← pbRow? (List.ofFn (beB j)) bf g
  uWord? c pa pb

/-- `peelExactPureP`'s entries as values. -/
def peelExactPureValP (p : Pipeline) (Aq : Matrix (Fin m) (Fin k) Code) (Bq : Matrix (Fin n) (Fin k) Code)
    (FA FB : Matrix (Fin k) (Fin 32) Code) (beA : Matrix (Fin m) (Fin 32) Nat) (beB : Matrix (Fin n) (Fin 32) Nat) :
    Matrix (Fin m) (Fin n) (Option ℚ) := fun i j => (peelExactPureP p Aq Bq FA FB beA beB i j).bind wordQ

end Pouw.PearlC.PeelExactP
