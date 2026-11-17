module

public import Definitions.Pouw.PearlC.Defs

/-! Π's promoted chain over a `Pipeline` with the atom's step a parameter; it imports no proofs, so the named protocol may read it. -/

@[expose] public section

namespace Pouw.PearlC.Accum

open Pouw.Fp8Atom Pouw.PearlC Matrix

variable {m n k : Nat}

/-- A promotion group's word on the step `p`: `p.step` folded over the group's atoms, starting from `+0`. -/
def groupWordP (p : Pipeline) (g : List AtomIn) : Nat :=
  g.foldl (fun w x => p.step w (List.ofFn x.1) (List.ofFn x.2)) 0

/-- The promoted total of a chain given by its groups, on the step `p`. -/
def totalP (p : Pipeline) (gs : List (List AtomIn)) : Option ℚ := (gs.map (groupWordP p)).foldl promote (some 0)

/-- The chain over a list of atoms on the step `p`, cut into groups of `G`. -/
def chainP (p : Pipeline) (G : Nat) (atoms : List AtomIn) : Option ℚ := totalP p (atoms.toChunks G)

/-- `C̃ = A′·B̃ᵀ` under the promoted chain on the step `p`: entry `(i, j)` is the chain over row `i` of `A′` and row
`j` of `B̃`. -/
def ctChainP (p : Pipeline) (G : Nat) (Aq : Matrix (Fin m) (Fin k) Code) (Bq : Matrix (Fin n) (Fin k) Code) :
    Matrix (Fin m) (Fin n) (Option ℚ) := fun i j => chainP p G (rowAtoms (Aq i) (Bq j))

/-- **The ticket chain at a device record** (`PearlCDevice.ticket`): `ctChainP` on the device's step `p`, promoted every
`G` atoms. At `G = 0`, `List.toChunks 0` keeps all of a row's atoms in one group, whose word is promoted once from `0`;
that promotion is exact, so `chainAt p 0` is the pure chain `ctPureChain p` (`AccumProofs.chainAt_zero_eq_pure`). -/
abbrev chainAt (p : Pipeline) (G : Nat) (Aq : Matrix (Fin m) (Fin k) Code) (Bq : Matrix (Fin n) (Fin k) Code) :
    Matrix (Fin m) (Fin n) (Option ℚ) := ctChainP p G Aq Bq

/-- **The pure chain** on the step `p`: `p.step` folded over all the atoms from `+0`, with no promotion, its word read as
a value. -/
def pureChain (p : Pipeline) (atoms : List AtomIn) : Option ℚ :=
  wordQ (atoms.foldl (fun w x => p.step w (List.ofFn x.1) (List.ofFn x.2)) 0)

def ctPureChain (p : Pipeline) (Aq : Matrix (Fin m) (Fin k) Code) (Bq : Matrix (Fin n) (Fin k) Code) :
    Matrix (Fin m) (Fin n) (Option ℚ) := fun i j => pureChain p (rowAtoms (Aq i) (Bq j))

end Pouw.PearlC.Accum
