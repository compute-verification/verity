module

public import Definitions.Pouw.PearlC.PeelExact

/-! `PeelExact`'s U with the E4M3 chains' step and window as parameters, so that a device whose FP8 step is another `Pipeline` has its own U; it imports no proofs, so the named protocol may read it. -/

@[expose] public section

namespace Pouw.PearlC.PeelExactP

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.PeelExact Matrix

variable {m n k : Nat}

/-- One window's word on the step `p`: its atoms' `p.step`s from `+0`, each word finite. -/
def windowWordP? (p : Pipeline) (g : List AtomIn) : Option Nat :=
  g.foldlM (fun w x => finite? (p.step w (List.ofFn x.1) (List.ofFn x.2))) 0

/-- **The chain on the step `p`**: the atoms in windows of `G`, each window's word promoted in order into a total from
`+0`. -/
def chainWordP? (p : Pipeline) (G : Nat) (atoms : List AtomIn) : Option Nat :=
  (atoms.toChunks G).foldlM (fun t g => (windowWordP? p g).bind (promoteWord? t)) 0

/-- **U under the pinned order, on the step `p` in windows of `G`**: `PeelExact.peelExact` with `chainWordP? p G` for
`C̃`, `A′·F_B`, `B̃·F_A` and `G = F_Bᵀ·F_A`; the factor rows and the four `k16Step`s unchanged. -/
def peelExactP (p : Pipeline) (G : Nat) (Aq : Matrix (Fin m) (Fin k) Code) (Bq : Matrix (Fin n) (Fin k) Code)
    (FA FB : Matrix (Fin k) (Fin 32) Code) (beA : Matrix (Fin m) (Fin 32) Nat) (beB : Matrix (Fin n) (Fin 32) Nat) :
    Matrix (Fin m) (Fin n) (Option Nat) := fun i j => do
  let c ← chainWordP? p G (rowAtoms (Aq i) (Bq j))
  let af ← (List.ofFn fun q => chainWordP? p G (rowAtoms (Aq i) (FBᵀ q))).mapM id
  let bf ← (List.ofFn fun q => chainWordP? p G (rowAtoms (Bq j) (FAᵀ q))).mapM id
  let g ← (List.ofFn fun q => (List.ofFn fun q' => chainWordP? p G (rowAtoms (FBᵀ q) (FAᵀ q'))).mapM id).mapM id
  let pa ← paRow? (List.ofFn (beA i)) af
  let pb ← pbRow? (List.ofFn (beB j)) bf g
  uWord? c pa pb

/-- `peelExactP`'s entries as values: `none` where a step leaves the finite range. -/
def peelExactValP (p : Pipeline) (G : Nat) (Aq : Matrix (Fin m) (Fin k) Code) (Bq : Matrix (Fin n) (Fin k) Code)
    (FA FB : Matrix (Fin k) (Fin 32) Code) (beA : Matrix (Fin m) (Fin 32) Nat) (beB : Matrix (Fin n) (Fin 32) Nat) :
    Matrix (Fin m) (Fin n) (Option ℚ) := fun i j => (peelExactP p G Aq Bq FA FB beA beB i j).bind wordQ

def pureWordP? (p : Pipeline) (atoms : List AtomIn) : Option Nat :=
  atoms.foldlM (fun w x => finite? (p.step w (List.ofFn x.1) (List.ofFn x.2))) 0

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

def peelExactPureValP (p : Pipeline) (Aq : Matrix (Fin m) (Fin k) Code) (Bq : Matrix (Fin n) (Fin k) Code)
    (FA FB : Matrix (Fin k) (Fin 32) Code) (beA : Matrix (Fin m) (Fin 32) Nat) (beB : Matrix (Fin n) (Fin 32) Nat) :
    Matrix (Fin m) (Fin n) (Option ℚ) := fun i j => (peelExactPureP p Aq Bq FA FB beA beB i j).bind wordQ

end Pouw.PearlC.PeelExactP
