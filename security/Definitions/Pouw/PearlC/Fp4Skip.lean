module

public import Definitions.Pouw.PearlC.Fp4Chain

/-! Terms 2 and 3 of TT_OUT-FP4's replayed debit, the lone-skip flags and the dead × dead block pairs; it imports no proofs, so the device record may read it. -/

@[expose] public section

namespace Pouw.PearlC.Fp4Skip

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.Fp4

/-- `Fp4Atom.chainN` with the atoms `ms` flags skipped in place: a skipped atom's incoming word passes through; past the
end of `ms`, every atom runs. -/
def chainMaskN (p : Fp4Atom) :
    Nat → List Bool → Nat → List Nib → List Nib → List (Fin 256) → List (Fin 256) → Option Nat
  | 0, _, c, _, _, _, _ => some c
  | n + 1, ms, c, a, b, sa, sb => do
    let d ← if ms.headD false then some c
      else p.step c (a.take p.k) (b.take p.k) (sa.take p.scales) (sb.take p.scales)
    chainMaskN p n ms.tail d (a.drop p.k) (b.drop p.k) (sa.drop p.scales) (sb.drop p.scales)

/-- `Fp4Atom.chain` with the atoms `ms` flags skipped in place; `none` if the lengths don't fit. -/
def chainMask (p : Fp4Atom) (ms : List Bool) (c : Nat) (a b : List Nib) (sa sb : List (Fin 256)) : Option Nat :=
  if b.length = a.length ∧ a.length % p.k = 0 ∧ sa.length = a.length / p.k * p.scales ∧
      sb.length = a.length / p.k * p.scales then
    chainMaskN p (a.length / p.k) ms c a b sa sb
  else none

/-- The mask of atom `t` alone. -/
def loneMask (t : Nat) : List Bool := List.replicate t false ++ [true]

/-- The chain from `c` with atom `t` skipped alone, in place. -/
def skipOne (p : Fp4Atom) (t c : Nat) (a b : List Nib) (sa sb : List (Fin 256)) : Option Nat :=
  chainMask p (loneMask t) c a b sa sb

/-- **The lone-skip flags** (term 2; `inert_work_pure` at the FP4 atom): atom `t < K/k` of the chain from `c` is flagged when
the chain is in the model and skipping `t` alone, in place, leaves its final word unchanged. -/
def fp4PureFlags (p : Fp4Atom) (c : Nat) (a b : List Nib) (sa sb : List (Fin 256)) (t : Nat) : Bool :=
  decide (t < a.length / p.k) && (p.chain c a b sa sb).isSome &&
    decide (skipOne p t c a b sa sb = p.chain c a b sa sb)

variable {m n k s : Nat}

def unitPureFlags (p : Fp4Atom) (Aq : Matrix (Fin m) (Fin k) Nib) (SA : Matrix (Fin m) (Fin s) (Fin 256))
    (Bq : Matrix (Fin n) (Fin k) Nib) (SB : Matrix (Fin n) (Fin s) (Fin 256)) : Fin m → Fin n → ℕ → Bool :=
  fun i j t => fp4PureFlags p 0 (List.ofFn (Aq i)) (List.ofFn (Bq j)) (List.ofFn (SA i)) (List.ofFn (SB j)) t

def flaggedAtoms (p : Fp4Atom) (k : ℕ) (flag : Fin m → Fin n → ℕ → Bool) (i : Fin m) (j : Fin n) : Finset ℕ :=
  (Finset.range (k / p.k)).filter fun t => flag i j t = true

def atomWords (p : Fp4Atom) (k : ℕ) (flag : Fin m → Fin n → ℕ → Bool) (rows : Finset (Fin m)) (cols : Finset (Fin n)) :
    ℕ :=
  ∑ i ∈ rows, ∑ j ∈ cols, (flaggedAtoms p k flag i j).card

/-- `c = 64` (`p.k`), at one unit per MAC. -/
def pureInertWork (p : Fp4Atom) (k : ℕ) (c : ℚ) (flag : Fin m → Fin n → ℕ → Bool) (rows : Finset (Fin m))
    (cols : Finset (Fin n)) : ℚ :=
  c * atomWords p k flag rows cols

def rhoSq (x : Fin k → ℚ) : ℚ :=
  let ps := (List.finRange k).filter fun l => l.val % 8 = 0
  (ps.map fun l => x l ^ 2).sum / ps.length

def blockAmax (p : Fp4Atom) (x : Fin k → ℚ) (β : ℕ) : ℚ :=
  (((List.finRange k).filter fun l => l.val / p.block = β).map fun l => |x l|).foldl max 0

/-- Exactly 90, the row rule's screen and the one term 3's soundness needs; it reads the clean row alone, so it is salt-free. -/
def dead (p : Fp4Atom) (x : Fin k → ℚ) (β : ℕ) : Bool := decide (90 * rhoSq x < blockAmax p x β ^ 2)

def deadPair (p : Fp4Atom) (x y : Fin k → ℚ) (β : ℕ) : Bool := dead p x β && dead p y β

def unitDeadPairs (p : Fp4Atom) (X : Matrix (Fin m) (Fin k) ℚ) (Y : Matrix (Fin n) (Fin k) ℚ) :
    Fin m → Fin n → ℕ → Bool :=
  fun i j β => deadPair p (X i) (Y j) β

def deadBlocks (p : Fp4Atom) (k : ℕ) (pair : Fin m → Fin n → ℕ → Bool) (i : Fin m) (j : Fin n) : Finset ℕ :=
  (Finset.range (k / p.block)).filter fun β => pair i j β = true

def pairWords (p : Fp4Atom) (k : ℕ) (pair : Fin m → Fin n → ℕ → Bool) (rows : Finset (Fin m)) (cols : Finset (Fin n)) :
    ℕ :=
  ∑ i ∈ rows, ∑ j ∈ cols, (deadBlocks p k pair i j).card

/-- `c = 16` (`p.block` on NVFP4), at one unit per MAC, as `pearl_c4.tile_inert_work` charges. -/
def deadInertWork (p : Fp4Atom) (k : ℕ) (c : ℚ) (pair : Fin m → Fin n → ℕ → Bool) (rows : Finset (Fin m))
    (cols : Finset (Fin n)) : ℚ :=
  c * pairWords p k pair rows cols

def blockOf (p : Fp4Atom) (codes : List Nib) (scales : List (Fin 256)) (β : Nat) : List Nib × Option (Fin 256) :=
  ((codes.drop (β * p.block)).take p.block, scales[β]?)

def blockTerms (p : Fp4Atom) (a b : List Nib) (sa sb : List (Fin 256)) (β : Nat) : List (Option GroupTerm) :=
  let t := β * p.block / p.k
  (List.range (p.block / p.group)).map fun g =>
    p.groupTerm ((a.drop (t * p.k)).take p.k) ((b.drop (t * p.k)).take p.k) ((sa.drop (t * p.scales)).take p.scales)
      ((sb.drop (t * p.scales)).take p.scales) (β * p.block % p.k / p.group + g)

end Pouw.PearlC.Fp4Skip
