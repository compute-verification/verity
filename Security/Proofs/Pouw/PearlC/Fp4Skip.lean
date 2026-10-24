import Definitions.Pouw.PearlC.Fp4Skip

/-! The proof-only part of `Definitions.Pouw.PearlC.Fp4Skip`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Pouw.PearlC.Fp4Skip

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.Fp4

variable {m n k s : Nat}

/-- **The unit's lone-skip flags**: atom `t` of word `(i, j)`, `C̃` from `+0` over row `i` of `A′` and row `j` of `B̃`
with their scale rows; `Forming.Flags m n`'s type. -/
def unitPureFlags (p : Fp4Atom) (Aq : Matrix (Fin m) (Fin k) Nib) (SA : Matrix (Fin m) (Fin s) (Fin 256))
    (Bq : Matrix (Fin n) (Fin k) Nib) (SB : Matrix (Fin n) (Fin s) (Fin 256)) : Fin m → Fin n → ℕ → Bool :=
  fun i j t => fp4PureFlags p 0 (List.ofFn (Aq i)) (List.ofFn (Bq j)) (List.ofFn (SA i)) (List.ofFn (SB j)) t

/-- The flagged atoms `t < k / p.k` of word `(i, j)`. -/
def flaggedAtoms (p : Fp4Atom) (k : ℕ) (flag : Fin m → Fin n → ℕ → Bool) (i : Fin m) (j : Fin n) : Finset ℕ :=
  (Finset.range (k / p.k)).filter fun t => flag i j t = true

/-- **Term 2's count**: the flagged atom-words of the tile `rows × cols`. -/
def atomWords (p : Fp4Atom) (k : ℕ) (flag : Fin m → Fin n → ℕ → Bool) (rows : Finset (Fin m)) (cols : Finset (Fin n)) :
    ℕ :=
  ∑ i ∈ rows, ∑ j ∈ cols, (flaggedAtoms p k flag i j).card

/-- **Term 2's debit** on the tile `rows × cols`: `c` per flagged atom-word (`c = 64`, `p.k`, at one unit per MAC). -/
def pureDebit (p : Fp4Atom) (k : ℕ) (c : ℚ) (flag : Fin m → Fin n → ℕ → Bool) (rows : Finset (Fin m))
    (cols : Finset (Fin n)) : ℚ :=
  c * atomWords p k flag rows cols

/-- `ρ²` exactly (`pearl_c4._exact_rho_sq`): the mean of `x_l²` over the positions `l = 0, 8, 16, …`. -/
def rhoSq (x : Fin k → ℚ) : ℚ :=
  let ps := (List.finRange k).filter fun l => l.val % 8 = 0
  (ps.map fun l => x l ^ 2).sum / ps.length

/-- The clean amax of k-block `β`: the largest `|x_l|` over its positions `l / p.block = β`, `0` if it has none. -/
def blockAmax (p : Fp4Atom) (x : Fin k → ℚ) (β : ℕ) : ℚ :=
  (((List.finRange k).filter fun l => l.val / p.block = β).map fun l => |x l|).foldl max 0

/-- **`dead(b)`** (D-SS, theory §1; `pearl_c4.dead_blocks`): k-block `β` of the clean row `x` is dead when
`amax_β² > 90·ρ²` (exactly 90, about 9.49ρ; ruled 10:40Z, was 144), i.e. the noise `σ = ρ/4` is below `3/(2√90) ≈ 0.16`
of the block's code step. It is the row rule's screen, the one term 3's soundness needs. A salt-free screen: it reads
the clean row alone. -/
def dead (p : Fp4Atom) (x : Fin k → ℚ) (β : ℕ) : Bool := decide (90 * rhoSq x < blockAmax p x β ^ 2)

/-- **The dead × dead pair** (term 3): k-block `β` is dead on both A's clean row `x` and B's clean row `y`. -/
def deadPair (p : Fp4Atom) (x y : Fin k → ℚ) (β : ℕ) : Bool := dead p x β && dead p y β

/-- **The unit's dead pairs**: k-block `β` of word `(i, j)`, over the clean rows `X i` of `A` and `Y j` of `B`. -/
def unitDeadPairs (p : Fp4Atom) (X : Matrix (Fin m) (Fin k) ℚ) (Y : Matrix (Fin n) (Fin k) ℚ) :
    Fin m → Fin n → ℕ → Bool :=
  fun i j β => deadPair p (X i) (Y j) β

/-- The dead × dead k-blocks `β < k / p.block` of word `(i, j)`. -/
def deadBlocks (p : Fp4Atom) (k : ℕ) (pair : Fin m → Fin n → ℕ → Bool) (i : Fin m) (j : Fin n) : Finset ℕ :=
  (Finset.range (k / p.block)).filter fun β => pair i j β = true

/-- **Term 3's count**: the dead × dead pair-words of the tile `rows × cols`. -/
def pairWords (p : Fp4Atom) (k : ℕ) (pair : Fin m → Fin n → ℕ → Bool) (rows : Finset (Fin m)) (cols : Finset (Fin n)) :
    ℕ :=
  ∑ i ∈ rows, ∑ j ∈ cols, (deadBlocks p k pair i j).card

/-- **Term 3's debit** on the tile `rows × cols`: `c` per dead × dead pair-word (`c = 16`, `p.block` on NVFP4, at one
unit per MAC; `pearl_c4.tile_debit` charges `p.block`). -/
def deadDebit (p : Fp4Atom) (k : ℕ) (c : ℚ) (pair : Fin m → Fin n → ℕ → Bool) (rows : Finset (Fin m))
    (cols : Finset (Fin n)) : ℚ :=
  c * pairWords p k pair rows cols

/-- One operand's k-block `β`: its `p.block` codes from position `β·p.block`, and its scale byte `β`. -/
def blockOf (p : Fp4Atom) (codes : List Nib) (scales : List (Fin 256)) (β : Nat) : List Nib × Option (Fin 256) :=
  ((codes.drop (β * p.block)).take p.block, scales[β]?)

/-- **K-block `β`'s contribution** to a word: the group terms of its `p.block / p.group` groups, as the atom holding it
(`β·p.block / p.k`) reads them from its slices of the two rows (`none` for an invalid scale byte). -/
def blockTerms (p : Fp4Atom) (a b : List Nib) (sa sb : List (Fin 256)) (β : Nat) : List (Option GroupTerm) :=
  let t := β * p.block / p.k
  (List.range (p.block / p.group)).map fun g =>
    p.groupTerm ((a.drop (t * p.k)).take p.k) ((b.drop (t * p.k)).take p.k) ((sa.drop (t * p.scales)).take p.scales)
      ((sb.drop (t * p.scales)).take p.scales) (β * p.block % p.k / p.group + g)

end Pouw.PearlC.Fp4Skip
