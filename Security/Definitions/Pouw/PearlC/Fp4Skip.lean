import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Definitions.Pouw.PearlC.Fp4Chain

/-!
# Pearl-C4's replayed debit: the lone-skip flags and the dead × dead block pairs (definitions only)

Terms 2 and 3 of TT_OUT-FP4's debit (`internal/pouw/rtx-pro/theory-pearl-c4-domain.md` §2), over any `Fp4Atom` and its
chain `Fp4Atom.chain` (no promotion): `sm120Nvf4` for `pearl-c-nvfp4-v0`, `sm120Mxf4` for `pearl-c-mxfp4-v0`. Their
soundness, and the exactness lemma that ties term 2 to zero-sum atoms, are in `Pouw.PearlC.Fp4Freeze`; this module
imports no proofs, so the device record may read it.

* **Term 2, the lone-skip flags** (`fp4PureFlags`): P1's `debit_pure` at the FP4 atom, the FP4 twin of `flagsAt P 0`.
  Atom `t` of a chain is flagged when the chain is in the model and skipping `t` alone, in place (its incoming word
  passes through, `skipOne`), leaves the chain's final word unchanged. Computed by replay (`chainMask`).
* **Term 3, the dead × dead pairs** (`dead`, `deadPair`): D-SS's `dead(b)` (theory §1; `pearl_c4.dead_blocks`), k-block
  `β` of a clean row `x` is dead when `amax_β² > 90·ρ²` (exactly 90, about 9.49ρ: the row rule's screen, ruled 10:40Z;
  was 144), with `ρ²` the exact mean of `x_l²` over `l = 0, 8, 16, …`
  (`rhoSq`, `pearl_c4._exact_rho_sq`). It reads the clean row alone, so it is salt-free. A pair is dead on both A's row
  and B's row.
* **The counts** (`atomWords`, `pairWords`) over a tile `rows × cols` (`Finset.univ` for the unit), and the debits at a
  price per atom-word (`pureDebit`, 64 at one unit per MAC) and per pair-word (`deadDebit`, 16 on NVFP4).
* **A block's contribution** (`blockTerms`): the group terms of k-block `β`'s groups, as the atom holding it reads them;
  what a dead × dead pair lets a program precompute. `blockOf` is one operand's codes and scale byte of the block.
-/

namespace Pouw.PearlC.Fp4Skip

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.Fp4

/-! ## Term 2: the lone-skip flags -/

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

/-- **The lone-skip flags** (term 2; `debit_pure` at the FP4 atom): atom `t < K/k` of the chain from `c` is flagged when
the chain is in the model and skipping `t` alone, in place, leaves its final word unchanged. -/
def fp4PureFlags (p : Fp4Atom) (c : Nat) (a b : List Nib) (sa sb : List (Fin 256)) (t : Nat) : Bool :=
  decide (t < a.length / p.k) && (p.chain c a b sa sb).isSome &&
    decide (skipOne p t c a b sa sb = p.chain c a b sa sb)

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

/-! ## Term 3: the dead × dead block pairs -/

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

/-! ## A block's contribution -/

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
