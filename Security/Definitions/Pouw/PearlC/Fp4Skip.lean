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

/-! ## Term 3: the dead × dead block pairs -/

/-! ## A block's contribution -/

end Pouw.PearlC.Fp4Skip
