import Definitions.Pouw.Eps1.M3

/-!
# Dither's CUDA-core lower bounds in model M3 (trusted)

EPS1-QSTAR's proved lower bounds (note:20261017T0344Z-draft-eps1-qstar, (b)), over `Pouw.Eps1`'s model M3, each to
be proved as `Pouw.SecurityProofs.Eps1.G` in `Proofs.Pouw.Eps1`; until then `lean-audit.json` lists none under
`guarantees`, and each under `shape.frozen` as `unstated`:

* `WordLocalZero`: Theorem B at the zero instance, a word-local program costs at least 4 instructions a word;
* `WordLocalGeneral`: Theorem B at a general instance, at least 8 a word (met by dp4a's 8 IDP.4A);
* `ClassPPacked`: Theorem C, a radix-packed program costs at least `1 + 28/27 ≈ 2.04` a word.

Free words (functions of one side alone) cost nothing, and the fold is not counted. M3 has no data-dependent
addressing and each instruction writes one 32-bit word from at most three, so these bound the gather-free `q*` in
M3; that the ISA's instructions are such is SCALAR-ROUTE's business, not these statements'.
-/

namespace Pouw.Guarantees.Eps1

open Pouw.Eps1

/-- **Theorem B, zero instance.** A word-local M3 program that computes Dither's prefix words `T_ℓ[i,j]` at the zero
instance has at least `4·L·m·n` instructions. -/
def WordLocalZero : Prop :=
  ∀ (m n Lseg : ℕ) (FR FU : Type) (G : ℕ) (P : Prog (Dither m Lseg) (Cols n) m n Lseg FR FU G)
    (blk : Fin G → Tau m n Lseg),
    P.Computes (fun _ U => ColsValid U) zeroOut → P.WordLocal blk → 4 * (Lseg * m * n) ≤ G

/-- **Theorem B, general instance.** A word-local M3 program that computes the prefix words `T_ℓ[i,j]` of `Ã·B̃` on
every int8 instance has at least `8·L·m·n` instructions. -/
def WordLocalGeneral : Prop :=
  ∀ (m n Lseg : ℕ) (FR FU : Type) (G : ℕ) (P : Prog (Rows m Lseg) (GCols n Lseg) m n Lseg FR FU G)
    (blk : Fin G → Tau m n Lseg),
    P.Computes (fun A B => (∀ i l p, Int8Valid (A i l p)) ∧ ∀ j l p, Int8Valid (B j l p)) genOut →
    P.WordLocal blk → 8 * (Lseg * m * n) ≤ G

/-- **Theorem C.** A class-P (radix-packed) M3 program that computes Dither's prefix words at the zero instance has at
least `(1 + 28/27)·L·m·n` instructions. `G` counts M3 instructions, each writing one 32-bit word, so it counts
destination words: read in sm_120 issue slots, an IMAD.WIDE (two destination words in one slot) counts two. -/
def ClassPPacked : Prop :=
  ∀ (m n Lseg : ℕ) (FR FU : Type) (G : ℕ) (P : Prog (Dither m Lseg) (Cols n) m n Lseg FR FU G)
    (pk : Fin G → Option (Packing m n Lseg)),
    P.Computes (fun _ U => ColsValid U) zeroOut → ClassP P pk → (1 + 28 / 27 : ℝ) * (Lseg * m * n) ≤ G

end Pouw.Guarantees.Eps1
