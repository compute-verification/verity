module

public import Definitions.Pouw.Assumptions.Dimension.EmbedAssumptions
public import Definitions.Pouw.Assumptions.Dimension.OpenPipe

/-! `M4090`, straight-line programs over 32-bit words at the RTX 4090's W1 prices, and its embedding into the open quadratic pipe (draft semantics, pending review). -/

@[expose] public section

namespace Pouw.Dimension.Embed

open Finset Pouw.NCP Pouw.NCP.Relation Pouw.Dimension Pouw.Dimension.Assumptions Pouw.Dimension.OpenPipe

/-- An `M4090` instruction over the grid `G` (draft semantics, pending review). -/
inductive M4090Instr (G : Type) where
  /-- `price` is the whole instruction's, not per register. -/
  | mma (outs : List (G → ℚ)) (price : ℕ)
  /-- `κ` is FFMA, FADD or FMUL with its operand modifiers, of registers `i`, `j`, `l`. -/
  | fp32 (κ i j l : ℕ)
  /-- An INT32 or any other instruction writing one register. It includes FSEL, SEL, FMNMX, the half2 family, FSET, an
  atomic or reduction per lane operation, and IMAD.WIDE per 32-bit lane result. -/
  | int (f : G → ℚ)
  /-- A data-dependent load, store or guarded instruction, one word (H5). -/
  | dd (f : G → ℚ)
  /-- A sub-word extraction, insertion or packing, one word. -/
  | subword (f : G → ℚ)
  /-- An oblivious whole-word move of register `i`. -/
  | move (i : ℕ)
  /-- A pre-salt or supplied word, at its first fetch. -/
  | presalt (f : G → ℚ)
  /-- An operand word of a correct unit. -/
  | operand (f : G → ℚ)

variable {G : Type}

/-- An FP32 instruction's value on the registers `Rs`, lane by lane; a missing register reads `0`. -/
def fpVal (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (κ i j l : ℕ) (Rs : List (G → ℚ)) : G → ℚ :=
  fun g => fpSem κ (Rs.getD i 0 g) (Rs.getD j 0 g) (Rs.getD l 0 g)

def M4090Instr.outs (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) : M4090Instr G → List (G → ℚ) → List (G → ℚ)
  | .mma outs _, _ => outs
  | .fp32 κ i j l, Rs => [fpVal fpSem κ i j l Rs]
  | .int f, _ => [f]
  | .dd f, _ => [f]
  | .subword f, _ => [f]
  | .move i, Rs => [Rs.getD i 0]
  | .presalt f, _ => [f]
  | .operand f, _ => [f]

def M4090Instr.price : M4090Instr G → ℕ
  | .mma _ p => p
  | .fp32 .. => 8
  | .int _ => 16
  | .dd _ => 16
  | .subword _ => 16
  | .move _ => 0
  | .presalt _ => 1380
  | .operand _ => 0

def M4090Instr.WF {N m n k : ℕ} : M4090Instr (Noise N m n k) → Prop
  | .mma outs p => 16 * outs.length ≤ p
  | .presalt f => f ∈ V0 N m n k
  | .operand f => f ∈ V0 N m n k
  | _ => True

def regs (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (P : List (M4090Instr G)) (L : List (G → ℚ)) : List (G → ℚ) :=
  P.foldl (fun Rs o => Rs ++ o.outs fpSem Rs) L

def cost4090 (P : List (M4090Instr G)) : ℕ := (P.map M4090Instr.price).sum

open Classical in
noncomputable def M4090Instr.nq {N m n k : ℕ} (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) :
    M4090Instr (Noise N m n k) → List (Noise N m n k → ℚ) → ℕ
  | .fp32 κ i j l, Rs => if QuadStep Rs (fpVal fpSem κ i j l Rs) then 0 else 1
  | _, _ => 0

noncomputable def nonQuad {N m n k : ℕ} (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) :
    List (M4090Instr (Noise N m n k)) → List (Noise N m n k → ℚ) → ℕ
  | [], _ => 0
  | o :: P, Rs => o.nq fpSem Rs + nonQuad fpSem P (Rs ++ o.outs fpSem Rs)

open Classical in
noncomputable def canonNaN (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (κ : ℕ) (a b c : ℚ) : ℚ :=
  if NaNGap (fpSem κ a b c) then 2147483647 else fpSem κ a b c

/-! ## Guarantees (draft semantics, pending review) -/

/-- **The embedding theorem** (draft semantics, pending review). -/
def EmbedM4090 : Prop :=
  ∀ (N m n k : ℕ) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (Noise N m n k → ℚ)) (P : List (M4090Instr (Noise N m n k))),
    (∀ f ∈ L, f ∈ V0 N m n k) → (∀ o ∈ P, o.WF) →
    ∃ P' : List (Instr (Noise N m n k)), OpenQuadPipe 16 8 P' ∧
      (∀ r ∈ regs fpSem P L, r ∈ V0 N m n k ∨ r ∈ slots P' L) ∧
      cost P' ≤ cost4090 P + 8 * nonQuad fpSem P L

/-- **The exact embedding into the FP32 pipe** (draft semantics, pending review). -/
def EmbedFp : Prop :=
  ∀ (N m n k : ℕ) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (Noise N m n k → ℚ)) (P : List (M4090Instr (Noise N m n k))),
    (∀ f ∈ L, f ∈ V0 N m n k) → (∀ o ∈ P, o.WF) →
    ∃ P' : List (Instr (Noise N m n k)), FpPipe fpSem P' ∧
      (∀ r ∈ regs fpSem P L, r ∈ V0 N m n k ∨ r ∈ slots P' L) ∧ cost P' ≤ cost4090 P

/-- **The per-word cost in `M4090`, less the non-quadratic FP32 instructions**, with no A1 or A2 (draft semantics,
pending review). -/
def PerWordM4090 : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (Noise N m n k → ℚ)) (P : List (M4090Instr (Noise N m n k))),
    CheckpointIndependent k 16 R → MinRankGe k R 5 → U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) →
    (∀ o ∈ P, o.WF) → ∀ S ⊆ U ×ˢ range (nonFinal k m n), (∀ x ∈ S, word k m n N R A B x.1 x.2 ∈ regs fpSem P L) →
      16 * S.card ≤ cost4090 P + 8 * nonQuad fpSem P L

/-- A satisfiability witness of `A1_fp`: it holds at `γ = 1/2` for every FP32 semantics. -/
def A1fpHalf : Prop := ∀ fpSem : ℕ → ℚ → ℚ → ℚ → ℚ, A1fp fpSem (1 / 2)

/-- A satisfiability witness of `A1_fp`: it holds at `γ = 0` for the FADD-only semantics. -/
def A1fpFadd : Prop := A1fp faddSem 0

/-- **`CanonicalNaN` is satisfiable**: every FP32 semantics with its NaN-pattern outputs canonicalized satisfies it. -/
def CanonicalNaNSat : Prop := ∀ fpSem : ℕ → ℚ → ℚ → ℚ → ℚ, CanonicalNaN (canonNaN fpSem)

/-- **`A1_fp` at `γ = 0` from a range gap** (draft semantics, pending review). -/
def A1fpOnRangeGap : Prop :=
  ∀ (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat)
    (U : Finset ℕ), CheckpointIndependent k 16 R → U ⊆ range N →
    (∀ x ∈ U ×ˢ range (nonFinal k m n), ∃ ω : Noise N m n k,
      ∀ κ a b c, fpSem κ a b c ≠ word k m n N R A B x.1 x.2 ω) →
    A1fpOn fpSem 0 k m n N R A B U

/-- **The per-word cost of every `M4090` program from a range gap**, with no A1, A2 or MinRank (draft semantics,
pending review). -/
def PerWordM4090Gap : Prop :=
  ∀ (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat)
    (U : Finset ℕ) (L : List (Noise N m n k → ℚ)) (P : List (M4090Instr (Noise N m n k))),
    CheckpointIndependent k 16 R → U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) → (∀ o ∈ P, o.WF) →
    (∀ x ∈ U ×ˢ range (nonFinal k m n), ∃ ω : Noise N m n k,
      ∀ κ a b c, fpSem κ a b c ≠ word k m n N R A B x.1 x.2 ω) →
    ∀ S ⊆ U ×ˢ range (nonFinal k m n), (∀ x ∈ S, word k m n N R A B x.1 x.2 ∈ regs fpSem P L) →
      16 * S.card ≤ cost4090 P

end Pouw.Dimension.Embed
