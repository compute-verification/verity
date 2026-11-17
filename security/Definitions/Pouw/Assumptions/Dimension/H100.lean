module

public import Definitions.Pouw.Assumptions.Dimension.H100Assumptions
public import Definitions.Pouw.Assumptions.Dimension.Embed

/-! `MH100`, the H100 analogue of `M4090`, with its embedding into AlgM_32 (`EmbedMH100`: draft semantics, pending review) and its per-word bounds. -/

@[expose] public section

namespace Pouw.Dimension.H100

open Finset Pouw.NCP Pouw.NCP.Relation Pouw.Dimension Pouw.Dimension.Assumptions Pouw.Dimension.Embed

/-- An `MH100` instruction over the grid `G`. -/
inductive Op100 (G : Type) where
  | wgmma (s : Shape100) (outs : List (G → ℚ))
  /-- Any other tensor-core instruction; `price` is the whole instruction's. -/
  | mma (outs : List (G → ℚ)) (price : ℕ)
  /-- FFMA, FADD or FMUL `κ` of registers `i`, `j`, `l`. -/
  | fp32 (κ i j l : ℕ)
  | int (f : G → ℚ)
  /-- An FP64 instruction's 32-bit result register: any function. -/
  | fp64 (f : G → ℚ)
  /-- Any other instruction writing one 32-bit word, in a register or in memory (data-dependent accesses, sub-word
  operations, memory-side reductions and atomics included). -/
  | other (f : G → ℚ)
  /-- An oblivious whole-word move of register `i`: a register copy, or a plain load or store of a whole 32-bit word. -/
  | move (i : ℕ)
  /-- A pre-salt, supplied or operand word. -/
  | operand (f : G → ℚ)

variable {G : Type}

/-- The registers an instruction writes, given the registers `Rs` before it. -/
def Op100.outs (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) : Op100 G → List (G → ℚ) → List (G → ℚ)
  | .wgmma _ outs, _ => outs
  | .mma outs _, _ => outs
  | .fp32 κ i j l, Rs => [fpVal fpSem κ i j l Rs]
  | .int f, _ => [f]
  | .fp64 f, _ => [f]
  | .other f, _ => [f]
  | .move i, Rs => [Rs.getD i 0]
  | .operand f, _ => [f]

/-- An instruction's price at the prices `pr`. -/
def Op100.price (pr : Prices100) : Op100 G → ℕ
  | .wgmma s outs => s.regPrice pr * outs.length
  | .mma _ p => p
  | .fp32 .. => pr.ffma
  | .int _ => pr.int32
  | .fp64 _ => pr.fp64
  | .other _ => pr.other
  | .move _ => 0
  | .operand _ => 0

def Op100.WF (pr : Prices100) (V : Submodule ℚ (G → ℚ)) : Op100 G → Prop
  | .mma outs p => pr.tc * outs.length ≤ p
  | .operand f => f ∈ V
  | _ => True

/-- A program's registers: its inputs `L`, then the registers each instruction writes, in order. -/
def regs100 (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (P : List (Op100 G)) (L : List (G → ℚ)) : List (G → ℚ) :=
  P.foldl (fun Rs o => Rs ++ o.outs fpSem Rs) L

/-- A program's cost at the prices `pr`: the sum of its prices. -/
def cost100 (pr : Prices100) (P : List (Op100 G)) : ℕ := (P.map (Op100.price pr)).sum

/-- **AlgM_c**; `H16` is `Hc 16`. -/
def Hc (c : ℕ) (P : List (Instr G)) : Prop := ∀ ins ∈ P, ins.isPriced = true → c ≤ ins.price

def IndepMod (V : Submodule ℚ (G → ℚ)) {ι : Type} [Fintype ι] (t : ι → G → ℚ) : Prop :=
  ∀ c : ι → ℚ, ∑ i, c i • t i ∈ V → ∀ i, c i = 0

/-- `RN_f32(R_τ + I_τ)` when `κ` is FADD, the third operand unused. -/
def runWords (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (κ : ℕ) (I : ℕ → G → ℚ) : ℕ → G → ℚ
  | 0 => 0
  | τ + 1 => fun g => fpSem κ (runWords fpSem κ I τ g) (I τ g) (runWords fpSem κ I τ g)

/-- The honest split-K schedule on `b` inputs. -/
def splitK (κ : ℕ) (I : ℕ → G → ℚ) (b : ℕ) : ℕ → List (Op100 G)
  | 0 => [Op100.operand 0]
  | τ + 1 => splitK κ I b τ ++
      [Op100.wgmma Shape100.e4m3F32 [I τ], Op100.fp32 κ (b + 2 * τ) (b + 2 * τ + 1) (b + 2 * τ)]

/-- **`H32` is satisfiable**: the nominal H100 table satisfies it. -/
def H32Sat : Prop := H32 h100Nominal

/-- **The embedding theorem for `MH100`** (draft semantics, pending review). -/
def EmbedMH100 : Prop :=
  ∀ pr : Prices100, H32 pr →
    ∀ (G : Type) (V : Submodule ℚ (G → ℚ)) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (G → ℚ)) (P : List (Op100 G)),
      (∀ f ∈ L, f ∈ V) → (∀ o ∈ P, o.WF pr V) →
      ∃ P' : List (Instr G), Hc 32 P' ∧ (∀ r ∈ regs100 fpSem P L, r ∈ V ∨ r ∈ slots P' L) ∧ cost P' ≤ cost100 pr P

/-- **The per-word cost in `MH100`** (Theorem D at `c = 32`), with no A1 or A2. -/
def PerWordMH100 : Prop :=
  ∀ pr : Prices100, H32 pr →
    ∀ (G : Type) [Fintype G] (V : Submodule ℚ (G → ℚ)), (1 : G → ℚ) ∈ V →
    ∀ (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (ι : Type) [Fintype ι] (t : ι → G → ℚ) (L : List (G → ℚ)) (P : List (Op100 G)),
      IndepMod V t → (∀ f ∈ L, f ∈ V) → (∀ o ∈ P, o.WF pr V) → (∀ i, t i ∈ regs100 fpSem P L) →
      32 * Fintype.card ι ≤ cost100 pr P

/-- **The per-word cost in `MH100` at any floor** (Theorem D at `c`), with no A1 or A2. -/
def PerWordMH100Floor : Prop :=
  ∀ (c : ℕ) (pr : Prices100), Floor c pr →
    ∀ (G : Type) [Fintype G] (V : Submodule ℚ (G → ℚ)), (1 : G → ℚ) ∈ V →
    ∀ (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (ι : Type) [Fintype ι] (t : ι → G → ℚ) (L : List (G → ℚ)) (P : List (Op100 G)),
      IndepMod V t → (∀ f ∈ L, f ∈ V) → (∀ o ∈ P, o.WF pr V) → (∀ i, t i ∈ regs100 fpSem P L) →
      c * Fintype.card ι ≤ cost100 pr P

/-- **The per-word cost in `MH100` on NCP's checked words**, with no A1, A2 or MinRank. It covers NCP's exact words at
16-step checkpoints only: the truncated k32 atom words are not linear and need `DistinctLiveMH100`. -/
def PerWordMH100Words : Prop :=
  ∀ pr : Prices100, H32 pr →
    ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
      (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (Noise N m n k → ℚ)) (P : List (Op100 (Noise N m n k))),
      CheckpointIndependent k 16 R → U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) → (∀ o ∈ P, o.WF pr (V0 N m n k)) →
      ∀ S ⊆ U ×ˢ range (nonFinal k m n), (∀ x ∈ S, word k m n N R A B x.1 x.2 ∈ regs100 fpSem P L) →
        32 * S.card ≤ cost100 pr P

/-- A nominal-model statement: on hardware it holds only for `wgmma` at `N ≥ 64` (`Shape100`). -/
def MH100Attains : Prop :=
  ∀ (G : Type) (V : Submodule ℚ (G → ℚ)) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (ι : Type) [Fintype ι] (t : ι → G → ℚ)
    (L : List (G → ℚ)),
    ∃ P : List (Op100 G), (∀ o ∈ P, o.WF h100Nominal V) ∧ (∀ i, t i ∈ regs100 fpSem P L) ∧
      cost100 h100Nominal P = 32 * Fintype.card ι

/-- A nominal-model statement: on hardware the `wgmma` price holds only at `N ≥ 64` (`Shape100`). -/
def HonestSplitK100 : Prop :=
  ∀ (G : Type) (V : Submodule ℚ (G → ℚ)) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (κ : ℕ) (I : ℕ → G → ℚ) (T : ℕ)
    (L : List (G → ℚ)),
    (∀ o ∈ splitK κ I L.length T, o.WF h100Nominal V) ∧
      (∀ τ < T, I τ ∈ regs100 fpSem (splitK κ I L.length T) L ∧
        runWords fpSem κ I (τ + 1) ∈ regs100 fpSem (splitK κ I L.length T) L) ∧
      cost100 h100Nominal (splitK κ I L.length T) = 32 * (2 * T)

/-- **`PerWordMH100Words`'s hypotheses hold.** -/
def MH100WordsConsistent : Prop :=
  ∀ k, 1024 ≤ k → 16 ∣ k → ∀ (m n N : ℕ) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ),
    U ⊆ range N →
    ∃ R : Mat, CheckpointIndependent k 16 R ∧
      IndepMod (V0 N m n k) (fun x : ↥(U ×ˢ range (nonFinal k m n)) => word k m n N R A B x.1.1 x.1.2)

end Pouw.Dimension.H100
