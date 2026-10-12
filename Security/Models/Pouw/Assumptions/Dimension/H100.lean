import Models.Pouw.Assumptions.Dimension.H100Assumptions
import Models.Pouw.Assumptions.Dimension.Embed

/-!
# `MH100`, the H100 analogue of `M4090`, and its per-word bound (trusted statements; draft semantics)

**Draft semantics, pending review** (`internal/pouw-fp8/genuine-fp8-interface.md` §1.1).

`MH100 pr` is `M4090` (`Pouw.Dimension.Embed`) with the H100's classes, at the prices `pr : Prices100`
(`Pouw.Dimension.H100Assumptions`). Its registers are SSA and hold functions on a finite noise grid `G`. Every
register written is an output. The free values are a subspace `V` of the functions on `G` that contains the constants.
NCP's instance is `G = Noise N m n k` with `V = V₀`; an FP8 scheme supplies its own grid and free values. The
instructions:

* `wgmma s outs`: a `wgmma` of shape `s`, writing the accumulator registers `outs` (any functions), at
  `regPrice pr s` per register.
* `mma outs p`: every other tensor-core instruction, at its full price `p`, well-formed when `pr.tc·len(outs) ≤ p`.
* `fp32 κ i j l`: FFMA, FADD or FMUL, `fpSem κ` of registers `i`, `j`, `l`, at `pr.ffma`.
* `int f`, `fp64 f`, `other f`: one register, any function, at `pr.int32`, `pr.fp64`, `pr.other`. A word written in
  memory by a reduction or an atomic is an `other` register.
* `move i`: an oblivious whole-word move, free. That includes a plain load or store of a whole 32-bit word.
* `operand f`: a pre-salt, supplied or operand word, free (which under-prices a fetch), `f ∈ V`.

The model may under-price W1 (every priced instruction computes any function; program text is free) but must not
over-price it. Unlike `M4090` it has **no cheap FP32 class**: under `H32` the FP32 pipe costs 32 per register, as much
as any other register writer.

* **The embedding** (`EmbedMH100`): under `H32`, every well-formed program on inputs in `V` maps into AlgM_32
  (`Hc 32`: Theorem D's model with every priced register at 32 or more) at no greater cost.
* **The per-word bound** (`PerWordMH100`, Theorem D at `c = 32`): under `H32`, every well-formed program on inputs in
  `V` whose registers include a family of target words independent modulo `V` costs at least 32 per target word.
  No A1 and no A2. `PerWordMH100Words` instantiates it on NCP's non-final checked words under checkpoint
  independence. `PerWordMH100Floor` is the same bound at any floor: under `Floor c pr`, `c` per target word, so a
  measured price table gives its own least price per register.
* **Which words (red team F18-3).** `PerWordMH100Words` and `MH100WordsConsistent` cover NCP's exact words at 16-step
  checkpoints (`CheckpointIndependent k 16`, linear independence over ℚ modulo `V₀`). They do not cover the truncated
  k32 `HOPPER_E4M3_K32` words of C-2, A-1 or D-2, which are not linear. For those the applicable pin is `MH100w`'s
  `DistinctLiveMH100` (`Pouw.Dimension.H100W`), whose hypothesis `DistinctLiveAt ε` each scheme must certify.
* **Tightness and consistency.** `MH100Attains`: at the nominal prices every family is output at exactly 32 per word
  (one FP8 `wgmma` register each). `MH100WordsConsistent`: independent families exist (NCP's checked words).
  `MH100Attains` and `HonestSplitK100` are nominal-model statements. On hardware they hold only at `N ≥ 64`: Round 7
  measured m64n8k32 at 144 per register (red team F18-4; `Shape100`).
* **FP8 targets.** For the split-K FP8 target of `docs/pouw-fp8/design.md` §6.1 (fresh-step words
  `I_τ = HOPPER_E4M3_K32.step(+0, Ã[S_τ], B̃[S_τ])` and running words `R_τ = RN_f32(R_(τ−1) + I_τ)`), the honest
  schedule `splitK` costs 32 per word at the nominal prices: a k32 `wgmma` register per `I_τ` and an FADD per `R_τ`
  (`HonestSplitK100`). If the scheme makes those words independent modulo its free values, `PerWordMH100` gives 32
  per word, so the bound is tight. That independence is the scheme lane's statement, not proved here. The truncated
  atom words are not linear, so for them the W1-faithful route is `MH100w` (`Pouw.Dimension.H100W`).
-/

namespace Pouw.Dimension.H100

open Finset Pouw.NCP Pouw.NCP.Relation Pouw.Dimension Pouw.Dimension.Assumptions Pouw.Dimension.Embed

/-- An `MH100` instruction over the grid `G` (draft semantics, pending review). -/
inductive Op100 (G : Type) where
  /-- A `wgmma` of shape `s`, writing the accumulator registers `outs`. -/
  | wgmma (s : Shape100) (outs : List (G → ℚ))
  /-- Any other tensor-core instruction at its full price, writing the accumulator registers `outs`. -/
  | mma (outs : List (G → ℚ)) (price : ℕ)
  /-- FFMA, FADD or FMUL `κ` of registers `i`, `j`, `l`. -/
  | fp32 (κ i j l : ℕ)
  /-- An INT32 instruction writing one register: any function. -/
  | int (f : G → ℚ)
  /-- An FP64 instruction's 32-bit result register: any function. -/
  | fp64 (f : G → ℚ)
  /-- Any other instruction writing one 32-bit word, in a register or in memory (data-dependent accesses, sub-word
  operations, memory-side reductions and atomics included): any function. -/
  | other (f : G → ℚ)
  /-- An oblivious whole-word move of register `i`, free: a register copy, or a plain load or store of a whole 32-bit
  word. -/
  | move (i : ℕ)
  /-- A pre-salt, supplied or operand word, free. -/
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

/-- Well-formed, for the free values `V`: a tensor-core instruction outside `Shape100` costs at least `pr.tc` per
accumulator register it writes, and operand words are free values. -/
def Op100.WF (pr : Prices100) (V : Submodule ℚ (G → ℚ)) : Op100 G → Prop
  | .mma outs p => pr.tc * outs.length ≤ p
  | .operand f => f ∈ V
  | _ => True

/-- A program's registers: its inputs `L`, then the registers each instruction writes, in order. -/
def regs100 (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (P : List (Op100 G)) (L : List (G → ℚ)) : List (G → ℚ) :=
  P.foldl (fun Rs o => Rs ++ o.outs fpSem Rs) L

/-- A program's cost at the prices `pr`: the sum of its prices. -/
def cost100 (pr : Prices100) (P : List (Op100 G)) : ℕ := (P.map (Op100.price pr)).sum

/-- **AlgM_c**: every priced instruction of the priced model, one register written, costs at least `c` (`H16` is
`Hc 16`). -/
def Hc (c : ℕ) (P : List (Instr G)) : Prop := ∀ ins ∈ P, ins.isPriced = true → c ≤ ins.price

/-- A family of target words, linearly independent modulo the free values `V`. -/
def IndepMod (V : Submodule ℚ (G → ℚ)) {ι : Type} [Fintype ι] (t : ι → G → ℚ) : Prop :=
  ∀ c : ι → ℚ, ∑ i, c i • t i ∈ V → ∀ i, c i = 0

/-- The split-K running words (`design.md` §6.1) over the fresh-step words `I`: `R 0 = +0` and
`R (τ + 1) = fpSem κ (R τ) (I τ) (R τ)`, which is `RN_f32(R_τ + I_τ)` when `κ` is FADD (the third operand unused). -/
def runWords (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (κ : ℕ) (I : ℕ → G → ℚ) : ℕ → G → ℚ
  | 0 => 0
  | τ + 1 => fun g => fpSem κ (runWords fpSem κ I τ g) (I τ g) (runWords fpSem κ I τ g)

/-- The honest split-K schedule on `b` inputs: `R 0` as an operand word, then for each step one FP8 `wgmma` register
`I τ` and one FP32 instruction `κ` on the registers of `R τ` and `I τ`. -/
def splitK (κ : ℕ) (I : ℕ → G → ℚ) (b : ℕ) : ℕ → List (Op100 G)
  | 0 => [Op100.operand 0]
  | τ + 1 => splitK κ I b τ ++
      [Op100.wgmma Shape100.e4m3F32 [I τ], Op100.fp32 κ (b + 2 * τ) (b + 2 * τ + 1) (b + 2 * τ)]

/-! ## Pinned statements (draft semantics, pending review) -/

/-- **`H32` is satisfiable**: the nominal H100 table satisfies it. -/
def H32Sat : Prop := H32 h100Nominal

/-- **The embedding theorem for `MH100`** (draft semantics, pending review): under `H32`, for every grid, free values
`V` and FP32 semantics, every well-formed `MH100` program on inputs in `V` maps to an AlgM_32 program whose slots
contain every register outside `V`, at no greater cost. -/
def EmbedMH100 : Prop :=
  ∀ pr : Prices100, H32 pr →
    ∀ (G : Type) (V : Submodule ℚ (G → ℚ)) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (G → ℚ)) (P : List (Op100 G)),
      (∀ f ∈ L, f ∈ V) → (∀ o ∈ P, o.WF pr V) →
      ∃ P' : List (Instr G), Hc 32 P' ∧ (∀ r ∈ regs100 fpSem P L, r ∈ V ∨ r ∈ slots P' L) ∧ cost P' ≤ cost100 pr P

/-- **The per-word cost in `MH100`** (Theorem D at `c = 32`; no A1, no A2; draft semantics, pending review): under
`H32`, for every finite grid, free values `V` containing the constants, and FP32 semantics, a well-formed program on
inputs in `V` whose registers include a family of target words `t`, linearly independent modulo `V`, costs at least
`32·|ι|`. -/
def PerWordMH100 : Prop :=
  ∀ pr : Prices100, H32 pr →
    ∀ (G : Type) [Fintype G] (V : Submodule ℚ (G → ℚ)), (1 : G → ℚ) ∈ V →
    ∀ (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (ι : Type) [Fintype ι] (t : ι → G → ℚ) (L : List (G → ℚ)) (P : List (Op100 G)),
      IndepMod V t → (∀ f ∈ L, f ∈ V) → (∀ o ∈ P, o.WF pr V) → (∀ i, t i ∈ regs100 fpSem P L) →
      32 * Fintype.card ι ≤ cost100 pr P

/-- **The per-word cost in `MH100` at any floor** (Theorem D at `c`; no A1, no A2; draft semantics, pending review):
under `Floor c pr`, for every finite grid, free values `V` containing the constants, and FP32 semantics, a well-formed
program on inputs in `V` whose registers include a family of target words `t`, linearly independent modulo `V`, costs
at least `c·|ι|`. -/
def PerWordMH100Floor : Prop :=
  ∀ (c : ℕ) (pr : Prices100), Floor c pr →
    ∀ (G : Type) [Fintype G] (V : Submodule ℚ (G → ℚ)), (1 : G → ℚ) ∈ V →
    ∀ (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (ι : Type) [Fintype ι] (t : ι → G → ℚ) (L : List (G → ℚ)) (P : List (Op100 G)),
      IndepMod V t → (∀ f ∈ L, f ∈ V) → (∀ o ∈ P, o.WF pr V) → (∀ i, t i ∈ regs100 fpSem P L) →
      c * Fintype.card ι ≤ cost100 pr P

/-- **The per-word cost in `MH100` on NCP's checked words** (no A1, no A2, no MinRank; draft semantics, pending
review): under `H32` and checkpoint independence, a well-formed program on inputs in `V₀` whose registers include the
non-final checked words at `S ⊆ U × [0, nonFinal)` costs at least `32·|S|`.

It covers NCP's exact words at 16-step checkpoints only. The truncated k32 atom words of C-2, A-1 and D-2 are not
linear, so they need `DistinctLiveMH100` instead (red team F18-3). -/
def PerWordMH100Words : Prop :=
  ∀ pr : Prices100, H32 pr →
    ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
      (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (Noise N m n k → ℚ)) (P : List (Op100 (Noise N m n k))),
      CheckpointIndependent k 16 R → U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) → (∀ o ∈ P, o.WF pr (V0 N m n k)) →
      ∀ S ⊆ U ×ˢ range (nonFinal k m n), (∀ x ∈ S, word k m n N R A B x.1 x.2 ∈ regs100 fpSem P L) →
        32 * S.card ≤ cost100 pr P

/-- **The bound is attained** at the nominal prices: for every grid, free values, FP32 semantics, inputs and family
of target words, one FP8 `wgmma` register per word is a well-formed program that outputs them all at exactly
`32·|ι|`. This is a nominal-model statement. On hardware it holds only for `wgmma` at `N ≥ 64` (red team F18-4;
`Shape100`). -/
def MH100Attains : Prop :=
  ∀ (G : Type) (V : Submodule ℚ (G → ℚ)) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (ι : Type) [Fintype ι] (t : ι → G → ℚ)
    (L : List (G → ℚ)),
    ∃ P : List (Op100 G), (∀ o ∈ P, o.WF h100Nominal V) ∧ (∀ i, t i ∈ regs100 fpSem P L) ∧
      cost100 h100Nominal P = 32 * Fintype.card ι

/-- **The honest split-K schedule costs 32 per checked word** at the nominal prices: for every grid, free values, FP32
semantics, opcode `κ`, fresh-step words `I`, `T` and inputs, the program `splitK` is well-formed, its registers include
`I τ` and `R (τ + 1)` for every `τ < T`, and it costs `32·2T`: one FP8 `wgmma` register (32) per `I τ` and one FP32
instruction (32) per `R (τ + 1)`.

This is a nominal-model statement (red team F18-4). On hardware the `wgmma` price holds only at `N ≥ 64`, since Round 7
measured m64n8k32, C-2's atom shape, at 144 per register. The FADD per `R (τ + 1)` measured 33.5, not 32. -/
def HonestSplitK100 : Prop :=
  ∀ (G : Type) (V : Submodule ℚ (G → ℚ)) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (κ : ℕ) (I : ℕ → G → ℚ) (T : ℕ)
    (L : List (G → ℚ)),
    (∀ o ∈ splitK κ I L.length T, o.WF h100Nominal V) ∧
      (∀ τ < T, I τ ∈ regs100 fpSem (splitK κ I L.length T) L ∧
        runWords fpSem κ I (τ + 1) ∈ regs100 fpSem (splitK κ I L.length T) L) ∧
      cost100 h100Nominal (splitK κ I L.length T) = 32 * (2 * T)

/-- **`PerWordMH100Words`'s hypotheses hold**: for every `k ≥ 1024` with `16 ∣ k`, every shape, `A = g(F₁)`, `B` and
unit set, some `R` is checkpoint independent, and NCP's non-final checked words are then independent modulo `V₀`. -/
def MH100WordsConsistent : Prop :=
  ∀ k, 1024 ≤ k → 16 ∣ k → ∀ (m n N : ℕ) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ),
    U ⊆ range N →
    ∃ R : Mat, CheckpointIndependent k 16 R ∧
      IndepMod (V0 N m n k) (fun x : ↥(U ×ˢ range (nonFinal k m n)) => word k m n N R A B x.1.1 x.1.2)

end Pouw.Dimension.H100
