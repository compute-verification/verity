import Definitions.Pouw.Assumptions.Dimension.EmbedAssumptions

/-!
# M4090 and the embedding theorem (trusted statements; draft semantics)

**Draft semantics, pending review**.

`M4090` makes W1 precise for straight-line programs over 32-bit words, on the
noise grid. Registers are SSA and hold functions of the noise (the int32 reading of each 32-bit pattern); every
register ever written is an output, and the inputs are the noise coordinates. The instructions and their prices:

* `mma outs p`: a tensor-core instruction at its full shape's price `p`, writing the accumulator registers `outs`
  (any functions); well-formed when `16·len(outs) ≤ p` (H_16 for the MMA table; legacy m8n8k4 meets it at 16 per
  register).
* `fp32 κ i j l`: an FP32-pipe instruction at 8, `fpSem κ` of registers `i`, `j`, `l`, lane by lane. The FP32 class
  is exactly FFMA, FADD and FMUL with their operand modifiers, the only opcodes W1 prices at 8 by measured throughput.
* `int f`, `dd f`, `subword f`: an INT32 or other instruction, a data-dependent access or guard (H5), a sub-word
  extraction (by ruling), each one register at 16, computing any function of the noise. `int` includes FSEL, SEL,
  FMNMX and the half2 family (HFMA2, HADD2, HMUL2, HMNMX2), which the 4090 runs at 16.1–16.6 per register written, and
  FSET: FSET.BF runs on the ALU at 16.4 per register, 16.6 with a data-dependent predicate, and `set.*.u32.f32` lowers
  to FSETP + SEL (§11, Round 6). It also includes atomics and reductions per lane operation, and IMAD.WIDE per 32-bit
  lane result. FSETP, a predicate writer, is not an instruction here.
* `move i`: an oblivious whole-word move, free.
* `presalt f`: a pre-salt or supplied word at 4·345, `f ∈ V₀`.
* `operand f`: a correct unit's operand word, free, `f ∈ V₀`.

The model may under-price W1 (every priced instruction computes any function; `int` is 16 where W1 charges more;
program text is free) but must not over-price it.

* **The embedding** (`EmbedM4090`): every well-formed program maps into `OpenQuadPipe 16 8` at no greater cost,
  except that a non-quadratic FP32 instruction costs 16 there (`nonQuad` counts them). `EmbedFp` maps it into the
  FP32 pipe at no greater cost, with no exception.
* **Consequences.** `PerWordM4090`: `16·|S| ≤ cost + 8·nonQuad`, with no assumption beyond checkpoint independence and
  MinRank ≥ 5. `A1fpIff`: `A1_fp` is equivalent to the per-word cost of `M4090`. `PerWordOfA1fp`, `TTNCPM4090OfA1fp`:
  `A1_fp` gives the per-word cost and TT_NCP. `M4090ChainConsistent`: the chain's hypotheses hold together at the
  tight `γ₀ = 16/(3k)`, with `A1fpFadd`.
* **The range gap**. `A1fpOnRangeGap`: if at some grid point each checked word takes a
  value that `fpSem` never outputs, `A1_fp` holds at `γ = 0` on that instance. `PerWordM4090Gap`: then every `M4090`
  program pays 16 per checked word, with no A1 and no MinRank. Under `CanonicalNaN`, `TTNCPM4090NanGap` gives TT_NCP
  for every `M4090` program on instances whose words take a value in the range gap `NaNGap` (`[−8388607, −2]` or
  `[2139095041, 2147483646]`), and `NanGapConsistent` shows its hypotheses hold together at the tight
  `γ₀ = 16/(3k)`.
-/

namespace Pouw.Dimension.Embed

open Finset Pouw.NCP Pouw.NCP.Relation Pouw.Dimension Pouw.Dimension.Assumptions Pouw.Dimension.OpenPipe

/-- An `M4090` instruction over the grid `G` (draft semantics, pending review). -/
inductive M4090Instr (G : Type) where
  /-- A tensor-core instruction at its full shape's price, writing the accumulator registers `outs`. -/
  | mma (outs : List (G → ℚ)) (price : ℕ)
  /-- An FP32-pipe instruction `κ` (FFMA, FADD or FMUL, with operand modifiers) of registers `i`, `j`, `l`, at 8. -/
  | fp32 (κ i j l : ℕ)
  /-- An INT32 or any other instruction writing one register, at 16: any function. It includes FSEL, SEL, FMNMX and
  the half2 family (Round 4: 16.1–16.6 per register), FSET (Round 6: 16.4, 16.6 with a data-dependent predicate), an
  atomic or reduction per lane operation, and IMAD.WIDE per 32-bit lane result. -/
  | int (f : G → ℚ)
  /-- A data-dependent load, store or guarded instruction, one word, at 16 (H5): any function. -/
  | dd (f : G → ℚ)
  /-- A sub-word extraction, insertion or packing, one word, at 16: any function. -/
  | subword (f : G → ℚ)
  /-- An oblivious whole-word move of register `i`, free. -/
  | move (i : ℕ)
  /-- A pre-salt or supplied word, first fetch, at 4·345. -/
  | presalt (f : G → ℚ)
  /-- An operand word of a correct unit, free. -/
  | operand (f : G → ℚ)

variable {G : Type}

/-- An FP32 instruction's value on the registers `Rs`, lane by lane; a missing register reads `0`. -/
def fpVal (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (κ i j l : ℕ) (Rs : List (G → ℚ)) : G → ℚ :=
  fun g => fpSem κ (Rs.getD i 0 g) (Rs.getD j 0 g) (Rs.getD l 0 g)

/-- The registers an instruction writes, given the registers `Rs` before it. -/
def M4090Instr.outs (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) : M4090Instr G → List (G → ℚ) → List (G → ℚ)
  | .mma outs _, _ => outs
  | .fp32 κ i j l, Rs => [fpVal fpSem κ i j l Rs]
  | .int f, _ => [f]
  | .dd f, _ => [f]
  | .subword f, _ => [f]
  | .move i, Rs => [Rs.getD i 0]
  | .presalt f, _ => [f]
  | .operand f, _ => [f]

/-- An instruction's W1 price. -/
def M4090Instr.price : M4090Instr G → ℕ
  | .mma _ p => p
  | .fp32 .. => 8
  | .int _ => 16
  | .dd _ => 16
  | .subword _ => 16
  | .move _ => 0
  | .presalt _ => 1380
  | .operand _ => 0

/-- Well-formed: a tensor-core instruction costs at least 16 per accumulator register it writes, and pre-salt and
operand words are free values. -/
def M4090Instr.WF {N m n k : ℕ} : M4090Instr (Noise N m n k) → Prop
  | .mma outs p => 16 * outs.length ≤ p
  | .presalt f => f ∈ V0 N m n k
  | .operand f => f ∈ V0 N m n k
  | _ => True

/-- A program's registers: its inputs `L`, then the registers each instruction writes, in order. -/
def regs (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (P : List (M4090Instr G)) (L : List (G → ℚ)) : List (G → ℚ) :=
  P.foldl (fun Rs o => Rs ++ o.outs fpSem Rs) L

/-- A program's W1 cost: the sum of its prices. -/
def cost4090 (P : List (M4090Instr G)) : ℕ := (P.map M4090Instr.price).sum

/-- **`M4090`** as a machine (draft semantics, pending review): its programs are the well-formed
straight-line programs on the noise coordinates, at their W1 cost, outputting their registers. -/
def m4090 (N m n k : ℕ) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) : Machine (Noise N m n k) where
  Prog := {P : List (M4090Instr (Noise N m n k)) // ∀ o ∈ P, o.WF}
  cost P := (cost4090 P.1 : ℝ)
  outputs P := {φ | φ ∈ regs fpSem P.1 (coords N m n k)}

open Classical in
/-- `1` for an FP32 instruction whose value is not a `QuadStep` over the registers before it, else `0`. -/
noncomputable def M4090Instr.nq {N m n k : ℕ} (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) :
    M4090Instr (Noise N m n k) → List (Noise N m n k → ℚ) → ℕ
  | .fp32 κ i j l, Rs => if QuadStep Rs (fpVal fpSem κ i j l Rs) then 0 else 1
  | _, _ => 0

/-- The number of **non-quadratic FP32 instructions** of a program on inputs `Rs`. -/
noncomputable def nonQuad {N m n k : ℕ} (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) :
    List (M4090Instr (Noise N m n k)) → List (Noise N m n k → ℚ) → ℕ
  | [], _ => 0
  | o :: P, Rs => o.nq fpSem Rs + nonQuad fpSem P (Rs ++ o.outs fpSem Rs)

open Classical in
/-- `fpSem` with every output in the range gap replaced by `0x7FFFFFFF` (`2147483647`). -/
noncomputable def canonNaN (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (κ : ℕ) (a b c : ℚ) : ℚ :=
  if NaNGap (fpSem κ a b c) then 2147483647 else fpSem κ a b c

/-! ## Pinned statements (draft semantics, pending review) -/

/-- **The embedding theorem** (draft semantics, pending review): for every FP32 semantics, every
well-formed `M4090` program on inputs in `V₀` maps to an `OpenQuadPipe 16 8` program whose slots contain every
register outside `V₀`, at no greater cost except 8 for each non-quadratic FP32 instruction. -/
def EmbedM4090 : Prop :=
  ∀ (N m n k : ℕ) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (Noise N m n k → ℚ)) (P : List (M4090Instr (Noise N m n k))),
    (∀ f ∈ L, f ∈ V0 N m n k) → (∀ o ∈ P, o.WF) →
    ∃ P' : List (Instr (Noise N m n k)), OpenQuadPipe 16 8 P' ∧
      (∀ r ∈ regs fpSem P L, r ∈ V0 N m n k ∨ r ∈ slots P' L) ∧
      cost P' ≤ cost4090 P + 8 * nonQuad fpSem P L

/-- **The exact embedding into the FP32 pipe** (draft semantics, pending review): every
well-formed `M4090 fpSem` program on inputs in `V₀` maps to an `FpPipe fpSem` program whose slots contain every
register outside `V₀`, at no greater cost. -/
def EmbedFp : Prop :=
  ∀ (N m n k : ℕ) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (Noise N m n k → ℚ)) (P : List (M4090Instr (Noise N m n k))),
    (∀ f ∈ L, f ∈ V0 N m n k) → (∀ o ∈ P, o.WF) →
    ∃ P' : List (Instr (Noise N m n k)), FpPipe fpSem P' ∧
      (∀ r ∈ regs fpSem P L, r ∈ V0 N m n k ∨ r ∈ slots P' L) ∧ cost P' ≤ cost4090 P

/-- **The per-word cost in `M4090`, less the non-quadratic FP32 instructions** (no A1, no A2; draft
semantics, pending review): under checkpoint independence and MinRank ≥ 5, a well-formed program
on inputs in `V₀` whose registers include the non-final checked words at `S ⊆ U × [0, nonFinal)` has
`16·|S| ≤ cost + 8·nonQuad`. -/
def PerWordM4090 : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (Noise N m n k → ℚ)) (P : List (M4090Instr (Noise N m n k))),
    CheckpointIndependent k 16 R → MinRankGe k R 5 → U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) →
    (∀ o ∈ P, o.WF) → ∀ S ⊆ U ×ˢ range (nonFinal k m n), (∀ x ∈ S, word k m n N R A B x.1 x.2 ∈ regs fpSem P L) →
      16 * S.card ≤ cost4090 P + 8 * nonQuad fpSem P L

/-- **`A1_fp` is the per-word cost of `M4090`** (draft semantics, pending review): under
checkpoint independence and MinRank ≥ 5, for every FP32 semantics and `γ`, `A1_fp`'s conclusion for an instance is
equivalent to `PerWordCost γ (M4090 fpSem)` for its non-final checked words. One way is `EmbedFp`; the other maps an
FP32-pipe program into `M4090` at no greater cost (a big instruction to `int`, an FP32 step to one `operand` or
`move` per operand and one `fp32`). -/
def A1fpIff : Prop :=
  ∀ (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (γ : ℝ) (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat)
    (B : ℕ → Mat) (U : Finset ℕ), CheckpointIndependent k 16 R → MinRankGe k R 5 → U ⊆ range N →
    (A1fpOn fpSem γ k m n N R A B U ↔ PerWordCost γ (m4090 N m n k fpSem) (word k m n N R A B) U)

/-- **`A1_fp` gives the per-word cost of `M4090`** (draft semantics, pending review):
`A1_fp(fpSem, γ)` implies `PerWordCost γ (M4090 fpSem)` for the non-final checked words, under checkpoint
independence and MinRank ≥ 5. -/
def PerWordOfA1fp : Prop :=
  ∀ (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (γ : ℝ), A1fp fpSem γ →
    ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ),
      CheckpointIndependent k 16 R → MinRankGe k R 5 → U ⊆ range N →
      PerWordCost γ (m4090 N m n k fpSem) (word k m n N R A B) U

/-- **TT_NCP for `M4090` from `A1_fp`** (draft semantics, pending review): if `A1_fp(fpSem, γ₂)`
holds, `R` is checkpoint independent with MinRank ≥ 5, an `M4090` program outputs the non-final checked words, and
`1 − γ₀ ≤ (1 − γ₂)·(1 − 16/(3k))`, then TT_NCP's event fails for that program. -/
def TTNCPM4090OfA1fp : Prop :=
  ∀ (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat)
    (U : Finset ℕ) (q : (m4090 N m n k fpSem).Prog) (γ₀ γ₂ : ℝ), A1fp fpSem γ₂ → 16 ∣ k → 0 < k →
    CheckpointIndependent k 16 R → MinRankGe k R 5 → U ⊆ range N →
    (∀ u ∈ U, ∀ p < nonFinal k m n, word k m n N R A B u p ∈ (m4090 N m n k fpSem).outputs q) →
    γ₂ < 1 → γ₀ < 1 → 1 - γ₀ ≤ (1 - γ₂) * (1 - 16 / (3 * k : ℝ)) →
    ¬ ((m4090 N m n k fpSem).cost q / (1 - γ₀) < ∑ _u ∈ U, 3 * ((m * k * n : ℕ) : ℝ))

/-- **`A1_fp` at `γ = 1/2` holds for every FP32 semantics** (a satisfiability witness): each checked word is a
distinct priced slot, and every priced instruction of the FP32 pipe costs at least 8. -/
def A1fpHalf : Prop := ∀ fpSem : ℕ → ℚ → ℚ → ℚ → ℚ, A1fp fpSem (1 / 2)

/-- **`A1_fp` at `γ = 0` holds for the FADD-only semantics** (a satisfiability witness): an FADD step `x + y` is a
`QuadStep`, so the FP32 pipe is an open quadratic pipe, which pays 16 per checked word (`theorem1Open`). -/
def A1fpFadd : Prop := A1fp faddSem 0

/-- **`TTNCPM4090OfA1fp`'s hypotheses hold together** (draft semantics, pending review), at the
tight `γ₂ = 0` and `γ₀ = 16/(3k)`: for every `k ≥ 1024` with `16 ∣ k`, every shape, `A = g(F₁)`, `B` and unit set, one
`R` and one `M4090 faddSem` program satisfy all of them, with `A1_fp faddSem 0`. -/
def M4090ChainConsistent : Prop :=
  ∀ k, 1024 ≤ k → 16 ∣ k → ∀ (m n N : ℕ) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ),
    U ⊆ range N →
    ∃ R : Mat, CheckpointIndependent k 16 R ∧ MinRankGe k R 5 ∧ A1fp faddSem 0 ∧
      (∃ q : (m4090 N m n k faddSem).Prog,
        ∀ u ∈ U, ∀ p < nonFinal k m n, word k m n N R A B u p ∈ (m4090 N m n k faddSem).outputs q) ∧
      (0 : ℝ) < 1 ∧ 16 / (3 * k : ℝ) < 1 ∧
      1 - 16 / (3 * k : ℝ) = (1 - 0) * (1 - 16 / (3 * k : ℝ))

/-- **`CanonicalNaN` is satisfiable**: every FP32 semantics with its NaN-pattern outputs canonicalized satisfies it. -/
def CanonicalNaNSat : Prop := ∀ fpSem : ℕ → ℚ → ℚ → ℚ → ℚ, CanonicalNaN (canonNaN fpSem)

/-- **`A1_fp` at `γ = 0` from a range gap** (`a1fpOn_of_rangeGap`; draft semantics, pending
review): under checkpoint independence, if at some grid point each non-final checked word takes a
value that `fpSem` never outputs, every FP32-pipe program that outputs the words at `S` costs at least `16·|S|`: no
FP32 step can be a word, so each word is a distinct big instruction. -/
def A1fpOnRangeGap : Prop :=
  ∀ (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat)
    (U : Finset ℕ), CheckpointIndependent k 16 R → U ⊆ range N →
    (∀ x ∈ U ×ˢ range (nonFinal k m n), ∃ ω : Noise N m n k,
      ∀ κ a b c, fpSem κ a b c ≠ word k m n N R A B x.1 x.2 ω) →
    A1fpOn fpSem 0 k m n N R A B U

/-- **The per-word cost of every `M4090` program from a range gap** (no A1, no A2, no MinRank; draft
semantics, pending review): under checkpoint independence, if at some grid point each non-final
checked word takes a value that `fpSem` never outputs, a well-formed program on inputs in `V₀` whose registers include
the checked words at `S ⊆ U × [0, nonFinal)` has `16·|S| ≤ cost`. -/
def PerWordM4090Gap : Prop :=
  ∀ (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat)
    (U : Finset ℕ) (L : List (Noise N m n k → ℚ)) (P : List (M4090Instr (Noise N m n k))),
    CheckpointIndependent k 16 R → U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) → (∀ o ∈ P, o.WF) →
    (∀ x ∈ U ×ˢ range (nonFinal k m n), ∃ ω : Noise N m n k,
      ∀ κ a b c, fpSem κ a b c ≠ word k m n N R A B x.1 x.2 ω) →
    ∀ S ⊆ U ×ˢ range (nonFinal k m n), (∀ x ∈ S, word k m n N R A B x.1 x.2 ∈ regs fpSem P L) →
      16 * S.card ≤ cost4090 P

/-- **TT_NCP for `M4090` with no A1** (`ttNCPM4090_of_nanGap`, here with no MinRank; draft semantics,
pending review): under `CanonicalNaN fpSem` and checkpoint independence, on an instance where each
non-final checked word takes a value in the range gap somewhere on the grid, TT_NCP's event fails for every
`M4090 fpSem` program that outputs the words, at `1 − γ₀ ≤ 1 − 16/(3k)`. -/
def TTNCPM4090NanGap : Prop :=
  ∀ (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ), CanonicalNaN fpSem →
    ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
      (q : (m4090 N m n k fpSem).Prog) (γ₀ : ℝ), 16 ∣ k → 0 < k → CheckpointIndependent k 16 R → U ⊆ range N →
      (∀ x ∈ U ×ˢ range (nonFinal k m n), ∃ ω : Noise N m n k, NaNGap (word k m n N R A B x.1 x.2 ω)) →
      (∀ u ∈ U, ∀ p < nonFinal k m n, word k m n N R A B u p ∈ (m4090 N m n k fpSem).outputs q) →
      γ₀ < 1 → 1 - γ₀ ≤ 1 - 16 / (3 * k : ℝ) →
      ¬ ((m4090 N m n k fpSem).cost q / (1 - γ₀) < ∑ _u ∈ U, 3 * ((m * k * n : ℕ) : ℝ))

/-- **`TTNCPM4090NanGap`'s hypotheses hold together** at the tight `γ₀ = 16/(3k)` (the no-A1 route is non-vacuous):
for every `k ≥ 1024` with `16 ∣ k`, every shape and unit set, there are `R` (checkpoint independent, MinRank ≥ 5),
inputs `A`, `B` whose every non-final checked word takes a value in the range gap at some grid point, the semantics
`canonNaN fpExact` satisfying `CanonicalNaN`, and an `M4090` program outputting the words. -/
def NanGapConsistent : Prop :=
  ∀ k, 1024 ≤ k → 16 ∣ k → ∀ (m n N : ℕ) (U : Finset ℕ), U ⊆ range N →
    ∃ (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat),
      CheckpointIndependent k 16 R ∧ MinRankGe k R 5 ∧
      (∀ x ∈ U ×ˢ range (nonFinal k m n), ∃ ω : Noise N m n k, NaNGap (word k m n N R A B x.1 x.2 ω)) ∧
      CanonicalNaN (canonNaN fpExact) ∧
      (∃ q : (m4090 N m n k (canonNaN fpExact)).Prog,
        ∀ u ∈ U, ∀ p < nonFinal k m n, word k m n N R A B u p ∈ (m4090 N m n k (canonNaN fpExact)).outputs q) ∧
      16 / (3 * k : ℝ) < 1

end Pouw.Dimension.Embed
