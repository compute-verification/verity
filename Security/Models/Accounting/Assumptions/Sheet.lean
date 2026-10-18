import Definitions.Accounting.TS.Sheet
import Models.Accounting.Assumptions.TS

/-!
# The data-sheet axioms A1–A7 (trusted)

Seven physical assumptions (note:20261017T1818Z-draft-axioms §2), each a `Prop` taken as a hypothesis, never an
`axiom`, and each a vendor statement (decision 96: data-sheet constants, no GPU self-attestation). Each docstring opens
with its claim id (`verity.core.claims`) and names its document. A1–A5 are claimed for a run of the stated device
against its sheet (`Accounting.TS.Sheet`). A6 and A7 are claimed for a cheater's labelled run (`Anatomy`) at §0.4's
and §1.3's labelling, as `Models.Accounting.Assumptions.TS`'s are.

What they give (`Models.Accounting.Guarantees.Sheet`): A1–A5 give (I1) `FacetCeilings` on the data-sheet machine; A6
gives (I2)/(I4) `WordsWritten` and, with the labelling's split, (I3) `ScalarRoute` at `q, q_T ≤ ½`, `b_T ≤ 0`; A7 is
(I5). With (I6)–(I8) they give Theorem D-ideal (2.7♭) on the data-sheet machine.

Data independence needs no axiom: a peak is a maximum over every operand value, so A3–A5 hold for any data.
-/

namespace Accounting.Assumptions.TS

open Accounting.TS

section Run

variable {K P : Type} [Fintype K] [DecidableEq P]

/-- `accounting-sheet-clock` (data sheet). **A1 CLOCK**: the run's SM clock is positive and at most `fmax`, the data
sheet's maximum boost clock (the B200's sheet gives none: its driver's reported maximum). A clock offset applied at run
time is a configuration, checked at run time, and outside this statement. -/
def SheetClock (S : Sheet K P) (Q : Run K) : Prop := 0 < Q.clock ∧ Q.clock ≤ S.fmax

/-- `accounting-sheet-dispatch` (architecture whitepapers; A2u in no vendor document). **A2 DISPATCH**: an SMSP issues
at most one warp instruction a cycle, whatever its pipe, so the run's warp instructions per SMSP are at most its cycles
per SMSP (whitepapers Volta through Blackwell: four partitions an SM, each with one warp scheduler and one dispatch
unit). **A2u**: the classes `K` include the uniform-datapath instructions, each taking an issue slot; no vendor
document states it (decision 84's issue ceiling). -/
def SheetDispatch (Q : Run K) : Prop := ∑ k, Q.n k ≤ Q.cycles

/-- `accounting-sheet-tensor` (data sheets and whitepapers). **A3 TENSOR**: the tensor cores do at most the sheet's
dense MACs an SM clock for each operand type (dense TOPS over 2 · SMs · boost clock), a quarter of them an SMSP, the
types sharing the pipe's time: the run's tensor MACs over a quarter of their type's peak are at most its cycles. -/
def SheetTensor (S : Sheet K P) (Q : Run K) : Prop := ∑ k, S.macs k / (S.peak k / 4) * Q.n k ≤ Q.cycles

/-- `accounting-sheet-pipe` (CUDA C++ Programming Guide, Arithmetic Instructions; Nsight Compute, Pipelines). **A4
PIPE**: the classes Nsight Compute puts on pipe `p` share that pipe's Programming-Guide rate, so their lane results (32
a warp instruction) per SMSP are at most a quarter of `width p` a cycle. -/
def SheetPipe (S : Sheet K P) (Q : Run K) : Prop :=
  ∀ p, ∑ k, (if S.pipe k = some p then (32 : ℝ) else 0) * Q.n k ≤ S.width p / 4 * Q.cycles

/-- `accounting-sheet-smem` (CUDA C++ Programming Guide, Shared Memory). **A5 SMEM**: shared memory serves a request's
distinct banks together, 32 banks × 4 B = 128 B an SM clock, loads and stores together: 32 B an SMSP-cycle. A
request's distinct bytes can be one word's (a broadcast load, a store to one address), so a shared-memory class holds
one request's 128 B whatever its width. -/
def SheetSmem (S : Sheet K P) (Q : Run K) : Prop := ∑ k, S.smem k * Q.n k ≤ 32 * Q.cycles

end Run

variable {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F]
  [Fintype X]

/-- `accounting-sheet-words` (PTX ISA). **A6 WORDS**: an instruction writes at most its destination registers' 32-bit
words. Writer kind `t`'s `ω_t` is its fragment's words (mma.sync m16n8k32 s32: 128 a warp instruction, m8n8k16: 64;
wgmma m64n256k32: 4096 a warp; tcgen05.mma 128 × 256: 32768; a uniform instruction: 1), and a lane instruction's
destination is at most 64 bits, so the vector- and table-headed words are at most two a lane of the vector
instructions the labelling gives them. -/
def SheetWords (Z : Model K R T L U F X) (A : Anatomy K T L U) : Prop :=
  (∀ t, A.words (.writer t) ≤ Z.omega t * A.nword (.writer t) (Z.cls t)) ∧
    A.words .vector ≤ 2 * (32 * ∑ k ∈ Z.vec, A.nword .vector k) ∧
    ∀ l, A.words (.table l) ≤ 2 * (32 * ∑ k ∈ Z.vec, A.nword (.table l) k)

/-- `accounting-sheet-semantics` (PTX ISA). **A7 SEMANTICS**: each instruction computes the function its ISA defines
and writes nothing else; whether a unit multiplies, and in which format, is read off its definition. Lean has no model
of instruction semantics, so it is stated as what RT-ISA's reconciliation reads off the definitions
(`benchmarks/pouw/theory/isa_escape.py`, with A3): no unit outside the menu heads a fresh word (`NoSum`), and a side
unit's words cost at least its per-word vector, read off its definition and priced on the sheet's rows
(`SideUnits`). -/
def SheetSemantics (Z : Model K R T L U F X) (A : Anatomy K T L U) : Prop := NoSum A ∧ SideUnits Z A

end Accounting.Assumptions.TS
