import Definitions.Accounting.TS.Sheet
import Definitions.Accounting.Assumptions.TS

/-!
# The data-sheet axioms A1–A7 (trusted)

Seven physical assumptions, each a `Prop` taken as a hypothesis, never an
`axiom`, and each a vendor statement (data-sheet constants, no GPU self-attestation). Each docstring opens
with its claim id (`verity.core.claims`) and names its document. A1–A5 are claimed for a run of the stated device
against its sheet (`Accounting.TS.Sheet`), the node's devices the stated ones by `accounting-declared-device` (`Run`'s
section, below). A6 and A7 are claimed for a cheater's labelled run (`Anatomy`) at §0.4's and §1.3's labelling, as
`Definitions.Accounting.Assumptions.TS`'s are.

What they give (`Definitions.Accounting.Guarantees.Sheet`): A1–A5 give (I1) `FacetCeilings` on the data-sheet machine; A6
gives (I2)/(I4) `WordsWritten` and, with the labelling's split, (I3) `ScalarRoute` at `q, q_T ≤ ½`, `b_T ≤ 0`; A7 is
(I5). With (I6)–(I8) they give Theorem D-ideal (2.7♭) on the data-sheet machine.

Data independence needs no axiom: a peak is a maximum over every operand value, so A3–A5 hold for any data.
-/

namespace Accounting.Assumptions.TS

open Accounting.TS

section Run

/-! A1–A5 are claimed for a run of the stated device, and which device a run is on is the auditor's to know, not the
run's: `accounting-declared-device`, per node and per window, says every GPU on the node is declared, each
is the declared board and vBIOS, and no SM clock exceeds `fmax`, the top of that vBIOS's clock table. A1's clock rests
on it: nothing checks a clock, an offset or a lock at run time but the prover's own driver. -/

variable {K P : Type} [Fintype K] [DecidableEq P]

/-- `accounting-sheet-clock` (assumed; `fmax` from each device's source). **A1 CLOCK**: the run's SM clock is positive
and at most `fmax`. sm_120's 2430 MHz is the maximum SM clock its driver reports (nvidia-smi on the RTX PRO 6000 Server
Edition, SPEC's P1), not a data sheet's, and the same die boosts to 2617 MHz as the Workstation Edition; the B200's
sheet gives none either, so its `fmax` is its driver's reported maximum too, and the H200's 1980 MHz is EPS1 DC's
measured f_max at default clocks (its sheet quotes TOPS at 1830 MHz). A driver's report and a measurement are the
device's own, not a source for a sheet, so A1 is assumed there. So is the absence of a clock offset: nothing checks it
at run time, and the clock is the prover's configuration. Our nodes' driver allows none (NVML's allowed offset range
`[0, 0]`), but a prover picks its own VBIOS and driver. -/
def SheetClock (S : Sheet K P) (Q : Run K) : Prop := 0 < Q.clock ∧ Q.clock ≤ S.fmax

/-- `accounting-sheet-dispatch` (architecture whitepapers; A2u in no vendor document). **A2 DISPATCH**: an SMSP issues
at most one warp instruction a cycle, whatever its pipe, so the run's warp instructions per SMSP are at most its cycles
per SMSP (whitepapers Volta through Blackwell: four partitions an SM, each with one warp scheduler and one dispatch
unit). **A2u**: the classes `K` include the uniform-datapath instructions, each taking an issue slot; no vendor document
states it. **Catch-all**: every warp instruction takes at least one issue slot, whether or not its class is listed, so
`K` counts every warp instruction the run issues, one no other class names in the device's catch-all class, priced at
its issue slot (`NOROW` alone on sm_120 and on the H200's and B200's sheets, `UNLISTED` on the measured issue row on
their measured readings DC-90-ri and DC-100-ri, and on DC-90-ri on the JI90 lines too, its instructions timed beside the
IGMMA; the exporters' `CATCH_ALL`). sm_120's is a vector class, so A6 reads such an instruction as a lane head of at
most two words a lane; one with a wider destination is assumed to be a memory access (`LDL.128`, `LD.128`), which heads
no vector word. The DC readings' are neither writers nor vector classes, so they head no word. A reading whose classes
have none leaves an unlisted instruction free on every row, as sm_120's measured reading does. **Bulk count**: a bulk or
TMA instruction (`cp.async.bulk`, `cp.reduce.async.bulk` and their tensor forms) of size `S` counts `S/99 KB` in
sm_120's UTMABIG (A5b), at most one, so `K` holds it in at most its slot. -/
def SheetDispatch (Q : Run K) : Prop := ∑ k, Q.n k ≤ Q.cycles

/-- `accounting-sheet-tensor` (data sheets and whitepapers). **A3 TENSOR**: the tensor cores do at most the sheet's
dense MACs an SM clock for each operand type (dense TOPS over 2 · SMs · the clock the figure is quoted at, which need
not be `fmax`: the H200's tensor boost clock, 1830 MHz, against its f_max of 1980 MHz), a quarter of them an SMSP, the
types sharing the pipe's time: the run's tensor MACs over a quarter of their type's peak are at most its cycles. -/
def SheetTensor (S : Sheet K P) (Q : Run K) : Prop := ∑ k, S.macs k / (S.peak k / 4) * Q.n k ≤ Q.cycles

/-- `accounting-sheet-pipe` (CUDA C++ Programming Guide, Arithmetic Instructions; Nsight Compute, Pipelines). **A4
PIPE**: the classes Nsight Compute puts on pipe `p` share that pipe's Programming-Guide rate, so their lane results (32
a warp instruction) per SMSP are at most a quarter of `width p` a cycle. -/
def SheetPipe (S : Sheet K P) (Q : Run K) : Prop :=
  ∀ p, ∑ k, (if S.pipe k = some p then (32 : ℝ) else 0) * Q.n k ≤ S.width p / 4 * Q.cycles

/-- `accounting-sheet-smem` (CUDA C++ Programming Guide, Shared Memory; A5b in no vendor document). **A5 SMEM**: shared
memory serves a request's distinct banks together, 32 banks × 4 B = 128 B an SM clock, loads and stores together: 32 B
an SMSP-cycle. A request's distinct bytes can be one word's (a broadcast load, a store to one address), so a
shared-memory class holds one request's 128 B whatever its width. **A5b**: a bulk or TMA instruction holds its size `S`
of bank bytes at that rate, the bytes it reads or writes on one side of a block's shared memory (a reduction into
shared memory moves `3S`), so its `S/99 KB` of a UTMABIG count holds `S` (sm_120's UTMABIG holds 99 KB, a block's
most). No vendor document states what the bulk engine's traffic costs the banks, so it is named, as A2u is: no census
program exceeds 1 on DS_smem with it, and R6r's measured row already charges TMAred512 its 512 B. Without it DS-120
charges a bulk reduction one issue slot and no other row, and a fold on the bulk engine costs less than the menu's. -/
def SheetSmem (S : Sheet K P) (Q : Run K) : Prop := ∑ k, S.smem k * Q.n k ≤ 32 * Q.cycles

end Run

variable {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F]
  [Fintype X]

/-- `accounting-sheet-words` (PTX ISA). **A6 WORDS**: an instruction writes at most its destination registers' 32-bit
words. Writer kind `t`'s `ω_t` is its fragment's words (mma.sync m16n8k32 s32: 128 a warp instruction, m8n8k16: 64;
wgmma m64n256k32: 4096 a warp; tcgen05.mma 128 × 256: 32768; a uniform instruction: 2, since `UIMAD.WIDE.U32` writes a
uniform register pair; DMMA, `mma.sync .f64`: two
registers an element, 128 at m8n8k4 and 256 at m16n8k16). A vector or table word's head computes it on a lane, at most
64 bits a lane (IMAD.WIDE, IADD.64, DFMA, sm_100's FFMA2), so the vector- and table-headed words are at most two a lane
of the vector instructions the labelling gives them. The wider lane destinations head no such word: a load, LDSM, a
texture fetch, a 128-bit atomic or a tensor-memory load (LDS.128 and LDSM.x4: 4 words a lane; LDG.E.ENL2.256: 8;
LDTM.32x32b.x128: 128) writes memory's content, a copy that keeps its writer's head or a table entry that its word's
lane head combines. -/
def SheetWords (Z : Model K R T L U F X) (A : Anatomy K T L U) : Prop :=
  (∀ t, A.words (.writer t) ≤ Z.omega t * A.nword (.writer t) (Z.cls t)) ∧
    A.words .vector ≤ 2 * (32 * ∑ k ∈ Z.vec, A.nword .vector k) ∧
    ∀ l, A.words (.table l) ≤ 2 * (32 * ∑ k ∈ Z.vec, A.nword (.table l) k)

/-- `accounting-sheet-semantics` (PTX ISA). **A7 SEMANTICS**: each instruction computes the function its ISA defines
and writes nothing else; whether a unit multiplies, and in which format, is read off its definition. Lean has no model
of instruction semantics, so it is stated as what the ISA reconciliation reads off the definitions
(`benchmarks/pouw/theory/isa_escape.py`, with A3): no unit outside the menu heads a fresh word (`NoSum`), and a side
unit's words cost at least its per-word vector, read off its definition and priced on the sheet's rows
(`SideUnits`). -/
def SheetSemantics (Z : Model K R T L U F X) (A : Anatomy K T L U) : Prop := NoSum A ∧ SideUnits Z A

end Accounting.Assumptions.TS
