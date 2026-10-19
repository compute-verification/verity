import Models.Accounting.Assumptions.Sheet

/-!
# Lemma J, the slot step (trusted; statements, not yet proved)

FP-JOINT's Lemma J (note:20261019T0745Z-draft-fp-joint, G2) on the data-sheet machine `Sheet.machine`. A tensor class
`k`'s instruction writes `slots k` output slots, its fragment's `M·N` (128 for mma.sync m16n8, 4096 a warp for wgmma
m64n256, 32768 for tcgen05.mma 128 × 256; `0` off the tensor pipe). The credited class `c`'s instruction holds the
tensor pipe `(macs c / slots c) / (peak c / 4)` SMSP-cycles a slot, its depth `K_c` over a quarter of its type's dense
MACs an SM clock: the step, 1/8 on sm_120, 1/32 on H200 and 1/64 on B200.

* (J1), the sheet's slot-step identity, is vendor arithmetic (PTX ISA 9.4's shapes, the sheets' rates), checked per
  form by `benchmarks/pouw/theory/isa_escape_joint.py`: every form's full-depth instruction has a tensor row of exactly
  `slots k` steps (FP64 more). A3 caps MACs per operand type, not per shape, so a shallower shape's row is half a step
  a slot or less, and a class of peak `0` (sm_120's FP64) has row `0`.
* `Accounting.Assumptions.TS.SlotStep`, premise (M): every tensor class holds the pipe at least a step a slot, the
  shallower shapes and the peak-`0` classes included.
* `SlotStepOfSheet`: A3 and (J1) on every class give (M), so (M) adds to A3 only at those classes.
* `SlotStepWords`, (J2): with A6 (`WordsWritten`) and (M), the words a tensor writer heads cost a step each,
  `step · N_T ≤ cycles`, on any form, joint encodings included. (J3), a word's gain over its credit `L/R_c` at most
  `L/K_c`, divides it.

A packed slot (REFEREE S14), more than one committed word through one slot's output, gives its words vector heads, so
they are outside `N_T`. The checker bounds that class per device: closed on B200 by A6 alone, on H200 at a fold of at
least ½ a lane instruction a word, and open on sm_120 for decodes between A6's ½ and the break-even.
-/

namespace Accounting.Assumptions.TS

open Accounting.TS

variable {K P : Type} [Fintype K] [DecidableEq P]

/-- `accounting-ts-slot-step` (measured, device-scoped). **(M) A STEP A SLOT**: every tensor instruction holds the
tensor pipe at least `slots k` of the credited instruction's steps as well as its sheet row (A3), the classes sharing
the pipe's time: the run's tensor time, each class at the larger of the two, is at most its cycles. At a kind's full
depth the two agree ((J1)); (M) adds the shallower shapes and the classes the sheet gives no peak. Measured on sm_120
(RTX PRO 6000) for IMMA.16816 (k16, and m8n8k16 with 64 of its slots), IMMA.SP.16832 and HMMA.1684 at a step a slot or
more, and DMMA.8x8x4 at 73 (TENSOR M1, r20261017-040443-f33c, r20261017-041103-d35d, r20261017-041754-7630); not
measured for sm_120's other shallower shapes, nor at all on H200 (sm_90a) and B200 (sm_100a), where it is (M′); false
on sm_89, where IMMA.16816 runs at the MAC rate (EPS1-ADA, r20261017-063245-fc25). -/
def SlotStep (S : Sheet K P) (slots : K → ℝ) (c : K) (Q : Run K) : Prop :=
  ∑ k, max (S.macs k / (S.peak k / 4)) (slots k * (S.macs c / slots c / (S.peak c / 4))) * Q.n k ≤ Q.cycles

end Accounting.Assumptions.TS

namespace Accounting.Guarantees.TS

open Accounting.TS Accounting.Assumptions.TS

/-- **(J1) gives (M).** If every class's sheet row is at least `slots k` steps (the vendor arithmetic
`isa_escape_joint.py` checks at each kind's full depth), A3 on a run gives (M) on it. -/
def SlotStepOfSheet : Prop :=
  ∀ {K P : Type} [Fintype K] [DecidableEq P] (S : Sheet K P) (slots : K → ℝ) (c : K) (Q : Run K),
    (∀ k, slots k * (S.macs c / slots c / (S.peak c / 4)) ≤ S.macs k / (S.peak k / 4)) →
    SheetTensor S Q → SlotStep S slots c Q

/-- **Lemma J (J2, the slot step).** For a credited class `c` with slots and a peak, tensor writer kinds `tw` each
writing at most its class's slots (A6's `ω_t` an MMA kind is its fragment's `M·N`), a labelled run that splits its
counts, A6 (`WordsWritten`) and (M) (`SlotStep`): the committed words the kinds of `tw` head cost the tensor pipe a
step each, `step · Σ_{t ∈ tw} words t ≤ cycles`, whatever forms the run uses. -/
def SlotStepWords : Prop :=
  ∀ {K P R T L U F X : Type} [Fintype K] [DecidableEq P] [Fintype R] [Fintype T] [Fintype L] [Fintype U]
    [Fintype F] [Fintype X] (S : Sheet K P) (slots : K → ℝ) (c : K) (Z : Model K R T L U F X)
    (A : Anatomy K T L U) (tw : Finset T),
    0 < slots c → 0 < S.peak c → 0 ≤ S.macs c → (∀ k, 0 ≤ slots k) → (∀ t ∈ tw, Z.omega t ≤ slots (Z.cls t)) →
    A.Split → WordsWritten Z A → SlotStep S slots c A.run →
    S.macs c / slots c / (S.peak c / 4) * ∑ t ∈ tw, A.words (.writer t) ≤ A.run.cycles

end Accounting.Guarantees.TS
