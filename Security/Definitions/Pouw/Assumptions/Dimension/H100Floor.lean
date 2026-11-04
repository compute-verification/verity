import Definitions.Pouw.Assumptions.Dimension.H100W
import Definitions.Pouw.Assumptions.Dimension.H100LoopFree

/-!
# `MH100`'s floor table, and the distinct-writes bound for citing the hardware (trusted statements; draft semantics)

**Draft semantics, pending review**.

**Which table is which.**
* `h100Nominal` is the datasheet's, unmeasured.
* `h100Measured` is Round 7's measurement. Since Round 8 it does not bound the hardware from below: two-source IADD
  measured 34.68 against its `int32` of 64, because ptxas now splits it across two pipes. So a bound at it does not
  transfer to the hardware.
* `h100LoopFree` is Round 8's measurement: each class at its least measured loop-free price. Its `int32` (34, two-source
  IADD) and `other` (36, two-source half2) are least measured, not floors. The immediate forms (IADD a+K, HADD2 a+K)
  were not measured, and could reach 32. So a bound at it transfers to programs of the measured forms.
* **`h100Floor`** (here) is a floor except at `tc`: `h100LoopFree` with `int32` and `other` at 32. That is the dispatch
  bound for a one-register writer (128 instructions per SM per clock), which no form can beat. `mac`, `ffma` and `fp64`
  sit on a physical bound too: the datasheet `wgmma` rate, dispatch for FADD, and the FP64 pipe for DADD. A
  multi-register writer is floored by its documented pipe, not by dispatch: IMAD.WIDE's 64 per SM per clock is 32 per
  register, and it measured 64.
* **Except `tc`**. `tc = 32` is legacy `mma.sync`'s measured 32.00, half the `wgmma` rate per MAC:
  at that rate m16n8k8 f16 → f32 would cost 16 per register. So `tc` is this board's measurement, with no slack at k8,
  and it rests on `H32`'s assertion about `sm_90` (the `h100Nominal` docstring). Two forms were never run on the H100
  SXM: m8n8k4 and sparse `mma.sp`, which would cost 8 and 16 per register at the `wgmma` rate. The H200 ran both
  (the review below).

**The hardware citations.** Both tables meet `H32` (`h32LoopFree`, `h32Floor`), so `DistinctWritesMH100On` holds at
each. `DistinctWritesLoopFree` covers programs of the measured forms.
`DistinctWritesFloor` covers every scalar form and every `wgmma` shape, and legacy tensor-core forms only as measured,
with no hypothesis on any instruction's accumulator or multiplicands. `DistinctWritesPeakTc`
(`Pouw.Dimension.H100PeakTc`) needs no measurement at `tc`: it prices legacy forms at 8, and charges 8 for a target a
cheap tensor-core instruction can write. All three are additive-W1 work, not time: co-issued mixes write a register as
often as every 21.45 units loop-free, and 21.33 at full dispatch (`H32`'s docstring).

## Review against the H200's measurements

Each definition below was checked against what was measured on one H200 SXM (`sm_90a`), as `Pouw.Dimension.H100W`'s
review lists the measurements.

* **`h100Floor`**: **matches**. The H200's table is `h100Floor` entry for entry: each class's cheapest writer costs
  32.003 (tc), 32.255 (ffma), 32.277 (int32), 32.003 (fp64) and 32.279 (other), and `mac` is `nominalMac`. So on the
  H200 even `tc` is a least price over every tensor-core form, since the census priced them all.
  - m8n8k4 with f16 inputs isn't a tensor instruction on `sm_90a`. ptxas lowers it to a scalar routine, priced as its
    instructions, and f64 m8n8k4 is DMMA at 128.
  - Sparse `mma.sp` costs at least 48.25, and sparse `wgmma` 32.003 with A from registers.
  - The entries are floors on the H200 for every priced form. 26 forms are unpriced (21 `LDGMC`, 4 `LDL` and `LEPC`),
    and `LDGMC` and a computed-address `LDL` are `other`. So the marker stays until they are priced.
* **`h100LoopFree`**: **looser** than the H200 at `int32` and `other`. IMAD beside LOP3 costs 32.277 and IADD with an
  immediate 32.35, against its 34, and two-source HADD2 with an immediate 32.28, against its 36. So a bound at it
  doesn't transfer to the H200, except for programs of the forms it was measured on.
* **`DistinctWritesLoopFree`**: the statement is as stated, but its table is looser than the H200, so it is no
  citation there and keeps its marker. `DistinctWritesFloor` gives the same 32 a target at a table the H200 meets.
* **`DistinctWritesFloor`**: at `h100Floor`, so it is checked with it. On the H200 it covers every tensor-core form
  too, m8n8k4 and `mma.sp` included, since every one was priced at 32 a register or more. Its marker stays with
  `h100Floor`'s.
-/

namespace Pouw.Dimension.H100

/-- **The H100 floor table** (draft semantics): `h100LoopFree` with `int32` and `other` at 32, the
dispatch bound for a one-register writer. It is a floor except at `tc`, which is this board's measurement of legacy
`mma.sync`. -/
def h100Floor : Prices100 := { h100LoopFree with int32 := 32, other := 32 }

/-! ## Pinned statements (draft semantics, pending review) -/

/-- **The distinct-writes bound at Round 8's loop-free prices** (draft semantics, pending review): at
`h100LoopFree`, every well-formed `MH100w` program on free inputs that matches on `S` a finite set `T` of targets,
pairwise distinct on `S` and none equal on `S` to a free value, costs at least `32·|T|`. This is the H100 citation for
programs of the measured forms, in place of `DistinctWritesMeasured`. -/
def DistinctWritesLoopFree : Prop :=
  ∀ (G : Type) (F : Set (G → ℚ)), (∀ q : ℚ, (fun _ : G => q) ∈ F) →
    ∀ (S : Set G) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (G → ℚ)) (P : List (Op100 G)) (T : Finset (G → ℚ)),
      (∀ f ∈ L, f ∈ F) → (∀ o ∈ P, o.WFw h100LoopFree F) → DistinctOn S F T → OutputsOn fpSem S P L T →
      32 * T.card ≤ cost100 h100LoopFree P

/-- **The distinct-writes bound at the floor table** (draft semantics, pending review): the same at
`h100Floor`. This is the H100 citation for every scalar form (the unmeasured immediate forms included) and every
`wgmma` shape, and for legacy tensor-core forms only as measured: `tc` is this board's measurement,
and m8n8k4 and sparse `mma.sp` were never run on the H100 SXM. On the H200 every tensor-core form was priced, at 32 a
register or more. -/
def DistinctWritesFloor : Prop :=
  ∀ (G : Type) (F : Set (G → ℚ)), (∀ q : ℚ, (fun _ : G => q) ∈ F) →
    ∀ (S : Set G) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (G → ℚ)) (P : List (Op100 G)) (T : Finset (G → ℚ)),
      (∀ f ∈ L, f ∈ F) → (∀ o ∈ P, o.WFw h100Floor F) → DistinctOn S F T → OutputsOn fpSem S P L T →
      32 * T.card ≤ cost100 h100Floor P

end Pouw.Dimension.H100
