import Models.Pouw.Assumptions.Dimension.H100W
import Models.Pouw.Assumptions.Dimension.H100LoopFree

/-!
# `MH100`'s floor table, and the distinct-writes bound for citing the hardware (trusted statements; draft semantics)

**Draft semantics, pending review** (`internal/pouw-fp8/genuine-fp8-interface.md` §1.1; red team Phase 18b in
`internal/pouw/new-crypto/red-team.md`, F18b-1 and F18b-2).

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
* **Except `tc`** (red team F18c-1). `tc = 32` is legacy `mma.sync`'s measured 32.00, half the `wgmma` rate per MAC:
  at that rate m16n8k8 f16 → f32 would cost 16 per register. So `tc` is this board's measurement, with no slack at k8,
  and it rests on `H32`'s assertion about `sm_90` (the `h100Nominal` docstring). Two forms were never run: m8n8k4 and
  sparse `mma.sp`, which would cost 8 and 16 per register at the `wgmma` rate.

**The hardware citations.** Both tables meet `H32` (`h32LoopFree`, `h32Floor`), so `DistinctWritesMH100On` holds at
each. `DistinctWritesLoopFree` is the one F18b-1 asks for, and it covers programs of the measured forms.
`DistinctWritesFloor` covers every scalar form and every `wgmma` shape, and legacy tensor-core forms only as measured
(F18c-1), with no hypothesis on any instruction's accumulator or multiplicands. `DistinctWritesPeakTc`
(`Pouw.Dimension.H100PeakTc`) needs no measurement at `tc`: it prices legacy forms at 8, and charges 8 for a target a
cheap tensor-core instruction can write. All three are additive-W1 work, not time: co-issued mixes write a register as
often as every 21.45 units loop-free, and 21.33 at full dispatch (`H32`'s docstring; `docs/pouw/w1-time-model.md`).
-/

namespace Pouw.Dimension.H100

/-- **The H100 floor table** (red team F18b-2; draft semantics): `h100LoopFree` with `int32` and `other` at 32, the
dispatch bound for a one-register writer. It is a floor except at `tc`, which is this board's measurement of legacy
`mma.sync` (F18c-1). -/
def h100Floor : Prices100 := { h100LoopFree with int32 := 32, other := 32 }

/-! ## Pinned statements (draft semantics, pending review) -/

/-- **The distinct-writes bound at Round 8's loop-free prices** (draft semantics, pending review; F18b-1): at
`h100LoopFree`, every well-formed `MH100w` program on free inputs that matches on `S` a finite set `T` of targets,
pairwise distinct on `S` and none equal on `S` to a free value, costs at least `32·|T|`. This is the H100 citation for
programs of the measured forms, in place of `DistinctWritesMeasured`. -/
def DistinctWritesLoopFree : Prop :=
  ∀ (G : Type) (F : Set (G → ℚ)), (∀ q : ℚ, (fun _ : G => q) ∈ F) →
    ∀ (S : Set G) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (G → ℚ)) (P : List (Op100 G)) (T : Finset (G → ℚ)),
      (∀ f ∈ L, f ∈ F) → (∀ o ∈ P, o.WFw h100LoopFree F) → DistinctOn S F T → OutputsOn fpSem S P L T →
      32 * T.card ≤ cost100 h100LoopFree P

/-- **The distinct-writes bound at the floor table** (draft semantics, pending review; F18b-2): the same at
`h100Floor`. This is the H100 citation for every scalar form (the unmeasured immediate forms included) and every
`wgmma` shape, and for legacy tensor-core forms only as measured (red team F18c-1): `tc` is this board's measurement,
and m8n8k4 and sparse `mma.sp` were never run. -/
def DistinctWritesFloor : Prop :=
  ∀ (G : Type) (F : Set (G → ℚ)), (∀ q : ℚ, (fun _ : G => q) ∈ F) →
    ∀ (S : Set G) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (G → ℚ)) (P : List (Op100 G)) (T : Finset (G → ℚ)),
      (∀ f ∈ L, f ∈ F) → (∀ o ∈ P, o.WFw h100Floor F) → DistinctOn S F T → OutputsOn fpSem S P L T →
      32 * T.card ≤ cost100 h100Floor P

end Pouw.Dimension.H100
