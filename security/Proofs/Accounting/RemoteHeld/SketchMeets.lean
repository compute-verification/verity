module

public import Proofs.Accounting.RemoteHeld.Sketch
import Proofs.Pouw.Prob

/-!
# The deployed sketch bound's premises meet, and its conclusion bites

`Accounting.Guarantees.Remote.SketchBoundDeployed` has its premises met at an honest session
(`SketchBoundDeployed_honest`) and its conclusion false on a trace that accepts everything
(`SketchBoundDeployed_nontrivial`).

* **Met.** The honest session (`SketchMeets.honest`): one pass of one candidate over a store of one 4096-byte block,
  at the deployed receipts (`m = 128`, `h = 512`, 64 sign rows). Its device holds the block's 128 sectors on chip
  (`S = 2^15` bits) and tags the block with one query, the receipt hash of its index and its sketch, so the block's
  receipt is right at every outcome (`SketchMeets.honest_accepts`). SHA-512's keys at the seed are the outcome's own
  coins (`Hs` the keys, `keys` the identity), so `SketchHashRandom` holds at any `ε`. Its trace
  (`SketchMeets.honestTrace`) accepts where the block's receipt is right, counts it correct and holds it in node memory
  at every tick: `KeyBudget` at one query a pass, `Receipted` and `Present` hold.
* **Bites.** The toy (`SketchMeets.acceptAll`): one pass over one block, counted correct and never present, on an audit
  that accepts every outcome. `SketchBound` there claims probability at most the deployed δs at `Q = 0`, `P = 1`,
  `ε = 0` and `N = 1`, below `2^-511`, for an event of probability `1`.
-/

@[expose] public section

namespace Accounting.SecurityProofs.Remote

open Finset Accounting.RateSketch Accounting.Remote.Held Accounting.Remote.Held.Assumptions Accounting.Remote.Sketch
  Accounting.Remote.Sketch.Assumptions

namespace SketchMeets

/-- One candidate seed a pass. -/
abbrev G₁ : Fin 1 → ℕ := fun _ => 1

/-- What SHA-512's outputs are read over: the session's keys themselves. -/
abbrev Hs₁ : Type := Keys 1 1 128 512 64 G₁

/-- The honest device: it holds the block's 128 sectors on chip and tags the block, its index and its sketch, with one
query. -/
def device : RateSketch.Prover 1 128 0 512 32768 0 1 256 Sector (SketchSigns 64 128) (Fin 64 → ZMod (2 ^ 32)) where
  held := univ
  held_le := by simp
  chip σ v := .tag (Fin.elim0, 0, sketch σ fun u => v ⟨(0, u), mem_univ _⟩) fun y => .done fun _ => y
  bounded σ v O D := by simp [Chip.linkSteps, Chip.queries]

/-- **The honest session**: one pass of one candidate, running `device`, its key the outcome's own. -/
def honest : Remote.Sketch.Run Unit Hs₁ 1 1 128 512 32768 0 1 64 G₁ where
  device _ _ _ _ := device
  keys _ _ x := x
  kept _ _ := 0

/-- The honest session's trace: it accepts where the block's receipt is right, counts the block correct and holds it
in node memory at every tick. -/
def honestTrace : Trace (Outcome Unit Hs₁ 1 128) where
  blocks := 1
  passes := 1
  accepts ω := (0 : Fin 1) ∈ honest.correctAt ω 0 0
  tick _ p := p
  plan _ _ := univ
  correct _ _ := univ
  mem _ _ := univ
  chip _ _ := ∅
  rebuilt _ _ := 0
  disk _ _ := 0
  link _ _ := 0

/-- The honest device's receipt is right at every outcome, so the honest session is accepted on all of them. -/
theorem honest_accepts (ω : Outcome Unit Hs₁ 1 128) : honestTrace.accepts ω := by
  show (0 : Fin 1) ∈ honest.correctAt ω 0 0
  simp only [Run.correctAt, Run.devAt, Run.coins, Prover.correct, Prover.tags, Prover.prog, honest, device, Chip.run,
    mem_filter, mem_univ, true_and]
  exact congrArg _ (Prod.ext (Subsingleton.elim _ _) rfl)

/-- The toy: one pass over one block, counted correct and never in node memory or on a chip, on an audit that accepts
every outcome. -/
def acceptAll : Trace Unit where
  blocks := 1
  passes := 1
  accepts _ := True
  tick _ _ := 0
  plan _ _ := univ
  correct _ _ := univ
  mem _ _ := ∅
  chip _ _ := ∅
  rebuilt _ _ := 0
  disk _ _ := 0
  link _ _ := 0

end SketchMeets

open SketchMeets

/-- **`SketchBoundDeployed`'s premises meet** at the honest session: one pass of one candidate at the deployed receipts,
its device holding the block and tagging it with one query, SHA-512's keys the outcome's own. -/
theorem SketchBoundDeployed_honest :
    KeyBudget honest (fun _ => 1) ∧ Receipted honest honestTrace ∧ Present honest honestTrace ∧
      Accounting.Assumptions.SketchHashRandom honest ((((1 : ℕ) : ℝ) + ((1 : ℕ) : ℝ)) ^ 2 / 2 ^ 513) := by
  refine ⟨fun b j η κ D => ?_, fun ω hω j b _ => ⟨0, ?_, ?_⟩, fun ω _ j _ _ b _ => ⟨j, le_rfl, Nat.lt_succ_self _, ?_⟩,
    fun b D E => le_self_add⟩
  · simp [honest, device, Prover.prog, Chip.queries]
  · rw [Subsingleton.elim j 0]
    exact hω
  · exact (Nat.lt_one_iff.1 b.isLt).symm
  · unfold Trace.present
    exact mem_union_left _ (mem_univ _)

/-- **`SketchBoundDeployed`'s conclusion bites**: on an audit that accepts everything, a pass's correct receipt of a
block never present breaks the sketch bound at the deployed δs (`Q = 0`, `P = 1`, `ε = 0`, `N = 1`). -/
theorem SketchBoundDeployed_nontrivial :
    ¬ SketchBound acceptAll (((0 : ℕ) : ℝ) / 2 ^ 216 + (((0 : ℕ) : ℝ) + ((1 : ℕ) : ℝ)) * (0 + ((1 : ℕ) : ℝ) / 2 ^ 512) +
      (((0 : ℕ) : ℝ) + ((1 : ℕ) : ℝ)) ^ 2 / 2 ^ 513) := by
  intro h
  have hev : ∀ ω : Unit, acceptAll.accepts ω ∧ ¬ (∀ p < acceptAll.passes, ∀ j ∈ acceptAll.correct ω p,
      ∃ t, acceptAll.tick ω p ≤ t ∧ t < acceptAll.tick ω (p + 1) ∧ j ∈ acceptAll.present ω t) := by
    intro ω
    refine ⟨trivial, fun hp => ?_⟩
    obtain ⟨t, -, -, ht⟩ := hp 0 Nat.zero_lt_one ⟨0, Nat.zero_lt_one⟩ (mem_univ _)
    unfold Trace.present at ht
    exact notMem_empty _ ((mem_union.1 ht).elim id id)
  have h1 : (1 : ℝ) ≤ _ := le_trans (Pouw.pr_eq_one hev).symm.le h
  have a : ((2 : ℝ) ^ 512)⁻¹ ≤ 1 / 4 :=
    le_trans (inv_anti₀ (by positivity) (pow_le_pow_right₀ (by norm_num) (by norm_num : 2 ≤ 512))) (by norm_num)
  have b : ((2 : ℝ) ^ 513)⁻¹ ≤ 1 / 4 :=
    le_trans (inv_anti₀ (by positivity) (pow_le_pow_right₀ (by norm_num) (by norm_num : 2 ≤ 513))) (by norm_num)
  norm_num at h1
  linarith

end Accounting.SecurityProofs.Remote
