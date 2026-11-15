module

public import Definitions.Claims.SamplingTwoReg
public import Proofs.Flock.Claims.Witnesses.SamplingTwoReg
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.ZK.NonVacuous
import Proofs.Flock.SamplingGeneric.TwoRegMiss

/-!
# The two-stage miss bound says something, and the headline's premises hold

`CompiledTwo.MissTwo` fails at `sksLTwo` (`Witnesses.SamplingTwo`: `sksL`, which accepts after one round, samples
nothing and opens nothing, with the two-stage design `toyTwo`), its silent prover and any hash, at both rates `1`, which
`toyTwo` meets (`capTwo_honest`): `CompiledTwo.MissTwo_fails` over the spec, and at coins `Bool` and SHA-512
(`FlockDeploy.sha512`) the nontrivial lemma of `CompilerMissTwo`.

* `sksLTwo`'s committed boundary trace opens nothing (`sksL_regVal`), so its one replay block holds an incorrect proof
  block at every interior (`toy_wrongInAt`) and is wrong at the boundary (`toy_wrongAtBoundaryAt`): the bound
  `(1 − 1·1)^{|W|}` is `0`.
* It samples nothing, so its sample holds no proof block of the committed incorrect set at any coins: the miss is `1`
  (`compiledTwoMiss_fails`).

The compilers' premises hold together at two sessions, at `twoLTwo`, which meets the two-stage contract
(`compilerKSTwo_honest`, which `CompilerKSRegTwo` reads: its conclusion holds that compiler's premises):
* `compilerMissTwo_honest`: its design `twoTwo` catches each of its two replay blocks and samples each cell, of one
  proof block each, at rate `1` (`twoTwo_firstRate`, `twoTwo_secondRate`).
* `compilerHeadlineTwo_honest`: it meets the committed form at two stages (`compilerKSRegTwo`).
-/

@[expose] public section

namespace Claims.Witness.SamplingGeneric

open Claims.SamplingGeneric
open Claims.Sampling (cell window wrongAtBoundaryAt)
open Claims.Witness.Sampling (toyStmt toyPart)

/-- The two-session toy's first stage `all` catches each of its two replay blocks at rate `1`. -/
theorem twoTwo_firstRate : twoTwo.FirstRate 1 := by
  intro T hT
  show twoStmt.design.Pa twoPart T ≤ (1 - 1) ^ T.card
  rw [two_Pa, sub_self]
  split_ifs with hw
  · have hT0 : T = ∅ := by
      refine Finset.eq_empty_of_forall_notMem fun v hv => ?_
      have hlen : twoTwo.part.blocks.length = 2 := rfl
      have hv2 : v < 2 := by
        have := Finset.mem_range.1 (hT hv)
        omega
      have hvw : v ∈ window 0 2 T := by
        simp [window, hv]
        omega
      rw [hw] at hvw
      simp at hvw
    subst hT0
    simp
  · exact pow_nonneg le_rfl _

/-- The two-session toy's second stage samples each of its cells, of one proof block each, whole: rate `1`. -/
theorem twoTwo_secondRate : twoTwo.SecondRate twoPart 1 := by
  intro ρ hρ t ht _
  have hlen : twoTwo.part.blocks.length = 2 := rfl
  have hs : twoPart.strata.length = 1 := rfl
  obtain rfl : t = 0 := by omega
  have hc : (cell twoPart twoTwo.part ρ 0).card = 1 := by
    obtain rfl | rfl : ρ = 0 ∨ ρ = 1 := by omega
    · decide
    · decide
  rw [hc]
  norm_num [twoTwo]

/-- At boundary values that commit nothing at the input, the toy's one replay block is wrong at its boundary. -/
theorem toy_wrongAtBoundaryAt {val : ℕ → Option Bool} (hv : val 0 = none) :
    0 ∈ wrongAtBoundaryAt toyStmt toyPart toyPart val := by
  classical
  unfold wrongAtBoundaryAt
  exact Finset.mem_filter.2 ⟨Finset.mem_range.2 (by decide), fun ι => ⟨0, by decide, ⟨0, toy_wrongInAt hv ι⟩⟩⟩

/-- **`CompiledTwo.MissTwo` fails** at `sksLTwo`'s silent prover, at both rates `1`: its sample misses the committed
incorrect set always, while its replay block is wrong at the committed boundary trace, so the bound is `0`. -/
theorem compiledTwoMiss_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ¬ (sksLTwo Coin).MissTwo H 1 1 := by
  intro h
  have h1 := h [] toyStmt toyPart toyTwo (sksLTwo_scope []) rfl rfl rfl toyTwo_firstRate toyTwo_secondRate Unit
    silentP ()
  have hW : 0 < (wrongAtBoundaryAt toyStmt toyPart toyTwo.part ((sksLTwo Coin).regVal H silentP [] ())).card :=
    Finset.card_pos.2 ⟨0, toy_wrongAtBoundaryAt (val := (sksLTwo Coin).regVal H silentP [] ())
      (sksL_regVal H silentP [] () 0)⟩
  have hp : Pr (fun r : Fin ((sksLTwo Coin).V.k []) → Coin =>
      ∀ u ∈ (sksLTwo Coin).checked [] r, u ∉ (sksLTwo Coin).FregTwo H silentP [] () r) = 1 :=
    Eq.trans (Pr_congr fun _ => iff_true_intro fun u hu => absurd (show u ∈ ([] : List ℕ) from hu) (by simp))
      Claims.ZKMal.Pr_true
  have h2 : (1 : ℝ) ≤ (1 - 1 * 1) ^
      (wrongAtBoundaryAt toyStmt toyPart toyTwo.part ((sksLTwo Coin).regVal H silentP [] ())).card :=
    hp.symm.trans_le h1
  rw [show (1 : ℝ) - 1 * 1 = 0 by norm_num, zero_pow hW.ne'] at h2
  norm_num at h2

end Claims.Witness.SamplingGeneric

/-! ## Stated over the spec -/

namespace Claims.SamplingGeneric

open Claims.Witness (silentP)
open Claims.Witness.SamplingGeneric

/-- **`CompiledTwo.MissTwo` fails** at some compiled two-stage verifier and rates. -/
theorem CompiledTwo.MissTwo_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ∃ (A : Arg Coin) (L : CompiledTwo Coin A) (r₁ r₂ : ℝ), ¬ L.MissTwo H r₁ r₂ :=
  ⟨_, _, 1, 1, compiledTwoMiss_fails H⟩

/-- **`CompilerMissTwo`'s conclusion fails** at `sksLTwo`, at both rates `1`, which its design meets: it samples
nothing, while its replay block is wrong at the committed boundary trace. -/
theorem compilerMissTwo_nontrivial : ¬ (sksLTwo Bool).MissTwo FlockDeploy.sha512 1 1 := compiledTwoMiss_fails _

/-- **`CompilerMissTwo`'s premises hold together at two sessions**: `twoLTwo` meets the two-stage contract, the empty
statement is in scope, and its design `twoTwo` catches each replay block and samples each cell at rate `1`. -/
theorem compilerMissTwo_honest :
    (twoLTwo Bool).Sessions FlockDeploy.sha512 ∧ (twoLTwo Bool).Scope [] ∧ twoTwo.FirstRate 1 ∧
      twoTwo.SecondRate twoPart 1 :=
  ⟨twoLTwo_sessions _, twoLTwo_scope [], twoTwo_firstRate, twoTwo_secondRate⟩

/-- **`CompilerHeadlineTwo`'s premises hold together at two sessions**: `twoLTwo` meets the two-stage contract and the
committed form at two stages, the empty statement is in scope, and each run samples both replay blocks and both proof
blocks and is accepted. -/
theorem compilerHeadlineTwo_honest :
    (twoLTwo Bool).Sessions FlockDeploy.sha512 ∧ (twoLTwo Bool).KSRegTwo FlockDeploy.sha512 ∧
      (twoLTwo Bool).Scope [] ∧
      ∀ r, (twoLTwo Bool).replaySample [] r = [0, 1] ∧ (twoLTwo Bool).checked [] r = [0, 1] ∧
        (run FlockDeploy.sha512 silentP (twoLTwo Bool).V [] () r).out = .acc := by
  obtain ⟨hA, hB, hS, hsc, hr⟩ := compilerKSTwo_honest
  exact ⟨hS, compilerKSRegTwo Bool (twoA Bool) (twoLTwo Bool) FlockDeploy.sha512 hA hB hS, hsc, hr⟩

end Claims.SamplingGeneric
