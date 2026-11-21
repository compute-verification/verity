module

public import Proofs.Flock.Soundness.Discharge.Composed.SampledJ
public import Proofs.Flock.Soundness.Discharge.ZkLink.Public
public import Proofs.Flock.Soundness.Discharge.ZkReg.SoundHJCustody
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Proofs.Flock.Soundness.Discharge.Composed.PublicJ
import Proofs.Flock.Soundness.Discharge.Tables.Headline
import Proofs.Flock.Soundness.Discharge.ZkReg.Checked
import Proofs.Flock.Soundness.Discharge.ZkReg.CheckedHJ
import Proofs.Flock.Soundness.Game

/-!
# `zk_session_soundHJR_custody`, "wrong" measured against the public file

`zk_session_soundHJR_custody_reg` is `zk_session_soundHJR_custody` with its first conjunct's "wrong" `incorrectRegZ` at the
prover's registration in place of `Partition.wrong`, at the same bound, and its second conjunct (the registered check at
every accepted outcome) unchanged. A unit counts as wrong unless it is correct in the committed transcript and one opening
per commit string makes the key program's evaluation on the committed inputs, on every committed wire the unit reads or
computes, the bit of an opening of that wire's registered commit string: the public file's `b ‖ c` at rows, which the
second conjunct ties, at each read the verifier owns, to the root `--registered F` holds.

* `rsZHJS_incorrectRegZ`: the registered commit strings are the public file's (`keyLayoutH`'s population is `fun _ => pub`),
  so `incorrectRegZ` is one set at every registration.
* `zk_session_soundHJR_custody_sampled_reg`: the sampled proof blocks' form of the first conjunct,
  `Pr[verity-verify-proof accepts ∧ some drawn unit in incorrectRegZ] ≤ ε_ks + ε_bind`, without A3, and the same second conjunct.
-/

@[expose] public section

namespace FlockSoundness.Discharge.ZkReg

open Game Spot.Sampling Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind ZkHidden HiddenExec
  Composed Placed StrictCR DecodeZL
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

variable {n : ℕ} {Reg : Type}

/-- **The registered commit strings are the public file's**, under hidden outputs with or without shared rows: the
rows' value binding (`rsZHJS`) reads the registration only through the public file `pub`, so its `incorrectRegZ` is one set
at every registration. -/
theorem rsZHJS_incorrectRegZ (dj : Finset (Fin n) → Lean.Json) {I : Inputs} {mPts : ℕ} {S₀ : Finset (Fin n)}
    {d₀ : Lean.Json} (y₀ : SampleSetupZHJ (inputsJ I 0) mPts S₀ d₀) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hscope : Layout.scopeOkH c = true)
    (hU : Built c.unit) {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
    (R R' : Reg) (Xv : Fin (Layout.keyN c hU n) → Bool) :
    incorrectRegZ execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
        ((rsZHJS dj y₀ ht hc hscope hU hpub).bindingZ (rsZHJS_computes dj y₀ ht hc hscope hU hpub)) R Xv =
      incorrectRegZ execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
        ((rsZHJS dj y₀ ht hc hscope hU hpub).bindingZ (rsZHJS_computes dj y₀ ht hc hscope hU hpub)) R' Xv := by
  apply incorrectRegZ_congr
  intro p
  rfl

end FlockSoundness.Discharge.ZkReg
