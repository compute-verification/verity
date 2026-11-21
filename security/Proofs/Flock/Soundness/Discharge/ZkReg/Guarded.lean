module

public import Proofs.Flock.Soundness.Discharge.ZkHidden.DefsJ
public import Proofs.Flock.Soundness.Assumptions.Zk
public import Proofs.Flock.Soundness.Discharge.ZkExec.LiveStatements
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
import Proofs.Flock.Soundness.Discharge.Composed.CustodyJ
import Proofs.Flock.Soundness.Discharge.Exec.SamplingDesignSample
import Proofs.Flock.Soundness.Discharge.Tables.Accepts
import Proofs.Flock.Soundness.Discharge.ZkExec.RunsLive
import Proofs.Flock.Soundness.Discharge.ZkHidden.CustodyJ
import Proofs.Flock.Soundness.Discharge.ZkHidden.SoundSampleOk

/-!
# The live `--zk` runs with the sample at accepted outcomes

`ZkExec.runsZC_live` takes its sample hypothesis (`SampleOkZK`) at every sample of the sampling design, and reads it only at
the sample of an outcome verity-verify-proof `--zk` accepts. `runsZC_liveA` takes it there alone: it is the old theorem at
the sampling design with the one coin that samples that sample.

The samples come from the record's custody (`SampleFileZK`):

* `SampleFileZK.proves`, `SampleFileZK.card`: a sample file custody fixes proves its sample's units, as many as the sample has;
* `sampleOkZK_execOS`: `SampleOkZK` at a sample of `SamplingDesign.execOS` whose run succeeds, at link points
  `mPts = mPtsOf c.g (∑ s, k s)` (`Exec.setupH_mPts_execOS`), from custody under any sampling design;
* `sampleOkZKHJ_one`: at one table, `SampleOkZKHJ` is `SampleOkZKH`.
-/

@[expose] public section

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL ZkSession

local notation "𝕂" => FlockLevel3.GF256

/-- **`RunsZCLive` with the sample hypothesis at accepted outcomes**: `RunsZCLive` with `SampleOkZK` at the sample `L.draw ω`
only where verity-verify-proof `--zk` accepts what `wr` writes at it. -/
def RunsZCLiveA : Prop :=
  ∀ {n mPts : ℕ} {Reg : Type} (I : Inputs) (_ht : I.tags.typed = false) (dj : Finset (Fin n) → Lean.Json)
    {S₀ : Finset (Fin n)} (y₀ : ZkLink.SampleSetupZK I mPts S₀ (dj S₀))
    (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript) (L : Spot.Sampling.SamplingDesign n),
    ZkLink.TablesAtZKLive (n := n) (mPts := mPts) (Reg := Reg) → TagsOk I → ZkLink.RecordsLiveZK I dj wr →
    (∀ ω R (o : LiveOut mPts), AcceptsZK I (wr (L.draw ω) R o) → ZkLink.SampleOkZK I dj mPts (L.draw ω)) →
    RunsOnZC (K := 𝕂) execArith hs512 (ZkLink.tablesAtZK (Reg := Reg) dj y₀) I wr
      (opnAtZ (ZkLink.tablesAtZK dj y₀) I dj) (decAtZ (ZkLink.tablesAtZK dj y₀)) (fun _ _ _ _ => [])
      (fun _ _ b _ => capOfRoot b) L

/-- **`RunsZC` at the live `--zk` tables, with the sample at accepted outcomes**: `runsZC_live` at the one-coin distribution of
the accepted outcome's sample. -/
theorem runsZC_liveA : RunsZCLiveA := by
  intro n mPts Reg I ht dj S₀ y₀ wr L hTab hT hRec hDraw ω R o hacc
  exact runsZC_live I ht dj y₀ wr ⟨Unit, fun _ => L.draw ω⟩ hTab hT hRec (fun _ => hDraw ω R o hacc) () R o hacc

end FlockSoundness.Discharge.ZkExec

namespace FlockSoundness.Discharge.ZkLink

open Flock Exec ZkExec Finset FlockSoundness.Refine

variable {n : ℕ}

/-- A sample file custody fixes proves its sample's units. -/
theorem SampleFileZK.proves {dj : Finset (Fin n) → Lean.Json} {samplingDesign : Draw.SamplingDesign} {pop : ℕ} (h : SampleFileZK dj samplingDesign pop)
    (S : Finset (Fin n)) : ∀ D, Draw.UnitSample.ofJson (dj S) = .ok D → ∀ u ∈ S, u.val ∈ D.proved := by
  intro D hD u hu
  rw [h S D hD, sampleOf_proved, sampleOf_mem]
  exact ⟨u, hu, rfl⟩

/-- A sample file custody fixes proves as many units as its sample has. -/
theorem SampleFileZK.card {dj : Finset (Fin n) → Lean.Json} {samplingDesign : Draw.SamplingDesign} {pop : ℕ} (h : SampleFileZK dj samplingDesign pop) :
    ∀ B D, Draw.UnitSample.ofJson (dj B) = .ok D → D.proved.size = B.card := by
  intro B D hD
  rw [h B D hD, sampleOf_proved, sampleOf_size]

/-- **`SampleOkZK` at a sample of the live sampling design whose run succeeds**, at `mPtsOf c.g (∑ s, k s)` link points. -/
theorem sampleOkZK_execOS (I : Inputs) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) {mPts m : ℕ} {σ : Fin n → Fin m} {k : Fin m → ℕ}
    {St : Fin m → Draw.Stratum} (hS : Spot.Sampling.SamplingDesign.ExecStrata σ k St) {samplingDesign : Draw.SamplingDesign} {pop' : ℕ}
    {dj : Finset (Fin n) → Lean.Json} (h : SampleFileZK dj samplingDesign pop') {K pop : ℕ} {Ω : Type} [Fintype Ω] [Nonempty Ω]
    {L : Fin m → ℕ} {t : Ω → (i : Fin m) → Fin (L i) → Fin 256} {ω : Ω} {d : Draw.UnitSample}
    (hrun : Spot.Sampling.SamplingDesign.runOS (t ω) (.stratified K (Array.ofFn St)) pop = some d)
    (hm : mPts = mPtsOf c.g (∑ s, k s)) :
    SampleOkZK I dj mPts ((Spot.Sampling.SamplingDesign.execOS (n := n) t (.stratified K (Array.ofFn St)) pop).draw ω) := by
  have hc' : (if (zkTags I).typed then HmRow.parseTyped (zkTags I) I.circuitFile I.tables
      else HmRow.parse (zkTags I) I.circuitFile I.tables) = .ok c := by
    rw [zkTags_typed, ht]; exact hc
  refine ⟨h.proves _, fun st hst => ?_⟩
  rw [hm]
  exact setupH_mPts_execOS hc' σ k St hS K pop t ω hrun dj h.card hst

end FlockSoundness.Discharge.ZkLink

namespace FlockSoundness.Discharge.ZkHidden

open Game Spot.Sampling Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Placed DecodeZL Composed
open FlockSoundness.Discharge.Tables (LiveOutJ)

local notation "𝕂" => FlockLevel3.GF256

variable {n mPts : ℕ} {Reg : Type}

/-- **At one table, `SampleOkZKHJ` is `SampleOkZKH`**: the table's sample is the session's, its units are the sampled ones, and
its tags are the `--zk` tags. -/
theorem sampleOkZKHJ_one {I : Inputs} (h1 : I.count ≤ 1) {dj : Finset (Fin n) → Lean.Json} {S : Finset (Fin n)}
    (h : SampleOkZKH I dj mPts S) : SampleOkZKHJ I dj mPts S := by
  intro j
  have hd : drawJ I (dj S) j = dj S := by simp [drawJ, h1]
  have ht : tagsAt (zkTags I) I.count j = zkTags I := by simp [tagsAt, h1]
  rw [hd, ht]
  exact ⟨fun D hD u hu => h.1 D hD u (Finset.mem_filter.1 hu).1, h.2⟩

end FlockSoundness.Discharge.ZkHidden
