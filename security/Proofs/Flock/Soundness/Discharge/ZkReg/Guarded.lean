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
# The live `--zk` runs with the sample at accepted outcomes, and the challenger's live-coin part's samples at V*'s sampling design

`ZkExec.runsZC_live` and `ZkHidden.runsZCJH_live` take their sample hypothesis (`SampleOkZK`, `SampleOkZKHJ`) at every sample of
the sampling design, and read it only at the sample of an outcome verity-verify-proof `--zk` accepts. `runsZC_liveA` and `runsZCJH_liveA` take
it there alone: each is the old theorem at the sampling design with the one coin that samples that sample.

The samples come from the record's custody (`SampleFileZK`):

* `SampleFileZK.proves`, `SampleFileZK.card`: a sample file custody fixes proves its sample's units, as many as the sample has;
* `sampleOkZK_execOS`, `sampleOkZKH_execOS`: `SampleOkZK` (`SampleOkZKH`) at a sample of `SamplingDesign.execOS` whose run succeeds, at link
  points `mPts = mPtsOf c.g (∑ s, k s)` (`Exec.setupH_mPts_execOS`, `ZkHidden.setupHidden_mPts_of_proved`), from custody
  under any sampling design. `ZkLink.sampleOkZK_of_exec` and `ZkHidden.sampleOkZKH_of_exec` give the same at every sample, the failed
  run's (every unit) included, from custody under the live sampling design's own stratified sampling design;
* `sampleOkZKHJ_one`, `sampleOkZKHJ_of_exec_one`: at one table, `SampleOkZKHJ` is `SampleOkZKH`, so `runsZCJH_live`'s `hDraw`
  holds at every sample of the live sampling design;
* `SampleFileZK.size_execOS`, `Composed.sampleOkZKJ_of_exec`, `sampleOkZKHJ_of_exec`: at every sample of the live stratified
  sampling design, custody's sample file has `Σ_s k_s` units, so where the session's `J` tables divide it, `SampleOkZKJ` (`SampleOkZKHJ`)
  holds at `(Σ_s k_s) / J` units a table: verity-verify-proof splits one stratified sample over several tables as it splits a
  subset sample (`Draw.tablesWhy`);
* `sampleOkZKHJ_of_accepts`, at any count: several tables split the sample's units evenly (`setupTables_units`), so at an
  accepted outcome the tables divide `Σ_s k_s` and `sampleOkZKHJ_of_exec` holds. `runsZCJH_live_execOS` is
  `runsZCJH_liveA` at the live sampling design with that sample hypothesis discharged. `sampleOkZKHJ_of_custodyJS` is the same from
  `Assumptions.Zk.RecordCustodyZKJS`, the custody `JointStratifiedMT` takes;
* `sampleOkZKHJ_of_accepts_subset`, at a subset sampling design and any count: several tables split the sample's units evenly
  (`setupTables_units`), so `sampleOkZKHJ_of_file` holds at `kd / nTab I` units a table. `runsZCJH_live_subset` is
  `runsZCJH_liveA` at any sampling design with that sample hypothesis discharged, V*'s full sample among them.
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
    (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript) (L : Audit.SamplingDesign n),
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
    {St : Fin m → Draw.Stratum} (hS : Audit.SamplingDesign.ExecStrata σ k St) {samplingDesign : Draw.SamplingDesign} {pop' : ℕ}
    {dj : Finset (Fin n) → Lean.Json} (h : SampleFileZK dj samplingDesign pop') {K pop : ℕ} {Ω : Type} [Fintype Ω] [Nonempty Ω]
    {L : Fin m → ℕ} {t : Ω → (i : Fin m) → Fin (L i) → Fin 256} {ω : Ω} {d : Draw.UnitSample}
    (hrun : Audit.SamplingDesign.runOS (t ω) (.stratified K (Array.ofFn St)) pop = some d)
    (hm : mPts = mPtsOf c.g (∑ s, k s)) :
    SampleOkZK I dj mPts ((Audit.SamplingDesign.execOS (n := n) t (.stratified K (Array.ofFn St)) pop).draw ω) := by
  have hc' : (if (zkTags I).typed then HmRow.parseTyped (zkTags I) I.circuitFile I.tables
      else HmRow.parse (zkTags I) I.circuitFile I.tables) = .ok c := by
    rw [zkTags_typed, ht]; exact hc
  refine ⟨h.proves _, fun st hst => ?_⟩
  rw [hm]
  exact setupH_mPts_execOS hc' σ k St hS K pop t ω hrun dj h.card hst

/-- **A sample file custody fixes under the live stratified sampling design has `Σ_s k_s` units**, at every sample of `SamplingDesign.execOS`: the
run that succeeds samples that many (`execOS_sample_card_or`), and the run that reads past the source's bytes samples every
unit, whose file decodes only when every stratum samples all its units (`sum_ks_of_univ`). -/
theorem SampleFileZK.size_execOS {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ)
    (hks : ∀ s, ks s ≤ (Audit.SamplingDesign.stratum σs s).card) (St : Fin m → Draw.Stratum) (hS : Audit.SamplingDesign.ExecStrata σs ks St)
    (Kd pop : ℕ) {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    {dj : Finset (Fin n) → Lean.Json} (hdj : SampleFileZK dj (.stratified Kd (Array.ofFn St)) pop) (ω : Ω) :
    ∀ D, Draw.UnitSample.ofJson (dj ((Audit.SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω))
      = .ok D → D.units.size = ∑ s, ks s := by
  intro D hD
  have hDe := hdj _ D hD
  rw [hDe, sampleOf_size]
  rcases execOS_sample_card_or σs ks St hS Kd pop src ω with hcard | huniv
  · exact hcard
  · rw [hDe, huniv] at hD
    rw [huniv, card_univ, Fintype.card_fin, sum_ks_of_univ σs ks hks St hS Kd pop hD]

end FlockSoundness.Discharge.ZkLink

namespace FlockSoundness.Discharge.ZkHidden

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Placed DecodeZL Composed
open FlockSoundness.Discharge.Tables (LiveOutJ)

local notation "𝕂" => FlockLevel3.GF256

variable {n mPts : ℕ} {Reg : Type}

/-- **`SampleOkZKH` at a sample of the live sampling design whose run succeeds**, at `mPtsOf c.g (∑ s, k s)` link points. -/
theorem sampleOkZKH_execOS (I : Inputs) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) {m : ℕ} {σ : Fin n → Fin m} {k : Fin m → ℕ}
    {St : Fin m → Draw.Stratum} (hS : SamplingDesign.ExecStrata σ k St) {samplingDesign : Draw.SamplingDesign} {pop' : ℕ}
    {dj : Finset (Fin n) → Lean.Json} (h : SampleFileZK dj samplingDesign pop') {K pop : ℕ} {Ω : Type} [Fintype Ω] [Nonempty Ω]
    {L : Fin m → ℕ} {t : Ω → (i : Fin m) → Fin (L i) → Fin 256} {ω : Ω} {d : Draw.UnitSample}
    (hrun : SamplingDesign.runOS (t ω) (.stratified K (Array.ofFn St)) pop = some d)
    (hm : mPts = mPtsOf c.g (∑ s, k s)) :
    SampleOkZKH I dj mPts ((SamplingDesign.execOS (n := n) t (.stratified K (Array.ofFn St)) pop).draw ω) := by
  refine ⟨h.proves _, fun st hst => ?_⟩
  rw [setupHidden_mPts_of_proved hc (h.card ((SamplingDesign.execOS (n := n) t (.stratified K (Array.ofFn St)) pop).draw ω)) hst,
    execOS_sample_card σ k St hS K pop t ω hrun, hm]

/-- **At one table, `SampleOkZKHJ` is `SampleOkZKH`**: the table's sample is the session's, its units are the sampled ones, and
its tags are the `--zk` tags. -/
theorem sampleOkZKHJ_one {I : Inputs} (h1 : I.count ≤ 1) {dj : Finset (Fin n) → Lean.Json} {S : Finset (Fin n)}
    (h : SampleOkZKH I dj mPts S) : SampleOkZKHJ I dj mPts S := by
  intro j
  have hd : drawJ I (dj S) j = dj S := by simp [drawJ, h1]
  have ht : tagsAt (zkTags I) I.count j = zkTags I := by simp [tagsAt, h1]
  rw [hd, ht]
  exact ⟨fun D hD u hu => h.1 D hD u (Finset.mem_filter.1 hu).1, h.2⟩

/-- **At one table, `runsZCJH_live`'s samples hold at every sample of the live sampling design**: `sampleOkZKH_of_exec` at one table. -/
theorem sampleOkZKHJ_of_exec_one (I : Inputs) (h1 : I.count ≤ 1) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (SamplingDesign.stratum σs s).card)
    (St : Fin m → Draw.Stratum) (hS : SamplingDesign.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (dj : Finset (Fin n) → Lean.Json) (hdj : SampleFileZK dj (.stratified Kd (Array.ofFn St)) pop) (ω : Ω) :
    SampleOkZKHJ I dj (mPtsOf c.g (∑ s, ks s)) ((SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) :=
  sampleOkZKHJ_one h1 (sampleOkZKH_of_exec I hc σs ks hks St hS Kd pop Lb src dj hdj ω)

/-- **`RunsZC` at the live hidden-output `--zk` tables of `J` tables, with the sample at accepted outcomes**:
`runsZCJH_live` at the one-coin distribution of the accepted outcome's sample. -/
theorem runsZCJH_liveA (I : Inputs) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (ho : c.outNet = c.unitNet)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : SampleSetupZHJ (inputsJ I 0) mPts S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length mPts → Exec.Transcript)
    (L : SamplingDesign n) (hT : TagsOkZHJ I) (hRec : RecordsLiveZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr)
    (hDraw : ∀ ω R (o : LiveOutJ (planZHJ (Reg := Reg) dj y₀ (L.draw ω) R).length mPts),
      AcceptsZK I (wr (L.draw ω) R o) → SampleOkZKHJ I dj mPts (L.draw ω)) :
    RunsOnZCJ (K := 𝕂) execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) L := by
  intro ω R o hacc
  exact runsZCJH_live I ht hc ho dj y₀ wr ⟨Unit, fun _ => L.draw ω⟩ hT hRec (fun _ => hDraw ω R o hacc) () R o hacc

/-- **Several tables split the sample's units evenly**: at `count > 1`, `setupTables` accepts a sample only if it decodes and
its units are a multiple of `count` (`Tables.setupTables_parts`, `setupTables_partsH`). -/
theorem setupTables_units {tags : Flock.Tags} {cf pf : ByteArray} {coins : Flock.Coins} {count : ℕ}
    {tables : List (String × ByteArray)} {d : Lean.Json} {partition : Option Lean.Json} {program : Option ByteArray}
    {work : Option (ℕ × Array (String × ℕ × ℕ))} {stratified : Option ℕ} {closure : Option (Std.HashMap ℕ (Array ℕ))}
    {spec : Flock.SessionSpec} {sts : Array Flock.Stmt} (hc1 : 1 < count)
    (h : Flock.Stmt.setupTables tags cf pf coins count tables (some d) partition program work stratified closure =
      .ok (spec, sts)) :
    ∃ u, Draw.UnitSample.ofJson d = .ok u ∧ u.units.size % count = 0 := by
  cases hh : tags.hiddenOutputs
  · obtain ⟨-, -, _, _, hd, -, -, hdiv, -⟩ := Tables.setupTables_parts hh hc1 h
    obtain ⟨u, hu, -, -, hun⟩ := hd d rfl
    exact ⟨u, hu, by rw [hun]; exact hdiv⟩
  · obtain ⟨-, -, _, _, hd, -, -, hdiv, -⟩ := setupTables_partsH hh hc1 h
    obtain ⟨u, hu, -, -, hun⟩ := hd d rfl
    exact ⟨u, hu, by rw [hun]; exact hdiv⟩

/-- **The challenger's live-coin part's samples at `J` tables under the live stratified sampling design, from custody**: when the record's sample file
is the verifier's sample under the live sampling design's own stratified sampling design (`SampleFileZK`) and the session's tables divide
`Σ_s k_s`, every sample of `SamplingDesign.execOS` is `SampleOkZKJ` at `mPtsOf c.g ((Σ_s k_s) / nTab I)` link points, the failed run's
(every unit) included: custody's file has `Σ_s k_s` units (`SampleFileZK.size_execOS`), which `partSample` splits into the
tables' equal parts (`sampleOkZKJ_of_size`). -/
theorem _root_.FlockSoundness.Discharge.Composed.sampleOkZKJ_of_exec (I : Inputs) (ht : I.tags.typed = false)
    {c : Flock.Circuit} (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c)
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (SamplingDesign.stratum σs s).card)
    (St : Fin m → Draw.Stratum) (hS : SamplingDesign.ExecStrata σs ks St) (Kd pop : ℕ) (hdiv : (∑ s, ks s) % nTab I = 0)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (dj : Finset (Fin n) → Lean.Json) (hdj : SampleFileZK dj (.stratified Kd (Array.ofFn St)) pop) (ω : Ω) :
    SampleOkZKJ I dj (mPtsOf c.g ((∑ s, ks s) / nTab I))
      ((SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) :=
  sampleOkZKJ_of_size I ht hc _ _ pop hdiv dj hdj _ (SampleFileZK.size_execOS σs ks hks St hS Kd pop Lb src hdj ω)

/-- **The challenger's live-coin part's samples at `J` tables under the live stratified sampling design, from custody, at hidden outputs**:
`sampleOkZKJ_of_exec` at setupHidden (`sampleOkZKHJ_of_size`). -/
theorem sampleOkZKHJ_of_exec (I : Inputs) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (SamplingDesign.stratum σs s).card)
    (St : Fin m → Draw.Stratum) (hS : SamplingDesign.ExecStrata σs ks St) (Kd pop : ℕ) (hdiv : (∑ s, ks s) % nTab I = 0)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (dj : Finset (Fin n) → Lean.Json) (hdj : SampleFileZK dj (.stratified Kd (Array.ofFn St)) pop) (ω : Ω) :
    SampleOkZKHJ I dj (mPtsOf c.g ((∑ s, ks s) / nTab I))
      ((SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) :=
  sampleOkZKHJ_of_size I ht hc _ _ pop hdiv dj hdj _ (SampleFileZK.size_execOS σs ks hks St hS Kd pop Lb src hdj ω)

/-- **`SampleOkZKHJ` at an accepted outcome of the live sampling design, at any count**: `sampleOkZKHJ_of_exec` at `(Σ_s k_s) / nTab I`
units a table. At one table `nTab I = 1`; at several, verity-verify-proof `--zk` accepts the record's sample (the challenger's live-coin part's,
`RecordsLiveZKJ`) only when its units are a multiple of the count (`setupTables_units`), and custody's sample has
`Σ_s k_s` units (`SampleFileZK.size_execOS`), so the tables divide `Σ_s k_s`. -/
theorem sampleOkZKHJ_of_accepts (I : Inputs) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (SamplingDesign.stratum σs s).card)
    (St : Fin m → Draw.Stratum) (hS : SamplingDesign.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (dj : Finset (Fin n) → Lean.Json) (hdj : SampleFileZK dj (.stratified Kd (Array.ofFn St)) pop)
    {Jf : Finset (Fin n) → Reg → ℕ} {wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (Jf S R) mPts → Exec.Transcript}
    (hRec : RecordsLiveZKJ I dj Jf wr) (ω : Ω) (R : Reg)
    (o : LiveOutJ (Jf ((SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) R) mPts)
    (hacc : AcceptsZK I (wr ((SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) R o)) :
    SampleOkZKHJ I dj (mPtsOf c.g ((∑ s, ks s) / nTab I))
      ((SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) := by
  refine sampleOkZKHJ_of_exec I ht hc σs ks hks St hS Kd pop ?_ Lb src dj hdj ω
  by_cases h1 : I.count ≤ 1
  · rw [show nTab I = 1 from if_pos h1]
    exact Nat.mod_one _
  · obtain ⟨hd, -⟩ := hRec _ R o hacc
    obtain ⟨spec, sts, hs, -⟩ := hacc
    rw [Exec.setupOf, hd] at hs
    obtain ⟨u, hu, hdu⟩ := setupTables_units (show 1 < (zkInputs I).count from not_le.1 h1) hs
    have hn : nTab I = I.count := by simp [nTab, h1]
    rw [hn, ← SampleFileZK.size_execOS σs ks hks St hS Kd pop Lb src hdj ω u hu]
    exact hdu

/-- **`SampleOkZKHJ` from `RecordCustodyZKJS`**, the custody `JointStratifiedMT` and `JointStratifiedPooledMT` take: at
a sample of the live stratified sampling design `stratified Kd (Array.ofFn St)` that `verify --zk --session-tables J` accepts, at any
`J`, several tables included (verity-verify-proof takes one stratified sample over the sampling block there when `J` divides its
units, and gives table `j` its `j`-th part), `SampleOkZKHJ` holds at `(Σ_s k_s) / nTab I` units a table:
`sampleOkZKHJ_of_accepts` at the custody's two parts. -/
theorem sampleOkZKHJ_of_custodyJS (I : Inputs) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (SamplingDesign.stratum σs s).card)
    (St : Fin m → Draw.Stratum) (hS : SamplingDesign.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (dj : Finset (Fin n) → Lean.Json)
    {Jf : Finset (Fin n) → Reg → ℕ} {wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (Jf S R) mPts → Exec.Transcript}
    (hCust : Assumptions.Zk.RecordCustodyZKJS I dj Jf wr Kd (Array.ofFn St) pop) (ω : Ω) (R : Reg)
    (o : LiveOutJ (Jf ((SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) R) mPts)
    (hacc : AcceptsZK I (wr ((SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) R o)) :
    SampleOkZKHJ I dj (mPtsOf c.g ((∑ s, ks s) / nTab I))
      ((SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) :=
  sampleOkZKHJ_of_accepts I ht hc σs ks hks St hS Kd pop Lb src dj hCust.2 hCust.1 ω R o hacc

/-- **`RunsZC` at the live hidden-output `--zk` tables, at any count, with the sample from custody**: `runsZCJH_liveA`
at the live sampling design, whose sample hypothesis `sampleOkZKHJ_of_accepts` discharges at `mPtsOf c.g ((Σ_s k_s) / nTab I)` link
points. -/
theorem runsZCJH_live_execOS (I : Inputs) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (ho : c.outNet = c.unitNet)
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (SamplingDesign.stratum σs s).card)
    (St : Fin m → Draw.Stratum) (hS : SamplingDesign.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (dj : Finset (Fin n) → Lean.Json) (hdj : SampleFileZK dj (.stratified Kd (Array.ofFn St)) pop)
    (hm : mPts = mPtsOf c.g ((∑ s, ks s) / nTab I)) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : SampleSetupZHJ (inputsJ I 0) mPts S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length mPts → Exec.Transcript)
    (hT : TagsOkZHJ I) (hRec : RecordsLiveZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr) :
    RunsOnZCJ (K := 𝕂) execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!)
      (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) := by
  subst hm
  exact runsZCJH_liveA I ht hc ho dj y₀ wr (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) hT hRec
    (sampleOkZKHJ_of_accepts I ht hc σs ks hks St hS Kd pop Lb src dj hdj hRec)

/-- **`SampleOkZKHJ` at an accepted outcome of a subset sampling design, at any count**: `sampleOkZKHJ_of_file` at `kd / nTab I` units a
table. At one table `nTab I = 1`; at several, verity-verify-proof `--zk` accepts the record's sample (the challenger's live-coin part's,
`RecordsLiveZKJ`) only when its units are a multiple of the count (`setupTables_units`), and custody's sample has `kd`
units (`SampleFileZK`, `ofJson_shape`), so `kd % nTab I = 0`. -/
theorem sampleOkZKHJ_of_accepts_subset (I : Inputs) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (kd pop : ℕ)
    (dj : Finset (Fin n) → Lean.Json) (hdj : SampleFileZK dj (.subset kd) pop)
    {Jf : Finset (Fin n) → Reg → ℕ} {wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (Jf S R) mPts → Exec.Transcript}
    (hRec : RecordsLiveZKJ I dj Jf wr) (S : Finset (Fin n)) (R : Reg) (o : LiveOutJ (Jf S R) mPts)
    (hacc : AcceptsZK I (wr S R o)) :
    SampleOkZKHJ I dj (mPtsOf c.g (kd / nTab I)) S := by
  refine sampleOkZKHJ_of_file I ht hc kd pop ?_ dj hdj S
  by_cases h1 : I.count ≤ 1
  · rw [show nTab I = 1 from if_pos h1]
    exact Nat.mod_one kd
  · obtain ⟨hd, -⟩ := hRec S R o hacc
    obtain ⟨spec, sts, hs, -⟩ := hacc
    rw [Exec.setupOf, hd] at hs
    obtain ⟨u, hu, hdu⟩ := setupTables_units (show 1 < (zkInputs I).count from not_le.1 h1) hs
    have hsz : u.units.size = kd := (ofJson_shape hu).2.2 kd (by rw [hdj S u hu]; rfl)
    have hn : nTab I = I.count := by simp [nTab, h1]
    rw [hn, ← hsz]
    exact hdu

/-- **`RunsZC` at the live hidden-output `--zk` tables under a subset sampling design, at any count, with the sample from custody**:
`runsZCJH_liveA` at any sampling design, whose sample hypothesis `sampleOkZKHJ_of_accepts_subset` discharges at
`mPtsOf c.g (kd / nTab I)` link points. -/
theorem runsZCJH_live_subset (I : Inputs) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (ho : c.outNet = c.unitNet) (kd pop : ℕ)
    (dj : Finset (Fin n) → Lean.Json) (hdj : SampleFileZK dj (.subset kd) pop)
    (hm : mPts = mPtsOf c.g (kd / nTab I)) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : SampleSetupZHJ (inputsJ I 0) mPts S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length mPts → Exec.Transcript)
    (L : SamplingDesign n) (hT : TagsOkZHJ I)
    (hRec : RecordsLiveZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr) :
    RunsOnZCJ (K := 𝕂) execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) L := by
  subst hm
  exact runsZCJH_liveA I ht hc ho dj y₀ wr L hT hRec
    fun ω R o hacc => sampleOkZKHJ_of_accepts_subset I ht hc kd pop dj hdj hRec (L.draw ω) R o hacc

end FlockSoundness.Discharge.ZkHidden
