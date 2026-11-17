module

public import Proofs.Flock.Soundness.Discharge.ZkHidden.Defs
public import Proofs.Flock.Soundness.Discharge.ZkLink.SampleOk
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
import Proofs.Flock.Soundness.Discharge.Exec.SamplingDesignSample
import Proofs.Flock.Soundness.Discharge.Hidden.Theorems
import Proofs.Flock.Soundness.Discharge.HiddenExec.Setup
import Proofs.Flock.Soundness.Discharge.ZkHidden.ParseStmt

/-! The challenger's sample at every sample of the live sampling design, at hidden-output `--zk`. -/

@[expose] public section

namespace FlockSoundness.Discharge.ZkHidden

open Flock Exec ZkExec ZkLink HiddenExec Finset FlockSoundness.Refine
open Lean (Json)

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {partition : Option Json} {program : Option ByteArray} {st : Flock.Stmt}

theorem setupHidden_mPts_sampled {d : Json}
    (h : Stmt.setupHidden tags cf pf coins tables (some d) partition program = .ok st) :
    ∃ ud, Draw.UnitSample.ofJson d = .ok ud ∧ st.mPts = mPtsOf st.c.g ud.proved.size := by
  have s := Hidden.setupHidden_spec h
  obtain ⟨pop, -, hd⟩ := setupHidden_sampled h
  obtain ⟨ud, hud, hn⟩ := sampled_n (c := { st.c with outPorts := #[] }) hd
  refine ⟨ud, hud, ?_⟩
  rw [s.mPts, s.nbl, s.blocksReal, s.n, hn]
  rfl

theorem setupHidden_mPts_of_proved {c : Circuit} {d : Json} {N₀ : ℕ} (hc : HmOut.parse tags cf tables = .ok c)
    (hD : ∀ D, Draw.UnitSample.ofJson d = .ok D → D.proved.size = N₀)
    (h : Stmt.setupHidden tags cf pf coins tables (some d) partition program = .ok st) : st.mPts = mPtsOf c.g N₀ := by
  obtain ⟨ud, hud, hm⟩ := setupHidden_mPts_sampled h
  rw [hm, setupHidden_c_of_parse hc h, hD ud hud]

theorem setupHidden_mPts_of_provedG {g : ℕ} {d : Json} {N₀ : ℕ} (hg : st.c.g = g)
    (hD : ∀ D, Draw.UnitSample.ofJson d = .ok D → D.proved.size = N₀)
    (h : Stmt.setupHidden tags cf pf coins tables (some d) partition program = .ok st) : st.mPts = mPtsOf g N₀ := by
  obtain ⟨ud, hud, hm⟩ := setupHidden_mPts_sampled h
  rw [hm, hg, hD ud hud]

theorem setupHidden_mPts_of_provedSt {c : Circuit} {d : Json} {N₀ : ℕ} (ht : tags.typed = false)
    (hc : HmOut.parseStmt tags cf tables = .ok c) (hD : ∀ D, Draw.UnitSample.ofJson d = .ok D → D.proved.size = N₀)
    (h : Stmt.setupHidden tags cf pf coins tables (some d) partition program = .ok st) : st.mPts = mPtsOf c.g N₀ :=
  setupHidden_mPts_of_provedG (stmt_g_of_setupHidden ht hc h).symm hD h

variable {n : ℕ}

theorem sampleOkZKH_of_execG (I : Inputs) {g : ℕ}
    (hg : ∀ {d : Json} {st : Flock.Stmt}, Stmt.setupHidden (zkTags I) I.circuitFile I.publicFile (zkCoins I) I.tables
      (some d) I.partition I.program = .ok st → st.c.g = g)
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Audit.SamplingDesign.stratum σs s).card)
    (St : Fin m → Draw.Stratum) (hS : Audit.SamplingDesign.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (dj : Finset (Fin n) → Lean.Json) (hdj : SampleFileZK dj (.stratified Kd (Array.ofFn St)) pop) (ω : Ω) :
    SampleOkZKH I dj (mPtsOf g (∑ s, ks s))
      ((Audit.SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) := by
  refine ⟨fun D hD u hu => ?_, fun st hst => ?_⟩
  · rw [hdj _ D hD, sampleOf_proved, sampleOf_mem]
    exact ⟨u, hu, rfl⟩
  · rw [setupHidden_mPts_of_provedG (hg hst) (fun D hD => by rw [hdj _ D hD, sampleOf_proved, sampleOf_size]) hst]
    rcases execOS_sample_card_or σs ks St hS Kd pop src ω with hcard | huniv
    · rw [hcard]
    · obtain ⟨ud, hud, -⟩ := setupHidden_mPts_sampled hst
      rw [hdj _ ud hud, huniv] at hud
      rw [huniv, card_univ, Fintype.card_fin, sum_ks_of_univ σs ks hks St hS Kd pop hud]

theorem sampleOkZKH_of_exec (I : Inputs) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Audit.SamplingDesign.stratum σs s).card)
    (St : Fin m → Draw.Stratum) (hS : Audit.SamplingDesign.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (dj : Finset (Fin n) → Lean.Json) (hdj : SampleFileZK dj (.stratified Kd (Array.ofFn St)) pop) (ω : Ω) :
    SampleOkZKH I dj (mPtsOf c.g (∑ s, ks s))
      ((Audit.SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) :=
  sampleOkZKH_of_execG I (fun h => congrArg Flock.Circuit.g (setupHidden_c_of_parse hc h)) σs ks hks
    St hS Kd pop Lb src dj hdj ω

theorem sampleOkZKH_of_execSt (I : Inputs) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmOut.parseStmt (zkTags I) I.circuitFile I.tables = .ok c)
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Audit.SamplingDesign.stratum σs s).card)
    (St : Fin m → Draw.Stratum) (hS : Audit.SamplingDesign.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (dj : Finset (Fin n) → Lean.Json) (hdj : SampleFileZK dj (.stratified Kd (Array.ofFn St)) pop) (ω : Ω) :
    SampleOkZKH I dj (mPtsOf c.g (∑ s, ks s))
      ((Audit.SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) :=
  sampleOkZKH_of_execG I (fun h => (stmt_g_of_setupHidden ((zkTags_typed I).trans ht) hc h).symm) σs ks hks
    St hS Kd pop Lb src dj hdj ω

end FlockSoundness.Discharge.ZkHidden
