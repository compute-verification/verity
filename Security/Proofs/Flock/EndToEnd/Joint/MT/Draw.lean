module

public import Proofs.Flock.Soundness.Discharge.ZkHidden.DefsJ
public import Proofs.Flock.Soundness.Discharge.ZkLink.DrawOk
public import Proofs.Flock.Soundness.Discharge.ZkReg.Guarded
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
import Proofs.Flock.Soundness.Discharge.Exec.LawDraw
import Proofs.Flock.Soundness.Discharge.ZkHidden.CustodyJ

/-!
# The challenger's live-coin part's stratified sample at `J` tables

`drawOkZKHJ_of_accepts_stratified`: at a sample of the live stratified sampling design (`Law.execOS` at `stratified Kd St`) that
flock-verify `--zk` accepts, at any number of tables, `DrawOkZKHJ` holds at `mPtsOf c.g ((∑ s, ks s) / nTab I)` link
points, from the record's custody alone (`RecordsLiveZKJ`, and `DrawFileZK` at the stratified sampling design). The sample file
decodes only to the verifier's sample of the sampled proof blocks, `∑ s, ks s` of them (`execOS_draw_card_or`; a run that reads
past its bytes samples every unit, and then `sum_ks_of_univ`), and an accepted record's units split into the tables evenly
(`setupTables_units`), so `drawOkZKHJ_of_size` applies. It is `drawOkZKHJ_of_accepts`'s statement, and
`drawOkZKHJ_of_custodyJS` gives the same from `RecordCustodyZKJS` (`ZkReg/Guarded.lean`), where `verify --zk` takes a
stratified sample at several tables.
-/

@[expose] public section

namespace FlockSoundness.Discharge.ZkHidden

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Placed DecodeZL Composed
  Finset
open FlockSoundness.Discharge.Tables (LiveOutJ)

variable {n mPts : ℕ} {Reg : Type}

/-- **`DrawOkZKHJ` at an accepted outcome of the live stratified sampling design, at any count.** -/
theorem drawOkZKHJ_of_accepts_stratified (I : Inputs) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (dj : Finset (Fin n) → Lean.Json) (hdj : DrawFileZK dj (.stratified Kd (Array.ofFn St)) pop)
    {Jf : Finset (Fin n) → Reg → ℕ} {wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (Jf S R) mPts → Exec.Transcript}
    (hRec : RecordsLiveZKJ I dj Jf wr) (ω : Ω) (R : Reg)
    (o : LiveOutJ (Jf ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) R) mPts)
    (hacc : AcceptsZK I (wr ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) R o)) :
    DrawOkZKHJ I dj (mPtsOf c.g ((∑ s, ks s) / nTab I))
      ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) := by
  have hsz : ∀ D, Draw.UnitDraw.ofJson (dj ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)) =
      .ok D → D.units.size = ∑ s, ks s := by
    intro D hD
    have hD' := hD
    rw [hdj _ D hD, drawOf_size]
    rcases execOS_draw_card_or σs ks St hS Kd pop src ω with hcard | huniv
    · exact hcard
    · rw [hdj _ D hD, huniv] at hD'
      rw [huniv, card_univ, Fintype.card_fin, sum_ks_of_univ σs ks hks St hS Kd pop hD']
  refine drawOkZKHJ_of_size I ht hc _ (∑ s, ks s) pop ?_ dj hdj _ hsz
  by_cases h1 : I.count ≤ 1
  · rw [show nTab I = 1 by simp [nTab, h1]]
    exact Nat.mod_one _
  · obtain ⟨hd, -⟩ := hRec _ R o hacc
    obtain ⟨spec, sts, hs, -⟩ := hacc
    rw [Exec.setupOf, hd] at hs
    obtain ⟨u, hu, hdu⟩ := setupTables_units (show 1 < (zkInputs I).count from not_le.1 h1) hs
    have hn : nTab I = I.count := by simp [nTab, h1]
    rw [hn, ← hsz u hu]
    exact hdu

end FlockSoundness.Discharge.ZkHidden
