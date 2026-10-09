import Proofs.Flock.Soundness.Discharge.ZkExec.RunsLive
import Proofs.Flock.Soundness.Discharge.ZkHidden.DefsJ
import Proofs.Flock.Soundness.Discharge.ZkHidden.SoundDrawOk

/-!
# The live `--zk` runs with the draw at accepted outcomes, and the coin server's draws at V*'s law

`ZkExec.runsZC_live` and `ZkHidden.runsZCJH_live` take their draw hypothesis (`DrawOkZK`, `DrawOkZKHJ`) at every draw of
the law, and read it only at the draw of an outcome flock-verify `--zk` accepts. `runsZC_liveA` and `runsZCJH_liveA` take
it there alone: each is the old theorem at the law with the one coin that draws that draw.

The draws come from the record's custody (`DrawFileZK`):

* `DrawFileZK.proves`, `DrawFileZK.card`: a draw file custody fixes proves its draw's units, as many as the draw has;
* `drawOkZK_execOS`, `drawOkZKH_execOS`: `DrawOkZK` (`DrawOkZKH`) at a draw of `Law.execOS` whose run succeeds, at link
  points `mPts = mPtsOf c.g (∑ s, k s)` (`Exec.setupH_mPts_execOS`, `ZkHidden.setupHidden_mPts_of_proved`), from custody
  under any law. `ZkLink.drawOkZK_of_exec` and `ZkHidden.drawOkZKH_of_exec` give the same at every draw, the failed
  run's (every unit) included, from custody under the live law's own stratified law;
* `drawOkZKHJ_one`, `drawOkZKHJ_of_exec_one`: at one table, `DrawOkZKHJ` is `DrawOkZKH`, so `runsZCJH_live`'s `hDraw`
  holds at every draw of the live law.
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL ZkSession

local notation "𝕂" => FlockLevel3.GF256

/-- **`RunsZCLive` with the draw hypothesis at accepted outcomes**: `RunsZCLive` with `DrawOkZK` at the draw `L.draw ω`
only where flock-verify `--zk` accepts what `wr` writes at it. -/
def RunsZCLiveA : Prop :=
  ∀ {n mPts : ℕ} {Reg : Type} (I : Inputs) (_ht : I.tags.typed = false) (dj : Finset (Fin n) → Lean.Json)
    {S₀ : Finset (Fin n)} (y₀ : ZkLink.DrawSetupZK I mPts S₀ (dj S₀))
    (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript) (L : Audit.Law n),
    ZkLink.TablesAtZKLive (n := n) (mPts := mPts) (Reg := Reg) → TagsOk I → ZkLink.RecordsLiveZK I dj wr →
    (∀ ω R (o : LiveOut mPts), AcceptsZK I (wr (L.draw ω) R o) → ZkLink.DrawOkZK I dj mPts (L.draw ω)) →
    RunsOnZC (K := 𝕂) execArith hs512 (ZkLink.tablesAtZK (Reg := Reg) dj y₀) I wr
      (opnAtZ (ZkLink.tablesAtZK dj y₀) I dj) (decAtZ (ZkLink.tablesAtZK dj y₀)) (fun _ _ _ _ => [])
      (fun _ _ b _ => capOfRoot b) L

/-- **`RunsZC` at the live `--zk` tables, with the draw at accepted outcomes**: `runsZC_live` at the one-coin law of
the accepted outcome's draw. -/
theorem runsZC_liveA : RunsZCLiveA := by
  intro n mPts Reg I ht dj S₀ y₀ wr L hTab hT hRec hDraw ω R o hacc
  exact runsZC_live I ht dj y₀ wr ⟨Unit, fun _ => L.draw ω⟩ hTab hT hRec (fun _ => hDraw ω R o hacc) () R o hacc

end FlockSoundness.Discharge.ZkExec

namespace FlockSoundness.Discharge.ZkLink

open Flock Exec ZkExec Finset FlockSoundness.Refine

variable {n : ℕ}

/-- A draw file custody fixes proves its draw's units. -/
theorem DrawFileZK.proves {dj : Finset (Fin n) → Lean.Json} {law : Draw.Law} {pop : ℕ} (h : DrawFileZK dj law pop)
    (S : Finset (Fin n)) : ∀ D, Draw.UnitDraw.ofJson (dj S) = .ok D → ∀ u ∈ S, u.val ∈ D.proved := by
  intro D hD u hu
  rw [h S D hD, drawOf_proved, drawOf_mem]
  exact ⟨u, hu, rfl⟩

/-- A draw file custody fixes proves as many units as its draw has. -/
theorem DrawFileZK.card {dj : Finset (Fin n) → Lean.Json} {law : Draw.Law} {pop : ℕ} (h : DrawFileZK dj law pop) :
    ∀ B D, Draw.UnitDraw.ofJson (dj B) = .ok D → D.proved.size = B.card := by
  intro B D hD
  rw [h B D hD, drawOf_proved, drawOf_size]

/-- **`DrawOkZK` at a draw of the live law whose run succeeds**, at `mPtsOf c.g (∑ s, k s)` link points. -/
theorem drawOkZK_execOS (I : Inputs) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) {mPts m : ℕ} {σ : Fin n → Fin m} {k : Fin m → ℕ}
    {St : Fin m → Draw.Stratum} (hS : Audit.Law.ExecStrata σ k St) {law : Draw.Law} {pop' : ℕ}
    {dj : Finset (Fin n) → Lean.Json} (h : DrawFileZK dj law pop') {K pop : ℕ} {Ω : Type} [Fintype Ω] [Nonempty Ω]
    {L : Fin m → ℕ} {t : Ω → (i : Fin m) → Fin (L i) → Fin 256} {ω : Ω} {d : Draw.UnitDraw}
    (hrun : Audit.Law.runOS (t ω) (.stratified K (Array.ofFn St)) pop = some d)
    (hm : mPts = mPtsOf c.g (∑ s, k s)) :
    DrawOkZK I dj mPts ((Audit.Law.execOS (n := n) t (.stratified K (Array.ofFn St)) pop).draw ω) := by
  have hc' : (if (zkTags I).typed then HmRow.parseTyped (zkTags I) I.circuitFile I.tables
      else HmRow.parse (zkTags I) I.circuitFile I.tables) = .ok c := by
    rw [zkTags_typed, ht]; exact hc
  refine ⟨h.proves _, fun st hst => ?_⟩
  rw [hm]
  exact setupH_mPts_execOS hc' σ k St hS K pop t ω hrun dj h.card hst

end FlockSoundness.Discharge.ZkLink

namespace FlockSoundness.Discharge.ZkHidden

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Placed DecodeZL Composed
open FlockSoundness.Discharge.Tables (LiveOutJ)

local notation "𝕂" => FlockLevel3.GF256

variable {n mPts : ℕ} {Reg : Type}

/-- **`DrawOkZKH` at a draw of the live law whose run succeeds**, at `mPtsOf c.g (∑ s, k s)` link points. -/
theorem drawOkZKH_execOS (I : Inputs) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) {m : ℕ} {σ : Fin n → Fin m} {k : Fin m → ℕ}
    {St : Fin m → Draw.Stratum} (hS : Law.ExecStrata σ k St) {law : Draw.Law} {pop' : ℕ}
    {dj : Finset (Fin n) → Lean.Json} (h : DrawFileZK dj law pop') {K pop : ℕ} {Ω : Type} [Fintype Ω] [Nonempty Ω]
    {L : Fin m → ℕ} {t : Ω → (i : Fin m) → Fin (L i) → Fin 256} {ω : Ω} {d : Draw.UnitDraw}
    (hrun : Law.runOS (t ω) (.stratified K (Array.ofFn St)) pop = some d)
    (hm : mPts = mPtsOf c.g (∑ s, k s)) :
    DrawOkZKH I dj mPts ((Law.execOS (n := n) t (.stratified K (Array.ofFn St)) pop).draw ω) := by
  refine ⟨h.proves _, fun st hst => ?_⟩
  rw [setupHidden_mPts_of_proved hc (h.card ((Law.execOS (n := n) t (.stratified K (Array.ofFn St)) pop).draw ω)) hst,
    execOS_draw_card σ k St hS K pop t ω hrun, hm]

/-- **At one table, `DrawOkZKHJ` is `DrawOkZKH`**: the table's draw is the session's, its units are the drawn ones, and
its tags are the `--zk` tags. -/
theorem drawOkZKHJ_one {I : Inputs} (h1 : I.count ≤ 1) {dj : Finset (Fin n) → Lean.Json} {S : Finset (Fin n)}
    (h : DrawOkZKH I dj mPts S) : DrawOkZKHJ I dj mPts S := by
  intro j
  have hd : drawJ I (dj S) j = dj S := by simp [drawJ, h1]
  have ht : tagsAt (zkTags I) I.count j = zkTags I := by simp [tagsAt, h1]
  rw [hd, ht]
  exact ⟨fun D hD u hu => h.1 D hD u (Finset.mem_filter.1 hu).1, h.2⟩

/-- **At one table, `runsZCJH_live`'s draws hold at every draw of the live law**: `drawOkZKH_of_exec` at one table. -/
theorem drawOkZKHJ_of_exec_one (I : Inputs) (h1 : I.count ≤ 1) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (dj : Finset (Fin n) → Lean.Json) (hdj : DrawFileZK dj (.stratified Kd (Array.ofFn St)) pop) (ω : Ω) :
    DrawOkZKHJ I dj (mPtsOf c.g (∑ s, ks s)) ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) :=
  drawOkZKHJ_one h1 (drawOkZKH_of_exec I hc σs ks hks St hS Kd pop Lb src dj hdj ω)

/-- **`RunsZC` at the live hidden-output `--zk` tables of `J` tables, with the draw at accepted outcomes**:
`runsZCJH_live` at the one-coin law of the accepted outcome's draw. -/
theorem runsZCJH_liveA (I : Inputs) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (ho : c.outNet = c.unitNet)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length mPts → Exec.Transcript)
    (L : Law n) (hT : TagsOkZHJ I) (hRec : RecordsLiveZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr)
    (hDraw : ∀ ω R (o : LiveOutJ (planZHJ (Reg := Reg) dj y₀ (L.draw ω) R).length mPts),
      AcceptsZK I (wr (L.draw ω) R o) → DrawOkZKHJ I dj mPts (L.draw ω)) :
    RunsOnZCJ (K := 𝕂) execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) L := by
  intro ω R o hacc
  exact runsZCJH_live I ht hc ho dj y₀ wr ⟨Unit, fun _ => L.draw ω⟩ hT hRec (fun _ => hDraw ω R o hacc) () R o hacc

end FlockSoundness.Discharge.ZkHidden
