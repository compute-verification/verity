import Proofs.Flock.Bridge.Canon

/-!
# The assembly with canonical openings (`Bridge.AssemblyTargetC`, proved)

`Flock.Bridge.assemblyTargetC : Bridge.AssemblyTargetC`. At `kd = 0` the miss term `C(n − K₀, 0)/C(n, 0)` is `1`.
Otherwise `EndToEndHiddenC` at `K₁ = min K₀ (n − kd + 1)` bounds the event: outside `RelS` at least `K₁` slots are
faulty (`CountTarget`), and each faulty slot is a unit wrong on the registration with canonical openings, since a unit
right in that sense is not faulty (`FaultyRightTargetC`, its transport from `TransportsLiveC`; the slot is a unit by
`PartRusTarget`). `K₁`'s miss term is `K₀`'s (`choose_min_count`): where they differ both are `0`.
-/

namespace Flock.Bridge

/-- At `1 ≤ kd ≤ n`, `C(n − min K₀ (n − kd + 1), kd) = C(n − K₀, kd)`: where the two differ, both are `0`. -/
theorem choose_min_count {n kd K₀ : ℕ} (hkd1 : 1 ≤ kd) (hkd : kd ≤ n) :
    (n - min K₀ (n - kd + 1)).choose kd = (n - K₀).choose kd := by
  rcases le_total K₀ (n - kd + 1) with h | h
  · rw [min_eq_left h]
  · rw [min_eq_right h, Nat.choose_eq_zero_of_lt (by omega), Nat.choose_eq_zero_of_lt (by omega)]

end Flock.Bridge

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

/-- **The assembly, with canonical openings** (`Bridge.AssemblyTargetC`). -/
theorem _root_.Flock.Bridge.assemblyTargetC : Bridge.AssemblyTargetC := by
  intro hT hF hPR hC hE n Reg I ht D hD hU pub hpub hpt hscope kd hkd hdiv dj S₀ d₀ y₀ wr dg N Ω _ _ Lb src hA3 hTags
    hCust k Rw M hk hRw hM ρ hρ hr t' ht' K₀ hK₀ τ qF qS desc q L P vb σ Xr δ Cc rows roots Cd hTab hLink
  by_cases hk0 : kd = 0
  · refine (FlockSoundness.Game.prob_le_one _ _ τ).trans (le_trans (le_of_eq ?_) (le_self_add.trans le_self_add))
    rw [hk0]
    simp
  have hkd1 : 1 ≤ kd := Nat.pos_of_ne_zero hk0
  have hTu := hT n Reg I ht D hD hU pub hpub hpt hscope kd hkd hdiv dj S₀ d₀ y₀ wr dg N Ω Lb src hTags hCust τ
  have hsub : Claims.Sampling.faulty Claims.Sampling.sha512Msg Cd (Bridge.stmtOf P kd desc q roots) (Bridge.partOf P)
      (Bridge.witDep Cc Cd Claims.Sampling.sha512Msg roots Xr) ⊆
      (wrongRegZc execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb (fun g v => Bridge.canonRow rows g v) (reg σ)
        Xr).map Fin.valEmbedding := by
    classical
    intro v hv
    have hlt : v < n := by
      unfold Claims.Sampling.faulty at hv
      have h := (Finset.mem_filter.1 hv).1
      rw [Finset.mem_range, hPR P] at h
      exact h
    refine Finset.mem_map.2 ⟨⟨v, hlt⟩, ?_, rfl⟩
    by_contra hu
    obtain ⟨-, op, hop⟩ := not_not.1 (mt (mem_wrongRegZc _ _ _ _ _ _).2 hu)
    exact hF P vb.pos vb.bit vb.val vb.com (vb.cs (reg σ)) (fun g v => Bridge.canonRow rows g v)
      Claims.Sampling.sha512Msg Cd roots Xr ⟨v, hlt⟩ ⟨op, hop⟩ (hTu ⟨v, hlt⟩) kd desc q hv
  have hcard : ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P kd desc q roots)
      (Bridge.partOf P) (Bridge.witDep Cc Cd Claims.Sampling.sha512Msg roots Xr) →
      min K₀ (n - kd + 1) ≤ (wrongRegZc execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb
        (fun g v => Bridge.canonRow rows g v) (reg σ) Xr).card := fun hrel =>
    (hC P kd K₀ hkd1 hkd _ _ desc q roots _ hrel).trans
      ((Finset.card_le_card hsub).trans (Finset.card_map _).le)
  have hE' := hE n Reg I ht D hD hU pub hpub hpt hscope kd hkd hdiv dj S₀ d₀ y₀ wr dg N Ω Lb src hA3 hTags hCust
    k Rw M hk hRw hM ρ hρ hr t' ht' (min K₀ (n - kd + 1)) ((min_le_left _ _).trans hK₀) τ qF qS hTab hLink
  rw [Bridge.choose_min_count hkd1 hkd] at hE'
  refine le_trans (FlockSoundness.Game.prob_mono ?_ _ τ) hE'
  exact fun o ho => ⟨ho.1, hcard ho.2⟩

end Flock.Guarantees
