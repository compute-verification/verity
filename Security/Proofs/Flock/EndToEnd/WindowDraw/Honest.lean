import Proofs.Flock.EndToEnd.WindowDraw.Bound

/-!
# `WindowBound`'s witnesses

`WindowBound_honest`, the deployed window: its draw file and its three sessions' draw files read as pinned
(`FlockSoundness.Assumptions.PinnedWindowDraws`); its strata are `[0, 2496)`, `[2496, 3120)` and `[3120, 3744)`, of
2496, 624 and 624 units (`card0`, `card1`, `card2`); its thirteen members tile it (`tile`), each inside its stratum
(`inside`); the service's split of its draw file is the three pinned session files; and the members' sizes over their
strata's sum to one per stratum (`share_sum`), so the composed bound is at most `3 · b`.

`WindowBound_nontrivial`: a member across two strata, whose composed bound fails. Every premise of `WindowBound` but
the member's units in its stratum holds at the toy (`WindowToy.fit_holds`, `event_nonempty_holds`,
`session_bound_holds`, with the law's `Flock.Guarantees.WindowToy.hk`), and that one fails (`in_stratum_fails`).
-/

namespace Flock.SecurityProofs

open Finset FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Audit.Window
open scoped ENNReal

namespace WindowDeployed

theorem stratum_eq_range (s : Fin 3) (lo hi : ℕ) (hlo : lo ≤ hi) (hhi : hi ≤ 3744)
    (h : ∀ u : Fin 3744, Flock.Guarantees.WindowDeployed.strata u = s ↔ lo ≤ u.val ∧ u.val < hi) :
    (Law.stratum Flock.Guarantees.WindowDeployed.strata s).card = hi - lo := by
  have he : Law.stratum Flock.Guarantees.WindowDeployed.strata s = (⟨lo, hi - lo⟩ : Member).range := by
    ext u
    rw [Law.mem_stratum, h u]
    simp only [Member.range, mem_filter, mem_univ, true_and]
    omega
  rw [he, Member.card_range (by simp only; omega)]

theorem card0 : (Law.stratum Flock.Guarantees.WindowDeployed.strata 0).card = 2496 :=
  stratum_eq_range 0 0 2496 (by norm_num) (by norm_num) fun u => by
    simp only [Flock.Guarantees.WindowDeployed.strata]
    split_ifs <;> simp_all

theorem card1 : (Law.stratum Flock.Guarantees.WindowDeployed.strata 1).card = 624 :=
  stratum_eq_range 1 2496 3120 (by norm_num) (by norm_num) fun u => by
    simp only [Flock.Guarantees.WindowDeployed.strata]
    split_ifs <;> simp_all

theorem card2 : (Law.stratum Flock.Guarantees.WindowDeployed.strata 2).card = 624 :=
  stratum_eq_range 2 3120 3744 (by norm_num) (by norm_num) fun u => by
    simp only [Flock.Guarantees.WindowDeployed.strata]
    have := u.isLt
    split_ifs <;> simp_all
    omega

theorem fit : ∀ j, (Flock.Guarantees.WindowDeployed.member j).base +
    (Flock.Guarantees.WindowDeployed.member j).size ≤ 3744 := by
  decide

theorem inside : ∀ j, ∀ u : Fin 3744, (Flock.Guarantees.WindowDeployed.member j).base ≤ u.val →
    u.val < (Flock.Guarantees.WindowDeployed.member j).base + (Flock.Guarantees.WindowDeployed.member j).size →
    Flock.Guarantees.WindowDeployed.strata u = Flock.Guarantees.WindowDeployed.str j := by
  intro j u h1 h2
  fin_cases j <;>
    simp only [Flock.Guarantees.WindowDeployed.member, Flock.Guarantees.WindowDeployed.str,
      Flock.Guarantees.WindowDeployed.strata] at * <;>
    simp at h1 h2 ⊢ <;> split_ifs <;> first | rfl | omega

theorem tile : ∀ u : Fin 3744, ∃! j, (Flock.Guarantees.WindowDeployed.member j).base ≤ u.val ∧
    u.val < (Flock.Guarantees.WindowDeployed.member j).base + (Flock.Guarantees.WindowDeployed.member j).size := by
  intro u
  have hu := u.isLt
  refine existsUnique_of_exists_of_unique ?_ ?_
  · simp only [Fin.exists_fin_succ, Flock.Guarantees.WindowDeployed.member]
    simp
    omega
  · intro y₁ y₂ h₁ h₂
    fin_cases y₁ <;> fin_cases y₂ <;> simp [Flock.Guarantees.WindowDeployed.member] at h₁ h₂ ⊢ <;> omega

theorem share_sum :
    ∑ j, ((Flock.Guarantees.WindowDeployed.member j).size : ℝ≥0∞) /
      ((Law.stratum Flock.Guarantees.WindowDeployed.strata (Flock.Guarantees.WindowDeployed.str j)).card : ℝ≥0∞)
      = 3 := by
  simp only [Fin.sum_univ_succ, Fin.sum_univ_zero, Flock.Guarantees.WindowDeployed.member,
    Flock.Guarantees.WindowDeployed.str]
  simp [card0, card1, card2, div_eq_mul_inv]
  have e0 : (2496 : ℝ≥0∞) * 2496⁻¹ = 1 := ENNReal.mul_inv_cancel (by norm_num) (by simp)
  have e1 : (624 : ℝ≥0∞) * 624⁻¹ = 1 := ENNReal.mul_inv_cancel (by norm_num) (by simp)
  calc _ = 2496 * (2496 : ℝ≥0∞)⁻¹ + 624 * (624 : ℝ≥0∞)⁻¹ + 624 * (624 : ℝ≥0∞)⁻¹ := by ring
    _ = 3 := by rw [e0, e1]; norm_num

end WindowDeployed

/-- **`WindowBound`'s honest lemma: the deployed window**, through `FlockSoundness.Assumptions.PinnedWindowDraws`. -/
theorem WindowBound_honest : Flock.Guarantees.WindowBound_honest := by
  intro files hpin
  obtain ⟨hW, h1, h10, h12⟩ := hpin
  have h1' : files Flock.Guarantees.WindowDeployed.m1Digest = some ⟨"subset", 256, 1, [], [165]⟩ := h1
  have h10' : files Flock.Guarantees.WindowDeployed.m10Digest = some ⟨"subset", 512, 1, [], [328]⟩ := h10
  have h12' : files Flock.Guarantees.WindowDeployed.m12Digest = some ⟨"subset", 624, 1, [], [150]⟩ := h12
  refine ⟨_, hW, rfl, rfl, by decide, ?_, ?_, WindowDeployed.card0,
    WindowDeployed.card1, WindowDeployed.card2, ?_, WindowDeployed.fit, WindowDeployed.inside, WindowDeployed.tile,
    ?_, ?_⟩
  · intro u
    simp only [Flock.Guarantees.WindowDeployed.strata, DrawFile.stratumOf, List.findIdx_cons, List.any_cons,
      List.any_nil, Bool.or_false, List.findIdx_nil]
    split_ifs <;> simp_all <;> omega
  · intro s
    fin_cases s <;> rfl
  · intro s
    fin_cases s <;>
      simp [Flock.Guarantees.WindowDeployed.k, WindowDeployed.card0, WindowDeployed.card1, WindowDeployed.card2]
  · intro j
    fin_cases j <;> simp only [h1', h10', h12'] <;> rfl
  · intro hk
    refine ⟨?_, fun ω j => share_card_le (hk := hk) (Flock.Guarantees.WindowDeployed.member j)
      (Flock.Guarantees.WindowDeployed.str j) (WindowDeployed.fit j) (WindowDeployed.inside j) ω, ?_⟩
    · let pick : Fin 3 → Fin 3744 := ![421, 2824, 3270]
      have hpick : ∀ s, ({pick s} : Finset (Fin 3744)) ∈
          (Law.stratum Flock.Guarantees.WindowDeployed.strata s).powersetCard (Flock.Guarantees.WindowDeployed.k s) := by
        intro s
        rw [mem_powersetCard, singleton_subset_iff, card_singleton, Law.mem_stratum]
        fin_cases s <;> exact ⟨rfl, rfl⟩
      refine ⟨fun s => ⟨{pick s}, hpick s⟩, ?_⟩
      ext u
      show u ∈ univ.biUnion (fun s => ({pick s} : Finset (Fin 3744))) ↔ _
      simp [pick, Fin.exists_fin_succ, Fin.ext_iff]
    · intro α Reg body E β hrun hsess τ
      have hb := window_bound_prob Flock.Guarantees.WindowDeployed.strata Flock.Guarantees.WindowDeployed.k hk body
        univ Flock.Guarantees.WindowDeployed.member Flock.Guarantees.WindowDeployed.str
        (fun j _ => WindowDeployed.fit j) (fun j _ => WindowDeployed.inside j) E (fun j _ => hrun j) β
        (fun j _ κ hκ h1 hk1 => hsess j κ hκ h1 hk1) τ
      rw [window_bound_one Flock.Guarantees.WindowDeployed.strata Flock.Guarantees.WindowDeployed.k hk univ
        Flock.Guarantees.WindowDeployed.member Flock.Guarantees.WindowDeployed.str
        (fun j _ => WindowDeployed.fit j) (fun j _ => WindowDeployed.inside j) (fun _ _ => rfl) β] at hb
      have hP := (prob_mono (p := fun o => ∃ j, E j o) (q := fun o => ∃ j ∈ univ, E j o)
        (fun _ ⟨j, hj⟩ => ⟨j, mem_univ j, hj⟩) _ τ).trans hb
      refine ⟨hP, fun b hβ => hP.trans ?_⟩
      calc ∑ j, ((Flock.Guarantees.WindowDeployed.member j).size : ℝ≥0∞) /
            ((Law.stratum Flock.Guarantees.WindowDeployed.strata (Flock.Guarantees.WindowDeployed.str j)).card :
              ℝ≥0∞) * β j 1
          ≤ ∑ j, ((Flock.Guarantees.WindowDeployed.member j).size : ℝ≥0∞) /
            ((Law.stratum Flock.Guarantees.WindowDeployed.strata (Flock.Guarantees.WindowDeployed.str j)).card :
              ℝ≥0∞) * b := sum_le_sum fun j _ => by gcongr; exact hβ j
        _ = 3 * b := by rw [← sum_mul, WindowDeployed.share_sum]

namespace WindowToy

/-- `WindowBound`'s premise that each member is inside the window holds at the toy. -/
theorem fit_holds : Flock.Guarantees.WindowToy.member.base + Flock.Guarantees.WindowToy.member.size ≤ 3 := by
  decide

/-- `WindowBound`'s premise that each member's event happens only at a nonempty share holds at the toy's event, a share
of two units. -/
theorem event_nonempty_holds : ∀ o : Unit × Finset (Fin 3) × Unit,
    (Flock.Guarantees.WindowToy.member.share o.2.1).card = 2 → (Flock.Guarantees.WindowToy.member.share o.2.1).Nonempty :=
  fun _ h => card_pos.1 (by omega)

/-- `WindowBound`'s premise on each member's session bound holds at the toy with `β = 0`: at its one share size, the
member's own `subset` session draws one of its units, so its share never has two. -/
theorem session_bound_holds : ∀ κ (hκ : κ ≤ Flock.Guarantees.WindowToy.member.size), 1 ≤ κ →
    κ ≤ Flock.Guarantees.WindowToy.k 0 →
    ∀ ctx : Finset (Fin 3), Disjoint ctx Flock.Guarantees.WindowToy.member.range →
    ∀ τ : Strategy (audit (Law.subset Flock.Guarantees.WindowToy.member.size κ hκ) Unit
        fun S R => Flock.Guarantees.WindowToy.body (ctx ∪ Flock.Guarantees.WindowToy.member.unshift S) R),
      prob (fun o => (Flock.Guarantees.WindowToy.member.share
          (ctx ∪ Flock.Guarantees.WindowToy.member.unshift o.2.1)).card = 2)
        (audit (Law.subset Flock.Guarantees.WindowToy.member.size κ hκ) Unit
          fun S R => Flock.Guarantees.WindowToy.body (ctx ∪ Flock.Guarantees.WindowToy.member.unshift S) R) τ ≤ 0 := by
  intro κ hκ h1 hk1 ctx hctx τ
  rw [prob_audit, avg]
  refine le_of_eq (sum_eq_zero fun S _ => ?_)
  rw [prob_map]
  refine mul_eq_zero_of_right _ (le_antisymm ((prob_mono (q := fun _ => False) (fun a ha => ?_) _ _).trans
    (prob_false _ _).le) zero_le)
  have ha' : (Flock.Guarantees.WindowToy.member.share
      (ctx ∪ Flock.Guarantees.WindowToy.member.unshift ((Law.subset _ κ hκ).draw S))).card = 2 := ha
  have hc : ((Law.subset _ κ hκ).draw S).card = κ := S.2
  rw [Member.share_union_unshift fit_holds ctx _ hctx, hc] at ha'
  have : κ ≤ 1 := hk1
  omega

/-- `WindowBound`'s premise that each member's units are in its stratum fails at the toy: the member `[0, 2)`, taken
in stratum 0, has unit 1, of stratum 1. -/
theorem in_stratum_fails : ¬ ∀ u : Fin 3, Flock.Guarantees.WindowToy.member.base ≤ u.val →
    u.val < Flock.Guarantees.WindowToy.member.base + Flock.Guarantees.WindowToy.member.size →
    Flock.Guarantees.WindowToy.strata u = 0 :=
  fun h => absurd (h 1 (by decide) (by decide)) (by decide)

end WindowToy

/-- **`WindowBound`'s nontrivial lemma**: without its members inside their strata, the composed bound fails. -/
theorem WindowBound_nontrivial : Flock.Guarantees.WindowBound_nontrivial := by
  intro hall
  let τ₀ : Strategy (audit (Law.stratified Flock.Guarantees.WindowToy.strata Flock.Guarantees.WindowToy.k
    Flock.Guarantees.WindowToy.hk) Unit Flock.Guarantees.WindowToy.body) := ⟨(), fun _ => PUnit.unit⟩
  have h := hall τ₀
  simp only [mul_zero, sum_const_zero, nonpos_iff_eq_zero] at h
  rw [prob_audit, avg] at h
  have hpick : ∀ s, ({(![0, 1] : Fin 2 → Fin 3) s} : Finset (Fin 3)) ∈
      (Law.stratum Flock.Guarantees.WindowToy.strata s).powersetCard (Flock.Guarantees.WindowToy.k s) := by
    intro s
    rw [mem_powersetCard, singleton_subset_iff, card_singleton, Law.mem_stratum]
    fin_cases s <;> exact ⟨rfl, rfl⟩
  let ω₀ : (Law.stratified Flock.Guarantees.WindowToy.strata Flock.Guarantees.WindowToy.k
      Flock.Guarantees.WindowToy.hk).Ω :=
    fun s => ⟨{(![0, 1] : Fin 2 → Fin 3) s}, hpick s⟩
  have hd : (Law.stratified Flock.Guarantees.WindowToy.strata Flock.Guarantees.WindowToy.k
      Flock.Guarantees.WindowToy.hk).draw ω₀ = {0, 1} := by
    show univ.biUnion (fun s => ({(![0, 1] : Fin 2 → Fin 3) s} : Finset (Fin 3))) = {0, 1}
    decide
  have hE : (Flock.Guarantees.WindowToy.member.share ((Law.stratified Flock.Guarantees.WindowToy.strata
      Flock.Guarantees.WindowToy.k Flock.Guarantees.WindowToy.hk).draw ω₀)).card = 2 := by
    rw [hd]
    decide
  have h' := (sum_eq_zero_iff.1 h) ω₀ (mem_univ _)
  rw [prob_map] at h'
  rcases mul_eq_zero.1 h' with h0 | h0
  · exact ENNReal.inv_ne_zero.2 (ENNReal.natCast_ne_top _) h0
  · simp only [Flock.Guarantees.WindowToy.body, prob] at h0
    split_ifs at h0 with hc
    · exact one_ne_zero h0
    · exact hc ⟨(), mem_univ _, hE⟩

end Flock.SecurityProofs
