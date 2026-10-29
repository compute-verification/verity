import Definitions.Pouw.Guarantees.LeadGame
import Proofs.Pouw.Prob
import Proofs.Pouw.Break
import Proofs.Pouw.Theorem1

/-!
# The lead in PoUW's game, from `G_tail`

A run of the lead's game that wins and hashes no collision has a unit whose call is accepted, whose own execution and
honest commitment hash no collision, and whose work is below `32 · (honestRegs − gapFixed)` (`exists_unit_of_wins`):
the accepted calls' honest work exceeds `B / (1 − γ)`, so their work, at most the budget `B`, is below
`32 · (honestRegs − gapFixed)` on average over them. Each unit's event has probability at most `etaLead`
(`pr_unitEvent_le`): its strategy is a function of the other units' coins, so on each value of those `G_tail` bounds
its share of the unit's own coins (`pr_unit_le`). A union bound over the `N` units gives `G_γ` with error
`N · etaLead` (`gamma_of_tail`), and Theorem 1 with work-weighted sampling the end-to-end bound (`endToEnd_of_gamma`).
-/

namespace Pouw.SecurityProofs.Lead.Bridge

open Finset Pouw.Lead Pouw.Lead.Game Pouw.Assumptions.Lead Pouw.Dimension.H100

theorem etaLead_nonneg (L : Pouw.Lead.Layout) : 0 ≤ etaLead L := by
  have h1 : (0 : ℚ) ≤ tableTail L := by unfold tableTail; positivity
  have h1' : (0 : ℝ) ≤ (tableTail L : ℝ) := by exact_mod_cast h1
  have h2 : 0 ≤ coinTail L := by unfold coinTail; exact Real.iSup_nonneg fun _ => Pouw.pr_nonneg _
  unfold etaLead
  linarith

/-- **One unit's share**: an event of unit `i`'s strategy and coins, at most `e` for every strategy `can` admits, is at
most `e` over the salt, the strategy being a function of the other units' coins. -/
theorem pr_unit_le {L : Pouw.Lead.Layout} {N₀ : ℕ} {can : Strategy L → Prop} (p : LeadProg L N₀ can) (i : Fin N₀)
    (E : Strategy L → Coins L → Prop) {e : ℝ} (he : 0 ≤ e) (h : ∀ σ, can σ → Pouw.pr (E σ) ≤ e) :
    Pouw.pr (fun s : Salt L N₀ => E (p.play s i) (s i)) ≤ e :=
  le_of_eq_of_le
    (Pouw.pr_equiv ((Equiv.piSplitAt i fun _ => Coins L).trans (Equiv.prodComm _ _))
      (fun y : ({j : Fin N₀ // j ≠ i} → Coins L) × Coins L => E (p.σ i y.1) y.2))
    (Pouw.pr_prod_le _ e he fun r => h _ (p.can_σ i r))

/-- Unit `u`'s event at the salt `s`: its call is accepted, its execution and honest commitment hash no collision, and
its work is below `32 · (honestRegs − gapFixed)`. -/
def unitEvent {L : Pouw.Lead.Layout} {N₀ : ℕ} (prc : Prices100) (sha : Hash.Msg → Hash.Digest) (C : Pouw.Lead.Commit L)
    (κ : Pouw.Workload → ℕ → Fin L.m → Fin L.S → Fin L.n → ℤ) {can : Strategy L → Prop} (p : LeadProg L N₀ can)
    (U : Pouw.Workload) (u : ℕ) (s : Salt L N₀) : Prop :=
  ∃ h : u < N₀, Accepted L sha C (κ U u) (p.play s ⟨u, h⟩) (s ⟨u, h⟩) ∧
    ¬ Break L sha C (κ U u) (p.play s ⟨u, h⟩) (s ⟨u, h⟩) ∧
    ((p.play s ⟨u, h⟩).work prc (s ⟨u, h⟩) : ℚ) < 32 * (honestRegs L C - gapFixed L C)

theorem pr_unitEvent_le {Q R : Type} [Fintype Q] [Fintype R] {L : Pouw.Lead.Layout} {N₀ : ℕ} (prc : Prices100)
    (sha : Hash.Msg → Hash.Digest) (C : Pouw.Lead.Commit L) (κ : Pouw.Workload → ℕ → Fin L.m → Fin L.S → Fin L.n → ℤ)
    {can : Strategy L → Prop} (p : LeadProg L N₀ can) (U : Pouw.Workload) (u : ℕ)
    (htail : ∀ σ : Strategy L, can σ →
      Pouw.pr (fun ω : Coins L => Accepted L sha C (κ U u) σ ω ∧ ¬ Break L sha C (κ U u) σ ω ∧
        (σ.work prc ω : ℚ) < 32 * (honestRegs L C - gapFixed L C)) ≤ etaLead L) :
    Pouw.pr (fun x : (Q → R) × Salt L N₀ => unitEvent prc sha C κ p U u x.2) ≤ etaLead L := by
  have hη := etaLead_nonneg L
  by_cases h : u < N₀
  · calc Pouw.pr (fun x : (Q → R) × Salt L N₀ => unitEvent prc sha C κ p U u x.2)
          = Pouw.pr (fun y : Salt L N₀ × (Q → R) => unitEvent prc sha C κ p U u y.1) :=
            Pouw.pr_equiv (Equiv.prodComm _ _) (fun y : Salt L N₀ × (Q → R) => unitEvent prc sha C κ p U u y.1)
      _ ≤ Pouw.pr (unitEvent prc sha C κ p U u) := Pouw.pr_fst_le _
      _ ≤ Pouw.pr (fun s : Salt L N₀ =>
            Accepted L sha C (κ U u) (p.play s ⟨u, h⟩) (s ⟨u, h⟩) ∧
              ¬ Break L sha C (κ U u) (p.play s ⟨u, h⟩) (s ⟨u, h⟩) ∧
              ((p.play s ⟨u, h⟩).work prc (s ⟨u, h⟩) : ℚ) < 32 * (honestRegs L C - gapFixed L C)) :=
            Pouw.pr_mono fun _ hs => hs.2
      _ ≤ etaLead L := pr_unit_le p ⟨u, h⟩
            (fun σ ω => Accepted L sha C (κ U u) σ ω ∧ ¬ Break L sha C (κ U u) σ ω ∧
              (σ.work prc ω : ℚ) < 32 * (honestRegs L C - gapFixed L C)) hη htail
  · exact (Pouw.pr_eq_zero fun _ hx => h hx.1).le.trans hη

/-- **A winning run without a break has a unit's event**: some accepted call worked below
`32 · (honestRegs − gapFixed)` and hashed no collision. -/
theorem exists_unit_of_wins {Q R : Type} {L : Pouw.Lead.Layout} {N₀ : ℕ} (prc : Prices100)
    (sha : Hash.Msg → Hash.Digest) (C : Pouw.Lead.Commit L) (κ : Pouw.Workload → ℕ → Fin L.m → Fin L.S → Fin L.n → ℤ)
    {can : Strategy L → Prop} (p : LeadProg L N₀ can) (U : Pouw.Workload) (B : ℝ) (hB : 0 ≤ B) (H : Q → R)
    (s : Salt L N₀) (hgap : gapFixed L C < honestRegs L C)
    (hcost : ∑ u ∈ range U.layout.N, (unitWork prc p s u : ℝ) ≤ B)
    (hwin : Wins (protocol L N₀ Q R C) U (gamma L C) B (unitCommit sha C κ p U s) H s)
    (hbr : ¬ Hash.CollisionIn sha ((List.range U.layout.N).flatMap (unitHashed sha C κ p U s))) :
    ∃ u ∈ range U.layout.N, unitEvent prc sha C κ p U u s := by
  have hw₀ : (0 : ℝ) < ((32 * (honestRegs L C - gapFixed L C) : ℚ) : ℝ) := by
    have : (0 : ℚ) < 32 * (honestRegs L C - gapFixed L C) := by linarith
    exact_mod_cast this
  have hmem : ∀ u ∈ correctSet (protocol L N₀ Q R C) U H s (unitCommit sha C κ p U s),
      u ∈ range U.layout.N ∧ ∃ h : u < N₀, Accepted L sha C (κ U u) (p.play s ⟨u, h⟩) (s ⟨u, h⟩) := by
    intro u hu
    unfold correctSet at hu
    rw [mem_filter] at hu
    obtain ⟨hu, c, hc, -⟩ := hu
    refine ⟨hu, ?_⟩
    by_cases h1 : u < N₀
    · by_cases h2 : Accepted L sha C (κ U u) (p.play s ⟨u, h1⟩) (s ⟨u, h1⟩)
      · exact ⟨h1, h2⟩
      · rw [unitCommit, dif_pos h1, if_neg h2] at hc
        exact Option.noConfusion hc
    · rw [unitCommit, dif_neg h1] at hc
      exact Option.noConfusion hc
  have hγ : 1 - gamma L C =
      ((32 * (honestRegs L C - gapFixed L C) : ℚ) : ℝ) / ((⌊32 * honestRegs L C⌋₊ : ℕ) : ℝ) := by
    unfold gamma; ring
  unfold Wins at hwin
  change B / (1 - gamma L C) <
    ∑ _u ∈ correctSet (protocol L N₀ Q R C) U H s (unitCommit sha C κ p U s), ((⌊32 * honestRegs L C⌋₊ : ℕ) : ℝ)
    at hwin
  rw [sum_const, nsmul_eq_mul, hγ] at hwin
  have hBK : B < (correctSet (protocol L N₀ Q R C) U H s (unitCommit sha C κ p U s)).card *
      ((32 * (honestRegs L C - gapFixed L C) : ℚ) : ℝ) := by
    rcases (Nat.cast_nonneg _ : (0 : ℝ) ≤ ((⌊32 * honestRegs L C⌋₊ : ℕ) : ℝ)).eq_or_lt with h0 | hpos
    · rw [← h0] at hwin
      simp at hwin
    · rw [div_div_eq_mul_div, div_lt_iff₀ hw₀] at hwin
      refine lt_of_mul_lt_mul_right ?_ hpos.le
      calc B * ((⌊32 * honestRegs L C⌋₊ : ℕ) : ℝ) < _ := hwin
        _ = _ := by ring
  have hsub : ∑ u ∈ correctSet (protocol L N₀ Q R C) U H s (unitCommit sha C κ p U s), (unitWork prc p s u : ℝ) ≤
      ∑ u ∈ range U.layout.N, (unitWork prc p s u : ℝ) :=
    sum_le_sum_of_subset_of_nonneg (fun u hu => (hmem u hu).1) fun _ _ _ => Nat.cast_nonneg _
  have hlt : ∑ u ∈ correctSet (protocol L N₀ Q R C) U H s (unitCommit sha C κ p U s), (unitWork prc p s u : ℝ) <
      ∑ _u ∈ correctSet (protocol L N₀ Q R C) U H s (unitCommit sha C κ p U s),
        ((32 * (honestRegs L C - gapFixed L C) : ℚ) : ℝ) := by
    rw [sum_const, nsmul_eq_mul]
    linarith
  obtain ⟨u, huK, hu⟩ := exists_lt_of_sum_lt hlt
  obtain ⟨huN, h, hacc⟩ := hmem u huK
  refine ⟨u, huN, h, hacc, fun hb => hbr (Pouw.Proofs.collisionIn_mono (fun m hm => ?_) hb), ?_⟩
  · refine List.mem_flatMap_of_mem (List.mem_range.2 (mem_range.1 huN)) ?_
    rw [unitHashed, dif_pos h]
    exact hm
  · rw [unitWork, dif_pos h] at hu
    exact_mod_cast hu

/-- **`G_γ` for the lead's game from each unit's tail.** -/
theorem gamma_of_tail (Q R : Type) [Fintype Q] [DecidableEq Q] [Fintype R] (L : Pouw.Lead.Layout) (N₀ : ℕ)
    (prc : Prices100) (sha : Hash.Msg → Hash.Digest) (C : Pouw.Lead.Commit L)
    (κ : Pouw.Workload → ℕ → Fin L.m → Fin L.S → Fin L.n → ℤ) (can : Strategy L → Prop)
    (htail : ∀ (U : Pouw.Workload) (u : ℕ) (σ : Strategy L), can σ →
      Pouw.pr (fun ω : Coins L => Accepted L sha C (κ U u) σ ω ∧ ¬ Break L sha C (κ U u) σ ω ∧
        (σ.work prc ω : ℚ) < 32 * (honestRegs L C - gapFixed L C)) ≤ etaLead L)
    (hgap : gapFixed L C < honestRegs L C) :
    Gγ (costModel L N₀ Q R prc sha C κ can) (protocol L N₀ Q R C) (domain N₀) (gamma L C) (eta L) := by
  intro U _ B hB q _ A hA
  calc _ ≤ Pouw.pr (fun x : (Q → R) × Salt L N₀ =>
          ∃ u ∈ range U.layout.N, unitEvent prc sha C κ (N₀ := N₀) (can := can) A.online U u x.2) :=
        Pouw.pr_mono fun x hx =>
          exists_unit_of_wins prc sha C κ (N₀ := N₀) (can := can) A.online U B hB x.1 x.2 hgap (hA x.1 x.2).1 hx.1 hx.2
    _ ≤ ∑ u ∈ range U.layout.N, Pouw.pr (fun x : (Q → R) × Salt L N₀ =>
          unitEvent prc sha C κ (N₀ := N₀) (can := can) A.online U u x.2) :=
        Pouw.Proofs.pr_exists_le_sum _ (fun u x => unitEvent prc sha C κ (N₀ := N₀) (can := can) A.online U u x.2)
    _ ≤ ∑ _u ∈ range U.layout.N, etaLead L :=
        sum_le_sum fun u _ => pr_unitEvent_le prc sha C κ (N₀ := N₀) (can := can) A.online U u (htail U u)
    _ = eta L q U.layout.N := by rw [sum_const, card_range, nsmul_eq_mul]; rfl

/-- **The lead's game end to end from `G_γ`**: Theorem 1 with work-weighted sampling; a protocol whose `W_ref` is
zero claims no work. -/
theorem endToEnd_of_gamma (Q R : Type) [Fintype Q] [DecidableEq Q] [Fintype R] (L : Pouw.Lead.Layout) (N₀ : ℕ)
    (prc : Prices100) (sha : Hash.Msg → Hash.Digest) (C : Pouw.Lead.Commit L)
    (κ : Pouw.Workload → ℕ → Fin L.m → Fin L.S → Fin L.n → ℤ) (can : Strategy L → Prop)
    (hgap : gapFixed L C < honestRegs L C)
    (hG : Gγ (costModel L N₀ Q R prc sha C κ can) (protocol L N₀ Q R C) (domain N₀) (gamma L C) (eta L))
    (U : Pouw.Workload) (hU : U.InDomain (domain N₀)) (B : ℝ) (hB : 0 ≤ B) (q : ℕ) (hq : q ≤ 2 ^ 64)
    (A : Adversary (costModel L N₀ Q R prc sha C κ can)) (hA : A.Bounded U B q) (t : ℕ) (εs : ℝ) (hεs : εs ≤ 1) :
    Pouw.pr (fun x : (Q → R) × Salt L N₀ × (Fin t → Ticket ((protocol L N₀ Q R C).Wref U.layout) U.layout.N) =>
        AuditAccepts (correctSet (protocol L N₀ Q R C) U x.1 x.2.1 (A.transcript U x.1 x.2.1)) x.2.2 ∧
        B < (1 - gamma L C) * (1 - εs) * totalWref (protocol L N₀ Q R C) U.layout ∧ ¬ A.Breaks U x.1 x.2.1)
      ≤ (1 - εs) ^ t + U.layout.N * etaLead L := by
  have hη := etaLead_nonneg L
  have hw₀ : (0 : ℝ) < ((32 * (honestRegs L C - gapFixed L C) : ℚ) : ℝ) := by
    have : (0 : ℚ) < 32 * (honestRegs L C - gapFixed L C) := by linarith
    exact_mod_cast this
  rcases (Nat.cast_nonneg _ : (0 : ℝ) ≤ ((⌊32 * honestRegs L C⌋₊ : ℕ) : ℝ)).eq_or_lt with h0 | hpos
  · have hW : totalWref (protocol L N₀ Q R C) U.layout = 0 := by
      change ∑ _u ∈ range U.layout.N, ((⌊32 * honestRegs L C⌋₊ : ℕ) : ℝ) = 0
      rw [← h0, sum_const_zero]
    refine le_trans (le_of_eq (Pouw.pr_eq_zero fun x hx => ?_))
      (add_nonneg (pow_nonneg (by linarith) t) (mul_nonneg (Nat.cast_nonneg _) hη))
    have := hx.2.1
    rw [hW, mul_zero] at this
    linarith
  · have hγ : gamma L C < 1 := by
      have := div_pos hw₀ hpos
      unfold gamma
      linarith
    refine Pouw.Proofs.theorem1 Q R (Salt L N₀) (costModel L N₀ Q R prc sha C κ can) (protocol L N₀ Q R C)
      (domain N₀) (gamma L C) (eta L) hγ hG U hU B hB q hq A hA
      (Fin t → Ticket ((protocol L N₀ Q R C).Wref U.layout) U.layout.N)
      (fun H s c => AuditAccepts (correctSet (protocol L N₀ Q R C) U H s (A.transcript U H s)) c) εs ((1 - εs) ^ t) ?_
    let G := fun y : ((Q → R) × Salt L N₀) × (Fin t → Ticket ((protocol L N₀ Q R C).Wref U.layout) U.layout.N) =>
      AuditAccepts (correctSet (protocol L N₀ Q R C) U y.1.1 y.1.2 (A.transcript U y.1.1 y.1.2)) y.2 ∧
        εs * ∑ u ∈ range U.layout.N, ((protocol L N₀ Q R C).Wref U.layout u : ℝ) <
          ∑ u ∈ range U.layout.N \ correctSet (protocol L N₀ Q R C) U y.1.1 y.1.2 (A.transcript U y.1.1 y.1.2),
            ((protocol L N₀ Q R C).Wref U.layout u : ℝ)
    calc _ = Pouw.pr G := Pouw.pr_equiv (Equiv.prodAssoc _ _ _).symm G
      _ ≤ _ := Pouw.Proofs.workWeightedSampling ((Q → R) × Salt L N₀) ((protocol L N₀ Q R C).Wref U.layout)
        U.layout.N t (fun y => correctSet (protocol L N₀ Q R C) U y.1 y.2 (A.transcript U y.1 y.2)) εs hεs

/-- **`G_γ` for the lead's calls, from `G_tail`** (`Pouw.Guarantees.Lead.GammaFromTail`). -/
theorem GammaFromTail : Pouw.Guarantees.Lead.GammaFromTail := by
  intro Q R _ _ _ prc sha T m n K N₀ stream C isA can real κ hG hκ h32 hW1 hOne hCard hSha hgap
  exact gamma_of_tail Q R (lead T m n K) N₀ prc sha C κ can
    (fun U u σ hσ => hG prc sha T m n K (κ U u) stream C isA can real (hκ U u).1 h32 hW1 (hκ U u).2.1 hOne hCard
      hSha (hκ U u).2.2.1 (hκ U u).2.2.2 σ hσ) hgap

/-- **The lead's calls end to end, from `G_tail`** (`Pouw.Guarantees.Lead.EndToEndFromTail`). -/
theorem EndToEndFromTail : Pouw.Guarantees.Lead.EndToEndFromTail := by
  intro Q R _ _ _ prc sha T m n K N₀ stream C isA can real κ hG hκ h32 hW1 hOne hCard hSha hgap
  exact endToEnd_of_gamma Q R (lead T m n K) N₀ prc sha C κ can hgap
    (GammaFromTail Q R prc sha T m n K N₀ stream C isA can real κ hG hκ h32 hW1 hOne hCard hSha hgap)

end Pouw.SecurityProofs.Lead.Bridge
