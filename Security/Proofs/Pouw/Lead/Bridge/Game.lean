import Definitions.Pouw.Guarantees.LeadGame
import Proofs.Pouw.Prob
import Proofs.Pouw.Break
import Proofs.Pouw.Theorem1
import Proofs.Pouw.Lead.Compose.Honest
import Proofs.Pouw.Lead.Network.Term

/-!
# The lead in PoUW's game, from `G_tail`

A run of the lead's game that wins and hashes no collision has a unit whose call is accepted, whose own execution and
honest commitment hash no collision, and whose cost, its work plus `betaLead` a bit it exchanged, is below
`32 · (honestRegs − gapFixed)` (`exists_unit_of_wins`): the accepted calls' honest work exceeds `B / (1 − γ)`, so their
cost, at most the budget `B`, is below `32 · (honestRegs − gapFixed)` on average over them. Each unit's event has
probability at most `etaLead` (`pr_unitEvent_le`): its strategy is a function of the other units' coins, so on each
value of those `G_tail` bounds its share of the unit's own coins (`pr_unit_le`), at a certifier whose entropy is the
coins (`Compose.certifierCoins`). A union bound over the `N` units bounds the winning runs at every oracle bound
(`pr_wins_le`), which is `G_γ` with error `N · etaLead` (`gamma_of_tail`), and Theorem 1's split with work-weighted
sampling gives the end-to-end bound (`endToEnd_of_tail`), at every oracle bound too.
-/

namespace Pouw.SecurityProofs.Lead.Bridge

open Finset Pouw.Lead Pouw.Lead.Game Pouw.Assumptions.Lead Pouw.Dimension.H100

theorem etaLead_nonneg (L : Pouw.Lead.Layout) : 0 ≤ etaLead L := by
  have h1 : (0 : ℚ) ≤ tableTail L := by unfold tableTail; positivity
  have h1' : (0 : ℝ) ≤ (tableTail L : ℝ) := by exact_mod_cast h1
  have h2 : 0 ≤ coinTail L := by unfold coinTail; exact Real.iSup_nonneg fun _ => Pouw.pr_nonneg _
  unfold etaLead
  linarith

theorem unitCost_nonneg {L : Pouw.Lead.Layout} {N₀ : ℕ} (prc : Prices100) (C : Pouw.Lead.Commit L)
    {can : Strategy L → Prop} (p : LeadProg L N₀ can) (s : Pouw.Lead.Game.Salt L N₀) (u : ℕ) :
    0 ≤ unitCost prc C p s u := by
  unfold unitCost
  split
  · exact add_nonneg (Nat.cast_nonneg _) (mul_nonneg (Network.betaLead_nonneg L C) (Nat.cast_nonneg _))
  · exact le_rfl

/-- **One unit's share**: an event of unit `i`'s strategy and coins, at most `e` for every strategy `can` admits, is at
most `e` over the salt, the strategy being a function of the other units' coins. -/
theorem pr_unit_le {L : Pouw.Lead.Layout} {N₀ : ℕ} {can : Strategy L → Prop} (p : LeadProg L N₀ can) (i : Fin N₀)
    (E : Strategy L → Coins L → Prop) {e : ℝ} (he : 0 ≤ e) (h : ∀ σ, can σ → Pouw.pr (E σ) ≤ e) :
    Pouw.pr (fun s : Pouw.Lead.Game.Salt L N₀ => E (p.play s i) (s i)) ≤ e := by
  let f : Pouw.Lead.Game.Salt L N₀ ≃ ({j : Fin N₀ // j ≠ i} → Coins L) × Coins L :=
    (Equiv.piSplitAt i fun _ => Coins L).trans (Equiv.prodComm _ _)
  let P : ({j : Fin N₀ // j ≠ i} → Coins L) × Coins L → Prop := fun y => E (p.σ i y.1) y.2
  have h1 : Pouw.pr (fun s : Pouw.Lead.Game.Salt L N₀ => P (f s)) = Pouw.pr P := Pouw.pr_equiv f P
  have h2 : Pouw.pr P ≤ e := Pouw.pr_prod_le P e he fun r => h (p.σ i r) (p.can_σ i r)
  have h3 : (fun s : Pouw.Lead.Game.Salt L N₀ => E (p.play s i) (s i)) = fun s => P (f s) := by
    funext s; rfl
  rw [h3, h1]
  exact h2

/-- `G_tail`'s event at a strategy `σ` and the coins `ω`: the call is accepted, its execution and honest commitment
hash no collision, and its work plus `betaLead` a bit it exchanged is below `32 · (honestRegs − gapFixed)`. -/
def TailEvent {L : Pouw.Lead.Layout} (prc : Prices100) (sha : Hash.Msg → Hash.Digest) (C : Pouw.Lead.Commit L)
    (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (σ : Strategy L) (ω : Coins L) : Prop :=
  Accepted L sha C κ σ ω ∧ ¬ Break L sha C κ σ ω ∧
    (σ.work prc ω : ℚ) + betaLead L C * (σ.exec ω).egress < 32 * (honestRegs L C - gapFixed L C)

/-- Unit `u`'s event at the salt `s`: `G_tail`'s event at its strategy and coins. -/
def unitEvent {L : Pouw.Lead.Layout} {N₀ : ℕ} (prc : Prices100) (sha : Hash.Msg → Hash.Digest) (C : Pouw.Lead.Commit L)
    (κ : Pouw.Workload → ℕ → Fin L.m → Fin L.S → Fin L.n → ℤ) {can : Strategy L → Prop} (p : LeadProg L N₀ can)
    (U : Pouw.Workload) (u : ℕ) (s : Pouw.Lead.Game.Salt L N₀) : Prop :=
  ∃ h : u < N₀, TailEvent prc sha C (κ U u) (p.play s ⟨u, h⟩) (s ⟨u, h⟩)

theorem pr_unitEvent_le {Q R : Type} [Fintype Q] [DecidableEq Q] [Fintype R] {L : Pouw.Lead.Layout} {N₀ : ℕ}
    (prc : Prices100) (sha : Hash.Msg → Hash.Digest) (C : Pouw.Lead.Commit L)
    (κ : Pouw.Workload → ℕ → Fin L.m → Fin L.S → Fin L.n → ℤ) {can : Strategy L → Prop} (p : LeadProg L N₀ can)
    (U : Pouw.Workload) (u : ℕ)
    (htail : ∀ σ : Strategy L, can σ → Pouw.pr (TailEvent prc sha C (κ U u) σ) ≤ etaLead L) :
    Pouw.pr (fun x : (Q → R) × Pouw.Lead.Game.Salt L N₀ => unitEvent prc sha C κ p U u x.2) ≤ etaLead L := by
  have hη := etaLead_nonneg L
  refine Pouw.pr_prod_le _ (etaLead L) hη fun _ => ?_
  show Pouw.pr (fun s : Pouw.Lead.Game.Salt L N₀ => unitEvent prc sha C κ p U u s) ≤ etaLead L
  by_cases h : u < N₀
  · have h1 := pr_unit_le p ⟨u, h⟩ (TailEvent prc sha C (κ U u)) hη htail
    refine le_trans (Pouw.pr_mono fun s hs => ?_) h1
    obtain ⟨_, hs⟩ := hs
    exact hs
  · exact (Pouw.pr_eq_zero fun _ ⟨hu, _⟩ => h hu).le.trans hη

/-- **A winning run without a break has a unit's event**: some accepted call cost below
`32 · (honestRegs − gapFixed)` and hashed no collision. -/
theorem exists_unit_of_wins {Q R : Type} {L : Pouw.Lead.Layout} {N₀ : ℕ} (prc : Prices100)
    (sha : Hash.Msg → Hash.Digest) (C : Pouw.Lead.Commit L) (κ : Pouw.Workload → ℕ → Fin L.m → Fin L.S → Fin L.n → ℤ)
    {can : Strategy L → Prop} (p : LeadProg L N₀ can) (U : Pouw.Workload) (B : ℝ) (H : Q → R)
    (s : Pouw.Lead.Game.Salt L N₀) (hgap : FixedGapBelow L C)
    (hcost : ∑ u ∈ range U.layout.N, (unitCost prc C p s u : ℝ) ≤ B)
    (hwin : Wins (protocol L N₀ Q R C) U (gamma L C) B (unitCommit sha C κ p U s) H s)
    (hbr : ¬ Hash.CollisionIn sha ((List.range U.layout.N).flatMap (unitHashed sha C κ p U s))) :
    ∃ u ∈ range U.layout.N, unitEvent prc sha C κ p U u s := by
  have hgap' : gapFixed L C < honestRegs L C := hgap
  have hw₀ : (0 : ℝ) < ((32 * (honestRegs L C - gapFixed L C) : ℚ) : ℝ) := by
    have : (0 : ℚ) < 32 * (honestRegs L C - gapFixed L C) := by linarith
    exact_mod_cast this
  have hmem : ∀ u ∈ correctSet (protocol L N₀ Q R C) U H s (unitCommit sha C κ p U s),
      u ∈ range U.layout.N ∧ ∃ h : u < N₀, Accepted L sha C (κ U u) (p.play s ⟨u, h⟩) (s ⟨u, h⟩) := by
    classical
    intro u hu
    unfold correctSet at hu
    rw [mem_filter] at hu
    obtain ⟨hu, c, hc, -⟩ := hu
    refine ⟨hu, ?_⟩
    by_cases h1 : u < N₀
    · by_cases h2 : Accepted L sha C (κ U u) (p.play s ⟨u, h1⟩) (s ⟨u, h1⟩)
      · exact ⟨h1, h2⟩
      · rw [unitCommit, dif_pos h1, if_neg h2] at hc
        simp at hc
    · rw [unitCommit, dif_neg h1] at hc
      simp at hc
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
      by_contra hc
      nlinarith [mul_le_mul_of_nonneg_right (not_lt.1 hc) hpos.le]
  have hsub : ∑ u ∈ correctSet (protocol L N₀ Q R C) U H s (unitCommit sha C κ p U s), (unitCost prc C p s u : ℝ) ≤
      ∑ u ∈ range U.layout.N, (unitCost prc C p s u : ℝ) :=
    sum_le_sum_of_subset_of_nonneg (fun u hu => (hmem u hu).1) fun u _ _ => by
      exact_mod_cast unitCost_nonneg prc C p s u
  have hlt : ∑ u ∈ correctSet (protocol L N₀ Q R C) U H s (unitCommit sha C κ p U s), (unitCost prc C p s u : ℝ) <
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
  · rw [unitCost, dif_pos h] at hu
    exact_mod_cast hu

/-- **The winning runs at every oracle bound**: for every workload, budget `B`, oracle bound `q` and adversary within
them, `Pr[win ∧ no collision] ≤ N · etaLead`. Neither the domain nor the oracle bound enters. -/
theorem pr_wins_le {Q R : Type} [Fintype Q] [DecidableEq Q] [Fintype R] {L : Pouw.Lead.Layout} {N₀ : ℕ}
    (prc : Prices100) (sha : Hash.Msg → Hash.Digest) (C : Pouw.Lead.Commit L)
    (κ : Pouw.Workload → ℕ → Fin L.m → Fin L.S → Fin L.n → ℤ) {can : Strategy L → Prop}
    (htail : ∀ (U : Pouw.Workload) (u : ℕ) (σ : Strategy L), can σ →
      Pouw.pr (TailEvent prc sha C (κ U u) σ) ≤ etaLead L)
    (hgap : FixedGapBelow L C) (U : Pouw.Workload) (B : ℝ) (q : ℕ)
    (A : Adversary (costModel L N₀ Q R prc sha C κ can)) (hA : A.Bounded U B q) :
    Pouw.pr (fun x : (Q → R) × Pouw.Lead.Game.Salt L N₀ =>
        Wins (protocol L N₀ Q R C) U (gamma L C) B (A.transcript U x.1 x.2) x.1 x.2 ∧ ¬ A.Breaks U x.1 x.2)
      ≤ U.layout.N * etaLead L := by
  have hev : ∀ x : (Q → R) × Pouw.Lead.Game.Salt L N₀,
      Wins (protocol L N₀ Q R C) U (gamma L C) B (A.transcript U x.1 x.2) x.1 x.2 ∧ ¬ A.Breaks U x.1 x.2 →
      ∃ u ∈ range U.layout.N, unitEvent prc sha C κ (N₀ := N₀) (can := can) A.online U u x.2 := by
    intro x hx
    have hcost : ∑ u ∈ range U.layout.N, (unitCost prc C (can := can) A.online x.2 u : ℝ) ≤ B :=
      (hA x.1 x.2).1
    have hwin : Wins (protocol L N₀ Q R C) U (gamma L C) B
        (unitCommit sha C κ (can := can) A.online U x.2) x.1 x.2 := hx.1
    have hbr : ¬ Hash.CollisionIn sha
        ((List.range U.layout.N).flatMap (unitHashed sha C κ (can := can) A.online U x.2)) := hx.2
    exact exists_unit_of_wins prc sha C κ A.online U B x.1 x.2 hgap hcost hwin hbr
  have h1 := Pouw.pr_mono hev
  have h2 := Pouw.Proofs.pr_exists_le_sum (α := (Q → R) × Pouw.Lead.Game.Salt L N₀) (range U.layout.N)
    fun u x => unitEvent prc sha C κ (N₀ := N₀) (can := can) A.online U u x.2
  have h3 : ∑ u ∈ range U.layout.N, Pouw.pr (fun x : (Q → R) × Pouw.Lead.Game.Salt L N₀ =>
      unitEvent prc sha C κ (N₀ := N₀) (can := can) A.online U u x.2) ≤ ∑ _u ∈ range U.layout.N, etaLead L :=
    sum_le_sum fun u _ => pr_unitEvent_le prc sha C κ (N₀ := N₀) (can := can) A.online U u (htail U u)
  have h4 : ∑ _u ∈ range U.layout.N, etaLead L = U.layout.N * etaLead L := by
    rw [sum_const, card_range, nsmul_eq_mul]
  exact h1.trans (h2.trans (h3.trans h4.le))

/-- **`G_γ` for the lead's game from each unit's tail.** -/
theorem gamma_of_tail (Q R : Type) [Fintype Q] [DecidableEq Q] [Fintype R] (L : Pouw.Lead.Layout) (N₀ : ℕ)
    (prc : Prices100) (sha : Hash.Msg → Hash.Digest) (C : Pouw.Lead.Commit L)
    (κ : Pouw.Workload → ℕ → Fin L.m → Fin L.S → Fin L.n → ℤ) (can : Strategy L → Prop)
    (htail : ∀ (U : Pouw.Workload) (u : ℕ) (σ : Strategy L), can σ →
      Pouw.pr (TailEvent prc sha C (κ U u) σ) ≤ etaLead L)
    (hgap : FixedGapBelow L C) :
    Gγ (costModel L N₀ Q R prc sha C κ can) (protocol L N₀ Q R C) (domain N₀) (gamma L C) (eta L) :=
  fun U _ B _ q _ A hA => pr_wins_le prc sha C κ htail hgap U B q A hA

/-- **The lead's game end to end from each unit's tail**: Theorem 1's split (an accepted run either has much
uncorrect work, which work-weighted sampling bounds, or wins) with `pr_wins_le`, at every oracle bound; a protocol
whose `W_ref` is zero claims no work. -/
theorem endToEnd_of_tail (Q R : Type) [Fintype Q] [DecidableEq Q] [Fintype R] (L : Pouw.Lead.Layout) (N₀ : ℕ)
    (prc : Prices100) (sha : Hash.Msg → Hash.Digest) (C : Pouw.Lead.Commit L)
    (κ : Pouw.Workload → ℕ → Fin L.m → Fin L.S → Fin L.n → ℤ) (can : Strategy L → Prop)
    (htail : ∀ (U : Pouw.Workload) (u : ℕ) (σ : Strategy L), can σ →
      Pouw.pr (TailEvent prc sha C (κ U u) σ) ≤ etaLead L)
    (hgap : FixedGapBelow L C) (U : Pouw.Workload) (B : ℝ) (hB : 0 ≤ B) (q : ℕ)
    (A : Adversary (costModel L N₀ Q R prc sha C κ can)) (hA : A.Bounded U B q) (t : ℕ) (εs : ℝ) (hεs : εs ≤ 1) :
    Pouw.pr (fun x : (Q → R) × Pouw.Lead.Game.Salt L N₀ ×
        (Fin t → Ticket ((protocol L N₀ Q R C).Wref U.layout) U.layout.N) =>
        AuditAccepts (correctSet (protocol L N₀ Q R C) U x.1 x.2.1 (A.transcript U x.1 x.2.1)) x.2.2 ∧
        B < (1 - gamma L C) * (1 - εs) * totalWref (protocol L N₀ Q R C) U.layout ∧ ¬ A.Breaks U x.1 x.2.1)
      ≤ (1 - εs) ^ t + U.layout.N * etaLead L := by
  have hgap' : gapFixed L C < honestRegs L C := hgap
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
    let P := protocol L N₀ Q R C
    let Tk := Fin t → Ticket (P.Wref U.layout) U.layout.N
    let E1 := fun x : (Q → R) × Pouw.Lead.Game.Salt L N₀ × Tk =>
      AuditAccepts (correctSet P U x.1 x.2.1 (A.transcript U x.1 x.2.1)) x.2.2 ∧
        εs * totalWref P U.layout <
          ∑ u ∈ range U.layout.N \ correctSet P U x.1 x.2.1 (A.transcript U x.1 x.2.1), (P.Wref U.layout u : ℝ)
    let E2 := fun y : (Q → R) × Pouw.Lead.Game.Salt L N₀ =>
      Wins P U (gamma L C) B (A.transcript U y.1 y.2) y.1 y.2 ∧ ¬ A.Breaks U y.1 y.2
    let G := fun y : ((Q → R) × Pouw.Lead.Game.Salt L N₀) × Tk =>
      AuditAccepts (correctSet P U y.1.1 y.1.2 (A.transcript U y.1.1 y.1.2)) y.2 ∧
        εs * ∑ u ∈ range U.layout.N, (P.Wref U.layout u : ℝ) <
          ∑ u ∈ range U.layout.N \ correctSet P U y.1.1 y.1.2 (A.transcript U y.1.1 y.1.2), (P.Wref U.layout u : ℝ)
    have hE1 : Pouw.pr E1 ≤ (1 - εs) ^ t :=
      calc Pouw.pr E1 = Pouw.pr G := Pouw.pr_equiv (Equiv.prodAssoc _ _ _).symm G
        _ ≤ _ := Pouw.Proofs.workWeightedSampling ((Q → R) × Pouw.Lead.Game.Salt L N₀) (P.Wref U.layout) U.layout.N t
          (fun y => correctSet P U y.1 y.2 (A.transcript U y.1 y.2)) εs hεs
    calc _ ≤ Pouw.pr (fun x : (Q → R) × Pouw.Lead.Game.Salt L N₀ × Tk => E1 x ∨ E2 (x.1, x.2.1)) :=
          Pouw.pr_mono fun x hx => by
            by_cases h1 : E1 x
            · exact Or.inl h1
            · refine Or.inr ⟨Pouw.Proofs.wins_of_accept P U x.1 x.2.1 _ hγ hx.2.1 ?_, hx.2.2⟩
              by_contra h2
              exact h1 ⟨hx.1, lt_of_not_ge h2⟩
      _ ≤ Pouw.pr E1 + Pouw.pr (fun x : (Q → R) × Pouw.Lead.Game.Salt L N₀ × Tk => E2 (x.1, x.2.1)) := Pouw.pr_or_le _ _
      _ ≤ (1 - εs) ^ t + U.layout.N * etaLead L :=
          add_le_add hE1 ((Pouw.pr_fst2_le E2).trans (pr_wins_le prc sha C κ htail hgap U B q A hA))

/-- Each unit's tail from `G_tail`, at a certifier whose entropy is the coins (`Compose.certifierCoins`), whose
released coins are the entropy itself. -/
theorem tail_of_G_tail (prc : Prices100) (sha : Hash.Msg → Hash.Digest) (T m n K : ℕ)
    (stream : CoinStream (lead T m n K)) (C : Pouw.Lead.Commit (lead T m n K)) (isA : Hash.Msg → Bool)
    (can real : Strategy (lead T m n K) → Prop)
    (κ : Pouw.Workload → ℕ → Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ)
    (hG : Pouw.Guarantees.Lead.G_tail)
    (hκ : ∀ U u, CommitL1w sha C isA (κ U u) ∧ TileGrain sha C (κ U u) stream real ∧
      HonestHashRuled sha C (κ U u) real ∧ OutsourceWithinEgressLead prc (betaLead (lead T m n K) C) sha C (κ U u) real)
    (h32 : H32 prc) (hW1 : W1 prc real) (hOne : OneInstructionMinimal real) (hCard : CardWithinAccelW1 can real)
    (hSha : ShaWrites isA real) (U : Pouw.Workload) (u : ℕ) (σ : Strategy (lead T m n K)) (hσ : can σ) :
    Pouw.pr (TailEvent prc sha C (κ U u) σ) ≤ etaLead (lead T m n K) :=
  hG prc sha T m n K (κ U u) stream C isA can real Unit (Coins (lead T m n K)) () (Equiv.refl _)
    (fun _ => certCode (Equiv.refl _)) (Compose.certifierCoins _) (hκ U u).1 h32 hW1 (hκ U u).2.1 hOne hCard hSha
    (hκ U u).2.2.1 (hκ U u).2.2.2 ⟨σ, hσ⟩

/-- **`G_γ` for the lead's calls, from `G_tail`** (`Pouw.Guarantees.Lead.GammaFromTail`). -/
theorem GammaFromTail : Pouw.Guarantees.Lead.GammaFromTail := by
  intro Q R _ _ _ prc sha T m n K N₀ stream C isA can real κ hG hκ h32 hW1 hOne hCard hSha hgap
  exact gamma_of_tail Q R (lead T m n K) N₀ prc sha C κ can
    (tail_of_G_tail prc sha T m n K stream C isA can real κ hG hκ h32 hW1 hOne hCard hSha) hgap

/-- **The lead's calls end to end, from `G_tail`** (`Pouw.Guarantees.Lead.EndToEndFromTail`). -/
theorem EndToEndFromTail : Pouw.Guarantees.Lead.EndToEndFromTail := by
  intro Q R _ _ _ prc sha T m n K N₀ stream C isA can real κ hG hκ h32 hW1 hOne hCard hSha hgap U _ B hB q A hA t
    εs hεs
  exact endToEnd_of_tail Q R (lead T m n K) N₀ prc sha C κ can
    (tail_of_G_tail prc sha T m n K stream C isA can real κ hG hκ h32 hW1 hOne hCard hSha) hgap U B hB q A hA t εs
    hεs

end Pouw.SecurityProofs.Lead.Bridge
