import Models.Accounting.Guarantees.PerRun
import Proofs.Accounting.PerRunJoint
import Proofs.Accounting.PerRunTruncation
import Proofs.Pouw.Sampling
import Proofs.Pouw.PearlC.W1BudgetAbort

/-!
# Proof of the priced per-run guarantees

Pearl-C's game with `t` tickets weighted by `w`, at units whose `W_ref` is `κ·w`: an accepted run whose correct units
carry at least `1 − ε_s` of the tickets' weight carry at least `1 − ε_s` of `W_ref`, so a cost below
`(1 − γ)(1 − ε_s)·∑ W_ref` wins the γ-gap game (`winsPearlC_of_accept`). Work-weighted sampling and `Gγ` bound the two
(`pearlC_endToEnd`, every adversary within a sure budget), and truncation (`pr_lt_of_abort`, with `abortPearlC`) makes it
per run. A run that realizes Pearl-C's game with its cost covered by the priced sum has the compute audit's per-run
guarantee on that sum.

When the reads are attributed, the bits read over the window are the bandwidth account's, the compute account's and the
rest (`Run.read_add_pouwRead_add_covertRead`), within the device's `BW·T`. So, beside the compute account's work and
every other account's within `F·T`, a run whose priced sum and window reads reach their certificates leaves the other
accounts at most `(F + pRead·BW)·T − (W + pRead·Q)` of priced work and reads (`Run.pricedLt_of_covert`).

`BudgetAbortPearlC` is compute-accounting's `Pouw.PearlC.W1BudgetAbort.BudgetAbortPearlC` (`budgetAbortPearlC_iff`), and a
cost model whose costs and calls don't depend on the coins satisfies it (`budgetAbortPearlC_of_oblivious`).
-/

namespace Accounting

open Pouw

variable {Q R S : Type}

theorem budgetAbortPearlC_iff (CM : PearlC.CostModel Q R S) :
    BudgetAbortPearlC CM ↔ PearlC.W1BudgetAbort.BudgetAbortPearlC CM :=
  Iff.rfl

/-- A Pearl-C cost model whose programs' costs and calls do not depend on the oracle, the preprocessed state or the salt,
with nonnegative costs and a program that costs nothing and calls nothing, truncates, so `BudgetAbortPearlC` is
satisfiable. -/
theorem budgetAbortPearlC_of_oblivious (CM : PearlC.CostModel Q R S)
    (hnn : ∀ prog H U act σ s, 0 ≤ CM.cost prog H U act σ s)
    (hobl : ∀ prog U act, ∃ c : ℝ, ∃ n : ℕ, ∀ H σ s, CM.cost prog H U act σ s = c ∧ CM.calls prog H U act σ s = n)
    (null : CM.Prog) (hnull : ∀ H U act σ s, CM.cost null H U act σ s = 0 ∧ CM.calls null H U act σ s = 0) :
    BudgetAbortPearlC CM := by
  classical
  refine ⟨hnn, fun prog U act b hb => ?_⟩
  obtain ⟨c, n, hc⟩ := hobl prog U act
  by_cases hcb : c ≤ b
  · exact ⟨prog, fun H σ s => ⟨(hc H σ s).1 ▸ hcb, le_rfl, fun _ => rfl⟩⟩
  · refine ⟨null, fun H σ s => ⟨(hnull H U act σ s).1 ▸ hb, (hnull H U act σ s).2 ▸ Nat.zero_le _, fun h => ?_⟩⟩
    exact absurd ((hc H σ s).1 ▸ h) hcb

/-- **Pearl-C's adversaries are closed under budget-abort** when its cost model truncates. -/
theorem abortPearlC {CM : PearlC.CostModel Q R S} (hCM : BudgetAbortPearlC CM) (A : PearlC.Adversary CM)
    (U : PearlC.Workload) {q : ℕ} (hq : CallsWithinPearlC A U q) {b : ℝ} (hb : 0 ≤ b) :
    ∃ A' : PearlC.Adversary CM, A'.Bounded U b q ∧
      ∀ H s, onlineCostPearlC A U H s ≤ b → A'.transcript U H s = A.transcript U H s := by
  obtain ⟨prog', h⟩ := hCM.2 A.online U A.act b hb
  refine ⟨⟨A.pre, prog', A.act⟩, fun H s => ⟨(h H _ s).1, ?_⟩, fun H s hc => (h H _ s).2.2 hc⟩
  exact (Nat.add_le_add_left (h H _ s).2.1 _).trans (hq H s)

/-- A run whose correct units carry all but `ε_s` of the tickets' weight, at units whose `W_ref` is `κ·w`, and whose
cost is below `(1 − γ)(1 − ε_s)·∑ W_ref`, wins Pearl-C's γ-gap game. -/
theorem winsPearlC_of_accept (P : PearlC.Protocol Q R S) (U : PearlC.Workload) (H : Q → R) (s : S)
    (τ : PearlC.Transcript) {w : ℕ → ℕ} {κ γ T εs : ℝ} (hκ : 0 ≤ κ)
    (hw : ∀ u < U.layout.N, P.Wref U.layout u = κ * w u) (hγ : γ < 1)
    (hT : T < (1 - γ) * (1 - εs) * ∑ u ∈ Finset.range U.layout.N, P.Wref U.layout u)
    (hbad : ∑ u ∈ Finset.range U.layout.N \ PearlC.correctSet P U H s τ, (w u : ℝ) ≤
      εs * ∑ u ∈ Finset.range U.layout.N, (w u : ℝ)) :
    PearlC.Wins P U γ T τ H s := by
  classical
  have hK : PearlC.correctSet P U H s τ ⊆ Finset.range U.layout.N := Finset.filter_subset _ _
  have hsum : ∀ A ⊆ Finset.range U.layout.N, ∑ u ∈ A, P.Wref U.layout u = κ * ∑ u ∈ A, (w u : ℝ) := by
    intro A hA
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun u hu => hw u (Finset.mem_range.1 (hA hu))
  have hsplit := Finset.sum_sdiff (f := fun u => P.Wref U.layout u) hK
  have hbadW : ∑ u ∈ Finset.range U.layout.N \ PearlC.correctSet P U H s τ, P.Wref U.layout u ≤
      εs * ∑ u ∈ Finset.range U.layout.N, P.Wref U.layout u := by
    rw [hsum _ Finset.sdiff_subset, hsum _ subset_rfl]
    calc κ * _ ≤ κ * (εs * ∑ u ∈ Finset.range U.layout.N, (w u : ℝ)) := mul_le_mul_of_nonneg_left hbad hκ
      _ = _ := by ring
  have hgood : (1 - εs) * ∑ u ∈ Finset.range U.layout.N, P.Wref U.layout u ≤
      ∑ u ∈ PearlC.correctSet P U H s τ, P.Wref U.layout u := by
    linarith
  have h1γ : 0 < 1 - γ := by linarith
  unfold PearlC.Wins
  rw [div_lt_iff₀ h1γ]
  calc T < (1 - γ) * (1 - εs) * ∑ u ∈ Finset.range U.layout.N, P.Wref U.layout u := hT
    _ = (1 - γ) * ((1 - εs) * ∑ u ∈ Finset.range U.layout.N, P.Wref U.layout u) := by ring
    _ ≤ (1 - γ) * ∑ u ∈ PearlC.correctSet P U H s τ, P.Wref U.layout u := mul_le_mul_of_nonneg_left hgood h1γ.le
    _ = _ := by ring

/-- **Pearl-C's γ-gap property, audited by work-weighted tickets**: an adversary within a sure budget `T` is accepted with
`T < (1 − γ)(1 − ε_s)·∑ W_ref` with probability at most `(1 − ε_s)^t + η(q, N)`. -/
theorem pearlC_endToEnd [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] {CM : PearlC.CostModel Q R S}
    {P : PearlC.Protocol Q R S} {D : PearlC.Layout → Prop} {γ : ℝ} {η : ℕ → ℕ → ℝ} (hγ : γ < 1)
    (hG : PearlC.Gγ CM P D γ η) {U : PearlC.Workload} (hU : U.InDomain P D) {w : ℕ → ℕ} {κ : ℝ} (hκ : 0 ≤ κ)
    (hw : ∀ u < U.layout.N, P.Wref U.layout u = κ * w u) {T : ℝ} (hT : 0 ≤ T) {q : ℕ} (hq : q ≤ 2 ^ 64)
    {A : PearlC.Adversary CM} (hA : A.Bounded U T q) (t : ℕ) {εs : ℝ} (hεs : εs ≤ 1) :
    pr (fun x : PearlCCoins Q R S U w t =>
        AuditAccepts (PearlC.correctSet P U x.1 x.2.1 (A.transcript U x.1 x.2.1)) x.2.2 ∧
        T < (1 - γ) * (1 - εs) * ∑ u ∈ Finset.range U.layout.N, P.Wref U.layout u)
      ≤ (1 - εs) ^ t + η q U.layout.N := by
  let K := fun y : (Q → R) × S => PearlC.correctSet P U y.1 y.2 (A.transcript U y.1 y.2)
  let G := fun y : ((Q → R) × S) × (Fin t → Ticket w U.layout.N) => AuditAccepts (K y.1) y.2 ∧
    εs * ∑ u ∈ Finset.range U.layout.N, (w u : ℝ) < ∑ u ∈ Finset.range U.layout.N \ K y.1, (w u : ℝ)
  let E1 := fun x : PearlCCoins Q R S U w t => G ((x.1, x.2.1), x.2.2)
  let E2 := fun y : (Q → R) × S => PearlC.Wins P U γ T (A.transcript U y.1 y.2) y.1 y.2
  have h1 : pr E1 ≤ (1 - εs) ^ t :=
    (pr_equiv (Equiv.prodAssoc _ _ _).symm G).le.trans
      (Pouw.Proofs.workWeightedSampling ((Q → R) × S) w U.layout.N t K εs hεs)
  calc _ ≤ pr (fun x : PearlCCoins Q R S U w t => E1 x ∨ E2 (x.1, x.2.1)) := pr_mono fun x hx => by
        by_cases hE : E1 x
        · exact Or.inl hE
        · refine Or.inr (winsPearlC_of_accept P U x.1 x.2.1 _ hκ hw hγ hx.2 ?_)
          by_contra h2
          exact hE ⟨hx.1, lt_of_not_ge h2⟩
    _ ≤ pr E1 + pr (fun x : PearlCCoins Q R S U w t => E2 (x.1, x.2.1)) := pr_or_le _ _
    _ ≤ (1 - εs) ^ t + η q U.layout.N := add_le_add h1 ((pr_fst2_le E2).trans (hG U hU T hT q hq A hA))

namespace Run

variable {ι : Type} {d : Device ι} {α ΩW ΩQ ΩS : Type} [Fintype α] (r : Run d α ΩW ΩQ ΩS)

section Sure

variable [Fintype ΩW]

/-- A compute guarantee on the work alone gives the priced one at any read price `pRead ≥ 0`. -/
theorem computeSoundPerRunPriced_of_perRun {p : Price} {pRead W ε : ℝ} (h0 : 0 ≤ pRead)
    (h : r.ComputeSoundPerRun p W ε) : r.ComputeSoundPerRunPriced p pRead W ε := fun o =>
  (pr_mono fun _ hx => ⟨hx.1, lt_of_le_of_lt (le_add_of_nonneg_right (mul_nonneg h0 (Nat.cast_nonneg _))) hx.2⟩).trans
    (h o)

/-- An ideal compute audit, accepting only runs whose priced sum reaches `W`, has its priced per-run guarantee with
error 0. -/
theorem computeSoundPerRunPriced_ideal {p : Price} {pRead W : ℝ}
    (h : ∀ ω, r.accepts .compute ω → W ≤ (r.work p ω : ℝ) + pRead * r.pouwRead ω) :
    r.ComputeSoundPerRunPriced p pRead W 0 := fun _ =>
  (pr_eq_zero fun _ hx => absurd (h _ hx.1) (not_le.2 hx.2)).le

variable [Fintype ΩQ] [Fintype ΩS]

theorem pr_computePricedLt_le [Nonempty ΩQ] [Nonempty ΩS] {p : Price} {pRead W ε : ℝ}
    (h : r.ComputeSoundPerRunPriced p pRead W ε) :
    pr (fun ω => r.accepts .compute ω ∧ (r.work p ω : ℝ) + pRead * r.pouwRead ω < W) ≤ ε := by
  have := pr_prod_le (fun y : (ΩQ × ΩS) × ΩW =>
      r.accepts .compute (y.2, y.1) ∧ (r.work p (y.2, y.1) : ℝ) + pRead * r.pouwRead (y.2, y.1) < W) ε
    ((pr_nonneg _).trans (h (Classical.arbitrary _))) h
  rw [← pr_equiv (Equiv.prodComm ΩW (ΩQ × ΩS))] at this
  exact this

theorem jointPerRunPriced [Nonempty ΩW] [Nonempty ΩQ] [Nonempty ΩS] {p : Price} {pRead W Q S εW εQ εS : ℝ}
    (hW : r.ComputeSoundPerRunPriced p pRead W εW) (hQ : r.BandwidthSoundPerRun Q εQ)
    (hS : r.CapacitySoundPerRun S εS) :
    pr (fun ω => r.Accepted ω ∧
        ((r.work p ω : ℝ) + pRead * r.pouwRead ω < W ∨ (r.read ω : ℝ) < Q ∨
          (r.held ω : ℝ) < S * (r.win .capacity).len))
      ≤ εW + εQ + εS := by
  calc _ ≤ pr (fun ω => (r.accepts .compute ω ∧ (r.work p ω : ℝ) + pRead * r.pouwRead ω < W) ∨
          (r.accepts .bandwidth ω ∧ (r.read ω : ℝ) < Q) ∨
          (r.accepts .capacity ω ∧ (r.held ω : ℝ) < S * (r.win .capacity).len)) :=
        pr_mono fun ω h => h.2.imp (⟨h.1 .compute, ·⟩) (Or.imp (⟨h.1 .bandwidth, ·⟩) (⟨h.1 .capacity, ·⟩))
    _ ≤ εW + (εQ + εS) :=
        (pr_or_le _ _).trans (add_le_add (r.pr_computePricedLt_le hW)
          ((pr_or_le _ _).trans (add_le_add (r.pr_bandwidthLt_le hQ) (r.pr_capacityLt_le hS))))
    _ = εW + εQ + εS := (add_assoc _ _ _).symm

end Sure

section Covert

theorem totalRead_le (ω : ΩW × ΩQ × ΩS) : r.totalRead ω ≤ d.bw * (r.win .bandwidth).len :=
  Schedule.sum_read_le (r.fits ω) _

variable [DecidableEq α]

/-- **No bit read is credited twice**: when the reads are attributed, the bits read over the window are the bandwidth
account's, the compute account's and the other accounts'. -/
theorem read_add_pouwRead_add_covertRead (h : r.ReadsAttributed) (ω : ΩW × ΩQ × ΩS) :
    r.read ω + r.pouwRead ω + r.covertRead ω = r.totalRead ω := by
  have hc : r.acct .compute ∈ Finset.univ.erase (r.acct .bandwidth) := Finset.mem_erase.2 ⟨h.1, Finset.mem_univ _⟩
  have h1 := Finset.add_sum_erase (Finset.univ.erase (r.acct .bandwidth))
    (fun a => (r.schedule ω).read (r.win .bandwidth) a) hc
  have h2 := Finset.add_sum_erase Finset.univ (fun a => (r.schedule ω).read (r.win .bandwidth) a)
    (Finset.mem_univ (r.acct .bandwidth))
  unfold read pouwRead covertRead totalRead
  rw [h.2, add_assoc, h1, h2]

/-- On a run whose bandwidth account reads at least `Q` bits, the other accounts beyond the priced budget
`(F + pRead·BW)·T − (W + pRead·Q)` put the priced sum below `W`. -/
theorem pricedLt_of_covert (hA : r.ReadsAttributed) {p : Price} {pRead W Q : ℝ} (h0 : 0 ≤ pRead)
    {ω : ΩW × ΩQ × ΩS} (hQ : Q ≤ r.read ω)
    (h : ((d.priced p * (r.win .compute).len : ℕ) : ℝ) + pRead * ((d.bw * (r.win .compute).len : ℕ) : ℝ) -
      (W + pRead * Q) < r.othersWork p ω + pRead * r.covertRead ω) :
    (r.work p ω : ℝ) + pRead * r.pouwRead ω < W := by
  have hw : ((r.work p ω : ℕ) : ℝ) + r.othersWork p ω ≤ ((d.priced p * (r.win .compute).len : ℕ) : ℝ) := by
    exact_mod_cast r.work_add_othersWork p ω
  have hr : ((r.read ω : ℕ) : ℝ) + r.pouwRead ω + r.covertRead ω ≤ ((d.bw * (r.win .compute).len : ℕ) : ℝ) := by
    have := (r.read_add_pouwRead_add_covertRead hA ω).symm ▸ r.totalRead_le ω
    rw [← hA.2] at this
    exact_mod_cast this
  have hcov : (r.covertRead ω : ℝ) ≤ ((d.bw * (r.win .compute).len : ℕ) : ℝ) - r.read ω - r.pouwRead ω := by
    linarith
  have m1 := mul_le_mul_of_nonneg_left hcov h0
  have m2 := mul_le_mul_of_nonneg_left hQ h0
  nlinarith

variable [Fintype ΩW] [Fintype ΩQ] [Fintype ΩS] [Nonempty ΩW] [Nonempty ΩQ] [Nonempty ΩS]

theorem covertResidualPerRunPriced {p : Price} {pRead W Q S εW εQ εS : ℝ} (h0 : 0 ≤ pRead)
    (hA : r.ReadsAttributed) (hW : r.ComputeSoundPerRunPriced p pRead W εW) (hQ : r.BandwidthSoundPerRun Q εQ)
    (hS : r.CapacitySoundPerRun S εS) :
    pr (fun ω => r.Accepted ω ∧
        (((d.priced p * (r.win .compute).len : ℕ) : ℝ) + pRead * ((d.bw * (r.win .compute).len : ℕ) : ℝ) -
            (W + pRead * Q) < r.othersWork p ω + pRead * r.covertRead ω ∨
          ((d.mem : ℝ) - S) * (r.win .capacity).len < r.othersHeld ω))
      ≤ εW + εQ + εS := by
  refine (pr_mono fun ω h => ⟨h.1, ?_⟩).trans (r.jointPerRunPriced hW hQ hS)
  rcases h.2 with hc | hc
  · by_cases hq : (r.read ω : ℝ) < Q
    · exact Or.inr (Or.inl hq)
    · exact Or.inl (r.pricedLt_of_covert hA h0 (le_of_not_gt hq) hc)
  · exact Or.inr (Or.inr (r.heldLt_of_othersHeld hc))

end Covert

end Run

namespace SecurityProofs

theorem PearlCEndToEndPerRun : Accounting.Guarantees.PearlCEndToEndPerRun := by
  intro Q R S _ _ _ _ CM P D γ η hγ hG hCM U hU w κ hκ hw q hq A hA t εs hεs
  refine pr_lt_of_abort
    (fun (A' : PearlC.Adversary CM) (x : PearlCCoins Q R S U w t) =>
      AuditAccepts (PearlC.correctSet P U x.1 x.2.1 (A'.transcript U x.1 x.2.1)) x.2.2)
    (fun A' T => A'.Bounded U T q) A (fun x => onlineCostPearlC A U x.1 x.2.1) _ _ (fun x => hCM.1 _ _ _ _ _ _)
    (fun A' T hT hA' => pearlC_endToEnd hγ hG hU hκ hw hT hq hA' t hεs)
    fun b hb => ?_
  obtain ⟨A', hA', htr⟩ := abortPearlC hCM A U hA hb
  exact ⟨A', hA', fun x hx hacc => (htr x.1 x.2.1 hx).symm ▸ hacc⟩

theorem ComputeOfPearlCPerRunPriced : Accounting.Guarantees.ComputeOfPearlCPerRunPriced := by
  intro Q R S _ _ _ _ CM P D γ η hγ hG hCM U hU w κ hκ hw t εs hεs hne ι α ΩQ ΩS _ _ _ d r p pRead hq real o
  obtain ⟨⟨_, _, tix⟩⟩ := hne
  have hA : CallsWithinPearlC (real.adv o) U (d.rate .oracle * (r.win .compute).len) := fun H s =>
    (real.calls_le (H, s, tix) o).trans (Schedule.ops_le (r.fits _) _ _ _)
  refine le_trans (pr_mono fun x hx => ?_)
    (PearlCEndToEndPerRun Q R S CM P D γ η hγ hG hCM U hU w κ hκ hw _ hq (real.adv o) hA t εs hεs)
  exact ⟨(real.accepts_iff x o).1 hx.1, (real.cost_le x o).trans_lt hx.2⟩

theorem JointPerRunPriced : Accounting.Guarantees.JointPerRunPriced := by
  intro ι α ΩW ΩQ ΩS _ _ _ _ _ _ _ _ d r p pRead W Q S εW εQ εS hA hW hQ hS
  exact ⟨r.read_add_pouwRead_add_covertRead hA, r.jointPerRunPriced hW hQ hS⟩

theorem CovertResidualPerRunPriced : Accounting.Guarantees.CovertResidualPerRunPriced := by
  intro ι α ΩW ΩQ ΩS _ _ _ _ _ _ _ _ d r p pRead W Q S εW εQ εS h0 hA hW hQ hS
  exact r.covertResidualPerRunPriced h0 hA hW hQ hS

end SecurityProofs

end Accounting
