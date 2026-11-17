module

public import Proofs.Flock.SamplingGeneric.TwoOffClassAt
public import Proofs.Flock.SamplingGeneric.TwoFirstHitAvg
import Proofs.Flock.Claims.Prob
import Proofs.Flock.SamplingGeneric.RegTrace

/-! The first-hit bounds by class of gates from the prover's average calls, no run held to a budget. -/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims Finset
open Claims.Sampling (AuditStatement Partition InScopeTwo)

namespace Compiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

structure CallsAvg (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} [Fintype Ω] (P : Prover Ω) (x : Msg)
    (needs : ℕ → Finset ℕ) (cls : Fin (L.nG x) → Fin (L.nG x)) (T As Ak Am Ks : ℝ) : Prop where
  /-- Both parties' calls counted. -/
  calls : Ex (fun q : Ω × (Fin (L.V.k x) → Coin) => (L.partiesCalls H P x q.1 q.2 : ℝ)) ≤ T
  sessions : Ex (fun q : Ω × (Fin (L.V.k x) → Coin) =>
    ((L.firstSlots x (L.checked x q.2)).card : ℝ) * (L.partiesCalls H P x q.1 q.2 : ℝ)) ≤ As
  classes : Ex (fun q : Ω × (Fin (L.V.k x) → Coin) =>
    (((L.sampleNeeds x needs (L.checked x q.2)).image cls).card : ℝ) * (L.partiesCalls H P x q.1 q.2 : ℝ)) ≤ Ak
  needed : Ex (fun q : Ω × (Fin (L.V.k x) → Coin) =>
    (L.sessCountIn x needs (fun l => (L.sampleNeeds x needs l).image cls) (L.checked x q.2) : ℝ) *
      (L.partiesCalls H P x q.1 q.2 : ℝ)) ≤ Am
  count : Ex (fun r : Fin (L.V.k x) → Coin =>
    (((L.sampleNeeds x needs (L.checked x r)).image cls).card : ℝ)) ≤ Ks

theorem CallsAvg.of_counts {H : Msg → Msg} {L : Compiled Coin A} {Ω : Type} [Fintype Ω] [Nonempty Ω]
    {P : Prover Ω} {x : Msg} {needs : ℕ → Finset ℕ} {cls : Fin (L.nG x) → Fin (L.nG x)} {T Ns Ks Km : ℝ}
    (hT : Ex (fun q : Ω × (Fin (L.V.k x) → Coin) => (L.partiesCalls H P x q.1 q.2 : ℝ)) ≤ T)
    (hNs : ∀ r : Fin (L.V.k x) → Coin, ((L.firstSlots x (L.checked x r)).card : ℝ) ≤ Ns)
    (hKs : ∀ r : Fin (L.V.k x) → Coin, (((L.sampleNeeds x needs (L.checked x r)).image cls).card : ℝ) ≤ Ks)
    (hKm : ∀ r : Fin (L.V.k x) → Coin,
      (L.sessCountIn x needs (fun l => (L.sampleNeeds x needs l).image cls) (L.checked x r) : ℝ) ≤ Km) :
    L.CallsAvg H P x needs cls T (Ns * T) (Ks * T) (Km * T) Ks where
  calls := hT
  sessions := Ex_mul_le_of_le (fun _ => Nat.cast_nonneg _) (fun q => hNs q.2) (fun _ => Nat.cast_nonneg _) hT
  classes := Ex_mul_le_of_le (fun _ => Nat.cast_nonneg _) (fun q => hKs q.2) (fun _ => Nat.cast_nonneg _) hT
  needed := Ex_mul_le_of_le (fun _ => Nat.cast_nonneg _) (fun q => hKm q.2) (fun _ => Nat.cast_nonneg _) hT
  count := (Ex_mono hKs).trans_eq (Ex_const Ks)

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem runCostAvg_counts (L : Compiled Coin A) (x : Msg) (T Ns : ℝ) :
    L.runCostAvg x T (Ns * T) = T * (1 + L.es x * Ns) + (L.nPU x : ℝ) * L.openCalls x := by
  unfold runCostAvg
  ring

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem firstHitCostAvg_counts (L : Compiled Coin A) (x : Msg) (T Ns Ks Km : ℝ) :
    L.firstHitCostAvg x T (Ns * T) (Ks * T) (Km * T) Ks =
      T * (1 + L.es x * Ns + Ks + L.es x * Km) + (1 + Ks) * ((L.nPU x : ℝ) * L.openCalls x) := by
  unfold firstHitCostAvg
  ring

theorem pr_offRegX_le_cls_avg {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H)
    (hS : L.CostContract H) (hopen : L.OpenCallsLe H) {x : Msg} {s : AuditStatement} (hs : L.parse x = some s)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u) (cls : Fin (L.nG x) → Fin (L.nG x))
    (hallX : ∀ (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g g' : Fin (L.nG x)), cls g = cls g' →
      (L.opensAt H P x a.1 a.2.1 a.2.2 g).isSome → (L.opensAt H P x a.1 a.2.1 a.2.2 g').isSome)
    {T As Ak Am Ks : ℝ} (hC : L.CallsAvg H P x needs cls T As Ak Am Ks)
    (hX : ∀ K : ℕ, ExpectedCR H (L.finderXCls H P x s needs cls K)) :
    Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.OffRegX H P x q.1 q.2.1 q.2.2) ≤
      2 * L.firstHitCostAvg x T As Ak Am Ks / 2 ^ (256 : ℝ) := by
  have hall : ∀ K : ℕ,
      Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.OffRegX H P x q.1 q.2.1 q.2.2) ≤
        2 * Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
            ((L.callsAt H P x s q.1 q.2.1 q.2.2 +
              ∑ c ∈ (univ.filter (fun g : Fin (L.nG x) =>
                  decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u))).image cls,
                L.callsBy H P x s needs c.val q.1 q.2.1 q.2.2 : ℕ) : ℝ)) /
            2 ^ (256 : ℝ) +
          (L.nG x : ℝ) / ((K : ℝ) + 1) :=
    fun K => pr_offRegX_le_cls_cap hbind hs P needs hpos cls hallX K (hX K)
  have h := Verity.FirstHit.le_of_forall_cap hall
  have hc := ex_calls_le_avg_cls hS hopen hs P needs cls hC.calls hC.sessions hC.classes hC.needed hC.count
  exact h.trans (div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hc (by norm_num : (0 : ℝ) ≤ 2))
    (by positivity : (0 : ℝ) ≤ 2 ^ (256 : ℝ)))

set_option synthInstance.maxSize 1024 in
/-- No run of the prover is held to a budget. -/
theorem eventX_le_cls_avg {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) (hS : L.Sessions H)
    (hSK : L.SessionKS H) (hopen : L.OpenCallsLe H) {x : Msg} {s : AuditStatement} {Pt : Partition}
    (hs : L.parse x = some s) (hPt : L.partition s = some Pt) (hin : Sampling.InScope Pt s) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u) (cls : Fin (L.nG x) → Fin (L.nG x))
    (hallX : ∀ (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g g' : Fin (L.nG x)), cls g = cls g' →
      (L.opensAt H P x a.1 a.2.1 a.2.2 g).isSome → (L.opensAt H P x a.1 a.2.1 a.2.2 g').isSome)
    {T As Ak Am Ks : ℝ} (hC : L.CallsAvg H P x needs cls T As Ak Am Ks)
    (hB : ExpectedCR H (BconfX H L Ω P x))
    (hX : ∀ K : ℕ, ExpectedCR H (L.finderXCls H P x s needs cls K)) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.FregX H P x) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) +
        (2 * L.runCostAvg x T As + 2 * L.firstHitCostAvg x T As Ak Am Ks) / 2 ^ (256 : ℝ) := by
  have hI := (pr_inConfX_le hbind hs P).trans
    (bconfX_le_avg hS.toCostContract hopen x P hC.calls hC.sessions hB)
  have hO := (Pr_mono (p := fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      (run H P L.V x q.1 q.2.1).out = .acc ∧ L.OffRegX H P x q.1 q.2.1 q.2.2) fun _ h => h.2).trans
    (pr_offRegX_le_cls_avg hbind hS.toCostContract hopen hs P needs hpos cls hallX hC hX)
  simp only [FregX_eq_FregAt]
  refine (event_le_of_off hS hSK hs hPt hin P (fun _ _ => L.regValX H P x) hI hO).trans ?_
  rw [add_div]
  linarith

set_option synthInstance.maxSize 1024 in
theorem eventX_le_cls_exp {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) (hS : L.Sessions H)
    (hSK : L.SessionKS H) (hopen : L.OpenCallsLe H) {x : Msg} {s : AuditStatement} {Pt : Partition}
    (hs : L.parse x = some s) (hPt : L.partition s = some Pt) (hin : Sampling.InScope Pt s) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u) (cls : Fin (L.nG x) → Fin (L.nG x))
    (hallX : ∀ (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g g' : Fin (L.nG x)), cls g = cls g' →
      (L.opensAt H P x a.1 a.2.1 a.2.2 g).isSome → (L.opensAt H P x a.1 a.2.1 a.2.2 g').isSome)
    {T Ns Ks Km : ℝ}
    (hT : Ex (fun q : Ω × (Fin (L.V.k x) → Coin) => (L.partiesCalls H P x q.1 q.2 : ℝ)) ≤ T)
    (hNs : ∀ r : Fin (L.V.k x) → Coin, ((L.firstSlots x (L.checked x r)).card : ℝ) ≤ Ns)
    (hKs : ∀ r : Fin (L.V.k x) → Coin, (((L.sampleNeeds x needs (L.checked x r)).image cls).card : ℝ) ≤ Ks)
    (hKm : ∀ r : Fin (L.V.k x) → Coin,
      (L.sessCountIn x needs (fun l => (L.sampleNeeds x needs l).image cls) (L.checked x r) : ℝ) ≤ Km)
    (hB : ExpectedCR H (BconfX H L Ω P x))
    (hX : ∀ K : ℕ, ExpectedCR H (L.finderXCls H P x s needs cls K)) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.FregX H P x) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) +
        (2 * (T * (1 + L.es x * Ns) + (L.nPU x : ℝ) * L.openCalls x) +
          2 * (T * (1 + L.es x * Ns + Ks + L.es x * Km) + (1 + Ks) * ((L.nPU x : ℝ) * L.openCalls x))) /
          2 ^ (256 : ℝ) := by
  have h := eventX_le_cls_avg hbind hS hSK hopen hs hPt hin P needs hpos cls hallX
    (CallsAvg.of_counts hT hNs hKs hKm) hB hX
  rw [runCostAvg_counts, firstHitCostAvg_counts] at h
  exact h

end Compiled

namespace CompiledTwo

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

theorem pr_offIreg_le_cls_avg {H : Msg → Msg} {L : CompiledTwo Coin A} (hbind : L.commit.Binds H)
    (hS : L.Sessions H) (hopen : L.toCompiled.OpenCallsLe H) {x : Msg} {s : AuditStatement}
    (hs : L.parse x = some s) (Pρ : Partition) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (needs : ℕ → Finset ℕ) (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u)
    (cls : Fin (L.nG x) → Fin (L.nG x))
    (hallI : ∀ (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g g' : Fin (L.nG x)),
      cls g = cls g' → (L.iopTable H P x t.1 Pρ t.2 ec g).isSome → (L.iopTable H P x t.1 Pρ t.2 ec g').isSome)
    {T As Ak Am Ks : ℝ} (hC : L.toCompiled.CallsAvg H P x needs cls T As Ak Am Ks)
    (hI : ∀ K : ℕ, ExpectedCR H (L.finderICls H P x s Pρ needs cls K)) :
    Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.OffIreg H P x q.1 Pρ q.2.1 q.2.2) ≤
      2 * L.toCompiled.firstHitCostAvg x T As Ak Am Ks / 2 ^ (256 : ℝ) := by
  have hall : ∀ K : ℕ,
      Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.OffIreg H P x q.1 Pρ q.2.1 q.2.2) ≤
        2 * Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
            ((L.callsAt H P x s q.1 q.2.1 q.2.2 +
              ∑ c ∈ (univ.filter (fun g : Fin (L.nG x) =>
                  decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u))).image cls,
                L.toCompiled.callsBy H P x s needs c.val q.1 q.2.1 q.2.2 : ℕ) : ℝ)) /
            2 ^ (256 : ℝ) +
          (L.nG x : ℝ) / ((K : ℝ) + 1) :=
    fun K => pr_offIreg_le_cls_cap hbind hS hs Pρ P needs hpos cls hallI K (hI K)
  have h := Verity.FirstHit.le_of_forall_cap hall
  have hc := Compiled.ex_calls_le_avg_cls hS.toCostContract hopen hs P needs cls hC.calls hC.sessions hC.classes
    hC.needed hC.count
  exact h.trans (div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hc (by norm_num : (0 : ℝ) ≤ 2))
    (by positivity : (0 : ℝ) ≤ 2 ^ (256 : ℝ)))

noncomputable def εclsAvg (H : Msg → Msg) (L : CompiledTwo Coin A) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (T As Ak Am Ks : ℝ) : ℝ :=
  L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) +
    (2 * L.toCompiled.runCostAvg x T As + 4 * L.toCompiled.firstHitCostAvg x T As Ak Am Ks) / 2 ^ (256 : ℝ)

/-- No run of the prover is held to a budget: `eventTwoX_le_cls_at_budget`'s `T` bounds every run's calls, this one's
their average. -/
theorem eventTwoX_le_cls_avg {H : Msg → Msg} {L : CompiledTwo Coin A} (hA : A.KS H)
    (hbind : L.commit.Binds H) (hS : L.Sessions H) (hopen : L.toCompiled.OpenCallsLe H) {x : Msg}
    {s : AuditStatement} {Pt : Partition} {d : Sampling.Design.Two} (hs : L.parse x = some s)
    (hPt : L.partition s = some Pt) (hd : L.two s = some d) (hin : InScopeTwo Pt d s) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u) (cls : Fin (L.nG x) → Fin (L.nG x))
    (hrep : ∀ u (g : Fin (L.nG x)), g.val ∈ needs u → (cls g).val ∈ needs u)
    (hallX : ∀ (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g g' : Fin (L.nG x)), cls g = cls g' →
      (L.opensAt H P x a.1 a.2.1 a.2.2 g).isSome → (L.opensAt H P x a.1 a.2.1 a.2.2 g').isSome)
    (hallI : ∀ (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g g' : Fin (L.nG x)),
      cls g = cls g' → (L.iopTable H P x t.1 d.part t.2 ec g).isSome →
        (L.iopTable H P x t.1 d.part t.2 ec g').isSome)
    {T As Ak Am Ks : ℝ} (hC : L.toCompiled.CallsAvg H P x needs cls T As Ak Am Ks)
    (hB : ExpectedCR H (Compiled.BconfX H L.toCompiled Ω P x))
    (hX : ∀ K : ℕ, ExpectedCR H (L.toCompiled.finderXCls H P x s needs cls K))
    (hI : ∀ K : ℕ, ExpectedCR H (L.finderICls H P x s d.part needs cls K)) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.FregTwoX H P x p.1 p.2) ≤ L.εclsAvg H x P T As Ak Am Ks := by
  have h := eventTwoX_le_firstHit_cls hA hbind hS hs hPt hd hin P needs hpos cls hrep hallX hallI hX hI
  have hc := Compiled.ex_calls_le_avg_cls hS.toCostContract hopen hs P needs cls hC.calls hC.sessions hC.classes
    hC.needed hC.count
  have hdiv := div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hc (by norm_num : (0 : ℝ) ≤ 4))
    (by positivity : (0 : ℝ) ≤ 2 ^ (256 : ℝ))
  have hb := Compiled.bconfX_le_avg hS.toCostContract hopen x P hC.calls hC.sessions hB
  unfold εclsAvg
  rw [add_div]
  exact (h.trans (add_le_add (le_refl _) hdiv)).trans
    ((add_le_add (add_le_add (le_refl _) hb) (le_refl _)).trans (le_of_eq (add_assoc _ _ _)))

theorem eventTwoX_le_cls_exp {H : Msg → Msg} {L : CompiledTwo Coin A} (hA : A.KS H)
    (hbind : L.commit.Binds H) (hS : L.Sessions H) (hopen : L.toCompiled.OpenCallsLe H) {x : Msg}
    {s : AuditStatement} {Pt : Partition} {d : Sampling.Design.Two} (hs : L.parse x = some s)
    (hPt : L.partition s = some Pt) (hd : L.two s = some d) (hin : InScopeTwo Pt d s) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u) (cls : Fin (L.nG x) → Fin (L.nG x))
    (hrep : ∀ u (g : Fin (L.nG x)), g.val ∈ needs u → (cls g).val ∈ needs u)
    (hallX : ∀ (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g g' : Fin (L.nG x)), cls g = cls g' →
      (L.opensAt H P x a.1 a.2.1 a.2.2 g).isSome → (L.opensAt H P x a.1 a.2.1 a.2.2 g').isSome)
    (hallI : ∀ (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g g' : Fin (L.nG x)),
      cls g = cls g' → (L.iopTable H P x t.1 d.part t.2 ec g).isSome →
        (L.iopTable H P x t.1 d.part t.2 ec g').isSome)
    {T Ns Ks Km : ℝ}
    (hT : Ex (fun q : Ω × (Fin (L.V.k x) → Coin) => (L.toCompiled.partiesCalls H P x q.1 q.2 : ℝ)) ≤ T)
    (hNs : ∀ r : Fin (L.V.k x) → Coin, ((L.toCompiled.firstSlots x (L.checked x r)).card : ℝ) ≤ Ns)
    (hKs : ∀ r : Fin (L.V.k x) → Coin,
      (((L.toCompiled.sampleNeeds x needs (L.checked x r)).image cls).card : ℝ) ≤ Ks)
    (hKm : ∀ r : Fin (L.V.k x) → Coin,
      (L.toCompiled.sessCountIn x needs (fun l => (L.toCompiled.sampleNeeds x needs l).image cls)
        (L.checked x r) : ℝ) ≤ Km)
    (hB : ExpectedCR H (Compiled.BconfX H L.toCompiled Ω P x))
    (hX : ∀ K : ℕ, ExpectedCR H (L.toCompiled.finderXCls H P x s needs cls K))
    (hI : ∀ K : ℕ, ExpectedCR H (L.finderICls H P x s d.part needs cls K)) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.FregTwoX H P x p.1 p.2) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) +
        (2 * (T * (1 + L.es x * Ns) + (L.nPU x : ℝ) * L.openCalls x) +
          4 * (T * (1 + L.es x * Ns + Ks + L.es x * Km) + (1 + Ks) * ((L.nPU x : ℝ) * L.openCalls x))) /
          2 ^ (256 : ℝ) := by
  have h := eventTwoX_le_cls_avg hA hbind hS hopen hs hPt hd hin P needs hpos cls hrep hallX hallI
    (Compiled.CallsAvg.of_counts hT hNs hKs hKm) hB hX hI
  unfold εclsAvg at h
  rw [Compiled.runCostAvg_counts, Compiled.firstHitCostAvg_counts] at h
  exact h

end CompiledTwo

end Claims.SamplingGeneric
