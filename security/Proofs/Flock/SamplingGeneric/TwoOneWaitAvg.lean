module

public import Proofs.Flock.SamplingGeneric.TwoOneWaitAt
public import Proofs.Flock.SamplingGeneric.TwoOffClassAvg
import Proofs.Flock.Claims.Prob

/-! The two-stage first-hit bound on the one-wait extraction, at the prover's expected calls. -/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims Finset
open Claims.Sampling (AuditStatement Partition InScopeTwo)

namespace CompiledTwo

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

noncomputable def εclsOneWaitAvg (H : Msg → Msg) (L : CompiledTwo Coin A) (x : Msg) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) (T As Ak Am Ks : ℝ) : ℝ :=
  3 * L.κs x + 2 * Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) +
    2 * (2 * L.oneWait.toCompiled.runCostAvg x T As + 4 * L.oneWait.toCompiled.firstHitCostAvg x T As Ak Am Ks) /
      2 ^ (256 : ℝ)

/-- No run of the prover is held to a budget. -/
theorem eventTwoX_oneWait_le_cls_avg {H : Msg → Msg} {L : CompiledTwo Coin A} (hF : A.OneWaitFinds H)
    (hA : A.KS H) (hbind : L.commit.Binds H) (hS : L.Sessions H) (hopen : L.toCompiled.OpenCallsLe H) {x : Msg}
    {s : AuditStatement} {Pt : Partition} {d : Sampling.Design.Two} (hs : L.parse x = some s)
    (hPt : L.partition s = some Pt) (hd : L.two s = some d) (hin : InScopeTwo Pt d s) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u) (cls : Fin (L.nG x) → Fin (L.nG x))
    (hallX : ∀ (a : Ω × ((Fin (L.V.k x) → Coin) × L.oneWait.SlotEC x)) (g g' : Fin (L.nG x)), cls g = cls g' →
      (L.oneWait.opensAt H P x a.1 a.2.1 a.2.2 g).isSome → (L.oneWait.opensAt H P x a.1 a.2.1 a.2.2 g').isSome)
    (hallI : ∀ (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.oneWait.SlotEC x)
      (g g' : Fin (L.nG x)), cls g = cls g' → (L.oneWait.iopTable H P x t.1 d.part t.2 ec g).isSome →
        (L.oneWait.iopTable H P x t.1 d.part t.2 ec g').isSome)
    {T As Ak Am Ks : ℝ} (hC : L.toCompiled.CallsAvg H P x needs cls T As Ak Am Ks)
    (hB : ExpectedCR H (Compiled.BconfX H L.oneWait.toCompiled Ω P x))
    (hX : ∀ K : ℕ, ExpectedCR H (L.oneWait.toCompiled.finderXCls H P x s needs cls K))
    (hI : ∀ K : ℕ, ExpectedCR H (L.oneWait.finderICls H P x s d.part needs cls K)) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.oneWait.FregTwoX H P x p.1 p.2) ≤ L.εclsOneWaitAvg H x P T As Ak Am Ks := by
  have hS₁ : L.oneWait.Sessions H := hS.oneWait
  have hopen₁ : L.oneWait.toCompiled.OpenCallsLe H := hopen.oneWait
  have key : ∀ a b c e n k : ℝ,
      a ≤ b + 2 * (c + (2 * e / 2 ^ (256 : ℝ) + n / k) + (2 * e / 2 ^ (256 : ℝ) + n / k)) →
        a ≤ b + 2 * c + 8 * e / 2 ^ (256 : ℝ) + (4 * n) / k :=
    fun a b c e n k h => h.trans (le_of_eq (by ring))
  have hcap : ∀ K : ℕ,
      Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
          ∃ u ∈ L.checked x p.2, u ∈ L.oneWait.FregTwoX H P x p.1 p.2) ≤
        3 * L.κs x + 2 * Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) +
            2 * (Compiled.BconfX H L.oneWait.toCompiled Ω P x).Pr H +
          8 * Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.oneWait.SlotEC x) =>
            ((L.oneWait.callsAt H P x s q.1 q.2.1 q.2.2 +
              ∑ c ∈ (univ.filter (fun g : Fin (L.nG x) =>
                  decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u))).image cls,
                L.oneWait.toCompiled.callsBy H P x s needs c.val q.1 q.2.1 q.2.2 : ℕ) : ℝ)) /
            2 ^ (256 : ℝ) +
          (4 * (L.nG x : ℝ)) / ((K : ℝ) + 1) := by
    intro K
    have hB' := (Pr_mono (p := fun q : Ω × ((Fin (L.V.k x) → Coin) × L.oneWait.SlotEC x) =>
        (run H P L.V x q.1 q.2.1).out = .acc ∧ L.oneWait.OffRegX H P x q.1 q.2.1 q.2.2) fun _ h => h.2).trans
      (Compiled.pr_offRegX_le_cls_cap (L := L.oneWait.toCompiled) hbind hs P needs hpos cls hallX K (hX K))
    have hN' := (Pr_mono (p := fun q : Ω × ((Fin (L.V.k x) → Coin) × L.oneWait.SlotEC x) =>
        (run H P L.V x q.1 q.2.1).out = .acc ∧ L.oneWait.OffIreg H P x q.1 d.part q.2.1 q.2.2) fun _ h => h.2).trans
      (pr_offIreg_le_cls_cap (L := L.oneWait) hbind hS₁ hs d.part P needs hpos cls hallI K (hI K))
    exact key _ _ _ _ _ _ (eventTwoX_oneWait_le_of_off hF hA hbind hS hs hPt hd hin P hB' hN')
  have h := Verity.FirstHit.le_of_forall_cap hcap
  have hC₁ : L.oneWait.toCompiled.CallsAvg H P x needs cls T As Ak Am Ks :=
    ⟨hC.calls, hC.sessions, hC.classes, hC.needed, hC.count⟩
  have hc := Compiled.ex_calls_le_avg_cls (L := L.oneWait.toCompiled) hS₁.toCostContract hopen₁ hs P needs cls
    hC₁.calls hC₁.sessions hC₁.classes hC₁.needed hC₁.count
  have hb := Compiled.bconfX_le_avg (L := L.oneWait.toCompiled) hS₁.toCostContract hopen₁ x P hC₁.calls
    hC₁.sessions hB
  have key2 : ∀ a b c e r f p : ℝ, a ≤ b + 2 * c + 8 * e / p → c ≤ 2 * r / p → e ≤ f → 0 < p →
      a ≤ b + 2 * (2 * r + 4 * f) / p := by
    intro a b c e r f p h1 h2 h3 hp
    have h4 : 8 * e / p ≤ 8 * f / p := div_le_div_of_nonneg_right (by linarith) hp.le
    have h5 : b + 2 * (2 * r + 4 * f) / p = b + 2 * (2 * r / p) + 8 * f / p := by ring
    rw [h5]
    linarith
  exact key2 _ _ _ _ _ _ _ h hb hc (by positivity)

theorem eventTwoX_oneWait_le_cls_exp {H : Msg → Msg} {L : CompiledTwo Coin A} (hF : A.OneWaitFinds H)
    (hA : A.KS H) (hbind : L.commit.Binds H) (hS : L.Sessions H) (hopen : L.toCompiled.OpenCallsLe H) {x : Msg}
    {s : AuditStatement} {Pt : Partition} {d : Sampling.Design.Two} (hs : L.parse x = some s)
    (hPt : L.partition s = some Pt) (hd : L.two s = some d) (hin : InScopeTwo Pt d s) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u) (cls : Fin (L.nG x) → Fin (L.nG x))
    (hallX : ∀ (a : Ω × ((Fin (L.V.k x) → Coin) × L.oneWait.SlotEC x)) (g g' : Fin (L.nG x)), cls g = cls g' →
      (L.oneWait.opensAt H P x a.1 a.2.1 a.2.2 g).isSome → (L.oneWait.opensAt H P x a.1 a.2.1 a.2.2 g').isSome)
    (hallI : ∀ (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.oneWait.SlotEC x)
      (g g' : Fin (L.nG x)), cls g = cls g' → (L.oneWait.iopTable H P x t.1 d.part t.2 ec g).isSome →
        (L.oneWait.iopTable H P x t.1 d.part t.2 ec g').isSome)
    {T Ns Ks Km : ℝ}
    (hT : Ex (fun q : Ω × (Fin (L.V.k x) → Coin) => (L.toCompiled.partiesCalls H P x q.1 q.2 : ℝ)) ≤ T)
    (hNs : ∀ r : Fin (L.V.k x) → Coin, ((L.toCompiled.firstSlots x (L.checked x r)).card : ℝ) ≤ Ns)
    (hKs : ∀ r : Fin (L.V.k x) → Coin,
      (((L.toCompiled.sampleNeeds x needs (L.checked x r)).image cls).card : ℝ) ≤ Ks)
    (hKm : ∀ r : Fin (L.V.k x) → Coin,
      (L.toCompiled.sessCountIn x needs (fun l => (L.toCompiled.sampleNeeds x needs l).image cls)
        (L.checked x r) : ℝ) ≤ Km)
    (hB : ExpectedCR H (Compiled.BconfX H L.oneWait.toCompiled Ω P x))
    (hX : ∀ K : ℕ, ExpectedCR H (L.oneWait.toCompiled.finderXCls H P x s needs cls K))
    (hI : ∀ K : ℕ, ExpectedCR H (L.oneWait.finderICls H P x s d.part needs cls K)) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.oneWait.FregTwoX H P x p.1 p.2) ≤
      3 * L.κs x + 2 * Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) +
        2 * (2 * (T * (1 + L.oneWait.toCompiled.es x * Ns) +
              (L.oneWait.toCompiled.nPU x : ℝ) * L.oneWait.toCompiled.openCalls x) +
            4 * (T * (1 + L.oneWait.toCompiled.es x * Ns + Ks + L.oneWait.toCompiled.es x * Km) +
              (1 + Ks) * ((L.oneWait.toCompiled.nPU x : ℝ) * L.oneWait.toCompiled.openCalls x))) /
          2 ^ (256 : ℝ) := by
  have h := eventTwoX_oneWait_le_cls_avg hF hA hbind hS hopen hs hPt hd hin P needs hpos cls hallX hallI
    (Compiled.CallsAvg.of_counts hT hNs hKs hKm) hB hX hI
  unfold εclsOneWaitAvg at h
  rw [Compiled.runCostAvg_counts, Compiled.firstHitCostAvg_counts] at h
  exact h

end CompiledTwo

end Claims.SamplingGeneric
