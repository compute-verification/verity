module

public import Proofs.WeightSecurity.TwoProd
import Proofs.Core.Game.Prob
import Proofs.Core.Game.Tape
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Sampling.Session
import Proofs.Flock.EndToEnd.Joint.Prod
import Proofs.Flock.SamplingGeneric.TwoReg
import Proofs.Flock.SamplingGeneric.TwoRegMiss
import Proofs.Flock.Soundness.Game

/-! A period of two-stage verification phases with one failure probability for the whole period, proved. -/

@[expose] public section

namespace Claims.Sampling

theorem Design.Two.Pa_congr_esc (d : Design.Two) (Pt : Partition) {fp fp' : ℕ → ℕ → ℕ}
    (h : ∀ ρ < d.part.blocks.length, d.esc Pt fp ρ = d.esc Pt fp' ρ) : d.Pa Pt fp = d.Pa Pt fp' := by
  unfold Design.Two.Pa
  refine Finset.sum_congr rfl fun T hT => ?_
  have hT' : T ⊆ Finset.range d.part.blocks.length := Finset.mem_powerset.1 hT
  have h1 : ∏ ρ ∈ T, (1 - d.esc Pt fp ρ) = ∏ ρ ∈ T, (1 - d.esc Pt fp' ρ) :=
    Finset.prod_congr rfl fun ρ hρ => by rw [h ρ (Finset.mem_range.1 (hT' hρ))]
  have h2 : ∏ ρ ∈ Finset.range d.part.blocks.length \ T, d.esc Pt fp ρ =
      ∏ ρ ∈ Finset.range d.part.blocks.length \ T, d.esc Pt fp' ρ :=
    Finset.prod_congr rfl fun ρ hρ => h ρ (Finset.mem_range.1 (Finset.mem_sdiff.1 hρ).1)
  rw [h1, h2]

theorem Design.Two.paBestAt_eq (d : Design.Two) (s : AuditStatement) (Pt : Partition) (val : ℕ → Option Bool)
    {ι : ℕ → ℕ → Bool} (hι : BestInteriorsAt s Pt d val ι) :
    d.PaBestAt s Pt val = d.Pa Pt (footprintAt s Pt d.part val ι) := by
  have hε : BestInteriorsAt s Pt d val (Classical.epsilon (BestInteriorsAt s Pt d val)) :=
    Classical.epsilon_spec ⟨ι, hι⟩
  exact d.Pa_congr_esc Pt (fp := footprintAt s Pt d.part val (Classical.epsilon (BestInteriorsAt s Pt d val)))
    (fp' := footprintAt s Pt d.part val ι) fun ρ hρ => le_antisymm (hι ρ hρ _) (hε ρ hρ _)

end Claims.Sampling

namespace Claims.SamplingGeneric.CompiledTwo

open Claims
open Claims.Sampling (AuditStatement Partition incorrectAllAt)

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

theorem FregTwoAt_eq (H : Msg → Msg) (L : CompiledTwo Coin A) {Ω : Type} (P : Prover Ω) {x : Msg}
    {s : AuditStatement} {Pt : Partition} {d : Sampling.Design.Two} (hs : L.parse x = some s)
    (hPt : L.partition s = some Pt) (hd : L.two s = some d) (ω : Ω) (val : ℕ → Option Bool)
    (r : Fin (L.V.k x) → Coin) :
    L.FregTwoAt H P x ω val r = incorrectAllAt s Pt d.part val (L.iregVal H P x ω r) := by
  simp only [FregTwoAt, hs, hPt, hd]

end Claims.SamplingGeneric.CompiledTwo

namespace WeightSecurity.SecurityProofs

open Claims Claims.Sampling Claims.SamplingGeneric
open FlockSoundness.Game hiding Msg
open scoped ENNReal

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin} {O V Tick Adv : Type}

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
open Classical in
theorem escAt_cases (L : CompiledTwo Coin A) (x : Msg) (val : ℕ → Option Bool) :
    escAt L x val = 1 ∨ ∃ s Pt d, L.parse x = some s ∧ L.partition s = some Pt ∧ L.two s = some d ∧
      InScopeTwo Pt d s ∧ escAt L x val = ENNReal.ofReal (d.PaBestAt s Pt val) := by
  rcases hs : L.parse x with _ | s
  · exact Or.inl (by simp only [escAt, hs])
  rcases hPt : L.partition s with _ | Pt
  · exact Or.inl (by simp only [escAt, hs, hPt])
  rcases hd : L.two s with _ | d
  · exact Or.inl (by simp only [escAt, hs, hPt, hd])
  by_cases hin : InScopeTwo Pt d s
  · exact Or.inr ⟨s, Pt, d, rfl, hPt, hd, hin, by simp only [escAt, hs, hPt, hd, ite_eq_left hin]⟩
  · exact Or.inl (by simp only [escAt, hs, hPt, hd, ite_eq_right hin])

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem escAt_ne_top (L : CompiledTwo Coin A) (x : Msg) (val : ℕ → Option Bool) : escAt L x val ≠ ⊤ := by
  rcases escAt_cases L x val with h | ⟨s, Pt, d, -, -, -, -, h⟩
  · rw [h]
    exact ENNReal.one_ne_top
  · rw [h]
    exact ENNReal.ofReal_ne_top

theorem twoAtRec_accepted_le (H : Msg → Msg) (L : CompiledTwo Coin A)
    (href : ∀ x (P : Prover Unit) r, (run H P L.V x () r).out = .acc → L.Scope x)
    (val : (O → V) × Tick × Adv → Msg → Prover Unit → ℕ → Option Bool) (n : ℕ) (e : ℝ≥0∞) {ε : ℝ}
    (τ : Strategy (twoGame L ((O → V) × Tick × Adv)))
    (hev : L.Scope τ.2.1.1 → acceptedSampledAt H L τ.2.1.1 τ.2.1.2 (val τ.1 τ.2.1.1 τ.2.1.2) ≤ ε) :
    prob (fun b => (twoAtRec H L (val τ.1 τ.2.1.1 τ.2.1.2) n e b).accepted ∧
        (twoAtRec H L (val τ.1 τ.2.1.1 τ.2.1.2) n e b).sampledIncorrect) (twoGame L _) τ ≤ ENNReal.ofReal ε := by
  obtain ⟨m, ⟨x, P⟩, k⟩ := τ
  show Verity.Game.prCoin (fun r => (run H P L.V x () r).out = .acc ∧
      ∃ u ∈ L.checked x r, u ∈ L.FregTwoAt H P x () (val m x P) r) ≤ ENNReal.ofReal ε
  by_cases hx : L.Scope x
  · have h : Pr (fun r : Fin (L.V.k x) → Coin => (run H P L.V x () r).out = .acc ∧
        ∃ u ∈ L.checked x r, u ∈ L.FregTwoAt H P x () (val m x P) r) ≤ ε := hev hx
    rw [Verity.Game.prCoin_eq_ofReal_pr]
    exact ENNReal.ofReal_le_ofReal h
  · rw [show (fun r => (run H P L.V x () r).out = .acc ∧
          ∃ u ∈ L.checked x r, u ∈ L.FregTwoAt H P x () (val m x P) r) =
        fun _ => False from funext fun r => propext (iff_false_intro fun h => hx (href x P r h.1)),
      Verity.Game.prCoin_false]
    exact zero_le

theorem twoAtRec_missed_le (H : Msg → Msg) (L : CompiledTwo Coin A) (hS : L.Sessions H)
    (val : (O → V) × Tick × Adv → Msg → Prover Unit → ℕ → Option Bool) (n : ℕ) (e : ℝ≥0∞)
    (τ : Strategy (twoGame L ((O → V) × Tick × Adv))) :
    prob (fun b => ¬ (twoAtRec H L (val τ.1 τ.2.1.1 τ.2.1.2) n e b).sampledIncorrect) (twoGame L _) τ ≤
      escAt L τ.2.1.1 (val τ.1 τ.2.1.1 τ.2.1.2) := by
  obtain ⟨m, ⟨x, P⟩, k⟩ := τ
  show prob (fun b => ¬ (twoAtRec H L (val m x P) n e b).sampledIncorrect) (twoGame L _) ⟨m, ⟨x, P⟩, k⟩ ≤
    escAt L x (val m x P)
  rcases escAt_cases L x (val m x P) with h1 | ⟨s, Pt, d, hs, hPt, hd, hin, he⟩
  · rw [h1]
    exact Verity.Game.prob_le_one _ _ _
  · rw [he]
    show Verity.Game.prCoin (fun r => ¬ ∃ u ∈ L.checked x r, u ∈ L.FregTwoAt H P x () (val m x P) r) ≤ _
    obtain ⟨ι, hι⟩ := exists_best_interiorsAt s Pt d (val m x P)
    rw [Verity.Game.prCoin_eq_ofReal_pr, d.paBestAt_eq s Pt (val m x P) (ι := ι) hι]
    refine ENNReal.ofReal_le_ofReal (le_trans (Pr_mono ?_)
      (CompiledTwo.pr_missTwo_le_Pa H L hS hs hPt hd hin P () (val m x P) ι hι))
    intro r hr u hu hF
    exact hr ⟨u, hu, by rw [CompiledTwo.FregTwoAt_eq H L P hs hPt hd]; exact hF⟩

theorem monitoringPeriodTwoProd_of (H : Msg → Msg) (Lv : ℕ → CompiledTwo Coin A)
    (val : (t : ℕ) → (O → V) × Tick × Adv → Msg → Prover Unit → ℕ → Option Bool) (hS : ∀ t, (Lv t).Sessions H) :
    MonitoringPeriodTwoProd (O := O) (V := V) (Tick := Tick) (Adv := Adv) H Lv val := by
  intro T ε s hev δ
  have hs := monitoringPeriodGood_of_plays (fun t => twoGame (Lv t) ((O → V) × Tick × Adv))
    (fun t τ => (Lv t).Scope τ.2.1.1 →
      acceptedSampledAt H (Lv t) τ.2.1.1 τ.2.1.2 (val t τ.1 τ.2.1.1 τ.2.1.2) ≤ ε t) T 0 s hev
  have hA := monitoringPeriod_union (fun t => twoGame (Lv t) ((O → V) × Tick × Adv))
    (fun t τ => twoAtRec H (Lv t) (val t τ.1 τ.2.1.1 τ.2.1.2) (regCountAt (Lv t) τ.2.1.1 (val t τ.1 τ.2.1.1 τ.2.1.2))
      (escAt (Lv t) τ.2.1.1 (val t τ.1 τ.2.1.1 τ.2.1.2)))
    (fun r : ServedEsc O V Tick Adv => r.accepted ∧ r.sampledIncorrect)
    (fun t τ => (Lv t).Scope τ.2.1.1 →
      acceptedSampledAt H (Lv t) τ.2.1.1 τ.2.1.2 (val t τ.1 τ.2.1.1 τ.2.1.2) ≤ ε t)
    (fun t => ENNReal.ofReal (ε t))
    (fun t τ hτ => twoAtRec_accepted_le H (Lv t) (fun x P r h => (hS t).refuses x Unit P () r h) (val t) _ _ τ hτ)
    T 0 s hs
  simp only [Nat.zero_add] at hA
  have hB := monitoringPeriod_prod (fun t => twoGame (Lv t) ((O → V) × Tick × Adv))
    (fun t τ => twoAtRec H (Lv t) (val t τ.1 τ.2.1.1 τ.2.1.2) (regCountAt (Lv t) τ.2.1.1 (val t τ.1 τ.2.1.1 τ.2.1.2))
      (escAt (Lv t) τ.2.1.1 (val t τ.1 τ.2.1.1 τ.2.1.2)))
    (fun r : ServedEsc O V Tick Adv => ¬ r.sampledIncorrect) ServedEsc.missProb
    (fun t τ => (Lv t).Scope τ.2.1.1 →
      acceptedSampledAt H (Lv t) τ.2.1.1 τ.2.1.2 (val t τ.1 τ.2.1.1 τ.2.1.2) ≤ ε t)
    (fun t τ _ => ⟨escAt (Lv t) τ.2.1.1 (val t τ.1 τ.2.1.1 τ.2.1.2), escAt_ne_top _ _ _, fun _ => rfl,
      twoAtRec_missed_le H (Lv t) (hS t) (val t) _ _ τ⟩)
    T 0 s (ENNReal.ofReal δ) hs
  calc Verity.Game.value (fun l : List (ServedEsc O V Tick Adv) =>
          (∀ r ∈ l, r.accepted) ∧ (l.map ServedEsc.missProb).prod < ENNReal.ofReal δ) (twoAtRun H Lv val T s)
      ≤ Verity.Game.value (fun l : List (ServedEsc O V Tick Adv) =>
          (∃ r ∈ l, r.accepted ∧ r.sampledIncorrect) ∨
            ((∀ r ∈ l, ¬ r.sampledIncorrect) ∧ (l.map ServedEsc.missProb).prod < ENNReal.ofReal δ))
          (twoAtRun H Lv val T s) := by
        refine Verity.Game.value_mono (fun l hl => ?_) _
        by_cases hd : ∃ r ∈ l, r.sampledIncorrect
        · obtain ⟨r, hr, hdr⟩ := hd
          exact Or.inl ⟨r, hr, hl.1 r hr, hdr⟩
        · exact Or.inr ⟨fun r hr hdr => hd ⟨r, hr, hdr⟩, hl.2⟩
    _ ≤ Verity.Game.value (fun l : List (ServedEsc O V Tick Adv) => ∃ r ∈ l, r.accepted ∧ r.sampledIncorrect)
          (twoAtRun H Lv val T s) +
        Verity.Game.value (fun l : List (ServedEsc O V Tick Adv) =>
          (∀ r ∈ l, ¬ r.sampledIncorrect) ∧ (l.map ServedEsc.missProb).prod < ENNReal.ofReal δ)
          (twoAtRun H Lv val T s) := Verity.Game.value_or_le _ _ _
    _ ≤ (∑ t ∈ Finset.range T, ENNReal.ofReal (ε t)) + ENNReal.ofReal δ := add_le_add hA hB
    _ = ENNReal.ofReal δ + ∑ t ∈ Finset.range T, ENNReal.ofReal (ε t) := add_comm _ _

theorem period_leadEscape {β : ℕ → Type} (G : (i : ℕ) → Verity.Game (β i))
    (rec : (i : ℕ) → Strategy (G i) → β i → ServedEsc O V Tick Adv) (good : (i : ℕ) → Strategy (G i) → Prop)
    (hstage : ∀ i τ, good i τ → ∃ e : ℝ≥0∞, e ≠ ⊤ ∧ (∀ b, (rec i τ b).missProb = e) ∧
      prob (fun b => ¬ (rec i τ b).sampledIncorrect) (G i) τ ≤ e) :
    ∀ (T i : ℕ) (s : Strategy (monitoringPeriodG G i T)) (c : ℝ≥0∞), MonitoringPeriodGood G good i T s →
      value (fun l => ServedEsc.leadMissProb l < c) (monitoringPeriodRun G rec i T s) ≤ c
  | 0, i, s, c, _ => by
      classical
      show value (fun l => ServedEsc.leadMissProb l < c) (Verity.Game.ret []) ≤ _
      rw [value_ret]
      split_ifs with h
      · exact le_of_lt (by simpa only [ServedEsc.leadMissProb] using h)
      · exact zero_le
  | T + 1, i, s, c, hs => by
      classical
      rcases lt_or_ge 1 c with hc | hc
      · exact (value_le_one _ _).trans hc.le
      obtain ⟨e, he, hw, hp⟩ := hstage i _ hs.1
      by_cases he0 : e = 0
      · have h := value_runBind_le_mul (fun _ => monitoringPeriodG G (i + 1) T)
          (fun b s' => (monitoringPeriodRun G rec (i + 1) T s').map (rec i (Strategy.ofBind _ (G i) s) b :: ·))
          (fun l => ServedEsc.leadMissProb l < c)
          (fun b => ¬ (rec i (Strategy.ofBind _ (G i) s) b).sampledIncorrect) 1 (G i) s
          (atEnds_mono (fun b s' _ => by
            rw [value_map]
            by_cases hm : (rec i (Strategy.ofBind _ (G i) s) b).sampledIncorrect
            · rw [ite_eq_right (not_not.2 hm)]
              refine (value_mono (q := fun _ => False) (fun l hl => ?_) _).trans_eq (value_false _)
              simp only [ServedEsc.leadMissProb, ite_eq_left hm] at hl
              exact not_lt.2 hc hl
            · rw [ite_eq_left hm]
              exact value_le_one _ _) (G i) s hs.2)
        refine h.trans ?_
        rw [mul_one]
        exact (hp.trans_eq he0).trans zero_le
      · have h := value_runBind_le_mul (fun _ => monitoringPeriodG G (i + 1) T)
          (fun b s' => (monitoringPeriodRun G rec (i + 1) T s').map (rec i (Strategy.ofBind _ (G i) s) b :: ·))
          (fun l => ServedEsc.leadMissProb l < c)
          (fun b => ¬ (rec i (Strategy.ofBind _ (G i) s) b).sampledIncorrect) (c / e) (G i) s
          (atEnds_mono (fun b s' hg => by
            rw [value_map]
            by_cases hm : (rec i (Strategy.ofBind _ (G i) s) b).sampledIncorrect
            · rw [ite_eq_right (not_not.2 hm)]
              refine (value_mono (q := fun _ => False) (fun l hl => ?_) _).trans_eq (value_false _)
              simp only [ServedEsc.leadMissProb, ite_eq_left hm] at hl
              exact not_lt.2 hc hl
            · rw [ite_eq_left hm]
              refine (value_mono (fun l hl => ?_) _).trans
                (period_leadEscape G rec good hstage T (i + 1) s' (c / e) hg)
              simp only [ServedEsc.leadMissProb, ite_eq_right hm, hw b] at hl
              exact (ENNReal.lt_div_iff_mul_lt (Or.inl he0) (Or.inl he)).2 (by rwa [mul_comm])) (G i) s hs.2)
        calc value (fun l => ServedEsc.leadMissProb l < c) (monitoringPeriodRun G rec i (T + 1) s)
            ≤ prob (fun b => ¬ (rec i (Strategy.ofBind _ (G i) s) b).sampledIncorrect) (G i)
                (Strategy.ofBind _ (G i) s) * (c / e) := h
          _ ≤ e * (c / e) := by gcongr
          _ ≤ c := ENNReal.mul_div_le

theorem monitoringPeriodTwoLead_of (H : Msg → Msg) (Lv : ℕ → CompiledTwo Coin A)
    (val : (t : ℕ) → (O → V) × Tick × Adv → Msg → Prover Unit → ℕ → Option Bool) (hS : ∀ t, (Lv t).Sessions H) :
    MonitoringPeriodTwoLead (O := O) (V := V) (Tick := Tick) (Adv := Adv) H Lv val := by
  intro T ε s hev δ
  have hs := monitoringPeriodGood_of_plays (fun t => twoGame (Lv t) ((O → V) × Tick × Adv))
    (fun t τ => (Lv t).Scope τ.2.1.1 →
      acceptedSampledAt H (Lv t) τ.2.1.1 τ.2.1.2 (val t τ.1 τ.2.1.1 τ.2.1.2) ≤ ε t) T 0 s hev
  have hA := monitoringPeriod_union (fun t => twoGame (Lv t) ((O → V) × Tick × Adv))
    (fun t τ => twoAtRec H (Lv t) (val t τ.1 τ.2.1.1 τ.2.1.2) (regCountAt (Lv t) τ.2.1.1 (val t τ.1 τ.2.1.1 τ.2.1.2))
      (escAt (Lv t) τ.2.1.1 (val t τ.1 τ.2.1.1 τ.2.1.2)))
    (fun r : ServedEsc O V Tick Adv => r.accepted ∧ r.sampledIncorrect)
    (fun t τ => (Lv t).Scope τ.2.1.1 →
      acceptedSampledAt H (Lv t) τ.2.1.1 τ.2.1.2 (val t τ.1 τ.2.1.1 τ.2.1.2) ≤ ε t)
    (fun t => ENNReal.ofReal (ε t))
    (fun t τ hτ => twoAtRec_accepted_le H (Lv t) (fun x P r h => (hS t).refuses x Unit P () r h) (val t) _ _ τ hτ)
    T 0 s hs
  simp only [Nat.zero_add] at hA
  have hB := period_leadEscape (fun t => twoGame (Lv t) ((O → V) × Tick × Adv))
    (fun t τ => twoAtRec H (Lv t) (val t τ.1 τ.2.1.1 τ.2.1.2) (regCountAt (Lv t) τ.2.1.1 (val t τ.1 τ.2.1.1 τ.2.1.2))
      (escAt (Lv t) τ.2.1.1 (val t τ.1 τ.2.1.1 τ.2.1.2)))
    (fun t τ => (Lv t).Scope τ.2.1.1 →
      acceptedSampledAt H (Lv t) τ.2.1.1 τ.2.1.2 (val t τ.1 τ.2.1.1 τ.2.1.2) ≤ ε t)
    (fun t τ _ => ⟨escAt (Lv t) τ.2.1.1 (val t τ.1 τ.2.1.1 τ.2.1.2), escAt_ne_top _ _ _, fun _ => rfl,
      twoAtRec_missed_le H (Lv t) (hS t) (val t) _ _ τ⟩)
    T 0 s (ENNReal.ofReal δ) hs
  rw [add_comm (ENNReal.ofReal δ)]
  show Verity.Game.value (fun l : List (ServedEsc O V Tick Adv) =>
      (∃ w ∈ l, w.accepted ∧ w.sampledIncorrect) ∨ ServedEsc.leadMissProb l < ENNReal.ofReal δ)
      (twoAtRun H Lv val T s) ≤ _
  exact (Verity.Game.value_or_le _ _ _).trans (add_le_add hA hB)

theorem compilerMonitoringPeriodTwoProd : CompilerMonitoringPeriodTwoProd := by
  intro Coin _ _ _ A O V Tick Adv H Lv val hS
  exact ⟨monitoringPeriodTwoLead_of H Lv val hS, monitoringPeriodTwoProd_of H Lv val hS⟩

theorem acceptedSampledAt_regVal_le {H : Msg → Msg} {L : CompiledTwo Coin A} (hA : A.KS H)
    (hbind : L.commit.Binds H) (hS : L.Sessions H) {x : Msg} (hx : L.Scope x) (P : Prover Unit) :
    acceptedSampledAt H L x P (L.regVal H P x ()) ≤ L.εregTwo H x P := by
  have h := CompiledTwo.eventTwo_le hA hbind hS x hx Unit P
  obtain ⟨s, Pt, d, hs, hPt, hd, -⟩ := hx
  simp only [CompiledTwo.FregTwo_eq H L P hs hPt hd] at h
  unfold acceptedSampledAt
  simp only [CompiledTwo.FregTwoAt_eq H L P hs hPt hd]
  exact (Pr_unit fun r : Fin (L.V.k x) → Coin => (run H P L.V x () r).out = .acc ∧
    ∃ u ∈ L.checked x r, u ∈ incorrectAllAt s Pt d.part (L.regVal H P x ()) (L.iregVal H P x () r)).symm.trans_le h

theorem monitoringPeriodTwoProd_regVal (H : Msg → Msg) (Lv : ℕ → CompiledTwo Coin A) (hA : A.KS H)
    (hS : ∀ t, (Lv t).commit.Binds H ∧ (Lv t).Sessions H) (T : ℕ) (ε : ℕ → ℝ)
    (s : Strategy (monitoringPeriodG (fun t => twoGame (Lv t) ((O → V) × Tick × Adv)) 0 T))
    (herr : ∀ t τ, MonitoringPeriodPlays (fun t => twoGame (Lv t) ((O → V) × Tick × Adv)) 0 T s t τ →
      (Lv t).εregTwo H τ.2.1.1 τ.2.1.2 ≤ ε t) (δ : ℝ) :
    Verity.Game.value (fun l : List (ServedEsc O V Tick Adv) =>
        (∀ r ∈ l, r.accepted) ∧ (l.map ServedEsc.missProb).prod < ENNReal.ofReal δ)
        (twoAtRun H Lv (fun t _ x P => (Lv t).regVal H P x ()) T s) ≤
      ENNReal.ofReal δ + ∑ t ∈ Finset.range T, ENNReal.ofReal (ε t) :=
  monitoringPeriodTwoProd_of H Lv (fun t _ x P => (Lv t).regVal H P x ()) (fun t => (hS t).2) T ε s
    (fun t τ hp hx => (acceptedSampledAt_regVal_le hA (hS t).1 (hS t).2 hx τ.2.1.2).trans (herr t τ hp)) δ

theorem monitoringPeriodTwoLead_regVal (H : Msg → Msg) (Lv : ℕ → CompiledTwo Coin A) (hA : A.KS H)
    (hS : ∀ t, (Lv t).commit.Binds H ∧ (Lv t).Sessions H) (T : ℕ) (ε : ℕ → ℝ)
    (s : Strategy (monitoringPeriodG (fun t => twoGame (Lv t) ((O → V) × Tick × Adv)) 0 T))
    (herr : ∀ t τ, MonitoringPeriodPlays (fun t => twoGame (Lv t) ((O → V) × Tick × Adv)) 0 T s t τ →
      (Lv t).εregTwo H τ.2.1.1 τ.2.1.2 ≤ ε t) (δ : ℝ) :
    Verity.Game.value (ServedEsc.UndetectedBelow δ)
        (twoAtRun H Lv (fun t _ x P => (Lv t).regVal H P x ()) T s) ≤
      ENNReal.ofReal δ + ∑ t ∈ Finset.range T, ENNReal.ofReal (ε t) :=
  monitoringPeriodTwoLead_of H Lv (fun t _ x P => (Lv t).regVal H P x ()) (fun t => (hS t).2) T ε s
    (fun t τ hp hx => (acceptedSampledAt_regVal_le hA (hS t).1 (hS t).2 hx τ.2.1.2).trans (herr t τ hp)) δ

end WeightSecurity.SecurityProofs
