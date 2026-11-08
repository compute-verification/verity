module

public import Proofs.Core.Game.Basic
public import Proofs.Core.Game.Interleave
public import Proofs.Core.Game.Prob
public import Proofs.Core.Game.Tape
public import Proofs.Core.Game.Union
public import Proofs.Flock.Claims.Prob
public import Proofs.Flock.Claims.Sampling.Session
public import Proofs.Flock.EndToEnd.Period.Union
public import Proofs.Flock.SamplingGeneric.TwoCap
public import Proofs.WeightSecurity.Served

/-!
# Weight security at a two-stage audit

`TheftServed` at a deployment whose audit is a compiled two-stage sampled verifier (`CompiledTwo`). Each window of the
monitoring period is one play of `twoGame`: the prover names the window's release, a statement and a tapeless prover,
and the verifier flips the run's coins. `twoRec` records the window as a `Served`. The window is sampled incorrect unless
its run accepts and an extraction lands in `RTwo`, and then its incorrect proof blocks are the fewest replay blocks such
an extraction marks (`TwoPlay.marks`).

* `twoRec_accepted_le`: a window is accepted and sampled incorrect with probability at most its KS error at `δ`
  (`twoErr`), by `CompiledTwo.KS` at a tapeless prover, and with probability `0` outside the scope.
* `twoRec_incorrect_le`: a window that isn't sampled incorrect carries at most `K − 1` incorrect proof blocks when every
  footprint in `M̄_δ` marks fewer than `K` replay blocks (`CapTwo`).
* `value_lead_eq_zero`: so `T` windows never carry a lead of `T·(K − 1) + 1`.
* `two_undetected`: the period leaves `T·(K − 1) + 1` incorrect proof blocks undetected with probability at most
  `Σ_t ε_t`, where `ε_t` bounds window `t`'s KS error.
* `TheftServedTwo`: `TheftServed` at that budget.
* `capTwo_of_rates`: `CapTwo` from a subset-inputs first stage and a per-cell second-stage rate
  (`Design.Two.card_wrongBlocks_lt_subsetInputs`).

The budget grows linearly in the windows, as `T·(K − 1) + 1`, with `K` the cap at `δ`.
-/

@[expose] public section

namespace WeightSecurity.SecurityProofs

open Claims Claims.Sampling Claims.SamplingGeneric FlockSoundness.Game WeightSecurity.Assumptions
open Claims.Sampling.EndToEndDraw (SubsetInputsDesign)
open scoped ENNReal

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin} {O V Tick Adv : Type}

/-- A window's play: the statement, the tapeless prover and the run's coins. -/
abbrev TwoPlay (L : CompiledTwo Coin A) := (x : Claims.Msg) × Prover Unit × (Fin (L.V.k x) → Coin)

/-- **A window of a two-stage audit**: the prover names the window's release, a statement and a tapeless prover, and the
verifier flips the run's coins. -/
noncomputable def twoGame (L : CompiledTwo Coin A) (Rls : Type) : Verity.Game (Rls × TwoPlay L) :=
  .send Rls fun m => .send (Claims.Msg × Prover Unit) fun xp =>
    .coin (Fin (L.V.k xp.1) → Coin) fun r => .ret (m, ⟨xp.1, xp.2, r⟩)

namespace TwoPlay

/-- The play's run accepts. -/
def accepted (H : Claims.Msg → Claims.Msg) {L : CompiledTwo Coin A} (p : TwoPlay L) : Prop :=
  (run H p.2.1 L.V p.1 () p.2.2).out = .acc

/-- The witness the compiled extractor returns on the play at its own coins `e`. -/
noncomputable def extracted (H : Claims.Msg → Claims.Msg) {L : CompiledTwo Coin A} (p : TwoPlay L) (e : L.E.C p.1) :
    Option Wit :=
  (L.E.eval H p.2.1 p.1 () p.2.2 e).1

/-- **The counts an extraction marks**: the number of replay blocks the footprint of an extracted witness marks
incorrect, at interior values that put the footprint in `M̄_δ`. -/
def marks (H : Claims.Msg → Claims.Msg) {L : CompiledTwo Coin A} (δ : ℝ) (p : TwoPlay L) : Set ℕ :=
  {n | ∃ (e : L.E.C p.1) (w : Wit) (s : AuditStatement) (Pt : Partition) (d : Design.Two) (ι : ℕ → ℕ → Bool),
    p.extracted H e = some w ∧ L.parse p.1 = some s ∧ L.partition s = some Pt ∧ L.two s = some d ∧
      footprint H L.commit s Pt d.part w ι ∈ d.M Pt δ ∧
      (d.wrongBlocks Pt (footprint H L.commit s Pt d.part w ι)).card = n}

/-- An extraction marks some count exactly when one lands in `RTwo`. -/
theorem marks_nonempty (H : Claims.Msg → Claims.Msg) {L : CompiledTwo Coin A} (δ : ℝ) (p : TwoPlay L) :
    (p.marks H δ).Nonempty ↔ ∃ e, (RTwo H L.commit L.parse L.partition L.two δ).ext p.1 (p.extracted H e) := by
  constructor
  · rintro ⟨_, e, w, s, Pt, d, ι, hw, hs, hPt, hd, hM, -⟩
    refine ⟨e, ?_⟩
    rw [hw]
    exact ⟨s, Pt, d, hs, hPt, hd, ι, hM⟩
  · rintro ⟨e, h⟩
    cases hw : p.extracted H e with
    | none =>
      rw [hw] at h
      exact (h : False).elim
    | some w =>
      rw [hw] at h
      obtain ⟨s, Pt, d, hs, hPt, hd, ι, hM⟩ := h
      exact ⟨_, e, w, s, Pt, d, ι, hw, hs, hPt, hd, hM, rfl⟩

end TwoPlay

open Classical in
/-- **A window's record**: sampled incorrect unless the run accepts and an extraction lands in `RTwo`, and then carrying
the fewest replay blocks such an extraction marks; otherwise it carries `|O|`. -/
noncomputable def twoRec [Fintype O] (H : Claims.Msg → Claims.Msg) (L : CompiledTwo Coin A) (δ : ℝ)
    (b : ((O → V) × Tick × Adv) × TwoPlay L) : Served O V Tick Adv where
  accepted := b.2.accepted H
  sampledIncorrect := ¬ (b.2.accepted H ∧ (b.2.marks H δ).Nonempty)
  incorrect := if b.2.accepted H ∧ (b.2.marks H δ).Nonempty then sInf (b.2.marks H δ) else Fintype.card O
  egress := b.1.1
  tick := b.1.2.1
  adv := b.1.2.2

/-- **A monitoring period of `T` two-stage windows**, window `t` audited by `Lv t`. -/
noncomputable def twoRun [Fintype O] (H : Claims.Msg → Claims.Msg) (Lv : ℕ → CompiledTwo Coin A) (δ : ℝ) (T : ℕ)
    (s : Strategy (periodG (fun t => twoGame (Lv t) ((O → V) × Tick × Adv)) 0 T)) :
    Verity.Game (List (Served O V Tick Adv)) :=
  periodRun (fun t => twoGame (Lv t) ((O → V) × Tick × Adv)) (fun t _ => twoRec H (Lv t) δ) 0 T s

/-- **A window's KS error at `δ`**, at the statement and prover its strategy names: `κ_δ + √(γ·Pr[B]) + Pr[B_ext]`. -/
noncomputable def twoErr (H : Claims.Msg → Claims.Msg) (L : CompiledTwo Coin A) (δ : ℝ) {Rls : Type}
    (τ : Strategy (twoGame L Rls)) : ℝ :=
  L.κ δ τ.2.1.1 + Real.sqrt (L.γ τ.2.1.1 * PrB H L.V τ.2.1.2 τ.2.1.1) + (Bext H L.V L.E Unit τ.2.1.2 τ.2.1.1).Pr H

/-- The run at coins `r` accepts and no extraction lands in `RTwo`. -/
def twoMiss (H : Claims.Msg → Claims.Msg) (L : CompiledTwo Coin A) (δ : ℝ) (x : Claims.Msg) (P : Prover Unit)
    (r : Fin (L.V.k x) → Coin) : Prop :=
  (run H P L.V x () r).out = .acc ∧
    ∀ e, ¬ (RTwo H L.commit L.parse L.partition L.two δ).ext x (L.E.eval H P x () r e).1

/-- **The per-window cap**: on every statement in scope, a footprint in `M̄_δ` that counts at most each cell marks fewer
than `K` replay blocks incorrect. -/
def CapTwo (L : CompiledTwo Coin A) (δ : ℝ) (K : ℕ) : Prop :=
  ∀ x, L.Scope x → ∀ s Pt d, L.parse x = some s → L.partition s = some Pt → L.two s = some d →
    ∀ fp ∈ d.M Pt δ, (∀ ρ t, fp ρ t ≤ (cell Pt d.part ρ t).card) → (d.wrongBlocks Pt fp).card < K

/-- A window is accepted and sampled incorrect exactly when its run accepts and no extraction lands in `RTwo`. -/
theorem twoRec_accepted_iff [Fintype O] (H : Claims.Msg → Claims.Msg) (L : CompiledTwo Coin A) (δ : ℝ)
    (b : ((O → V) × Tick × Adv) × TwoPlay L) :
    ((twoRec H L δ b).accepted ∧ (twoRec H L δ b).sampledIncorrect) ↔
      b.2.accepted H ∧ ∀ e, ¬ (RTwo H L.commit L.parse L.partition L.two δ).ext b.2.1 (b.2.extracted H e) := by
  show (b.2.accepted H ∧ ¬ (b.2.accepted H ∧ (b.2.marks H δ).Nonempty)) ↔ _
  rw [TwoPlay.marks_nonempty]
  exact ⟨fun ⟨ha, hn⟩ => ⟨ha, fun e he => hn ⟨ha, e, he⟩⟩, fun ⟨ha, hn⟩ => ⟨ha, fun ⟨_, e, he⟩ => hn e he⟩⟩

/-- At a tapeless prover, accepting with every extraction outside `R` is `PrBad`'s event at the extractor's coins. -/
theorem pr_acc_noext_le (H : Claims.Msg → Claims.Msg) {W : Type} (V : Verifier Coin) (R : Rel W)
    (E : Extractor Coin W V) (x : Claims.Msg) (P : Prover Unit) :
    Pr (fun r : Fin (V.k x) → Coin => (run H P V x () r).out = .acc ∧ ∀ e, ¬ R.ext x (E.eval H P x () r e).1) ≤
      PrBad H V R E x P :=
  calc Pr (fun r : Fin (V.k x) → Coin => (run H P V x () r).out = .acc ∧ ∀ e, ¬ R.ext x (E.eval H P x () r e).1)
      = Pr (fun s : (Fin (V.k x) → Coin) × E.C x =>
          (run H P V x () s.1).out = .acc ∧ ∀ e, ¬ R.ext x (E.eval H P x () s.1 e).1) :=
        (Pr_fst (fun r : Fin (V.k x) → Coin =>
          (run H P V x () r).out = .acc ∧ ∀ e, ¬ R.ext x (E.eval H P x () r e).1)).symm
    _ = Pr (fun s : Unit × (Fin (V.k x) → Coin) × E.C x =>
          (run H P V x () s.2.1).out = .acc ∧ ∀ e, ¬ R.ext x (E.eval H P x () s.2.1 e).1) :=
        (Pr_unit (fun s : (Fin (V.k x) → Coin) × E.C x =>
          (run H P V x () s.1).out = .acc ∧ ∀ e, ¬ R.ext x (E.eval H P x () s.1 e).1)).symm
    _ ≤ PrBad H V R E x P := Pr_mono fun s hs => ⟨hs.1, hs.2 s.2.2⟩

/-- **A window is accepted and sampled incorrect with probability at most its KS error**, by `CompiledTwo.KS` at the
tapeless prover its strategy names; outside the scope the run never accepts. -/
theorem twoRec_accepted_le [Fintype O] (H : Claims.Msg → Claims.Msg) (L : CompiledTwo Coin A) {δ : ℝ} (hδ : 0 ≤ δ)
    (hks : L.KS H) (href : ∀ x (P : Prover Unit) r, (run H P L.V x () r).out = .acc → L.Scope x)
    (τ : Strategy (twoGame L ((O → V) × Tick × Adv))) :
    prob (fun b => (twoRec H L δ b).accepted ∧ (twoRec H L δ b).sampledIncorrect) (twoGame L _) τ ≤
      ENNReal.ofReal (twoErr H L δ τ) := by
  obtain ⟨m, ⟨x, P⟩, k⟩ := τ
  have hq : prob (fun b => (twoRec H L δ b).accepted ∧ (twoRec H L δ b).sampledIncorrect) (twoGame L _)
      ⟨m, ⟨x, P⟩, k⟩ = prCoin (twoMiss H L δ x P) := by
    rw [show twoMiss H L δ x P = fun r => (twoRec H L δ (m, ⟨x, P, r⟩)).accepted ∧
        (twoRec H L δ (m, ⟨x, P, r⟩)).sampledIncorrect from
      funext fun r => propext (twoRec_accepted_iff H L δ (m, ⟨x, P, r⟩)).symm]
    rfl
  rw [hq]
  by_cases hx : L.Scope x
  · rw [Verity.Game.prCoin_eq_ofReal_pr]
    refine ENNReal.ofReal_le_ofReal ?_
    have h := (pr_acc_noext_le H L.V (RTwo H L.commit L.parse L.partition L.two δ) L.E x P).trans
      (hks δ hδ x hx Unit P)
    simp only [one_mul] at h
    exact h
  · rw [show twoMiss H L δ x P = fun _ => False from
      funext fun r => propext (iff_false_intro fun h => hx (href x P r h.1)), Verity.Game.prCoin_false]
    exact zero_le

/-- **A window that isn't sampled incorrect carries fewer than `K` incorrect proof blocks** under the cap: some
extraction marks its count, and that extraction's footprint is in `M̄_δ`. -/
theorem twoRec_incorrect_le [Fintype O] (H : Claims.Msg → Claims.Msg) (L : CompiledTwo Coin A) {δ : ℝ} {K : ℕ}
    (href : ∀ x (P : Prover Unit) r, (run H P L.V x () r).out = .acc → L.Scope x) (hcap : CapTwo L δ K)
    (b : ((O → V) × Tick × Adv) × TwoPlay L) (hb : ¬ (twoRec H L δ b).sampledIncorrect) :
    (twoRec H L δ b).incorrect ≤ K - 1 := by
  have hb' : b.2.accepted H ∧ (b.2.marks H δ).Nonempty := not_not.1 hb
  obtain ⟨e, w, s, Pt, d, ι, -, hs, hPt, hd, hM, hn⟩ := Nat.sInf_mem hb'.2
  have h := hcap b.2.1 (href b.2.1 b.2.2.1 b.2.2.2 hb'.1) s Pt d hs hPt hd _ hM
    fun ρ t => footprint_le_card H L.commit s Pt d.part w ι ρ t
  dsimp only [twoRec]
  split_ifs with hc
  · omega
  · exact absurd hb' hc

/-- **`T` windows never carry a lead of `M > T·c`** when each window that isn't sampled incorrect carries at most `c`
incorrect proof blocks. -/
theorem value_lead_eq_zero {β : ℕ → Type} (G : (i : ℕ) → Verity.Game (β i))
    (rec : (i : ℕ) → Strategy (G i) → β i → Served O V Tick Adv) (c : ℕ) :
    ∀ (T i : ℕ) (s : Strategy (periodG G i T)) (M : ℕ),
      (∀ j, i ≤ j → j < i + T → ∀ τ b, ¬ (rec j τ b).sampledIncorrect → (rec j τ b).incorrect ≤ c) →
      T * c < M → Verity.Game.value (fun l => M ≤ Served.lead l) (periodRun G rec i T s) = 0
  | 0, _, _, M, _, hM => by
      classical
      show Verity.Game.value (fun l => M ≤ Served.lead l) (Verity.Game.ret []) = 0
      rw [Verity.Game.value_ret, ite_eq_right]
      simp only [Served.lead]
      omega
  | T + 1, i, s, M, hc, hM => by
      classical
      have hpos : 0 < M := lt_of_le_of_lt (Nat.zero_le _) hM
      have hM' : T * c < M - c := by
        have := add_one_mul T c
        omega
      refine le_antisymm ((value_runBind_le_mul (fun _ => periodG G (i + 1) T)
        (fun b s' => (periodRun G rec (i + 1) T s').map (rec i (Strategy.ofBind _ (G i) s) b :: ·))
        (fun l => M ≤ Served.lead l) (fun _ => True) 0 (G i) s
        (atEnds_of_forall (fun b s' => ?_) (G i) s)).trans_eq (mul_zero _)) zero_le
      rw [ite_self, Verity.Game.value_map]
      by_cases hm : (rec i (Strategy.ofBind _ (G i) s) b).sampledIncorrect
      · refine (Verity.Game.value_mono (q := fun _ => False) (fun l hl => ?_) _).trans_eq (Verity.Game.value_false _)
        simp only [Served.lead, ite_eq_left hm] at hl
        omega
      · have hb := hc i le_rfl (by omega) _ b hm
        refine (Verity.Game.value_mono (fun l hl => ?_) _).trans
          (value_lead_eq_zero G rec c T (i + 1) s' (M - c) (fun j hj hj' => hc j (by omega) (by omega)) hM').le
        simp only [Served.lead, ite_eq_right hm] at hl
        omega

/-- **The period leaves `T·(K − 1) + 1` incorrect proof blocks undetected with probability at most `Σ_t ε_t`**: a
window is accepted and sampled incorrect with probability at most its KS error, and under the cap the windows' lead
never reaches `T·(K − 1) + 1`. -/
theorem two_undetected [Fintype O] (H : Claims.Msg → Claims.Msg) (Lv : ℕ → CompiledTwo Coin A) {δ : ℝ} (hδ : 0 ≤ δ)
    (T K : ℕ) (hks : ∀ t < T, (Lv t).KS H)
    (href : ∀ t < T, ∀ x (P : Prover Unit) r, (run H P (Lv t).V x () r).out = .acc → (Lv t).Scope x)
    (hcap : ∀ t < T, CapTwo (Lv t) δ K) (ε : ℕ → ℝ)
    (s : Strategy (periodG (fun t => twoGame (Lv t) ((O → V) × Tick × Adv)) 0 T))
    (herr : ∀ t τ, PeriodPlays (fun t => twoGame (Lv t) ((O → V) × Tick × Adv)) 0 T s t τ →
      twoErr H (Lv t) δ τ ≤ ε t) :
    Verity.Game.value (Served.Undetected (T * (K - 1) + 1)) (twoRun H Lv δ T s) ≤
      ∑ t ∈ Finset.range T, ENNReal.ofReal (ε t) := by
  have hs : PeriodGood (fun t => twoGame (Lv t) ((O → V) × Tick × Adv))
      (fun t τ => t < T ∧ twoErr H (Lv t) δ τ ≤ ε t) 0 T s :=
    periodGood_mono _ T 0 s (fun j τ _ hj h => ⟨by omega, h⟩)
      (periodGood_of_plays (fun t => twoGame (Lv t) ((O → V) × Tick × Adv))
        (fun t τ => twoErr H (Lv t) δ τ ≤ ε t) T 0 s herr)
  have h1 := period_union (fun t => twoGame (Lv t) ((O → V) × Tick × Adv)) (fun t _ => twoRec H (Lv t) δ)
    (fun w => w.accepted ∧ w.sampledIncorrect) (fun t τ => t < T ∧ twoErr H (Lv t) δ τ ≤ ε t)
    (fun t => ENNReal.ofReal (ε t))
    (fun t τ hτ => (twoRec_accepted_le H (Lv t) hδ (hks t hτ.1) (href t hτ.1) τ).trans
      (ENNReal.ofReal_le_ofReal hτ.2)) T 0 s hs
  have h2 := value_lead_eq_zero (fun t => twoGame (Lv t) ((O → V) × Tick × Adv)) (fun t _ => twoRec H (Lv t) δ)
    (K - 1) T 0 s (T * (K - 1) + 1)
    (fun j _ hj _ b hb => twoRec_incorrect_le H (Lv j) (href j (by omega)) (hcap j (by omega)) b hb)
    (Nat.lt_succ_self _)
  simp only [Nat.zero_add] at h1
  show Verity.Game.value (fun l => (∃ r ∈ l, r.accepted ∧ r.sampledIncorrect) ∨ T * (K - 1) + 1 ≤ Served.lead l)
    (twoRun H Lv δ T s) ≤ _
  exact (Verity.Game.value_or_le _ _ _).trans ((add_le_add h1 h2.le).trans_eq (add_zero _))

/-- **`TheftServed` at a two-stage audit**: a party in `A` steals the weights through a period of `T` two-stage windows
with probability at most `|S|·p + Σ_t ε_t + Pr[¬Local]`, at the budget `T·(K − 1) + 1`, where `S` is the served
channel's covert strings, `ε_t` bounds window `t`'s KS error at `δ` and `K` is the per-window cap at `δ`. -/
theorem TheftServedTwo {Ω R Θ Q Θ' : Type} [Fintype Ω] [Fintype R] [Fintype Adv] [Inhabited Adv]
    [Fintype O] [DecidableEq O] [Fintype V] [Fintype Tick] [Inhabited Tick]
    (H : Claims.Msg → Claims.Msg) (Lv : ℕ → CompiledTwo Coin A) (T L E K : ℕ) {δ : ℝ} (hδ : 0 ≤ δ) (ε : ℕ → ℝ)
    (C : Θ → Q × Adv → O → V) (gen : Ω → Θ) (Stolen : Θ → Θ' → Prop) (Aset : Set (Outsider R (Q × Adv) (O → V) Θ'))
    (p : ℝ) (Ou : Outsider R (Q × Adv) (Option ((O → V) × Tick)) Θ')
    (s : Θ → R → Strategy (periodG (fun t => twoGame (Lv t) ((O → V) × Tick × Adv)) 0 T))
    (hhon : HonestUseHard C gen Stolen T Aset p)
    (hsim : Simulable Aset Ou
      (servedChannel R Q Adv O V Tick T (E * (T * (K - 1) + 1) + (L + 1) * Fintype.card O)))
    (hks : ∀ t < T, (Lv t).KS H)
    (href : ∀ t < T, ∀ x (P : Prover Unit) r, (run H P (Lv t).V x () r).out = .acc → (Lv t).Scope x)
    (hcap : ∀ t < T, CapTwo (Lv t) δ K)
    (herr : ∀ θ r t τ, PeriodPlays (fun t => twoGame (Lv t) ((O → V) × Tick × Adv)) 0 T (s θ r) t τ →
      twoErr H (Lv t) δ τ ≤ ε t) :
    Verity.Game.avg (fun z : Ω × R => Verity.Game.value
        (fun l => Stolen (gen z.1) (Ou.out z.2 (Served.view Ou L z.2 l T))) (twoRun H Lv δ T (s (gen z.1) z.2))) ≤
      ENNReal.ofReal (Fintype.card
          (servedChannel R Q Adv O V Tick T (E * (T * (K - 1) + 1) + (L + 1) * Fintype.card O)).S * p) +
        ∑ t ∈ Finset.range T, ENNReal.ofReal (ε t) +
        Verity.Game.avg (fun z : Ω × R => Verity.Game.value
          (fun l => ¬ Served.Local C E Ou L (gen z.1) z.2 l) (twoRun H Lv δ T (s (gen z.1) z.2))) :=
  TheftServed C gen Stolen Aset p Ou T L E (T * (K - 1) + 1) _ (fun θ r => twoRun H Lv δ T (s θ r)) hhon hsim
    fun θ r => two_undetected H Lv hδ T K hks href hcap ε (s θ r) (herr θ r)

/-- **The cap from the design's rates**: a subset-inputs first stage of `Q` among the population's replay blocks and a
per-cell second-stage rate `r₂` give `CapTwo` at every `K` with `(1 − (Q/N)·r₂)^K ≤ δ`. -/
theorem capTwo_of_rates (L : CompiledTwo Coin A) {δ r₂ : ℝ} {K : ℕ} (hr₂ : 0 ≤ r₂) (hr₂' : r₂ ≤ 1)
    (h : ∀ x, L.Scope x → ∀ s Pt d, L.parse x = some s → L.partition s = some Pt → L.two s = some d →
      ∃ Q c ic, SubsetInputsDesign d.replay Q ∧ d.part.classes = [c, ic] ∧ Q ≤ c.flatten.length ∧
        (1 - (Q : ℝ) / c.flatten.length * r₂) ^ K ≤ δ ∧
        ∀ ρ < d.part.blocks.length, ∀ t < Pt.strata.length, 0 < (cell Pt d.part ρ t).card →
          r₂ * (cell Pt d.part ρ t).card ≤ ((min (d.k.getD t 0) (cell Pt d.part ρ t).card : ℕ) : ℝ)) :
    CapTwo L δ K := by
  intro x hx s Pt d hs hPt hd fp hfp hcell
  obtain ⟨Q, c, ic, hD, hP, hQ, hK, hrate⟩ := h x hx s Pt d hs hPt hd
  exact d.card_wrongBlocks_lt_subsetInputs Pt hD hP hQ hr₂ hr₂' hrate hK hfp hcell

end WeightSecurity.SecurityProofs
