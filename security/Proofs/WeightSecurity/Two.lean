module

public import Proofs.Core.Game.Basic
public import Proofs.Core.Game.Interleave
public import Proofs.Core.Game.Tape
public import Proofs.Core.Game.Union
public import Proofs.Flock.Claims.Prob
public import Proofs.Flock.Claims.Sampling.Session
public import Proofs.Flock.EndToEnd.Period.Union
public import Proofs.Flock.SamplingGeneric.TwoCap
public import Proofs.WeightSecurity.Served

/-!
# Weight security at a two-stage audit

`TheftServed` over windows each audited by SPOT's two-stage verifier (`CompiledTwo`), each window's verifier knowledge
sound at the footprint (`CompiledTwo.KS`). A window (`twoGame`): the datacenter commits what leaves on the egress slots,
its timing bucket and its advice, then sends the window's statement and prover, and the verifier flips the run's live
coins. Its record (`twoRec`): accepted when the run accepts; sampled incorrect when it rejects or no extraction is in
`RTwo` at `δ`; incorrect, the fewest replay blocks such an extraction marks (`Design.Two.wrongBlocks`).

* `twoRec_accepted_le`: a window is accepted with no extraction in `RTwo` with probability at most `KSLinOn`'s bound at
  the statement and prover it sends, `κ_δ(x) + √(γ(x)·Pr[B]) + Pr[B_ext]` (`twoErr`), and never out of scope.
* `value_lead_eq_zero`: when every record not sampled incorrect counts at most `c`, the windows before the first
  sampled incorrect one count at most `T·c`.
* `two_undetected`: under `WindowCap` (a footprint in `M̄_δ` marks fewer than `K` replay blocks), the run leaves
  `M = T·(K − 1) + 1` incorrect replay blocks undetected with probability at most `Σ_t ε_t`.
* `TheftServedTwo`: `TheftServed` at that `M`. `windowCap_of_rates` gives `WindowCap` from the two stages' rates
  (`Design.Two.card_wrongBlocks_lt_subsetInputs`) at every `K` with `(1 − (Q/N)·r₂)^K ≤ δ`, so the budget grows as
  `T·K_δ`.
-/

@[expose] public section

namespace WeightSecurity.SecurityProofs

open Claims Claims.Sampling Claims.SamplingGeneric WeightSecurity.Assumptions
open FlockSoundness.Game hiding Msg
open Claims.Sampling.EndToEndDraw (SubsetInputsDesign)
open scoped ENNReal

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin} {O V Tick Adv : Type}

/-- A window's play at a two-stage audit: the statement and the prover the datacenter sent, and the run's coins. -/
abbrev TwoPlay (L : CompiledTwo Coin A) := (x : Msg) × Prover Unit × (Fin (L.V.k x) → Coin)

/-- **A window at a two-stage audit**: the datacenter commits the window's release, then sends the statement and the
prover, and the verifier flips the run's live coins. -/
noncomputable def twoGame (L : CompiledTwo Coin A) (Rls : Type) : Verity.Game (Rls × TwoPlay L) :=
  .send Rls fun m => .send (Msg × Prover Unit) fun xp =>
    .coin (Fin (L.V.k xp.1) → Coin) fun r => .ret (m, ⟨xp.1, xp.2, r⟩)

namespace TwoPlay

/-- The run accepts. -/
def accepted (H : Msg → Msg) {L : CompiledTwo Coin A} (p : TwoPlay L) : Prop :=
  (run H p.2.1 L.V p.1 () p.2.2).out = .acc

/-- The witness the extractor extracts on its coins `e`. -/
noncomputable def extracted (H : Msg → Msg) {L : CompiledTwo Coin A} (p : TwoPlay L) (e : L.E.C p.1) : Option Wit :=
  (L.E.eval H p.2.1 p.1 () p.2.2 e).1

/-- **The replay blocks an extraction marks incorrect**: their number, for each extractor coin whose extraction is in
`RTwo` at `δ` and each interior putting its footprint in `M̄_δ`. -/
def marks (H : Msg → Msg) {L : CompiledTwo Coin A} (δ : ℝ) (p : TwoPlay L) : Set ℕ :=
  {n | ∃ (e : L.E.C p.1) (w : Wit) (s : AuditStatement) (Pt : Partition) (d : Design.Two) (ι : ℕ → ℕ → Bool),
    p.extracted H e = some w ∧ L.parse p.1 = some s ∧ L.partition s = some Pt ∧ L.two s = some d ∧
      footprint H L.commit s Pt d.part w ι ∈ d.M Pt δ ∧
      (d.wrongBlocks Pt (footprint H L.commit s Pt d.part w ι)).card = n}

theorem marks_nonempty (H : Msg → Msg) {L : CompiledTwo Coin A} (δ : ℝ) (p : TwoPlay L) :
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
/-- **A two-stage window's record**: accepted when the run accepts; sampled incorrect when it rejects or no extraction
is in `RTwo` at `δ`; incorrect, the fewest replay blocks an extraction in `RTwo` marks, and every egress slot when there
is none; and the release. -/
noncomputable def twoRec [Fintype O] (H : Msg → Msg) (L : CompiledTwo Coin A) (δ : ℝ)
    (b : ((O → V) × Tick × Adv) × TwoPlay L) : Served O V Tick Adv where
  accepted := b.2.accepted H
  sampledIncorrect := ¬ (b.2.accepted H ∧ (b.2.marks H δ).Nonempty)
  incorrect := if b.2.accepted H ∧ (b.2.marks H δ).Nonempty then sInf (b.2.marks H δ) else Fintype.card O
  egress := b.1.1
  tick := b.1.2.1
  adv := b.1.2.2

/-- **The deployment's run over `T` two-stage windows** at the datacenter's strategy `s`, each window recorded by
`twoRec`. -/
noncomputable def twoRun [Fintype O] (H : Msg → Msg) (Lv : ℕ → CompiledTwo Coin A) (δ : ℝ) (T : ℕ)
    (s : Strategy (periodG (fun t => twoGame (Lv t) ((O → V) × Tick × Adv)) 0 T)) :
    Verity.Game (List (Served O V Tick Adv)) :=
  periodRun (fun t => twoGame (Lv t) ((O → V) × Tick × Adv)) (fun t _ => twoRec H (Lv t) δ) 0 T s

/-- **A window's error at its strategy**: `KSLinOn`'s bound at the statement `x` and prover `P` it sends,
`κ_δ(x) + √(γ(x)·Pr[B]) + Pr[B_ext]`. -/
noncomputable def twoErr (H : Msg → Msg) (L : CompiledTwo Coin A) (δ : ℝ) {Rls : Type}
    (τ : Strategy (twoGame L Rls)) : ℝ :=
  L.κ δ τ.2.1.1 + Real.sqrt (L.γ τ.2.1.1 * PrB H L.V τ.2.1.2 τ.2.1.1) + (Bext H L.V L.E Unit τ.2.1.2 τ.2.1.1).Pr H

/-- The run on coins `r` accepts and no extractor coin extracts a witness in `RTwo` at `δ`. -/
def twoMiss (H : Msg → Msg) (L : CompiledTwo Coin A) (δ : ℝ) (x : Msg) (P : Prover Unit)
    (r : Fin (L.V.k x) → Coin) : Prop :=
  (run H P L.V x () r).out = .acc ∧
    ∀ e, ¬ (RTwo H L.commit L.parse L.partition L.two δ).ext x (L.E.eval H P x () r e).1

/-- **The cap at `δ`**: at every statement in scope, a footprint in `M̄_δ` that counts at most each cell marks fewer
than `K` replay blocks incorrect. -/
def WindowCap (L : CompiledTwo Coin A) (δ : ℝ) (K : ℕ) : Prop :=
  ∀ x s Pt d, L.parse x = some s → L.partition s = some Pt → L.two s = some d → InScopeTwo Pt d s →
    ∀ fp ∈ d.M Pt δ, (∀ ρ t, fp ρ t ≤ (cell Pt d.part ρ t).card) → (d.wrongBlocks Pt fp).card < K

theorem twoRec_accepted_iff [Fintype O] (H : Msg → Msg) (L : CompiledTwo Coin A) (δ : ℝ)
    (b : ((O → V) × Tick × Adv) × TwoPlay L) :
    ((twoRec H L δ b).accepted ∧ (twoRec H L δ b).sampledIncorrect) ↔
      b.2.accepted H ∧ ∀ e, ¬ (RTwo H L.commit L.parse L.partition L.two δ).ext b.2.1 (b.2.extracted H e) := by
  show (b.2.accepted H ∧ ¬ (b.2.accepted H ∧ (b.2.marks H δ).Nonempty)) ↔ _
  rw [TwoPlay.marks_nonempty]
  exact ⟨fun ⟨ha, hn⟩ => ⟨ha, fun e he => hn ⟨ha, e, he⟩⟩, fun ⟨ha, hn⟩ => ⟨ha, fun ⟨_, e, he⟩ => hn e he⟩⟩

omit [Nonempty Coin] in
/-- **Every extractor coin failing on an accepted run** is at most as likely as one uniform extractor coin failing:
`PrBad` at a prover with no tape. -/
theorem pr_acc_noext_le (H : Msg → Msg) {W : Type} (V : Verifier Coin) (R : Rel W) (E : Extractor Coin W V)
    (x : Msg) (P : Prover Unit) :
    Pr (fun r : Fin (V.k x) → Coin => (run H P V x () r).out = .acc ∧ ∀ e, ¬ R.ext x (E.eval H P x () r e).1) ≤
      PrBad H V R E x P := by
  calc Pr (fun r : Fin (V.k x) → Coin => (run H P V x () r).out = .acc ∧ ∀ e, ¬ R.ext x (E.eval H P x () r e).1)
      = Pr (fun s : Unit × (Fin (V.k x) → Coin) × E.C x =>
          (run H P V x () s.2.1).out = .acc ∧ ∀ e, ¬ R.ext x (E.eval H P x () s.2.1 e).1) := by
        rw [Pr_unit (fun s : (Fin (V.k x) → Coin) × E.C x =>
            (run H P V x () s.1).out = .acc ∧ ∀ e, ¬ R.ext x (E.eval H P x () s.1 e).1),
          Pr_fst (fun r : Fin (V.k x) → Coin =>
            (run H P V x () r).out = .acc ∧ ∀ e, ¬ R.ext x (E.eval H P x () r e).1)]
    _ ≤ PrBad H V R E x P := Pr_mono fun s hs => ⟨hs.1, hs.2 s.2.2⟩

/-- **A window is accepted with no extraction in `RTwo` at most as often as `KSLinOn` bounds**: at a statement in
scope by `CompiledTwo.KS` at a prover with no tape, and out of scope never, since the verifier refuses it. -/
theorem twoRec_accepted_le [Fintype O] (H : Msg → Msg) (L : CompiledTwo Coin A) {δ : ℝ} (hδ : 0 ≤ δ) (hks : L.KS H)
    (href : ∀ x (P : Prover Unit) r, (run H P L.V x () r).out = .acc → L.Scope x)
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

/-- **A record not sampled incorrect counts fewer than `K`** under `WindowCap`: its run accepts, so its statement is in
scope, and the fewest replay blocks its extractions mark is one of them. -/
theorem twoRec_incorrect_le [Fintype O] (H : Msg → Msg) (L : CompiledTwo Coin A) {δ : ℝ} {K : ℕ}
    (href : ∀ x (P : Prover Unit) r, (run H P L.V x () r).out = .acc → L.Scope x) (hcap : WindowCap L δ K)
    (b : ((O → V) × Tick × Adv) × TwoPlay L) (hb : ¬ (twoRec H L δ b).sampledIncorrect) :
    (twoRec H L δ b).incorrect ≤ K - 1 := by
  classical
  have hb' : b.2.accepted H ∧ (b.2.marks H δ).Nonempty := not_not.1 hb
  show (if b.2.accepted H ∧ (b.2.marks H δ).Nonempty then sInf (b.2.marks H δ) else Fintype.card O) ≤ K - 1
  rw [ite_eq_left hb']
  obtain ⟨e, w, s, Pt, d, ι, -, hs, hPt, hd, hM, hn⟩ := Nat.sInf_mem hb'.2
  obtain ⟨s', Pt', d', hs', hPt', hd', hin⟩ := href b.2.1 b.2.2.1 b.2.2.2 hb'.1
  rw [hs] at hs'
  cases hs'
  rw [hPt] at hPt'
  cases hPt'
  rw [hd] at hd'
  cases hd'
  have h := hcap b.2.1 s Pt d hs hPt hd hin _ hM fun ρ t => footprint_le_card H L.commit s Pt d.part w ι ρ t
  omega

/-- **No run reaches a lead above `T·c`** when every record not sampled incorrect, at the windows the run plays, counts
at most `c`: the windows before the first sampled incorrect one count at most `c` each. -/
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
        have := Nat.add_one_mul T c
        omega
      refine le_antisymm ((value_runBind_le_mul (fun _ => periodG G (i + 1) T)
        (fun b s' => (periodRun G rec (i + 1) T s').map (rec i (Strategy.ofBind _ (G i) s) b :: ·))
        (fun l => M ≤ Served.lead l) (fun _ => True) 0 (G i) s
        (atEnds_of_forall (fun b s' => ?_) (G i) s)).trans_eq (mul_zero _)) zero_le
      rw [ite_eq_left trivial, Verity.Game.value_map]
      by_cases hm : (rec i (Strategy.ofBind _ (G i) s) b).sampledIncorrect
      · refine (Verity.Game.value_mono (fun l hl => ?_) _).trans (Verity.Game.value_false _).le
        simp only [Served.lead, ite_eq_left hm] at hl
        exact absurd (lt_of_lt_of_le hpos hl) (lt_irrefl 0)
      · have hb := hc i le_rfl (by omega) _ b hm
        refine (Verity.Game.value_mono (fun l hl => ?_) _).trans
          (value_lead_eq_zero G rec c T (i + 1) s' (M - c) (fun j hj hj' => hc j (by omega) (by omega)) hM').le
        simp only [Served.lead, ite_eq_right hm] at hl
        omega

/-- **The run leaves `T·(K − 1) + 1` incorrect replay blocks undetected with probability at most `Σ_t ε_t`**: an
accepted window with no extraction in `RTwo` at `δ` has probability at most its error (`twoRec_accepted_le`), and
outside those windows each window before the first sampled incorrect one marks fewer than `K` (`WindowCap`). -/
theorem two_undetected [Fintype O] (H : Msg → Msg) (Lv : ℕ → CompiledTwo Coin A) {δ : ℝ} (hδ : 0 ≤ δ) (T K : ℕ)
    (hks : ∀ t < T, (Lv t).KS H)
    (href : ∀ t < T, ∀ x (P : Prover Unit) r, (run H P (Lv t).V x () r).out = .acc → (Lv t).Scope x)
    (hcap : ∀ t < T, WindowCap (Lv t) δ K) (ε : ℕ → ℝ)
    (s : Strategy (periodG (fun t => twoGame (Lv t) ((O → V) × Tick × Adv)) 0 T))
    (herr : ∀ t τ, PeriodPlays (fun t => twoGame (Lv t) ((O → V) × Tick × Adv)) 0 T s t τ →
      twoErr H (Lv t) δ τ ≤ ε t) :
    Verity.Game.value (Served.Undetected (T * (K - 1) + 1)) (twoRun H Lv δ T s) ≤
      ∑ t ∈ Finset.range T, ENNReal.ofReal (ε t) := by
  have hs : PeriodGood (fun t => twoGame (Lv t) ((O → V) × Tick × Adv))
      (fun t τ => t < T ∧ twoErr H (Lv t) δ τ ≤ ε t) 0 T s :=
    monitoringPeriodGood_mono _ T 0 s (fun j τ _ hj h => ⟨by omega, h⟩)
      (monitoringPeriodGood_of_plays (fun t => twoGame (Lv t) ((O → V) × Tick × Adv))
        (fun t τ => twoErr H (Lv t) δ τ ≤ ε t) T 0 s herr)
  have h1 := monitoringPeriod_union (fun t => twoGame (Lv t) ((O → V) × Tick × Adv)) (fun t _ => twoRec H (Lv t) δ)
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
  refine (Verity.Game.value_or_le _ _ _).trans ?_
  rw [show twoRun H Lv δ T s = periodRun (fun t => twoGame (Lv t) ((O → V) × Tick × Adv))
    (fun t _ => twoRec H (Lv t) δ) 0 T s from rfl, h2, add_zero]
  exact h1

/-- **`TheftServed` at a two-stage audit**: with each window's verifier knowledge sound at the footprint, refusing what
is out of scope, and capped at `K` (`WindowCap`), the outsider steals the weights with probability at most the channel's
`|S|·p` at the budget `E·M + (L + 1)·|O|`, `M = T·(K − 1) + 1`, plus `Σ_t ε_t`, each `ε_t` bounding window `t`'s
`KSLinOn` error at every strategy the run plays it with, plus the probability that a window changes more slots than
`E` per incorrect replay block. -/
theorem TheftServedTwo {Ω R Θ Q Θ' : Type} [Fintype Ω] [Fintype R] [Fintype Adv] [Inhabited Adv]
    [Fintype O] [DecidableEq O] [Fintype V] [Fintype Tick] [Inhabited Tick]
    (H : Msg → Msg) (Lv : ℕ → CompiledTwo Coin A) (T L E K : ℕ) {δ : ℝ} (hδ : 0 ≤ δ) (ε : ℕ → ℝ)
    (C : Θ → Q × Adv → O → V) (gen : Ω → Θ) (Stolen : Θ → Θ' → Prop) (Aset : Set (Outsider R (Q × Adv) (O → V) Θ'))
    (p : ℝ) (Ou : Outsider R (Q × Adv) (Option ((O → V) × Tick)) Θ')
    (s : Θ → R → Strategy (periodG (fun t => twoGame (Lv t) ((O → V) × Tick × Adv)) 0 T))
    (hhon : HonestUseHard C gen Stolen T Aset p)
    (hsim : Simulable Aset Ou
      (servedChannel R Q Adv O V Tick T (E * (T * (K - 1) + 1) + (L + 1) * Fintype.card O)))
    (hks : ∀ t < T, (Lv t).KS H)
    (href : ∀ t < T, ∀ x (P : Prover Unit) r, (run H P (Lv t).V x () r).out = .acc → (Lv t).Scope x)
    (hcap : ∀ t < T, WindowCap (Lv t) δ K)
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

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **The cap from the two stages' rates**: where each statement in scope draws its replay sample as a uniform
`Q`-subset of its `N` correct-class replay blocks and every input replay block, and its second stage samples at least
`r₂` of every cell, `WindowCap` holds at every `K` with `(1 − (Q/N)·r₂)^K ≤ δ`. -/
theorem windowCap_of_rates (L : CompiledTwo Coin A) {δ r₂ : ℝ} {K : ℕ} (hr₂ : 0 ≤ r₂) (hr₂' : r₂ ≤ 1)
    (h : ∀ x s Pt d, L.parse x = some s → L.partition s = some Pt → L.two s = some d → InScopeTwo Pt d s →
      ∃ Q c ic, SubsetInputsDesign d.replay Q ∧ d.part.classes = [c, ic] ∧ Q ≤ c.flatten.length ∧
        (1 - (Q : ℝ) / c.flatten.length * r₂) ^ K ≤ δ ∧
        ∀ ρ < d.part.blocks.length, ∀ t < Pt.strata.length, 0 < (cell Pt d.part ρ t).card →
          r₂ * (cell Pt d.part ρ t).card ≤ ((min (d.k.getD t 0) (cell Pt d.part ρ t).card : ℕ) : ℝ)) :
    WindowCap L δ K := by
  intro x s Pt d hs hPt hd hin fp hfp hcell
  obtain ⟨Q, c, ic, hD, hP, hQ, hK, hrate⟩ := h x s Pt d hs hPt hd hin
  exact d.card_wrongBlocks_lt_subsetInputs Pt hD hP hQ hr₂ hr₂' hrate hK hfp hcell

end WeightSecurity.SecurityProofs
