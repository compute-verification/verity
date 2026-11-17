module

public import Proofs.Flock.Claims.KS.Joint
public import Proofs.Flock.Claims.KS.Flock
public import Proofs.Flock.Claims.Recursion.AbortWait
public import Proofs.Flock.Claims.Recursion.AbortFork
public import Proofs.Flock.Claims.Recursion.InnerAbort

/-! Flock meets `Inner.KSAbort` from the fixed-table bound at the abort distribution (`KSAbort_of_tableBoundKept`),
which `TableBoundθ` alone doesn't give (`flockKS_not_KSAbort`). -/

@[expose] public section

namespace Claims.Recursion

open Claims Claims.Flock FlockSoundness FlockSoundness.Game FlockSoundness.Extract FlockSoundness.Rewinding

section SampleGame

variable {Dr : Type} [Fintype Dr] [Nonempty Dr]

def sampleGame (Dr : Type) [Fintype Dr] [Nonempty Dr] : Game Dr := .coin Dr .ret

def drawS (Dr : Type) [Fintype Dr] [Nonempty Dr] : Strategy (sampleGame Dr) := fun _ => PUnit.unit

theorem expect_sampleGame (f : Dr → ℝ) : expect f (sampleGame Dr) (drawS Dr) = Ex f := rfl

theorem expectN_sampleGame : ∀ (k : ℕ) (F : List Dr → ℝ),
    expectN (sampleGame Dr) (drawS Dr) k F = Ex fun σ : Fin k → Dr => F ((List.finRange k).map σ)
  | 0, F => by
    simp only [expectN, List.finRange_zero, List.map_nil]
    exact (Ex_const _).symm
  | k + 1, F => by
    rw [expectN, expect_sampleGame, Claims.Bridge.Ex_cons]
    refine congrArg Ex (funext fun c => ?_)
    rw [expectN_sampleGame k]
    refine congrArg Ex (funext fun r => ?_)
    simp [List.finRange_succ, List.map_cons, List.map_map, Function.comp_def]

def AccD (tr : Dr → Transcript) (d : Dr) : Prop := (tr d).out = .acc

theorem accPrD_eq (tr : Dr → Transcript) : accPrD tr = expect (ind (AccD tr)) (sampleGame Dr) (drawS Dr) := by
  rw [expect_sampleGame]
  exact Claims.Bridge.Pr_eq_Ex _

theorem accFailD_eq (tr : Dr → Transcript) (k : ℕ) (G : List Seen → Prop) :
    accFailD tr k G = expectN (sampleGame Dr) (drawS Dr) k fun bs =>
      (bs.map (ind (AccD tr))).prod * ind (fun bs => ¬ G (bs.map fun d => (tr d).seen)) bs := by
  classical
  rw [expectN_sampleGame]
  unfold accFailD
  rw [Claims.Bridge.Pr_eq_Ex]
  refine congrArg Ex (funext fun σ => ?_)
  rw [prod_map_ind]
  have hseen : ((List.finRange k).map σ).map (fun d => (tr d).seen) =
      (List.finRange k).map fun i => (tr (σ i)).seen := by
    simp [List.map_map, Function.comp_def]
  unfold ind
  beta_reduce
  rw [hseen]
  by_cases h1 : ∀ i, (tr (σ i)).out = .acc <;>
    by_cases h2 : G ((List.finRange k).map fun i => (tr (σ i)).seen) <;> simp [AccD, h1, h2]

theorem oneWaitD_lhs_eq (tr : Dr → Transcript) (k : ℕ) (G : List Seen → Prop) :
    accPrD tr * (accFailD tr k G / accPrD tr ^ k) =
      expect (ind (AccD tr)) (sampleGame Dr) (drawS Dr) *
        expectAcc (sampleGame Dr) (drawS Dr) (AccD tr) k (ind fun bs => ¬ G (bs.map fun d => (tr d).seen)) := by
  rw [accFailD_eq, accPrD_eq]
  rfl

end SampleGame

section OneWaitD

variable {Dr Coin Col T : Type} [Fintype Dr] [Nonempty Dr] [DecidableEq Col] [Zero Col]

open Classical in
noncomputable def opD (D : Flock.Deploy Coin Col T) (N : Msg → ℕ) (x : Msg) (tr : Dr → Transcript) (d : Dr)
    (p : Fin (N x)) : Option Col :=
  if (tr d).out = .acc then lookCol (D.cols (tr d).seen 0) p.val else none

omit [Fintype Dr] [Nonempty Dr] [DecidableEq Col] in
theorem firstTable_opD (D : Flock.Deploy Coin Col T) (N : Msg → ℕ) (x : Msg) (tr : Dr → Transcript) (bs : List Dr)
    (hacc : ∀ b ∈ bs, AccD tr b) (p : Fin (N x)) :
    firstTable (opD D N x tr) (fun _ => 0) bs p =
      (lookCol ((bs.map fun d => (tr d).seen).flatMap fun v => D.cols v 0) p.val).getD 0 := by
  unfold firstTable
  rw [List.flatMap_map,
    findSome?_lookCol _ (fun d => D.cols (tr d).seen 0) p.val bs fun b hb => by
      unfold opD
      simp only [show (tr b).out = .acc from hacc b hb, ↓reduceIte]]

theorem oneWaitD (D : Flock.Deploy Coin Col T) (N : Msg → ℕ) (x : Msg) (tr : Dr → Transcript) {s : Flock.Stmt}
    (hs : D.parse x = some s) (h1 : s.tables = 1) (Ok : Msg → Col → Prop) (hz : Ok x 0)
    (hok : ∀ d q u, (tr d).out = .acc → q < N x → lookCol (D.cols (tr d).seen 0) q = some u → Ok x u)
    (εc : ℝ) (hεc : 0 ≤ εc)
    (hb : ∀ C : Fin (N x) → Col, (∀ p, Ok x (C p)) → ¬ GoodTable D N x C →
      accPrD tr ≤ εc + Pr fun d => (tr d).out = .acc ∧
        ∃ (p : Fin (N x)) (u : Col), lookCol (D.cols (tr d).seen 0) p.val = some u ∧ u ≠ C p)
    (k : ℕ) (hk : 1 ≤ k) :
    accPrD tr * (accFailD tr k (Extracts D x) / accPrD tr ^ k) ≤
      εc + N x * accPrD tr / (Real.exp 1 * k) +
        Real.sqrt (k * Pr fun s : Dr × Dr => ∃ p, ConflictAt (opD D N x tr) s p) := by
  classical
  set Xt : List Dr → Prop := fun bs =>
    (∀ b ∈ bs, AccD tr b) → Extracts D x (bs.map fun d => (tr d).seen) with hXt
  have hXb : ∀ bs, ¬ Xt bs → expect (ind (AccD tr)) (sampleGame Dr) (drawS Dr) ≤
      εc + expect (ind fun d => AccD tr d ∧
        ∃ p, Off (opD D N x tr) (firstTable (opD D N x tr) (fun _ => 0) bs) d p) (sampleGame Dr) (drawS Dr) := by
    intro bs hnX
    simp only [hXt, Classical.not_imp] at hnX
    obtain ⟨hall, hnot⟩ := hnX
    have hft : firstTable (opD D N x tr) (fun _ => 0) bs =
        fun p => (lookCol ((bs.map fun d => (tr d).seen).flatMap fun v => D.cols v 0) p.val).getD 0 :=
      funext (firstTable_opD D N x tr bs hall)
    have hgood : ¬ GoodTable D N x (firstTable (opD D N x tr) (fun _ => 0) bs) := by
      intro hg
      rw [hft] at hg
      exact hnot (extracts_of_good D N x _ hs h1 hg)
    have hOk : ∀ p, Ok x (firstTable (opD D N x tr) (fun _ => 0) bs p) := by
      intro p
      rw [firstTable_opD D N x tr bs hall p]
      cases hl : lookCol ((bs.map fun d => (tr d).seen).flatMap fun v => D.cols v 0) p.val with
      | none => exact hz
      | some u =>
        rw [List.flatMap_map] at hl
        obtain ⟨b, hb', hu⟩ := lookCol_flatMap_some (fun d => D.cols (tr d).seen 0) p.val u bs hl
        exact hok b p.val u (hall b hb') p.isLt hu
    rw [← accPrD_eq, expect_sampleGame, ← Claims.Bridge.Pr_eq_Ex]
    refine (hb _ hOk hgood).trans (le_of_eq ?_)
    congr 1
    refine Pr_congr fun d => ?_
    unfold AccD Off opD
    constructor
    · rintro ⟨ha, p, u, hu, hne⟩
      refine ⟨ha, p, ?_⟩
      rw [ite_eq_left ha, hu]
      exact ⟨rfl, fun h => hne (Option.some_injective _ h)⟩
    · rintro ⟨ha, p, h1, h2⟩
      rw [ite_eq_left ha] at h1 h2
      obtain ⟨u, hu⟩ := Option.isSome_iff_exists.1 h1
      exact ⟨ha, p, u, hu, fun h => h2 (by rw [hu, h])⟩
  have hj := joint_acc_le (opD D N x tr) (fun _ => 0) (sampleGame Dr) (drawS Dr) (AccD tr) Xt εc hεc hXb k hk
  have hlhs : accPrD tr * (accFailD tr k (Extracts D x) / accPrD tr ^ k) =
      expect (ind (AccD tr)) (sampleGame Dr) (drawS Dr) *
        expectAcc (sampleGame Dr) (drawS Dr) (AccD tr) k (ind fun bs => ¬ Xt bs) := by
    rw [oneWaitD_lhs_eq]
    congr 1
    unfold expectAcc
    congr 1
    refine congrArg (expectN _ _ _) (funext fun bs => ?_)
    rw [prod_map_ind]
    by_cases hall : ∀ b ∈ bs, AccD tr b
    · congr 1
      unfold ind
      have : (¬ Xt bs) ↔ ¬ Extracts D x (bs.map fun d => (tr d).seen) := by
        rw [hXt]
        exact not_congr ⟨fun h => h hall, fun h _ => h⟩
      beta_reduce
      split_ifs with h1 h2 h2 <;> first | rfl | (exfalso; tauto)
    · rw [ite_eq_right hall, zero_mul, zero_mul]
  have hE₂ : expect₂ (fun a a' => ind (fun y : Dr × Dr => ∃ p, ConflictAt (opD D N x tr) y p) (a, a'))
      (sampleGame Dr) (drawS Dr) = Pr fun s : Dr × Dr => ∃ p, ConflictAt (opD D N x tr) s p := by
    unfold expect₂
    simp only [expect_sampleGame]
    rw [Claims.Bridge.Pr_eq_Ex, Claims.Bridge.Ex_prod]
  rw [hE₂, ← accPrD_eq, Fintype.card_fin] at hj
  rw [hlhs, ← accPrD_eq]
  exact hj

end OneWaitD

section ForkD

variable {Ω Coin A : Type} {H : Msg → Msg} {n : ℕ} {tr : Ω → (Fin n → Coin) × A → Transcript}

def EscapesD.addFork (e : EscapesD H n tr) (c : ℝ) (hc : 0 ≤ c) (j₀ : Fin n)
    (ev : Ω → (Fin n → Coin) × A → (Fin n → Coin) × A → Prop)
    (tied : ∀ ω d d', ev ω d d' → TiedColl H ((tr ω d).vLog ++ (tr ω (forkSample j₀ d d')).vLog)) :
    EscapesD H n tr where
  I := Option e.I
  c i := i.elim c e.c
  j i := i.elim j₀ e.j
  ev i := i.elim ev e.ev
  L := e.L
  β := e.β
  ev₁ := e.ev₁
  c_nonneg i := by cases i; exacts [hc, e.c_nonneg _]
  β_nonneg := e.β_nonneg
  tied i := by cases i; exacts [tied, e.tied _]
  tied₁ := e.tied₁

def EscapesD.FitsFork (e : EscapesD H n tr) (C c : ℝ) : Prop :=
  ∃ mult : ℕ, (∀ r : Fin n, (Finset.univ.filter fun i => e.j i = r).card + (if r.val = 0 then 1 else 0) ≤ mult) ∧
    mult * (c + ∑ i, e.c i) ≤ 2 * C ∧ ∑ l, e.β l ≤ n

theorem EscapesD.term_addFork [Fintype Coin] [Fintype A] (e : EscapesD H n tr) (c : ℝ) (hc : 0 ≤ c) (j₀ : Fin n)
    (ev : Ω → (Fin n → Coin) × A → (Fin n → Coin) × A → Prop) (tied) (ω : Ω) :
    (e.addFork c hc j₀ ev tied).term ω =
      Real.sqrt (c * Pr fun s : ((Fin n → Coin) × A) × ((Fin n → Coin) × A) => ev ω s.1 s.2) + e.term ω := by
  have h1 : (∑ i : (e.addFork c hc j₀ ev tied).I, Real.sqrt ((e.addFork c hc j₀ ev tied).c i *
        Pr fun s : ((Fin n → Coin) × A) × ((Fin n → Coin) × A) => (e.addFork c hc j₀ ev tied).ev i ω s.1 s.2)) =
      Real.sqrt (c * Pr fun s : ((Fin n → Coin) × A) × ((Fin n → Coin) × A) => ev ω s.1 s.2) +
        ∑ i, Real.sqrt (e.c i * Pr fun s : ((Fin n → Coin) × A) × ((Fin n → Coin) × A) => e.ev i ω s.1 s.2) :=
    sum_option_any _ fun i : Option e.I => Real.sqrt (i.elim c e.c *
      Pr fun s : ((Fin n → Coin) × A) × ((Fin n → Coin) × A) => (i.elim ev e.ev) ω s.1 s.2)
  unfold EscapesD.term
  rw [h1, add_assoc]
  rfl

theorem EscapesD.fits_addFork (e : EscapesD H n tr) (C c : ℝ) (hc : 0 ≤ c) (j₀ : Fin n) (hj₀ : j₀.val = 0)
    (ev : Ω → (Fin n → Coin) × A → (Fin n → Coin) × A → Prop) (tied) (h : e.FitsFork C c) :
    (e.addFork c hc j₀ ev tied).Fits C := by
  classical
  obtain ⟨mult, hm, hmc, hβ⟩ := h
  refine ⟨mult, fun r => ?_, ?_, hβ⟩
  · refine le_trans (le_of_eq ?_) (hm r)
    refine (card_filter_option _ (fun i : Option e.I => i.elim j₀ e.j) r).trans ?_
    by_cases h0 : r.val = 0
    · rw [ite_eq_left (show (none : Option e.I).elim j₀ e.j = r from Fin.ext (hj₀.trans h0.symm)),
        ite_eq_left h0]
      exact Nat.add_comm _ _
    · rw [ite_eq_right (show ¬ (none : Option e.I).elim j₀ e.j = r from fun h => h0 (by
          rw [← (h : j₀ = r)]; exact hj₀)), ite_eq_right h0]
      exact (Nat.zero_add _).trans (Nat.add_zero _).symm
  · show mult * ∑ i : Option e.I, i.elim c e.c ≤ 2 * C
    rw [sum_option_any]
    exact hmc

end ForkD

section AbortSamplingDesign

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

omit [Fintype Coin] [Nonempty Coin] [DecidableEq Col] [Zero Col] in
theorem abortRun_acc_iff {H : Msg → Msg} {Ω A : Type} {P : Prover Ω} {V : Verifier Coin} {x : Msg}
    {keep : Ω → (Fin (V.k x) → Coin) → A → Bool} {ω : Ω} {d : (Fin (V.k x) → Coin) × A} :
    (abortRun H P V x keep ω d).out = .acc ↔ keep ω d.1 d.2 = true ∧ (run H P V x ω d.1).out = .acc := by
  unfold abortRun
  by_cases h : keep ω d.1 d.2 = true
  · simp [h]
  · simp [h, abortTranscript]

def TableBoundθAbort (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (N : Msg → ℕ) (Ok : Msg → Col → Prop) : Prop :=
  ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (A : Type) [Fintype A] [Nonempty A]
    (keep : Ω → (Fin (D.V.k x) → Coin) → A → Bool),
    ∃ e : EscapesD H (D.V.k x) (abortRun H P D.V x keep), e.FitsFork (D.weight x) (D.k x) ∧
      ∀ (ω : Ω) (C : Fin (N x) → Col), (∀ p, Ok x (C p)) → ¬ GoodTable D N x C →
        accPrD (abortRun H P D.V x keep ω) ≤ D.εtable x + e.term ω +
          Pr fun d : (Fin (D.V.k x) → Coin) × A => (abortRun H P D.V x keep ω d).out = .acc ∧
            ∃ (p : Fin (N x)) (u : Col),
              lookCol (D.cols (abortRun H P D.V x keep ω d).seen 0) p.val = some u ∧ u ≠ C p

def TableBoundθKept (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (N : Msg → ℕ) (Ok : Msg → Col → Prop) : Prop :=
  ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (A : Type) [Fintype A] [Nonempty A]
    (keep : Ω → (Fin (D.V.k x) → Coin) → A → Bool),
    ∃ e : EscapesD H (D.V.k x) (fun ω d => run H P D.V x ω d.1), e.Kept keep ∧ e.FitsFork (D.weight x) (D.k x) ∧
      ∀ (ω : Ω) (C : Fin (N x) → Col), (∀ p, Ok x (C p)) → ¬ GoodTable D N x C →
        Pr (fun d : (Fin (D.V.k x) → Coin) × A => keep ω d.1 d.2 = true ∧ (run H P D.V x ω d.1).out = .acc) ≤
          D.εtable x + e.term ω +
            Pr fun d : (Fin (D.V.k x) → Coin) × A => keep ω d.1 d.2 = true ∧ (run H P D.V x ω d.1).out = .acc ∧
              ∃ (p : Fin (N x)) (u : Col), lookCol (D.cols (run H P D.V x ω d.1).seen 0) p.val = some u ∧ u ≠ C p

omit [Nonempty Coin] in
theorem tableBoundθAbort_of_kept (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (N : Msg → ℕ)
    (Ok : Msg → Col → Prop) (h : TableBoundθKept H D N Ok) : TableBoundθAbort H D N Ok := by
  intro x Ω _ _ P A _ _ keep
  obtain ⟨e, hk, hfit, hb⟩ := h x Ω P A keep
  refine ⟨e.toAbort hk, hfit, fun ω C hC hg => ?_⟩
  have h1 : accPrD (abortRun H P D.V x keep ω) =
      Pr fun d : (Fin (D.V.k x) → Coin) × A => keep ω d.1 d.2 = true ∧ (run H P D.V x ω d.1).out = .acc :=
    Pr_congr fun _ => abortRun_acc_iff
  rw [h1, EscapesD.term_toAbort]
  refine (hb ω C hC hg).trans (le_of_eq (congrArg _ (Pr_congr fun d => ?_)))
  constructor
  · rintro ⟨hkeep, ha, rest⟩
    rw [abortRun_of_keep hkeep]
    exact ⟨ha, rest⟩
  · rintro ⟨ha, rest⟩
    obtain ⟨hkeep, ha'⟩ := abortRun_acc_iff.1 ha
    rw [abortRun_of_keep hkeep] at rest
    exact ⟨hkeep, ha', rest⟩

theorem Pr_unit {R : Type} [Fintype R] (p : R → Prop) : Pr (fun d : R × Unit => p d.1) = Pr p :=
  Pr_equiv (Equiv.prodPUnit R) p

theorem Pr_unit₂ {R : Type} [Fintype R] (p : R → R → Prop) :
    Pr (fun s : (R × Unit) × (R × Unit) => p s.1.1 s.2.1) = Pr fun s : R × R => p s.1 s.2 :=
  Pr_equiv ((Equiv.prodPUnit R).prodCongr (Equiv.prodPUnit R)) fun s : R × R => p s.1 s.2

def EscapesD.toEscapes {H : Msg → Msg} {Ω : Type} {V : Verifier Coin} {P : Prover Ω} {x : Msg}
    (e : EscapesD H (V.k x) fun ω (d : (Fin (V.k x) → Coin) × Unit) => run H P V x ω d.1) : Escapes H V P x where
  I := e.I
  c := e.c
  j := e.j
  ev i ω r r' := e.ev i ω (r, ()) (r', ())
  L := e.L
  β := e.β
  ev₁ l ω r := e.ev₁ l ω (r, ())
  c_nonneg := e.c_nonneg
  β_nonneg := e.β_nonneg
  tied i ω r r' h := e.tied i ω (r, ()) (r', ()) h
  tied₁ l ω r h := e.tied₁ l ω (r, ()) h

omit [Nonempty Coin] [DecidableEq Col] [Zero Col] in
theorem EscapesD.term_toEscapes {H : Msg → Msg} {Ω : Type} {V : Verifier Coin} {P : Prover Ω} {x : Msg}
    (e : EscapesD H (V.k x) fun ω (d : (Fin (V.k x) → Coin) × Unit) => run H P V x ω d.1) (ω : Ω) :
    e.toEscapes.term ω = e.term ω := by
  unfold Escapes.term EscapesD.term
  congr 1
  · refine Finset.sum_congr rfl fun i _ => ?_
    congr 2
    exact (Pr_unit₂ fun r r' => e.ev i ω (r, ()) (r', ())).symm
  · refine Finset.sum_congr rfl fun l _ => ?_
    congr 1
    exact (Pr_unit fun r => e.ev₁ l ω (r, ())).symm

omit [Nonempty Coin] in
theorem tableBoundθ_of_kept (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (N : Msg → ℕ) (Ok : Msg → Col → Prop)
    (h : TableBoundθKept H D N Ok) : TableBoundθ H D N Ok := by
  intro x Ω _ _ P
  obtain ⟨e, -, hfit, hb⟩ := h x Ω P Unit fun _ _ _ => true
  refine ⟨e.toEscapes, hfit, fun ω C hC hg => ?_⟩
  rw [EscapesD.term_toEscapes]
  refine le_trans (le_of_eq ?_) ((hb ω C hC hg).trans (le_of_eq ?_))
  · unfold accPr
    rw [← Pr_unit fun r : Fin (D.V.k x) → Coin => (run H P D.V x ω r).out = .acc]
    exact Pr_congr fun d => by simp
  · congr 1
    rw [← Pr_unit fun r : Fin (D.V.k x) → Coin => (run H P D.V x ω r).out = .acc ∧
      ∃ (p : Fin (N x)) (u : Col), lookCol (D.cols (run H P D.V x ω r).seen 0) p.val = some u ∧ u ≠ C p]
    exact Pr_congr fun d => by simp

end AbortSamplingDesign

section KSAbort

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

def OneWaitAbort (H : Msg → Msg) (D : Flock.Deploy Coin Col T) : Prop :=
  ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (A : Type) [Fintype A] [Nonempty A]
    (keep : Ω → (Fin (D.V.k x) → Coin) → A → Bool),
    ∃ e : EscapesD H (D.V.k x) (abortRun H P D.V x keep), e.Fits (D.weight x) ∧
      ∀ ω, 0 < accPrD (abortRun H P D.V x keep ω) →
        accPrD (abortRun H P D.V x keep ω) * (accFailD (abortRun H P D.V x keep ω) (D.k x) (Extracts D x) /
          accPrD (abortRun H P D.V x keep ω) ^ D.k x) ≤
          D.εtable x + e.term ω + D.N₀ x / (Real.exp 1 * D.k x) * accPrD (abortRun H P D.V x keep ω)

theorem opD_tied (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (N : Msg → ℕ) (hconf : ConflictTied H D N)
    {Ω A : Type} (P : Prover Ω) (x : Msg) (keep : Ω → (Fin (D.V.k x) → Coin) → A → Bool) (j₀ : Fin (D.V.k x))
    (hj₀ : j₀.val = 0) (ω : Ω) (d d' : (Fin (D.V.k x) → Coin) × A)
    (h : ∃ p, ConflictAt (opD D N x (abortRun H P D.V x keep ω)) (d, d') p) :
    TiedColl H ((abortRun H P D.V x keep ω d).vLog ++ (abortRun H P D.V x keep ω (forkSample j₀ d d')).vLog) := by
  classical
  obtain ⟨p, h1, h2, h3⟩ := h
  have hfd : forkSample j₀ d d' = d' := by
    show (forkCoins j₀ d.1 d'.1, d'.2) = d'
    rw [Claims.Bridge.forkCoins_zero j₀ hj₀]
  rw [hfd]
  simp only [opD] at h1 h2 h3
  split_ifs at h1 h2 h3 with ha ha'
  · obtain ⟨hk1, hr1⟩ := abortRun_acc_iff.1 ha
    obtain ⟨hk2, hr2⟩ := abortRun_acc_iff.1 ha'
    obtain ⟨u, hu⟩ := Option.isSome_iff_exists.1 h1
    obtain ⟨v, hv⟩ := Option.isSome_iff_exists.1 h2
    have hne : u ≠ v := fun huv => h3 (by rw [hu, hv, huv])
    rw [abortRun_of_keep hk1] at hu ⊢
    rw [abortRun_of_keep hk2] at hv ⊢
    exact hconf x Ω P ω d.1 d'.1 p.val u v hr1 hr2 p.isLt hu hv hne
  all_goals simp at h1 h2

theorem oneWaitAbort_le_of_tableBound (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (N : Msg → ℕ)
    (hN : ∀ x, (N x : ℝ) ≤ D.N₀ x) (hεt : ∀ x, 0 ≤ D.εtable x) (hVk : ∀ x, 0 < D.V.k x)
    (hk : ∀ x, 1 ≤ D.k x ∨ NoAcc H D.V x)
    (hone : ∀ x, (∃ s, D.parse x = some s ∧ s.tables = 1) ∨ NoAcc H D.V x) (Ok : Msg → Col → Prop) (hz : ∀ x, Ok x 0)
    (hok : ColsOk H D N Ok) (hconf : ConflictTied H D N) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) {A : Type} [Fintype A] [Nonempty A] (keep : Ω → (Fin (D.V.k x) → Coin) → A → Bool)
    (e₀ : EscapesD H (D.V.k x) (abortRun H P D.V x keep))
    (hb : ∀ (ω : Ω) (C : Fin (N x) → Col), (∀ p, Ok x (C p)) → ¬ GoodTable D N x C →
      accPrD (abortRun H P D.V x keep ω) ≤ D.εtable x + e₀.term ω +
        Pr fun d : (Fin (D.V.k x) → Coin) × A => (abortRun H P D.V x keep ω d).out = .acc ∧
          ∃ (p : Fin (N x)) (u : Col), lookCol (D.cols (abortRun H P D.V x keep ω d).seen 0) p.val = some u ∧ u ≠ C p)
    (ω : Ω) (hε : 0 < accPrD (abortRun H P D.V x keep ω)) :
    accPrD (abortRun H P D.V x keep ω) * (accFailD (abortRun H P D.V x keep ω) (D.k x) (Extracts D x) /
      accPrD (abortRun H P D.V x keep ω) ^ D.k x) ≤
      D.εtable x + (e₀.addFork (D.k x) (Nat.cast_nonneg _) ⟨0, hVk x⟩
        (fun ω d d' => ∃ p, ConflictAt (opD D N x (abortRun H P D.V x keep ω)) (d, d') p)
        (opD_tied H D N hconf P x keep ⟨0, hVk x⟩ rfl)).term ω +
        D.N₀ x / (Real.exp 1 * D.k x) * accPrD (abortRun H P D.V x keep ω) := by
  classical
  have hacc : ¬ NoAcc H D.V x := fun hno => by
    have : accPrD (abortRun H P D.V x keep ω) = 0 := by
      unfold accPrD
      rw [Pr_congr (q := fun _ => False) fun d =>
        ⟨fun h => hno Ω P ω d.1 (abortRun_acc_iff.1 h).2, False.elim⟩, Pr_false]
    linarith
  have hk1 : 1 ≤ D.k x := (hk x).resolve_right hacc
  obtain ⟨s, hs, h1⟩ := (hone x).resolve_right hacc
  have hokD : ∀ d q u, (abortRun H P D.V x keep ω d).out = .acc → q < N x →
      lookCol (D.cols (abortRun H P D.V x keep ω d).seen 0) q = some u → Ok x u := by
    intro d q u ha hq hu
    obtain ⟨hkeep, hr⟩ := abortRun_acc_iff.1 ha
    rw [abortRun_of_keep hkeep] at hu
    exact hok x Ω P ω d.1 q u hr hq hu
  have hεc : 0 ≤ D.εtable x + e₀.term ω := add_nonneg (hεt x) (e₀.term_nonneg ω)
  have hj := oneWaitD D N x (abortRun H P D.V x keep ω) hs h1 Ok (hz x) hokD _ hεc
    (fun C hC hg => hb ω C hC hg) (D.k x) hk1
  have hE : (Pr fun s : ((Fin (D.V.k x) → Coin) × A) × ((Fin (D.V.k x) → Coin) × A) =>
      ∃ p, ConflictAt (opD D N x (abortRun H P D.V x keep ω)) s p) =
      Pr fun s : ((Fin (D.V.k x) → Coin) × A) × ((Fin (D.V.k x) → Coin) × A) =>
        ∃ p, ConflictAt (opD D N x (abortRun H P D.V x keep ω)) (s.1, s.2) p := rfl
  rw [hE] at hj
  rw [EscapesD.term_addFork]
  have hrate : (N x : ℝ) * accPrD (abortRun H P D.V x keep ω) / (Real.exp 1 * D.k x) ≤
      D.N₀ x / (Real.exp 1 * D.k x) * accPrD (abortRun H P D.V x keep ω) := by
    rw [div_mul_eq_mul_div]
    exact div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_right (hN x) hε.le) (by positivity)
  linarith

theorem oneWaitAbort_of_tableBound (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (N : Msg → ℕ)
    (hN : ∀ x, (N x : ℝ) ≤ D.N₀ x) (hεt : ∀ x, 0 ≤ D.εtable x) (hVk : ∀ x, 0 < D.V.k x)
    (hk : ∀ x, 1 ≤ D.k x ∨ NoAcc H D.V x)
    (hone : ∀ x, (∃ s, D.parse x = some s ∧ s.tables = 1) ∨ NoAcc H D.V x) (Ok : Msg → Col → Prop) (hz : ∀ x, Ok x 0)
    (hok : ColsOk H D N Ok) (hconf : ConflictTied H D N) (htab : TableBoundθAbort H D N Ok) :
    OneWaitAbort H D := by
  intro x Ω _ _ P A _ _ keep
  obtain ⟨e₀, hfit, hb⟩ := htab x Ω P A keep
  exact ⟨_, e₀.fits_addFork (D.weight x) (D.k x) (Nat.cast_nonneg _) ⟨0, hVk x⟩ rfl
    (fun ω d d' => ∃ p, ConflictAt (opD D N x (abortRun H P D.V x keep ω)) (d, d') p)
    (opD_tied H D N hconf P x keep ⟨0, hVk x⟩ rfl) hfit,
    oneWaitAbort_le_of_tableBound H D N hN hεt hVk hk hone Ok hz hok hconf x P keep e₀ hb⟩

theorem KSAbort_of_oneWaitAbort (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (hK : ∀ x, 0 < D.cap x)
    (hk : ∀ x, (8 ≤ D.k x ∧ Splits D x) ∨ NoAcc H D.V x) (hεt : ∀ x, 0 ≤ D.εtable x) (h : OneWaitAbort H D)
    (decode : List (ℕ × Col) → List T) (hlw : ListWithin D.len D.enc D.θ decode) :
    (Inner.ofFlock D decode).KSAbort H := by
  intro x Ω _ _ P A _ _ keep
  rcases hk x with ⟨hk8, hsp⟩ | hno
  swap
  · have h0 : (Inner.ofFlock D decode).PrBadAbort H x P keep = 0 := by
      unfold Inner.PrBadAbort
      rw [Pr_congr (q := fun _ => False) fun s =>
        ⟨fun hs => hno Ω P s.1 s.2.1.1 (abortRun_acc_iff.1 hs.1).2, False.elim⟩, Pr_false]
    rw [h0]
    exact add_nonneg (Flock.κ_pos D x (hεt x)).le (Real.sqrt_nonneg _)
  obtain ⟨e, hfit, hw⟩ := h x Ω P A keep
  have hwait : ∀ ω, 0 < accPrD (abortRun H P D.V x keep ω) →
      accPrD (abortRun H P D.V x keep ω) * (accFailD (abortRun H P D.V x keep ω) (D.k x)
        (Assembles (fun v => [D.cols v]) (assemble D.parse D.table decode D.read x) (check D.parse x)) /
          accPrD (abortRun H P D.V x keep ω) ^ D.k x) ≤
        (D.εtable x + e.term ω) + D.N₀ x / (Real.exp 1 * D.k x) * accPrD (abortRun H P D.V x keep ω) := by
    intro ω hε
    refine le_trans ?_ ((hw ω hε).trans (le_of_eq (by ring)))
    refine mul_le_mul_of_nonneg_left (div_le_div_of_nonneg_right ?_ (pow_nonneg hε.le _)) hε.le
    unfold accFailD
    exact Pr_mono fun σ ⟨hacc, hno⟩ => ⟨hacc, fun hex => hno (assembles_of_extracts D decode hlw x _ hsp hex)⟩
  have hb := PrBadAbort_trunc H (Inner.ofFlock D decode) (check_sound D.parse) x P keep (hK x) hk8
    (fun ω => D.εtable x + e.term ω) (fun ω => add_nonneg (hεt x) (e.term_nonneg ω)) (rate_le D x (by omega))
    hwait
  have hEx : 4 * Ex (fun ω => D.εtable x + e.term ω) ≤
      4 * D.εtable x + Real.sqrt (Flock.γ D x * PrBAbort H D.V P x keep) := by
    rw [Ex_add, Ex_const]
    have := e.le (D.weight x) hfit
    unfold Flock.γ
    linarith
  show (Inner.ofFlock D decode).PrBadAbort H x P keep ≤
    Flock.κ D x + Real.sqrt (Flock.γ D x * PrBAbort H D.V P x keep)
  have hb' : (Inner.ofFlock D decode).PrBadAbort H x P keep ≤ 4 * Ex (fun ω => D.εtable x + e.term ω) +
      2 ^ (-(D.waits x : ℝ)) + 2 * D.k x / D.cap x + Real.exp (-(D.k x : ℝ) / 8) := hb
  unfold Flock.κ
  linarith

theorem KSAbort_of_tableBoundKept (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (N : Msg → ℕ)
    (Ok : Msg → Col → Prop) (hN : ∀ x, (N x : ℝ) ≤ D.N₀ x) (hεt : ∀ x, 0 ≤ D.εtable x) (hVk : ∀ x, 0 < D.V.k x)
    (hK : ∀ x, 0 < D.cap x) (hk : ∀ x, (8 ≤ D.k x ∧ Splits D x) ∨ NoAcc H D.V x)
    (hone : ∀ x, (∃ s, D.parse x = some s ∧ s.tables = 1) ∨ NoAcc H D.V x) (hz : ∀ x, Ok x 0)
    (hok : ColsOk H D N Ok) (hconf : ConflictTied H D N) (htab : TableBoundθKept H D N Ok)
    (decode : List (ℕ × Col) → List T) (hdec : decOK D decode) : (Inner.ofFlock D decode).KSAbort H :=
  KSAbort_of_oneWaitAbort H D hK hk hεt
    (oneWaitAbort_of_tableBound H D N hN hεt hVk (fun x => (hk x).imp (fun h => by omega) id) hone Ok hz hok hconf
      (tableBoundθAbort_of_kept H D N Ok htab)) decode hdec

end KSAbort

end Claims.Recursion
