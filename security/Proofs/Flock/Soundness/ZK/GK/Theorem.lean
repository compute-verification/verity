module

public import Proofs.Flock.Soundness.ZK.GK.Sim
public import Proofs.Flock.Soundness.Assumptions
import Proofs.Flock.Soundness.Game
import Proofs.Flock.Soundness.ZK.Masking
import VCVio.Prelude.Core

/-! Zero knowledge, Theorem GK: the simulator's output is close to `V*`'s real view, with Lemma B (`RewindClose`) and
Lemma C (`PrefinalClose`) taken as hypotheses at `δ_B` and `δ_C`. -/

@[expose] public section

namespace FlockSoundness.ZK.GK

open Finset Game

/-- The simulator uses only `dummy`, `rwS`, `comp` and `bad`; `real`, `rwR`, `pre`, `compP` and `inc` are the
analysis's. -/
structure Model (V P Ω Ωd Ωs : Type) where
  real : Ω → V
  dummy : Ωd → V
  rwS : V → Ωs → V
  rwR : V → Ω → V
  comp : V → Bool
  bad : V → V → Bool
  pre : V → P
  compP : P → Bool
  inc : P → V
  comp_eq : ∀ v, comp v = compP (pre v)
  inc_real : ∀ ω, comp (real ω) = false → real ω = inc (pre (real ω))
  inc_dummy : ∀ ω, comp (dummy ω) = false → dummy ω = inc (pre (dummy ω))
  bind_real : ∀ ω₀, comp (dummy ω₀) = true → ∀ ω, bad (dummy ω₀) (rwR (dummy ω₀) ω) = false →
    rwR (dummy ω₀) ω = real ω

variable {V P Ω Ωd Ωs : Type} [Fintype Ω] [Nonempty Ω] [Fintype Ωd] [Nonempty Ωd] [Fintype Ωs] [Nonempty Ωs]

namespace Model

variable (m : Model V P Ω Ωd Ωs)

def PrefinalClose (δC : ℝ) : Prop :=
  ∀ U : P → Prop, |avgR (fun ω => ind U (m.pre (m.real ω))) - avgR (fun ω => ind U (m.pre (m.dummy ω)))| ≤ δC

def RewindClose (δB : ℝ) : Prop :=
  0 ≤ δB ∧ ∀ ω₀, m.comp (m.dummy ω₀) = true → ∀ U : V → Prop,
    |avgR (fun σ => ind U (m.rwS (m.dummy ω₀) σ)) - avgR (fun ω => ind U (m.rwR (m.dummy ω₀) ω))| ≤ δB

def sim (t M : ℕ) : Rand (Out V × ℕ) := gk Ωd Ωs m.dummy m.rwS m.comp m.bad t M

def hyb₁ (t M : ℕ) : Rand (Out V × ℕ) := gk Ωd Ω m.dummy m.rwR m.comp m.bad t M

def hyb₂ (t M : ℕ) : Rand (Out V × ℕ) := gk Ωd Ω m.dummy (fun _ => m.real) m.comp (fun _ _ => false) t M

def finder (pick : V → V → Pair) (t M : ℕ) : Rand Pair := (m.sim t M).map fun o => Out.pickBy pick o.1

noncomputable abbrev pD : ℝ := rate Ωd fun ω => m.comp (m.dummy ω)

noncomputable abbrev pR : ℝ := rate Ω fun ω => m.comp (m.real ω)

end Model

theorem avgR_ite_split {C : Type} [Fintype C] (c : C → Bool) (x y : C → ℝ) :
    avgR (fun ω => if c ω = true then x ω else y ω) =
      avgR (fun ω => if c ω = true then x ω else 0) + avgR (fun ω => if c ω = true then 0 else y ω) := by
  rw [← avgR_add]
  congr 1
  funext ω
  split_ifs <;> simp

namespace Model

variable (m : Model V P Ω Ωd Ωs)

theorem sim_hyb₁ {δB : ℝ} (hB : m.RewindClose δB) (t M : ℕ) (f : Out V × ℕ → ℝ) (h0 : ∀ x, 0 ≤ f x)
    (h1 : ∀ x, f x ≤ 1) : |Rand.ev f (m.sim t M) - Rand.ev f (m.hyb₁ t M)| ≤ 4 * t * δB := by
  obtain ⟨hδ, hB⟩ := hB
  unfold sim hyb₁
  rw [ev_gk, ev_gk]
  set est := estimate Ωd (fun ω => m.comp (m.dummy ω)) M t
  have hL : m.pD * Rand.ev (fun r => (r.1 : ℝ)) est ≤ t := estimate_runs Ωd _ M t
  refine (abs_avgR_sub_le (k := fun ω₀ => if m.comp (m.dummy ω₀) = true then
    4 * δB * Rand.ev (fun r => (r.1 : ℝ)) est else 0) fun ω₀ => ?_).trans ?_
  · split_ifs with hc
    · refine (Rand.abs_ev_sub_le (k := fun r => if r.2 = true then 4 * r.1 * δB else 0) (fun r => ?_) est).trans ?_
      · split_ifs
        · have := rewinds_close (m.dummy ω₀) (m.rwS (m.dummy ω₀)) (m.rwR (m.dummy ω₀)) m.comp (m.bad (m.dummy ω₀))
            (hB ω₀ hc) (4 * r.1) (fun o => f (o.1, o.2 + 1 + r.1)) (fun _ => h0 _) (fun _ => h1 _)
          push_cast at this
          exact this
        · simp
      · rw [← Rand.ev_const_mul]
        refine Rand.ev_mono (fun r => ?_) est
        split_ifs
        · linarith
        · positivity
    · simp
  · rw [avgR_split Ωd (fun ω => m.comp (m.dummy ω))]
    have := mul_le_mul_of_nonneg_left hL (by linarith : (0 : ℝ) ≤ 4 * δB)
    show m.pD * (4 * δB * _) + (1 - m.pD) * 0 ≤ _
    nlinarith

omit [Fintype Ωs] [Nonempty Ωs] in
theorem hyb₁_hyb₂ (t M : ℕ) (f : Out V × ℕ → ℝ) (h0 : ∀ x, 0 ≤ f x) (h1 : ∀ x, f x ≤ 1) :
    |Rand.ev f (m.hyb₁ t M) - Rand.ev f (m.hyb₂ t M)| ≤ Rand.pr (fun o => Out.bindBy m.bad o.1) (m.hyb₁ t M) := by
  unfold Rand.pr hyb₁ hyb₂
  rw [ev_gk, ev_gk, ev_gk]
  refine abs_avgR_sub_le fun ω₀ => ?_
  try dsimp only
  split_ifs with hc
  · refine Rand.abs_ev_sub_le (fun r => ?_) _
    try dsimp only
    split_ifs
    · exact rewinds_couple Ω (m.dummy ω₀) (m.rwR (m.dummy ω₀)) m.real m.comp (m.bad (m.dummy ω₀)) (Out.bindBy m.bad)
        (fun v hv => hv) (m.bind_real ω₀ hc) (4 * r.1) (fun o => f (o.1, o.2 + 1 + r.1)) (fun _ => h0 _)
        (fun _ => h1 _)
    · rw [sub_self, abs_zero]
      exact ind_nonneg _ _
  · rw [sub_self, abs_zero]
    exact ind_nonneg _ _

omit [Fintype Ωs] [Nonempty Ωs] in
theorem hyb₂_real {δC : ℝ} (hC : m.PrefinalClose δC) (t M : ℕ) (hM : 0 < M) (φ : V → ℝ) (h0 : ∀ v, 0 ≤ φ v)
    (h1 : ∀ v, φ v ≤ 1) :
    |Rand.ev (fun o => Out.val φ o.1) (m.hyb₂ t M) - avgR (fun ω => φ (m.real ω))| ≤
      3 * δC + t / M + (1 / 3) ^ t := by
  set p := m.pR with hp
  set pD := m.pD with hpD
  set est := estimate Ωd (fun ω => m.comp (m.dummy ω)) M t
  set S := avgR (fun ω => if m.comp (m.real ω) = true then φ (m.real ω) else 0) with hS
  have hp0 : 0 ≤ p := rate_nonneg _ _
  have hp1 : p ≤ 1 := rate_le_one _ _
  have hpD0 : 0 ≤ pD := rate_nonneg _ _
  have hpD1 : pD ≤ 1 := rate_le_one _ _
  have hS0 : 0 ≤ S := avgR_nonneg fun ω => by split_ifs; exacts [h0 _, le_rfl]
  have hSp : S ≤ p := avgR_mono fun ω => by
    unfold ind
    split_ifs <;> first | exact h1 _ | simp_all
  set A := if p = 0 then 0 else S / p with hA
  have hSA : S = p * A := by
    by_cases h : p = 0
    · rw [hA, ite_eq_left h, h, zero_mul]
      linarith
    · rw [hA, ite_eq_right h]
      field_simp
  have hA0 : 0 ≤ A := by
    rw [hA]
    split_ifs
    · exact le_rfl
    · positivity
  have hA1 : A ≤ 1 := by
    rw [hA]
    split_ifs with h
    · norm_num
    · exact (div_le_one (lt_of_le_of_ne hp0 (Ne.symm h))).2 hSp
  set Xr := Rand.ev (fun r => if r.2 = true then A * (1 - (1 - p) ^ (4 * r.1)) else 0) est with hX
  have eH2 : Rand.ev (fun o => Out.val φ o.1) (m.hyb₂ t M) =
      avgR fun ω₀ => if m.comp (m.dummy ω₀) = true then Xr else φ (m.dummy ω₀) := by
    unfold hyb₂
    rw [ev_gk]
    congr 1
    funext ω₀
    split_ifs
    · congr 1
      funext r
      split_ifs
      · exact ev_rewinds_real Ω (m.dummy ω₀) m.real m.comp φ A hSA (4 * r.1)
      · rfl
    · rfl
  let G : P → ℝ := fun x => if m.compP x = true then A else φ (m.inc x)
  have hG0 : ∀ x, 0 ≤ G x := fun x => by simp only [G]; split_ifs; exacts [hA0, h0 _]
  have hG1 : ∀ x, G x ≤ 1 := fun x => by simp only [G]; split_ifs; exacts [hA1, h1 _]
  have hGR : avgR (fun ω => G (m.pre (m.real ω))) = avgR (fun ω => φ (m.real ω)) := by
    have e1 : (fun ω => G (m.pre (m.real ω))) =
        fun ω => if m.comp (m.real ω) = true then A else φ (m.real ω) := by
      funext ω
      simp only [G, ← m.comp_eq]
      split_ifs with h
      · rfl
      · rw [← m.inc_real ω (by simpa using h)]
    have e3 : avgR (fun ω => φ (m.real ω)) =
        S + avgR (fun ω => if m.comp (m.real ω) = true then 0 else φ (m.real ω)) := by
      rw [hS, ← avgR_add]
      congr 1
      funext ω
      split_ifs <;> simp
    rw [e1, avgR_ite_split, avgR_split Ω (fun ω => m.comp (m.real ω)), e3, hSA]
    simp only [mul_zero, add_zero]
    rfl
  have hGD : avgR (fun ω => G (m.pre (m.dummy ω))) =
      pD * A + avgR (fun ω => if m.comp (m.dummy ω) = true then 0 else φ (m.dummy ω)) := by
    have e1 : (fun ω => G (m.pre (m.dummy ω))) =
        fun ω => if m.comp (m.dummy ω) = true then A else φ (m.dummy ω) := by
      funext ω
      simp only [G, ← m.comp_eq]
      split_ifs with h
      · rfl
      · rw [← m.inc_dummy ω (by simpa using h)]
    rw [e1, avgR_ite_split, avgR_split Ωd (fun ω => m.comp (m.dummy ω))]
    simp only [mul_zero, add_zero]
    rfl
  have hH2 : Rand.ev (fun o => Out.val φ o.1) (m.hyb₂ t M) =
      pD * Xr + avgR (fun ω => if m.comp (m.dummy ω) = true then 0 else φ (m.dummy ω)) := by
    rw [eH2, avgR_ite_split, avgR_split Ωd (fun ω => m.comp (m.dummy ω))]
    simp only [mul_zero, add_zero]
    rfl
  have hGclose : |avgR (fun ω => G (m.pre (m.real ω))) - avgR (fun ω => G (m.pre (m.dummy ω)))| ≤ δC :=
    abs_avgR_sub_le_of_events (fun ω => m.pre (m.real ω)) (fun ω => m.pre (m.dummy ω)) hC G hG0 hG1
  have hrate : |p - pD| ≤ δC := by
    have eR : (fun ω => ind (fun x => m.compP x = true) (m.pre (m.real ω))) =
        ind (fun ω => m.comp (m.real ω) = true) := by
      funext ω
      simp only [ind, m.comp_eq]
      split_ifs <;> rfl
    have eD : (fun ω => ind (fun x => m.compP x = true) (m.pre (m.dummy ω))) =
        ind (fun ω => m.comp (m.dummy ω) = true) := by
      funext ω
      simp only [ind, m.comp_eq]
      split_ifs <;> rfl
    have := hC (fun x => m.compP x = true)
    rw [eR, eD] at this
    exact this
  set a := (1 - p) ^ 4 with ha
  have ha0 : 0 ≤ a := by positivity
  have ha1 : a ≤ 1 := pow_le_one₀ (by linarith) (by linarith)
  set Y₁ := Rand.ev (fun r => if r.2 = true then a ^ r.1 else 0) est with hY₁
  set Y₂ := Rand.ev (fun r => if r.2 = true then 0 else 1) est with hY₂
  have hY₁0 : 0 ≤ Y₁ := Rand.ev_nonneg (fun r => by split_ifs; exacts [pow_nonneg ha0 _, le_rfl]) _
  have hY₁1 : Y₁ ≤ 1 := Rand.ev_le_one (fun r => by split_ifs; exacts [pow_le_one₀ ha0 ha1, zero_le_one]) _
  have hAX : A - Xr = Rand.ev (fun r => if r.2 = true then A * a ^ r.1 else A) est := by
    rw [hX, ← Rand.ev_const A est, ← Rand.ev_sub, Rand.ev_const]
    congr 1
    funext r
    split_ifs
    · rw [ha, ← pow_mul]
      ring
    · ring
  have hAX0 : 0 ≤ A - Xr := by
    rw [hAX]
    exact Rand.ev_nonneg (fun r => by split_ifs; exacts [mul_nonneg hA0 (pow_nonneg ha0 _), hA0]) _
  have hAX1 : A - Xr ≤ Y₁ + Y₂ := by
    rw [hAX, hY₁, hY₂, ← Rand.ev_add]
    refine Rand.ev_mono (fun r => ?_) est
    split_ifs
    · have := pow_nonneg ha0 r.1
      nlinarith
    · linarith
  have hcap : pD * Y₂ ≤ t / M := by
    have hc := estimate_cap Ωd (fun ω => m.comp (m.dummy ω)) M t
    have hr := estimate_runs Ωd (fun ω => m.comp (m.dummy ω)) M t
    have hMpos : (0 : ℝ) < M := by exact_mod_cast hM
    rw [le_div_iff₀ hMpos]
    have := mul_le_mul_of_nonneg_left hc hpD0
    change pD * (M * Y₂) ≤ pD * Rand.ev (fun r => (r.1 : ℝ)) est at this
    change pD * Rand.ev (fun r => (r.1 : ℝ)) est ≤ t at hr
    calc pD * Y₂ * M = pD * (M * Y₂) := by ring
      _ ≤ _ := this
      _ ≤ t := hr
  have hab : pD * Y₁ ≤ (1 / 3) ^ t + 2 * δC := by
    have hδ : 0 ≤ δC := (abs_nonneg _).trans hrate
    by_cases hh : pD / 2 ≤ p
    · have hb : a * (1 + 2 * pD) ≤ 1 := by
        have h4 : a ≤ (1 - pD / 2) ^ 4 := pow_le_pow_left₀ (by linarith) (by linarith) 4
        have hr := gk_ratio_le_third pD hpD1
        have : (1 - pD / 2) ^ 4 * (1 + 2 * pD) ≤ 1 := by linarith
        calc a * (1 + 2 * pD) ≤ (1 - pD / 2) ^ 4 * (1 + 2 * pD) :=
              mul_le_mul_of_nonneg_right h4 (by linarith)
          _ ≤ 1 := this
      have := estimate_abort Ωd (fun ω => m.comp (m.dummy ω)) ha0 hb M t
      change Y₁ ≤ (1 / 3) ^ t at this
      calc pD * Y₁ ≤ 1 * Y₁ := mul_le_mul_of_nonneg_right hpD1 hY₁0
        _ ≤ _ := by linarith
    · push Not at hh
      have : pD ≤ 2 * δC := by
        have := (abs_le.1 hrate).1
        linarith
      have h3 : (0 : ℝ) ≤ (1 / 3) ^ t := by positivity
      calc pD * Y₁ ≤ pD * 1 := mul_le_mul_of_nonneg_left hY₁1 hpD0
        _ ≤ _ := by linarith
  have k0 : pD * Xr ≤ pD * A := mul_le_mul_of_nonneg_left (by linarith) hpD0
  have k1 : pD * A - pD * Xr ≤ pD * Y₁ + pD * Y₂ := by
    have := mul_le_mul_of_nonneg_left hAX1 hpD0
    rw [mul_sub, mul_add] at this
    exact this
  rw [← hGR, hH2]
  rw [hGD] at hGclose
  clear_value G Xr A S Y₁ Y₂ a est p pD
  rw [abs_le] at hGclose ⊢
  constructor <;> linarith

theorem gk_close {δB δC : ℝ} (hB : m.RewindClose δB) (hC : m.PrefinalClose δC) (t M : ℕ) (hM : 0 < M) (φ : V → ℝ)
    (h0 : ∀ v, 0 ≤ φ v) (h1 : ∀ v, φ v ≤ 1) :
    |Rand.ev (fun o => Out.val φ o.1) (m.sim t M) - avgR (fun ω => φ (m.real ω))| ≤
      3 * δC + 8 * t * δB + (1 / 3) ^ t + t / M + Rand.pr (fun o => Out.bindBy m.bad o.1) (m.sim t M) := by
  have s1 := m.sim_hyb₁ hB t M (fun o => Out.val φ o.1) (fun o => Out.val_nonneg h0 o.1)
    (fun o => Out.val_le_one h1 o.1)
  have s1b := m.sim_hyb₁ hB t M (ind fun o => Out.bindBy m.bad o.1) (ind_nonneg _) (ind_le_one _)
  have s2 := m.hyb₁_hyb₂ t M (fun o => Out.val φ o.1) (fun o => Out.val_nonneg h0 o.1)
    (fun o => Out.val_le_one h1 o.1)
  have s3 := m.hyb₂_real hC t M hM φ h0 h1
  unfold Rand.pr at s2 ⊢
  rw [abs_le] at s1 s1b s2 s3 ⊢
  constructor <;> linarith [s1.1, s1.2, s1b.1, s1b.2, s2.1, s2.2, s3.1, s3.2]

omit [Fintype Ω] [Nonempty Ω] in
/-- Lean does not check that the budget `q` is honest: it counts the simulator's runs, at most `1 + 5M`, and the
extraction's evaluations, at most `2·(|path| + 3 + n)`. -/
theorem pr_bind_le {D : Type} (H : List UInt8 → D) (pick : V → V → Pair)
    (hpick : ∀ v₀ v, m.bad v₀ v = true → IsColl H (pick v₀ v)) (t M : ℕ) {q : ℝ}
    (hcr : Assumptions.SHA512CRStrict H (m.finder pick t M).toGame (m.finder pick t M).strat id (fun _ => q) q) :
    Rand.pr (fun o => Out.bindBy m.bad o.1) (m.sim t M) ≤ q ^ 2 / 2 ^ (513 : ℝ) := by
  have h := hcr fun _ => le_rfl
  rw [Rand.prob_toGame] at h
  refine le_trans ?_ h
  unfold Rand.pr finder
  rw [Rand.ev_map]
  refine Rand.ev_mono (fun o => ?_) _
  unfold ind
  split_ifs with hb hc
  · exact le_rfl
  · exfalso
    apply hc
    rcases o with ⟨o, n⟩
    cases o with
    | bind v₀ v => exact hpick v₀ v hb
    | view v => exact hb.elim
    | fail => exact hb.elim
  · norm_num
  · exact le_rfl

omit [Fintype Ω] [Nonempty Ω] in
theorem sim_runs_le (t M : ℕ) : Rand.All (fun o : Out V × ℕ => o.2 ≤ 1 + 5 * M) (m.sim t M) := by
  unfold sim gk
  intro ω₀
  try dsimp only
  split_ifs
  · refine Rand.All.bind _ (fun r hr => ?_) _ (estimate_all Ωd _ M t)
    try dsimp only
    split_ifs
    · refine Rand.All.map (P := fun o : Out V × ℕ => o.2 ≤ 4 * r.1) _ (fun o ho => ?_) _ (rewinds_all ..)
      try dsimp only at ho ⊢
      omega
    · show 1 + r.1 ≤ 1 + 5 * M
      omega
  · show 1 ≤ 1 + 5 * M
    omega

omit [Fintype Ω] [Nonempty Ω] in
theorem sim_runs_ev (t M : ℕ) : Rand.ev (fun o => (o.2 : ℝ)) (m.sim t M) ≤ 1 + 5 * t := by
  unfold sim
  rw [ev_gk]
  set est := estimate Ωd (fun ω => m.comp (m.dummy ω)) M t
  have hL : m.pD * Rand.ev (fun r => (r.1 : ℝ)) est ≤ t := estimate_runs Ωd _ M t
  have hle : ∀ ω₀, (if m.comp (m.dummy ω₀) = true then
        Rand.ev (fun r => if r.2 = true then
            Rand.ev (fun o => ((o.2 + 1 + r.1 : ℕ) : ℝ))
              (rewinds Ωs (m.dummy ω₀) (m.rwS (m.dummy ω₀)) m.comp (m.bad (m.dummy ω₀)) (4 * r.1))
          else ((1 + r.1 : ℕ) : ℝ)) est
      else ((1 : ℕ) : ℝ)) ≤
      if m.comp (m.dummy ω₀) = true then 1 + 5 * Rand.ev (fun r => (r.1 : ℝ)) est else 1 := by
    intro ω₀
    split_ifs
    · rw [← Rand.ev_const_mul, ← Rand.ev_const 1 est, ← Rand.ev_add]
      refine Rand.ev_mono (fun r => ?_) est
      split_ifs
      · refine (Rand.ev_le_of_All (g := fun _ => 1 + 5 * (r.1 : ℝ)) (fun o ho => ?_) _ (rewinds_all ..)).trans ?_
        · have : ((o.2 + 1 + r.1 : ℕ) : ℝ) ≤ ((4 * r.1 + 1 + r.1 : ℕ) : ℝ) := by exact_mod_cast (by omega)
          push_cast at this ⊢
          linarith
        · rw [Rand.ev_const]
      · push_cast
        have : (0 : ℝ) ≤ r.1 := Nat.cast_nonneg _
        linarith
    · norm_num
  refine (avgR_mono hle).trans ?_
  rw [avgR_split Ωd (fun ω => m.comp (m.dummy ω))]
  have := rate_le_one Ωd fun ω => m.comp (m.dummy ω)
  change m.pD * _ + (1 - m.pD) * 1 ≤ _
  nlinarith

omit [Fintype Ω] [Nonempty Ω] in
theorem sim_runs_tail (t M : ℕ) {x : ℝ} (hx : 0 < x) :
    Rand.pr (fun o => x ≤ (o.2 : ℝ)) (m.sim t M) ≤ (1 + 5 * t) / x := by
  rw [le_div_iff₀ hx]
  have h := m.sim_runs_ev t M
  have : Rand.pr (fun o => x ≤ (o.2 : ℝ)) (m.sim t M) * x ≤ Rand.ev (fun o => (o.2 : ℝ)) (m.sim t M) := by
    unfold Rand.pr
    rw [mul_comm, ← Rand.ev_const_mul]
    refine Rand.ev_mono (fun o => ?_) _
    unfold ind
    split_ifs with h'
    · linarith
    · simp
  linarith

theorem gk_simulate {D : Type} (H : List UInt8 → D) (pick : V → V → Pair)
    (hpick : ∀ v₀ v, m.bad v₀ v = true → IsColl H (pick v₀ v)) {δB δC q : ℝ} (hB : m.RewindClose δB)
    (hC : m.PrefinalClose δC) (t M : ℕ) (hM : 0 < M)
    (hcr : Assumptions.SHA512CRStrict H (m.finder pick t M).toGame (m.finder pick t M).strat id (fun _ => q) q)
    (φ : V → ℝ) (h0 : ∀ v, 0 ≤ φ v) (h1 : ∀ v, φ v ≤ 1) :
    |Rand.ev (fun o => Out.val φ o.1) (m.sim t M) - avgR (fun ω => φ (m.real ω))| ≤
      3 * δC + 8 * t * δB + (1 / 3) ^ t + t / M + q ^ 2 / 2 ^ (513 : ℝ) :=
  (m.gk_close hB hC t M hM φ h0 h1).trans (by linarith [m.pr_bind_le H pick hpick t M hcr])

theorem gk_simulate_coins {D Coin Salt : Type} [DecidableEq D] [DecidableEq Coin] (H : List UInt8 → D)
    (T : CoinTree D Coin Salt) (root : V → D) (opens : V → List (CoinOpening D Coin Salt))
    (hbad : ∀ v₀ v, m.bad v₀ v = conflicts H T (root v₀) (opens v₀) (opens v)) {δB δC q : ℝ}
    (hB : m.RewindClose δB) (hC : m.PrefinalClose δC) (t M : ℕ) (hM : 0 < M)
    (hcr : Assumptions.SHA512CRStrict H (m.finder (fun v₀ v => bindColl H T (root v₀) (opens v₀) (opens v)) t M).toGame
      (m.finder (fun v₀ v => bindColl H T (root v₀) (opens v₀) (opens v)) t M).strat id (fun _ => q) q)
    (φ : V → ℝ) (h0 : ∀ v, 0 ≤ φ v) (h1 : ∀ v, φ v ≤ 1) :
    |Rand.ev (fun o => Out.val φ o.1) (m.sim t M) - avgR (fun ω => φ (m.real ω))| ≤
      3 * δC + 8 * t * δB + (1 / 3) ^ t + t / M + q ^ 2 / 2 ^ (513 : ℝ) :=
  m.gk_simulate H _ (fun v₀ v hb => bindColl_isColl H T (root v₀) (opens v₀) (opens v) ((hbad v₀ v).symm.trans hb))
    hB hC t M hM hcr φ h0 h1

theorem gk_simulate_coins_keyed {D Coin Salt : Type} [DecidableEq D] [DecidableEq Coin] (H : List UInt8 → D)
    (T : CoinTree D Coin Salt) (ν : V → List UInt8) (root : V → D) (opens : V → List (CoinOpening D Coin Salt))
    (hbad : ∀ v₀ v, m.bad v₀ v = conflicts (fun x => H (ν v₀ ++ x)) T (root v₀) (opens v₀) (opens v)) {δB δC q : ℝ}
    (hB : m.RewindClose δB) (hC : m.PrefinalClose δC) (t M : ℕ) (hM : 0 < M)
    (hcr : Assumptions.SHA512CRStrict H
      (m.finder (fun v₀ v => bindCollKeyed H T (ν v₀) (root v₀) (opens v₀) (opens v)) t M).toGame
      (m.finder (fun v₀ v => bindCollKeyed H T (ν v₀) (root v₀) (opens v₀) (opens v)) t M).strat id (fun _ => q) q)
    (φ : V → ℝ) (h0 : ∀ v, 0 ≤ φ v) (h1 : ∀ v, φ v ≤ 1) :
    |Rand.ev (fun o => Out.val φ o.1) (m.sim t M) - avgR (fun ω => φ (m.real ω))| ≤
      3 * δC + 8 * t * δB + (1 / 3) ^ t + t / M + q ^ 2 / 2 ^ (513 : ℝ) :=
  m.gk_simulate H _ (fun v₀ v hb =>
    bindCollKeyed_isColl H T (ν v₀) (root v₀) (opens v₀) (opens v) ((hbad v₀ v).symm.trans hb)) hB hC t M hM hcr φ h0 h1

end Model

theorem gk_simulate_avg {T : Type} [Fintype T] [Nonempty T] {D : Type} (H : List UInt8 → D)
    (m : T → Model V P Ω Ωd Ωs) (pick : V → V → Pair)
    (hpick : ∀ τ v₀ v, (m τ).bad v₀ v = true → IsColl H (pick v₀ v)) {δB δC q : ℝ}
    (hB : ∀ τ, (m τ).RewindClose δB) (hC : ∀ τ, (m τ).PrefinalClose δC) (t M : ℕ) (hM : 0 < M)
    (hcr : Assumptions.SHA512CRStrict H (Rand.coin T fun τ => (m τ).finder pick t M).toGame
      (Rand.coin T fun τ => (m τ).finder pick t M).strat id (fun _ => q) q)
    (φ : V → ℝ) (h0 : ∀ v, 0 ≤ φ v) (h1 : ∀ v, φ v ≤ 1) :
    |Rand.ev (fun o => Out.val φ o.1) (Rand.coin T fun τ => (m τ).sim t M) -
        avgR (fun τ => avgR (fun ω => φ ((m τ).real ω)))| ≤
      3 * δC + 8 * t * δB + (1 / 3) ^ t + t / M + q ^ 2 / 2 ^ (513 : ℝ) := by
  have h := hcr fun _ => le_rfl
  rw [Rand.prob_toGame] at h
  have hbind : avgR (fun τ => Rand.pr (fun o => Out.bindBy (m τ).bad o.1) ((m τ).sim t M)) ≤ q ^ 2 / 2 ^ (513 : ℝ) := by
    refine le_trans (avgR_mono fun τ => ?_) h
    unfold Rand.pr Model.finder
    rw [Rand.ev_map]
    refine Rand.ev_mono (fun o => ?_) _
    unfold ind
    split_ifs with hb hc
    · exact le_rfl
    · exfalso
      apply hc
      rcases o with ⟨o, n⟩
      cases o with
      | bind v₀ v => exact hpick τ v₀ v hb
      | view v => exact hb.elim
      | fail => exact hb.elim
    · norm_num
    · exact le_rfl
  rw [Rand.ev_coin]
  refine (abs_avgR_sub_le (k := fun τ => 3 * δC + 8 * t * δB + (1 / 3) ^ t + t / M +
    Rand.pr (fun o => Out.bindBy (m τ).bad o.1) ((m τ).sim t M)) fun τ => (m τ).gk_close (hB τ) (hC τ) t M hM φ h0 h1).trans ?_
  rw [avgR_add, avgR_const]
  linarith

end FlockSoundness.ZK.GK
