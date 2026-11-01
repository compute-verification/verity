import Proofs.Flock.EndToEnd.Carry.Statement
import Proofs.Flock.EndToEnd.Blocks
import Proofs.Flock.Soundness.Discharge.ZkHidden.PublicJ

/-!
# Proof of the integrity audit over a monitoring period of blocks, with carried state

* `SamplingBlock.accepted_drawnWrong_le_at`: one session's bound at the value layer of any values `Xv`
  (`zk_session_soundHJ_custody_drawn_at`): `ε_ks` and the binding errors against `Xv` (`errBoundZC`).
* `CarriedBlocks.errBound_le`: at the values `Xcar`, the binding errors are `δ_link`'s at every commit string no carried
  leaf sits at, and at a carried leaf's commit string the leaf's trial's binding error at the pulled-back value
  (`CarriedBlocks.carriedE`).
* `CarriedBlocks.periodE_errX_le`: that trial's binding error, averaged over the monitoring period up to the session, is the
  binding error of the leaf's trial in that session played after the monitoring period before it (`expect_compTo`), at most `m`
  times the mixture's (`errAt_le_mix`), which `ecr/sha-512` at the leaf's finder bounds (`link_mass_le`), with the
  finder's cost at most `Σ_{t ∈ ℓ} (Σ_{j < t} t'_j + t'_t(1 + k_t))/m`.
* `Flock.SecurityProofs.IntegrityHorizonCarried`: `IntegrityHorizon`'s proof, `period_unionE` in place of
  `period_union` for the sessions accepted with a drawn unit wrong, each carried leaf's term summed over the monitoring period.
-/

namespace FlockSoundness.Game

theorem expect_optG_pos {γ : Type} (F : γ → ℝ) (c : Prop) (g : c → Game γ) (a : γ) (s : (h : c) → Strategy (g h))
    (h : c) : ∀ d : Decidable c, expect F (optG c g a d) (optS c g a s d) = expect F (g h) (s h)
  | .isTrue _ => rfl
  | .isFalse hn => absurd h hn

theorem expect_optG_neg {γ : Type} (F : γ → ℝ) (c : Prop) (g : c → Game γ) (a : γ) (s : (h : c) → Strategy (g h))
    (hn : ¬ c) : ∀ d : Decidable c, expect F (optG c g a d) (optS c g a s d) = F a
  | .isTrue h => absurd h hn
  | .isFalse _ => rfl

theorem periodGood_true {β : ℕ → Type} (G : (i : ℕ) → Game (β i)) (i T : ℕ) (s : Strategy (periodG G i T)) :
    PeriodGood G (fun _ _ => True) i T s :=
  periodGood_of_plays G _ T i s fun _ _ _ => trivial

end FlockSoundness.Game

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden FlockSoundness.Discharge.ZkBind
  FlockSoundness.Binding FlockSoundness.LinkFinder
open scoped ENNReal

namespace SamplingBlock

variable (w : SamplingBlock)

open Classical in
/-- **One session's drawn-unit bound at the value layer of any values `Xv`**, at the strategy `τ` that plays it. -/
theorem accepted_drawnWrong_le_at
    (Xv : (R : w.Reg) → Cont w.L w.Reg (batchedSessionZC execArith hs512 (planZHJ w.dj w.y₀)) R → w.vb.Pos →
      w.vb.Val)
    (τ : Strategy w.game) (hcr : w.CR τ) :
    prob (fun o => LiveAcceptsZK w.I o ∧ ∃ u ∈ o.2.1,
        u ∈ wrongRegZ execArith w.P (planZHJ w.dj w.y₀) (tabZHJ w.dj w.y₀) w.vb (reg (w.sim τ))
          (XlayerZC execArith w.P (planZHJ w.dj w.y₀) (tabZHJ w.dj w.y₀) w.vb
            (Xv (reg (w.sim τ)) (cont (w.sim τ))))) w.game τ ≤
      w.ks + errBoundZC execArith hs512 w.P (planZHJ w.dj w.y₀) (tabZHJ w.dj w.y₀) w.vb Xv w.k w.Rw w.ρ
        (reg (w.sim τ)) (cont (w.sim τ)) := by
  have h := zk_session_soundHJ_custody_drawn_at w.I w.ht (Flock.Desc.parse_flat w.hD) w.hU w.hpub w.hpt w.hscope
    w.kd w.hdiv w.dj w.y₀ w.wr w.dg w.N w.Lb w.src w.hTags w.hCust w.hk w.hRw w.hρ w.hr Xv τ (qF := w.qF)
    (qS := w.qS) hcr.1
  refine h.trans (add_le_add ?_ le_rfl)
  exact le_iSup (fun R => ksAvgStrictZ (planZHJ w.dj w.y₀) w.L w.k w.qF w.qS R) _

/-- **An opening of a row's registered commitment hashes to its leaf**: `hm96`'s leaf of its bytes and salt. -/
theorem leaf_of_com {R : w.Reg} {p : w.vb.Pos} (hrow : w.rs.isRow p) (o : w.lay.Pos × w.lay.Val × List UInt8)
    (h : w.vb.com o = w.vb.cs R p) : H512 (hm96Input (w.rs.enc p o.2.1) o.2.2) = w.lay.regLeaf R p := by
  obtain ⟨p', v, y⟩ := o
  have h' : (p', (w.rs.scheme p').leaf v y) = (p, w.lay.regLeaf R p) := h
  obtain ⟨rfl, h2⟩ := Prod.mk.inj h'
  unfold HmRowsZX.scheme at h2
  rw [ite_eq_left_of_eq_true _ _ (eq_true hrow)] at h2
  exact h2

theorem readP_nonneg (c : w.vb.Pos) : 0 ≤ w.readP c := div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)

private theorem mul_add_le {π E K B g : ℝ} (hπ : 0 ≤ π) (h : E ≤ B + g) : π * (E + K) ≤ π * (B + K) + π * g := by
  nlinarith

/-- **The binding errors against `Xv`, split**: at most `δ_link` plus `Σ_c π_c g_c/(1 − ρ)` when each commit string's
binding error is at most `δ_link`'s per-string term plus `g_c`. -/
theorem errBound_le_of (R : w.Reg) (κ : Cont w.L w.Reg (batchedSessionZC execArith hs512 (planZHJ w.dj w.y₀)) R)
    (Xv : (R : w.Reg) → Cont w.L w.Reg (batchedSessionZC execArith hs512 (planZHJ w.dj w.y₀)) R → w.vb.Pos →
      w.vb.Val)
    (g : w.vb.Pos → ℝ)
    (hg : ∀ c, errZC execArith hs512 w.P (planZHJ w.dj w.y₀) (tabZHJ w.dj w.y₀) w.vb R κ c w.k w.Rw (Xv R κ c) ≤
      2 * (w.t' * (1 + w.k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * w.M) + g c) :
    errBoundZC execArith hs512 w.P (planZHJ w.dj w.y₀) (tabZHJ w.dj w.y₀) w.vb Xv w.k w.Rw w.ρ R κ ≤
      w.link + ENNReal.ofReal ((∑ c, w.readP c * g c) / (1 - w.ρ)) := by
  unfold errBoundZC link
  have hρ1 : 0 < 1 - w.ρ := by linarith [w.hρ]
  refine (ENNReal.ofReal_le_ofReal ?_).trans ENNReal.ofReal_add_le
  rw [← add_div]
  refine div_le_div_of_nonneg_right ?_ hρ1.le
  rw [Finset.sum_mul, ← Finset.sum_add_distrib]
  exact Finset.sum_le_sum fun c _ => mul_add_le (w.readP_nonneg c) (hg c)

end SamplingBlock

namespace CarriedLeaf

variable {W : ℕ → SamplingBlock} (x : CarriedLeaf W)

theorem enc_mem {t : ℕ} (ht : t ∈ x.sessions) (v : (W t).vb.Val) : (W t).rs.enc (x.pos t) v ∈ x.encs :=
  Finset.mem_biUnion.2 ⟨t, ht, Finset.mem_image_of_mem _ (Finset.mem_univ v)⟩

theorem e_eq {t : ℕ} (ht : t ∈ x.sessions) (v : (W t).vb.Val) : x.e t v = some ⟨_, x.enc_mem ht v⟩ := by
  unfold e
  rw [dite_eq_left_of_eq_true (eq_true (x.enc_mem ht v))]

theorem e_inj {t : ℕ} (ht : t ∈ x.sessions) : Function.Injective (x.e t) := fun v v' h => by
  rw [x.e_eq ht v, x.e_eq ht v'] at h
  exact x.inj t ht (congrArg Subtype.val (Option.some.inj h))

theorem enc_ne {t t' : ℕ} (ht : t ∈ x.sessions) (ht' : t' ∈ x.sessions) {v : (W t).vb.Val} {v' : (W t').vb.Val}
    (h : x.e t v ≠ x.e t' v') : (W t).rs.enc (x.pos t) v ≠ (W t').rs.enc (x.pos t') v' := fun he => h (by
  rw [x.e_eq ht v, x.e_eq ht' v']
  exact congrArg some (Subtype.ext he))

theorem rd_some {t : ℕ} (ht : t ∈ x.sessions) (τ : Strategy (W t).game)
    (o : TrialOutZC FlockLevel3.GF128 (W t).L Dig (List UInt8)) :
    x.rd (some ⟨t, τ, o⟩) = (rdTZC execArith hs512 (W t).P (planZHJ (W t).dj (W t).y₀) (tabZHJ (W t).dj (W t).y₀)
      (W t).vb (reg ((W t).sim τ)) (cont ((W t).sim τ)) (x.pos t) o).image (x.e t) := by
  simp only [rd, ht, ↓reduceIte]

theorem rd_some_out {t : ℕ} (ht : t ∉ x.sessions) (τ : Strategy (W t).game)
    (o : TrialOutZC FlockLevel3.GF128 (W t).L Dig (List UInt8)) : x.rd (some ⟨t, τ, o⟩) = ∅ := by
  simp only [rd, ht, ↓reduceIte]

theorem expect_trial {t : ℕ} (τ : Strategy (W t).game) (h : x.Read t) (F : TrialAt W → ℝ) :
    expect F (x.trial t τ) (x.trialS t τ) =
      haveI := h
      expect (fun o => F (some ⟨t, τ, o⟩))
        (trialGZC execArith hs512 (W t).P (planZHJ (W t).dj (W t).y₀) (tabZHJ (W t).dj (W t).y₀) (W t).vb
          (reg ((W t).sim τ)) (cont ((W t).sim τ)) (x.pos t) (W t).k (W t).Rw)
        (trialStratZC execArith hs512 (W t).P (planZHJ (W t).dj (W t).y₀) (tabZHJ (W t).dj (W t).y₀) (W t).vb
          (reg ((W t).sim τ)) (cont ((W t).sim τ)) (x.pos t) (W t).k (W t).Rw) := by
  have := h
  exact (expect_optG_pos F _ _ _ _ h _).trans (expect_map _ _ _ _)

theorem expect_trial_neg {t : ℕ} (τ : Strategy (W t).game) (h : ¬ x.Read t) (F : TrialAt W → ℝ) :
    expect F (x.trial t τ) (x.trialS t τ) = F none :=
  expect_optG_neg F _ _ _ _ h _

/-- **A session's binding error at the leaf, at a pulled-back common value, is at most its trial's** at that value. -/
theorem errZC_le_trial {t : ℕ} (ht : t ∈ x.sessions) (τ : Strategy (W t).game) (w : x.V) :
    errZC execArith hs512 (W t).P (planZHJ (W t).dj (W t).y₀) (tabZHJ (W t).dj (W t).y₀) (W t).vb
        (reg ((W t).sim τ)) (cont ((W t).sim τ)) (x.pos t) (W t).k (W t).Rw (pull (x.e t) w) ≤
      errAt (x.trial t τ) (x.trialS t τ) x.rd w := by
  by_cases hne : x.Read t
  · unfold errZC
    rw [dite_eq_left_of_eq_true (eq_true hne)]
    refine (errAt_pull_le _ _ _ (x.e_inj ht) w).trans (le_of_eq ?_)
    rw [errAt_eq_expect, errAt_eq_expect, x.expect_trial τ hne]
    refine congrArg (fun f => expect f _ _) (funext fun o => ?_)
    simp only [ind, x.rd_some ht]
    congr
  · unfold errZC
    rw [dite_eq_right_of_eq_false (eq_false hne)]
    exact errAt_nonneg _ _ _ w

theorem trialCost_nonneg : ∀ y : TrialAt W, 0 ≤ trialCost y
  | none => le_rfl
  | some ⟨t, τ, o⟩ => cT_nonnegZC (planZHJ (W t).dj (W t).y₀) (reg ((W t).sim τ)) (W t).ht' o

theorem costMix_nonneg (y : Σ _ : x.Ses, TrialAt W) : 0 ≤ x.costMix y :=
  add_nonneg (Finset.sum_nonneg fun j _ => (W j).ht') (trialCost_nonneg y.2)

/-- A trial's expected cost: at most `t'(1 + k)`. -/
theorem expect_cost_le (t : ℕ) (τ : Strategy (W t).game) :
    expect trialCost (x.trial t τ) (x.trialS t τ) ≤ (W t).t' * (1 + (W t).k) := by
  by_cases hne : x.Read t
  · have := hne
    rw [x.expect_trial τ hne]
    exact expect_cT_leZC execArith hs512 (W t).P (planZHJ (W t).dj (W t).y₀) (tabZHJ (W t).dj (W t).y₀) (W t).vb
      (reg ((W t).sim τ)) (cont ((W t).sim τ)) (x.pos t) (W t).k (W t).Rw (W t).ht'
  · rw [x.expect_trial_neg τ hne]
    show (0 : ℝ) ≤ _
    exact mul_nonneg (W t).ht' (by positivity)

/-- An opening from one of the leaf's sessions, of its registered commitment there. -/
def Opens : OpenAt W → Prop
  | none => False
  | some ⟨t, o⟩ => t ∈ x.sessions ∧ ∃ R, (W t).vb.com o = (W t).vb.cs R (x.pos t)

/-- An opening's value, as a common value. -/
noncomputable def valOf : OpenAt W → x.V
  | none => none
  | some ⟨t, o⟩ => x.e t ((W t).vb.val o)

/-- **Every success of the leaf's finder outputs a SHA-512 collision**: two readings of different bytes, from any two of
the leaf's sessions, open one leaf. -/
theorem out_spec (z : (Σ _ : x.Ses, TrialAt W) × List (Σ _ : x.Ses, TrialAt W))
    (hS : Succ (rdSig fun _ => x.rd) z) :
    ∃ a b, outOf (rdSig fun _ => x.rd) x.op x.col z = some (a, b) ∧ a ≠ b ∧ H512 a = H512 b := by
  refine outOf_spec (rdSig fun _ => x.rd) x.op x.col (Hc := H512) x.Opens x.valOf ?_ ?_ z hS
  · rintro ⟨i, _ | ⟨t, τ, o⟩⟩ v hv
    · exact absurd hv (Finset.notMem_empty v)
    · by_cases ht : t ∈ x.sessions
      · have hv' : v ∈ x.rd (some ⟨t, τ, o⟩) := hv
        rw [x.rd_some ht] at hv'
        obtain ⟨u, hu, rfl⟩ := Finset.mem_image.1 hv'
        have hp := pull_spec (x.e_inj ht) u
        have hspec := opOfVZC_spec execArith hs512 (W t).P (planZHJ (W t).dj (W t).y₀) (tabZHJ (W t).dj (W t).y₀)
          (W t).vb (reg ((W t).sim τ)) (cont ((W t).sim τ)) hu
        refine ⟨⟨ht, reg ((W t).sim τ), ?_⟩, ?_⟩
        · show (W t).vb.com (opOfVZC execArith hs512 (W t).P (planZHJ (W t).dj (W t).y₀) (tabZHJ (W t).dj (W t).y₀)
            (W t).vb (reg ((W t).sim τ)) (cont ((W t).sim τ)) (x.pos t) o (pull (x.e t) (x.e t u))) = _
          rw [hp]
          exact hspec.1
        · show x.e t ((W t).vb.val (opOfVZC execArith hs512 (W t).P (planZHJ (W t).dj (W t).y₀)
            (tabZHJ (W t).dj (W t).y₀) (W t).vb (reg ((W t).sim τ)) (cont ((W t).sim τ)) (x.pos t) o
            (pull (x.e t) (x.e t u)))) = x.e t u
          rw [hp, hspec.2]
      · have hv' : v ∈ x.rd (some ⟨t, τ, o⟩) := hv
        rw [x.rd_some_out ht] at hv'
        exact absurd hv' (Finset.notMem_empty v)
  · rintro (_ | ⟨t, o⟩) (_ | ⟨t', o'⟩) hg hg' hne
    · exact (hg : False).elim
    · exact (hg : False).elim
    · exact (hg' : False).elim
    · obtain ⟨ht, R, hR⟩ := hg
      obtain ⟨ht', R', hR'⟩ := hg'
      have hl : H512 (hm96Input ((W t).rs.enc (x.pos t) ((W t).vb.val o)) ((W t).salt o)) =
          H512 (hm96Input ((W t').rs.enc (x.pos t') ((W t').vb.val o')) ((W t').salt o')) :=
        ((W t).leaf_of_com (x.row t ht) o hR).trans ((x.registered t ht R).trans
          ((x.registered t' ht' R').symm.trans ((W t').leaf_of_com (x.row t' ht') o' hR').symm))
      exact hm96Pair_spec _ _ _ _ (x.enc_ne (v := (W t).vb.val o) (v' := (W t').vb.val o') ht ht' hne) hl

end CarriedLeaf

namespace CarriedBlocks

variable (S : CarriedBlocks) (s : Strategy S.game)

/-- The binding error, in session `t`, at leaf `x`'s commit string against the leaf's value. -/
noncomputable def errX (x : Fin S.nL) (t : ℕ) (τ : Strategy (S.W t).game) : ℝ :=
  errZC execArith hs512 (S.W t).P (planZHJ (S.W t).dj (S.W t).y₀) (tabZHJ (S.W t).dj (S.W t).y₀) (S.W t).vb
    (reg ((S.W t).sim τ)) (cont ((S.W t).sim τ)) ((S.leaf x).pos t) (S.W t).k (S.W t).Rw
    (pull ((S.leaf x).e t) (S.Xm s x))

/-- Leaf `x`'s weight in session `t`: `π/(1 − ρ)` at its commit string there, if the session shares it. -/
noncomputable def wt (x : Fin S.nL) (t : ℕ) : ℝ :=
  if t ∈ (S.leaf x).sessions then (S.W t).readP ((S.leaf x).pos t) / (1 - (S.W t).ρ) else 0

/-- **Session `t`'s carried term**: each leaf's binding error there, weighted. -/
noncomputable def carriedE (t : ℕ) (τ : Strategy (S.W t).game) : ℝ := ∑ x, S.wt x t * S.errX s x t τ

/-- Leaf `x`'s binding error at commit string `c` of session `t`, if it sits there. -/
noncomputable def gTerm (t : ℕ) (τ : Strategy (S.W t).game) (c : (S.W t).vb.Pos) (x : Fin S.nL) : ℝ :=
  if t ∈ (S.leaf x).sessions then (if (S.leaf x).pos t = c then S.errX s x t τ else 0) else 0

/-- The carried leaves' binding errors at commit string `c` of session `t`. -/
noncomputable def gAt (t : ℕ) (τ : Strategy (S.W t).game) (c : (S.W t).vb.Pos) : ℝ := ∑ x, S.gTerm s t τ c x

theorem errX_nonneg (x : Fin S.nL) (t : ℕ) (τ : Strategy (S.W t).game) : 0 ≤ S.errX s x t τ := by
  unfold errX
  exact errZC_nonneg _ _ _ _ _ _ _ _ _ _ _ _

theorem wt_nonneg (x : Fin S.nL) (t : ℕ) : 0 ≤ S.wt x t := by
  unfold wt
  split_ifs
  · exact div_nonneg ((S.W t).readP_nonneg _) (by linarith [(S.W t).hρ])
  · exact le_rfl

theorem carriedE_nonneg (t : ℕ) (τ : Strategy (S.W t).game) : 0 ≤ S.carriedE s t τ :=
  Finset.sum_nonneg fun x _ => mul_nonneg (S.wt_nonneg x t) (S.errX_nonneg s x t τ)

theorem gTerm_nonneg (t : ℕ) (τ : Strategy (S.W t).game) (c : (S.W t).vb.Pos) (x : Fin S.nL) :
    0 ≤ S.gTerm s t τ c x := by
  unfold gTerm
  split_ifs
  · exact S.errX_nonneg s x t τ
  · exact le_rfl
  · exact le_rfl

theorem gAt_nonneg (t : ℕ) (τ : Strategy (S.W t).game) (c : (S.W t).vb.Pos) : 0 ≤ S.gAt s t τ c :=
  Finset.sum_nonneg fun x _ => S.gTerm_nonneg s t τ c x

theorem Xcar_carried (t : ℕ) (R : (S.W t).Reg)
    (κ : Cont (S.W t).L (S.W t).Reg (batchedSessionZC execArith hs512 (planZHJ (S.W t).dj (S.W t).y₀)) R)
    {c : (S.W t).vb.Pos} (h : ∃ x, t ∈ (S.leaf x).sessions ∧ (S.leaf x).pos t = c) :
    ∃ x0, t ∈ (S.leaf x0).sessions ∧ (S.leaf x0).pos t = c ∧
      S.Xcar s t R κ c = pull ((S.leaf x0).e t) (S.Xm s x0) :=
  ⟨h.choose, h.choose_spec.1, h.choose_spec.2, by
    unfold Xcar
    rw [dite_eq_left_of_eq_true (eq_true h)]⟩

theorem Xcar_free (t : ℕ) (R : (S.W t).Reg)
    (κ : Cont (S.W t).L (S.W t).Reg (batchedSessionZC execArith hs512 (planZHJ (S.W t).dj (S.W t).y₀)) R)
    {c : (S.W t).vb.Pos} (h : ¬ ∃ x, t ∈ (S.leaf x).sessions ∧ (S.leaf x).pos t = c) :
    S.Xcar s t R κ c = XvalZC execArith hs512 (S.W t).P (planZHJ (S.W t).dj (S.W t).y₀) (tabZHJ (S.W t).dj (S.W t).y₀)
      (S.W t).vb R κ (S.W t).k (S.W t).Rw c := by
  unfold Xcar
  rw [dite_eq_right_of_eq_false (eq_false h)]

/-- **The binding error at `Xcar`, per commit string**: `δ_link`'s term where no carried leaf sits, under A2 for the
session's link finders, and where one sits, its binding error at the leaf's value. -/
theorem err_le_gAt (t : ℕ) (τ : Strategy (S.W t).game)
    (hL : LinkCRZC execArith hs512 (S.W t).P (planZHJ (S.W t).dj (S.W t).y₀) (tabZHJ (S.W t).dj (S.W t).y₀) (S.W t).vb
      (reg ((S.W t).sim τ)) (cont ((S.W t).sim τ)) (S.W t).k (S.W t).Rw (S.W t).M (S.W t).t')
    (c : (S.W t).vb.Pos) :
    errZC execArith hs512 (S.W t).P (planZHJ (S.W t).dj (S.W t).y₀) (tabZHJ (S.W t).dj (S.W t).y₀) (S.W t).vb
        (reg ((S.W t).sim τ)) (cont ((S.W t).sim τ)) c (S.W t).k (S.W t).Rw
        (S.Xcar s t (reg ((S.W t).sim τ)) (cont ((S.W t).sim τ)) c) ≤
      2 * ((S.W t).t' * (1 + (S.W t).k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * (S.W t).M) + S.gAt s t τ c := by
  by_cases h : ∃ x, t ∈ (S.leaf x).sessions ∧ (S.leaf x).pos t = c
  · obtain ⟨x0, hxt, hpos, hX⟩ := S.Xcar_carried s t (reg ((S.W t).sim τ)) (cont ((S.W t).sim τ)) h
    rw [hX]
    subst hpos
    have hB : 0 ≤ 2 * ((S.W t).t' * (1 + (S.W t).k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * (S.W t).M) := by
      have := (S.W t).ht'
      positivity
    have he : S.gTerm s t τ ((S.leaf x0).pos t) x0 = S.errX s x0 t τ := by
      unfold gTerm
      rw [ite_eq_left_of_eq_true _ _ (eq_true hxt), ite_eq_left_of_eq_true _ _ (eq_true rfl)]
    have h1 := Finset.single_le_sum (fun x _ => S.gTerm_nonneg s t τ ((S.leaf x0).pos t) x) (Finset.mem_univ x0)
    exact (he.symm.le.trans h1).trans (le_add_of_nonneg_left hB)
  · rw [S.Xcar_free s t _ _ h]
    exact (errZC_le execArith hs512 (S.W t).P (planZHJ (S.W t).dj (S.W t).y₀) (tabZHJ (S.W t).dj (S.W t).y₀)
      (S.W t).vb (reg ((S.W t).sim τ)) (cont ((S.W t).sim τ)) (S.W t).k (S.W t).Rw (S.W t).M (S.W t).hM
      (S.W t).ht' c (hL c)).trans (le_add_of_nonneg_right (S.gAt_nonneg s t τ c))

theorem sum_gAt (t : ℕ) (τ : Strategy (S.W t).game) :
    (∑ c, (S.W t).readP c * S.gAt s t τ c) / (1 - (S.W t).ρ) = S.carriedE s t τ := by
  unfold gAt carriedE
  simp only [Finset.mul_sum]
  rw [Finset.sum_comm, Finset.sum_div]
  refine Finset.sum_congr rfl fun x _ => ?_
  unfold gTerm wt
  by_cases hx : t ∈ (S.leaf x).sessions
  · simp only [hx, ↓reduceIte, mul_ite, mul_zero, Finset.sum_ite_eq, Finset.mem_univ]
    ring
  · simp only [hx, ↓reduceIte, mul_zero, Finset.sum_const_zero, zero_div, zero_mul]

/-- **The binding errors at `Xcar`**: `δ_link` plus the carried term. -/
theorem errBound_le (t : ℕ) (τ : Strategy (S.W t).game)
    (hL : LinkCRZC execArith hs512 (S.W t).P (planZHJ (S.W t).dj (S.W t).y₀) (tabZHJ (S.W t).dj (S.W t).y₀) (S.W t).vb
      (reg ((S.W t).sim τ)) (cont ((S.W t).sim τ)) (S.W t).k (S.W t).Rw (S.W t).M (S.W t).t') :
    errBoundZC execArith hs512 (S.W t).P (planZHJ (S.W t).dj (S.W t).y₀) (tabZHJ (S.W t).dj (S.W t).y₀) (S.W t).vb
        (S.Xcar s t) (S.W t).k (S.W t).Rw (S.W t).ρ (reg ((S.W t).sim τ)) (cont ((S.W t).sim τ)) ≤
      (S.W t).link + ENNReal.ofReal (S.carriedE s t τ) := by
  refine ((S.W t).errBound_le_of _ _ (S.Xcar s t) (S.gAt s t τ) (S.err_le_gAt s t τ hL)).trans ?_
  rw [S.sum_gAt s t τ]

/-- **One session's stage**: accepted with a drawn unit wrong, each carried leaf read at its value, with probability
at most `ε_ks + δ_link` plus the carried term. -/
theorem stage_le (t : ℕ) (τ : Strategy (S.W t).game) (hcr : (S.W t).CR τ) :
    prob (fun b => (S.record s t τ b).accepted ∧ (S.record s t τ b).drawnWrong) (S.W t).game τ ≤
      ((S.W t).ks + (S.W t).link) + ENNReal.ofReal (S.carriedE s t τ) := by
  refine ((S.W t).accepted_drawnWrong_le_at (S.Xcar s t) τ hcr).trans ?_
  rw [add_assoc]
  exact add_le_add le_rfl (S.errBound_le s t τ hcr.2)

/-- **A carried leaf's binding error in one of its sessions, over the monitoring period before it**, is at most `m` times its
finder's bound under A2. -/
theorem periodE_errX_le (x : Fin S.nL)
    (hA2 : Assumptions.SHA512CRExpected H512 (finderG (S.mix s x) (S.leaf x).M)
      (finderStrat (S.mix s x) (S.mixS s x) (S.leaf x).M)
      (outOf (rdSig fun _ => (S.leaf x).rd) (S.leaf x).op (S.leaf x).col)
      (finderCost (rdSig fun _ => (S.leaf x).rd) (S.leaf x).costMix))
    {d : ℕ} (hd : d ∈ (S.leaf x).sessions) :
    periodE (fun t => (S.W t).game) (S.errX s x) d 0 S.T s ≤
      2 * (∑ t ∈ (S.leaf x).sessions, ((∑ j ∈ Finset.range t, (S.W j).t') + (S.W t).t' * (1 + (S.W t).k))) /
        2 ^ (256 : ℝ) + (S.leaf x).sessions.card / (Real.exp 1 * (S.leaf x).M) := by
  have h1 : periodE (fun t => (S.W t).game) (S.errX s x) d 0 S.T s ≤
      periodE (fun t => (S.W t).game)
        (fun j τ => errAt ((S.leaf x).trial j τ) ((S.leaf x).trialS j τ) (S.leaf x).rd (S.Xm s x)) d 0 S.T s :=
    periodE_mono (fun t => (S.W t).game) d 0 S.T s (periodGood_true _ 0 S.T s) fun j τ hj _ => by
      rw [zero_add] at hj
      subst hj
      exact (S.leaf x).errZC_le_trial hd τ (S.Xm s x)
  have h2 : errAt (S.comp s x ⟨d, hd⟩) (S.compS s x ⟨d, hd⟩) (S.leaf x).rd (S.Xm s x) =
      periodE (fun t => (S.W t).game)
        (fun j τ => errAt ((S.leaf x).trial j τ) ((S.leaf x).trialS j τ) (S.leaf x).rd (S.Xm s x)) d 0 S.T s := by
    rw [errAt_eq_expect]
    refine (expect_compTo (fun t => (S.W t).game) _ (S.leaf x).trial (S.leaf x).trialS d 0 S.T (S.hin x d hd) s).trans
      ?_
    exact congrArg (fun f => periodE (fun t => (S.W t).game) f d 0 S.T s)
      (funext fun j => funext fun τ => (errAt_eq_expect _ _ _ _).symm)
  have h3 := errAt_le_mix (S.comp s x) (S.compS s x) (fun _ => (S.leaf x).rd) ⟨d, hd⟩ (S.Xm s x)
  have hX : ∀ v, pRd (S.mix s x) (S.mixS s x) (rdSig fun _ => (S.leaf x).rd) v ≤
      pRd (S.mix s x) (S.mixS s x) (rdSig fun _ => (S.leaf x).rd) (S.Xm s x) := fun v =>
    (Classical.choose_spec (Finset.exists_max_image Finset.univ
      (pRd (S.mix s x) (S.mixS s x) (rdSig fun _ => (S.leaf x).rd)) Finset.univ_nonempty)).2 v (Finset.mem_univ v)
  have h4 : errAt (S.mix s x) (S.mixS s x) (rdSig fun _ => (S.leaf x).rd) (S.Xm s x) ≤
      2 * expect (S.leaf x).costMix (S.mix s x) (S.mixS s x) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * (S.leaf x).M) :=
    link_mass_le (S.mix s x) (S.mixS s x) (rdSig fun _ => (S.leaf x).rd) H512
      (outOf (rdSig fun _ => (S.leaf x).rd) (S.leaf x).op (S.leaf x).col) (S.leaf x).costMix (S.leaf x).costMix_nonneg
      (S.leaf x).M (S.leaf x).hM (S.leaf x).out_spec hA2 hX
  have hi : ∀ i : (S.leaf x).Ses, expect (fun e => (S.leaf x).costMix ⟨i, e⟩) (S.comp s x i) (S.compS s x i) ≤
      (∑ j ∈ Finset.range i.1, (S.W j).t') + (S.W i.1).t' * (1 + (S.W i.1).k) := by
    intro i
    have ha : expect (fun e => (S.leaf x).costMix ⟨i, e⟩) (S.comp s x i) (S.compS s x i) =
        (∑ j ∈ Finset.range i.1, (S.W j).t') + expect trialCost (S.comp s x i) (S.compS s x i) :=
      (expect_add (fun _ => ∑ j ∈ Finset.range i.1, (S.W j).t') trialCost _ _).trans (by rw [expect_const])
    rw [ha]
    refine add_le_add le_rfl ?_
    refine le_trans (le_of_eq (expect_compTo (fun t => (S.W t).game) _ (S.leaf x).trial (S.leaf x).trialS i.1 0 S.T
      (S.hin x i.1 i.2) s)) ?_
    refine (periodE_mono (fun t => (S.W t).game) i.1 0 S.T s (periodGood_true _ 0 S.T s)
      (f' := fun j _ => (S.W j).t' * (1 + (S.W j).k)) fun j τ _ _ => (S.leaf x).expect_cost_le j τ).trans ?_
    refine (periodE_const_le (fun t => (S.W t).game) (fun j => (S.W j).t' * (1 + (S.W j).k))
      (fun j => mul_nonneg (S.W j).ht' (by positivity)) i.1 0 S.T s).trans (le_of_eq ?_)
    simp only [zero_add]
  have hm0 : (0 : ℝ) < Fintype.card (S.leaf x).Ses := by exact_mod_cast Fintype.card_pos
  have hE : (Fintype.card (S.leaf x).Ses : ℝ) * expect (S.leaf x).costMix (S.mix s x) (S.mixS s x) ≤
      ∑ t ∈ (S.leaf x).sessions, ((∑ j ∈ Finset.range t, (S.W j).t') + (S.W t).t' * (1 + (S.W t).k)) := by
    rw [show expect (S.leaf x).costMix (S.mix s x) (S.mixS s x) = _ from
      expect_mixG (S.comp s x) (S.compS s x) (S.leaf x).costMix, mul_div_cancel₀ _ hm0.ne']
    refine (Finset.sum_le_sum fun i _ => hi i).trans (le_of_eq ?_)
    exact Finset.sum_coe_sort (S.leaf x).sessions
      (fun t => (∑ j ∈ Finset.range t, (S.W j).t') + (S.W t).t' * (1 + (S.W t).k))
  have hcard : (Fintype.card (S.leaf x).Ses : ℝ) = (S.leaf x).sessions.card := by
    exact_mod_cast Fintype.card_coe (S.leaf x).sessions
  calc periodE (fun t => (S.W t).game) (S.errX s x) d 0 S.T s
      ≤ errAt (S.comp s x ⟨d, hd⟩) (S.compS s x ⟨d, hd⟩) (S.leaf x).rd (S.Xm s x) := h1.trans h2.symm.le
    _ ≤ Fintype.card (S.leaf x).Ses * errAt (S.mix s x) (S.mixS s x) (rdSig fun _ => (S.leaf x).rd) (S.Xm s x) := h3
    _ ≤ Fintype.card (S.leaf x).Ses * (2 * expect (S.leaf x).costMix (S.mix s x) (S.mixS s x) / 2 ^ (256 : ℝ) +
          1 / (Real.exp 1 * (S.leaf x).M)) := mul_le_mul_of_nonneg_left h4 hm0.le
    _ = 2 * (Fintype.card (S.leaf x).Ses * expect (S.leaf x).costMix (S.mix s x) (S.mixS s x)) / 2 ^ (256 : ℝ) +
          Fintype.card (S.leaf x).Ses / (Real.exp 1 * (S.leaf x).M) := by ring
    _ ≤ _ := by
        rw [hcard] at hE ⊢
        gcongr

/-- **A carried leaf's term over the monitoring period**: its weighted binding errors, summed over the monitoring period's sessions, are at
most `δ_carry`. -/
theorem sum_periodE_le (x : Fin S.nL)
    (hA2 : Assumptions.SHA512CRExpected H512 (finderG (S.mix s x) (S.leaf x).M)
      (finderStrat (S.mix s x) (S.mixS s x) (S.leaf x).M)
      (outOf (rdSig fun _ => (S.leaf x).rd) (S.leaf x).op (S.leaf x).col)
      (finderCost (rdSig fun _ => (S.leaf x).rd) (S.leaf x).costMix)) :
    ENNReal.ofReal (∑ d ∈ Finset.range S.T, S.wt x d * periodE (fun t => (S.W t).game) (S.errX s x) d 0 S.T s) ≤
      (S.leaf x).carry := by
  unfold CarriedLeaf.carry
  apply ENNReal.ofReal_le_ofReal
  have hsub : (S.leaf x).sessions ⊆ Finset.range S.T := fun d hd => Finset.mem_range.2 (S.hin x d hd)
  rw [← Finset.sum_subset hsub fun d _ hd => by simp only [wt, hd, ↓reduceIte, zero_mul], Finset.sum_mul]
  refine Finset.sum_le_sum fun d hd => ?_
  simp only [wt, hd, ↓reduceIte]
  exact mul_le_mul_of_nonneg_left (S.periodE_errX_le s x hA2 hd)
    (div_nonneg ((S.W d).readP_nonneg _) (by linarith [(S.W d).hρ]))

/-- **The carried terms over the monitoring period**: at most `Σ_ℓ δ_carry,ℓ`. -/
theorem carry_le
    (hA2 : ∀ x, Assumptions.SHA512CRExpected H512 (finderG (S.mix s x) (S.leaf x).M)
      (finderStrat (S.mix s x) (S.mixS s x) (S.leaf x).M)
      (outOf (rdSig fun _ => (S.leaf x).rd) (S.leaf x).op (S.leaf x).col)
      (finderCost (rdSig fun _ => (S.leaf x).rd) (S.leaf x).costMix)) :
    ENNReal.ofReal (∑ d ∈ Finset.range S.T, periodE (fun t => (S.W t).game) (S.carriedE s) d 0 S.T s) ≤
      ∑ x, (S.leaf x).carry := by
  have hsplit : ∑ d ∈ Finset.range S.T, periodE (fun t => (S.W t).game) (S.carriedE s) d 0 S.T s =
      ∑ x, ∑ d ∈ Finset.range S.T, S.wt x d * periodE (fun t => (S.W t).game) (S.errX s x) d 0 S.T s := by
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun d _ => ?_
    refine (periodE_sum (fun t => (S.W t).game) Finset.univ (fun x j τ => S.wt x j * S.errX s x j τ) d 0 S.T s).trans
      (Finset.sum_congr rfl fun x _ => ?_)
    refine (periodE_mul (fun t => (S.W t).game) (S.wt x) (S.errX s x) d 0 S.T s).trans ?_
    rw [zero_add]
  have hnn : ∀ x ∈ (Finset.univ : Finset (Fin S.nL)),
      0 ≤ ∑ d ∈ Finset.range S.T, S.wt x d * periodE (fun t => (S.W t).game) (S.errX s x) d 0 S.T s :=
    fun x _ => Finset.sum_nonneg fun d _ => mul_nonneg (S.wt_nonneg x d)
      (periodE_nonneg (fun t => (S.W t).game) (S.errX_nonneg s x) d 0 S.T s)
  rw [hsplit, ENNReal.ofReal_sum_of_nonneg hnn]
  exact Finset.sum_le_sum fun x _ => S.sum_periodE_le s x (hA2 x)

end CarriedBlocks

end Flock.Guarantees

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.Game Flock.Guarantees
open scoped ENNReal

/-- **The integrity audit over a monitoring period of server-time blocks, with carried state.** -/
theorem IntegrityHorizonCarried : Flock.Guarantees.IntegrityHorizonCarried := by
  intro S q δ hq hδ hqt hA3 s hT hL hA2
  classical
  have hs : PeriodGood (fun t => (S.W t).game) (fun t => (S.W t).CR) 0 S.T s :=
    periodGood_of_plays _ _ S.T 0 s fun t τ hp => And.intro (hT t τ hp) (hL t τ hp)
  set M : ℕ := ⌈Real.log (1 / δ) / q⌉₊
  set xq : ℝ≥0∞ := ENNReal.ofReal (1 - q)
  have hx : xq ≤ 1 := ENNReal.ofReal_le_one.2 (by linarith)
  have hfac : ∀ t < S.T, ∀ m : ℕ, (1 - ((S.W t).kd : ℝ≥0∞) / (S.W t).n) ^ m ≤ xq ^ m :=
    fun t ht m => pow_le_pow_left₀ zero_le ((S.W t).one_sub_le hq (hqt t ht)) m
  have hgood := periodGood_mono _ S.T 0 s (fun j τ _ hj hcr => (⟨by omega, hcr⟩ : j < S.T ∧ (S.W j).CR τ)) hs
  -- some session accepted with a drawn unit wrong, each carried leaf read at its value
  have hA := period_unionE (fun t => (S.W t).game) (fun t => S.record s t) (fun r => r.accepted ∧ r.drawnWrong)
    (fun t τ => t < S.T ∧ (S.W t).CR τ) (fun t => (S.W t).ks + (S.W t).link) (S.carriedE s) (S.carriedE_nonneg s)
    (fun t τ hτ => S.stage_le s t τ hτ.2) S.T 0 s hgood
  -- every draw misses, with at least `M` wrong units in all
  have hB := period_horizon (fun t => (S.W t).game) (fun t => S.record s t) (fun r => ¬ r.drawnWrong)
    WindowRec.wrong (fun t τ => t < S.T ∧ (S.W t).CR τ) xq hx
    (fun t τ hτ => ⟨(S.wrongSet s t τ).card, fun _ => rfl,
      ((S.W t).drawMiss_le (hA3 t hτ.1) _ τ).trans (hfac t hτ.1 _)⟩) S.T 0 s M hgood
  have hxM : xq ^ M ≤ ENNReal.ofReal δ := one_sub_pow_le_of_log hq hδ (Nat.le_ceil _)
  have hε : ∑ t ∈ Finset.range S.T, (fun t => (S.W t).ks + (S.W t).link) (0 + t) =
      ∑ b ∈ Finset.range S.B, S.err b := by
    simp only [zero_add]
    exact sum_range_start S.toServerBlocks (fun t => (S.W t).ks + (S.W t).link) S.B
  calc value (fun l => (∀ r ∈ l, r.accepted) ∧ Real.log (1 / δ) / q ≤ ((l.map WindowRec.wrong).sum : ℝ)) (S.run s)
      ≤ value (fun l => (∃ r ∈ l, r.accepted ∧ r.drawnWrong) ∨
          ((∀ r ∈ l, ¬ r.drawnWrong) ∧ M ≤ (l.map WindowRec.wrong).sum)) (S.run s) := by
        refine value_mono (fun l hl => ?_) _
        by_cases hd : ∃ r ∈ l, r.drawnWrong
        · obtain ⟨r, hr, hdr⟩ := hd
          exact Or.inl ⟨r, hr, hl.1 r hr, hdr⟩
        · exact Or.inr ⟨fun r hr hdr => hd ⟨r, hr, hdr⟩, Nat.ceil_le.2 hl.2⟩
    _ ≤ value (fun l => ∃ r ∈ l, r.accepted ∧ r.drawnWrong) (S.run s) +
          value (fun l => (∀ r ∈ l, ¬ r.drawnWrong) ∧ M ≤ (l.map WindowRec.wrong).sum) (S.run s) :=
        value_or_le _ _ _
    _ ≤ (∑ b ∈ Finset.range S.B, S.err b + ∑ x, (S.leaf x).carry) + ENNReal.ofReal δ := by
        refine add_le_add ?_ (hB.trans hxM)
        refine hA.trans ?_
        rw [hε]
        exact add_le_add le_rfl (S.carry_le s hA2)
    _ = ENNReal.ofReal δ + ∑ b ∈ Finset.range S.B, S.err b + ∑ x, (S.leaf x).carry := by ring

end Flock.SecurityProofs
