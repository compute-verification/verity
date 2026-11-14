module

public import Proofs.Flock.Soundness.Game.Tape
public import Proofs.Flock.Soundness.TmplUnitAt
public import Proofs.Flock.EndToEnd.Joint.V5.Session

/-!
# A member's session in the window's audit, played alone

A strategy of the window's audit reads the sample and every earlier member's session before it plays member `k`'s, so
where the sample ended member `k`'s session in the audit is its session played alone at `JointAuditV5.memberLaw`, whose
coins are the sample's at which the sample ended and the earlier sessions' sampled together, by
`JointAuditV5.memberStrat`:

* `JointAuditV5.prob_member`: an event of member `k`'s part of the sample and its session's transcript, with the sample
  ended, has in the window's audit the probability it has in member `k`'s session played alone times the share of the
  sample's coins at which the sample ended, `JointAuditV5.endShare` (`prob_seqG`, then `avg_ite_subtype` and
  `avg_prod_pi` over the sample's coins and, for each, the earlier sessions' coins);
* `JointAuditV5.endShare_mul_epsKs_le`, `JointAuditV5.endShare_mul_link_le`: `ε_ks` and `ε_bind` read only member `k`'s
  part of the sample, so that share of each at `memberLaw` is at most the term at `design k`, which samples the sample's
  coins alone and counts the samples that ran out too;
* `JointAuditV5.member_accepted_drawnWrong_le`: the sample ends and member `k`'s session in the window's audit accepts
  with a sampled proof block of its part incorrect with probability at most `ε_ks,k + ε_bind,k` at `design k`
  (`MemberV5.accepted_drawnWrong_le` at `memberLaw`).

`JointAuditV5.hcls_of_bounds` gives `JointAuditV5.hcls` from the registration check's refusal of a member whose fewest
and most sampled proof blocks have different link points (`mPtsOf_mono`).
-/

@[expose] public section

namespace FlockSoundness.Discharge.Exec

/-- `m_pts` grows with the number of proved units. -/
theorem mPtsOf_mono (g : ℕ) {a b : ℕ} (h : a ≤ b) : mPtsOf g a ≤ mPtsOf g b := by
  have hd : (a + g - 1) / g ≤ (b + g - 1) / g := Nat.div_le_div_right (by omega)
  exact Nat.add_le_add_left (max_le_max (le_refl 3) (log2ceil_mono (max_le_max hd (le_refl 1)))) _

end FlockSoundness.Discharge.Exec

namespace Flock.Guarantees.JointAuditV5

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open scoped ENNReal

open Classical

variable (w : JointAuditV5)

/-- An average of a function that is zero off `p` is `p`'s share of the coins times the average over `p`. -/
theorem avg_ite_subtype {C : Type} [Fintype C] (p : C → Prop) [DecidablePred p] [Fintype {c // p c}]
    [Nonempty {c // p c}] (f : C → ℝ≥0∞) :
    avg (fun c => if p c then f c else 0) =
      (Fintype.card {c // p c} : ℝ≥0∞) / Fintype.card C * avg (fun x : {c // p c} => f x.1) := by
  have h0 : (Fintype.card {c // p c} : ℝ≥0∞) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  have ht : (Fintype.card {c // p c} : ℝ≥0∞) ≠ ⊤ := ENNReal.natCast_ne_top _
  calc avg (fun c => if p c then f c else 0)
      = ∑ c ∈ Finset.univ.filter p, (Fintype.card C : ℝ≥0∞)⁻¹ * f c := by
        unfold avg
        rw [Finset.sum_filter]
        refine Finset.sum_congr rfl fun c _ => ?_
        beta_reduce
        by_cases hc : p c
        · rw [ite_eq_left hc, ite_eq_left hc]
        · rw [ite_eq_right hc, ite_eq_right hc, mul_zero]
    _ = ∑ x : {c // p c}, (Fintype.card C : ℝ≥0∞)⁻¹ * f x.1 :=
        Finset.sum_subtype _ (fun c => by simp) _
    _ = _ := by
        unfold avg
        rw [Finset.mul_sum]
        refine Finset.sum_congr rfl fun x _ => ?_
        rw [div_eq_mul_inv, mul_mul_mul_comm, ENNReal.mul_inv_cancel h0 ht, one_mul]

/-- **The share of the sample's coins at which the sample ended.** -/
noncomputable def endShare : ℝ≥0∞ := (Fintype.card w.EndΩ : ℝ≥0∞) / Fintype.card w.L.Ω

section Ended

variable [Nonempty w.EndΩ]

/-- **Where the sample ended, member `k`'s session in the window's audit is its session played alone**: an event `E` of
member `k`'s part of the sample and its session's transcript, with the sample ended, has in the window's audit the
probability `E` has in the session played alone times the share of the sample's coins at which the sample ended. -/
theorem prob_member (τ : Strategy w.game) (k : Fin w.ms.length)
    (E : Finset (Fin (w.ms.get k).n) → Discharge.Exec.Transcript → Prop) :
    prob (fun o => (o.2.2[k.1]?).elim False (E ((w.ms.get k).part o.2.1)) ∧ o.1.2 = true) w.game τ =
      w.endShare * prob (fun o => E o.2.1 o.2.2) ((w.ms.get k).game (w.memberLaw τ k)) (w.memberStrat τ k) := by
  let h : (ω : w.L.Ω) → w.before τ k ω → ℝ≥0∞ := fun ω t =>
    prob (E ((w.ms.get k).part (w.L.draw ω))) ((w.ms.get k).live ((w.ms.get k).part (w.L.draw ω)) τ.1)
      (seqStrat (fun m : MemberV5 w.Reg w.nJ => m.live (m.part (w.L.draw ω)) τ.1) w.ms k.1 k.2
        (Strategy.ofMap (fun ts => ((τ.1, w.ended ω), w.L.draw ω, ts)) _ (τ.2 ω)) t)
  have hL : prob (fun o => (o.2.2[k.1]?).elim False (E ((w.ms.get k).part o.2.1)) ∧ o.1.2 = true) w.game τ =
      avg fun ω : w.L.Ω => if w.ended ω = true then avg (h ω) else 0 := by
    show avg (fun ω => prob _
      ((w.sessions (w.L.draw ω) τ.1).map fun ts => ((τ.1, w.ended ω), w.L.draw ω, ts)) (τ.2 ω)) = _
    refine congrArg avg (funext fun ω => ?_)
    rw [prob_map]
    show prob (fun ts : List Discharge.Exec.Transcript =>
      (ts[k.1]?).elim False (E ((w.ms.get k).part (w.L.draw ω))) ∧ w.ended ω = true) _ _ = _
    rw [prob_and_const]
    by_cases he : w.ended ω = true
    · rw [ite_eq_left he, ite_eq_left he]
      exact (prob_seqG (fun m : MemberV5 w.Reg w.nJ => m.live (m.part (w.L.draw ω)) τ.1)
        (fun x => x.elim False (E ((w.ms.get k).part (w.L.draw ω)))) w.ms k.1 k.2 _).trans
        (avg_congr_fintype _ _ _)
    · rw [ite_eq_right he, ite_eq_right he]
  have hR : prob (fun o => E o.2.1 o.2.2) ((w.ms.get k).game (w.memberLaw τ k)) (w.memberStrat τ k) =
      avg fun x : (w.memberLaw τ k).Ω => h x.1.1 (x.2 x.1) := by
    show avg (fun x => prob (fun o => E o.2.1 o.2.2) (((w.ms.get k).live ((w.memberLaw τ k).draw x) τ.1).map
      fun t => (τ.1, (w.memberLaw τ k).draw x, t))
        (Strategy.map (fun t => (τ.1, (w.memberLaw τ k).draw x, t)) _
          (seqStrat (fun m : MemberV5 w.Reg w.nJ => m.live (m.part (w.L.draw x.1.1)) τ.1) w.ms k.1 k.2
            (Strategy.ofMap (fun ts => ((τ.1, w.ended x.1.1), w.L.draw x.1.1, ts)) _ (τ.2 x.1.1)) (x.2 x.1)))) = _
    refine congrArg avg (funext fun x => ?_)
    rw [prob_map]
    exact congrArg (prob _ _) (Strategy.ofMap_map _ _ _)
  rw [hL, hR]
  refine (avg_ite_subtype (fun ω : w.L.Ω => w.ended ω = true) fun ω => avg (h ω)).trans ?_
  refine congrArg (fun x : ℝ≥0∞ => w.endShare * x) ?_
  exact (avg_prod_pi fun (c : w.EndΩ) (t : w.before τ k c.1) => h c.1 t).symm.trans (avg_congr_fintype _ _ _)

/-- **The sampling design of member `k`'s part of the sample, given that the sample ended**: the sample's coins at which
it ended, member `k`'s part. -/
noncomputable def endDesign (k : Fin w.ms.length) : Law (w.ms.get k).n where
  Ω := w.EndΩ
  draw c := (w.ms.get k).part (w.L.draw c.1)

/-- An average over member `k`'s sampling design played alone of a function of its part of the sample is the average
over the sample's coins at which the sample ended. -/
theorem avg_memberLaw (τ : Strategy w.game) (k : Fin w.ms.length) (G : Finset (Fin (w.ms.get k).n) → ℝ≥0∞) :
    avg (fun x : (w.memberLaw τ k).Ω => G ((w.memberLaw τ k).draw x)) =
      avg (fun c : (w.endDesign k).Ω => G ((w.endDesign k).draw c)) :=
  (avg_congr_fintype _ _ _).trans
    (avg_fst (B := (c : w.EndΩ) → w.before τ k c.1) fun c : w.EndΩ => G ((w.ms.get k).part (w.L.draw c.1)))

/-- **The ended share of an average over member `k`'s sampling design played alone** of a function of its part of the
sample is at most the average over `design k`, which counts the samples that ran out too. -/
theorem endShare_mul_avg_le (τ : Strategy w.game) (k : Fin w.ms.length) (G : Finset (Fin (w.ms.get k).n) → ℝ≥0∞) :
    w.endShare * avg (fun x : (w.memberLaw τ k).Ω => G ((w.memberLaw τ k).draw x)) ≤
      avg (fun ω : (w.design k).Ω => G ((w.design k).draw ω)) := by
  rw [w.avg_memberLaw τ k G]
  refine (avg_ite_subtype (fun ω : w.L.Ω => w.ended ω = true)
    fun ω => G ((w.ms.get k).part (w.L.draw ω))).symm.trans_le ?_
  refine avg_mono fun ω => ?_
  by_cases he : w.ended ω = true
  · rw [ite_eq_left he]
    exact le_rfl
  · rw [ite_eq_right he]
    exact zero_le

/-- **`ε_ks` reads only the member's part of the sample**: its ended share at `memberLaw` is at most `ε_ks` at
`design k`. -/
theorem endShare_mul_epsKs_le (τ : Strategy w.game) (k : Fin w.ms.length) :
    w.endShare * (w.ms.get k).epsKs (w.memberLaw τ k) ≤ (w.ms.get k).epsKs (w.design k) := by
  unfold MemberV5.epsKs
  rw [ENNReal.mul_iSup]
  exact iSup_mono fun R => w.endShare_mul_avg_le τ k fun S =>
    ⨆ j : Fin (planZHJP (w.ms.get k).dj (w.ms.get k).y₀ S R).length,
      ENNReal.ofReal (ksStrictZ (planZHJP (w.ms.get k).dj (w.ms.get k).y₀ S R) j (w.ms.get k).k (w.ms.get k).qF
        (w.ms.get k).qS / (1 - rateZ (planZHJP (w.ms.get k).dj (w.ms.get k).y₀ S R) j (w.ms.get k).k))

/-- The share of member `k`'s sampling design played alone whose samples read commit string `p` is that of the sample's
coins at which the sample ended. -/
theorem readShare_memberLaw (τ : Strategy w.game) (k : Fin w.ms.length) (p : (w.ms.get k).vb.Pos) :
    (Fintype.card (ΩcZ execArith (w.ms.get k).P (planZHJP (w.ms.get k).dj (w.ms.get k).y₀)
        (tabZHJP (w.ms.get k).dj (w.ms.get k).y₀) (w.ms.get k).vb (w.memberLaw τ k) p) : ℝ) /
        Fintype.card (w.memberLaw τ k).Ω =
      (Fintype.card (ΩcZ execArith (w.ms.get k).P (planZHJP (w.ms.get k).dj (w.ms.get k).y₀)
        (tabZHJP (w.ms.get k).dj (w.ms.get k).y₀) (w.ms.get k).vb (w.endDesign k) p) : ℝ) /
        Fintype.card w.EndΩ := by
  have e : ΩcZ execArith (w.ms.get k).P (planZHJP (w.ms.get k).dj (w.ms.get k).y₀)
        (tabZHJP (w.ms.get k).dj (w.ms.get k).y₀) (w.ms.get k).vb (w.memberLaw τ k) p ≃
      ΩcZ execArith (w.ms.get k).P (planZHJP (w.ms.get k).dj (w.ms.get k).y₀)
        (tabZHJP (w.ms.get k).dj (w.ms.get k).y₀) (w.ms.get k).vb (w.endDesign k) p ×
          ((c : w.EndΩ) → w.before τ k c.1) :=
    Equiv.prodSubtypeFstEquivSubtypeProd (p := fun c : w.EndΩ => ReadsDrawZ (L := w.endDesign k) execArith
      (w.ms.get k).P (planZHJP (w.ms.get k).dj (w.ms.get k).y₀) (tabZHJP (w.ms.get k).dj (w.ms.get k).y₀)
      (w.ms.get k).vb p c)
  have hΩ : Fintype.card (w.memberLaw τ k).Ω =
      Fintype.card w.EndΩ * Fintype.card ((c : w.EndΩ) → w.before τ k c.1) :=
    Fintype.card_prod _ _
  rw [Fintype.card_congr e, Fintype.card_prod, hΩ, Nat.cast_mul, Nat.cast_mul,
    mul_div_mul_right _ _ (Nat.cast_ne_zero.2 (Fintype.card_ne_zero (α := (c : w.EndΩ) → w.before τ k c.1)))]

/-- The samples that ended and read commit string `p` are among the samples that read it. -/
theorem card_endDesign_le (k : Fin w.ms.length) (p : (w.ms.get k).vb.Pos) :
    Fintype.card (ΩcZ execArith (w.ms.get k).P (planZHJP (w.ms.get k).dj (w.ms.get k).y₀)
        (tabZHJP (w.ms.get k).dj (w.ms.get k).y₀) (w.ms.get k).vb (w.endDesign k) p) ≤
      Fintype.card (ΩcZ execArith (w.ms.get k).P (planZHJP (w.ms.get k).dj (w.ms.get k).y₀)
        (tabZHJP (w.ms.get k).dj (w.ms.get k).y₀) (w.ms.get k).vb (w.design k) p) :=
  Fintype.card_le_of_injective
    (fun c : ΩcZ execArith (w.ms.get k).P (planZHJP (w.ms.get k).dj (w.ms.get k).y₀)
        (tabZHJP (w.ms.get k).dj (w.ms.get k).y₀) (w.ms.get k).vb (w.endDesign k) p =>
      (⟨c.1.1, c.2⟩ : ΩcZ execArith (w.ms.get k).P (planZHJP (w.ms.get k).dj (w.ms.get k).y₀)
        (tabZHJP (w.ms.get k).dj (w.ms.get k).y₀) (w.ms.get k).vb (w.design k) p))
    fun a b hab => by
      have h := congrArg Subtype.val hab
      exact Subtype.ext (Subtype.ext h)

/-- A share `cE / cL` of a binding term at shares `a p / cE` is at most the term at shares `b p / cL`, for `a ≤ b`. -/
theorem share_scale {ι : Type} [Fintype ι] {cE cL c d : ℝ} (hcE : 0 < cE) (hcL : 0 < cL) (hc : 0 ≤ c) (hd : 0 < d)
    (a b s : ι → ℝ) (hs : ∀ p, s p = a p / cE) (hab : ∀ p, a p ≤ b p) :
    ENNReal.ofReal (cE / cL) * ENNReal.ofReal ((∑ p, s p) * c / d) ≤ ENNReal.ofReal ((∑ p, b p / cL) * c / d) := by
  rw [← ENNReal.ofReal_mul (div_nonneg hcE.le hcL.le)]
  refine ENNReal.ofReal_le_ofReal ?_
  have key : cE / cL * ∑ p, s p ≤ ∑ p, b p / cL := by
    rw [Finset.mul_sum]
    refine Finset.sum_le_sum fun p _ => ?_
    rw [hs p]
    calc cE / cL * (a p / cE) = cE * a p / (cL * cE) := div_mul_div_comm _ _ _ _
      _ = a p * cE / (cL * cE) := by rw [mul_comm cE (a p)]
      _ = a p / cL := mul_div_mul_right _ _ hcE.ne'
      _ ≤ b p / cL := div_le_div_of_nonneg_right (hab p) hcL.le
  calc cE / cL * ((∑ p, s p) * c / d) = cE / cL * (∑ p, s p) * (c / d) := by ring
    _ ≤ (∑ p, b p / cL) * (c / d) := mul_le_mul_of_nonneg_right key (div_nonneg hc hd.le)
    _ = (∑ p, b p / cL) * c / d := by ring

/-- **`ε_bind` reads only the member's part of the sample**: its ended share at `memberLaw` is at most `ε_bind` at
`design k`. -/
theorem endShare_mul_link_le (τ : Strategy w.game) (k : Fin w.ms.length) :
    w.endShare * (w.ms.get k).link (w.memberLaw τ k) ≤ (w.ms.get k).link (w.design k) := by
  have hE : (0 : ℝ) < Fintype.card w.EndΩ := by exact_mod_cast Fintype.card_pos
  have hL : (0 : ℝ) < Fintype.card w.L.Ω := by exact_mod_cast Fintype.card_pos
  have hs : w.endShare = ENNReal.ofReal ((Fintype.card w.EndΩ : ℝ) / (Fintype.card w.L.Ω : ℝ)) := by
    show (Fintype.card w.EndΩ : ℝ≥0∞) / (Fintype.card w.L.Ω : ℝ≥0∞) = _
    rw [ENNReal.ofReal_div_of_pos hL, ENNReal.ofReal_natCast, ENNReal.ofReal_natCast]
  have ht := (w.ms.get k).ht'
  rw [hs]
  unfold MemberV5.link
  exact share_scale hE hL (by positivity) (sub_pos.2 (w.ms.get k).hρ)
    (fun p => (Fintype.card (ΩcZ execArith (w.ms.get k).P (planZHJP (w.ms.get k).dj (w.ms.get k).y₀)
      (tabZHJP (w.ms.get k).dj (w.ms.get k).y₀) (w.ms.get k).vb (w.endDesign k) p) : ℝ))
    (fun p => (Fintype.card (ΩcZ execArith (w.ms.get k).P (planZHJP (w.ms.get k).dj (w.ms.get k).y₀)
      (tabZHJP (w.ms.get k).dj (w.ms.get k).y₀) (w.ms.get k).vb (w.design k) p) : ℝ))
    _ (fun p => w.readShare_memberLaw τ k p) fun p => by exact_mod_cast w.card_endDesign_le k p

/-- **The sample ends and member `k`'s session in the window's audit accepts with a sampled proof block of its part
incorrect** with probability at most `ε_ks,k + ε_bind,k` at `design k`, under `cr/sha-512` and `ecr/sha-512` at its
session played alone. -/
theorem member_accepted_drawnWrong_le (τ : Strategy w.game) (k : Fin w.ms.length)
    (hCR : ∀ ω j, TableCRZ execArith hs512
      (planZHJP (w.ms.get k).dj (w.ms.get k).y₀ ((w.memberLaw τ k).draw ω) (reg (w.memberSim τ k))) j
      (zcStrat execArith hs512 (planZHJP (w.ms.get k).dj (w.ms.get k).y₀) (cont (w.memberSim τ k) ω))
      (w.ms.get k).qF (w.ms.get k).qS)
    (hA2 : LinkCRZC execArith hs512 (w.ms.get k).P (planZHJP (w.ms.get k).dj (w.ms.get k).y₀)
      (tabZHJP (w.ms.get k).dj (w.ms.get k).y₀) (w.ms.get k).vb (reg (w.memberSim τ k)) (cont (w.memberSim τ k))
      (w.ms.get k).k (w.ms.get k).Rw (w.ms.get k).M (w.ms.get k).t') :
    prob (fun o => (o.2.2[k.1]?).elim False (fun t => AcceptsZKP (w.ms.get k).I (w.ms.get k).own t ∧
        ∃ u ∈ (w.ms.get k).part o.2.1, u ∈ w.memberWrong τ k) ∧ o.1.2 = true) w.game τ ≤
      (w.ms.get k).epsKs (w.design k) + (w.ms.get k).link (w.design k) := by
  refine (w.prob_member τ k fun S t => AcceptsZKP (w.ms.get k).I (w.ms.get k).own t ∧
    ∃ u ∈ S, u ∈ w.memberWrong τ k).trans_le ?_
  refine (mul_le_mul' le_rfl ((w.ms.get k).accepted_drawnWrong_le (w.memberLaw τ k) (w.oneClass_memberLaw τ k)
    (w.memberStrat τ k) hCR hA2)).trans ?_
  rw [mul_add]
  exact add_le_add (w.endShare_mul_epsKs_le τ k) (w.endShare_mul_link_le τ k)

end Ended

/-- **`hcls` from the registration's refusal**: if at every sample that ended member `k`'s part holds between `lo k` and
`hi k` of its proof blocks, its record's count `kd / nTab I` lies between them too, and `lo k` and `hi k` give the same
link points (the registration check refuses a member whose fewest and most sampled proof blocks don't), then the part
of every sample that ended has the record's link points. -/
theorem hcls_of_bounds {Reg : Type} {nJ : ℕ} (ms : List (MemberV5 Reg nJ)) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    {Lb : Fin 1 → ℕ} (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (law : Flock.Draw.Law)
    (lo hi : Fin ms.length → ℕ)
    (hdraw : ∀ k ω, sampleEnded src law nJ ω = true →
      lo k ≤ ((ms.get k).part ((Bridge.execSel (n := nJ) src law nJ).draw ω)).card ∧
        ((ms.get k).part ((Bridge.execSel (n := nJ) src law nJ).draw ω)).card ≤ hi k)
    (hrec : ∀ k, lo k ≤ (ms.get k).kd / nTab (ms.get k).I ∧ (ms.get k).kd / nTab (ms.get k).I ≤ hi k)
    (hends : ∀ k, mPtsOf (ms.get k).D.unroll.g (lo k) = mPtsOf (ms.get k).D.unroll.g (hi k)) :
    ∀ (k : Fin ms.length) (ω : Ω), sampleEnded src law nJ ω = true →
      mPtsOf (ms.get k).D.unroll.g ((ms.get k).part ((Bridge.execSel (n := nJ) src law nJ).draw ω)).card =
        mPtsOf (ms.get k).D.unroll.g ((ms.get k).kd / nTab (ms.get k).I) := by
  intro k ω he
  have h1 := mPtsOf_mono (ms.get k).D.unroll.g (hdraw k ω he).1
  have h2 := mPtsOf_mono (ms.get k).D.unroll.g (hdraw k ω he).2
  have h3 := mPtsOf_mono (ms.get k).D.unroll.g (hrec k).1
  have h4 := mPtsOf_mono (ms.get k).D.unroll.g (hrec k).2
  have h5 := hends k
  omega

end Flock.Guarantees.JointAuditV5
