import Proofs.Flock.Soundness.Game.Tape
import Proofs.Flock.Soundness.TmplUnitAt
import Proofs.Flock.EndToEnd.Joint.V5.Session

/-!
# A member's session in the window's audit, played alone

A strategy of the window's audit reads the sample and every earlier member's session before it plays member `k`'s, so
member `k`'s session in the audit is its session played alone at `JointAuditV5.memberLaw`, whose coins are the sample's
on which the draw ends and the earlier sessions' drawn together, by `JointAuditV5.memberStrat`:

* `JointAuditV5.prob_member`: an event of member `k`'s part of the sample and its session's transcript has probability in
  the window's audit at most its probability in member `k`'s session played alone: a draw that runs out records no
  transcript, so the event needs a draw that ends (`avg_ite_le`), and there the two are equal (`prob_seqG`, then
  `avg_prod_pi` over the sample's coins and, for each, the earlier sessions' coins);
* `JointAuditV5.epsKs_memberLaw`, `JointAuditV5.link_memberLaw`: `ε_ks` and `ε_bind` read only member `k`'s part of the
  sample, so they are the same at `memberLaw` as at `design k`, which draws the sample's coins that end alone;
* `JointAuditV5.member_accepted_drawnWrong_le`: member `k`'s session in the window's audit accepts with a sampled proof
  block of its part incorrect with probability at most `ε_ks,k + ε_bind,k` at `design k`
  (`MemberV5.accepted_drawnWrong_le` at `memberLaw`).

`JointAuditV5.hcls_of_bounds` gives `JointAuditV5.hcls` from the registration check's refusal of a member whose fewest
and most proof blocks over the draws that end have different link points (`mPtsOf_mono`).
-/

namespace FlockSoundness.Discharge.Exec

/-- `m_pts` grows with the number of proved units. -/
theorem mPtsOf_mono (g : ℕ) {a b : ℕ} (h : a ≤ b) : mPtsOf g a ≤ mPtsOf g b := by
  have hd : (a + g - 1) / g ≤ (b + g - 1) / g := Nat.div_le_div_right (by omega)
  exact Nat.add_le_add_left (max_le_max (le_refl 3) (log2ceil_mono (max_le_max hd (le_refl 1)))) _

end FlockSoundness.Discharge.Exec

namespace FlockSoundness.Game

open scoped ENNReal

/-- **An average that is zero off `p`** is at most the average over the coins where `p` holds. -/
theorem avg_ite_le {C : Type} [Fintype C] (p : C → Prop) [DecidablePred p] (f : C → ℝ≥0∞) :
    avg (fun c => if p c then f c else 0) ≤ avg fun c : {c // p c} => f c.1 := by
  have hc : (Fintype.card C : ℝ≥0∞)⁻¹ ≤ (Fintype.card {c // p c} : ℝ≥0∞)⁻¹ :=
    ENNReal.inv_le_inv.2 (by exact_mod_cast Fintype.card_subtype_le p)
  unfold avg
  calc ∑ c : C, (Fintype.card C : ℝ≥0∞)⁻¹ * (if p c then f c else 0)
      ≤ ∑ c ∈ Finset.univ.filter p, (Fintype.card {c // p c} : ℝ≥0∞)⁻¹ * f c := by
        rw [Finset.sum_filter]
        refine Finset.sum_le_sum fun c _ => ?_
        split_ifs
        · exact mul_le_mul_of_nonneg_right hc zero_le
        · exact (mul_zero _).le
    _ = ∑ c : {c // p c}, (Fintype.card {c // p c} : ℝ≥0∞)⁻¹ * f c.1 :=
        Finset.sum_subtype _ (fun c => by simp) _

end FlockSoundness.Game

namespace Flock.Guarantees.JointAuditV5

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open scoped ENNReal

open Classical

variable (w : JointAuditV5)

/-- **Member `k`'s session in the window's audit is at most its session played alone**: an event `E` of member `k`'s
part of the sample and its session's transcript is no likelier in the window's audit, which records no transcript when
its draw runs out, than in member `k`'s session played alone on the coins on which the draw ends. -/
theorem prob_member (τ : Strategy w.game) (k : Fin w.ms.length)
    (E : Finset (Fin (w.ms.get k).n) → Discharge.Exec.Transcript → Prop) :
    prob (fun o => (o.2.2[k.1]?).elim False (E ((w.ms.get k).part o.2.1))) w.game τ ≤
      prob (fun o => E o.2.1 o.2.2) ((w.ms.get k).game (w.memberLaw τ k)) (w.memberStrat τ k) := by
  let h : (ω : w.L.Ω) → w.before τ k ω → ℝ≥0∞ := fun ω t =>
    prob (E ((w.ms.get k).part (w.L.draw ω))) ((w.ms.get k).live ((w.ms.get k).part (w.L.draw ω)) τ.1)
      (seqStrat (fun m : MemberV5 w.Reg w.nJ => m.live (m.part (w.L.draw ω)) τ.1) w.ms k.1 k.2
        (Strategy.ofMap (w.out τ.1 ω) _ (τ.2 ω)) t)
  have hL : prob (fun o => (o.2.2[k.1]?).elim False (E ((w.ms.get k).part o.2.1))) w.game τ =
      avg fun ω : w.L.Ω => if DrawEnds w.src w.law w.nJ ω then avg (h ω) else 0 := by
    show avg (fun ω => prob _ ((w.sessions (w.L.draw ω) τ.1).map (w.out τ.1 ω)) (τ.2 ω)) = _
    refine congrArg avg (funext fun ω => ?_)
    rw [prob_map]
    by_cases hω : DrawEnds w.src w.law w.nJ ω
    · rw [if_pos hω]
      have hp : ∀ ts : List Discharge.Exec.Transcript,
          ((w.out τ.1 ω ts).2.2[k.1]?).elim False (E ((w.ms.get k).part (w.out τ.1 ω ts).2.1)) ↔
            (ts[k.1]?).elim False (E ((w.ms.get k).part (w.L.draw ω))) := fun ts => by
        simp only [JointAuditV5.out, if_pos hω]
      refine (le_antisymm (prob_mono (fun ts hts => (hp ts).1 hts) _ _)
        (prob_mono (fun ts hts => (hp ts).2 hts) _ _)).trans ?_
      exact (prob_seqG (fun m : MemberV5 w.Reg w.nJ => m.live (m.part (w.L.draw ω)) τ.1)
        (fun x => x.elim False (E ((w.ms.get k).part (w.L.draw ω)))) w.ms k.1 k.2 _).trans
        (avg_congr_fintype _ _ _)
    · rw [if_neg hω]
      refine le_antisymm ((prob_mono (q := fun _ => False) (fun ts hts => ?_) _ _).trans_eq (prob_false _ _))
        zero_le
      simp [JointAuditV5.out, hω] at hts
  have hR : prob (fun o => E o.2.1 o.2.2) ((w.ms.get k).game (w.memberLaw τ k)) (w.memberStrat τ k) =
      avg fun x : (w.memberLaw τ k).Ω => h x.1.1 (x.2 x.1) := by
    show avg (fun x => prob (fun o => E o.2.1 o.2.2) (((w.ms.get k).live ((w.memberLaw τ k).draw x) τ.1).map
      fun t => (τ.1, (w.memberLaw τ k).draw x, t))
        (Strategy.map (fun t => (τ.1, (w.memberLaw τ k).draw x, t)) _
          (seqStrat (fun m : MemberV5 w.Reg w.nJ => m.live (m.part (w.L.draw x.1.1)) τ.1) w.ms k.1 k.2
            (Strategy.ofMap (w.out τ.1 x.1.1) _ (τ.2 x.1.1)) (x.2 x.1)))) = _
    refine congrArg avg (funext fun x => ?_)
    rw [prob_map]
    exact congrArg (prob _ _) (Strategy.ofMap_map _ _ _)
  rw [hL, hR]
  exact (avg_ite_le _ _).trans_eq
    ((avg_prod_pi fun ω : w.EΩ => h ω.1).symm.trans (avg_congr_fintype _ _ _))

/-- An average over member `k`'s sampling design played alone of a function of its part of the sample is the average
over `design k`. -/
theorem avg_memberLaw (τ : Strategy w.game) (k : Fin w.ms.length) (G : Finset (Fin (w.ms.get k).n) → ℝ≥0∞) :
    avg (fun x : (w.memberLaw τ k).Ω => G ((w.memberLaw τ k).draw x)) =
      avg (fun ω : (w.design k).Ω => G ((w.design k).draw ω)) :=
  (avg_congr_fintype _ _ _).trans
    (avg_fst (B := (ω : w.EΩ) → w.before τ k ω.1) fun ω : w.EΩ => G ((w.ms.get k).part (w.L.draw ω.1)))

/-- **`ε_ks` reads only the member's part of the sample**: it is the same at `memberLaw` as at `design k`. -/
theorem epsKs_memberLaw (τ : Strategy w.game) (k : Fin w.ms.length) :
    (w.ms.get k).epsKs (w.memberLaw τ k) = (w.ms.get k).epsKs (w.design k) :=
  iSup_congr fun R => w.avg_memberLaw τ k fun S =>
    ⨆ j : Fin (planZHJP (w.ms.get k).dj (w.ms.get k).y₀ S R).length,
      ENNReal.ofReal (ksStrictZ (planZHJP (w.ms.get k).dj (w.ms.get k).y₀ S R) j (w.ms.get k).k (w.ms.get k).qF
        (w.ms.get k).qS / (1 - rateZ (planZHJP (w.ms.get k).dj (w.ms.get k).y₀ S R) j (w.ms.get k).k))

/-- The share of member `k`'s sampling design played alone whose draws read commit string `p` is `design k`'s. -/
theorem readShare_memberLaw (τ : Strategy w.game) (k : Fin w.ms.length) (p : (w.ms.get k).vb.Pos) :
    (Fintype.card (ΩcZ execArith (w.ms.get k).P (planZHJP (w.ms.get k).dj (w.ms.get k).y₀)
        (tabZHJP (w.ms.get k).dj (w.ms.get k).y₀) (w.ms.get k).vb (w.memberLaw τ k) p) : ℝ) /
        Fintype.card (w.memberLaw τ k).Ω =
      (Fintype.card (ΩcZ execArith (w.ms.get k).P (planZHJP (w.ms.get k).dj (w.ms.get k).y₀)
        (tabZHJP (w.ms.get k).dj (w.ms.get k).y₀) (w.ms.get k).vb (w.design k) p) : ℝ) /
        Fintype.card (w.design k).Ω := by
  have e : ΩcZ execArith (w.ms.get k).P (planZHJP (w.ms.get k).dj (w.ms.get k).y₀)
        (tabZHJP (w.ms.get k).dj (w.ms.get k).y₀) (w.ms.get k).vb (w.memberLaw τ k) p ≃
      ΩcZ execArith (w.ms.get k).P (planZHJP (w.ms.get k).dj (w.ms.get k).y₀)
        (tabZHJP (w.ms.get k).dj (w.ms.get k).y₀) (w.ms.get k).vb (w.design k) p × ((ω : w.EΩ) → w.before τ k ω.1) :=
    Equiv.prodSubtypeFstEquivSubtypeProd (p := fun ω : w.EΩ => ReadsDrawZ (L := w.design k) execArith
      (w.ms.get k).P (planZHJP (w.ms.get k).dj (w.ms.get k).y₀) (tabZHJP (w.ms.get k).dj (w.ms.get k).y₀)
      (w.ms.get k).vb p ω)
  have hΩ : Fintype.card (w.memberLaw τ k).Ω =
      Fintype.card (w.design k).Ω * Fintype.card ((ω : w.EΩ) → w.before τ k ω.1) :=
    Fintype.card_prod _ _
  rw [Fintype.card_congr e, Fintype.card_prod, hΩ, Nat.cast_mul, Nat.cast_mul,
    mul_div_mul_right _ _ (Nat.cast_ne_zero.2 (Fintype.card_ne_zero (α := (ω : w.EΩ) → w.before τ k ω.1)))]

/-- **`ε_bind` reads only the member's part of the sample**: it is the same at `memberLaw` as at `design k`. -/
theorem link_memberLaw (τ : Strategy w.game) (k : Fin w.ms.length) :
    (w.ms.get k).link (w.memberLaw τ k) = (w.ms.get k).link (w.design k) := by
  simp only [MemberV5.link, w.readShare_memberLaw τ k]

/-- **Member `k`'s session in the window's audit accepts with a sampled proof block of its part incorrect** with
probability at most `ε_ks,k + ε_bind,k` at `design k`, under `cr/sha-512` and `ecr/sha-512` at its session played
alone. -/
theorem member_accepted_drawnWrong_le (τ : Strategy w.game) (k : Fin w.ms.length)
    (hCR : ∀ ω j, TableCRZ execArith hs512
      (planZHJP (w.ms.get k).dj (w.ms.get k).y₀ ((w.memberLaw τ k).draw ω) (reg (w.memberSim τ k))) j
      (zcStrat execArith hs512 (planZHJP (w.ms.get k).dj (w.ms.get k).y₀) (cont (w.memberSim τ k) ω))
      (w.ms.get k).qF (w.ms.get k).qS)
    (hA2 : LinkCRZC execArith hs512 (w.ms.get k).P (planZHJP (w.ms.get k).dj (w.ms.get k).y₀)
      (tabZHJP (w.ms.get k).dj (w.ms.get k).y₀) (w.ms.get k).vb (reg (w.memberSim τ k)) (cont (w.memberSim τ k))
      (w.ms.get k).k (w.ms.get k).Rw (w.ms.get k).M (w.ms.get k).t') :
    prob (fun o => (o.2.2[k.1]?).elim False (fun t => AcceptsZKP (w.ms.get k).I (w.ms.get k).own t ∧
        ∃ u ∈ (w.ms.get k).part o.2.1, u ∈ w.memberWrong τ k)) w.game τ ≤
      (w.ms.get k).epsKs (w.design k) + (w.ms.get k).link (w.design k) := by
  refine (w.prob_member τ k fun S t => AcceptsZKP (w.ms.get k).I (w.ms.get k).own t ∧
    ∃ u ∈ S, u ∈ w.memberWrong τ k).trans ?_
  rw [← w.epsKs_memberLaw τ k, ← w.link_memberLaw τ k]
  exact (w.ms.get k).accepted_drawnWrong_le (w.memberLaw τ k) (w.oneClass_memberLaw τ k) (w.memberStrat τ k) hCR hA2

/-- **`hcls` from the registration's refusal**: if at every draw that ends member `k`'s part holds between `lo k` and
`hi k` of its proof blocks, its record's count `kd / nTab I` lies between them too, and `lo k` and `hi k` give the same
link points (the registration check refuses a member whose fewest and most proof blocks over the draws that end don't),
then every draw that ends has a part with the record's link points. -/
theorem hcls_of_bounds {Reg : Type} {nJ : ℕ} (ms : List (MemberV5 Reg nJ)) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    {Lb : Fin 1 → ℕ} (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (law : Flock.Draw.Law)
    (lo hi : Fin ms.length → ℕ)
    (hdraw : ∀ k ω, DrawEnds src law nJ ω →
      lo k ≤ ((ms.get k).part ((Bridge.execSel (n := nJ) src law nJ).draw ω)).card ∧
        ((ms.get k).part ((Bridge.execSel (n := nJ) src law nJ).draw ω)).card ≤ hi k)
    (hrec : ∀ k, lo k ≤ (ms.get k).kd / nTab (ms.get k).I ∧ (ms.get k).kd / nTab (ms.get k).I ≤ hi k)
    (hends : ∀ k, mPtsOf (ms.get k).D.unroll.g (lo k) = mPtsOf (ms.get k).D.unroll.g (hi k)) :
    ∀ (k : Fin ms.length) (ω : Ω), DrawEnds src law nJ ω →
      mPtsOf (ms.get k).D.unroll.g ((ms.get k).part ((Bridge.execSel (n := nJ) src law nJ).draw ω)).card =
        mPtsOf (ms.get k).D.unroll.g ((ms.get k).kd / nTab (ms.get k).I) := by
  intro k ω hω
  have h1 := mPtsOf_mono (ms.get k).D.unroll.g (hdraw k ω hω).1
  have h2 := mPtsOf_mono (ms.get k).D.unroll.g (hdraw k ω hω).2
  have h3 := mPtsOf_mono (ms.get k).D.unroll.g (hrec k).1
  have h4 := mPtsOf_mono (ms.get k).D.unroll.g (hrec k).2
  have h5 := hends k
  omega

end Flock.Guarantees.JointAuditV5
