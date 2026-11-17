module

public import Proofs.Flock.EndToEnd.Joint.V5.Layout
public import Proofs.Properties.Verity.Sampled

/-! A window's audit with its sample's coins widened. -/

@[expose] public section

namespace Flock.Guarantees.JointAuditV5

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open scoped ENNReal

open Classical

variable (w : JointAuditV5)

noncomputable def extend (S : Verity.Sampled.Widening) : JointAuditV5 where
  __ := w
  Ω := w.Ω × S.T
  fintypeΩ := inferInstance
  nonemptyΩ := inferInstance
  src ω := w.src ω.1
  hcls k ω hω := by
    have h := w.hcls k ω.1 hω
    dsimp only [Bridge.execSel] at h ⊢
    exact h

/-- By `dsimp`, not `rfl`: elaborating the two samples as definitionally equal reduces their `match` and
evaluates `sampleUnits`. -/
theorem extend_sample (S : Verity.Sampled.Widening) (ω : (w.extend S).L.Ω) :
    (w.extend S).L.draw ω = w.L.draw ω.1 := by
  dsimp only [JointAuditV5.L, Bridge.execSel, JointAuditV5.extend]

theorem extend_design_sample (S : Verity.Sampled.Widening) (k : Fin w.ms.length) (ω : (w.extend S).L.Ω) :
    ((w.extend S).design k).draw ω = (w.design k).draw ω.1 :=
  congrArg (w.ms.get k).part (w.extend_sample S ω)

theorem uniform_extend (hA3 : Assumptions.UniformRandomBytes w.Lb w.src) (S : Verity.Sampled.Widening) :
    Assumptions.UniformRandomBytes (w.extend S).Lb (w.extend S).src :=
  Verity.Sampled.uniform_fst hA3

variable {w} in
theorem Laid.extend (h : w.Laid) (S : Verity.Sampled.Widening) : (w.extend S).Laid :=
  ⟨h.total, h.emb⟩

theorem avg_extend_design (S : Verity.Sampled.Widening) (k : Fin w.ms.length)
    (G : Finset (Fin (w.ms.get k).n) → ℝ≥0∞) :
    avg (fun x : ((w.extend S).design k).Ω => G (((w.extend S).design k).draw x)) =
      avg (fun ω : (w.design k).Ω => G ((w.design k).draw ω)) :=
  (congrArg avg (funext fun x => congrArg G (w.extend_design_sample S k x))).trans
    ((avg_congr_fintype _ _ _).trans (avg_fst (B := S.T) fun ω => G ((w.design k).draw ω)))

theorem epsKs_extend (S : Verity.Sampled.Widening) (k : Fin w.ms.length) :
    (w.ms.get k).epsKs ((w.extend S).design k) = (w.ms.get k).epsKs (w.design k) :=
  iSup_congr fun R => w.avg_extend_design S k fun T =>
    ⨆ j : Fin (planZHJP (w.ms.get k).dj (w.ms.get k).y₀ T R).length,
      ENNReal.ofReal (ksStrictZ (planZHJP (w.ms.get k).dj (w.ms.get k).y₀ T R) j (w.ms.get k).k (w.ms.get k).qF
        (w.ms.get k).qS / (1 - rateZ (planZHJP (w.ms.get k).dj (w.ms.get k).y₀ T R) j (w.ms.get k).k))

theorem readShare_extend (S : Verity.Sampled.Widening) (k : Fin w.ms.length) (p : (w.ms.get k).vb.Pos) :
    (Fintype.card (ΩcZ execArith (w.ms.get k).P (planZHJP (w.ms.get k).dj (w.ms.get k).y₀)
        (tabZHJP (w.ms.get k).dj (w.ms.get k).y₀) (w.ms.get k).vb ((w.extend S).design k) p) : ℝ) /
        Fintype.card ((w.extend S).design k).Ω =
      (Fintype.card (ΩcZ execArith (w.ms.get k).P (planZHJP (w.ms.get k).dj (w.ms.get k).y₀)
        (tabZHJP (w.ms.get k).dj (w.ms.get k).y₀) (w.ms.get k).vb (w.design k) p) : ℝ) /
        Fintype.card (w.design k).Ω := by
  have e : ΩcZ execArith (w.ms.get k).P (planZHJP (w.ms.get k).dj (w.ms.get k).y₀)
        (tabZHJP (w.ms.get k).dj (w.ms.get k).y₀) (w.ms.get k).vb ((w.extend S).design k) p ≃
      ΩcZ execArith (w.ms.get k).P (planZHJP (w.ms.get k).dj (w.ms.get k).y₀)
        (tabZHJP (w.ms.get k).dj (w.ms.get k).y₀) (w.ms.get k).vb (w.design k) p × S.T :=
    (Equiv.subtypeEquivRight fun x : w.Ω × S.T => by
      show ReadsSampleZ (L := (w.extend S).design k) execArith (w.ms.get k).P
          (planZHJP (w.ms.get k).dj (w.ms.get k).y₀) (tabZHJP (w.ms.get k).dj (w.ms.get k).y₀) (w.ms.get k).vb p
          x ↔
        ReadsSampleZ (L := w.design k) execArith (w.ms.get k).P
          (planZHJP (w.ms.get k).dj (w.ms.get k).y₀) (tabZHJP (w.ms.get k).dj (w.ms.get k).y₀) (w.ms.get k).vb p
          x.1
      unfold ReadsSampleZ
      rw [w.extend_design_sample S k x]).trans
      (Equiv.prodSubtypeFstEquivSubtypeProd (p := fun ω : w.L.Ω => ReadsSampleZ (L := w.design k) execArith
        (w.ms.get k).P (planZHJP (w.ms.get k).dj (w.ms.get k).y₀) (tabZHJP (w.ms.get k).dj (w.ms.get k).y₀)
        (w.ms.get k).vb p ω))
  have hΩ : Fintype.card ((w.extend S).design k).Ω = Fintype.card (w.design k).Ω * Fintype.card S.T :=
    Fintype.card_prod _ _
  rw [Fintype.card_congr e, Fintype.card_prod, hΩ, Nat.cast_mul, Nat.cast_mul,
    mul_div_mul_right _ _ (Nat.cast_ne_zero.2 (Fintype.card_ne_zero (α := S.T)))]

theorem link_extend (S : Verity.Sampled.Widening) (k : Fin w.ms.length) :
    (w.ms.get k).link ((w.extend S).design k) = (w.ms.get k).link (w.design k) := by
  simp only [MemberV5.link, w.readShare_extend S k]

theorem eps_extend (S : Verity.Sampled.Widening) : (w.extend S).eps = w.eps := by
  show (∑ k : Fin w.ms.length, ((w.ms.get k).epsKs ((w.extend S).design k) +
      (w.ms.get k).link ((w.extend S).design k))) =
    ∑ k : Fin w.ms.length, ((w.ms.get k).epsKs (w.design k) + (w.ms.get k).link (w.design k))
  exact Finset.sum_congr rfl fun k _ => congrArg₂ (· + ·) (w.epsKs_extend S k) (w.link_extend S k)

theorem pa_extend (S : Verity.Sampled.Widening) (B : Finset (Fin w.nJ)) : (w.extend S).pa B = w.pa B :=
  rfl

theorem sampled_extend (hA3 : Assumptions.UniformRandomBytes w.Lb w.src) (S : Verity.Sampled.Widening)
    [Nonempty (w.extend S).EndΩ] (τ : Strategy (w.extend S).game)
    (hCR : ∀ k ω j, Assumptions.TableCRZAt execArith hs512
      (planZHJP ((w.extend S).ms.get k).dj ((w.extend S).ms.get k).y₀ (((w.extend S).memberSamplingDesign τ k).draw ω)
        (reg ((w.extend S).memberSim τ k))) j
      (zcStrat execArith hs512 (planZHJP ((w.extend S).ms.get k).dj ((w.extend S).ms.get k).y₀)
        (cont ((w.extend S).memberSim τ k) ω))
      ((w.extend S).ms.get k).qF ((w.extend S).ms.get k).qS)
    (hA2 : ∀ k, Assumptions.LinkCRZCAt execArith hs512 ((w.extend S).ms.get k).P
      (planZHJP ((w.extend S).ms.get k).dj ((w.extend S).ms.get k).y₀)
      (tabZHJP ((w.extend S).ms.get k).dj ((w.extend S).ms.get k).y₀) ((w.extend S).ms.get k).vb
      (reg ((w.extend S).memberSim τ k)) (cont ((w.extend S).memberSim τ k))
      ((w.extend S).ms.get k).k ((w.extend S).ms.get k).Rw ((w.extend S).ms.get k).M ((w.extend S).ms.get k).t')
    (W : Finset (Fin w.nJ)) :
    prob (fun o => (w.extend S).Accepts o ∧ W ⊆ (w.extend S).incorrectSet τ) (w.extend S).game τ ≤
      ENNReal.ofReal (w.pa W) + w.eps :=
  (Flock.SecurityProofs.JointSampledV5 (w.extend S) (w.uniform_extend hA3 S) τ (fun _ => hCR) (fun _ => hA2)
    W).trans_eq
    (congrArg (ENNReal.ofReal (w.pa W) + ·) (w.eps_extend S))

end Flock.Guarantees.JointAuditV5
