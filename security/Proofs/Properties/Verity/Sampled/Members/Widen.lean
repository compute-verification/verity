module

public import Proofs.Flock.EndToEnd.Joint.V5.Layout
public import Proofs.Properties.Verity.Sampled

/-!
# A window's audit with its sample's coins widened

A strategy of the period's game is read as a strategy of the window's audit (`Flock.Guarantees.JointAuditV5`) that
samples the period's other coins beside the sample's (`Verity.Sampled.Widening`), as one session's is
(`Verity.Sampled.Live.extend`). The sample reads only the window's bytes, so the widened audit is the registered one in
everything its bound reads:

* `JointAuditV5.extend`: the audit with its sample's coins widened by the coins `S`; its members, law, strata and
  layout are the registered audit's (`Laid.extend`);
* `extend_draw`, `extend_design_draw`: the widened audit samples, and each member's part is, what the registered audit
  does at the sample's own coins;
* `uniform_extend`: A3 at the window's bytes is A3 at the widened audit's (`Verity.Sampled.uniform_fst`);
* `avg_extend_design`, `epsKs_extend`, `link_extend`, `eps_extend`, `pa_extend`: a mean over a member's widened design
  of a function of its sample is the mean over its registered design, so each member's `ε_ks` and `ε_bind`, their sum
  and the sampling design's own miss are the registered audit's;
* `sampled_extend`: `JointSampledV5`'s bound at the widened audit, in the registered audit's terms.
-/

@[expose] public section

namespace Flock.Guarantees.JointAuditV5

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open scoped ENNReal

open Classical

variable (w : JointAuditV5)

/-- **The window's audit with its sample's coins widened by the coins `S`**: the sample reads only the window's bytes,
so every member's part of every sample still has its session's link points. -/
noncomputable def extend (S : Verity.Sampled.Widening) : JointAuditV5 where
  __ := w
  Ω := w.Ω × S.T
  fintypeΩ := inferInstance
  nonemptyΩ := inferInstance
  src ω := w.src ω.1
  hcls k ω := by
    have h := w.hcls k ω.1
    dsimp only [Bridge.execSel] at h ⊢
    exact h

/-- The widened audit samples what the registered audit does at the sample's own coins. By `dsimp`, not `rfl`:
elaborating the two samples as definitionally equal reduces their `match` and evaluates `drawUnits`. -/
theorem extend_draw (S : Verity.Sampled.Widening) (ω : (w.extend S).L.Ω) :
    (w.extend S).L.draw ω = w.L.draw ω.1 := by
  dsimp only [JointAuditV5.L, Bridge.execSel, JointAuditV5.extend]

/-- Member `k`'s part of the widened audit's sample is its part of the registered audit's at the sample's own coins. -/
theorem extend_design_draw (S : Verity.Sampled.Widening) (k : Fin w.ms.length) (ω : (w.extend S).L.Ω) :
    ((w.extend S).design k).draw ω = (w.design k).draw ω.1 :=
  congrArg (w.ms.get k).part (w.extend_draw S ω)

/-- **A3 at the widened audit's bytes** is A3 at the window's. -/
theorem uniform_extend (hA3 : Assumptions.UniformRandomBytes w.Lb w.src) (S : Verity.Sampled.Widening) :
    Assumptions.UniformRandomBytes (w.extend S).Lb (w.extend S).src :=
  Verity.Sampled.uniform_fst hA3

variable {w} in
/-- The widened audit's members are laid out as the registered audit's. -/
theorem Laid.extend (h : w.Laid) (S : Verity.Sampled.Widening) : (w.extend S).Laid :=
  ⟨h.total, h.emb⟩

/-- **A mean over member `k`'s widened design of a function of its sample is the mean over its registered design.** -/
theorem avg_extend_design (S : Verity.Sampled.Widening) (k : Fin w.ms.length)
    (G : Finset (Fin (w.ms.get k).n) → ℝ≥0∞) :
    avg (fun x : ((w.extend S).design k).Ω => G (((w.extend S).design k).draw x)) =
      avg (fun ω : (w.design k).Ω => G ((w.design k).draw ω)) :=
  (congrArg avg (funext fun x => congrArg G (w.extend_design_draw S k x))).trans
    ((avg_congr_fintype _ _ _).trans (avg_fst (B := S.T) fun ω => G ((w.design k).draw ω)))

/-- **Member `k`'s `ε_ks` at the widened audit is the registered audit's.** -/
theorem epsKs_extend (S : Verity.Sampled.Widening) (k : Fin w.ms.length) :
    (w.ms.get k).epsKs ((w.extend S).design k) = (w.ms.get k).epsKs (w.design k) :=
  iSup_congr fun R => w.avg_extend_design S k fun T =>
    ⨆ j : Fin (planZHJP (w.ms.get k).dj (w.ms.get k).y₀ T R).length,
      ENNReal.ofReal (ksStrictZ (planZHJP (w.ms.get k).dj (w.ms.get k).y₀ T R) j (w.ms.get k).k (w.ms.get k).qF
        (w.ms.get k).qS / (1 - rateZ (planZHJP (w.ms.get k).dj (w.ms.get k).y₀ T R) j (w.ms.get k).k))

/-- The share of member `k`'s widened design whose samples read commit string `p` is its registered design's: a
widened sample reads it exactly when the sample's own coins' sample does. -/
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
      show ReadsDrawZ (L := (w.extend S).design k) execArith (w.ms.get k).P
          (planZHJP (w.ms.get k).dj (w.ms.get k).y₀) (tabZHJP (w.ms.get k).dj (w.ms.get k).y₀) (w.ms.get k).vb p
          x ↔
        ReadsDrawZ (L := w.design k) execArith (w.ms.get k).P
          (planZHJP (w.ms.get k).dj (w.ms.get k).y₀) (tabZHJP (w.ms.get k).dj (w.ms.get k).y₀) (w.ms.get k).vb p
          x.1
      unfold ReadsDrawZ
      rw [w.extend_design_draw S k x]).trans
      (Equiv.prodSubtypeFstEquivSubtypeProd (p := fun ω : w.L.Ω => ReadsDrawZ (L := w.design k) execArith
        (w.ms.get k).P (planZHJP (w.ms.get k).dj (w.ms.get k).y₀) (tabZHJP (w.ms.get k).dj (w.ms.get k).y₀)
        (w.ms.get k).vb p ω))
  have hΩ : Fintype.card ((w.extend S).design k).Ω = Fintype.card (w.design k).Ω * Fintype.card S.T :=
    Fintype.card_prod _ _
  rw [Fintype.card_congr e, Fintype.card_prod, hΩ, Nat.cast_mul, Nat.cast_mul,
    mul_div_mul_right _ _ (Nat.cast_ne_zero.2 (Fintype.card_ne_zero (α := S.T)))]

/-- **Member `k`'s `ε_bind` at the widened audit is the registered audit's.** -/
theorem link_extend (S : Verity.Sampled.Widening) (k : Fin w.ms.length) :
    (w.ms.get k).link ((w.extend S).design k) = (w.ms.get k).link (w.design k) := by
  simp only [MemberV5.link, w.readShare_extend S k]

/-- **The members' knowledge and binding terms at the widened audit are the registered audit's.** -/
theorem eps_extend (S : Verity.Sampled.Widening) : (w.extend S).eps = w.eps := by
  show (∑ k : Fin w.ms.length, ((w.ms.get k).epsKs ((w.extend S).design k) +
      (w.ms.get k).link ((w.extend S).design k))) =
    ∑ k : Fin w.ms.length, ((w.ms.get k).epsKs (w.design k) + (w.ms.get k).link (w.design k))
  exact Finset.sum_congr rfl fun k _ => congrArg₂ (· + ·) (w.epsKs_extend S k) (w.link_extend S k)

/-- The sampling design's own miss at the widened audit is the registered audit's. -/
theorem pa_extend (S : Verity.Sampled.Widening) (B : Finset (Fin w.nJ)) : (w.extend S).pa B = w.pa B :=
  rfl

/-- **`JointSampledV5`'s bound at the widened audit, in the registered audit's terms**: against any strategy `τ` of
the audit widened by `S`, the probability that every member's session with a proof block in the sample is accepted and
every proof block of `W` is incorrect is at most the registered sampling design's own miss of `W` plus the registered
members' `ε_ks` and `ε_bind`, under A3 at the window's bytes and `cr/sha-512` and `ecr/sha-512` at each member's
session played alone at `τ`. -/
theorem sampled_extend (hA3 : Assumptions.UniformRandomBytes w.Lb w.src) (S : Verity.Sampled.Widening)
    (τ : Strategy (w.extend S).game)
    (hCR : ∀ k ω j, Assumptions.TableCRZAt execArith hs512
      (planZHJP ((w.extend S).ms.get k).dj ((w.extend S).ms.get k).y₀ (((w.extend S).memberLaw τ k).draw ω)
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
    prob (fun o => (w.extend S).Accepts o ∧ W ⊆ (w.extend S).wrongSet τ) (w.extend S).game τ ≤
      ENNReal.ofReal (w.pa W) + w.eps :=
  (Flock.SecurityProofs.JointSampledV5 (w.extend S) (w.uniform_extend hA3 S) τ hCR hA2 W).trans_eq
    (congrArg (ENNReal.ofReal (w.pa W) + ·) (w.eps_extend S))

end Flock.Guarantees.JointAuditV5
