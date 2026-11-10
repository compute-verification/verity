module

public import Proofs.Flock.SampledGeneric.Joint

/-!
# The textbook form's transfer, and its theorem

The binding error of an extractor that is a game (`JointExtractor.toExtractor`) transfers to its trials
(`JointExtractor.toTrials_transfers`): an accepted session whose witness for the picked proof block opens a position `c`
to a value other than the trace's makes the extractor's game read a value other than the trace's at `c`
(`wrongAt_eq_one`), so the binding error at the picked proof block is at most the sum over the positions the sample reads
(`atTgt_link_le`); averaged over the sample and exchanged with the sum over positions, that is `Σ_c wt_c · errOf_c`
(`sum_wt_errOf`), with no slack. `Flock.SecurityProofs.GenericSampledKSJoint` is then `GenericSampledKSPa` at the
extractor and its trials, with the knowledge term at most `κ`.
-/

@[expose] public section

namespace FlockSoundness.SampledGeneric

open Game Audit LinkFinder
open scoped ENNReal

variable {n : ℕ} {Reg : Type} {Dh : Type} {Hc : List UInt8 → Dh} {B : Binding Reg Hc} {PS : ProofSystem n Reg}

private theorem avg_ofReal_eq {C : Type} [Fintype C] [Nonempty C] (f : C → ℝ) (hf : ∀ c, 0 ≤ f c) :
    avg (fun c => ENNReal.ofReal (f c)) = ENNReal.ofReal ((∑ c, f c) / Fintype.card C) := by
  have hc : (0 : ℝ) < Fintype.card C := by exact_mod_cast Fintype.card_pos
  unfold avg
  rw [div_eq_mul_inv, Finset.sum_mul, ENNReal.ofReal_sum_of_nonneg fun c _ =>
    mul_nonneg (hf c) (inv_nonneg.2 hc.le)]
  refine Finset.sum_congr rfl fun c _ => ?_
  rw [ENNReal.ofReal_mul (hf c), ENNReal.ofReal_inv_of_pos hc, ENNReal.ofReal_natCast, mul_comm]

namespace JointExtractor

variable (J : JointExtractor PS B) (L : Law n)

/-- **What the extractor's game reads wrong at `c`** after the sample, against `x`: `1` if it reads a value there and
not exactly `x`, else `0` (the integrand of `LinkFinder.errAt`, `errAt_eq_expect`). -/
noncomputable def wrongAt (R : Reg) (c : B.Pos) (x : B.Val) (e : L.Ω × (Bool × (Fin n → Option J.Wit))) : ℝ :=
  ind (fun e => (J.readsOut L R c e).Nonempty) e - ind (fun e => J.readsOut L R c e = {x}) e

theorem wrongAt_nonneg (R : Reg) (c : B.Pos) (x : B.Val) (e : L.Ω × (Bool × (Fin n → Option J.Wit))) :
    0 ≤ J.wrongAt L R c x e := by
  simp only [wrongAt, ind]
  split_ifs with h1 h2 h2
  · norm_num
  · norm_num
  · exact (h1 (h2 ▸ Finset.singleton_nonempty x)).elim
  · norm_num

theorem wrongAt_eq_one {R : Reg} {c : B.Pos} {x : B.Val} {e : L.Ω × (Bool × (Fin n → Option J.Wit))} {v : B.Val}
    (hv : v ∈ J.readsOut L R c e) (hx : v ≠ x) : J.wrongAt L R c x e = 1 := by
  have h1 : (J.readsOut L R c e).Nonempty := ⟨v, hv⟩
  have h2 : J.readsOut L R c e ≠ {x} := fun h => hx (by rw [h] at hv; exact Finset.mem_singleton.1 hv)
  simp only [wrongAt, ind, h1, h2, ↓reduceIte, sub_zero]

/-- What the extractor's game reads wrong at `c` after the sample `ω`, in expectation. -/
noncomputable def wrongExp (R : Reg) (τ : Cont L Reg PS.session R) (X : B.Trace) (ω : L.Ω) (c : B.Pos) : ℝ :=
  expect (fun o => J.wrongAt L R c (X c) (ω, o)) (J.G (L.draw ω) R) (J.strat (L.draw ω) R (τ ω))

theorem wrongExp_nonneg (R : Reg) (τ : Cont L Reg PS.session R) (X : B.Trace) (ω : L.Ω) (c : B.Pos) :
    0 ≤ J.wrongExp L R τ X ω c :=
  expect_nonneg (fun o => J.wrongAt_nonneg L R c (X c) (ω, o)) _ _

/-- **The binding error at a sampled proof block** is at most what the extractor's game reads wrong, in expectation, at
the positions it reads. -/
theorem link_toReal_le (R : Reg) (τ : Cont L Reg PS.session R) (X : B.Trace) (ω : L.Ω) (u : Fin n)
    (hu : u ∈ L.draw ω) :
    (J.toExtractor.link (L.draw ω) R (τ ω) u X).toReal ≤ ∑ c ∈ J.reads u, J.wrongExp L R τ X ω c := by
  show (prob _ _ _).toReal ≤ _
  rw [prob_toReal]
  unfold wrongExp
  rw [← expect_sum (J.reads u) (fun c o => J.wrongAt L R c (X c) (ω, o))]
  refine expect_mono (fun o => ?_) _ _
  simp only [ind]
  split_ifs with h
  · obtain ⟨h1, w, hw, hR, c, hc, hne⟩ := h
    have hv : B.val (J.opens (L.draw ω) R u w c) ∈ J.readsOut L R c (ω, o) :=
      (J.mem_readsOut (L := L)).2 ⟨h1, u, hu, hc, w, hw, hR, rfl⟩
    calc (1 : ℝ) = J.wrongAt L R c (X c) (ω, o) := (J.wrongAt_eq_one L hv hne).symm
      _ ≤ ∑ c ∈ J.reads u, J.wrongAt L R c (X c) (ω, o) :=
        Finset.single_le_sum (f := fun c => J.wrongAt L R c (X c) (ω, o))
          (fun c _ => J.wrongAt_nonneg L R c (X c) (ω, o)) hc
  · exact Finset.sum_nonneg fun c _ => J.wrongAt_nonneg L R c (X c) (ω, o)

/-- **The binding error at the picked proof block**, at the sample `ω`, is at most what the extractor's game reads wrong at
every position the sample reads. -/
theorem atTgt_link_le (R : Reg) (τ : Cont L Reg PS.session R) (X : B.Trace) (ω : L.Ω) (t : Option (Fin n)) :
    atTgt (L.draw ω) t (fun u => J.toExtractor.link (L.draw ω) R (τ ω) u X) ≤
      ENNReal.ofReal (∑ c ∈ Finset.univ.filter (fun c => ∃ u ∈ L.draw ω, c ∈ J.reads u), J.wrongExp L R τ X ω c) := by
  refine atTgt_le (fun u hu => ?_) t
  have hne : J.toExtractor.link (L.draw ω) R (τ ω) u X ≠ ⊤ :=
    ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ _ _)
  have hsub : ∑ c ∈ J.reads u, J.wrongExp L R τ X ω c ≤
      ∑ c ∈ Finset.univ.filter (fun c => ∃ u ∈ L.draw ω, c ∈ J.reads u), J.wrongExp L R τ X ω c :=
    Finset.sum_le_sum_of_subset_of_nonneg (fun c hc => Finset.mem_filter.2 ⟨Finset.mem_univ c, u, hu, hc⟩)
      fun c _ _ => J.wrongExp_nonneg L R τ X ω c
  calc J.toExtractor.link (L.draw ω) R (τ ω) u X
      = ENNReal.ofReal (J.toExtractor.link (L.draw ω) R (τ ω) u X).toReal := (ENNReal.ofReal_toReal hne).symm
    _ ≤ _ := ENNReal.ofReal_le_ofReal ((J.link_toReal_le L R τ X ω u hu).trans hsub)

/-- **The trials' binding error at a position some sample reads**: what the extractor's game reads wrong there, averaged
over the samples that read it (`LinkFinder.errAt_eq_expect`). -/
theorem errOf_toTrials (R : Reg) (τ : Cont L Reg PS.session R) (X : B.Trace) (c : B.Pos)
    (h : Nonempty (J.ReadsAt L c)) :
    (J.toTrials L).errOf R τ c (X c) =
      (∑ w : J.ReadsAt L c, J.wrongExp L R τ X w.1 c) / Fintype.card (J.ReadsAt L c) := by
  unfold Trials.errOf
  split_ifs with h'
  · rw [errAt_eq_expect]
    refine (expect_bind _ (Game.draw (J.ReadsAt L c)) (fun w => (J.G (L.draw w.1) R).map fun o => (w.1, o))
      (drawStrat _) (fun w => Strategy.map _ _ (J.strat (L.draw w.1) R (τ w.1)))).trans
      ((Verity.Game.expect_draw _ _).trans (congrArg (· / (Fintype.card (J.ReadsAt L c) : ℝ))
        (Finset.sum_congr rfl fun w _ => ?_)))
    exact expect_map _ _ _ _
  · exact absurd h h'

/-- **The trials' weighted binding error** is what the extractor's game reads wrong at the positions each sample reads,
averaged over the sample. -/
theorem sum_wt_errOf (R : Reg) (τ : Cont L Reg PS.session R) (X : B.Trace) :
    ∑ c, (J.toTrials L).wt c * (J.toTrials L).errOf R τ c (X c) =
      (∑ ω, ∑ c ∈ Finset.univ.filter (fun c => ∃ u ∈ L.draw ω, c ∈ J.reads u), J.wrongExp L R τ X ω c) /
        Fintype.card L.Ω := by
  classical
  have hΩ : (Fintype.card L.Ω : ℝ) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  have hc : ∀ c, (J.toTrials L).wt c * (J.toTrials L).errOf R τ c (X c) =
      (∑ ω ∈ Finset.univ.filter (fun ω => ∃ u ∈ L.draw ω, c ∈ J.reads u), J.wrongExp L R τ X ω c) /
        Fintype.card L.Ω := by
    intro c
    rw [Finset.sum_subtype (F := J.instFintypeReadsAt L c)
      (Finset.univ.filter (fun ω => ∃ u ∈ L.draw ω, c ∈ J.reads u)) (fun ω => by simp)]
    show (Fintype.card (J.ReadsAt L c) : ℝ) / Fintype.card L.Ω * _ = _
    by_cases h : Nonempty (J.ReadsAt L c)
    · rw [J.errOf_toTrials L R τ X c h]
      have hR : (Fintype.card (J.ReadsAt L c) : ℝ) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
      rw [div_mul_div_comm, mul_comm (Fintype.card (J.ReadsAt L c) : ℝ), mul_div_mul_right _ _ hR]
    · have : IsEmpty (J.ReadsAt L c) := not_nonempty_iff.1 h
      simp [Fintype.card_eq_zero]
  rw [Finset.sum_congr rfl fun c _ => hc c, ← Finset.sum_div]
  congr 1
  simp only [Finset.sum_filter]
  exact Finset.sum_comm

theorem toTrials_transfers : (J.toTrials L).Transfers J.toExtractor := by
  intro R τ tgt X
  refine (avg_mono fun ω => J.atTgt_link_le L R τ X ω (tgt ω)).trans (le_of_eq ?_)
  rw [avg_ofReal_eq _ fun ω => Finset.sum_nonneg fun c _ => J.wrongExp_nonneg L R τ X ω c]
  show _ = ENNReal.ofReal ((∑ c, (J.toTrials L).wt c * ((J.toTrials L).errOf R τ c (X c) + 0)) / (1 - 0))
  rw [sub_zero, div_one]
  simp only [add_zero]
  rw [J.sum_wt_errOf L R τ X]

end JointExtractor

end FlockSoundness.SampledGeneric

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.SampledGeneric in
/-- **The textbook form** (`Flock.Guarantees.GenericSampledKSJoint`): `GenericSampledKSPa` at the extractor that is a
game and its trials, whose knowledge term is at most `κ` at every sampled proof block. -/
theorem Flock.SecurityProofs.GenericSampledKSJoint : Flock.Guarantees.GenericSampledKSJoint := by
  intro n Reg Dh Hc L PS B Correct J κ M hJ hS hc hks hM σ hCR
  refine (Flock.SecurityProofs.GenericSampledKSPa n Reg Dh Hc L PS B Correct J.toExtractor (J.toTrials L)
    (J.toTrials L).plur M (J.toExtractor_covers hJ hS) (J.toTrials_opens hS L) (J.toTrials_wellFormed hc L)
    (J.toTrials_transfers L) (J.toTrials L).plurality_plur hM σ hCR).trans ?_
  gcongr
  exact avg_mono fun ω => iSup₂_mono fun u hu => hks _ _ _ u hu
