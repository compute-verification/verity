module

public import Proofs.Flock.SampledGeneric.Hiding

/-!
# Sessions in turn are zero knowledge at the sum of their errors (proved)

A hybrid over the sessions, first session first. With the first session's honest view fixed, the sessions after it are
sessions in turn against the verifier that saw that view, so by induction their honest views are close to their
simulated ones at their own summed error; averaged over the first session's honest view, that is the tail of
`Zk.seqErr` (`Close.andThen_right`). Then the first session's honest view is replaced by its simulator's: the sessions
after it, simulated, are a post-processing of that view, and closeness survives post-processing (`Close.andThen`).

Closeness is stated on events, and a post-processing tests a view by a probability, a number in `[0,1]`. The step
between the two is the statistical-distance argument on the outcomes either experiment can end with:

* `outcomes`, `exG_eq_sum`: an experiment's expectation is a sum over a finite set that holds its outcomes;
* `sum_mul_le_of_gap`: two masses on a finite set that the event "the first exceeds the second" separates by at most
  `δ` give every `[0,1]`-valued test expectations within `δ`;
* `Close.ex_le`: so do two close experiments. `Close.andThen`: closeness survives post-processing;
* `Close.andThen_right`, `Close.map`, `Close.refl`: closeness under a shared first experiment, at the continuations'
  errors averaged over it; under a relabelling; of an experiment to itself;
* `Zk.seq_close`: the hybrid. `Zk.seqErr_le_sum`: its error is at most the sum of the sessions' bounds.
  `Flock.SecurityProofs.GenericSessionsZKAt`, `GenericSampledSessionsZK`, `GenericSampledSessionsZKSum`,
  `SpotFlockSessionsZK`.
-/

@[expose] public section

namespace FlockSoundness.SampledGeneric

open Game
open scoped ENNReal

/-! ## From events to tests -/

open Classical in
/-- **The outcomes `g` can end with against `s`**: a finite set. -/
noncomputable def outcomes {α : Type} : (g : Game α) → Strategy g → Finset α
  | .ret a, _ => {a}
  | .send _ k, s => outcomes (k s.1) s.2
  | @Game.coin _ C _ _ k, s => Finset.univ.biUnion fun c : C => outcomes (k c) (s c)

open Classical in
/-- **An expectation is a sum over the outcomes**: on any finite set that holds them, of each outcome's probability
times the test's value there. -/
theorem exG_eq_sum {α : Type} (F : α → ℝ≥0∞) : ∀ (g : Game α) (s : Strategy g) (S : Finset α), outcomes g s ⊆ S →
    Discharge.Recursion.exG F g s = ∑ v ∈ S, prob (· = v) g s * F v
  | .ret a, _, S, h => by
      have ha : a ∈ S := h (Finset.mem_singleton_self a)
      have h0 : ∀ v ∈ S, v ≠ a → @ite _ (a = v) (Classical.propDecidable _) (1 : ℝ≥0∞) 0 * F v = 0 :=
        fun v _ hv => by rw [ite_eq_right (Ne.symm hv), zero_mul]
      show F a = ∑ v ∈ S, @ite _ (a = v) (Classical.propDecidable _) (1 : ℝ≥0∞) 0 * F v
      rw [Finset.sum_eq_single_of_mem a ha h0, ite_eq_left rfl, one_mul]
  | .send _ k, s, S, h => exG_eq_sum F (k s.1) s.2 S h
  | @Game.coin _ C _ _ k, s, S, h => by
      have hc : ∀ c : C, outcomes (k c) (s c) ⊆ S := fun c =>
        (Finset.subset_biUnion_of_mem (fun c : C => outcomes (k c) (s c)) (Finset.mem_univ c)).trans h
      show avg (fun c : C => Discharge.Recursion.exG F (k c) (s c)) =
        ∑ v ∈ S, avg (fun c : C => prob (· = v) (k c) (s c)) * F v
      rw [show (fun c : C => Discharge.Recursion.exG F (k c) (s c)) =
          fun c : C => ∑ v ∈ S, prob (· = v) (k c) (s c) * F v from
        funext fun c => exG_eq_sum F (k c) (s c) S (hc c), Discharge.Recursion.avg_finset_sum]
      exact Finset.sum_congr rfl fun v _ => avg_mul_const _ _

/-- **From one event to every `[0,1]`-valued test, on a finite set**: if the event "`p` exceeds `q`" has mass under `p`
at most its mass under `q` plus `δ`, every test with values in `[0,1]` has expectation under `p` at most its
expectation under `q` plus `δ` (`Discharge.ZkCoins.avg_le_of_prCoin`'s argument, on any two masses). -/
theorem sum_mul_le_of_gap {V : Type} (S : Finset V) (p q : V → ℝ≥0∞) (hq : ∀ v, q v ≠ ∞) {δ : ℝ≥0∞}
    (hE : ∑ v ∈ S, p v * @ite _ (q v < p v) (Classical.propDecidable _) 1 0 ≤
      ∑ v ∈ S, q v * @ite _ (q v < p v) (Classical.propDecidable _) 1 0 + δ)
    (F : V → ℝ≥0∞) (hF : ∀ v, F v ≤ 1) :
    ∑ v ∈ S, p v * F v ≤ ∑ v ∈ S, q v * F v + δ := by
  have split : ∑ v ∈ S, p v * @ite _ (q v < p v) (Classical.propDecidable _) 1 0 =
      ∑ v ∈ S, @ite _ (q v < p v) (Classical.propDecidable _) (p v - q v) 0 +
        ∑ v ∈ S, q v * @ite _ (q v < p v) (Classical.propDecidable _) 1 0 := by
    rw [← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun v _ => ?_
    by_cases hp : q v < p v
    · rw [ite_eq_left hp, ite_eq_left hp, mul_one, mul_one]
      exact (tsub_add_cancel_of_le hp.le).symm
    · rw [ite_eq_right hp, ite_eq_right hp, mul_zero, mul_zero, add_zero]
  have hgap : ∑ v ∈ S, @ite _ (q v < p v) (Classical.propDecidable _) (p v - q v) 0 ≤ δ := by
    rw [split, add_comm] at hE
    refine (ENNReal.add_le_add_iff_left ?_).1 hE
    exact ENNReal.sum_ne_top.2 fun v _ => ENNReal.mul_ne_top (hq v) (by split_ifs <;> simp)
  have hpt : ∀ v ∈ S, p v * F v ≤
      q v * F v + @ite _ (q v < p v) (Classical.propDecidable _) (p v - q v) 0 := by
    intro v _
    by_cases hp : q v < p v
    · rw [ite_eq_left hp]
      calc p v * F v = (p v - q v) * F v + q v * F v := by rw [← add_mul, tsub_add_cancel_of_le hp.le]
        _ ≤ (p v - q v) * 1 + q v * F v := by gcongr; exact hF v
        _ = q v * F v + (p v - q v) := by rw [mul_one, add_comm]
    · rw [ite_eq_right hp, add_zero]
      exact mul_le_mul_left (not_lt.1 hp) _
  calc ∑ v ∈ S, p v * F v
      ≤ ∑ v ∈ S, (q v * F v + @ite _ (q v < p v) (Classical.propDecidable _) (p v - q v) 0) :=
        Finset.sum_le_sum hpt
    _ = ∑ v ∈ S, q v * F v + ∑ v ∈ S, @ite _ (q v < p v) (Classical.propDecidable _) (p v - q v) 0 :=
        Finset.sum_add_distrib
    _ ≤ ∑ v ∈ S, q v * F v + δ := by gcongr

/-- A post-processed experiment's probability: the expectation over the first experiment's view of the
post-processing's. -/
theorem Exp.andThen_pr {V W : Type} (e : Exp V) (f : V → Exp W) (E : W → Prop) :
    (e.andThen f).pr E = e.ex fun v => (f v).pr E :=
  Discharge.Recursion.prob_bind_strat E e.g (fun v => (f v).g) e.s fun v => (f v).s

/-- A relabelled experiment's probability. -/
theorem Exp.map_pr {V W : Type} (e : Exp V) (f : V → W) (E : W → Prop) : (e.map f).pr E = e.pr fun v => E (f v) :=
  prob_map_strat E f e.g e.s

namespace Close

variable {V W : Type}

/-- An experiment is `0`-close to itself. -/
theorem refl (a : Exp V) : Close a a 0 := fun _ => ⟨le_self_add, le_self_add⟩

/-- **From events to `[0,1]`-valued tests**: two `ε`-close experiments give every test with values in `[0,1]`
expectations within `ε`. The event that decides is "this view is likelier in the first experiment". -/
theorem ex_le {a b : Exp V} {ε : ℝ≥0∞} (h : Close a b ε) (T : V → ℝ≥0∞) (hT : ∀ v, T v ≤ 1) :
    a.ex T ≤ b.ex T + ε := by
  classical
  have hSa : outcomes a.g a.s ⊆ outcomes a.g a.s ∪ outcomes b.g b.s :=
    Finset.subset_iff.2 fun _ hv => Finset.mem_union_left _ hv
  have hSb : outcomes b.g b.s ⊆ outcomes a.g a.s ∪ outcomes b.g b.s :=
    Finset.subset_iff.2 fun _ hv => Finset.mem_union_right _ hv
  have key : ∀ (e : Exp V) (S : Finset V), outcomes e.g e.s ⊆ S → ∀ P : V → Prop,
      e.pr P = ∑ v ∈ S, prob (· = v) e.g e.s * @ite _ (P v) (Classical.propDecidable _) 1 0 := fun e S hS P =>
    (Discharge.Recursion.prob_eq_exG P e.g e.s).trans (exG_eq_sum _ e.g e.s S hS)
  have hE := (h fun v => prob (· = v) b.g b.s < prob (· = v) a.g a.s).1
  rw [key a _ hSa, key b _ hSb] at hE
  have hsum := sum_mul_le_of_gap (outcomes a.g a.s ∪ outcomes b.g b.s) (fun v => prob (· = v) a.g a.s)
    (fun v => prob (· = v) b.g b.s) (fun v => ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ _ _)) hE T hT
  show Discharge.Recursion.exG T a.g a.s ≤ Discharge.Recursion.exG T b.g b.s + ε
  rw [exG_eq_sum T a.g a.s _ hSa, exG_eq_sum T b.g b.s _ hSb]
  exact hsum

/-- **Closeness survives post-processing**: two `ε`-close experiments, each followed by the same experiment on its
view, are `ε`-close. -/
theorem andThen {a b : Exp V} {ε : ℝ≥0∞} (h : Close a b ε) (f : V → Exp W) :
    Close (a.andThen f) (b.andThen f) ε := by
  intro E
  rw [Exp.andThen_pr, Exp.andThen_pr]
  exact ⟨h.ex_le _ fun v => prob_le_one E (f v).g (f v).s, h.symm.ex_le _ fun v => prob_le_one E (f v).g (f v).s⟩

/-- **Closeness under a shared first experiment**: if at each of its views the two continuations are `δ v`-close, the
two composites are close at `δ` averaged over the first experiment. -/
theorem andThen_right (e : Exp V) {f f' : V → Exp W} {δ : V → ℝ≥0∞} (h : ∀ v, Close (f v) (f' v) (δ v)) :
    Close (e.andThen f) (e.andThen f') (e.ex δ) := by
  intro E
  rw [Exp.andThen_pr, Exp.andThen_pr]
  have h1 : Discharge.Recursion.exG (fun v => (f v).pr E) e.g e.s ≤
      Discharge.Recursion.exG (fun v => (f' v).pr E + δ v) e.g e.s :=
    Discharge.Recursion.exG_mono (fun v => (h v E).1) e.g e.s
  have h2 : Discharge.Recursion.exG (fun v => (f' v).pr E) e.g e.s ≤
      Discharge.Recursion.exG (fun v => (f v).pr E + δ v) e.g e.s :=
    Discharge.Recursion.exG_mono (fun v => (h v E).2) e.g e.s
  exact ⟨h1.trans_eq (Discharge.Recursion.exG_add (fun v => (f' v).pr E) δ e.g e.s),
    h2.trans_eq (Discharge.Recursion.exG_add (fun v => (f v).pr E) δ e.g e.s)⟩

/-- Closeness survives relabelling the view. -/
theorem map {a b : Exp V} {ε : ℝ≥0∞} (h : Close a b ε) (f : V → W) : Close (a.map f) (b.map f) ε := by
  intro E
  rw [Exp.map_pr, Exp.map_pr]
  exact h _

end Close

/-! ## The hybrid over the sessions -/

/-- **Sessions in turn are close to their simulators in turn, at the sessions' error** (`Zk.seqErr`): for statements
and witnesses in the relation pairwise, against every verifier that carries what it saw. The middle hybrid at each step
keeps the first session honest and simulates the sessions after it. -/
theorem Zk.seq_close {Stmt Wit VS View : Type} {Zs : Zk Stmt Wit VS View} {Rel : Stmt → Wit → Prop}
    {ε : Stmt → Wit → VS → ℝ≥0∞} (hZ : Zs.ZKAt Rel ε) {xs : List Stmt} {ws : List Wit}
    (hR : List.Forall₂ Rel xs ws) :
    ∀ v : List View → VS, Close (Zs.seqReal (xs.zip ws) v) (Zs.seqSim xs v) (Zs.seqErr ε (xs.zip ws) v) := by
  induction hR with
  | nil =>
    intro v
    show Close (⟨.ret [], PUnit.unit⟩ : Exp (List View)) ⟨.ret [], PUnit.unit⟩ 0
    exact Close.refl _
  | @cons x w xs ws hxw _ ih =>
    intro v
    show Close ((Zs.real x w (v [])).andThen fun a => (Zs.seqReal (xs.zip ws) fun h => v (a :: h)).map (a :: ·))
      ((Zs.sim x (v [])).andThen fun a => (Zs.seqSim xs fun h => v (a :: h)).map (a :: ·))
      (ε x w (v []) + (Zs.real x w (v [])).ex fun a => Zs.seqErr ε (xs.zip ws) fun h => v (a :: h))
    have hA : Close
        ((Zs.real x w (v [])).andThen fun a => (Zs.seqReal (xs.zip ws) fun h => v (a :: h)).map (a :: ·))
        ((Zs.real x w (v [])).andThen fun a => (Zs.seqSim xs fun h => v (a :: h)).map (a :: ·))
        ((Zs.real x w (v [])).ex fun a => Zs.seqErr ε (xs.zip ws) fun h => v (a :: h)) :=
      Close.andThen_right _ fun a => (ih fun h => v (a :: h)).map (a :: ·)
    have hB : Close
        ((Zs.real x w (v [])).andThen fun a => (Zs.seqSim xs fun h => v (a :: h)).map (a :: ·))
        ((Zs.sim x (v [])).andThen fun a => (Zs.seqSim xs fun h => v (a :: h)).map (a :: ·))
        (ε x w (v [])) :=
      (hZ x w hxw (v [])).andThen _
    exact (hA.trans hB).mono (le_of_eq (add_comm _ _))

/-- **The sessions' error is at most the sum of the sessions' bounds**: where a session's error is at most `ε_b x w`
against every verifier, the average over the honest views before it is too. -/
theorem Zk.seqErr_le_sum {Stmt Wit VS View : Type} (Zs : Zk Stmt Wit VS View) {ε : Stmt → Wit → VS → ℝ≥0∞}
    {εb : Stmt → Wit → ℝ≥0∞} (hε : ∀ x w v, ε x w v ≤ εb x w) :
    ∀ (l : List (Stmt × Wit)) (v : List View → VS), Zs.seqErr ε l v ≤ (l.map fun xw => εb xw.1 xw.2).sum
  | [], _ => le_of_eq rfl
  | xw :: l, v => by
      show ε xw.1 xw.2 (v []) + (Zs.real xw.1 xw.2 (v [])).ex (fun a => Zs.seqErr ε l fun hs => v (a :: hs)) ≤
        ((xw :: l).map fun xw => εb xw.1 xw.2).sum
      rw [List.map_cons, List.sum_cons]
      refine add_le_add (hε _ _ _) ?_
      have h1 : Discharge.Recursion.exG (fun a => Zs.seqErr ε l fun hs => v (a :: hs)) (Zs.real xw.1 xw.2 (v [])).g
          (Zs.real xw.1 xw.2 (v [])).s ≤
          Discharge.Recursion.exG (fun _ => (l.map fun xw => εb xw.1 xw.2).sum) (Zs.real xw.1 xw.2 (v [])).g
            (Zs.real xw.1 xw.2 (v [])).s :=
        Discharge.Recursion.exG_mono (fun a => Zk.seqErr_le_sum Zs hε l fun hs => v (a :: hs)) _ _
      exact h1.trans_eq (Discharge.Recursion.exG_const _ _ _)

end FlockSoundness.SampledGeneric

open FlockSoundness FlockSoundness.Game FlockSoundness.SampledGeneric in
/-- **Sessions in turn are zero knowledge at the sum of their errors** (`Flock.Guarantees.GenericSessionsZKAt`):
`Zk.seq_close`. -/
theorem Flock.SecurityProofs.GenericSessionsZKAt : Flock.Guarantees.GenericSessionsZKAt := by
  intro Stmt Wit VS View Zs Rel ε hZ xs ws hR v
  exact Zk.seq_close hZ hR (v xs)

open FlockSoundness FlockSoundness.Game FlockSoundness.SampledGeneric in
/-- **The sampled audit's zero knowledge with a session per sampled statement**
(`Flock.Guarantees.GenericSampledSessionsZK`): `GenericSampledZKAt` at the sessions in turn as one proof system
(`GenericSessionsZKAt`), on the sample's list of statements. -/
theorem Flock.SecurityProofs.GenericSampledSessionsZK : Flock.Guarantees.GenericSampledSessionsZK := by
  intro n Tr Rnd Com Stmt Wit VS View _ _ H Zs Rel ε εh stmts samp wits hZ hH Xt hX X₀ v
  exact Flock.SecurityProofs.GenericSampledZKAt n Tr Rnd Com (List Wit)
    (Com × Finset (Fin n) → List Stmt → List View → VS) (List View) H
    (Zs.seq.after stmts) (fun s ws => List.Forall₂ Rel (stmts s) ws)
    (fun s ws v => Zs.seqErr ε ((stmts s).zip ws) (v s (stmts s))) εh samp wits
    ((Flock.SecurityProofs.GenericSessionsZKAt Stmt Wit VS View Zs Rel ε hZ).after stmts) hH Xt hX X₀ v

open FlockSoundness FlockSoundness.Game FlockSoundness.SampledGeneric in
/-- **The sampled audit's zero knowledge with a session per sampled statement, at the sum of the sessions' errors**
(`Flock.Guarantees.GenericSampledSessionsZKSum`): `GenericSampledSessionsZK`, with `Zk.seqErr_le_sum` at each
commitment coin. -/
theorem Flock.SecurityProofs.GenericSampledSessionsZKSum : Flock.Guarantees.GenericSampledSessionsZKSum := by
  intro n Tr Rnd Com Stmt Wit VS View _ _ H Zs Rel εb εh stmts samp wits hZ hH Xt hX X₀ v
  exact (Flock.SecurityProofs.GenericSampledSessionsZK n Tr Rnd Com Stmt Wit VS View H Zs Rel (fun x w _ => εb x w) εh
    stmts samp wits hZ hH Xt hX X₀ v).mono
      (add_le_add le_rfl (avg_mono fun _ =>
        Zk.seqErr_le_sum Zs (ε := fun x w _ => εb x w) (εb := εb) (fun _ _ _ => le_rfl) _ _))

open FlockSoundness FlockSoundness.Game FlockSoundness.SampledGeneric in
/-- **The sampled audit over Flock's masked sessions, a session per sampled statement, is zero knowledge**
(`Flock.Guarantees.SpotFlockSessionsZK`): `GenericSampledSessionsZK` at the masked session (`flockSession_zkAt`) and
the executable's commit strings (`hm96Exe_hiding`). -/
theorem Flock.SecurityProofs.SpotFlockSessionsZK : Flock.Guarantees.SpotFlockSessionsZK := by
  intro nr t ht ι _ _ n Θ stmts samp wits Xt hX X₀ v
  exact Flock.SecurityProofs.GenericSampledSessionsZK n _ _ _ _ _ _ _ (hm96Exe ι) (flockSession nr t Θ)
    (Claims.Flock.R (flockZk nr t).parse) (fun x w v => flockErr nr t x w v) _ stmts samp wits
    (flockSession_zkAt nr t ht Θ) (hm96Exe_hiding ι) Xt hX X₀ v
