import Proofs.Flock.Soundness.Audit.TwoStage
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Tactic.FinCases

/-!
# Tightness and insecurity: NOT chains

`K` chains of `nv` NOT gates. Chain `c` is one coarse unit and each of its gates one fine unit, so a
prover can move a lie to any fine unit of a wrong chain. The second stage draws a uniform
`k₂`-subset of each drawn chain's gates. The session is ideal: it accepts exactly when every drawn
fine unit is correct.

* `two_stage_b_tight`: for **every** first-stage law, a prover whose chains are all wrong is accepted
  with probability exactly `effEscape univ`, so `twoStage_count`'s bound is attained;
* `naive_two_level_unsound`: that is more than `L₁.miss K`, what one-stage over the coarse
  partition with `L₁` (and the `m`-dependent reading at `m = nv`) would promise. Nested sampling
  dilutes: after the first draw the prover concentrates each lie into one fine unit;
* `late_interior_insecure`: if the interior is committed after the fine draw, a wrong chain is
  accepted with certainty.
-/

namespace FlockSoundness.Audit

open Game
open scoped ENNReal

namespace Law

/-- A law relabelled along an embedding of its units. -/
def map {m n : ℕ} (L : Law m) (e : Fin m ↪ Fin n) : Law n where
  Ω := L.Ω
  draw ω := (L.draw ω).map e

end Law

namespace NotChains

variable (K nv : ℕ)

/-- Gate `j` of chain `c`; gate `0` is the chain's input. -/
def E : Fin K × Fin (nv + 1) ≃ Fin (K * (nv + 1)) := finProdFinEquiv

/-- Fine unit `i` of chain `c`: its gate `i + 1`. -/
def F : Fin K × Fin nv ≃ Fin (K * nv) := finProdFinEquiv

variable {K nv}

theorem E_val (x : Fin K × Fin (nv + 1)) : (E K nv x).val = x.2.val + (nv + 1) * x.1.val :=
  finProdFinEquiv_apply_val x

/-- The previous gate of the chain. -/
def prev (x : Fin K × Fin (nv + 1)) (_h : x.2.val ≠ 0) : Fin K × Fin (nv + 1) :=
  (x.1, ⟨x.2.val - 1, by have := x.2.isLt; omega⟩)

variable (K nv)

/-- Gate `0` of each chain is an input, gate `j + 1` is the NOT of gate `j`. -/
def op (g : Fin (K * (nv + 1))) : Op (K * (nv + 1)) :=
  if h : ((E K nv).symm g).2.val = 0 then .input else .not (E K nv (prev ((E K nv).symm g) h))

theorem topo : ∀ g, ∀ a ∈ (op K nv g).args, a < g := by
  intro g a ha
  unfold op at ha
  split_ifs at ha with h
  · simp [Op.args] at ha
  · simp only [Op.args, List.mem_singleton] at ha
    subst ha
    rw [Fin.lt_def, E_val]
    conv_rhs => rw [← (E K nv).apply_symm_apply g, E_val]
    simp only [prev]
    omega

/-- The chains, their last gates the outputs. -/
abbrev circuit : Circuit where
  N := K * (nv + 1)
  op := op K nv
  topo := topo K nv
  outputs := Finset.univ.filter fun g => ((E K nv).symm g).2.val = nv

variable {K nv}

theorem op_zero (c : Fin K) : (circuit K nv).op (E K nv (c, 0)) = .input := by
  simp [circuit, op]

theorem op_succ (c : Fin K) (i : ℕ) (h : i + 1 < nv + 1) :
    (circuit K nv).op (E K nv (c, ⟨i + 1, h⟩)) = .not (E K nv (c, ⟨i, by omega⟩)) := by
  simp [circuit, op, prev]

theorem isInput_iff (g : Fin (K * (nv + 1))) :
    ((circuit K nv).op g).isInput = true ↔ ((E K nv).symm g).2.val = 0 := by
  simp only [circuit, op]
  split_ifs with h <;> simp [h, Op.isInput]

variable (K nv)

/-- The coarse partition: one unit per chain. -/
def coarse : Partition (circuit K nv) K where
  unit g := if ((E K nv).symm g).2.val = 0 then none else some ((E K nv).symm g).1
  input_iff g := by rw [isInput_iff]; split_ifs with h <;> simp [h]

/-- The fine partition: one unit per NOT gate. -/
def fine : Partition (circuit K nv) (K * nv) where
  unit g := if h : ((E K nv).symm g).2.val = 0 then none else
    some (F K nv (((E K nv).symm g).1, ⟨((E K nv).symm g).2.val - 1, by
      have := ((E K nv).symm g).2.isLt; omega⟩))
  input_iff g := by rw [isInput_iff]; split_ifs with h <;> simp [h]

/-- Each gate's fine unit lies in its chain. -/
def refines : Partition.Refines (fine K nv) (coarse K nv) where
  parent v := ((F K nv).symm v).1
  sub g := by
    simp only [coarse, fine]
    split_ifs with h <;> simp

variable {K nv}

theorem coarse_unit (c : Fin K) (j : Fin (nv + 1)) :
    (coarse K nv).unit (E K nv (c, j)) = if j.val = 0 then none else some c := by
  simp [coarse]

theorem fine_unit_succ (c : Fin K) (i : ℕ) (h : i + 1 < nv + 1) :
    (fine K nv).unit (E K nv (c, ⟨i + 1, h⟩)) = some (F K nv (c, ⟨i, by omega⟩)) := by
  simp [fine]

theorem fine_unit_zero (c : Fin K) : (fine K nv).unit (E K nv (c, 0)) = none := by
  simp [fine]

/-- The gate a fine unit holds. -/
theorem fine_unit_eq' {x : Fin K × Fin (nv + 1)} {c : Fin K} {i : Fin nv}
    (h : (fine K nv).unit (E K nv x) = some (F K nv (c, i))) : x = (c, ⟨i.val + 1, by omega⟩) := by
  obtain ⟨c', j⟩ := x
  simp only [fine, Equiv.symm_apply_apply] at h
  split_ifs at h with hj
  simp only [Option.some.injEq, EmbeddingLike.apply_eq_iff_eq, Prod.mk.injEq] at h
  obtain ⟨rfl, hi⟩ := h
  refine Prod.ext rfl (Fin.ext ?_)
  have := congrArg Fin.val hi
  simp only at this ⊢
  omega

theorem fine_unit_eq {g : Fin (K * (nv + 1))} {c : Fin K} {i : Fin nv}
    (h : (fine K nv).unit g = some (F K nv (c, i))) : g = E K nv (c, ⟨i.val + 1, by omega⟩) := by
  rw [← (E K nv).apply_symm_apply g] at h ⊢
  rw [fine_unit_eq' h]

/-- A coarse unit's local evaluation runs its chain from the committed input. -/
theorem coarse_localEval (X : Fin (K * (nv + 1)) → Bool) (c : Fin K) :
    ∀ (j : ℕ) (hj : j < nv + 1),
      (coarse K nv).localEval X c (E K nv (c, ⟨j, hj⟩)) = (fun b => !b)^[j] (X (E K nv (c, 0))) := by
  intro j
  induction j with
  | zero =>
      intro hj
      rw [Partition.localEval_of_not_mem _ _ (by rw [coarse_unit]; simp)]
      rfl
  | succ i ih =>
      intro hj
      rw [Partition.localEval_of_mem _ _ (by rw [coarse_unit]; simp), op_succ c i hj]
      simp only [Op.apply]
      rw [ih (by omega), Function.iterate_succ_apply']

/-- A fine unit's local evaluation is the NOT of the gate before it, read from the transcript. -/
theorem fine_localEval (Y : Fin (K * (nv + 1)) → Bool) (c : Fin K) (i : ℕ) (h : i + 1 < nv + 1) :
    (fine K nv).localEval Y (F K nv (c, ⟨i, by omega⟩)) (E K nv (c, ⟨i + 1, h⟩)) =
      !Y (E K nv (c, ⟨i, by omega⟩)) := by
  rw [Partition.localEval_of_mem _ _ (fine_unit_succ c i h), op_succ c i h]
  simp only [Op.apply]
  rw [Partition.localEval_of_not_mem]
  rcases i with _ | i
  · rw [show (⟨0, _⟩ : Fin (nv + 1)) = 0 from rfl, fine_unit_zero]
    simp
  · rw [fine_unit_succ c i (by omega)]
    simp

theorem output_mem (c : Fin K) : E K nv (c, Fin.last nv) ∈ (circuit K nv).outputs := by
  simp [circuit]

/-- A chain is wrong when its committed output is not its gates applied to its committed input. -/
theorem coarse_wrong {X : Fin (K * (nv + 1)) → Bool} {c : Fin K} (hnv : 1 ≤ nv)
    (h : X (E K nv (c, Fin.last nv)) ≠ (fun b => !b)^[nv] (X (E K nv (c, 0)))) :
    c ∈ (coarse K nv).wrong X := by
  refine (coarse K nv).mem_wrong.2 fun hc => h ?_
  have := hc _ ((Partition.mem_committed_of_output _ (output_mem c))) (by rw [coarse_unit]; simp; omega)
  rw [this]
  exact coarse_localEval X c nv (by omega)

/-- A fine unit is correct when its gate is the NOT of the gate before it. -/
theorem fine_correct {Y : Fin (K * (nv + 1)) → Bool} {c : Fin K} {i : Fin nv}
    (h : Y (E K nv (c, ⟨i.val + 1, by omega⟩)) = !Y (E K nv (c, ⟨i.val, by omega⟩))) :
    (fine K nv).Correct Y (F K nv (c, i)) := by
  intro g _ hg
  rw [fine_unit_eq hg, h]
  exact (fine_localEval Y c i.val (by omega)).symm

/-- The last fine unit of a chain is wrong when the output is not the NOT of the gate before it. -/
theorem fine_wrong_last {Y : Fin (K * (nv + 1)) → Bool} {c : Fin K} (hnv : 1 ≤ nv)
    (h : Y (E K nv (c, Fin.last nv)) ≠ !Y (E K nv (c, ⟨nv - 1, by omega⟩))) :
    F K nv (c, ⟨nv - 1, by omega⟩) ∈ (fine K nv).wrong Y := by
  refine (fine K nv).mem_wrong.2 fun hc => h ?_
  have e : (Fin.last nv : Fin (nv + 1)) = ⟨nv - 1 + 1, by omega⟩ := Fin.ext (by simp; omega)
  have := hc _ ((Partition.mem_committed_of_output _ (output_mem c))) (by rw [e, fine_unit_succ])
  rw [this, e]
  exact fine_localEval Y c (nv - 1) (by omega)

/-! ## The two-stage audit on NOT chains -/

theorem F_injective : Function.Injective (F K nv) := (F K nv).injective

variable (K nv)

/-- Fine unit `i` of chain `c`, as an embedding of `c`'s units. -/
def emb (c : Fin K) : Fin nv ↪ Fin (K * nv) :=
  ⟨fun i => F K nv (c, i), fun i j h => by simpa using F_injective h⟩

/-- The second stage inside chain `c`: a uniform `k₂`-subset of its gates. -/
def L₂ {k₂ : ℕ} (hk : k₂ ≤ nv) (c : Fin K) : Law (K * nv) :=
  (Law.subset nv k₂ hk).map (emb K nv c)

/-- Transcripts of the chains. -/
abbrev Tr := Fin (K * (nv + 1)) → Bool

open Classical in
/-- The ideal session: it accepts exactly when every drawn fine unit is correct in the fine
transcript (the coarse committed wires from the registration, the rest from the interior). -/
noncomputable def session (S : Finset (Fin (K * nv))) (R : Tr K nv) (_S₁ : Finset (Fin K))
    (I : Tr K nv) : Game Bool :=
  .ret (decide (∀ v ∈ S, (fine K nv).Correct (Partition.extend (Pc := coarse K nv) R I) v))

/-- Every chain's output is the NOT of the honest value before it: each chain is wrong, and its
only wrong fine unit is its last gate. -/
def T : Tr K nv := fun g =>
  if ((E K nv).symm g).2.val = nv then !((fun b => !b)^[nv] false)
  else (fun b => !b)^[((E K nv).symm g).2.val] false

variable {K nv}

theorem T_apply (c : Fin K) (j : Fin (nv + 1)) :
    T K nv (E K nv (c, j)) = if j.val = nv then !((fun b => !b)^[nv] false)
      else (fun b => !b)^[j.val] false := by
  simp [T]

theorem extend_self (X : Tr K nv) : Partition.extend (Pc := coarse K nv) X X = X := by
  classical
  funext g
  simp [Partition.extend]

theorem T_coarse_wrong (hnv : 1 ≤ nv) (c : Fin K) : c ∈ (coarse K nv).wrong (T K nv) := by
  refine coarse_wrong hnv ?_
  rw [T_apply, T_apply]
  simp only [Fin.val_last, ite_true, Fin.val_zero, show (0 : ℕ) ≠ nv by omega, ite_false,
    Function.iterate_zero, id]
  exact Bool.not_ne_self _

theorem T_coarse_wrong_card (hnv : 1 ≤ nv) : K ≤ ((coarse K nv).wrong (T K nv)).card := by
  have : (coarse K nv).wrong (T K nv) = Finset.univ :=
    Finset.eq_univ_of_forall (T_coarse_wrong hnv)
  rw [this, Finset.card_univ, Fintype.card_fin]

/-- In `T`, a chain's fine units are all correct but the last. -/
theorem T_fine_correct_iff (hnv : 1 ≤ nv) (S : Finset (Fin (K * nv))) :
    (∀ v ∈ S, (fine K nv).Correct (T K nv) v) ↔ ∀ c, F K nv (c, ⟨nv - 1, by omega⟩) ∉ S := by
  constructor
  · intro h c hc
    have hw := fine_wrong_last (Y := T K nv) (c := c) hnv (by
      rw [T_apply, T_apply]
      simp only [Fin.val_last, ite_true, show nv - 1 ≠ nv by omega, ite_false]
      have e : (fun b => !b)^[nv] false = !((fun b => !b)^[nv - 1] false) := by
        conv_lhs => rw [show nv = nv - 1 + 1 by omega]
        rw [Function.iterate_succ_apply']
      rw [e, Bool.not_not]
      exact (Bool.not_ne_self _).symm)
    exact (fine K nv).mem_wrong.1 hw (h _ hc)
  · intro h v hv
    obtain ⟨⟨c, i⟩, rfl⟩ := (F K nv).surjective v
    have hi : i.val + 1 < nv := by
      by_contra hc
      exact h c (by rwa [show i = ⟨nv - 1, by omega⟩ from Fin.ext (by
        have := i.isLt; simp only; omega)] at hv)
    refine fine_correct ?_
    rw [T_apply, T_apply]
    simp only [show i.val + 1 ≠ nv by omega, show i.val ≠ nv by omega, ite_false,
      Function.iterate_succ_apply']

theorem mem_fineDraw_last {L₁ : Law K} {k₂ : ℕ} (hk : k₂ ≤ nv) (hnv : 1 ≤ nv) (ω₁ : L₁.Ω)
    (ω₂ : ∀ c, (L₂ K nv hk c).Ω) (c : Fin K) :
    F K nv (c, ⟨nv - 1, by omega⟩) ∈ fineDraw L₁ (L₂ K nv hk) ω₁ ω₂ ↔
      c ∈ L₁.draw ω₁ ∧ (⟨nv - 1, by omega⟩ : Fin nv) ∈ (ω₂ c).1 := by
  simp only [fineDraw, Finset.mem_biUnion, L₂, Law.map, Finset.mem_map, emb]
  constructor
  · rintro ⟨c', hc', i, hi, he⟩
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj (F_injective he)
    exact ⟨hc', hi⟩
  · rintro ⟨hc, hi⟩
    exact ⟨c, hc, _, hi, rfl⟩

theorem mem_map_emb (c : Fin K) (S : Finset (Fin nv)) (i : Fin nv) :
    F K nv (c, i) ∈ S.map (emb K nv c) ↔ i ∈ S :=
  Finset.mem_map' (emb K nv c)

theorem incl_L₂ {k₂ : ℕ} (hk : k₂ ≤ nv) (c : Fin K) (i : Fin nv) :
    (L₂ K nv hk c).incl (F K nv (c, i)) = (Law.subset nv k₂ hk).incl i :=
  congrArg prCoin (funext fun ω => propext (mem_map_emb c ω.1 i))

/-- The probability that a uniform `k₂`-subset of `nv` misses one given unit. -/
theorem subset_miss_one {k₂ : ℕ} (hk : k₂ ≤ nv) (i : Fin nv) :
    prCoin (fun ω : (Law.subset nv k₂ hk).Ω => i ∉ ω.1) =
      ((nv - 1).choose k₂ : ℝ≥0∞) / (nv.choose k₂ : ℝ≥0∞) := by
  have := Law.subset_escape hk {i}
  rw [Finset.card_singleton] at this
  rw [← this]
  unfold Law.escape
  exact congrArg prCoin (funext fun ω => propext Finset.disjoint_singleton_right.symm)

/-- The effective escape of the chains, for any first-stage law. -/
theorem effEscape_chains {L₁ : Law K} {k₂ : ℕ} (hk : k₂ ≤ nv) (hnv : 1 ≤ nv) (B : Finset (Fin K)) :
    effEscape L₁ (L₂ K nv hk) (refines K nv).parent B =
      avg fun ω₁ => (((nv - 1).choose k₂ : ℝ≥0∞) / (nv.choose k₂ : ℝ≥0∞)) ^ (B ∩ L₁.draw ω₁).card := by
  unfold effEscape
  refine congrArg avg (funext fun ω₁ => ?_)
  rw [← Finset.prod_const]
  refine Finset.prod_congr rfl fun c _ => le_antisymm (iSup₂_le fun v hv => ?_) ?_
  · obtain ⟨⟨c', i⟩, rfl⟩ := (F K nv).surjective v
    have hc : c' = c := by simpa [refines] using hv
    subst hc
    rw [incl_L₂, Law.incl, ← Law.prCoin_not]
    exact (subset_miss_one hk i).le
  · refine le_iSup₂_of_le (f := fun v (_ : (refines K nv).parent v = c) => 1 - (L₂ K nv hk c).incl v)
      (F K nv (c, ⟨0, by omega⟩)) (by simp [refines]) (le_of_eq ?_)
    rw [incl_L₂, Law.incl, ← Law.prCoin_not]
    exact (subset_miss_one hk _).symm

/-- A chain's interior gates are not coarse-committed: only its own chain reads them. -/
theorem interior_not_committed (c : Fin K) (j : Fin (nv + 1)) (h0 : j.val ≠ 0) (hn : j.val ≠ nv) :
    E K nv (c, j) ∉ (coarse K nv).committed := by
  classical
  simp only [Partition.committed, Finset.mem_filter, Finset.mem_univ, true_and, not_or]
  refine ⟨by rw [isInput_iff]; simpa using h0, by simpa [circuit] using hn, ?_⟩
  rintro ⟨h, hh, hne⟩
  rw [← (E K nv).apply_symm_apply h] at hh hne
  obtain ⟨c', j'⟩ := (E K nv).symm h
  simp only [op, Equiv.symm_apply_apply] at hh
  split_ifs at hh with hj'
  · simp [Op.args] at hh
  · simp only [Op.args, List.mem_singleton, EmbeddingLike.apply_eq_iff_eq, prev,
      Prod.mk.injEq] at hh
    obtain ⟨rfl, -⟩ := hh
    apply hne
    rw [coarse_unit, coarse_unit]
    simp [h0, hj']

/-- The ideal session's acceptance, for a prover whose interior depends only on the first draw. -/
theorem prob_ideal (L₁ : Law K) (L₂ : Fin K → Law (K * nv)) (R : Tr K nv) (I : L₁.Ω → Tr K nv) :
    prob (fun o => o.1 = true) (twoStage L₁ L₂ (Tr K nv) (fun _ => Tr K nv) (session K nv))
        ⟨R, fun ω₁ => ⟨I ω₁, fun _ => PUnit.unit⟩⟩ =
      avg fun ω₁ => prCoin fun ω₂ => ∀ v ∈ fineDraw L₁ L₂ ω₁ ω₂,
        (fine K nv).Correct (Partition.extend (Pc := coarse K nv) R (I ω₁)) v := by
  classical
  refine congrArg avg (funext fun ω₁ => ?_)
  rw [← avg_ite_eq_prCoin]
  refine congrArg avg (funext fun ω₂ => ?_)
  rw [prob_map]
  simp [session, prob]

variable (K nv)

/-- The dishonest prover: every chain's output wrong, every interior honest from the input. -/
noncomputable def σT (L₁ : Law K) {k₂ : ℕ} (hk : k₂ ≤ nv) :
    Strategy (twoStage L₁ (L₂ K nv hk) (Tr K nv) (fun _ => Tr K nv) (session K nv)) :=
  ⟨T K nv, fun _ => ⟨T K nv, fun _ => PUnit.unit⟩⟩

variable {K nv}

/-- `σT` is accepted with probability `E[q^{|S₁|}]`, `q = C(nv - 1, k₂) / C(nv, k₂)`. -/
theorem prob_σT (L₁ : Law K) {k₂ : ℕ} (hk : k₂ ≤ nv) (hnv : 1 ≤ nv) :
    prob (fun o => o.1 = true) (twoStage L₁ (L₂ K nv hk) (Tr K nv) (fun _ => Tr K nv) (session K nv))
        (σT K nv L₁ hk) =
      avg fun ω₁ => (((nv - 1).choose k₂ : ℝ≥0∞) / (nv.choose k₂ : ℝ≥0∞)) ^ (L₁.draw ω₁).card := by
  rw [σT, prob_ideal]
  refine congrArg avg (funext fun ω₁ => ?_)
  simp only [extend_self]
  have h : ∀ ω₂ : ∀ c, (L₂ K nv hk c).Ω, (∀ v ∈ fineDraw L₁ (L₂ K nv hk) ω₁ ω₂, (fine K nv).Correct (T K nv) v) ↔
      ∀ c ∈ L₁.draw ω₁, (⟨nv - 1, by omega⟩ : Fin nv) ∉ (ω₂ c).1 := by
    intro ω₂
    rw [T_fine_correct_iff hnv]
    simp only [mem_fineDraw_last hk hnv, not_and]
  rw [show (fun ω₂ : ∀ c, (L₂ K nv hk c).Ω => ∀ v ∈ fineDraw L₁ (L₂ K nv hk) ω₁ ω₂,
      (fine K nv).Correct (T K nv) v) = fun ω₂ => ∀ c ∈ L₁.draw ω₁,
      (⟨nv - 1, by omega⟩ : Fin nv) ∉ (ω₂ c).1 from funext fun ω₂ => propext (h ω₂)]
  rw [prCoin_pi_forall (L₁.draw ω₁) (fun c (ω : (L₂ K nv hk c).Ω) => (⟨nv - 1, by omega⟩ : Fin nv) ∉ ω.1)]
  rw [← Finset.prod_const]
  exact Finset.prod_congr rfl fun c _ => subset_miss_one hk _

/-- The profile's bound for `K` wrong chains is the effective escape of all of them. -/
theorem bound_chains (L₁ : Law K) {k₂ : ℕ} (hk : k₂ ≤ nv) :
    (⨆ (B : Finset (Fin K)) (_ : K ≤ B.card), effEscape L₁ (L₂ K nv hk) (refines K nv).parent B) =
      effEscape L₁ (L₂ K nv hk) (refines K nv).parent Finset.univ := by
  refine le_antisymm (iSup₂_le fun B hB => ?_) (le_iSup₂_of_le (f := fun (B : Finset (Fin K)) (_ : K ≤ B.card) =>
    effEscape L₁ (L₂ K nv hk) (refines K nv).parent B) Finset.univ (by simp) le_rfl)
  rw [Finset.eq_univ_of_card B (le_antisymm (by simpa using Finset.card_le_univ B) (by simpa using hB))]

/-- **Tight.** For every first-stage law, a prover whose `K` chains are all wrong is accepted with
probability exactly `twoStage_count`'s bound at `K`. -/
theorem two_stage_b_tight (L₁ : Law K) {k₂ : ℕ} (hk : k₂ ≤ nv) (hnv : 1 ≤ nv) :
    K ≤ ((coarse K nv).wrong (reg₂ (σT K nv L₁ hk))).card ∧
    prob (fun o => o.1 = true) (twoStage L₁ (L₂ K nv hk) (Tr K nv) (fun _ => Tr K nv) (session K nv))
        (σT K nv L₁ hk) =
      ⨆ (B : Finset (Fin K)) (_ : K ≤ B.card), effEscape L₁ (L₂ K nv hk) (refines K nv).parent B := by
  refine ⟨T_coarse_wrong_card hnv, ?_⟩
  rw [bound_chains, effEscape_chains hk hnv, prob_σT L₁ hk hnv]
  simp only [Finset.univ_inter]

/-- **Nested sampling dilutes.** When the second stage leaves some fine unit undrawn and the first
draws something, a prover with `K` wrong chains is accepted with probability above `L₁.miss K`:
more than one-stage over the coarse partition with `L₁` promises, and more than the `m`-dependent
reading of `TwoStageLaw.profile` gives at `m = nv`. -/
theorem naive_two_level_unsound (L₁ : Law K) {k₂ : ℕ} (hk : k₂ + 1 ≤ nv)
    (hdraw : ∃ ω, (L₁.draw ω).Nonempty) :
    K ≤ ((coarse K nv).wrong (reg₂ (σT K nv L₁ (by omega : k₂ ≤ nv)))).card ∧
    L₁.miss K < prob (fun o => o.1 = true)
      (twoStage L₁ (L₂ K nv (by omega : k₂ ≤ nv)) (Tr K nv) (fun _ => Tr K nv) (session K nv))
      (σT K nv L₁ (by omega)) := by
  classical
  refine ⟨T_coarse_wrong_card (by omega), ?_⟩
  rw [prob_σT L₁ (by omega) (by omega)]
  set q := (((nv - 1).choose k₂ : ℝ≥0∞) / (nv.choose k₂ : ℝ≥0∞))
  have hq : q ≠ 0 := by
    refine ENNReal.div_ne_zero.2 ⟨?_, ENNReal.natCast_ne_top _⟩
    exact_mod_cast (Nat.choose_pos (by omega)).ne'
  -- `miss K` is the probability that the first draw is empty
  have hmiss : L₁.miss K = prCoin fun ω => L₁.draw ω = ∅ := by
    refine le_antisymm (iSup₂_le fun B hB => ?_) ?_
    · rw [Finset.eq_univ_of_card B (le_antisymm (by simpa using Finset.card_le_univ B) (by simpa using hB))]
      exact prCoin_mono fun ω h =>
        Finset.subset_empty.1 fun x hx => absurd (Finset.mem_univ x) (Finset.disjoint_left.1 h hx)
    · refine (prCoin_mono fun ω h => ?_).trans (L₁.escape_le_miss (B := Finset.univ) (by simp))
      rw [h]
      exact Finset.disjoint_empty_left _
  rw [hmiss, ← avg_ite_eq_prCoin]
  have hsplit : (fun ω => q ^ (L₁.draw ω).card) =
      fun ω => (if L₁.draw ω = ∅ then (1 : ℝ≥0∞) else 0) +
        (if L₁.draw ω = ∅ then 0 else q ^ (L₁.draw ω).card) := by
    funext ω
    by_cases h : L₁.draw ω = ∅ <;> simp [h]
  rw [hsplit, avg_add]
  refine ENNReal.lt_add_right (ne_top_of_le_ne_top ENNReal.one_ne_top
    ((avg_mono fun ω => by split_ifs <;> simp).trans (avg_const (C := L₁.Ω) 1).le)) ?_
  obtain ⟨ω, hω⟩ := hdraw
  refine (lt_of_lt_of_le ?_ (Finset.single_le_sum (fun c _ => zero_le) (Finset.mem_univ ω))).ne'
  simp only [hω.ne_empty, ite_false]
  exact ENNReal.mul_pos (ENNReal.inv_ne_zero.2 (ENNReal.natCast_ne_top _)) (pow_ne_zero _ hq)

/-! ## (b2) with a Bernoulli first stage is one-stage over the coarse partition, thinned -/

/-- The factor identity behind thinning: a coordinate drawn with probability `num/den` and then
missed with probability `1 − k₂/nv` is a coordinate drawn with probability `num·k₂/(den·nv)`. -/
theorem bernFactor_thin {num den k₂ : ℕ} (hnum : num ≤ den) (hden : 1 ≤ den) (hk : k₂ ≤ nv)
    (hnv : 1 ≤ nv) :
    Law.bernFactor num den (((nv - k₂ : ℕ) : ℝ≥0∞) / (nv : ℝ≥0∞)) =
      Law.bernFactor (num * k₂) (den * nv) 0 := by
  unfold Law.bernFactor
  rw [mul_zero, zero_add]
  have hnv0 : (nv : ℝ≥0∞) ≠ 0 := by exact_mod_cast (show nv ≠ 0 by omega)
  rw [ENNReal.div_eq_div_iff (by exact_mod_cast (show den * nv ≠ 0 by positivity))
    (ENNReal.natCast_ne_top _) (by exact_mod_cast (show den ≠ 0 by omega)) (ENNReal.natCast_ne_top _)]
  have hnat : num * (nv - k₂) + nv * (den - num) = den * nv - num * k₂ := by
    have := Nat.mul_le_mul hnum hk
    zify [hk, hnum, this]
    ring
  calc ((den * nv : ℕ) : ℝ≥0∞) * ((num : ℝ≥0∞) * (((nv - k₂ : ℕ) : ℝ≥0∞) / nv) + ((den - num : ℕ) : ℝ≥0∞))
      = den * (num * (nv * (((nv - k₂ : ℕ) : ℝ≥0∞) / nv)) + nv * ((den - num : ℕ) : ℝ≥0∞)) := by
        push_cast; ring
    _ = den * (num * ((nv - k₂ : ℕ) : ℝ≥0∞) + nv * ((den - num : ℕ) : ℝ≥0∞)) := by
        rw [ENNReal.mul_div_cancel hnv0 (ENNReal.natCast_ne_top _)]
    _ = den * ((den * nv - num * k₂ : ℕ) : ℝ≥0∞) := by
        congr 1
        exact_mod_cast hnat

/-- **(b2) is one-stage over the coarse partition, thinned.** With a Bernoulli(`num/den`) first
stage and a uniform `k₂`-subset of each drawn chain's `nv` gates, the effective escape of any set of
chains is its escape under the one-stage Bernoulli(`num·k₂/(den·nv)`) draw. -/
theorem effEscape_bernoulli {num den k₂ : ℕ} [NeZero den] [NeZero nv] (hnum : num ≤ den)
    (hk : k₂ ≤ nv) (B : Finset (Fin K)) :
    effEscape (Law.bernoulli K num den) (L₂ K nv hk) (refines K nv).parent B =
      (Law.bernoulli K (num * k₂) (den * nv)).escape B := by
  have hnv : 1 ≤ nv := Nat.one_le_iff_ne_zero.2 (NeZero.ne nv)
  have hden : 1 ≤ den := Nat.one_le_iff_ne_zero.2 (NeZero.ne den)
  rw [effEscape_chains hk hnv B, Law.bernoulli_escape (Nat.mul_le_mul hnum hk) B]
  simp only [← Finset.prod_const]
  rw [Law.bernoulli_avg_prod hnum, Law.choose_ratio hk hnv, bernFactor_thin hnum hden hk hnv]

/-! ## Fine values chosen after the fine draw -/

/-- **(b) with the interior committed after the second draw**: the naive ordering. -/
def twoStageLate {nc nf : ℕ} (L₁ : Law nc) (L₂ : Fin nc → Law nf) (Reg : Type)
    (Int : Finset (Fin nc) → Type)
    (session : Finset (Fin nf) → Reg → (S₁ : Finset (Fin nc)) → Int S₁ → Game Bool) :
    Game (Bool × Finset (Fin nc) × Finset (Fin nf)) :=
  .send Reg fun R => .coin L₁.Ω fun ω₁ => .coin (∀ u, (L₂ u).Ω) fun ω₂ =>
  .send (Int (L₁.draw ω₁)) fun I =>
  (session (fineDraw L₁ L₂ ω₁ ω₂) R (L₁.draw ω₁) I).map fun ok => (ok, L₁.draw ω₁, fineDraw L₁ L₂ ω₁ ω₂)

/-- The ideal session's acceptance in the late ordering. -/
theorem prob_ideal_late (L₁ : Law K) (L₂ : Fin K → Law (K * nv)) (R : Tr K nv)
    (I : L₁.Ω → (∀ c, (L₂ c).Ω) → Tr K nv) :
    prob (fun o => o.1 = true) (twoStageLate L₁ L₂ (Tr K nv) (fun _ => Tr K nv) (session K nv))
        ⟨R, fun ω₁ ω₂ => ⟨I ω₁ ω₂, PUnit.unit⟩⟩ =
      avg fun ω₁ => prCoin fun ω₂ => ∀ v ∈ fineDraw L₁ L₂ ω₁ ω₂,
        (fine K nv).Correct (Partition.extend (Pc := coarse K nv) R (I ω₁ ω₂)) v := by
  classical
  refine congrArg avg (funext fun ω₁ => ?_)
  rw [← avg_ite_eq_prCoin]
  refine congrArg avg (funext fun ω₂ => ?_)
  simp [prob, session, Game.map]

/-- One chain of two NOT gates; the second stage draws one of its two gates. -/
noncomputable abbrev lateGame :=
  twoStageLate (Law.subset 1 1 le_rfl) (L₂ 1 2 (k₂ := 1) (by omega)) (Tr 1 2) (fun _ => Tr 1 2)
    (session 1 2)

/-- The prover sets the middle gate after seeing which gate is drawn, so that the drawn one is
correct and the lie sits in the other. -/
def lateInterior (ω₂ : ∀ c, (L₂ 1 2 (k₂ := 1) (by omega) c).Ω) : Tr 1 2 := fun g =>
  if g = E 1 2 (0, 1) then decide ((0 : Fin 2) ∈ (ω₂ 0).1) else T 1 2 g

/-- The late prover. -/
noncomputable def σLate : Strategy lateGame :=
  ⟨T 1 2, fun _ ω₂ => ⟨lateInterior ω₂, PUnit.unit⟩⟩

theorem T_first : T 1 2 (E 1 2 (0, 0)) = false := by rw [T_apply]; rfl

theorem T_last : T 1 2 (E 1 2 (0, 2)) = true := by rw [T_apply]; rfl

theorem late_extend (ω₂ : ∀ c, (L₂ 1 2 (k₂ := 1) (by omega) c).Ω) (g : Fin (1 * (2 + 1))) :
    Partition.extend (Pc := coarse 1 2) (T 1 2) (lateInterior ω₂) g =
      if g = E 1 2 (0, 1) then decide ((0 : Fin 2) ∈ (ω₂ 0).1) else T 1 2 g := by
  classical
  unfold Partition.extend lateInterior
  by_cases hg : g = E 1 2 (0, 1)
  · subst hg
    simp [interior_not_committed (K := 1) (nv := 2) 0 1 (by decide) (by decide)]
  · simp [hg]

/-- **Fine values chosen after the fine draw break the audit**: a wrong chain is accepted with
certainty. -/
theorem late_interior_insecure :
    ((coarse 1 2).wrong (T 1 2)).card = 1 ∧ prob (fun o => o.1 = true) lateGame σLate = 1 := by
  classical
  refine ⟨le_antisymm ((Finset.card_le_univ _).trans (by simp)) (T_coarse_wrong_card (by omega)), ?_⟩
  have hacc : ∀ (ω₁ : (Law.subset 1 1 le_rfl).Ω) (ω₂ : ∀ c, (L₂ 1 2 (k₂ := 1) (by omega) c).Ω),
      ∀ v ∈ fineDraw (Law.subset 1 1 le_rfl) (L₂ 1 2 (k₂ := 1) (by omega)) ω₁ ω₂,
      (fine 1 2).Correct (Partition.extend (Pc := coarse 1 2) (T 1 2) (lateInterior ω₂)) v := by
    intro ω₁ ω₂ v hv
    simp only [fineDraw, Finset.mem_biUnion, L₂, Law.map, Finset.mem_map] at hv
    obtain ⟨c, -, i, hi, rfl⟩ := hv
    obtain ⟨a, ha⟩ := Finset.card_eq_one.1 (ω₂ c).2
    have hc : c = 0 := Subsingleton.elim _ _
    subst hc
    have hia : i = a := by
      have := hi
      rw [show ((Law.subset 2 1 (by omega)).draw (ω₂ 0)) = (ω₂ 0).1 from rfl, ha] at this
      simpa using this
    subst hia
    refine fine_correct ?_
    rw [late_extend, late_extend]
    fin_cases i
    · simp only [ha]; simpa using T_first
    · simp only [ha]; simpa using T_last
  rw [σLate, prob_ideal_late]
  refine (congrArg avg (funext fun ω₁ => prCoin_eq_one_of fun ω₂ => hacc ω₁ ω₂)).trans (avg_const 1)

end NotChains

end FlockSoundness.Audit
