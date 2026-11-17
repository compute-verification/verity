module

public import Proofs.Flock.Soundness.Audit.OneStage
public import Proofs.Flock.Soundness.Instance
public import Definitions.Flock.Game
public import Mathlib.Order.CompletePartialOrder
import Mathlib.Algebra.Order.Floor.Semifield
import Mathlib.Combinatorics.Matroid.Init
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Proofs.Flock.Soundness.Accounting.Numbers
import Proofs.Flock.Soundness.Audit.SamplingDesign
import Proofs.Flock.Soundness.Game
import Proofs.Flock.Soundness.Soundness
import VCVio.Prelude.Core

/-! The Flock instantiation at the oracle layer, with mutual correlated agreement proved (`mcaError_le_bchks25`), so the lowerings are its only hypotheses. -/

@[expose] public section

namespace FlockSoundness.Audit

open Game Model Accounting
open scoped ENNReal

theorem value_batch_le {α : Type} (Q : α → Prop) :
    ∀ (gs : List (Game α)) (g : Game α), g ∈ gs →
      value (fun outs => ∀ o ∈ outs, Q o) (batch gs) ≤ value Q g
  | [], _, h => absurd h List.not_mem_nil
  | g' :: gs, g, h => by
      rw [batch, value_map]
      have e : (fun x : α × List α => ∀ o ∈ x.1 :: x.2, Q o) = fun x => Q x.1 ∧ ∀ o ∈ x.2, Q o := by
        funext x
        simp
      rw [e]
      refine (value_interleave_le Q (fun outs => ∀ o ∈ outs, Q o) g' (batch gs)).trans ?_
      rcases List.mem_cons.1 h with rfl | h
      · calc value Q g * value (fun outs => ∀ o ∈ outs, Q o) (batch gs) ≤ value Q g * 1 :=
              mul_le_mul' le_rfl (value_le_one _ _)
          _ = value Q g := mul_one _
      · calc value Q g' * value (fun outs => ∀ o ∈ outs, Q o) (batch gs)
            ≤ 1 * value Q g := mul_le_mul' (value_le_one _ _) (value_batch_le Q gs g h)
          _ = value Q g := one_mul _

structure FlockTable where
  S : Statement
  sch : Schedule
  hsch : fast100 S.m = some sch
  hm1 : 22 ≤ S.m
  hm2 : S.m ≤ 35
  ptLocal : ℕ
  mPts : ℕ
  hshape : (S.shape mPts).InRange
  hlay : S.LinkLayout ptLocal mPts

namespace FlockTable

noncomputable def game (T : FlockTable) : Game (Bool × Oracle FlockLevel3.GF128) :=
  table execArith T.S T.sch (by have := T.hm1; omega) T.ptLocal T.mPts

theorem value_le (T : FlockTable) :
    value (fun out => out.1 = true ∧ ¬ Committed execArith T.S T.sch out.2) T.game ≤
      ENNReal.ofReal ((2 : ℝ) ^ (-(205 : ℝ))) := by
  refine (table_value_sound execArith execArith_correct T.S T.sch T.hsch _ T.ptLocal T.mPts
    T.hlay).trans (ENNReal.ofReal_le_ofReal ?_)
  rcases le_or_gt T.S.m 33 with h | h
  · exact (tableError_le_of_le_33 T.hm1 h T.hsch _ T.hshape).trans
      (Real.rpow_le_rpow_of_exponent_le (by norm_num) (by norm_num))
  · exact tableError_le_of_34_35 (by omega) T.hm2 T.hsch _ T.hshape

end FlockTable

namespace Partition

variable {C : Verity.Boolean.Circuit} {n : ℕ} (P : Partition C n)

noncomputable def flockSession (plan : Finset (Fin n) → (Fin C.N → Bool) → List FlockTable)
    (S : Finset (Fin n)) (tr : Fin C.N → Bool) : Game Bool :=
  (batch ((plan S tr).map FlockTable.game)).map fun outs => decide (∀ o ∈ outs, o.1 = true)

/-- Named hypothesis. At the oracle layer it is the lowering theorem's content: a unit's statement is satisfiable
only if its gates map its committed inputs to its committed outputs. -/
def LoweringSound (plan : Finset (Fin n) → (Fin C.N → Bool) → List FlockTable) : Prop :=
  ∀ S tr u, u ∈ S → u ∈ P.incorrect tr → ∃ T ∈ plan S tr, ∀ C₀, ¬ Committed execArith T.S T.sch C₀

/-- Named hypothesis. -/
def InputsLoweringSound (plan : Finset (Fin n) → (Fin C.N → Bool) → List FlockTable)
    (a : Fin C.N → Bool) : Prop :=
  ∀ S tr, ¬ C.InputsAgree tr a → ∃ T ∈ plan S tr, ∀ C₀, ¬ Committed execArith T.S T.sch C₀

theorem flockSession_le
    (plan : Finset (Fin n) → (Fin C.N → Bool) → List FlockTable) (S : Finset (Fin n))
    (tr : Fin C.N → Bool) {T : FlockTable} (hT : T ∈ plan S tr)
    (hno : ∀ C₀, ¬ Committed execArith T.S T.sch C₀) (τ : Strategy (flockSession plan S tr)) :
    prob (· = true) (flockSession plan S tr) τ ≤ ENNReal.ofReal ((2 : ℝ) ^ (-(205 : ℝ))) := by
  refine (prob_le_value _ _ τ).trans ?_
  rw [flockSession, value_map]
  refine (value_mono (q := fun outs => ∀ o ∈ outs, o.1 = true) (fun outs h => by simpa using h)
    _).trans ?_
  refine (value_batch_le _ _ T.game (List.mem_map.2 ⟨T, hT, rfl⟩)).trans ?_
  exact (value_mono (fun out h => ⟨h, hno out.2⟩) _).trans (T.value_le)

theorem flock_session_sound
    (plan : Finset (Fin n) → (Fin C.N → Bool) → List FlockTable) (hlow : P.LoweringSound plan) :
    P.SessionSound (flockSession plan) (ENNReal.ofReal ((2 : ℝ) ^ (-(205 : ℝ)))) := by
  intro S tr u hu hw τ
  obtain ⟨T, hT, hno⟩ := hlow S tr u hu hw
  exact flockSession_le plan S tr hT hno τ

theorem flock_inputs_sound
    (plan : Finset (Fin n) → (Fin C.N → Bool) → List FlockTable) {a : Fin C.N → Bool}
    (hin : InputsLoweringSound plan a) :
    InputsSound (flockSession plan) a (ENNReal.ofReal ((2 : ℝ) ^ (-(205 : ℝ)))) := by
  intro S tr hbad τ
  obtain ⟨T, hT, hno⟩ := hin S tr hbad
  exact flockSession_le plan S tr hT hno τ

theorem flock_oracle_count
    (plan : Finset (Fin n) → (Fin C.N → Bool) → List FlockTable) (hlow : P.LoweringSound plan)
    (L : SamplingDesign n) (K : ℕ) (σ : Strategy (audit L (Fin C.N → Bool) (flockSession plan))) :
    prob (fun o => o.1 = true ∧ K ≤ (P.incorrect (reg σ)).card) (audit L _ (flockSession plan)) σ ≤
      L.miss K + ENNReal.ofReal ((2 : ℝ) ^ (-(205 : ℝ))) :=
  P.oracle_count (P.flock_session_sound plan hlow) K σ

theorem flock_oracle_count_subset
    (plan : Finset (Fin n) → (Fin C.N → Bool) → List FlockTable) (hlow : P.LoweringSound plan)
    {k : ℕ} (hk : k ≤ n) {K : ℕ} (hK : K ≤ n)
    (σ : Strategy (audit (SamplingDesign.subset n k hk) (Fin C.N → Bool) (flockSession plan))) :
    prob (fun o => o.1 = true ∧ K ≤ (P.incorrect (reg σ)).card)
        (audit (SamplingDesign.subset n k hk) _ (flockSession plan)) σ ≤
      ((n - K).choose k : ℝ≥0∞) / (n.choose k : ℝ≥0∞) + ENNReal.ofReal ((2 : ℝ) ^ (-(205 : ℝ))) := by
  have := P.flock_oracle_count plan hlow (SamplingDesign.subset n k hk) K σ
  rwa [SamplingDesign.subset_miss hk hK] at this

/-- Holds even for a consumer who looks after the sample. -/
theorem flock_oracle_sampled
    (plan : Finset (Fin n) → (Fin C.N → Bool) → List FlockTable) (hlow : P.LoweringSound plan)
    (L : SamplingDesign n) (σ : Strategy (audit L (Fin C.N → Bool) (flockSession plan))) :
    prob (fun o => o.1 = true ∧ ¬ Disjoint (P.incorrect (reg σ)) o.2) (audit L _ (flockSession plan)) σ ≤
      ENNReal.ofReal ((2 : ℝ) ^ (-(205 : ℝ))) :=
  P.oracle_sampled (P.flock_session_sound plan hlow) σ

theorem flock_oracle_cone
    (plan : Finset (Fin n) → (Fin C.N → Bool) → List FlockTable) (hlow : P.LoweringSound plan)
    {a : Fin C.N → Bool} (hin : InputsLoweringSound plan a) (L : SamplingDesign n) (T : Finset (Fin C.N))
    (σ : Strategy (audit L (Fin C.N → Bool) (flockSession plan))) :
    prob (fun o => o.1 = true ∧ ∃ t ∈ T, t ∈ P.committed ∧ reg σ t ≠ C.eval a t)
        (audit L _ (flockSession plan)) σ ≤
      (⨆ u ∈ P.cone T, (1 - L.incl u)) + ENNReal.ofReal ((2 : ℝ) ^ (-(205 : ℝ))) +
        ENNReal.ofReal ((2 : ℝ) ^ (-(205 : ℝ))) :=
  P.oracle_cone (P.flock_session_sound plan hlow) (flock_inputs_sound plan hin) T σ

end Partition

end FlockSoundness.Audit
