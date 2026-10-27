import Proofs.Flock.Soundness.Audit.OneStage
import Proofs.Flock.Soundness.Audit.TwoStage
import Proofs.Flock.Soundness.Game.Batch
import Proofs.Flock.Soundness.Instance

/-!
# The Flock instantiation, at the oracle layer

The verifier's session on a draw runs the tables its plan derives from the draw and the transcript,
concurrently, and accepts when every table accepts (`flockSession`). The prover chooses the
schedule, so every other table's coins are part of its strategy.

* `value_batch_le`: every table accepting is at most as likely as any one table accepting
  (`Game.value_interleave_le`, table by table);
* `FlockTable.value_le`: one table, with the executable's arithmetic and the `fast100` numbers, is
  accepted while its committed values do not satisfy its statement with probability at most
  `2^-205` (`table_value_sound`, `execArith_correct`);
* `flock_session_sound`: so the session is `SessionSound` at `2^-205`, given **the lowering**
  (`LoweringSound`, named): a drawn wrong unit lies in a table whose statement no committed table
  satisfies;
* `flock_inputs_sound`: the input unit, an always-proved table, likewise (`InputsLoweringSound`);
* `flock_oracle_count`, `flock_oracle_count_subset`: the audit's count curve,
  `δ(K) = miss K + 2^-205`, and for the uniform `k`-subset `C(n − K, k)/C(n, k) + 2^-205`;
* `flock_oracle_drawn`, `flock_oracle_cone`: drawn units, and a consumer's value through its cone;
* `flock_two_stage_sound`, `flock_two_stage_full_profile`: two-stage (b1) on Flock, one-stage over
  the coarse partition plus `2^-205`.

Mutual correlated agreement is proved (`mcaError_le_bchks25`), so there is no other hypothesis.
Of `Audit/`, only this file and the compiled instances (`FlockCompiled.lean`, `FlockBatched.lean`)
import the Flock theorems.
-/

namespace FlockSoundness.Audit

open Game Model Accounting
open scoped ENNReal

/-! ## Tables run concurrently (`Game.batch`) -/

/-- **Projection to one table.** Every interaction of a batch satisfying `Q` is at most as likely,
for an optimal prover, as any one of them satisfying `Q`. -/
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

/-! ## One Flock table -/

/-- A table of the verifier's session: a statement with its `fast100` schedule and link-point
layout, at a size the numbers cover (`22 ≤ m ≤ 35`, shapes in range). -/
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

/-- The table's session, with the executable's arithmetic. -/
noncomputable def game (T : FlockTable) : Game (Bool × Oracle FlockLevel3.GF128) :=
  table execArith T.S T.sch (by have := T.hm1; omega) T.ptLocal T.mPts

/-- **One table, with the numbers**: accepted while its committed values do not satisfy its
statement with probability at most `2^-205`, for an optimal prover. -/
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

/-! ## The session and the audit -/

namespace Partition

variable {C : Circuit} {n : ℕ} (P : Partition C n)

/-- **The verifier's session on a draw**: the tables its plan derives from the draw and the
transcript, run concurrently; it accepts when every table accepts. -/
noncomputable def flockSession (plan : Finset (Fin n) → (Fin C.N → Bool) → List FlockTable)
    (S : Finset (Fin n)) (tr : Fin C.N → Bool) : Game Bool :=
  (batch ((plan S tr).map FlockTable.game)).map fun outs => decide (∀ o ∈ outs, o.1 = true)

/-- **The lowering** (named hypothesis): a drawn unit that is wrong in the transcript lies in a
table of the plan whose statement no committed table satisfies. At the oracle layer a unit's
statement carries its committed inputs and outputs, so this is the lowering theorem's content: the
statement is satisfiable only if the unit's gates map its committed inputs to its committed
outputs. -/
def LoweringSound (plan : Finset (Fin n) → (Fin C.N → Bool) → List FlockTable) : Prop :=
  ∀ S tr u, u ∈ S → u ∈ P.wrong tr → ∃ T ∈ plan S tr, ∀ C₀, ¬ Committed execArith T.S T.sch C₀

/-- **The input unit's lowering** (named hypothesis): when the transcript's inputs are not the
anchors' values `a`, some table of the plan (the always-proved input statement) is satisfied by no
committed table. -/
def InputsLoweringSound (plan : Finset (Fin n) → (Fin C.N → Bool) → List FlockTable)
    (a : Fin C.N → Bool) : Prop :=
  ∀ S tr, ¬ C.InputsAgree tr a → ∃ T ∈ plan S tr, ∀ C₀, ¬ Committed execArith T.S T.sch C₀

/-- A batched session one of whose tables no committed table satisfies accepts with probability
at most `2^-205`, whatever the prover does. -/
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

/-- **Flock's batched session is `SessionSound` at `2^-205`**, given the lowering: the first
wrong drawn unit's table bounds the whole session. -/
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

/-- **The Flock audit's count curve, at the oracle layer**: `δ(K) = miss K + 2^-205`. -/
theorem flock_oracle_count
    (plan : Finset (Fin n) → (Fin C.N → Bool) → List FlockTable) (hlow : P.LoweringSound plan)
    (L : Law n) (K : ℕ) (σ : Strategy (audit L (Fin C.N → Bool) (flockSession plan))) :
    prob (fun o => o.1 = true ∧ K ≤ (P.wrong (reg σ)).card) (audit L _ (flockSession plan)) σ ≤
      L.miss K + ENNReal.ofReal ((2 : ℝ) ^ (-(205 : ℝ))) :=
  P.oracle_count (P.flock_session_sound plan hlow) K σ

/-- **With the uniform `k`-subset**: `δ(K) = C(n − K, k)/C(n, k) + 2^-205`. -/
theorem flock_oracle_count_subset
    (plan : Finset (Fin n) → (Fin C.N → Bool) → List FlockTable) (hlow : P.LoweringSound plan)
    {k : ℕ} (hk : k ≤ n) {K : ℕ} (hK : K ≤ n)
    (σ : Strategy (audit (Law.subset n k hk) (Fin C.N → Bool) (flockSession plan))) :
    prob (fun o => o.1 = true ∧ K ≤ (P.wrong (reg σ)).card)
        (audit (Law.subset n k hk) _ (flockSession plan)) σ ≤
      ((n - K).choose k : ℝ≥0∞) / (n.choose k : ℝ≥0∞) + ENNReal.ofReal ((2 : ℝ) ^ (-(205 : ℝ))) := by
  have := P.flock_oracle_count plan hlow (Law.subset n k hk) K σ
  rwa [Law.subset_miss hk hK] at this

/-- **Drawn units of the Flock audit** are correct up to `2^-205`, even for a consumer who looks
after the draw. -/
theorem flock_oracle_drawn
    (plan : Finset (Fin n) → (Fin C.N → Bool) → List FlockTable) (hlow : P.LoweringSound plan)
    (L : Law n) (σ : Strategy (audit L (Fin C.N → Bool) (flockSession plan))) :
    prob (fun o => o.1 = true ∧ ¬ Disjoint (P.wrong (reg σ)) o.2) (audit L _ (flockSession plan)) σ ≤
      ENNReal.ofReal ((2 : ℝ) ^ (-(205 : ℝ))) :=
  P.oracle_drawn (P.flock_session_sound plan hlow) σ

/-- **A consumer's value in the Flock audit**: for committed wires `T` fixed before the draw, the
audit accepts with a wrong wire of `T` with probability at most the largest miss probability in
`T`'s cone, plus `2^-205` for the drawn units and `2^-205` for the input unit. -/
theorem flock_oracle_cone
    (plan : Finset (Fin n) → (Fin C.N → Bool) → List FlockTable) (hlow : P.LoweringSound plan)
    {a : Fin C.N → Bool} (hin : InputsLoweringSound plan a) (L : Law n) (T : Finset (Fin C.N))
    (σ : Strategy (audit L (Fin C.N → Bool) (flockSession plan))) :
    prob (fun o => o.1 = true ∧ ∃ t ∈ T, t ∈ P.committed ∧ reg σ t ≠ C.eval a t)
        (audit L _ (flockSession plan)) σ ≤
      (⨆ u ∈ P.cone T, (1 - L.incl u)) + ENNReal.ofReal ((2 : ℝ) ^ (-(205 : ℝ))) +
        ENNReal.ofReal ((2 : ℝ) ^ (-(205 : ℝ))) :=
  P.oracle_cone (P.flock_session_sound plan hlow) (flock_inputs_sound plan hin) T σ

/-! ### Two stages -/

variable {nf nc : ℕ} {Pf : Partition C nf} {Pc : Partition C nc}

/-- **Flock's batched session at the fine level is `SessionSound₂` at `2^-205`**, given the fine
partition's lowering: the tables are planned from the fine transcript, the coarse registration
with the interior filled in. -/
theorem flock_two_stage_sound
    (plan : Finset (Fin nf) → (Fin C.N → Bool) → List FlockTable) (hlow : Pf.LoweringSound plan)
    {Int : Finset (Fin nc) → Type} (ext : ∀ S₁, Int S₁ → (Fin C.N → Bool)) :
    SessionSound₂ Pf Pc ext (fun S R S₁ I => flockSession plan S (extend (Pc := Pc) R (ext S₁ I)))
      (ENNReal.ofReal ((2 : ℝ) ^ (-(205 : ℝ)))) := by
  intro S R S₁ I v hv hw τ
  obtain ⟨T, hT, hno⟩ := hlow S _ v hv hw
  exact flockSession_le plan S _ hT hno τ

/-- **Two-stage (b1) on Flock is one-stage over the coarse partition**: when the second stage
proves every fine unit of each drawn coarse unit,
`Pr[accept ∧ wrong_c(R) ∈ 𝓑] ≤ sup_{B ∈ 𝓑} e_{L₁}(B) + 2^-205`. -/
theorem flock_two_stage_full_profile (r : Refines Pf Pc)
    (plan : Finset (Fin nf) → (Fin C.N → Bool) → List FlockTable) (hlow : Pf.LoweringSound plan)
    {L₁ : Law nc} {L₂ : Fin nc → Law nf} (hfull : ∀ u ω v, r.parent v = u → v ∈ (L₂ u).draw ω)
    {Int : Finset (Fin nc) → Type} (ext : ∀ S₁, Int S₁ → (Fin C.N → Bool))
    (𝓑 : Finset (Fin nc) → Prop)
    (σ : Strategy (twoStage L₁ L₂ (Fin C.N → Bool) Int
      (fun S R S₁ I => flockSession plan S (extend (Pc := Pc) R (ext S₁ I))))) :
    prob (fun o => o.1 = true ∧ 𝓑 (Pc.wrong (reg₂ σ)))
        (twoStage L₁ L₂ _ Int fun S R S₁ I => flockSession plan S (extend (Pc := Pc) R (ext S₁ I))) σ ≤
      (⨆ (B : Finset (Fin nc)) (_ : 𝓑 B), L₁.escape B) + ENNReal.ofReal ((2 : ℝ) ^ (-(205 : ℝ))) :=
  r.oracle₂_full_profile (flock_two_stage_sound plan hlow ext) hfull 𝓑 σ

end Partition

end FlockSoundness.Audit
