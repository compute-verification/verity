import Proofs.Flock.Soundness.Audit.Extraction
import Proofs.Flock.Soundness.Audit.Flock
import Proofs.Flock.Soundness.SessionBatch
import Proofs.Flock.Soundness.Lowering

/-!
# The Flock audit at the compiled layer, several tables per session

The protocol's session (PROTOCOL.md §5, `SessionBatch.lean`): the verifier plans the tables from the
draw and the registration (`plan`), `Commit` sends every table's level-0 cap, one link-point draw
serves all of them, and then they run concurrently; the session accepts when every table accepts
(`batchedSession`). Each drawn unit is read from one table of the plan (`tab S R u`), with its own
decoder (`DecoderB`): two drawn units that share a wire hold it in their own columns.

**The analysis** (`analysisB`) reads the extractor of the unit's table, over `Kr` reruns of
everything after `Commit`:

* `ks`: `Pr[the session accepts] · Pr[the unit's table extracts to no satisfying witness]`
  (`failProbB`), the joint probability, since the fresh run and the reruns are independent
  (`expect_mul_expectN`);
* `link`: likewise with the link event on the extracted table (`LinkEvB`): it is committed, but
  every close message packing a satisfying witness decodes, on the unit's committed wires, to values
  other than the committed transcript's.

`cover` is `one_le_fail_add_B'`: by **the lowering** (`LoweringSoundB`, named, per table), a
committed table gives the link event for a wrong unit. `loweringSoundB_of_place` discharges it from
each drawn unit's place in its table (`UnitPlace`), with `C` the rows the verifier parses.

**The discharge** (`flock_batched_knowledgeSound`): per draw, the unit's table's
`ε_c⁻ + K·Adv₀ + N₀/(eK)`, computed in the session (`joint_le_B'`, from `session_knowledge_sound`),
bounded by the largest over the plan's tables, and averaged over the draw (`ksAvgB`), with no
named assumption; `δ_link` stays named (`LinkSound`).

`flock_batched_count`: `Pr[accept ∧ at least K wrong units] ≤ miss K + ε_ks(σ) + δ_link(σ)`.
-/

namespace FlockSoundness.Audit

open Game Model Accounting
open scoped ENNReal

/-- The executable's field. -/
local notation "𝔽" => FlockLevel3.GF128

namespace Partition

variable {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type} {D : Type} [Inhabited D]
  (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)

/-- The table of the plan that carries drawn unit `u`. -/
abbrev tabOf (S : Finset (Fin n)) (R : Reg) (u : Fin n) : TabSpec mPts := (plan S R).get (tab S R u)

open Classical in
/-- **The batched session on a draw**: the plan's tables in one session; it accepts when every
table accepts. -/
noncomputable def batchedSession (S : Finset (Fin n)) (R : Reg) : Game Bool :=
  (sessionB execArith H E (plan S R)).map fun outs => decide (acceptedB (plan S R) outs)

/-- The strategy in the protocol's session inside a verdict strategy. -/
noncomputable def batchStrat {S : Finset (Fin n)} {R : Reg} (τ : Strategy (batchedSession H E plan S R)) :
    Strategy (sessionB execArith H E (plan S R)) :=
  Strategy.ofMap _ _ τ

theorem prob_batchedSession (S : Finset (Fin n)) (R : Reg)
    (τ : Strategy (batchedSession H E plan S R)) :
    prob (· = true) (batchedSession H E plan S R) τ =
      prob (acceptedB (plan S R)) (sessionB execArith H E (plan S R)) (batchStrat H E plan τ) := by
  classical
  refine (prob_map _ _ _ _).trans ?_
  congr 1
  funext outs
  exact propext decide_eq_true_iff

/-- A decoder: for each drawn unit, the values a level-0 message of its table packs for it. -/
abbrev DecoderB := (S : Finset (Fin n)) → (R : Reg) → (u : Fin n) →
  Message 𝔽 (tabOf plan tab S R u).sch.k0 (tabOf plan tab S R u).sch.level₀.logCols → (Fin C.N → Bool)

/-- **The lowering, compiled, per table** (named hypothesis): a message packing a witness that
satisfies the statement of drawn unit `u`'s table decodes to values on which `u` is correct. -/
def LoweringSoundB (decode : DecoderB (C := C) (Reg := Reg) plan tab) : Prop :=
  ∀ S R u, u ∈ S → ∀ M, (tabOf plan tab S R u).S.Satisfies
    (witnessOf execArith (tabOf plan tab S R u).S.m (tabOf plan tab S R u).sch.k0
      (tabOf plan tab S R u).sch.level₀.logCols M) →
    P.Correct (decode S R u M) u

/-- **The link event** on the table extracted for unit `u`'s table: it is committed, but every
close message packing a satisfying witness decodes, on `u`'s committed wires, to values other than
`X`'s. -/
def LinkEvB (decode : DecoderB (C := C) (Reg := Reg) plan tab) (S : Finset (Fin n)) (R : Reg)
    (u : Fin n) (Xc : Fin C.N → Bool) (T : Oracle 𝔽) : Prop :=
  Committed execArith (tabOf plan tab S R u).S (tabOf plan tab S R u).sch T ∧
    ∀ M : Message 𝔽 (tabOf plan tab S R u).sch.k0 (tabOf plan tab S R u).sch.level₀.logCols,
      Close (tabOf plan tab S R u).sch.level₀.radius (asTable (tabOf plan tab S R u).sch T)
        (encode execArith (tabOf plan tab S R u).sch.level₀.logLen M) →
      (tabOf plan tab S R u).S.Satisfies
        (witnessOf execArith (tabOf plan tab S R u).S.m (tabOf plan tab S R u).sch.k0
          (tabOf plan tab S R u).sch.level₀.logCols M) →
      ∃ g ∈ P.io u, decode S R u M g ≠ Xc g

/-- **The lowering from the drawn units' places** (`Lowering.lean`): if every drawn unit has a place
in its table, reading each unit from its block discharges `LoweringSoundB`. -/
theorem loweringSoundB_of_place (place : ∀ S R u, u ∈ S → UnitPlace P (tabOf plan tab S R u).S u) :
    P.LoweringSoundB plan tab fun S R u M =>
      if h : u ∈ S then (place S R u h).decode
        (witnessOf execArith (tabOf plan tab S R u).S.m (tabOf plan tab S R u).sch.k0
          (tabOf plan tab S R u).sch.level₀.logCols M)
      else fun _ => false := by
  intro S R u hu M hM
  simp only [hu, ↓reduceDIte]
  exact (place S R u hu).correct hM

variable {L : Law n}

/-- **The batched audit's analysis.** `X` is the committed transcript (the link theorem defines it),
`decode` reads each drawn unit from its table's witness, `Kr` is the extractor's rerun count. -/
noncomputable def analysisB (decode : DecoderB (C := C) (Reg := Reg) plan tab)
    (Xc : (R : Reg) → Cont L Reg (batchedSession H E plan) R → (Fin C.N → Bool)) (Kr : ℕ)
    (hlow : P.LoweringSoundB plan tab decode) : ExtractionAnalysis L Reg (batchedSession H E plan) where
  Tr := Fin C.N → Bool
  wrong := P.wrong
  committed := Xc
  ks S R τ u := prob (acceptedB (plan S R)) (sessionB execArith H E (plan S R)) (batchStrat H E plan τ) *
    ENNReal.ofReal (failProbB execArith H E (plan S R) (tab S R u) Kr (batchStrat H E plan τ))
  link S R τ u Xt := prob (acceptedB (plan S R)) (sessionB execArith H E (plan S R)) (batchStrat H E plan τ) *
    ENNReal.ofReal (extractProbB execArith H E (plan S R) (tab S R u) Kr (batchStrat H E plan τ)
      (P.LinkEvB plan tab decode S R u Xt))
  cover S R τ u Xt hS hu := by
    have hG : ∀ T, Committed execArith (tabOf plan tab S R u).S (tabOf plan tab S R u).sch T →
        P.LinkEvB plan tab decode S R u Xt T := fun T hc =>
      ⟨hc, fun M _ hsat => by
        by_contra hno
        push Not at hno
        exact P.mem_wrong.1 hu (P.correct_congr (fun g hg => (hno g hg).symm) (hlow S R u hS M hsat))⟩
    rw [prob_batchedSession, ← mul_add]
    exact le_mul_of_one_le_right zero_le
      (one_le_fail_add_B' execArith H E (plan S R) (tab S R u) Kr (batchStrat H E plan τ) _ hG)

/-- **The batched audit's knowledge term**, per prover: after each draw, the largest over the
plan's tables of the table's bound in the session, averaged over the draw. -/
noncomputable def ksAvgB (Kr : ℕ) (R : Reg) (τ : Cont L Reg (batchedSession H E plan) R) : ℝ≥0∞ :=
  avg fun ω => ⨆ j : Fin (plan (L.draw ω) R).length,
    ENNReal.ofReal (ksBoundB execArith H E (plan (L.draw ω) R) j Kr (batchStrat H E plan (τ ω)))

/-- **`ε_ks` is discharged for sessions of several tables.** -/
theorem flock_batched_knowledgeSound
    (decode : DecoderB (C := C) (Reg := Reg) plan tab)
    (Xc : (R : Reg) → Cont L Reg (batchedSession H E plan) R → (Fin C.N → Bool)) {Kr : ℕ}
    (hK : 1 ≤ Kr) (hlow : P.LoweringSoundB plan tab decode) :
    (P.analysisB H E plan tab decode Xc Kr hlow).KnowledgeSound (ksAvgB H E plan Kr) := by
  intro R τ tgt
  refine avg_mono fun ω => atTgt_le (fun u _ => ?_) (tgt ω)
  exact (joint_le_B' execArith execArith_correct H E (plan (L.draw ω) R) (tab (L.draw ω) R u)
    Kr hK (batchStrat H E plan (τ ω))).trans (le_iSup (fun j : Fin (plan (L.draw ω) R).length =>
      ENNReal.ofReal (ksBoundB execArith H E (plan (L.draw ω) R) j Kr (batchStrat H E plan (τ ω))))
      (tab (L.draw ω) R u))

/-- **The batched Flock audit's count curve**: `δ(K) = miss K + ε_ks(σ) + δ_link(σ)`. -/
theorem flock_batched_count
    (decode : DecoderB (C := C) (Reg := Reg) plan tab)
    (Xc : (R : Reg) → Cont L Reg (batchedSession H E plan) R → (Fin C.N → Bool)) {Kr : ℕ}
    (hK : 1 ≤ Kr) (hlow : P.LoweringSoundB plan tab decode)
    {δlink : (R : Reg) → Cont L Reg (batchedSession H E plan) R → ℝ≥0∞}
    (hlink : (P.analysisB H E plan tab decode Xc Kr hlow).LinkSound δlink) (K : ℕ)
    (σ : Strategy (audit L Reg (batchedSession H E plan))) :
    prob (fun o => o.1 = true ∧ K ≤ (P.wrong (Xc (reg σ) (cont σ))).card)
        (audit L Reg (batchedSession H E plan)) σ ≤
      L.miss K + ksAvgB H E plan Kr (reg σ) (cont σ) + δlink (reg σ) (cont σ) :=
  extraction_audit_count (P.analysisB H E plan tab decode Xc Kr hlow)
    (P.flock_batched_knowledgeSound H E plan tab decode Xc hK hlow) hlink K σ

/-- **Drawn units of the batched Flock audit** are correct up to `ε_ks(σ) + δ_link(σ)`. -/
theorem flock_batched_drawn
    (decode : DecoderB (C := C) (Reg := Reg) plan tab)
    (Xc : (R : Reg) → Cont L Reg (batchedSession H E plan) R → (Fin C.N → Bool)) {Kr : ℕ}
    (hK : 1 ≤ Kr) (hlow : P.LoweringSoundB plan tab decode)
    {δlink : (R : Reg) → Cont L Reg (batchedSession H E plan) R → ℝ≥0∞}
    (hlink : (P.analysisB H E plan tab decode Xc Kr hlow).LinkSound δlink)
    (σ : Strategy (audit L Reg (batchedSession H E plan))) :
    prob (fun o => o.1 = true ∧ ¬ Disjoint (P.wrong (Xc (reg σ) (cont σ))) o.2)
        (audit L Reg (batchedSession H E plan)) σ ≤
      ksAvgB H E plan Kr (reg σ) (cont σ) + δlink (reg σ) (cont σ) :=
  extraction_audit_drawn (P.analysisB H E plan tab decode Xc Kr hlow)
    (P.flock_batched_knowledgeSound H E plan tab decode Xc hK hlow) hlink σ

end Partition

end FlockSoundness.Audit
