import Proofs.Flock.Soundness.Audit.Extraction
import Proofs.Flock.Soundness.Audit.Flock
import Proofs.Flock.Soundness.Knowledge
import Proofs.Flock.Soundness.Lowering

/-!
# The Flock audit at the compiled layer, one table: `ε_ks` from `table_knowledge_sound_joint_tight`

The configuration of `docs/audit-protocols.md` §0.2: the verifier holds the leaf layer (every
commit string and output word) before the draw, so the serving-tree term is zero, and the drawn
units are packed into **one** compiled table, whose statement the verifier derives from the draw and
the registration (`plan`). The session is that table on its own (`sessionC`); the protocol's batched
session of several tables is `FlockBatched.lean`.

**The extractor** of a drawn unit is the table's (`Knowledge.lean`): `Kr` reruns of the prover after
its level-0 cap, the first verifying opening at each position, then decoding. The analysis
(`analysisC`) reads it jointly over the fresh run and the reruns, as the table theorem states it:

* `ks`: the table accepts and the extracted table packs no satisfying witness within the level-0
  radius (`FlockTableC.ksJoint`, `table_knowledge_sound_joint_tight`'s left side);
* `link`: the table accepts, the extracted table is committed, but every close message packing a
  satisfying witness decodes, on the unit's committed wires, to values other than the committed
  transcript's (`LinkEvent`, `linkJoint`). This is the weakest link event: the link theorem need
  only show that some satisfying candidate agrees with `X`.

The decoder is **per drawn unit** (`Decoder`): two drawn units that share a wire hold it in their
own columns, and only the link makes the two readings agree. `cover` (`Pr[accept] ≤ ks + link` at a
wrong drawn unit, pointwise in the reruns) follows from **the lowering** (`LoweringSoundC`, named):
a message packing a witness that satisfies the table's statement decodes, for each drawn unit, to
values on which that unit is correct. `loweringSoundC_of_place` discharges it from each drawn
unit's place in the table (`UnitPlace`, `Lowering.lean`), where the audit's circuit `C` is the rows
the verifier parses; what a correct unit computes is its certificate's (`IsRowsUnit.computes_of_cert`).

**The discharge** (`flock_compiled_knowledgeSound`): the audit's knowledge term is the table
theorem's bound for the prover's strategy after each draw, averaged over the draw,
$$\varepsilon_{ks}(R, \tau) = \mathbb E_\omega\big[\varepsilon_c^-(\sigma_\omega) + K\,\mathrm{Adv}_0(\sigma_\omega) + N_0/(eK)\big],$$
from `table_knowledge_sound_joint_tight`, with no named assumption. The link term
stays a named hypothesis (`LinkSound`) until the link theorem, under expected-time collision
resistance (`Assumptions.SHA512CRExpected`, DESIGN.md §3).

`flock_compiled_count`: `Pr[accept ∧ at least K wrong units] ≤ miss K + ε_ks(σ) + δ_link(σ)`.
-/

namespace FlockSoundness.Audit

open Game Model Accounting
open scoped ENNReal

/-- The executable's field. -/
local notation "𝔽" => FlockLevel3.GF128

/-- A compiled table of the verifier's session: a statement, its `fast100` schedule written out by
levels, and its link-point layout. -/
structure FlockTableC where
  S : Statement
  m : ℕ
  k0 : ℕ
  l₀ : Level
  l₁ : Level
  rest : List Level
  ext : ℕ
  hsch : fast100 S.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
  hm : 13 ≤ S.m
  ptLocal : ℕ
  mPts : ℕ
  hlay : S.LinkLayout ptLocal mPts

namespace FlockTableC

variable {D : Type} (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (T : FlockTableC)

/-- The table's schedule. -/
abbrev sch : Schedule := ⟨T.m, T.k0, T.l₀ :: T.l₁ :: T.rest, T.ext⟩

/-- The table's compiled session, with the executable's arithmetic. -/
noncomputable abbrev game : Game (OutC 𝔽 D) := tableC execArith H E T.S T.sch T.hm T.ptLocal T.mPts

open Classical in
/-- **Accept and extraction fails**, jointly over the fresh run and `Kr` reruns after the level-0
cap: the table accepts, and the extractor's table packs no satisfying witness within the level-0
radius. This is `table_knowledge_sound_joint`'s left side. -/
noncomputable def ksJoint (Kr : ℕ) (σ : Strategy (T.game H E)) : ℝ :=
  expectN (tableCmp execArith H E T.S T.sch T.hm T.ptLocal T.mPts σ.1) σ.2 Kr
    (fun bs => expect (ind fun out : OutC 𝔽 D => out.accepted = true ∧
      ¬ Committed execArith T.S T.sch (extractTable H E T.sch σ.1 bs)) (T.game H E) σ)

/-- `table_knowledge_sound_joint_tight`'s bound for the strategy `σ`:
`ε_c⁻ + K·adv₀ + N₀/(eK)`. -/
noncomputable def ksBound [Inhabited D] (Kr : ℕ) (σ : Strategy (T.game H E)) : ℝ :=
  epsCminus execArith H E T.S T.m T.k0 T.l₀ T.l₁ T.rest T.ext (fast100_shape T.S.m _ T.hsch).2.1
      T.hm T.ptLocal T.mPts σ +
    (Kr * adv₀ execArith H E T.S T.sch T.hm T.ptLocal T.mPts σ +
      2 ^ T.l₀.logLen / (Real.exp 1 * Kr))

theorem ksJoint_nonneg (Kr : ℕ) (σ : Strategy (T.game H E)) : 0 ≤ T.ksJoint H E Kr σ :=
  expectN_nonneg _ _ (fun _ => expect_nonneg (ind_nonneg _) _ _) _

/-- **The table**: `table_knowledge_sound_joint_tight`. -/
theorem ksJoint_le [Inhabited D] (Kr : ℕ) (hK : 1 ≤ Kr)
    (σ : Strategy (T.game H E)) : T.ksJoint H E Kr σ ≤ T.ksBound H E Kr σ :=
  table_knowledge_sound_joint_tight execArith execArith_correct H E T.S T.m T.k0 T.l₀ T.l₁
    T.rest T.ext T.hsch T.hm T.ptLocal T.mPts T.hlay σ Kr hK

end FlockTableC

namespace Partition

variable {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type} {D : Type} [Inhabited D]
  (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (plan : Finset (Fin n) → Reg → FlockTableC)

/-- **The compiled session on a draw**: one compiled table, planned by the verifier from the draw
and the registration; it accepts when the table accepts. -/
noncomputable def sessionC (S : Finset (Fin n)) (R : Reg) : Game Bool :=
  ((plan S R).game H E).map OutC.accepted

/-- The table strategy inside a session strategy. -/
noncomputable def tabStrat {S : Finset (Fin n)} {R : Reg} (τ : Strategy (sessionC H E plan S R)) :
    Strategy ((plan S R).game H E) :=
  Strategy.ofMap _ _ τ

/-- A decoder: for each drawn unit, the values a level-0 message of the table packs for it. Two
drawn units that share a wire hold it in different columns, so each has its own reading. -/
abbrev Decoder := (S : Finset (Fin n)) → (R : Reg) → Fin n →
  Message 𝔽 (plan S R).sch.k0 (plan S R).sch.level₀.logCols → (Fin C.N → Bool)

/-- **The lowering, compiled** (named hypothesis): a message packing a witness that satisfies the
table's statement decodes to values on which every drawn unit is correct. -/
def LoweringSoundC (decode : Decoder (C := C) (Reg := Reg) plan) : Prop :=
  ∀ S R u, u ∈ S → ∀ M, (plan S R).S.Satisfies
    (witnessOf execArith (plan S R).S.m (plan S R).sch.k0 (plan S R).sch.level₀.logCols M) →
    P.Correct (decode S R u M) u

/-- **The link event** on the extractor's reruns `bs`: the extracted table is committed, but every
close message packing a satisfying witness decodes, on `u`'s committed wires, to values other than
`X`'s. -/
def LinkEvent (decode : Decoder (C := C) (Reg := Reg) plan) (S : Finset (Fin n)) (R : Reg)
    (cap₀ : Cap D) (bs : List (OutC 𝔽 D)) (u : Fin n) (Xc : Fin C.N → Bool) : Prop :=
  Committed execArith (plan S R).S (plan S R).sch (extractTable H E (plan S R).sch cap₀ bs) ∧
    ∀ M : Message 𝔽 (plan S R).sch.k0 (plan S R).sch.level₀.logCols,
      Close (plan S R).sch.level₀.radius
        (asTable (plan S R).sch (extractTable H E (plan S R).sch cap₀ bs))
        (encode execArith (plan S R).sch.level₀.logLen M) →
      (plan S R).S.Satisfies
        (witnessOf execArith (plan S R).S.m (plan S R).sch.k0 (plan S R).sch.level₀.logCols M) →
      ∃ g ∈ P.io u, decode S R u M g ≠ Xc g

/-- **The lowering from the drawn units' places** (`Lowering.lean`): if every drawn unit has a place
in its table, reading each unit from its block discharges `LoweringSoundC`. -/
theorem loweringSoundC_of_place (place : ∀ S R u, u ∈ S → UnitPlace P (plan S R).S u) :
    P.LoweringSoundC plan fun S R u M =>
      if h : u ∈ S then (place S R u h).decode
        (witnessOf execArith (plan S R).S.m (plan S R).sch.k0 (plan S R).sch.level₀.logCols M)
      else fun _ => false := by
  intro S R u hu M hM
  simp only [hu, ↓reduceDIte]
  exact (place S R u hu).correct hM

open Classical in
/-- **Accept and the link fails**, jointly over the fresh run and the extractor's `Kr` reruns. -/
noncomputable def linkJoint (decode : Decoder (C := C) (Reg := Reg) plan) (Kr : ℕ)
    (S : Finset (Fin n)) (R : Reg) (τ : Strategy (sessionC H E plan S R)) (u : Fin n)
    (Xc : Fin C.N → Bool) : ℝ :=
  expectN (tableCmp execArith H E (plan S R).S (plan S R).sch (plan S R).hm (plan S R).ptLocal
      (plan S R).mPts (tabStrat H E plan τ).1) (tabStrat H E plan τ).2 Kr
    (fun bs => expect (ind fun out : OutC 𝔽 D => out.accepted = true ∧
        P.LinkEvent H E plan decode S R (tabStrat H E plan τ).1 bs u Xc)
      ((plan S R).game H E) (tabStrat H E plan τ))

omit [Inhabited D] in
theorem linkJoint_nonneg (decode : Decoder (C := C) (Reg := Reg) plan) (Kr : ℕ)
    (S : Finset (Fin n)) (R : Reg) (τ : Strategy (sessionC H E plan S R)) (u : Fin n)
    (Xc : Fin C.N → Bool) : 0 ≤ P.linkJoint H E plan decode Kr S R τ u Xc :=
  expectN_nonneg _ _ (fun _ => expect_nonneg (ind_nonneg _) _ _) _

omit [Inhabited D] in
theorem prob_sessionC (S : Finset (Fin n)) (R : Reg) (τ : Strategy (sessionC H E plan S R)) :
    prob (· = true) (sessionC H E plan S R) τ =
      prob (fun out : OutC 𝔽 D => out.accepted = true) ((plan S R).game H E) (tabStrat H E plan τ) :=
  prob_map _ _ _ _

variable {L : Law n}

/-- **The compiled audit's analysis.** `X` is the committed transcript (the link theorem defines
it), `decode` reads the witness, `Kr` is the extractor's rerun count, one for both terms. -/
noncomputable def analysisC (decode : Decoder (C := C) (Reg := Reg) plan)
    (Xc : (R : Reg) → Cont L Reg (sessionC H E plan) R → (Fin C.N → Bool)) (Kr : ℕ)
    (hlow : P.LoweringSoundC plan decode) : ExtractionAnalysis L Reg (sessionC H E plan) where
  Tr := Fin C.N → Bool
  wrong := P.wrong
  committed := Xc
  ks S R τ _ := ENNReal.ofReal ((plan S R).ksJoint H E Kr (tabStrat H E plan τ))
  link S R τ u Xt := ENNReal.ofReal (P.linkJoint H E plan decode Kr S R τ u Xt)
  cover S R τ u Xt hS hu := by
    classical
    set σ := tabStrat H E plan τ
    have hpt : ∀ (bs : List (OutC 𝔽 D)) (out : OutC 𝔽 D),
        ind (fun out : OutC 𝔽 D => out.accepted = true) out ≤
          ind (fun out : OutC 𝔽 D => out.accepted = true ∧ ¬ Committed execArith (plan S R).S
            (plan S R).sch (extractTable H E (plan S R).sch σ.1 bs)) out +
          ind (fun out : OutC 𝔽 D => out.accepted = true ∧
            P.LinkEvent H E plan decode S R σ.1 bs u Xt) out := by
      intro bs out
      by_cases ha : out.accepted = true
      · by_cases hc : Committed execArith (plan S R).S (plan S R).sch
            (extractTable H E (plan S R).sch σ.1 bs)
        · have hlink : P.LinkEvent H E plan decode S R σ.1 bs u Xt := by
            refine ⟨hc, fun M _ hsat => ?_⟩
            by_contra hno
            push Not at hno
            exact P.mem_wrong.1 hu (P.correct_congr (fun g hg => (hno g hg).symm)
              (hlow S R u hS M hsat))
          simp [ind, ha, hc, hlink]
        · have h1 : ind (fun out : OutC 𝔽 D => out.accepted = true ∧ ¬ Committed execArith
              (plan S R).S (plan S R).sch (extractTable H E (plan S R).sch σ.1 bs)) out = 1 := by
            simp [ind, ha, hc]
          rw [h1, show ind (fun out : OutC 𝔽 D => out.accepted = true) out = 1 by simp [ind, ha]]
          exact le_add_of_nonneg_right (ind_nonneg _ _)
      · simp [ind, ha]
    have hfin : prob (fun out : OutC 𝔽 D => out.accepted = true) ((plan S R).game H E) σ ≠ ⊤ :=
      ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ _ _)
    rw [prob_sessionC H E plan S R τ, ← ENNReal.ofReal_toReal hfin,
      ← ENNReal.ofReal_add (FlockTableC.ksJoint_nonneg H E _ _ _)
        (P.linkJoint_nonneg H E plan decode Kr S R τ u Xt)]
    refine ENNReal.ofReal_le_ofReal ?_
    rw [prob_toReal]
    unfold FlockTableC.ksJoint linkJoint
    calc expect (ind fun out : OutC 𝔽 D => out.accepted = true) ((plan S R).game H E) σ
        = expectN _ _ Kr (fun _ =>
            expect (ind fun out : OutC 𝔽 D => out.accepted = true) ((plan S R).game H E) σ) :=
          (expectN_const _ _ _ Kr).symm
      _ ≤ _ := expectN_mono _ _ (fun bs => (expect_mono (hpt bs) _ _).trans_eq
          (expect_add _ _ _ _)) Kr
      _ = _ := expectN_add _ _ _ _ Kr

/-- **The compiled audit's knowledge term**, per prover: `table_knowledge_sound_joint_tight`'s bound
for its table strategy after each draw, averaged over the draw. -/
noncomputable def ksAvg (Kr : ℕ) (R : Reg) (τ : Cont L Reg (sessionC H E plan) R) : ℝ≥0∞ :=
  avg fun ω => ENNReal.ofReal ((plan (L.draw ω) R).ksBound H E Kr (tabStrat H E plan (τ ω)))

theorem atTgt_le_const (S : Finset (Fin n)) (t : Option (Fin n)) (c : ℝ≥0∞) :
    atTgt S t (fun _ => c) ≤ c := by
  cases t with
  | none => exact zero_le
  | some u => simp only [atTgt, Option.elim_some]; split_ifs <;> simp

/-- **`ε_ks` is discharged at the compiled layer.** The compiled audit's analysis is
`KnowledgeSound` with the table theorem's bound averaged over the draw. -/
theorem flock_compiled_knowledgeSound
    (decode : Decoder (C := C) (Reg := Reg) plan)
    (Xc : (R : Reg) → Cont L Reg (sessionC H E plan) R → (Fin C.N → Bool)) {Kr : ℕ} (hK : 1 ≤ Kr)
    (hlow : P.LoweringSoundC plan decode) :
    (P.analysisC H E plan decode Xc Kr hlow).KnowledgeSound (ksAvg H E plan Kr) := by
  intro R τ tgt
  exact avg_mono fun ω => (atTgt_le_const _ _ _).trans
    (ENNReal.ofReal_le_ofReal ((plan (L.draw ω) R).ksJoint_le H E Kr hK _))

/-- **The compiled Flock audit's count curve**: `δ(K) = miss K + ε_ks(σ) + δ_link(σ)`, with
`ε_ks` the table theorem's bound averaged over the draw, and `δ_link` named. -/
theorem flock_compiled_count
    (decode : Decoder (C := C) (Reg := Reg) plan)
    (Xc : (R : Reg) → Cont L Reg (sessionC H E plan) R → (Fin C.N → Bool)) {Kr : ℕ} (hK : 1 ≤ Kr)
    (hlow : P.LoweringSoundC plan decode)
    {δlink : (R : Reg) → Cont L Reg (sessionC H E plan) R → ℝ≥0∞}
    (hlink : (P.analysisC H E plan decode Xc Kr hlow).LinkSound δlink) (K : ℕ)
    (σ : Strategy (audit L Reg (sessionC H E plan))) :
    prob (fun o => o.1 = true ∧ K ≤ (P.wrong (Xc (reg σ) (cont σ))).card)
        (audit L Reg (sessionC H E plan)) σ ≤
      L.miss K + ksAvg H E plan Kr (reg σ) (cont σ) + δlink (reg σ) (cont σ) := by
  show prob (fun o => o.1 = true ∧ K ≤ ((P.analysisC H E plan decode Xc Kr hlow).wrong
    ((P.analysisC H E plan decode Xc Kr hlow).committedOf σ)).card) _ σ ≤ _
  exact extraction_audit_count (P.analysisC H E plan decode Xc Kr hlow)
    (P.flock_compiled_knowledgeSound H E plan decode Xc hK hlow) hlink K σ

/-- **Drawn units of the compiled Flock audit** are correct up to `ε_ks(σ) + δ_link(σ)`. -/
theorem flock_compiled_drawn
    (decode : Decoder (C := C) (Reg := Reg) plan)
    (Xc : (R : Reg) → Cont L Reg (sessionC H E plan) R → (Fin C.N → Bool)) {Kr : ℕ} (hK : 1 ≤ Kr)
    (hlow : P.LoweringSoundC plan decode)
    {δlink : (R : Reg) → Cont L Reg (sessionC H E plan) R → ℝ≥0∞}
    (hlink : (P.analysisC H E plan decode Xc Kr hlow).LinkSound δlink)
    (σ : Strategy (audit L Reg (sessionC H E plan))) :
    prob (fun o => o.1 = true ∧ ¬ Disjoint (P.wrong (Xc (reg σ) (cont σ))) o.2)
        (audit L Reg (sessionC H E plan)) σ ≤
      ksAvg H E plan Kr (reg σ) (cont σ) + δlink (reg σ) (cont σ) :=
  extraction_audit_drawn (P.analysisC H E plan decode Xc Kr hlow)
    (P.flock_compiled_knowledgeSound H E plan decode Xc hK hlow) hlink σ

end Partition

end FlockSoundness.Audit
