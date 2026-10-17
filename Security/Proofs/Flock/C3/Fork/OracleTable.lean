import Proofs.Flock.C3.Fork.OracleClose
import Proofs.Flock.C3.Fork.OracleRep

/-!
# Piece (a), the oracle count: candidate 3's table with its oracles sent whole

`holoTable_sound`, `table_sound`'s analogue at `Model.holoTable` (`holoTableOf`): every prover makes the table accept
while no table close to the registered word has its hidden statement satisfied by a codeword near the committed level-0
table, with probability at most `holoError`. a1s and a1r are used here:

* within the unique-decoding radius at most one message `M` is near `w` (`card_closeL_ud_le`), and its cells are a table
  `R` (`RegSchedule.Close`); no claim reads the padding (`RegClaim.coord` is zero there), so a claim holds on
  `witnessOf M` iff it holds on `R.bits`;
* a rep whose hidden fold `h` is `R`'s (`sparseFold` at `R.bits`, which needs `hunits`) is a plain rep of
  `R.toStatement`, counted as `rep_sound` counts one from a doomed start (`repError`), after the link points as
  `link_sound` counts them (`linkError`);
* a rep whose `h` is not `R`'s: its sparse phase hands on a claim that holds on `R.bits` with probability at most
  `sparseError` (`sparse_sound`);
* a claim that fails on `R.bits`, or no message near `w`: R's opening accepts with probability at most `regOpenError`
  (`regOpen_sound`, at `fast100`'s schedule for `mReg`; at the empty schedule Ligerito rejects);
* each rep with its sparse phase plays on its own coins (the sparse phase reads only its rep's outputs), so the two
  multiply as `value_interleave_le` multiplies the reps: `(repError + sparseError)^2`.
-/

namespace FlockC3.Fork

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Model Accounting
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

omit [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] in
theorem value_bind_bind_ret {α β γ : Type} (P : γ → Prop) (g : Game α) (f : α → Game β) (k : β → γ) :
    value P (g.bind fun a => (f a).bind fun b => .ret (k b)) = value (fun b => P (k b)) (g.bind f) := by
  rw [← Game.bind_assoc, value_bind_ret]

omit [Fintype K] in
/-- **`regPhase` is both reps' sparse phases, then R's opening**: it accepts no more often than the two `stage`s
followed by `openTail`. -/
theorem regPhase_le (A : Arith F K) (S : Statement) (sh : SparseShape) (units : Fin (2 ^ sh.sLog) → Finset ℕ)
    (τ : Fin (2 ^ sh.sLog) → ℕ) (rsch : Schedule) (w : Oracle F)
    (reps : Option (ZcOut F S.m × HoloLcOut F S.m (S.kLog - 6)) ×
      Option (ZcOut F S.m × HoloLcOut F S.m (S.kLog - 6))) :
    value (fun b => b = true) (regPhase A S sh units τ false (fun op => ligerito A rsch op.T op.b w) reps) ≤
      value (fun b => b = true) (((stage A S sh units τ reps.1).bind fun c₁ =>
        (stage A S sh units τ reps.2).bind fun c₂ => .ret (c₁, c₂)).bind (openTail A sh rsch w)) := by
  rcases reps with ⟨_ | ⟨zc₁, lc₁⟩, _ | ⟨zc₂, lc₂⟩⟩
  · simp [regPhase]
  · simp [regPhase]
  · simp [regPhase]
  simp only [regPhase, stage, bind_eq, pure_eq, Game.bind_assoc, bind_ret]
  refine Discharge.ZkSession.value_bind_mono _ fun c₁ => ?_
  rcases c₁ with _ | c₁
  · simp
  refine Discharge.ZkSession.value_bind_mono _ fun c₂ => ?_
  rcases c₂ with _ | c₂
  · simp
  exact le_of_eq rfl

/-- **`holoTable_sound` for the optimal prover**, at the table's parts. -/
theorem holoTable_value_sound (A : Arith F K) (hA : A.Correct) (cls : HoloClass)
    (regs : List (Region cls.kLog (cls.m - cls.kLog))) (sch : Schedule) (rsch : RegSchedule cls)
    (τ : Fin (2 ^ cls.sLog) → Fin (2 ^ cls.tLog)) (hsch : fast100 cls.m = some sch)
    (hlay : ∀ R : RegTable cls, (R.toStatement τ regs).LinkLayout cls.ptLocal cls.mPts)
    (hR : rsch.sch.levels = [] ∨ fast100 cls.shape.mReg = some rsch.sch)
    (hunits : ∀ σ, ∀ u ∈ cls.units σ, u < 2 ^ (cls.kLog - cls.slotLog)) (w : Oracle F) :
    value (fun out => out.1 = true ∧ ¬ ∃ R : RegTable cls, rsch.Close A w R ∧
        Committed A (R.toStatement τ regs) sch out.2) (holoTableOf A cls regs sch rsch τ w) ≤
      ENNReal.ofReal (linkError sch ⟨cls.kLog, regs.length, cls.mPts⟩ +
        (repError sch ⟨cls.kLog, regs.length, cls.mPts⟩ + sparseError cls.shape) ^ 2 + regOpenError rsch.sch) := by
  classical
  have hrep : 0 ≤ repError sch ⟨cls.kLog, regs.length, cls.mPts⟩ :=
    repError_nonneg _ _ (fast100_levels_ok cls.m sch hsch)
  have hsp : 0 ≤ sparseError cls.shape := by unfold sparseError qF; positivity
  have hlink : 0 ≤ linkError sch ⟨cls.kLog, regs.length, cls.mPts⟩ := by unfold linkError qF; positivity
  have hro : 0 ≤ regOpenError rsch.sch := by
    have hl : 0 ≤ ligeritoError rsch.sch.levels := by
      rcases hR with h | h
      · rw [h]; simp [ligeritoError]
      · exact ligeritoError_nonneg _ (fast100_levels_ok _ _ h)
    have hu : (0 : ℝ) ≤ udQueryError rsch.sch.level₀ := by
      unfold udQueryError Level.rate Accounting.eta; positivity
    have hq : (0 : ℝ) ≤ ((7 * Flock.RecVStar.STACKED_CLAIMS + 2 : ℕ) : ℝ) / qF := by unfold qF; positivity
    exact add_nonneg (add_nonneg hq hu) hl
  simp only [holoTableOf, holoTable, bind_eq, pure_eq, recv, draw, bind_send, bind_ret, bind_coin, value_send]
  refine iSup_le fun C₀ => ?_
  by_cases hgood : ∃ R : RegTable cls, rsch.Close A w R ∧ Committed A (R.toStatement τ regs) sch C₀
  · refine le_of_eq_of_le ?_ zero_le
    rw [value_coin]
    refine Finset.sum_eq_zero fun pts _ => ?_
    rw [value_bind_bind_ret, value_congr (q := fun _ => False) (fun ok => by simp [hgood]), value_false, mul_zero]
  by_cases hnone : ∃ R : RegTable cls, rsch.Close A w R
  · obtain ⟨R₀, hR₀⟩ := hnone
    have hC : ¬ Committed A (R₀.toStatement τ regs) sch C₀ := fun h => hgood ⟨R₀, hR₀, h⟩
    let Q : Option (RegClaim F cls.shape) → Prop := fun o => ∃ c, o = some c ∧ c.toClaim.Holds R₀.bits
    refine (value_coin_le _ _ _ (fun pts => ¬ RepDoomed A (R₀.toStatement τ regs) sch C₀
        (extraClaims (R₀.toStatement τ regs) cls.ptLocal cls.mPts pts))
      (ENNReal.ofReal (linkError sch ⟨cls.kLog, regs.length, cls.mPts⟩))
      (ENNReal.ofReal (repError sch ⟨cls.kLog, regs.length, cls.mPts⟩ + sparseError cls.shape) ^ 2 +
        ENNReal.ofReal (regOpenError rsch.sch))
      (link_sound A hA _ sch hsch cls.ptLocal cls.mPts C₀ hC (hlay R₀)) fun pts hpts => ?_).trans ?_
    · rw [not_not] at hpts
      rw [value_bind_bind_ret]
      refine (value_mono (q := fun b => b = true) (fun ok h => h.1) _).trans ?_
      refine (Discharge.ZkSession.value_bind_mono _ fun reps =>
        regPhase_le A _ cls.shape cls.units _ rsch.sch w reps).trans ?_
      rw [← Game.bind_assoc]
      refine (value_bind_le _ _ _ (fun cc : Option (RegClaim F cls.shape) × Option (RegClaim F cls.shape) =>
        ¬ (Q cc.1 ∧ Q cc.2)) _ (fun cc hcc =>
        openTail_le A hA cls.shape rsch.sch hR w cc ?_)).trans (add_le_add ?_ le_rfl)
      · rintro c₁ c₂ rfl h M hM ⟨h₁, h₂⟩
        have hfit := (levelGood_spec (levelsChained_head_good (fast100_shape _ _ h).2.2.2)).2.1
        have hcell := cells_of_close A hA rsch hfit w R₀ hR₀ M hM
        exact hcc ⟨⟨c₁, rfl, (holds_congr_cells c₁ _ _ hcell).1 h₁⟩, ⟨c₂, rfl, (holds_congr_cells c₂ _ _ hcell).1 h₂⟩⟩
      · refine (value_mono (q := fun cc : Option (RegClaim F cls.shape) × Option (RegClaim F cls.shape) =>
          Q cc.1 ∧ Q cc.2) (fun cc h => not_not.1 h) _).trans ?_
        refine (value_interleave_bind_le Q Q _ _ _ _).trans ?_
        have h1 := holoRep_stage_le A hA cls regs τ R₀ hunits sch hsch cls.mPts C₀ _
          (extraClaims_length _ _ _ _) hpts
        rw [sq]
        exact mul_le_mul' h1 h1
    · rw [ENNReal.ofReal_add (add_nonneg hlink (by positivity)) hro, ENNReal.ofReal_add hlink (by positivity),
        ENNReal.ofReal_pow (add_nonneg hrep hsp), add_assoc]
  · refine le_trans ?_ (ENNReal.ofReal_le_ofReal (le_add_of_nonneg_left (add_nonneg hlink (by positivity))))
    refine value_coin_le_of_forall _ _ _ _ fun pts => ?_
    rw [value_bind_bind_ret]
    refine (value_mono (q := fun b => b = true) (fun ok h => h.1) _).trans ?_
    refine (Discharge.ZkSession.value_bind_mono _ fun reps =>
      regPhase_le A _ cls.shape cls.units _ rsch.sch w reps).trans ?_
    rw [← Game.bind_assoc]
    refine (value_bind_le _ _ _ (fun _ => True) _ fun cc _ => openTail_le A hA cls.shape rsch.sch hR w cc
      fun c₁ c₂ _ _ M hM => absurd ⟨tableOf A rsch M, close_tableOf A rsch w M hM⟩ hnone).trans ?_
    simp [value_false]

/-- **(a1o) Candidate 3's table is sound with its oracles sent whole.** At the reps' schedule `fast100`'s, a link
layout at every table, R's schedule empty or `fast100`'s at `mReg` and every unit slot within the table: for every
registered word `w` and every prover, the table accepts while no table close to `w` has a codeword near the committed
level-0 table packing a witness of its hidden statement with probability at most `holoError`. -/
theorem holoTable_sound (A : Arith F K) (hA : A.Correct) (e : ExecHolo)
    (hsch : Accounting.fast100 e.cls.m = some e.sch)
    (hlay : ∀ R : RegTable e.cls, (R.toStatement e.τ e.regs).LinkLayout e.cls.ptLocal e.cls.mPts)
    (hR : e.rsch.sch.levels = [] ∨ Accounting.fast100 e.cls.shape.mReg = some e.rsch.sch)
    (hunits : ∀ σ, ∀ u ∈ e.cls.units σ, u < 2 ^ (e.cls.kLog - e.cls.slotLog)) (w : Oracle F)
    (σ : Strategy (holoTableOf A e.cls e.regs e.sch e.rsch e.τ w)) :
    prob (fun out => out.1 = true ∧ ¬ ∃ R : RegTable e.cls, e.rsch.Close A w R ∧
        Committed A (R.toStatement e.τ e.regs) e.sch out.2)
      (holoTableOf A e.cls e.regs e.sch e.rsch e.τ w) σ ≤ ENNReal.ofReal (holoError e) := by
  unfold holoError
  exact (prob_le_value _ _ σ).trans (holoTable_value_sound A hA e.cls e.regs e.sch e.rsch e.τ hsch hlay hR hunits w)

end FlockC3.Fork
