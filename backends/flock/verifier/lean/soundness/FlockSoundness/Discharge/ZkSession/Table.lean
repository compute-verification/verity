import FlockSoundness.Discharge.ZkSession.Level0

/-!
# Soundness of a `--zk` table (live/PROTOCOL.md §1, §5, §7)

`tablePadZK` is `Model.tablePad` with each rep run as under `--zk`: the rep commits its pads (`Ch, Cμ`, §3), runs the
algebraic phase and the masked level 0 (`repPadZK`: the restart `T'` comes from the prover), and closes with the inner
proof on level 0's two lane-phase rows and the sacrifice's three (`repZK`, §5). The table's level-0 commitment and the
link points are as in `tablePad`.

**`table_sound_pad_zk`.** For every prover strategy, the verifier accepts a `--zk` table while no family of witness
lanes within the padded radius of the committed table packs a witness satisfying the statement (`CommittedPad`) with
probability at most `padLinkError + (padRepError + εInner)²`: `table_sound_pad`'s bound with each rep's term increased by
the inner proof's `εInner = (N + 3)/|F| + (1 − δ)^q`. Per rep it is `zk_compose` (the inner proof), whose coupling
premise is `repPadZK_couple` (the masked rep against the clear one) followed by `rep_sound_pad`.

The algebraic phase runs in the clear here; M1 also masks it (§4's zerocheck, lincheck and ring-switch rows), which
this statement does not cover.
-/

namespace FlockSoundness.Discharge.ZkSession

open Game Model Accounting
open scoped ENNReal NNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-- **One `--zk` table, level 0 masked**: the level-0 commitment, the link points, and both reps, each with its own
pads commitment, masked phase and inner proof. Both reps place their pads by `P` and `tr` (the same layout, §2, over
each rep's own pads). -/
noncomputable def tablePadZK (A : Arith F K) {np e : ℕ} (P : Pads0 np e) (tr : Triple np) {N : ℕ} (dom : Fin N ↪ F)
    (nP q : ℕ) (pos : F → Fin N) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (t : ℕ) (c₀ : F)
    (ptLocal mPts : ℕ) : Game (Bool × Oracle F) := do
  let C₀ ← recv (Oracle F)
  let pts ← draw (Fin 2 → Fin mPts → F)
  let extra := extraClaims S ptLocal mPts pts
  let accepts ← interleave (repZK dom np nP q pos tr (repPadZK A P S sch hm t c₀ 0 C₀ extra))
    (repZK dom np nP q pos tr (repPadZK A P S sch hm t c₀ 1 C₀ extra))
  pure (accepts.1 && accepts.2, C₀)

/-- **A `--zk` rep is sound**: from a doomed start over the padded witnesses, the rep (pads commitment, masked phase,
inner proof) accepts with probability at most `padRepError + εInner`. -/
theorem rep_sound_pad_zk (A : Arith F K) (hA : A.Correct) (S : Statement) (sch : Schedule)
    (hshape : 2 ≤ sch.levels.length ∧ (sch.levels.headD default).lanes = sch.k0 ∧
      sch.m - 7 = sch.k0 + (sch.levels.headD default).logCols ∧
        levelsChained sch.levels sch.ext = true)
    (hmsch : sch.m = S.m) (hm : 13 ≤ S.m) (t : ℕ) (hpad : PadOK sch.level₀ t) (c₀ : F)
    (e r mPts : ℕ) (C : Oracle F) (extra : List (Claim F S.m))
    (hextra : extra.length = 2 * S.regions.length)
    (hdoom : RepDoomedW A S sch (padWits A sch t c₀ C) extra)
    {np : ℕ} (P : Pads0 np e) (tr : Triple np) {N : ℕ} (dom : Fin N ↪ F) (nP q : ℕ) [NeZero (np + nP)]
    (pos : F → Fin N) (hpos : ∀ D : Finset (Fin N), prCoin (fun x => pos x ∈ D) ≤ (D.card : ℝ≥0∞) / N)
    (δ : ℝ≥0) (hL : np + nP ≤ N)
    (hδ : δ ≤ Code.relativeUniqueDecodingRadius (ReedSolomon.code dom (np + nP) : Set (Fin N → F))) :
    value (· = true) (repZK dom np nP q pos tr (repPadZK A P S sch hm t c₀ r C extra)) ≤
      ENNReal.ofReal (padRepError sch (S.shape mPts) t e) + εInner N F q δ := by
  have := hA.charTwo
  exact zk_compose dom np nP q pos hpos tr δ hL hδ _ _ fun h _ =>
    (repPadZK_couple A hA P S sch hm t c₀ r C extra h).trans
      (rep_sound_pad A hA S sch hshape hmsch hm t hpad c₀ e r mPts C extra hextra hdoom)

/-- **Soundness of one `--zk` table, level 0 masked** (interactive-oracle layer). For every prover strategy, the
verifier accepts both reps while the committed values do not satisfy the statement with probability at most
`padLinkError + (padRepError + εInner)²`. -/
theorem table_sound_pad_zk (A : Arith F K) (hA : A.Correct)
    (S : Statement) (sch : Schedule)
    (hshape : 2 ≤ sch.levels.length ∧ (sch.levels.headD default).lanes = sch.k0 ∧
      sch.m - 7 = sch.k0 + (sch.levels.headD default).logCols ∧
        levelsChained sch.levels sch.ext = true)
    (hmsch : sch.m = S.m) (hm : 13 ≤ S.m) (t : ℕ) (hpad : PadOK sch.level₀ t) (c₀ : F) (e : ℕ)
    (ptLocal mPts : ℕ) (hlay : S.LinkLayout ptLocal mPts)
    {np : ℕ} (P : Pads0 np e) (tr : Triple np) {N : ℕ} (dom : Fin N ↪ F) (nP q : ℕ) [NeZero (np + nP)]
    (pos : F → Fin N) (hpos : ∀ D : Finset (Fin N), prCoin (fun x => pos x ∈ D) ≤ (D.card : ℝ≥0∞) / N)
    (δ : ℝ≥0) (hL : np + nP ≤ N)
    (hδ : δ ≤ Code.relativeUniqueDecodingRadius (ReedSolomon.code dom (np + nP) : Set (Fin N → F)))
    (σ : Strategy (tablePadZK A P tr dom nP q pos S sch hm t c₀ ptLocal mPts)) :
    prob (fun out => out.1 = true ∧ ¬ CommittedPad A S sch t c₀ out.2)
      (tablePadZK A P tr dom nP q pos S sch hm t c₀ ptLocal mPts) σ ≤
      ENNReal.ofReal (padLinkError sch (S.shape mPts) t) +
        (ENNReal.ofReal (padRepError sch (S.shape mPts) t e) + εInner N F q δ) ^ 2 := by
  refine (prob_le_value _ _ σ).trans ?_
  set ε := ENNReal.ofReal (padRepError sch (S.shape mPts) t e) + εInner N F q δ with hε
  simp only [tablePadZK, bind_eq, pure_eq, recv, draw, bind_send, bind_ret, bind_coin, value_send]
  refine iSup_le fun C => ?_
  by_cases hC : CommittedPad A S sch t c₀ C
  · have h0 : value (fun out : Bool × Oracle F => out.1 = true ∧ ¬ CommittedPad A S sch t c₀ out.2)
        (.coin (Fin 2 → Fin mPts → F) fun pts =>
          ((repZK dom np nP q pos tr (repPadZK A P S sch hm t c₀ 0 C (extraClaims S ptLocal mPts pts))).interleave
              (repZK dom np nP q pos tr (repPadZK A P S sch hm t c₀ 1 C (extraClaims S ptLocal mPts pts)))).bind
            fun accepts => .ret (accepts.1 && accepts.2, C)) = 0 := by
      rw [value_coin]
      refine Finset.sum_eq_zero fun pts _ => ?_
      rw [value_bind_ret, value_congr (q := fun _ => False) (fun acc => by simp [hC]),
        value_false, mul_zero]
    rw [h0]
    exact zero_le
  · refine (value_coin_le _ _ _
      (fun pts => ¬ RepDoomedW A S sch (padWits A sch t c₀ C) (extraClaims S ptLocal mPts pts))
      (ENNReal.ofReal (padLinkError sch (S.shape mPts) t)) (ε * ε)
      (link_sound_pad A hA S sch t hpad c₀ ptLocal mPts C hC hlay) fun pts hpts => ?_).trans ?_
    · rw [not_not] at hpts
      rw [value_bind_ret]
      have hrep := fun r => rep_sound_pad_zk A hA S sch hshape hmsch hm t hpad c₀ e r mPts C _
        (extraClaims_length S ptLocal mPts pts) hpts P tr dom nP q pos hpos δ hL hδ
      calc value (fun acc : Bool × Bool =>
              (acc.1 && acc.2, C).1 = true ∧ ¬ CommittedPad A S sch t c₀ (acc.1 && acc.2, C).2)
              ((repZK dom np nP q pos tr (repPadZK A P S sch hm t c₀ 0 C (extraClaims S ptLocal mPts pts))).interleave
                (repZK dom np nP q pos tr (repPadZK A P S sch hm t c₀ 1 C (extraClaims S ptLocal mPts pts))))
          = value (fun acc : Bool × Bool => acc.1 = true ∧ acc.2 = true)
              ((repZK dom np nP q pos tr (repPadZK A P S sch hm t c₀ 0 C (extraClaims S ptLocal mPts pts))).interleave
                (repZK dom np nP q pos tr (repPadZK A P S sch hm t c₀ 1 C (extraClaims S ptLocal mPts pts)))) :=
            value_congr (fun acc => by simp [hC]) _
        _ ≤ _ := value_interleave_le (fun b : Bool => b = true) (fun b : Bool => b = true) _ _
        _ ≤ ε * ε := mul_le_mul' (hrep 0) (hrep 1)
    · rw [sq]

end FlockSoundness.Discharge.ZkSession
