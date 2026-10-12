import Proofs.Flock.Soundness.LigeritoPadSound
import Proofs.Flock.Soundness.Soundness
import Proofs.Flock.Soundness.Accounting.PaddedNumbers

/-!
# Soundness of a table with C-Flock ZK's padded level 0

**`table_sound_pad`.** For every prover strategy, the probability that the verifier accepts a table
with the padded level 0 (both reps, each folding its own `e` extra lanes) while no family of witness
lanes within the padded radius of the committed table packs a witness satisfying the statement
(`CommittedPad`), is at most `padTableError sch shape t e`.

It is `table_sound` with the padded level 0, for any padding `t` and extra lanes per rep `e`:
* the phases before Ligerito pay their union bounds over the padded code's witnesses (`padWits`,
  at most `padListBound` of them), through the phases' list forms
  (`zerocheck_phase_of_list` and its three siblings);
* Ligerito is `ligeritoPad_sound` for each rep's list, which the rep's families' witness lanes
  send into `padWits` (`padWit_mem_padWits`).
-/

namespace FlockSoundness

open Game Model Accounting
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-- **The padded table's witnesses**: the messages of the families of witness lanes within the
padded radius of the committed witness lanes. -/
def padWits (A : Arith F K) (sch : Schedule) (t : ℕ) (c₀ : F) (C : Oracle F) :
    Set (Message F sch.k0 sch.level₀.logCols) :=
  {P | ∃ R ∈ padList A sch.level₀ t sch.k0 0 0 C, padWit c₀ R = P}

/-- **The committed values satisfy the statement, padded level 0**: some family of witness lanes
within the padded radius of the committed table packs a witness that satisfies `S`. -/
def CommittedPad (A : Arith F K) (S : Statement) (sch : Schedule) (t : ℕ) (c₀ : F)
    (C : Oracle F) : Prop :=
  ∃ P ∈ padWits A sch t c₀ C, S.Satisfies (wit A S sch P)

open Classical in
theorem padWits_card_le (A : Arith F K) (hA : A.Correct) (sch : Schedule) (t : ℕ)
    (hpad : PadOK sch.level₀ t) (c₀ : F) (C : Oracle F) :
    (Finset.univ.filter fun P => P ∈ padWits A sch t c₀ C).card ≤ sch.level₀.padListBound t := by
  have hsub : (Finset.univ.filter fun P => P ∈ padWits A sch t c₀ C) ⊆
      (padList A sch.level₀ t sch.k0 0 0 C).image (padWit c₀) := by
    intro P hP
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, padWits, Set.mem_setOf_eq] at hP
    obtain ⟨R, hR, rfl⟩ := hP
    exact Finset.mem_image_of_mem _ hR
  refine (Finset.card_le_card hsub).trans (Finset.card_image_le.trans ?_)
  exact card_closeRows_le A hA _ _ hpad.hd (by have := hpad.hd; omega) hpad.hk
    (by have := Nat.two_pow_pos sch.level₀.logCols; omega) (s := 2 ^ sch.k0 + 0)
    (by have := Nat.two_pow_pos sch.k0; omega) _

/-- **A rep's families send their witness lanes into the padded witnesses.** -/
theorem padWit_mem_padWits (A : Arith F K) (sch : Schedule) (t : ℕ) (c₀ : F) (e r : ℕ)
    (C : Oracle F) (M : Fin (2 ^ sch.k0 + e) → Fin (2 ^ sch.level₀.logCols + t) → F)
    (hM : M ∈ padList A sch.level₀ t sch.k0 e r C) : padWit c₀ M ∈ padWits A sch t c₀ C := by
  classical
  let R : Fin (2 ^ sch.k0 + 0) → Fin (2 ^ sch.level₀.logCols + t) → F :=
    fun j => M (Fin.castAdd e (Fin.cast (Nat.add_zero _) j))
  refine ⟨R, ?_, ?_⟩
  · rw [padList, Finset.mem_filter] at hM ⊢
    refine ⟨Finset.mem_univ _, ?_⟩
    have hM2 := hM.2
    unfold CloseRows at hM2 ⊢
    refine le_trans hM2 ((Nat.cast_le (α := ℝ)).2 (Finset.card_le_card fun q hq => ?_))
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hq ⊢
    intro j
    refine Fin.addCases (fun ℓ => ?_) (fun i => i.elim0) j
    have h := hq (Fin.castAdd e ℓ)
    simp only [padRows, padLane, Fin.addCases_left] at h ⊢
    have hc : Fin.cast (Nat.add_zero _) (Fin.castAdd 0 ℓ) = ℓ := Fin.ext rfl
    simp only [R, hc]
    exact h
  · funext ℓ
    simp only [padWit, R]
    congr 2

/-! ## One rep -/

/-- Zerocheck and the pin, over a list of `L`. -/
noncomputable def εZcP (s : Schedule) (st : Stmt) (L : ℕ) : ℝ :=
  (((L + 1) * (s.m - 13) + L * 127 + L * (2 * (s.m - 6)) + L * (s.m - st.kLog) : ℕ) : ℝ) / qF

/-- Lincheck, over a list of `L`. -/
noncomputable def εLcP (st : Stmt) (L : ℕ) : ℝ := ((L * (2 + 2 * (st.kLog - 6) + 63) : ℕ) : ℝ) / qF

/-- Opening, over a list of `L`. -/
noncomputable def εOpP (st : Stmt) (L : ℕ) : ℝ := ((L * (7 * st.nClaims + 1) : ℕ) : ℝ) / qF

theorem padRepError_eq (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ) (st : Stmt)
    (t e : ℕ) :
    padRepError ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ st t e =
      εZcP ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ st (l₀.padListBound t) + εLcP st (l₀.padListBound t) +
        εOpP st (l₀.padListBound t) + padLigeritoError ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ t e := by
  show piopError m st.kLog st.nClaims (l₀.padListBound t) + padMidLevelError l₀ l₁ t e +
      ligeritoError (l₁ :: rest) = _
  unfold piopError εZcP εLcP εOpP padLigeritoError
  push_cast
  ring

/-- **One rep with the padded level 0 is sound**: from a doomed start over the padded witnesses,
the rep accepts with probability at most `padRepError`. -/
theorem rep_sound_pad (A : Arith F K) (hA : A.Correct)
    (S : Statement) (sch : Schedule)
    (hshape : 2 ≤ sch.levels.length ∧ (sch.levels.headD default).lanes = sch.k0 ∧
      sch.m - 7 = sch.k0 + (sch.levels.headD default).logCols ∧
        levelsChained sch.levels sch.ext = true)
    (hmsch : sch.m = S.m) (hm : 13 ≤ S.m) (t : ℕ) (hpad : PadOK sch.level₀ t) (c₀ : F)
    (e r mPts : ℕ) (C : Oracle F) (extra : List (Claim F S.m))
    (hextra : extra.length = 2 * S.regions.length)
    (hdoom : RepDoomedW A S sch (padWits A sch t c₀ C) extra) :
    value (fun b => b = true) (repPad A S sch hm t c₀ e r C extra) ≤
      ENNReal.ofReal (padRepError sch (S.shape mPts) t e) := by
  classical
  obtain ⟨m, k0, levels, ext⟩ := sch
  rcases levels with _ | ⟨l₀, _ | ⟨l₁, rest⟩⟩
  · simp at hshape
  · simp at hshape
  set sch : Schedule := ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ with hsch
  set W := padWits A sch t c₀ C
  set L := l₀.padListBound t
  have hWL : (Finset.univ.filter fun M => M ∈ W).card ≤ L := padWits_card_le A hA sch t hpad c₀ C
  have hmS : m = S.m := hmsch
  have hq : (0 : ℝ) < qF := by unfold qF; positivity
  have hZc : 0 ≤ εZcP sch (S.shape mPts) L := by unfold εZcP; positivity
  have hLc : 0 ≤ εLcP (S.shape mPts) L := by unfold εLcP; positivity
  have hOp : 0 ≤ εOpP (S.shape mPts) L := by unfold εOpP; positivity
  have hLig : 0 ≤ padLigeritoError sch t e := by
    have hok := levelsChained_ok hshape.2.2.2
    have hr := ligeritoError_nonneg (l₁ :: rest) (fun l hl => hok l (List.mem_cons_of_mem _ hl))
    have hmca := padMcaA_nonneg l₀ t hpad.hr.le hpad.hk1
    have hqK : (0 : ℝ) < qK := by unfold qK; positivity
    have hfe : 0 ≤ l₀.padFoldError t e L := by
      unfold Level.padFoldError
      have h2l : (1 : ℝ) ≤ 2 ^ l₀.lanes := one_le_pow₀ (by norm_num)
      have t1 : (0 : ℝ) ≤ l₀.lanes * (2 * L) / qK :=
        div_nonneg (mul_nonneg (Nat.cast_nonneg _) (mul_nonneg (by norm_num)
          (Nat.cast_nonneg _))) hqK.le
      have t2 : (0 : ℝ) ≤ (2 ^ l₀.lanes - 1) * l₀.padMcaA t / qK :=
        div_nonneg (mul_nonneg (by linarith) hmca) hqK.le
      have t3 : (0 : ℝ) ≤ e * l₀.padMcaA t / qK :=
        div_nonneg (mul_nonneg (Nat.cast_nonneg _) hmca) hqK.le
      have t4 : (0 : ℝ) ≤ e * L / qK :=
        div_nonneg (mul_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)) hqK.le
      linarith
    have hmid : 0 ≤ padMidLevelError l₀ l₁ t e := by
      rw [padMidLevelError_eq]
      exact add_nonneg hfe (midAPad_nonneg l₀ l₁ t)
    have h0 : (0 : ℝ) ≤ (l₀.padListBound t : ℝ) / qF := div_nonneg (Nat.cast_nonneg _) hq.le
    show (0 : ℝ) ≤ (l₀.padListBound t : ℝ) / qF + padMidLevelError l₀ l₁ t e +
      ligeritoError (l₁ :: rest)
    linarith
  rw [padRepError_eq, ENNReal.ofReal_add (by linarith) hLig, ENNReal.ofReal_add (by linarith) hOp,
    ENNReal.ofReal_add hZc hLc]
  unfold repPad
  simp only [bind_eq]
  -- zerocheck
  refine (value_bind_le _ _ _ (DZcW A S sch W extra)
    (ENNReal.ofReal (εLcP (S.shape mPts) L) + ENNReal.ofReal (εOpP (S.shape mPts) L)
      + ENNReal.ofReal (padLigeritoError sch t e)) (fun o ho => ?_)).trans ?_
  · cases o with
    | none => simp
    | some zc =>
      simp only
      -- lincheck
      refine (value_bind_le _ _ _ (DLcW A S sch W extra zc)
        (ENNReal.ofReal (εOpP (S.shape mPts) L) + ENNReal.ofReal (padLigeritoError sch t e))
        (fun o₂ ho₂ => ?_)).trans ?_
      · cases o₂ with
        | none => simp
        | some ab =>
          simp only
          -- opening
          refine (value_bind_le _ _ _ (DOpW S sch W) (ENNReal.ofReal (padLigeritoError sch t e))
            (fun o₃ ho₃ => ?_)).trans ?_
          · cases o₃ with
            | none => simp
            | some op =>
              refine ligeritoPad_sound A hA sch hshape t hpad c₀ e r op.T op.b C fun M hM => ?_
              have := ho₃ (padWit c₀ M) (padWit_mem_padWits A sch t c₀ e r C M hM)
              rw [show sch.m = S.m from hmS]
              exact this
          · refine add_le_add ?_ le_rfl
            have hdoomO : ∀ M ∈ W, ∃ c ∈ ab :: cClaim A zc :: extra, ¬ c.Holds (wit A S sch M) := by
              intro M hM
              by_contra hall
              push Not at hall
              exact ho₂ M hM ⟨hall ab (by simp), hall _ (by simp), fun e he => hall e (by simp [he])⟩
            have hlen : (ab :: cClaim A zc :: extra).length = (S.shape mPts).nClaims := by
              simp [Stmt.nClaims, Statement.shape, hextra]; ring
            refine (opening_phase_of_list A hA S sch (by omega) W _ hdoomO _ hWL).trans (le_of_eq ?_)
            rw [hlen, hA.card_F, εOpP, ofReal_nat_div_qF]
            push_cast
            rw [mul_div_assoc]
            norm_num
      · rw [← add_assoc]
        refine add_le_add (add_le_add ?_ le_rfl) le_rfl
        refine (lincheck_phase_of_list A hA S sch W extra zc ho _ hWL).trans (le_of_eq ?_)
        rw [hA.card_F, εLcP, ofReal_nat_div_qF]
        simp only [Statement.shape]
        push_cast
        rw [mul_div_assoc]
        norm_num
  · calc value (fun o => ¬ DZcW A S sch W extra o) (zerocheck A S.m)
          + (ENNReal.ofReal (εLcP (S.shape mPts) L) + ENNReal.ofReal (εOpP (S.shape mPts) L)
            + ENNReal.ofReal (padLigeritoError sch t e))
        ≤ ENNReal.ofReal (εZcP sch (S.shape mPts) L)
          + (ENNReal.ofReal (εLcP (S.shape mPts) L) + ENNReal.ofReal (εOpP (S.shape mPts) L)
            + ENNReal.ofReal (padLigeritoError sch t e)) := by
          refine add_le_add ?_ le_rfl
          refine (zerocheck_phase_of_list A hA S sch hm W extra hdoom _ hWL).trans (le_of_eq ?_)
          rw [hA.card_F, εZcP, ofReal_nat_div_qF]
          simp only [Statement.shape]
          rw [show sch.m = S.m from hmS, ← mul_div_assoc, ENNReal.div_add_div_same]
          congr 1
          have key : S.m - 13 + L * (S.m - 13 + 127 + 2 * (S.m - 6) + (S.m - S.kLog)) =
              (L + 1) * (S.m - 13) + L * 127 + L * (2 * (S.m - 6)) + L * (S.m - S.kLog) := by ring
          · rw [← key]
            push_cast
            ring
          · norm_num
      _ = _ := by ring

/-! ## One table -/

/-- **The link points, padded level 0**: if the committed values do not satisfy the statement, the
two link points leave some rep start undoomed with probability at most `padLinkError`. -/
theorem link_sound_pad (A : Arith F K) (hA : A.Correct) (S : Statement) (sch : Schedule) (t : ℕ)
    (hpad : PadOK sch.level₀ t) (c₀ : F) (ptLocal mPts : ℕ) (C : Oracle F)
    (hC : ¬ CommittedPad A S sch t c₀ C) (hlay : S.LinkLayout ptLocal mPts) :
    prCoin (fun pts : Fin 2 → Fin mPts → F =>
        ¬ RepDoomedW A S sch (padWits A sch t c₀ C) (extraClaims S ptLocal mPts pts)) ≤
      ENNReal.ofReal (padLinkError sch (S.shape mPts) t) := by
  classical
  refine (link_sound_of_list A hA S sch ptLocal mPts (padWits A sch t c₀ C) hC hlay _
    (padWits_card_le A hA sch t hpad c₀ C)).trans (le_of_eq ?_)
  rw [hA.card_F, padLinkError]
  simp only [Statement.shape]
  rw [ENNReal.ofReal_mul (by positivity), ENNReal.ofReal_mul (by positivity),
    ENNReal.ofReal_natCast, ENNReal.ofReal_natCast, ENNReal.ofReal_pow (by unfold qF; positivity),
    ofReal_nat_div_qF]
  simp only [Schedule.level₀]
  push_cast
  ring_nf

/-- **Soundness of one table with the padded level 0** (interactive-oracle layer). For every prover
strategy, the verifier accepts both reps while the committed values do not satisfy the statement
with probability at most `padTableError`. -/
theorem table_sound_pad (A : Arith F K) (hA : A.Correct)
    (S : Statement) (sch : Schedule)
    (hshape : 2 ≤ sch.levels.length ∧ (sch.levels.headD default).lanes = sch.k0 ∧
      sch.m - 7 = sch.k0 + (sch.levels.headD default).logCols ∧
        levelsChained sch.levels sch.ext = true)
    (hmsch : sch.m = S.m) (hm : 13 ≤ S.m) (t : ℕ) (hpad : PadOK sch.level₀ t) (c₀ : F) (e : ℕ)
    (ptLocal mPts : ℕ) (hlay : S.LinkLayout ptLocal mPts)
    (hε : 0 ≤ padRepError sch (S.shape mPts) t e)
    (σ : Strategy (tablePad A S sch hm t c₀ e ptLocal mPts)) :
    prob (fun out => out.1 = true ∧ ¬ CommittedPad A S sch t c₀ out.2)
      (tablePad A S sch hm t c₀ e ptLocal mPts) σ ≤
      ENNReal.ofReal (padTableError sch (S.shape mPts) t e) := by
  refine (prob_le_value _ _ σ).trans ?_
  set ε := padRepError sch (S.shape mPts) t e
  have hlink0 : 0 ≤ padLinkError sch (S.shape mPts) t := by
    unfold padLinkError qF; positivity
  simp only [tablePad, bind_eq, pure_eq, recv, draw, bind_send, bind_ret, bind_coin, value_send]
  refine iSup_le fun C => ?_
  by_cases hC : CommittedPad A S sch t c₀ C
  · have h0 : value (fun out : Bool × Oracle F => out.1 = true ∧ ¬ CommittedPad A S sch t c₀ out.2)
        (.coin (Fin 2 → Fin mPts → F) fun pts =>
          ((repPad A S sch hm t c₀ e 0 C (extraClaims S ptLocal mPts pts)).interleave
              (repPad A S sch hm t c₀ e 1 C (extraClaims S ptLocal mPts pts))).bind
            fun accepts => .ret (accepts.1 && accepts.2, C)) = 0 := by
      rw [value_coin]
      refine Finset.sum_eq_zero fun pts _ => ?_
      rw [value_bind_ret, value_congr (q := fun _ => False) (fun acc => by simp [hC]),
        value_false, mul_zero]
    rw [h0]
    exact zero_le
  · refine (value_coin_le _ _ _
      (fun pts => ¬ RepDoomedW A S sch (padWits A sch t c₀ C) (extraClaims S ptLocal mPts pts))
      (ENNReal.ofReal (padLinkError sch (S.shape mPts) t)) (ENNReal.ofReal ε * ENNReal.ofReal ε)
      (link_sound_pad A hA S sch t hpad c₀ ptLocal mPts C hC hlay) fun pts hpts => ?_).trans ?_
    · rw [not_not] at hpts
      rw [value_bind_ret]
      have h0 := rep_sound_pad A hA S sch hshape hmsch hm t hpad c₀ e 0 mPts C _
        (extraClaims_length S ptLocal mPts pts) hpts
      have h1 := rep_sound_pad A hA S sch hshape hmsch hm t hpad c₀ e 1 mPts C _
        (extraClaims_length S ptLocal mPts pts) hpts
      calc value (fun acc : Bool × Bool =>
              (acc.1 && acc.2, C).1 = true ∧ ¬ CommittedPad A S sch t c₀ (acc.1 && acc.2, C).2)
              ((repPad A S sch hm t c₀ e 0 C (extraClaims S ptLocal mPts pts)).interleave
                (repPad A S sch hm t c₀ e 1 C (extraClaims S ptLocal mPts pts)))
          = value (fun acc : Bool × Bool => acc.1 = true ∧ acc.2 = true)
              ((repPad A S sch hm t c₀ e 0 C (extraClaims S ptLocal mPts pts)).interleave
                (repPad A S sch hm t c₀ e 1 C (extraClaims S ptLocal mPts pts))) :=
            value_congr (fun acc => by simp [hC]) _
        _ ≤ _ := value_interleave_le (fun b : Bool => b = true) (fun b : Bool => b = true) _ _
        _ ≤ ENNReal.ofReal ε * ENNReal.ofReal ε := mul_le_mul' h0 h1
    · rw [padTableError, ENNReal.ofReal_add hlink0 (by positivity), sq, ENNReal.ofReal_mul hε]

/-! ## At C-Flock ZK's values -/

/-- C-Flock ZK's schedules have the `fast100` shape. -/
theorem m1_shape : ∀ m, 25 ≤ m → m ≤ 35 →
    2 ≤ (m1Schedule m).levels.length ∧
      ((m1Schedule m).levels.headD default).lanes = (m1Schedule m).k0 ∧
      (m1Schedule m).m - 7 = (m1Schedule m).k0 + ((m1Schedule m).levels.headD default).logCols ∧
      levelsChained (m1Schedule m).levels (m1Schedule m).ext = true := by
  intro m h1 h2
  interval_cases m <;> decide +kernel

/-- **The table theorem at C-Flock ZK's values** (`verity/ml/flock/live/PROTOCOL.md` §7 item 6): every `m`
the verifier serves, `25 ≤ m ≤ 35`, level-0 query counts `277, 242, 229` at `m = 25, 26, 27`, 219 at
`m = 28..31` and 218 at `m = 32..35`, padding `t = 2q₀`, a pair of extra lanes per rep. For every prover
strategy, the verifier accepts a table while the committed values do not satisfy the statement with
probability at most `2^-205`, for every statement shape in range. -/
theorem table_sound_pad_m1 (A : Arith F K) (hA : A.Correct)
    (S : Statement) (hm1 : 25 ≤ S.m) (hm2 : S.m ≤ 35) (c₀ : F) (ptLocal mPts : ℕ)
    (hshape : (S.shape mPts).InRange) (hlay : S.LinkLayout ptLocal mPts)
    (σ : Strategy (tablePad A S (m1Schedule S.m) (by omega) (m1Pad S.m) c₀ 2 ptLocal mPts)) :
    prob (fun out => out.1 = true ∧ ¬ CommittedPad A S (m1Schedule S.m) (m1Pad S.m) c₀ out.2)
      (tablePad A S (m1Schedule S.m) (by omega) (m1Pad S.m) c₀ 2 ptLocal mPts) σ ≤
      ENNReal.ofReal ((2 : ℝ) ^ (-(205 : ℝ))) :=
  (table_sound_pad A hA S (m1Schedule S.m) (m1_shape S.m hm1 hm2) (m1Schedule_m S.m hm1 hm2)
    (by omega) (m1Pad S.m) (m1_padOK S.m hm1 hm2) c₀ 2 ptLocal mPts hlay
    (m1_rep_nonneg S.m hm1 hm2 _) σ).trans
    (ENNReal.ofReal_le_ofReal (padTableError_le_m1 S.m hm1 hm2 _ hshape))

end FlockSoundness
