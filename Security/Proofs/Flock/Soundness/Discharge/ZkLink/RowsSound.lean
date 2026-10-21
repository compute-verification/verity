import Proofs.Flock.Soundness.Discharge.ZkLink.Rows
import Proofs.Flock.Soundness.Discharge.ZkLink.Public

/-!
# The rows link at the compiled `--zk` audit

`analysisZCReg` counts a drawn unit wrong when it is wrong on the registration (`wrongRegZ`). This file adds the units
whose value layer differs, at a registered read, from the registrant's row (`rowsWrongZ`), and bounds the drawn units of
both kinds by `ε_ks + δ_link + δ_rows`, at a registrant's openings `wo` of the registered reads `isRead` whose
commitments are the registered ones:

* `RowsLinkEvZK`: the link event with the rows added: every close satisfying message decodes off the value layer on
  some wire of the unit, or opens a registered read to a value other than the registrant's;
* `psiRowsZC`, `rowsLinkEv_le_psiZC`: its trace at the commit strings, the link's (`psiZC`) plus the rows' (an
  extraction reads, at a registered read, a value other than the registrant's);
* `trial_rows_leZC`: the trial's rows mass at a registered read is at most `t'(1 + k)/2^256`, under A2 for the rows
  finder (`RowsCRZC`), since the trial's opening and the registrant's then collide (`ValueBindingZ.collide_spec`);
* `ideal_le_trialZC`: `ideal_mass_leZC`'s step from the trial to the audit's extractor, for any trace;
* `analysisZCRows`, `zk_rowsLinkSoundC`: the analysis, and its link term at `rowsBoundZC`, `δ_link + δ_rows` with
  `δ_rows` at most `δ_link`'s closed form;
* `zk_flock_drawnC_rows`: `Audit.extraction_audit_drawn` at it.
-/

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting Extract LinkFinder Audit Audit.Partition ZkSession
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type} [Inhabited D] [Inhabited Sl]
variable (A : Arith F K) (Hs : Hashes F D Sl) {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)

variable {L : Law n} {decode : DecoderZK (C := C) planZ tab} {Dh : Type}
  {Hc : List UInt8 → Dh} (vb : ValueBindingZ A P planZ tab decode Hc)
  (isRead : vb.Pos → Prop) (wo : vb.Pos → vb.Opening)

/-- **The link event with the rows**: on the oracle of unit `u`'s table, `CommittedPad`, and every padded witness near
it that satisfies the statement decodes, on some wire of `u`, to a value other than `Xc`'s, or opens a registered read
on `u`'s wires to a value other than the registrant's (`wo`). -/
def RowsLinkEvZK (S : Finset (Fin n)) (R : Reg) (u : Fin n) (Xc : Fin C.N → Bool) (C₀ : Oracle F) : Prop :=
  (tabOfZ planZ tab S R u).Committed A C₀ ∧
    ∀ M ∈ padWits A (tabOfZ planZ tab S R u).sch (tabOfZ planZ tab S R u).t (tabOfZ planZ tab S R u).c₀ C₀,
      (tabOfZ planZ tab S R u).S.Satisfies (wit A (tabOfZ planZ tab S R u).S (tabOfZ planZ tab S R u).sch M) →
        (∃ g ∈ P.io u, decode S R u M g ≠ Xc g) ∨
          ∃ g ∈ P.io u, isRead (vb.pos g) ∧ vb.val (vb.opening S R u M (vb.pos g)) ≠ vb.val (wo (vb.pos g))

open Classical in
/-- **The units off the registrant's rows** at the value layer `Xv`: those with a committed wire at a registered read
whose bit in `Xv` is not the registrant's row's. -/
noncomputable def rowsWrongZ (Xv : Fin C.N → Bool) : Finset (Fin n) :=
  Finset.univ.filter fun u => ∃ g ∈ P.io u, isRead (vb.pos g) ∧ Xv g ≠ vb.bit (vb.val (wo (vb.pos g))) g

omit [Fintype K] [DecidableEq K] in
theorem mem_rowsWrongZ {Xv : Fin C.N → Bool} {u : Fin n} :
    u ∈ rowsWrongZ A P planZ tab vb isRead wo Xv ↔
      ∃ g ∈ P.io u, isRead (vb.pos g) ∧ Xv g ≠ vb.bit (vb.val (wo (vb.pos g))) g := by
  classical
  unfold rowsWrongZ
  rw [Finset.mem_filter]
  exact ⟨fun h => h.2, fun h => ⟨Finset.mem_univ u, h⟩⟩

section Trial

variable (R : Reg) (τ : Cont L Reg (batchedSessionZC A Hs planZ) R)

open Classical in
/-- **The rows' trace at commit string `c`**: `c` is a registered read, and the extraction from the reruns `bs` reads
a value there other than the registrant's. -/
noncomputable def psiRowsZC (c : vb.Pos) (ω : L.Ω) (bs : List (List (OutZC F D Sl))) : ℝ :=
  if isRead c ∧ ∃ v ∈ readsAtZC A Hs P planZ tab vb R τ ω bs c, v ≠ vb.val (wo c) then 1 else 0

omit [Fintype K] in
theorem psiRows_nonnegZC (c : vb.Pos) (ω : L.Ω) (bs : List (List (OutZC F D Sl))) :
    0 ≤ psiRowsZC A Hs P planZ tab vb isRead wo R τ c ω bs := by
  unfold psiRowsZC; split_ifs <;> norm_num

omit [Fintype K] in
theorem psiRows_le_oneZC (c : vb.Pos) (ω : L.Ω) (bs : List (List (OutZC F D Sl))) :
    psiRowsZC A Hs P planZ tab vb isRead wo R τ c ω bs ≤ 1 := by
  unfold psiRowsZC; split_ifs <;> norm_num

omit [Fintype K] in
theorem trialVal_psiRowsZC (c : vb.Pos) (e : TrialOutZC F L D Sl) :
    trialValZC planZ R (psiRowsZC A Hs P planZ tab vb isRead wo R τ c) e =
      ind (fun e => isRead c ∧ ∃ v ∈ rdTZC A Hs P planZ tab vb R τ c e, v ≠ vb.val (wo c)) e := by
  classical
  unfold trialValZC rdTZC
  by_cases ha : accωZC planZ R e.1 e.2.1
  · cases hf : firsts (accωZC planZ R e.1) e.2.2 with
    | none => simp [ind, ha, hf]
    | some bs => simp [ind, ha, hf, psiRowsZC]
  · simp [ind, ha]

/-- **The trial's rows mass at a registered read `c`**, under A2 for the rows finder: at most `t'(1 + k)/2^256`. A
value the trial reads at `c` other than the registrant's comes with an opening of `c`'s registered commitment
(`opOfVZC_spec`), which is the registrant's opening's (`hon`), so the binding's extractor collides on the two. -/
theorem trial_rows_leZC {t' : ℝ} (ht : 0 ≤ t') (k Rw : ℕ) (c : vb.Pos) [Nonempty (ΩcZ A P planZ tab vb L c)]
    (_hrc : isRead c) (hon : vb.com (wo c) = vb.cs R c)
    (hCR : Assumptions.SHA512CRExpected Hc (trialGZC A Hs P planZ tab vb R τ c k Rw)
      (trialStratZC A Hs P planZ tab vb R τ c k Rw) (rowsOutZC A Hs P planZ tab vb R τ wo c) (cTZC planZ R t')) :
    expect (trialValZC planZ R (psiRowsZC A Hs P planZ tab vb isRead wo R τ c))
        (trialGZC A Hs P planZ tab vb R τ c k Rw) (trialStratZC A Hs P planZ tab vb R τ c k Rw) ≤
      t' * (1 + k) / 2 ^ (256 : ℝ) := by
  classical
  rw [show trialValZC planZ R (psiRowsZC A Hs P planZ tab vb isRead wo R τ c) =
      ind (fun e => isRead c ∧ ∃ v ∈ rdTZC A Hs P planZ tab vb R τ c e, v ≠ vb.val (wo c)) from
    funext fun e => trialVal_psiRowsZC A Hs P planZ tab vb isRead wo R τ c e, ← prob_toReal]
  have hsub : prob (fun e => isRead c ∧ ∃ v ∈ rdTZC A Hs P planZ tab vb R τ c e, v ≠ vb.val (wo c))
        (trialGZC A Hs P planZ tab vb R τ c k Rw) (trialStratZC A Hs P planZ tab vb R τ c k Rw) ≤
      prob (fun e => ∃ x y, rowsOutZC A Hs P planZ tab vb R τ wo c e = some (x, y) ∧ x ≠ y ∧ Hc x = Hc y)
        (trialGZC A Hs P planZ tab vb R τ c k Rw) (trialStratZC A Hs P planZ tab vb R τ c k Rw) := by
    refine prob_mono (fun e he => ?_) _ _
    obtain ⟨-, hex⟩ := he
    show ∃ x y, rowsOutZC A Hs P planZ tab vb R τ wo c e = some (x, y) ∧ x ≠ y ∧ Hc x = Hc y
    unfold rowsOutZC
    rw [dite_eq_left_of_eq_true (eq_true hex)]
    obtain ⟨h1, hne⟩ := Classical.choose_spec hex
    have s1 := opOfVZC_spec A Hs P planZ tab vb R τ h1
    exact ⟨_, _, rfl, vb.collide_spec _ _ (s1.1.trans hon.symm) (by rw [s1.2]; exact hne)⟩
  have hfin : prob (fun e => ∃ x y, rowsOutZC A Hs P planZ tab vb R τ wo c e = some (x, y) ∧ x ≠ y ∧ Hc x = Hc y)
      (trialGZC A Hs P planZ tab vb R τ c k Rw) (trialStratZC A Hs P planZ tab vb R τ c k Rw) ≠ ⊤ :=
    ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ _ _)
  have hcost := expect_cT_leZC A Hs P planZ tab vb R τ c k Rw ht
  calc _ ≤ _ := ENNReal.toReal_mono hfin hsub
    _ ≤ expect (cTZC planZ R t') (trialGZC A Hs P planZ tab vb R τ c k Rw)
          (trialStratZC A Hs P planZ tab vb R τ c k Rw) / 2 ^ (256 : ℝ) := hCR
    _ ≤ _ := div_le_div_of_nonneg_right hcost (by positivity)

/-- **From the trial to the audit's extractor, for any trace `Ψ` at `c`** (`ideal_mass_leZC`'s step): a trace in
`[0, 1]` that vanishes at draws that do not read `c`, jointly with acceptance and averaged over the draw, is at most
`π c` times the trial's bound `Bt` plus `k/(eRw)` for the waits that give up. -/
theorem ideal_le_trialZC (k Rw : ℕ) (hRw : 1 ≤ Rw) (c : vb.Pos) (Ψ : L.Ω → List (List (OutZC F D Sl)) → ℝ)
    (h0 : ∀ ω bs, 0 ≤ Ψ ω bs) (h1 : ∀ ω bs, Ψ ω bs ≤ 1)
    (hoff : ∀ ω, ¬ ReadsDrawZ A P planZ tab vb c ω → Ψ ω = fun _ => 0) {Bt : ℝ} (hBt : 0 ≤ Bt)
    (htr : ∀ _ : Nonempty (ΩcZ A P planZ tab vb L c), expect (trialValZC planZ R Ψ)
      (trialGZC A Hs P planZ tab vb R τ c k Rw) (trialStratZC A Hs P planZ tab vb R τ c k Rw) ≤ Bt) :
    (∑ ω : L.Ω, expect (ind (accωZC planZ R ω)) (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 *
        expectAcc (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 (accωZC planZ R ω) k (Ψ ω)) /
        Fintype.card L.Ω ≤
      (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω * (Bt + k / (Real.exp 1 * Rw)) := by
  classical
  set f : L.Ω → ℝ := fun ω => expect (ind (accωZC planZ R ω)) (gameωZC A Hs planZ R τ ω)
      (stratωZC A Hs planZ R τ ω).2 *
    expectAcc (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 (accωZC planZ R ω) k (Ψ ω) with hf
  have hΩ : (0 : ℝ) < Fintype.card L.Ω := by exact_mod_cast Fintype.card_pos
  have hB : 0 ≤ Bt + k / (Real.exp 1 * Rw) := add_nonneg hBt (by positivity)
  have hoff' : ∀ ω, ¬ ReadsDrawZ A P planZ tab vb c ω → f ω = 0 := by
    intro ω hω
    simp only [hf, hoff ω hω, expectAcc_zero, mul_zero]
  have hsum : ∑ ω : L.Ω, f ω = ∑ w : ΩcZ A P planZ tab vb L c, f w.1 := by
    rw [← Finset.sum_filter_of_ne (p := fun ω => ReadsDrawZ A P planZ tab vb c ω)
      (fun ω _ hne => by by_contra h; exact hne (hoff' ω h))]
    exact Finset.sum_subtype _ (fun ω => by simp) f
  rw [hsum, div_le_iff₀ hΩ]
  by_cases hne : Nonempty (ΩcZ A P planZ tab vb L c)
  · have hw : ∀ w : ΩcZ A P planZ tab vb L c, f w.1 ≤
        expect (ind (accωZC planZ R w.1)) (gameωZC A Hs planZ R τ w.1) (stratωZC A Hs planZ R τ w.1).2 *
            geom (expect (ind (accωZC planZ R w.1)) (gameωZC A Hs planZ R τ w.1)
              (stratωZC A Hs planZ R τ w.1).2) Rw ^ k *
            expectN (gameωZC A Hs planZ R τ w.1) (stratωZC A Hs planZ R τ w.1).2 k
              (fun bs => (bs.map (ind (accωZC planZ R w.1))).prod * Ψ w.1 bs) + k / (Real.exp 1 * Rw) := fun w =>
      expectAcc_le_waits _ _ _ k Rw hRw _ (h0 w.1) (h1 w.1)
    have htr' := expect_trialValZC A Hs P planZ tab vb R τ c k Rw Ψ
    have hmass := htr hne
    have hcard : (0 : ℝ) < Fintype.card (ΩcZ A P planZ tab vb L c) := by exact_mod_cast Fintype.card_pos
    rw [htr', div_le_iff₀ hcard] at hmass
    calc ∑ w : ΩcZ A P planZ tab vb L c, f w.1
        ≤ ∑ w : ΩcZ A P planZ tab vb L c, (expect (ind (accωZC planZ R w.1)) (gameωZC A Hs planZ R τ w.1)
              (stratωZC A Hs planZ R τ w.1).2 *
            geom (expect (ind (accωZC planZ R w.1)) (gameωZC A Hs planZ R τ w.1)
              (stratωZC A Hs planZ R τ w.1).2) Rw ^ k *
            expectN (gameωZC A Hs planZ R τ w.1) (stratωZC A Hs planZ R τ w.1).2 k
              (fun bs => (bs.map (ind (accωZC planZ R w.1))).prod * Ψ w.1 bs) + k / (Real.exp 1 * Rw)) :=
          Finset.sum_le_sum fun w _ => hw w
      _ = (∑ w : ΩcZ A P planZ tab vb L c, expect (ind (accωZC planZ R w.1)) (gameωZC A Hs planZ R τ w.1)
              (stratωZC A Hs planZ R τ w.1).2 *
            geom (expect (ind (accωZC planZ R w.1)) (gameωZC A Hs planZ R τ w.1)
              (stratωZC A Hs planZ R τ w.1).2) Rw ^ k *
            expectN (gameωZC A Hs planZ R τ w.1) (stratωZC A Hs planZ R τ w.1).2 k
              (fun bs => (bs.map (ind (accωZC planZ R w.1))).prod * Ψ w.1 bs)) +
          Fintype.card (ΩcZ A P planZ tab vb L c) * (k / (Real.exp 1 * Rw)) := by
        rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
      _ ≤ Bt * Fintype.card (ΩcZ A P planZ tab vb L c) +
          Fintype.card (ΩcZ A P planZ tab vb L c) * (k / (Real.exp 1 * Rw)) := by linarith
      _ = _ := by rw [div_mul_eq_mul_div, div_mul_cancel₀ _ hΩ.ne']; ring
  · have h0' : (∑ w : ΩcZ A P planZ tab vb L c, f w.1) = 0 :=
      Finset.sum_eq_zero fun w _ => (hne ⟨w⟩).elim
    rw [h0']
    exact mul_nonneg (mul_nonneg (div_nonneg (Nat.cast_nonneg _) hΩ.le) hB) hΩ.le

/-- **The ideal rows mass at `c`**: the audit extractor's reading, at a registered read `c` whose registrant's
commitment is the registered one, of a value other than the registrant's, jointly with acceptance and averaged over the
draw, is at most `π c (t'(1 + k)/2^256 + k/(eRw))` under A2 for the rows finder. -/
theorem ideal_rows_leZC (k Rw : ℕ) (hRw : 1 ≤ Rw) {t' : ℝ} (ht : 0 ≤ t') (c : vb.Pos)
    (hon : isRead c → vb.com (wo c) = vb.cs R c)
    (hCR : ∀ _ : Nonempty (ΩcZ A P planZ tab vb L c), isRead c →
      Assumptions.SHA512CRExpected Hc (trialGZC A Hs P planZ tab vb R τ c k Rw)
        (trialStratZC A Hs P planZ tab vb R τ c k Rw) (rowsOutZC A Hs P planZ tab vb R τ wo c) (cTZC planZ R t')) :
    (∑ ω : L.Ω, expect (ind (accωZC planZ R ω)) (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 *
        expectAcc (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 (accωZC planZ R ω) k
          (psiRowsZC A Hs P planZ tab vb isRead wo R τ c ω)) / Fintype.card L.Ω ≤
      (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω *
        (t' * (1 + k) / 2 ^ (256 : ℝ) + k / (Real.exp 1 * Rw)) := by
  classical
  have hBt : 0 ≤ t' * (1 + k) / 2 ^ (256 : ℝ) := by positivity
  refine ideal_le_trialZC A Hs P planZ tab vb R τ k Rw hRw c _
    (psiRows_nonnegZC A Hs P planZ tab vb isRead wo R τ c) (psiRows_le_oneZC A Hs P planZ tab vb isRead wo R τ c)
    (fun ω hω => funext fun bs => ?_) hBt (fun hne => ?_)
  · simp [psiRowsZC, readsAt_eq_emptyZC A Hs P planZ tab vb R τ hω bs]
  · by_cases hrc : isRead c
    · exact trial_rows_leZC A Hs P planZ tab vb isRead wo R τ ht k Rw c hrc (hon hrc) (hCR hne hrc)
    · have h0 : trialValZC planZ R (psiRowsZC A Hs P planZ tab vb isRead wo R τ c) = fun _ => 0 :=
        funext fun e => by rw [trialVal_psiRowsZC]; simp [ind, hrc]
      rw [h0, expect_const]
      exact hBt

/-- **The rows link event leaves a trace**: if drawn unit `u`'s extracted table is committed but every close satisfying
message decodes off the value layer on some wire of `u`, or opens a registered read to a value other than the
registrant's, then some commit string reads a value other than its plurality, or a registered read a value other than
the registrant's. -/
theorem rowsLinkEv_le_psiZC (k Rw : ℕ) {ω : L.Ω} {u : Fin n} (hu : u ∈ L.draw ω)
    (bs : List (List (OutZC F D Sl))) :
    ind (fun bs => RowsLinkEvZK A P planZ tab vb isRead wo (L.draw ω) R u (XplurZC A Hs P planZ tab vb R τ k Rw)
      (tabExtZC A Hs planZ tab R τ ω u bs)) bs ≤
      ∑ c, (psiZC A Hs P planZ tab vb R τ k Rw c ω bs + psiRowsZC A Hs P planZ tab vb isRead wo R τ c ω bs) := by
  classical
  rw [Finset.sum_add_distrib]
  have hs0 : 0 ≤ ∑ c, psiZC A Hs P planZ tab vb R τ k Rw c ω bs :=
    Finset.sum_nonneg fun c _ => psi_nonnegZC A Hs P planZ tab vb R τ k Rw c ω bs
  have hr0 : 0 ≤ ∑ c, psiRowsZC A Hs P planZ tab vb isRead wo R τ c ω bs :=
    Finset.sum_nonneg fun c _ => psiRows_nonnegZC A Hs P planZ tab vb isRead wo R τ c ω bs
  have hi1 := ind_le_one (fun bs => RowsLinkEvZK A P planZ tab vb isRead wo (L.draw ω) R u
    (XplurZC A Hs P planZ tab vb R τ k Rw) (tabExtZC A Hs planZ tab R τ ω u bs)) bs
  by_cases hL : RowsLinkEvZK A P planZ tab vb isRead wo (L.draw ω) R u (XplurZC A Hs P planZ tab vb R τ k Rw)
      (tabExtZC A Hs planZ tab R τ ω u bs)
  · obtain ⟨hc, hall⟩ := hL
    have hspec := Classical.choose_spec hc
    have hcand : candZC A Hs planZ tab R τ ω u bs = some (Classical.choose hc) := by
      unfold candZC; rw [dite_eq_left_of_eq_true (eq_true hc)]
    have hmem : ∀ g ∈ P.io u, vb.val (vb.opening (L.draw ω) R u (Classical.choose hc) (vb.pos g)) ∈
        readsAtZC A Hs P planZ tab vb R τ ω bs (vb.pos g) := by
      intro g hg
      have hop : OpensAtZC A Hs P planZ tab vb R τ ω bs (vb.pos g)
          (vb.val (vb.opening (L.draw ω) R u (Classical.choose hc) (vb.pos g)))
          (vb.opening (L.draw ω) R u (Classical.choose hc) (vb.pos g)) :=
        ⟨u, hu, ⟨g, hg, rfl⟩, _, hcand, rfl, rfl⟩
      unfold readsAtZC; exact Finset.mem_filter.2 ⟨Finset.mem_univ _, _, hop⟩
    rcases hall (Classical.choose hc) hspec.1 hspec.2 with ⟨g, hg, hdec⟩ | ⟨g, hg, hrd, hne⟩
    · have hbit := (vb.opening_spec (L.draw ω) R u hu (Classical.choose hc) hspec.2 g hg).2
      have hne : vb.val (vb.opening (L.draw ω) R u (Classical.choose hc) (vb.pos g)) ≠
          XvalZC A Hs P planZ tab vb R τ k Rw (vb.pos g) := by
        intro heq
        apply hdec
        rw [← hbit, heq]
        rfl
      have hpsi : psiZC A Hs P planZ tab vb R τ k Rw (vb.pos g) ω bs = 1 := by
        have hex : ∃ v ∈ readsAtZC A Hs P planZ tab vb R τ ω bs (vb.pos g),
            v ≠ XvalZC A Hs P planZ tab vb R τ k Rw (vb.pos g) := ⟨_, hmem g hg, hne⟩
        unfold psiZC; rw [if_pos hex]
      have hle := Finset.single_le_sum (fun c _ => psi_nonnegZC A Hs P planZ tab vb R τ k Rw c ω bs)
        (Finset.mem_univ (vb.pos g))
      linarith
    · have hpsi : psiRowsZC A Hs P planZ tab vb isRead wo R τ (vb.pos g) ω bs = 1 := by
        have hex : isRead (vb.pos g) ∧ ∃ v ∈ readsAtZC A Hs P planZ tab vb R τ ω bs (vb.pos g),
            v ≠ vb.val (wo (vb.pos g)) := ⟨hrd, _, hmem g hg, hne⟩
        unfold psiRowsZC; rw [if_pos hex]
      have hle := Finset.single_le_sum (fun c _ => psiRows_nonnegZC A Hs P planZ tab vb isRead wo R τ c ω bs)
        (Finset.mem_univ (vb.pos g))
      linarith
  · have h0 : ind (fun bs => RowsLinkEvZK A P planZ tab vb isRead wo (L.draw ω) R u
        (XplurZC A Hs P planZ tab vb R τ k Rw) (tabExtZC A Hs planZ tab R τ ω u bs)) bs = 0 := by
      unfold ind; rw [if_neg hL]
    rw [h0]
    linarith

end Trial

/-! ## The analysis, and its link term -/

open Classical in
/-- **The compiled `--zk` audit's analysis, "wrong" against the registration and the registrant's rows**:
`analysisZCReg` with `rowsWrongZ` added to "wrong", and the rows added to its link event (`RowsLinkEvZK`). -/
noncomputable def analysisZCRows
    (Xc : (R : Reg) → Cont L Reg (batchedSessionZC A Hs planZ) R → (Fin C.N → Bool)) (k : ℕ)
    (hr : ∀ S R j, rateZ (planZ S R) j k < 1) (hlow : LoweringSoundZK A P planZ tab decode)
    (hown : ∀ u, P.ReadsOwn u)
    (a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ)
    (hj : JointZC A Hs planZ k a) :
    ExtractionAnalysis L Reg (batchedSessionZC A Hs planZ) where
  Tr := Reg × (Fin C.N → Bool)
  wrong Xt := wrongRegZ A P planZ tab vb Xt.1 Xt.2 ∪ rowsWrongZ A P planZ tab vb isRead wo Xt.2
  committed R τ := (R, Xc R τ)
  ks S R τ u := ENNReal.ofReal (a S R (zcStrat A Hs planZ τ) (tab S R u) / (1 - rateZ (planZ S R) (tab S R u) k))
  link S R τ u Xt := if Xt.1 = R then
      ENNReal.ofReal (accProbZC A Hs (planZ S R) (zcStrat A Hs planZ τ) *
        extractProbAccZ A Hs (planZ S R) (tab S R u) k (zcStrat A Hs planZ τ)
          (RowsLinkEvZK A P planZ tab vb isRead wo S R u Xt.2) / (1 - rateZ (planZ S R) (tab S R u) k))
    else ⊤
  cover S R τ u Xt hS hu := by
    obtain ⟨R', Xt⟩ := Xt
    by_cases hR : R' = R
    swap
    · simp only [hR, ↓reduceIte, add_top, le_top]
    subst hR
    simp only [↓reduceIte]
    have hG : ∀ T, (tabOfZ planZ tab S R' u).Committed A T →
        RowsLinkEvZK A P planZ tab vb isRead wo S R' u Xt T := fun T hc =>
      ⟨hc, fun M _ hsat => by
        by_contra hno
        have hno1 : ∀ g ∈ P.io u, decode S R' u M g = Xt g := fun g hg => by
          by_contra h; exact hno (Or.inl ⟨g, hg, h⟩)
        have hno2 : ∀ g ∈ P.io u, isRead (vb.pos g) →
            vb.val (vb.opening S R' u M (vb.pos g)) = vb.val (wo (vb.pos g)) := fun g hg hrd => by
          by_contra h; exact hno (Or.inr ⟨g, hg, hrd, h⟩)
        have hcX : P.Correct Xt u := P.correct_congr (fun g hg => (hno1 g hg).symm) (hlow S R' u hS M hsat)
        rcases Finset.mem_union.1 hu with hw | hw
        · refine (mem_wrongRegZ A P planZ tab vb).1 hw ⟨hcX, fun c => vb.opening S R' u M c, fun g hg => ?_⟩
          obtain ⟨h1, h2⟩ := vb.opening_spec S R' u hS M hsat g hg
          exact ⟨h1, h2.trans ((hno1 g hg).trans (P.correct_eval (hown u) hcX g hg))⟩
        · obtain ⟨g, hg, hrd, hne⟩ := (mem_rowsWrongZ A P planZ tab vb isRead wo).1 hw
          exact hne ((hno1 g hg).symm.trans ((vb.opening_spec S R' u hS M hsat g hg).2.symm.trans
            (congrArg (fun v => vb.bit v g) (hno2 g hg hrd))))⟩
    set ε := accProbZC A Hs (planZ S R') (zcStrat A Hs planZ τ)
    set r := rateZ (planZ S R') (tab S R' u) k
    set a' := a S R' (zcStrat A Hs planZ τ) (tab S R' u)
    set Lp := extractProbAccZ A Hs (planZ S R') (tab S R' u) k (zcStrat A Hs planZ τ)
      (RowsLinkEvZK A P planZ tab vb isRead wo S R' u Xt)
    have hr1 : 0 < 1 - r := by linarith [hr S R' (tab S R' u)]
    have ha : 0 ≤ a' := (hj S R' _ (tab S R' u)).1
    have hL : 0 ≤ Lp := extractProbAccZ_nonneg _ _ _ _ _ _ _
    have hε0 : 0 ≤ ε := accProbZC_nonneg _ _ _ _
    have key : ε ≤ a' / (1 - r) + ε * Lp / (1 - r) := by
      rcases hε0.eq_or_lt with h0 | hpos
      · rw [← h0]
        positivity
      · have h1 := one_le_fail_add_accZ A Hs (planZ S R') (tab S R' u) k (zcStrat A Hs planZ τ) _ hG hpos
        have h2 := (hj S R' (zcStrat A Hs planZ τ) (tab S R' u)).2
        rw [← add_div, le_div_iff₀ hr1]
        nlinarith
    show prob (· = true) (batchedSessionZC A Hs planZ S R') τ ≤
      ENNReal.ofReal (a' / (1 - r)) + ENNReal.ofReal (ε * Lp / (1 - r))
    rw [prob_batchedSessionZC_eq, ← ENNReal.ofReal_add (div_nonneg ha hr1.le)
      (div_nonneg (mul_nonneg hε0 hL) hr1.le)]
    exact ENNReal.ofReal_le_ofReal key

omit [Fintype K] in
/-- `analysisZCRows`'s knowledge term is `analysisZC`'s, `ksAvgZC`. -/
theorem zc_knowledgeSound_rows
    (Xc : (R : Reg) → Cont L Reg (batchedSessionZC A Hs planZ) R → (Fin C.N → Bool)) {k : ℕ}
    (hr : ∀ S R j, rateZ (planZ S R) j k < 1) (hlow : LoweringSoundZK A P planZ tab decode)
    (hown : ∀ u, P.ReadsOwn u)
    {a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ}
    (hj : JointZC A Hs planZ k a) :
    (analysisZCRows A Hs P planZ tab vb isRead wo Xc k hr hlow hown a hj).KnowledgeSound
      (ksAvgZC A Hs planZ k a) := by
  intro R τ tgt
  refine avg_mono fun ω => atTgt_le (fun u _ => ?_) (tgt ω)
  exact le_iSup (fun j : Fin (planZ (L.draw ω) R).length =>
    ENNReal.ofReal (a (L.draw ω) R (zcStrat A Hs planZ (τ ω)) j / (1 - rateZ (planZ (L.draw ω) R) j k)))
    (tab (L.draw ω) R u)

open Classical in
/-- **The link and rows bound, prover by prover**: twice `linkBoundZC`'s closed form at a prover whose link finders
(`LinkCRZC`) and rows finders (`RowsCRZC`) satisfy A2 and whose registration's commitments at the registered reads are
the registrant's, and `⊤` at any other. -/
noncomputable def rowsBoundZC (k Rw M : ℕ) (ρ t' : ℝ) (R : Reg) (τ : Cont L Reg (batchedSessionZC A Hs planZ) R) :
    ℝ≥0∞ :=
  if LinkCRZC A Hs P planZ tab vb R τ k Rw M t' ∧ RowsCRZC A Hs P planZ tab vb R τ isRead wo k Rw t' ∧
      ∀ c, isRead c → vb.com (wo c) = vb.cs R c then
    ENNReal.ofReal ((∑ c : vb.Pos, (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω) *
      (2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ)) +
    ENNReal.ofReal ((∑ c : vb.Pos, (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω) *
      (2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ))
  else ⊤

/-- **The rows link theorem**: `zk_linkSoundC` at `analysisZCRows`, whose link term carries the rows: at a prover whose
link and rows finders satisfy A2 and whose registered reads' commitments are the registrant's, `δ_link + δ_rows` with
each at most `Q_s/(1 − ρ)·(2t'(1 + k)/2^256 + 1/(eM) + k/(eR_w))` (`rowsBoundZC`). -/
theorem zk_rowsLinkSoundC (hlow : LoweringSoundZK A P planZ tab decode) (hown : ∀ u, P.ReadsOwn u) {k : ℕ}
    {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1) (hr : ∀ S R j, rateZ (planZ S R) j k ≤ ρ)
    {a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ}
    (hj : JointZC A Hs planZ k a) {t' : ℝ} (ht : 0 ≤ t') :
    (analysisZCRows (L := L) A Hs P planZ tab vb isRead wo (fun R τ => XplurZC A Hs P planZ tab vb R τ k Rw) k
        (fun S R j => (hr S R j).trans_lt hρ) hlow hown a hj).LinkSound
      (rowsBoundZC A Hs P planZ tab vb isRead wo k Rw M ρ t') := by
  intro R τ tgt
  unfold rowsBoundZC
  split_ifs with hCR
  swap
  · exact le_top
  obtain ⟨hCL, hCRr, hon⟩ := hCR
  set B := 2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)
    with hBdef
  have hρ1 : 0 < 1 - ρ := by linarith
  set f : vb.Pos → L.Ω → ℝ := fun c ω => expect (ind (accωZC planZ R ω)) (gameωZC A Hs planZ R τ ω)
      (stratωZC A Hs planZ R τ ω).2 *
    expectAcc (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 (accωZC planZ R ω) k
      (fun bs => psiZC A Hs P planZ tab vb R τ k Rw c ω bs + psiRowsZC A Hs P planZ tab vb isRead wo R τ c ω bs)
    with hf
  set Λ : L.Ω → ℝ := fun ω => (∑ c, f c ω) / (1 - ρ) with hΛ
  have hf0 : ∀ c ω, 0 ≤ f c ω := fun c ω => mul_nonneg (expect_nonneg (ind_nonneg _) _ _)
    (expectAcc_nonneg _ _ _ _ fun bs => add_nonneg (psi_nonnegZC A Hs P planZ tab vb R τ k Rw c ω bs)
      (psiRows_nonnegZC A Hs P planZ tab vb isRead wo R τ c ω bs))
  have hΛ0 : ∀ ω, 0 ≤ Λ ω := fun ω =>
    div_nonneg (Finset.sum_nonneg fun c _ => hf0 c ω) hρ1.le
  have step1 : ∀ ω, atTgt (L.draw ω) (tgt ω) (fun u =>
      (analysisZCRows (L := L) A Hs P planZ tab vb isRead wo (fun R τ => XplurZC A Hs P planZ tab vb R τ k Rw) k
        (fun S R j => (hr S R j).trans_lt hρ) hlow hown a hj).link (L.draw ω) R (τ ω) u
        ((analysisZCRows (L := L) A Hs P planZ tab vb isRead wo (fun R τ => XplurZC A Hs P planZ tab vb R τ k Rw) k
          (fun S R j => (hr S R j).trans_lt hρ) hlow hown a hj).committed R τ)) ≤ ENNReal.ofReal (Λ ω) := by
    intro ω
    refine atTgt_le (fun u hu => ?_) (tgt ω)
    simp only [analysisZCRows, ↓reduceIte]
    apply ENNReal.ofReal_le_ofReal
    have hε0 : 0 ≤ accProbZC A Hs (planZ (L.draw ω) R) (zcStrat A Hs planZ (τ ω)) :=
      expect_nonneg (ind_nonneg _) _ _
    have hLp0 := extractProbAccZ_nonneg A Hs (planZ (L.draw ω) R) (tab (L.draw ω) R u) k
      (zcStrat A Hs planZ (τ ω))
      (RowsLinkEvZK A P planZ tab vb isRead wo (L.draw ω) R u (XplurZC A Hs P planZ tab vb R τ k Rw))
    have hLp : extractProbAccZ A Hs (planZ (L.draw ω) R) (tab (L.draw ω) R u) k
        (zcStrat A Hs planZ (τ ω))
        (RowsLinkEvZK A P planZ tab vb isRead wo (L.draw ω) R u (XplurZC A Hs P planZ tab vb R τ k Rw)) ≤
          expectAcc (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 (accωZC planZ R ω) k
            (fun bs => ∑ c, (psiZC A Hs P planZ tab vb R τ k Rw c ω bs +
              psiRowsZC A Hs P planZ tab vb isRead wo R τ c ω bs)) :=
      expectAcc_mono _ _ _ _ fun bs => rowsLinkEv_le_psiZC A Hs P planZ tab vb isRead wo R τ k Rw hu bs
    have hrr : 1 - ρ ≤ 1 - rateZ (planZ (L.draw ω) R) (tab (L.draw ω) R u) k := by
      linarith [hr (L.draw ω) R (tab (L.draw ω) R u)]
    have hsum : expect (ind (accωZC planZ R ω)) (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 *
        expectAcc (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 (accωZC planZ R ω) k
          (fun bs => ∑ c, (psiZC A Hs P planZ tab vb R τ k Rw c ω bs +
            psiRowsZC A Hs P planZ tab vb isRead wo R τ c ω bs)) = ∑ c, f c ω := by
      rw [expectAcc_sum, Finset.mul_sum]
    refine (div_le_div_of_nonneg_left (mul_nonneg hε0 hLp0) hρ1 hrr).trans ?_
    show _ ≤ (∑ c, f c ω) / (1 - ρ)
    rw [← hsum]
    exact div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hLp hε0) hρ1.le
  refine (avg_mono step1).trans ?_
  rw [avg_ofReal Λ hΛ0]
  have hΩ : (0 : ℝ) < Fintype.card L.Ω := by exact_mod_cast Fintype.card_pos
  have hB0 : 0 ≤ B := by rw [hBdef]; positivity
  have hX : 0 ≤ (∑ c : vb.Pos, (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω) * B / (1 - ρ) :=
    div_nonneg (mul_nonneg (Finset.sum_nonneg fun c _ => div_nonneg (Nat.cast_nonneg _) hΩ.le) hB0) hρ1.le
  rw [← ENNReal.ofReal_add hX hX]
  apply ENNReal.ofReal_le_ofReal
  have hc : ∀ c, (∑ ω, f c ω) / Fintype.card L.Ω ≤
      (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω * (2 * B) := fun c => by
    have h1 : _ ≤ (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω * B :=
      ideal_mass_leZC A Hs P planZ tab vb R τ k Rw M hRw hM ht c (hCL c)
    have h2 := ideal_rows_leZC A Hs P planZ tab vb isRead wo R τ k Rw hRw ht c (hon c) (hCRr c)
    have hsplit : (∑ ω, f c ω) / Fintype.card L.Ω =
        (∑ ω : L.Ω, expect (ind (accωZC planZ R ω)) (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 *
          expectAcc (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 (accωZC planZ R ω) k
            (psiZC A Hs P planZ tab vb R τ k Rw c ω)) / Fintype.card L.Ω +
        (∑ ω : L.Ω, expect (ind (accωZC planZ R ω)) (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 *
          expectAcc (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 (accωZC planZ R ω) k
            (psiRowsZC A Hs P planZ tab vb isRead wo R τ c ω)) / Fintype.card L.Ω := by
      rw [← add_div, ← Finset.sum_add_distrib]
      congr 1
      refine Finset.sum_congr rfl fun ω _ => ?_
      rw [← mul_add, ← expectAcc_add]
    have hcc : 0 ≤ (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω :=
      div_nonneg (Nat.cast_nonneg _) hΩ.le
    have hle : t' * (1 + k) / 2 ^ (256 : ℝ) + k / (Real.exp 1 * Rw) ≤ B := by
      have e1 : 0 ≤ t' * (1 + k) / 2 ^ (256 : ℝ) := by positivity
      have e2 : 0 ≤ 1 / (Real.exp 1 * M) := by positivity
      have e3 : 2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) = 2 * (t' * (1 + k) / 2 ^ (256 : ℝ)) := by ring
      rw [hBdef, e3]
      linarith
    have h3 := mul_le_mul_of_nonneg_left hle hcc
    rw [hsplit]
    linarith
  calc (∑ ω, Λ ω) / Fintype.card L.Ω
      = (∑ c, (∑ ω, f c ω) / Fintype.card L.Ω) / (1 - ρ) := by
        simp only [hΛ, div_eq_mul_inv, ← Finset.sum_mul]
        rw [Finset.sum_comm]
        ring
    _ ≤ (∑ c, (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω * (2 * B)) / (1 - ρ) :=
        div_le_div_of_nonneg_right (Finset.sum_le_sum fun c _ => hc c) hρ1.le
    _ = _ := by rw [← Finset.sum_mul]; ring

/-- **The drawn units of the compiled `--zk` audit, against the registration and the registrant's rows**: it accepts
with a drawn unit in `wrongRegZ` or in `rowsWrongZ` with probability at most `ε_ks + δ_link + δ_rows`. -/
theorem zk_flock_drawnC_rows (hlow : LoweringSoundZK A P planZ tab decode) (hown : ∀ u, P.ReadsOwn u) {k Rw M : ℕ}
    (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1) (hr : ∀ S R j, rateZ (planZ S R) j k ≤ ρ)
    {a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ}
    (hj : JointZC A Hs planZ k a) {t' : ℝ} (ht : 0 ≤ t')
    (σ : Strategy (audit L Reg (batchedSessionZC A Hs planZ))) :
    prob (fun o => o.1 = true ∧
        ¬ Disjoint (wrongRegZ A P planZ tab vb (reg σ) (XplurZC A Hs P planZ tab vb (reg σ) (cont σ) k Rw) ∪
          rowsWrongZ A P planZ tab vb isRead wo (XplurZC A Hs P planZ tab vb (reg σ) (cont σ) k Rw)) o.2)
        (audit L Reg (batchedSessionZC A Hs planZ)) σ ≤
      ksAvgZC A Hs planZ k a (reg σ) (cont σ) +
        rowsBoundZC A Hs P planZ tab vb isRead wo k Rw M ρ t' (reg σ) (cont σ) :=
  extraction_audit_drawn (analysisZCRows A Hs P planZ tab vb isRead wo
      (fun R τ => XplurZC A Hs P planZ tab vb R τ k Rw) k (fun S R j => (hr S R j).trans_lt hρ) hlow hown a hj)
    (zc_knowledgeSound_rows A Hs P planZ tab vb isRead wo _ _ hlow hown hj)
    (zk_rowsLinkSoundC A Hs P planZ tab vb isRead wo hlow hown hRw hM hρ hr hj ht) σ

end FlockSoundness.Discharge.ZkLink
