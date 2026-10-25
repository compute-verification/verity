import Proofs.Flock.Claims.KS.Mixture
import Proofs.Flock.Claims.KS.SessionSC

/-!
# The plain table bound from salted sessions after `Commit`

`TableBoundθAt` at the deployed plain record, at a statement whose setup has one table, from a mixture of the old
salted sessions after `Commit` at table 0's model statement (`plain_tableBoundθAt_of_sessions`). The escapes are the
old ones (`Flock.oldEsc`) at the plain fork rounds (`plainJ`), so they fit the weight (`oldEsc_fitsFork_plainJ`). At
a tape `ω` and a table `C`, `ε_ω` is at most the deployed runs' probability of accepting with every opened column on
`C`, plus that of accepting with one off `C`, which the bound keeps. The first is at most a mixture's mean probability
of accepting after `Commit` with no level-0 opening off `C` unless it self-clashes. At a table the model does not
commit to, that is at most each strategy's `ε_c⁻` (`session_sound_of_table_accL_sc`, whose term off `C` without a
self-clash is then empty), and the mean `ε_c⁻` is at most `ε_table` and the escapes' term (`mean_epsSL_le`). That
bound at one tape and table is `plain_le_sessions`, and at any escapes whose term bounds the mean `ε_c⁻` it is
`plain_le_sessionsE`, which the slack's escapes reuse (`KS.ForkSlack`). What is left are the hypotheses on the
deployed prover:

* at each `ω` and `C`, a mixture bounds the deployed runs that accept on `C` (the transport of `Deploy.V false`'s runs
  to the live session's, with the decoder taking openings on `C` to openings on `C` or to a self-clash);
* its mean `Adv` and self-clash are at most the escapes' events (the escapes);
* a table that does not extract is not committed to table 0's model statement, at every table `Ok` allows.

With the mixture the simulated strategies after `Commit` of live strategies whose roots agree (`simAt`), the first
is the live verdict bounding the deployed runs that accept on `C`, and the decoder right for the verdict accepting
with no opening off `C` unless it self-clashes (`plain_tableBoundθAt_of_live`).
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.Exec

local notation "𝔽" => FlockLevel3.GF128

theorem tables_head_one {zk : Bool} {x : Claims.Msg} {spec : SessionSpec} {st : Stmt}
    (h : tables zk x = [(spec, st)]) : (tables zk x).head? = some (spec, st) := by
  rw [h]
  rfl

open Classical in
/-- **The plain bound at one tape and table, from salted sessions after `Commit`, at any escapes and any verifier of `Coin`s**: at a statement
`st`, a tape `ω` and a table `C` the model does not commit to, when a mixture `σ` of salted sessions after a `Commit`
of cap `cap₀` bounds the deployed runs accepting on `C` by its acceptance with no opening off `C` unless it
self-clashes, and the mixture's mean `ε_c⁻` is at most `ε_table` and the term of escapes `e`, `ε_ω` is at most
`ε_table`, `e`'s term and the deployed runs' probability of accepting with a column off `C`. -/
theorem plain_le_sessionsEV (Vr : Claims.Verifier Coin) {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} (L : Layout st) (m k0 : ℕ)
    (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m) (ptLocal : ℕ)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout ptLocal spec.mPts) {Ω : Type} (P : Claims.Prover Ω)
    (e : Claims.Escapes sha512 Vr P x) (ω : Ω)
    (C : Fin (wordLen false x) → List 𝔽)
    (hT : ¬ Committed execArith (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
      (Claims.Flock.oracleOf C))
    (τ : Type) [Fintype τ] [Nonempty τ] (cap₀ : Model.Cap Refine.Dig)
    (σ : τ → Strategy (sessAfterL execArith Refine.H512 Refine.enc512 Refine.leaf512
      (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal spec.mPts [] [] cap₀))
    (hacc : (Claims.Pr fun r : Fin (Vr.k x) → Coin => (Claims.run sha512 P Vr x ω r).out = .acc ∧
        ¬ ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512 P Vr x ω r).seen 0) p.val = some u ∧ u ≠ C p) ≤
      (∑ t, expect (ind fun outs => AccOne outs ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leaf512
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL) ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (outs.getD 0 dfltOutL)))
        (sessAfterL execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st))
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal spec.mPts [] [] cap₀) (σ t)) / Fintype.card τ)
    (hmean : (∑ t, epsSL execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st)) m k0 l₀
        l₁ rest ext hk0 hm ptLocal spec.mPts [] [] cap₀ (σ t)) / Fintype.card τ ≤ εtable false x + e.term ω) :
    Claims.accPr sha512 Vr P x ω ≤ εtable false x +
      e.term ω +
      Claims.Pr fun r : Fin (Vr.k x) → Coin => (Claims.run sha512 P Vr x ω r).out = .acc ∧
        ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512 P Vr x ω r).seen 0) p.val = some u ∧ u ≠ C p := by
  have hper : ∀ t, expect (ind fun outs => AccOne outs ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leaf512
        ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL) ∧
        ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (outs.getD 0 dfltOutL)))
      (sessAfterL execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st))
        ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal spec.mPts [] [] cap₀) (σ t) ≤
      epsSL execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st)) m k0 l₀ l₁ rest ext
          hk0 hm ptLocal spec.mPts [] [] cap₀ (σ t) := by
    intro t
    have hs := session_sound_of_table_accL_sc execArith execArith_correct Refine.H512 Refine.enc512 Refine.leaf512
      (L.model (Refine.regionsOf st)) m k0 l₀ l₁ rest ext hsch hm ptLocal spec.mPts hlay [] [] cap₀
      (fun outs => AccOne outs ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
        cap₀ (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL) ∧
        ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (outs.getD 0 dfltOutL)))
      (fun _ h => h.1) (Claims.Flock.oracleOf C) (σ t)
    have e : (ind fun outs : List (Model.OutCL 𝔽 Refine.Dig (List UInt8)) =>
        (AccOne outs ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀
          (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL) ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
            (outs.getD 0 dfltOutL))) ∧
        ¬ Committed execArith (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
          (Claims.Flock.oracleOf C)) = ind fun outs => AccOne outs ∧ ¬ (Off₀L Refine.H512 Refine.enc512
          Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL) ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
            (outs.getD 0 dfltOutL)) :=
      congrArg ind (funext fun _ => propext (and_iff_left hT))
    have e₀ : (ind fun outs : List (Model.OutCL 𝔽 Refine.Dig (List UInt8)) =>
        (AccOne outs ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀
          (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL) ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
            (outs.getD 0 dfltOutL))) ∧
        ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
          (outs.getD 0 dfltOutL) ∧
        Off₀L Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀
          (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL)) = fun _ => 0 := by
      funext outs
      unfold ind
      split_ifs with h0
      · exact absurd ⟨h0.2.2, h0.2.1⟩ h0.1.2
      · rfl
    rw [← e]
    refine le_trans hs (le_of_eq ?_)
    simp only [List.length_nil]
    rw [e₀, expect_const, add_zero]
  have hsum := Finset.sum_le_sum fun t (_ : t ∈ Finset.univ) => hper t
  have hcard : (0 : ℝ) ≤ Fintype.card τ := Nat.cast_nonneg _
  have hdiv := div_le_div_of_nonneg_right hsum hcard
  have hsplit : Claims.accPr sha512 Vr P x ω ≤
      (Claims.Pr fun r : Fin (Vr.k x) → Coin => (Claims.run sha512 P Vr x ω r).out = .acc ∧
        ¬ ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512 P Vr x ω r).seen 0) p.val = some u ∧ u ≠ C p) +
      Claims.Pr fun r : Fin (Vr.k x) → Coin => (Claims.run sha512 P Vr x ω r).out = .acc ∧
        ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512 P Vr x ω r).seen 0) p.val = some u ∧ u ≠ C p := by
    unfold Claims.accPr
    refine le_trans (Claims.Pr_mono fun r hr => ?_) (Claims.Pr_or_le _ _)
    by_cases hq : ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
        Claims.Flock.lookCol (cols false (Claims.run sha512 P Vr x ω r).seen 0) p.val = some u ∧ u ≠ C p
    · exact Or.inr ⟨hr, hq⟩
    · exact Or.inl ⟨hr, hq⟩
  have key : Claims.accPr sha512 Vr P x ω ≤ εtable false x +
      e.term ω +
      Claims.Pr fun r : Fin (Vr.k x) → Coin => (Claims.run sha512 P Vr x ω r).out = .acc ∧
        ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512 P Vr x ω r).seen 0) p.val = some u ∧ u ≠ C p := by
    linarith [hsplit, hacc, hmean, hdiv]
  exact key

open Classical in
/-- **The plain bound at one tape and table, from salted sessions after `Commit`, at any escapes**: `plain_le_sessionsEV` at
`V false`. -/
theorem plain_le_sessionsE {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} (L : Layout st) (m k0 : ℕ)
    (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m) (ptLocal : ℕ)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout ptLocal spec.mPts) {Ω : Type} (P : Claims.Prover Ω)
    (e : Claims.Escapes sha512 (V false) P x) (ω : Ω)
    (C : Fin (wordLen false x) → List 𝔽)
    (hT : ¬ Committed execArith (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
      (Claims.Flock.oracleOf C))
    (τ : Type) [Fintype τ] [Nonempty τ] (cap₀ : Model.Cap Refine.Dig)
    (σ : τ → Strategy (sessAfterL execArith Refine.H512 Refine.enc512 Refine.leaf512
      (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal spec.mPts [] [] cap₀))
    (hacc : (Claims.Pr fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (V false) x ω r).out = .acc ∧
        ¬ ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512 P (V false) x ω r).seen 0) p.val = some u ∧ u ≠ C p) ≤
      (∑ t, expect (ind fun outs => AccOne outs ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leaf512
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL) ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (outs.getD 0 dfltOutL)))
        (sessAfterL execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st))
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal spec.mPts [] [] cap₀) (σ t)) / Fintype.card τ)
    (hmean : (∑ t, epsSL execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st)) m k0 l₀
        l₁ rest ext hk0 hm ptLocal spec.mPts [] [] cap₀ (σ t)) / Fintype.card τ ≤ εtable false x + e.term ω) :
    Claims.accPr sha512 (V false) P x ω ≤ εtable false x +
      e.term ω +
      Claims.Pr fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (V false) x ω r).out = .acc ∧
        ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512 P (V false) x ω r).seen 0) p.val = some u ∧ u ≠ C p :=
  plain_le_sessionsEV (V false) L m k0 l₀ l₁ rest ext hsch hk0 hm ptLocal hlay P e ω C hT τ cap₀ σ hacc hmean

open Classical in
/-- **The plain bound at one tape and table, from salted sessions after `Commit`**: where the setup has the one table
`st`, at a tape `ω` and a table `C` the model does not commit to, when a mixture `σ` of salted sessions after a
`Commit` of cap `cap₀` bounds the deployed runs accepting on `C` by its acceptance with no opening off `C` unless it
self-clashes, and the events `ev`, `ev₁` bound its `Adv` and self-clash, `ε_ω` is at most `ε_table`, the old escapes'
term and the deployed runs' probability of accepting with a column off `C`. -/
theorem plain_le_sessions {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} (h : tables false x = [(spec, st)])
    (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m) (ptLocal : ℕ)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout ptLocal spec.mPts) {Ω : Type} (P : Claims.Prover Ω)
    (ev : Fin 2 → Fin (l₀ :: l₁ :: rest).length → Ω → (Fin ((V false).k x) → Coin) →
      (Fin ((V false).k x) → Coin) → Prop)
    (ev₁ : Ω → (Fin ((V false).k x) → Coin) → Prop)
    (tied : ∀ r ℓ ω s s', ev r ℓ ω s s' → Claims.TiedColl sha512
      ((Claims.run sha512 P (V false) x ω s).vLog ++
        (Claims.run sha512 P (V false) x ω (Claims.forkCoins (plainJ (tables_head_one h) hsch r ℓ) s s')).vLog))
    (tied₁ : ∀ ω s, ev₁ ω s → Claims.TiedColl sha512 (Claims.run sha512 P (V false) x ω s).vLog) (ω : Ω)
    (C : Fin (wordLen false x) → List 𝔽)
    (hT : ¬ Committed execArith (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
      (Claims.Flock.oracleOf C))
    (τ : Type) [Fintype τ] [Nonempty τ] (cap₀ : Model.Cap Refine.Dig)
    (σ : τ → Strategy (sessAfterL execArith Refine.H512 Refine.enc512 Refine.leaf512
      (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal spec.mPts [] [] cap₀))
    (hacc : (Claims.Pr fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (V false) x ω r).out = .acc ∧
        ¬ ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512 P (V false) x ω r).seen 0) p.val = some u ∧ u ≠ C p) ≤
      (∑ t, expect (ind fun outs => AccOne outs ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leaf512
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL) ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (outs.getD 0 dfltOutL)))
        (sessAfterL execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st))
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal spec.mPts [] [] cap₀) (σ t)) / Fintype.card τ)
    (hadv : ∀ r ℓ, (∑ t, advRSL execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st)) m
        k0 l₀ l₁ rest ext hk0 hm ptLocal spec.mPts [] [] cap₀ r.val (ℓ.val + 1) (σ t)) / Fintype.card τ ≤
      Claims.Pr fun s : (Fin ((V false).k x) → Coin) × (Fin ((V false).k x) → Coin) => ev r ℓ ω s.1 s.2)
    (hsc : (∑ t, expect (ind fun outs : List (Model.OutCL 𝔽 Refine.Dig (List UInt8)) =>
        SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (outs.getD 0 dfltOutL))
      (sessAfterL execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st))
        ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal spec.mPts [] [] cap₀) (σ t)) / Fintype.card τ ≤
      Claims.Pr fun s : Fin ((V false).k x) → Coin => ev₁ ω s) :
    Claims.accPr sha512 (V false) P x ω ≤ εtable false x +
      (Claims.Flock.oldEsc sha512 (V false) P x ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (plainJ (tables_head_one h) hsch) ev
        ev₁ tied tied₁).term ω +
      Claims.Pr fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (V false) x ω r).out = .acc ∧
        ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512 P (V false) x ω r).seen 0) p.val = some u ∧ u ≠ C p :=
  plain_le_sessionsE L m k0 l₀ l₁ rest ext hsch hk0 hm ptLocal hlay P
    (Claims.Flock.oldEsc sha512 (V false) P x ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (plainJ (tables_head_one h) hsch) ev ev₁
      tied tied₁) ω C hT τ cap₀ σ hacc
    (mean_epsSL_le execArith Refine.H512 Refine.enc512 Refine.leaf512 h L m k0 l₀ l₁ rest ext hsch hk0 hm ptLocal [] []
      cap₀ σ sha512 (V false) P (plainJ (tables_head_one h) hsch) ev ev₁ tied tied₁ ω hadv hsc)

open Classical in
/-- **The plain table bound at a statement, from salted sessions after `Commit`**: where the setup has the one table
`st`, when every prover has escapes at the plain fork rounds such that, at each tape and each table `C` `Ok` allows, a
mixture `σ` of salted sessions after a `Commit` of cap `cap₀` bounds the deployed runs accepting on `C` by its
acceptance with no opening off `C` unless it self-clashes, and the escapes' events bound its `Adv` and self-clash; and
no table `Ok` allows that does not extract is committed to table 0's model statement. -/
theorem plain_tableBoundθAt_of_sessions (rel : Readable.Parsed → Readable.Relation) (Ok : Claims.Msg → List 𝔽 → Prop)
    {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} (h : tables false x = [(spec, st)]) (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m) (ptLocal : ℕ)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout ptLocal spec.mPts)
    (hnc : ∀ C : Fin (wordLen false x) → List 𝔽, (∀ p, Ok x (C p)) →
      ¬ Claims.Flock.GoodTable (deployPlain rel) (wordLen false) x C →
      ¬ Committed execArith (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (Claims.Flock.oracleOf C))
    (hmix : ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω),
      ∃ (ev : Fin 2 → Fin (l₀ :: l₁ :: rest).length → Ω → (Fin ((V false).k x) → Coin) →
          (Fin ((V false).k x) → Coin) → Prop)
        (ev₁ : Ω → (Fin ((V false).k x) → Coin) → Prop)
        (_tied : ∀ r ℓ ω s s', ev r ℓ ω s s' → Claims.TiedColl sha512
          ((Claims.run sha512 P (V false) x ω s).vLog ++
            (Claims.run sha512 P (V false) x ω (Claims.forkCoins (plainJ (tables_head_one h) hsch r ℓ) s s')).vLog))
        (_tied₁ : ∀ ω s, ev₁ ω s → Claims.TiedColl sha512 (Claims.run sha512 P (V false) x ω s).vLog),
        ∀ ω (C : Fin (wordLen false x) → List 𝔽), (∀ p, Ok x (C p)) →
          ∃ (τ : Type) (_ : Fintype τ) (_ : Nonempty τ) (cap₀ : Model.Cap Refine.Dig)
            (σ : τ → Strategy (sessAfterL execArith Refine.H512 Refine.enc512 Refine.leaf512
              (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal spec.mPts [] [] cap₀)),
            (Claims.Pr fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (V false) x ω r).out = .acc ∧
                ¬ ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
                  Claims.Flock.lookCol (cols false (Claims.run sha512 P (V false) x ω r).seen 0) p.val = some u ∧
                    u ≠ C p) ≤
              (∑ t, expect (ind fun outs => AccOne outs ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leaf512
                  ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL) ∧
                  ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
                    (outs.getD 0 dfltOutL)))
                (sessAfterL execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st))
                  ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal spec.mPts [] [] cap₀) (σ t)) / Fintype.card τ ∧
            (∀ r ℓ, (∑ t, advRSL execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st)) m
                k0 l₀ l₁ rest ext hk0 hm ptLocal spec.mPts [] [] cap₀ r.val (ℓ.val + 1) (σ t)) / Fintype.card τ ≤
              Claims.Pr fun s : (Fin ((V false).k x) → Coin) × (Fin ((V false).k x) → Coin) => ev r ℓ ω s.1 s.2) ∧
            (∑ t, expect (ind fun outs : List (Model.OutCL 𝔽 Refine.Dig (List UInt8)) =>
                SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
                  (outs.getD 0 dfltOutL))
              (sessAfterL execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st))
                ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal spec.mPts [] [] cap₀) (σ t)) / Fintype.card τ ≤
              Claims.Pr fun s : Fin ((V false).k x) → Coin => ev₁ ω s) :
    Claims.Flock.TableBoundθAt sha512 (deployPlain rel) (wordLen false) Ok x := by
  intro Ω _ _ P
  obtain ⟨ev, ev₁, tied, tied₁, hω⟩ := hmix Ω P
  refine ⟨Claims.Flock.oldEsc sha512 (V false) P x ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (plainJ (tables_head_one h) hsch)
    ev ev₁ tied tied₁, oldEsc_fitsFork_plainJ rel P x (tables_head_one h) hsch ev ev₁ tied tied₁,
    fun ω C hok hg => ?_⟩
  obtain ⟨τ, _, _, cap₀, σ, hacc, hadv, hsc⟩ := hω ω C hok
  exact plain_le_sessions h L m k0 l₀ l₁ rest ext hsch hk0 hm ptLocal hlay P ev ev₁ tied tied₁ ω C (hnc C hok hg) τ
    cap₀ σ hacc hadv hsc

/-- Table 0's model spec at a plain statement of one table. -/
def plainTab {spec : SessionSpec} {st : Stmt} (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level)
    (rest : List Accounting.Level) (ext : ℕ) (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m) (ptLocal : ℕ)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout ptLocal spec.mPts) : TabSpec spec.mPts :=
  ⟨L.model (Refine.regionsOf st), m, k0, l₀, l₁, rest, ext, hsch, hm, ptLocal, hlay⟩

open Classical Discharge.Tables in
/-- **The plain table bound at a statement, from live sessions**: `plain_tableBoundθAt_of_sessions` with the mixture
the simulated strategies after `Commit` (`simAt`) of live strategies `σL` whose roots agree. At each tape and table
`C` `Ok` allows, what is left is that the live verdict `pL` bounds the deployed runs accepting on `C`, that the decoder
is right for the verdict accepting with no opening off `C` unless it self-clashes, and that the escapes' events bound
the simulated strategies' `Adv` and self-clash. -/
theorem plain_tableBoundθAt_of_live (rel : Readable.Parsed → Readable.Relation) (Ok : Claims.Msg → List 𝔽 → Prop)
    {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} (h : tables false x = [(spec, st)]) (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m) (ptLocal : ℕ)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout ptLocal spec.mPts)
    (hnc : ∀ C : Fin (wordLen false x) → List 𝔽, (∀ p, Ok x (C p)) →
      ¬ Claims.Flock.GoodTable (deployPlain rel) (wordLen false) x C →
      ¬ Committed execArith (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (Claims.Flock.oracleOf C))
    (hlive : ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω),
      ∃ (ev : Fin 2 → Fin (l₀ :: l₁ :: rest).length → Ω → (Fin ((V false).k x) → Coin) →
          (Fin ((V false).k x) → Coin) → Prop)
        (ev₁ : Ω → (Fin ((V false).k x) → Coin) → Prop)
        (_tied : ∀ r ℓ ω s s', ev r ℓ ω s s' → Claims.TiedColl sha512
          ((Claims.run sha512 P (V false) x ω s).vLog ++
            (Claims.run sha512 P (V false) x ω (Claims.forkCoins (plainJ (tables_head_one h) hsch r ℓ) s s')).vLog))
        (_tied₁ : ∀ ω s, ev₁ ω s → Claims.TiedColl sha512 (Claims.run sha512 P (V false) x ω s).vLog),
        ∀ ω (C : Fin (wordLen false x) → List 𝔽), (∀ p, Ok x (C p)) →
          ∃ (pL : LiveOutJ [plainTab L m k0 l₀ l₁ rest ext hsch hm ptLocal hlay].length spec.mPts → Prop)
            (opn : Array ByteArray → Fin [plainTab L m k0 l₀ l₁ rest ext hsch hm ptLocal hlay].length →
              Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
            (dg : ByteArray → ByteArray)
            (dec : Fin [plainTab L m k0 l₀ l₁ rest ext hsch hm ptLocal hlay].length → ℕ → ByteArray → List Msg)
            (one : Fin [plainTab L m k0 l₀ l₁ rest ext hsch hm ptLocal hlay].length → ℕ → Bool)
            (fin : Fin [plainTab L m k0 l₀ l₁ rest ext hsch hm ptLocal hlay].length → Fin 2 → Array ByteArray →
              List Msg)
            (capOf : Array ByteArray → Fin [plainTab L m k0 l₀ l₁ rest ext hsch hm ptLocal hlay].length →
              Model.Cap Refine.Dig)
            (N : ℕ) (rootB₀ : Array ByteArray)
            (hdec : ∀ j, DecodesTable
              (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leaf512 cap pend os)
              (fun _ o => o.accepted = true ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leaf512
                ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (capOf rootB₀ ⟨0, Nat.one_pos⟩) (Claims.Flock.oracleOf C) o ∧
                ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ o))
              pL opn dec one fin capOf
              (fun j => repsOf (D := Refine.Dig) execArith
                ([plainTab L m k0 l₀ l₁ rest ext hsch hm ptLocal hlay].get j))
              j)
            (τ : Type) (_ : Fintype τ) (_ : Nonempty τ)
            (σL : τ → Strategy
              (liveSession dg [plainTab L m k0 l₀ l₁ rest ext hsch hm ptLocal hlay].length spec.mPts N))
            (hσ : ∀ u, (σL u).1 = rootB₀),
            (Claims.Pr fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (V false) x ω r).out = .acc ∧
                ¬ ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
                  Claims.Flock.lookCol (cols false (Claims.run sha512 P (V false) x ω r).seen 0) p.val = some u ∧
                    u ≠ C p) ≤
              (∑ u, (prob pL _ (σL u)).toReal) / Fintype.card τ ∧
            (∀ r ℓ, (∑ u, advRSL execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st)) m
                k0 l₀ l₁ rest ext hk0 hm ptLocal spec.mPts [] [] (capOf rootB₀ ⟨0, Nat.one_pos⟩) r.val (ℓ.val + 1)
                (simAt execArith Refine.H512 Refine.enc512 Refine.leaf512
                  (plainTab L m k0 l₀ l₁ rest ext hsch hm ptLocal hlay) _ pL opn dg dec one fin capOf N hdec rootB₀
                  (σL u) (hσ u))) / Fintype.card τ ≤
              Claims.Pr fun s : (Fin ((V false).k x) → Coin) × (Fin ((V false).k x) → Coin) => ev r ℓ ω s.1 s.2) ∧
            (∑ u, expect (ind fun outs : List (Model.OutCL 𝔽 Refine.Dig (List UInt8)) =>
                SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
                  (outs.getD 0 dfltOutL))
              (sessAfterL execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st))
                ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal spec.mPts [] [] (capOf rootB₀ ⟨0, Nat.one_pos⟩))
              (simAt execArith Refine.H512 Refine.enc512 Refine.leaf512
                (plainTab L m k0 l₀ l₁ rest ext hsch hm ptLocal hlay) _ pL opn dg dec one fin capOf N hdec rootB₀
                (σL u) (hσ u))) / Fintype.card τ ≤
              Claims.Pr fun s : Fin ((V false).k x) → Coin => ev₁ ω s) :
    Claims.Flock.TableBoundθAt sha512 (deployPlain rel) (wordLen false) Ok x := by
  refine plain_tableBoundθAt_of_sessions rel Ok h L m k0 l₀ l₁ rest ext hsch hk0 hm ptLocal hlay hnc
    fun Ω _ _ P => ?_
  obtain ⟨ev, ev₁, tied, tied₁, hω⟩ := hlive Ω P
  refine ⟨ev, ev₁, tied, tied₁, fun ω C hok => ?_⟩
  obtain ⟨pL, opn, dg, dec, one, fin, capOf, N, rootB₀, hdec, τ, _, _, σL, hσ, htr, hadv, hsc⟩ := hω ω C hok
  exact ⟨τ, inferInstance, inferInstance, capOf rootB₀ ⟨0, Nat.one_pos⟩, fun u => simAt execArith Refine.H512
    Refine.enc512 Refine.leaf512 (plainTab L m k0 l₀ l₁ rest ext hsch hm ptLocal hlay) _ pL opn dg dec one fin capOf N
    hdec rootB₀ (σL u) (hσ u),
    le_trans htr (mean_live_le_simAt execArith Refine.H512 Refine.enc512 Refine.leaf512
      (plainTab L m k0 l₀ l₁ rest ext hsch hm ptLocal hlay) _ pL opn dg dec one fin capOf N hdec rootB₀ σL hσ),
    hadv, hsc⟩

end FlockDeploy.KS
