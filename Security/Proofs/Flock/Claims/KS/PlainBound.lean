import Proofs.Flock.Claims.KS.Mixture

/-!
# The plain table bound from salted sessions after `Commit`

`TableBoundθAt` at the deployed plain record, at a statement whose setup has one table, from a mixture of the old
salted sessions after `Commit` at table 0's model statement (`plain_tableBoundθAt_of_sessions`). The escapes are the
old ones (`Flock.oldEsc`) at the plain fork rounds (`plainJ`), so they fit the weight (`oldEsc_fitsFork_plainJ`). At
a tape `ω` and a table `C`, `ε_ω` is at most the deployed runs' probability of accepting with every opened column on
`C`, plus that of accepting with one off `C`, which the bound keeps. The first is at most a mixture's mean probability
of accepting after `Commit` with no level-0 opening off `C`. At a table the model does not commit to, that is at most
each strategy's `ε_c⁻` (`session_sound_of_table_accL`, whose term off `C` is then empty), and the mean `ε_c⁻` is at
most `ε_table` and the escapes' term (`mean_epsSL_le`). What is left are the hypotheses on the deployed prover:

* at each `ω` and `C`, a mixture bounds the deployed runs that accept on `C` (the transport of `Deploy.V false`'s runs
  to the live session's, with the decoder taking openings on `C` to openings on `C`);
* its mean `Adv` and self-clash are at most the escapes' events (the escapes);
* a table that does not extract is not committed to table 0's model statement, at every table `Ok` allows.
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.Exec

local notation "𝔽" => FlockLevel3.GF128

theorem tables_head_one {zk : Bool} {x : Claims.Msg} {spec : SessionSpec} {st : Stmt}
    (h : tables zk x = [(spec, st)]) : (tables zk x).head? = some (spec, st) := by
  rw [h]
  rfl

open Classical in
/-- **The plain table bound at a statement, from salted sessions after `Commit`**: where the setup has the one table
`st`, when every prover has escapes at the plain fork rounds such that, at each tape and each table `C` `Ok` allows, a
mixture `σ` of salted sessions after a `Commit` of cap `cap₀` bounds the deployed runs accepting on `C` by its
acceptance with no opening off `C`, and the escapes' events bound its `Adv` and self-clash; and no table `Ok` allows
that does not extract is committed to table 0's model statement. -/
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
                  Claims.Flock.lookCol (cols false (Claims.run sha512 P (V false) x ω r).seen) p.val = some u ∧
                    u ≠ C p) ≤
              (∑ t, expect (ind fun outs => AccOne outs ∧ ¬ Off₀L Refine.H512 Refine.enc512 Refine.leaf512
                  ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL))
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
  have hT := hnc C hok hg
  have hper : ∀ t, expect (ind fun outs => AccOne outs ∧ ¬ Off₀L Refine.H512 Refine.enc512 Refine.leaf512
        ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL))
      (sessAfterL execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st))
        ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal spec.mPts [] [] cap₀) (σ t) ≤
      epsSL execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st)) m k0 l₀ l₁ rest ext
          hk0 hm ptLocal spec.mPts [] [] cap₀ (σ t) := by
    intro t
    have hs := session_sound_of_table_accL execArith execArith_correct Refine.H512 Refine.enc512 Refine.leaf512
      (L.model (Refine.regionsOf st)) m k0 l₀ l₁ rest ext hsch hm ptLocal spec.mPts hlay [] [] cap₀
      (fun outs => AccOne outs ∧ ¬ Off₀L Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀
        (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL))
      (fun _ h => h.1) (Claims.Flock.oracleOf C) (σ t)
    have e : (ind fun outs : List (Model.OutCL 𝔽 Refine.Dig (List UInt8)) =>
        (AccOne outs ∧ ¬ Off₀L Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀
          (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL)) ∧
        ¬ Committed execArith (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
          (Claims.Flock.oracleOf C)) = ind fun outs => AccOne outs ∧ ¬ Off₀L Refine.H512 Refine.enc512
          Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL) :=
      congrArg ind (funext fun _ => propext (and_iff_left hT))
    have e₀ : (ind fun outs : List (Model.OutCL 𝔽 Refine.Dig (List UInt8)) =>
        (AccOne outs ∧ ¬ Off₀L Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀
          (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL)) ∧
        Off₀L Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀
          (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL)) = fun _ => 0 := by
      funext outs
      unfold ind
      split_ifs with h0
      · exact absurd h0.2 h0.1.2
      · rfl
    rw [← e]
    refine le_trans hs (le_of_eq ?_)
    simp only [List.length_nil]
    rw [e₀, expect_const, add_zero]
  have hmean := mean_epsSL_le execArith Refine.H512 Refine.enc512 Refine.leaf512 h L m k0 l₀ l₁ rest ext hsch hk0 hm
    ptLocal [] [] cap₀ σ sha512 (V false) P (plainJ (tables_head_one h) hsch) ev ev₁ tied tied₁ ω hadv hsc
  have hsum := Finset.sum_le_sum fun t (_ : t ∈ Finset.univ) => hper t
  have hcard : (0 : ℝ) ≤ Fintype.card τ := Nat.cast_nonneg _
  have hdiv := div_le_div_of_nonneg_right hsum hcard
  have hsplit : Claims.accPr sha512 (V false) P x ω ≤
      (Claims.Pr fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (V false) x ω r).out = .acc ∧
        ¬ ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512 P (V false) x ω r).seen) p.val = some u ∧ u ≠ C p) +
      Claims.Pr fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (V false) x ω r).out = .acc ∧
        ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512 P (V false) x ω r).seen) p.val = some u ∧ u ≠ C p := by
    unfold Claims.accPr
    refine le_trans (Claims.Pr_mono fun r hr => ?_) (Claims.Pr_or_le _ _)
    by_cases hq : ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
        Claims.Flock.lookCol (cols false (Claims.run sha512 P (V false) x ω r).seen) p.val = some u ∧ u ≠ C p
    · exact Or.inr ⟨hr, hq⟩
    · exact Or.inl ⟨hr, hq⟩
  have key : Claims.accPr sha512 (V false) P x ω ≤ εtable false x +
      (Claims.Flock.oldEsc sha512 (V false) P x ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (plainJ (tables_head_one h) hsch) ev
        ev₁ tied tied₁).term ω +
      Claims.Pr fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (V false) x ω r).out = .acc ∧
        ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512 P (V false) x ω r).seen) p.val = some u ∧ u ≠ C p := by
    linarith [hsplit, hacc, hmean, hdiv]
  exact key

end FlockDeploy.KS
