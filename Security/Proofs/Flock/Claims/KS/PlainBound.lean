import Proofs.Flock.Claims.KS.Mixture

/-!
# The plain table bound from salted sessions after `Commit`

`TableBoundθAt` at the deployed plain record, at a statement whose setup has one table, from a mixture of the old
salted sessions after `Commit` at table 0's model statement (`plain_tableBoundθAt_of_sessions`). The escapes are the
old ones (`Flock.oldEsc`) at the plain fork rounds (`plainJ`), so they fit the weight (`oldEsc_fitsFork_plainJ`). At
each tape `ω` the mixture's mean acceptance after `Commit` bounds `ε_ω`. At a table `C` the model does not commit to,
each strategy's acceptance is at most its `ε_c⁻` and its probability of accepting off `C`
(`session_sound_of_table_accL`). The mean `ε_c⁻` is at most `ε_table` and the escapes' term (`mean_epsSL_le`). What is
left are the hypotheses on the deployed prover:

* the mixture bounds `ε_ω` (the transport of `Deploy.V false`'s runs to the live session's);
* its mean probability of accepting off `C` is at most the deployed runs' of accepting with an opened column off `C`
  (the openings);
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
`st`, when every prover's escapes at the plain fork rounds and, at each tape, a mixture `σ` of salted sessions after a
`Commit` of cap `cap₀` bound `ε_ω`, the openings off each table `C` and the escapes' events, and no table `Ok` allows
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
        ∀ ω, ∃ (τ : Type) (_ : Fintype τ) (_ : Nonempty τ) (cap₀ : Model.Cap Refine.Dig)
          (σ : τ → Strategy (sessAfterL execArith Refine.H512 Refine.enc512 Refine.leaf512
            (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal spec.mPts [] [] cap₀)),
          Claims.accPr sha512 (V false) P x ω ≤
              (∑ t, expect (ind AccOne) (sessAfterL execArith Refine.H512 Refine.enc512 Refine.leaf512
                (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal spec.mPts [] [] cap₀)
                (σ t)) / Fintype.card τ ∧
            (∀ C : Fin (wordLen false x) → List 𝔽, (∀ p, Ok x (C p)) →
              (∑ t, expect (ind fun outs => AccOne outs ∧ Off₀L Refine.H512 Refine.enc512 Refine.leaf512
                  ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL))
                (sessAfterL execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st))
                  ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal spec.mPts [] [] cap₀) (σ t)) / Fintype.card τ ≤
                Claims.Pr fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (V false) x ω r).out = .acc ∧
                  ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
                    Claims.Flock.lookCol (cols false (Claims.run sha512 P (V false) x ω r).seen) p.val = some u ∧
                      u ≠ C p) ∧
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
  obtain ⟨τ, _, _, cap₀, σ, hacc, hopen, hadv, hsc⟩ := hω ω
  have hT := hnc C hok hg
  have hper : ∀ t, expect (ind AccOne) (sessAfterL execArith Refine.H512 Refine.enc512 Refine.leaf512
      (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal spec.mPts [] [] cap₀) (σ t) ≤
      epsSL execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st)) m k0 l₀ l₁ rest ext
          hk0 hm ptLocal spec.mPts [] [] cap₀ (σ t) +
        expect (ind fun outs => AccOne outs ∧ Off₀L Refine.H512 Refine.enc512 Refine.leaf512
            ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL))
          (sessAfterL execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st))
            ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal spec.mPts [] [] cap₀) (σ t) := by
    intro t
    have hs := session_sound_of_table_accL execArith execArith_correct Refine.H512 Refine.enc512 Refine.leaf512
      (L.model (Refine.regionsOf st)) m k0 l₀ l₁ rest ext hsch hm ptLocal spec.mPts hlay [] [] cap₀ AccOne
      (fun _ h => h) (Claims.Flock.oracleOf C) (σ t)
    have e : (ind fun outs : List (Model.OutCL 𝔽 Refine.Dig (List UInt8)) => AccOne outs ∧
        ¬ Committed execArith (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
          (Claims.Flock.oracleOf C)) = ind AccOne := by
      funext outs
      simp only [ind, hT, not_false_eq_true, and_true]
    rw [e] at hs
    exact hs
  have hmean := mean_epsSL_le execArith Refine.H512 Refine.enc512 Refine.leaf512 h L m k0 l₀ l₁ rest ext hsch hk0 hm
    ptLocal [] [] cap₀ σ sha512 (V false) P (plainJ (tables_head_one h) hsch) ev ev₁ tied tied₁ ω hadv hsc
  have hsum := Finset.sum_le_sum fun t (_ : t ∈ Finset.univ) => hper t
  rw [Finset.sum_add_distrib] at hsum
  have hcard : (0 : ℝ) ≤ Fintype.card τ := Nat.cast_nonneg _
  have hdiv := div_le_div_of_nonneg_right hsum hcard
  rw [add_div] at hdiv
  have key : Claims.accPr sha512 (V false) P x ω ≤ εtable false x +
      (Claims.Flock.oldEsc sha512 (V false) P x ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (plainJ (tables_head_one h) hsch) ev
        ev₁ tied tied₁).term ω +
      Claims.Pr fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (V false) x ω r).out = .acc ∧
        ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512 P (V false) x ω r).seen) p.val = some u ∧ u ≠ C p := by
    linarith [hacc, hopen C hok, hmean, hdiv]
  exact key

end FlockDeploy.KS
