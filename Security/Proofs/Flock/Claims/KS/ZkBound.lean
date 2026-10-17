import Proofs.Flock.Claims.KS.MixtureZk
import Proofs.Flock.Claims.KS.PlainBound

/-!
# The masked table bound from masked sessions after `Commit`

`PlainBound` at `--zk`. `TableBoundθAt` at the deployed masked record, at a statement whose setup has one table, from a
mixture of the old masked sessions after `Commit` at a padded table `T` (`zk_tableBoundθAt_of_sessions`). The escapes
are the old masked ones (`Flock.oldEscZ`) at fork rounds `j`, one round per rep and level, rep's pads root and rep's
`τ`, no two sharing one, so they fit the weight (`oldEscZ_fitsFork_zk`). At a tape `ω` and a table `C`, `ε_ω` is at
most the deployed runs' probability of accepting with every opened column on `C`, plus that of accepting with one off
`C`, which the bound keeps. The first is at most a mixture's mean probability of accepting after `Commit` with no
level-0 opening off `C` (`Off₀Z`). At a table the model does not commit to, that is at most each strategy's `ε_c⁻`
(`session_sound_of_table_accZ`, whose term off `C` is then empty), and the mean `ε_c⁻` is at most `ε_table` and the
escapes' term (`mean_epsSZ_le`). What is left are the hypotheses on the deployed prover:

* at each `ω` and `C`, a mixture bounds the deployed runs that accept on `C` (the transport of `Deploy.V true`'s runs
  to the live session's, with the decoder taking openings on `C` to openings on `C`);
* its mean `Adv`s (levels, pads roots, `τ`) and self-clash are at most the escapes' events (the escapes);
* a table that does not extract is not committed to `T`, at every table `Ok` allows (`zk_not_committed`, at `T`'s
  statement, schedule, padding and `c₀`);
* `T` is C-Flock ZK's table at the statement: its schedule, padding, extra lanes, shape, pads domain, queries and
  radius (`mean_epsSZ_le`'s), and a pads domain within the deployed one (`oldEscZ_fitsFork_zk`'s).

With the mixture the simulated strategies after `Commit` (`simAtZ`) of live strategies whose roots agree, the first is
the live verdict bounding the deployed runs that accept on `C`, and the decoder right for the verdict accepting with
no opening off `C` (`zk_tableBoundθAt_of_live`).
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge.Tables (LiveOutJ liveSession DecodesTable)
open FlockSoundness.Discharge.Composed (outJ repJ)

local notation "𝔽" => FlockLevel3.GF128

/-- A masked run opens a level-0 position, with a verifying path, to a row other than the oracle `C`'s. -/
def Off₀Z {D Sl : Type} {mPts : ℕ} (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts) (cap₀ : Model.Cap D) (C : Model.Oracle 𝔽)
    (out : OutZC 𝔽 D Sl) : Prop :=
  ∃ p, Rewinding.Off (op₀Z Hs.H Hs.E Hs.Lf T cap₀) (rowTW _ _ C) out p

/-- One masked table's acceptance: the output at its position is accepted. -/
def AccOneZ {D Sl : Type} [Inhabited D] [Inhabited Sl] (outs : List (OutZC 𝔽 D Sl)) : Prop :=
  (outs.getD 0 dfltOutZC).accepted = true

open Classical in
/-- **The masked table bound at a statement, from masked sessions after `Commit`**: where the setup has the one table
`st` and `T` is C-Flock ZK's padded table there, when every prover has escapes at fork rounds `j`, no two one, such
that, at each tape and each table `C` `Ok` allows, a mixture `σ` of masked sessions after a `Commit` of cap `cap₀`
bounds the deployed runs accepting on `C` by its acceptance with no opening off `C`, and the escapes' events bound its
`Adv`s and self-clash; and no table `Ok` allows that does not extract is committed to `T`. -/
theorem zk_tableBoundθAt_of_sessions {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D]
    [Inhabited Sl] (A : Model.Arith 𝔽 K) (hA : A.Correct) (Hs : Hashes 𝔽 D Sl)
    (rel : Readable.Parsed → Readable.Relation) (Ok : Claims.Msg → List 𝔽 → Prop)
    {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} (h : tables true x = [(spec, st)]) {mPts : ℕ}
    (T : TabZK 𝔽 mPts) (h25 : 25 ≤ st.m) (h35 : st.m ≤ 35) (hsch : T.sch = Accounting.m1Schedule st.m)
    (ht : T.t = Accounting.m1Pad st.m) (he : T.e = 2) (hS : T.S.shape mPts = acct spec st) (hN : T.N ≤ 2 ^ 13)
    (hq : T.q = 168) (hδ : 7 / 16 ≤ T.δ)
    (hNp : T.N ≤ Claims.Flock.Numbers.padDomain (Zk.PadLayout.new st.m st.c.kLog (Claims.Flock.zkSched st.m).1.k0))
    (hnc : ∀ C : Fin (wordLen true x) → List 𝔽, (∀ p, Ok x (C p)) →
      ¬ Claims.Flock.GoodTable (deployZk rel) (wordLen true) x C → ¬ T.Committed A (Claims.Flock.oracleOf C))
    (hmix : ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω),
      ∃ (j : Claims.Flock.IZ T.sch.levels.length → Fin ((V true).k x)) (_ : Function.Injective j)
        (ev : Claims.Flock.IZ T.sch.levels.length → Ω → (Fin ((V true).k x) → Coin) →
          (Fin ((V true).k x) → Coin) → Prop)
        (ev₁ : Ω → (Fin ((V true).k x) → Coin) → Prop)
        (_tied : ∀ i ω s s', ev i ω s s' → Claims.TiedColl sha512
          ((Claims.run sha512 P (V true) x ω s).vLog ++
            (Claims.run sha512 P (V true) x ω (Claims.forkCoins (j i) s s')).vLog))
        (_tied₁ : ∀ ω s, ev₁ ω s → Claims.TiedColl sha512 (Claims.run sha512 P (V true) x ω s).vLog),
        ∀ ω (C : Fin (wordLen true x) → List 𝔽), (∀ p, Ok x (C p)) →
          ∃ (τ : Type) (_ : Fintype τ) (_ : Nonempty τ) (cap₀ : Model.Cap D)
            (σ : τ → Strategy (sessAtZ [] [] (tabAfterZC A Hs T cap₀))),
            (Claims.Pr fun r : Fin ((V true).k x) → Coin => (Claims.run sha512 P (V true) x ω r).out = .acc ∧
                ¬ ∃ (p : Fin (wordLen true x)) (u : List 𝔽),
                  Claims.Flock.lookCol (cols true (Claims.run sha512 P (V true) x ω r).seen 0) p.val = some u ∧
                    u ≠ C p) ≤
              (∑ t, expect (ind fun outs => AccOneZ outs ∧
                  ¬ Off₀Z Hs T cap₀ (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutZC))
                (sessAtZ [] [] (tabAfterZC A Hs T cap₀)) (σ t)) / Fintype.card τ ∧
            (∀ r ℓ, (∑ t, advRSZ A Hs T [] [] cap₀ r.val (ℓ.val + 1) (σ t)) / Fintype.card τ ≤
              Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
                ev (.inl (r, ℓ)) ω s.1 s.2) ∧
            (∀ r : Fin 2, (∑ t, advPadsSZ A Hs T [] [] cap₀ r.val (σ t)) / Fintype.card τ ≤
              Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
                ev (.inr (.inl r)) ω s.1 s.2) ∧
            (∀ r : Fin 2, (∑ t, advTauSZ A Hs T [] [] cap₀ r.val (σ t)) / Fintype.card τ ≤
              Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
                ev (.inr (.inr r)) ω s.1 s.2) ∧
            (∑ t, expect (ind fun outs : List (OutZC 𝔽 D Sl) => SelfClashZ Hs T (outs.getD 0 dfltOutZC))
              (sessAtZ [] [] (tabAfterZC A Hs T cap₀)) (σ t)) / Fintype.card τ ≤
              Claims.Pr fun s : Fin ((V true).k x) → Coin => ev₁ ω s) :
    Claims.Flock.TableBoundθAt sha512 (deployZk rel) (wordLen true) Ok x := by
  intro Ω _ _ P
  obtain ⟨j, hj, ev, ev₁, tied, tied₁, hω⟩ := hmix Ω P
  refine ⟨Claims.Flock.oldEscZ sha512 (V true) P x T.sch T.N j ev ev₁ tied tied₁,
    oldEscZ_fitsFork_zk rel P x (tables_head_one h) h25 h35 hsch T.N hNp j hj ev ev₁ tied tied₁,
    fun ω C hok hg => ?_⟩
  obtain ⟨τ, _, _, cap₀, σ, hacc, hadv, hP, hT, hsc⟩ := hω ω C hok
  have hTc := hnc C hok hg
  have hper : ∀ t, expect (ind fun outs => AccOneZ outs ∧
        ¬ Off₀Z Hs T cap₀ (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutZC))
      (sessAtZ [] [] (tabAfterZC A Hs T cap₀)) (σ t) ≤ epsSZ A Hs T [] [] cap₀ (σ t) := by
    intro t
    have hs := session_sound_of_table_accZ A hA Hs T [] [] cap₀
      (fun outs => AccOneZ outs ∧ ¬ Off₀Z Hs T cap₀ (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutZC))
      (fun _ h => h.1) (Claims.Flock.oracleOf C) (σ t)
    have e : (ind fun outs : List (OutZC 𝔽 D Sl) =>
        (AccOneZ outs ∧ ¬ Off₀Z Hs T cap₀ (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutZC)) ∧
          ¬ T.Committed A (Claims.Flock.oracleOf C)) = ind fun outs => AccOneZ outs ∧
          ¬ Off₀Z Hs T cap₀ (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutZC) :=
      congrArg ind (funext fun _ => propext (and_iff_left hTc))
    have e₀ : (ind fun outs : List (OutZC 𝔽 D Sl) =>
        (AccOneZ outs ∧ ¬ Off₀Z Hs T cap₀ (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutZC)) ∧
          ∃ p, Rewinding.Off (op₀Z Hs.H Hs.E Hs.Lf T cap₀) (rowTW _ _ (Claims.Flock.oracleOf C))
            (outs.getD 0 dfltOutZC) p) = fun _ => 0 := by
      funext outs
      unfold ind
      split_ifs with h0
      · exact absurd h0.2 h0.1.2
      · rfl
    simp only [List.length_nil] at hs
    rw [e, e₀, expect_const, add_zero] at hs
    exact hs
  have hmean := mean_epsSZ_le FlockLevel3.card_F A Hs T [] [] cap₀ σ h h25 h35 hsch ht he hS hN hq hδ sha512
    (V true) P j ev ev₁ tied tied₁ ω hadv hP hT hsc
  have hsum := Finset.sum_le_sum fun t (_ : t ∈ Finset.univ) => hper t
  have hcard : (0 : ℝ) ≤ Fintype.card τ := Nat.cast_nonneg _
  have hdiv := div_le_div_of_nonneg_right hsum hcard
  have hsplit : Claims.accPr sha512 (V true) P x ω ≤
      (Claims.Pr fun r : Fin ((V true).k x) → Coin => (Claims.run sha512 P (V true) x ω r).out = .acc ∧
        ¬ ∃ (p : Fin (wordLen true x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols true (Claims.run sha512 P (V true) x ω r).seen 0) p.val = some u ∧ u ≠ C p) +
      Claims.Pr fun r : Fin ((V true).k x) → Coin => (Claims.run sha512 P (V true) x ω r).out = .acc ∧
        ∃ (p : Fin (wordLen true x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols true (Claims.run sha512 P (V true) x ω r).seen 0) p.val = some u ∧ u ≠ C p := by
    unfold Claims.accPr
    refine le_trans (Claims.Pr_mono fun r hr => ?_) (Claims.Pr_or_le _ _)
    by_cases hq : ∃ (p : Fin (wordLen true x)) (u : List 𝔽),
        Claims.Flock.lookCol (cols true (Claims.run sha512 P (V true) x ω r).seen 0) p.val = some u ∧ u ≠ C p
    · exact Or.inr ⟨hr, hq⟩
    · exact Or.inl ⟨hr, hq⟩
  have key : Claims.accPr sha512 (V true) P x ω ≤ εtable true x +
      (Claims.Flock.oldEscZ sha512 (V true) P x T.sch T.N j ev ev₁ tied tied₁).term ω +
      Claims.Pr fun r : Fin ((V true).k x) → Coin => (Claims.run sha512 P (V true) x ω r).out = .acc ∧
        ∃ (p : Fin (wordLen true x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols true (Claims.run sha512 P (V true) x ω r).seen 0) p.val = some u ∧ u ≠ C p := by
    linarith [hsplit, hacc, hmean, hdiv]
  exact key

open Classical in
/-- **The masked table bound at a statement, from live sessions**: `zk_tableBoundθAt_of_sessions` with the mixture the
simulated strategies after `Commit` (`simAtZ`) of live strategies `σL` whose roots agree. At each tape and table `C`
`Ok` allows, what is left is that the live verdict `pL` bounds the deployed runs accepting on `C`, that the decoder is
right for the verdict accepting with no opening off `C`, and that the escapes' events bound the simulated strategies'
`Adv`s and self-clash. -/
theorem zk_tableBoundθAt_of_live {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D]
    [Inhabited Sl] (A : Model.Arith 𝔽 K) (hA : A.Correct) (Hs : Hashes 𝔽 D Sl)
    (rel : Readable.Parsed → Readable.Relation) (Ok : Claims.Msg → List 𝔽 → Prop)
    {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} (h : tables true x = [(spec, st)]) {mPts : ℕ}
    (T : TabZK 𝔽 mPts) (h25 : 25 ≤ st.m) (h35 : st.m ≤ 35) (hsch : T.sch = Accounting.m1Schedule st.m)
    (ht : T.t = Accounting.m1Pad st.m) (he : T.e = 2) (hS : T.S.shape mPts = acct spec st) (hN : T.N ≤ 2 ^ 13)
    (hq : T.q = 168) (hδ : 7 / 16 ≤ T.δ)
    (hNp : T.N ≤ Claims.Flock.Numbers.padDomain (Zk.PadLayout.new st.m st.c.kLog (Claims.Flock.zkSched st.m).1.k0))
    (hnc : ∀ C : Fin (wordLen true x) → List 𝔽, (∀ p, Ok x (C p)) →
      ¬ Claims.Flock.GoodTable (deployZk rel) (wordLen true) x C → ¬ T.Committed A (Claims.Flock.oracleOf C))
    (hlive : ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω),
      ∃ (j : Claims.Flock.IZ T.sch.levels.length → Fin ((V true).k x)) (_ : Function.Injective j)
        (ev : Claims.Flock.IZ T.sch.levels.length → Ω → (Fin ((V true).k x) → Coin) →
          (Fin ((V true).k x) → Coin) → Prop)
        (ev₁ : Ω → (Fin ((V true).k x) → Coin) → Prop)
        (_tied : ∀ i ω s s', ev i ω s s' → Claims.TiedColl sha512
          ((Claims.run sha512 P (V true) x ω s).vLog ++
            (Claims.run sha512 P (V true) x ω (Claims.forkCoins (j i) s s')).vLog))
        (_tied₁ : ∀ ω s, ev₁ ω s → Claims.TiedColl sha512 (Claims.run sha512 P (V true) x ω s).vLog),
        ∀ ω (C : Fin (wordLen true x) → List 𝔽), (∀ p, Ok x (C p)) →
          ∃ (pL : LiveOutJ [T].length mPts → Prop) (opn : Array ByteArray → Fin [T].length → OpensZ 𝔽 D Sl)
            (dg : ByteArray → ByteArray) (dec : Fin [T].length → ℕ → ByteArray → List Msg)
            (one : Fin [T].length → ℕ → Bool) (fin : Fin [T].length → Fin 2 → Array ByteArray → List Msg)
            (capOf : Array ByteArray → Fin [T].length → Model.Cap D) (N : ℕ) (rootB₀ : Array ByteArray)
            (hdec : ∀ i, DecodesTable (outJ Hs [T]) (fun _ o => o.accepted = true ∧
                ¬ Off₀Z Hs T (capOf rootB₀ ⟨0, Nat.one_pos⟩) (Claims.Flock.oracleOf C) o)
              pL opn dec one fin capOf (repJ A Hs [T]) i)
            (τ : Type) (_ : Fintype τ) (_ : Nonempty τ) (σL : τ → Strategy (liveSession dg [T].length mPts N))
            (hσ : ∀ u, (σL u).1 = rootB₀),
            (Claims.Pr fun r : Fin ((V true).k x) → Coin => (Claims.run sha512 P (V true) x ω r).out = .acc ∧
                ¬ ∃ (p : Fin (wordLen true x)) (u : List 𝔽),
                  Claims.Flock.lookCol (cols true (Claims.run sha512 P (V true) x ω r).seen 0) p.val = some u ∧
                    u ≠ C p) ≤
              (∑ u, (prob pL _ (σL u)).toReal) / Fintype.card τ ∧
            (∀ r ℓ, (∑ u, advRSZ A Hs T [] [] (capOf rootB₀ ⟨0, Nat.one_pos⟩) r.val (ℓ.val + 1)
                (simAtZ A Hs T _ pL opn dg dec one fin capOf N hdec rootB₀ (σL u) (hσ u))) / Fintype.card τ ≤
              Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
                ev (.inl (r, ℓ)) ω s.1 s.2) ∧
            (∀ r : Fin 2, (∑ u, advPadsSZ A Hs T [] [] (capOf rootB₀ ⟨0, Nat.one_pos⟩) r.val
                (simAtZ A Hs T _ pL opn dg dec one fin capOf N hdec rootB₀ (σL u) (hσ u))) / Fintype.card τ ≤
              Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
                ev (.inr (.inl r)) ω s.1 s.2) ∧
            (∀ r : Fin 2, (∑ u, advTauSZ A Hs T [] [] (capOf rootB₀ ⟨0, Nat.one_pos⟩) r.val
                (simAtZ A Hs T _ pL opn dg dec one fin capOf N hdec rootB₀ (σL u) (hσ u))) / Fintype.card τ ≤
              Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
                ev (.inr (.inr r)) ω s.1 s.2) ∧
            (∑ u, expect (ind fun outs : List (OutZC 𝔽 D Sl) => SelfClashZ Hs T (outs.getD 0 dfltOutZC))
              (sessAtZ [] [] (tabAfterZC A Hs T (capOf rootB₀ ⟨0, Nat.one_pos⟩)))
              (simAtZ A Hs T _ pL opn dg dec one fin capOf N hdec rootB₀ (σL u) (hσ u))) / Fintype.card τ ≤
              Claims.Pr fun s : Fin ((V true).k x) → Coin => ev₁ ω s) :
    Claims.Flock.TableBoundθAt sha512 (deployZk rel) (wordLen true) Ok x := by
  refine zk_tableBoundθAt_of_sessions A hA Hs rel Ok h T h25 h35 hsch ht he hS hN hq hδ hNp hnc fun Ω _ _ P => ?_
  obtain ⟨j, hj, ev, ev₁, tied, tied₁, hω⟩ := hlive Ω P
  refine ⟨j, hj, ev, ev₁, tied, tied₁, fun ω C hok => ?_⟩
  obtain ⟨pL, opn, dg, dec, one, fin, capOf, N, rootB₀, hdec, τ, _, _, σL, hσ, htr, hadv, hP, hT, hsc⟩ := hω ω C hok
  exact ⟨τ, inferInstance, inferInstance, capOf rootB₀ ⟨0, Nat.one_pos⟩,
    fun u => simAtZ A Hs T _ pL opn dg dec one fin capOf N hdec rootB₀ (σL u) (hσ u),
    le_trans htr (mean_live_le_simAtZ A Hs T _ pL opn dg dec one fin capOf N hdec rootB₀ σL hσ),
    hadv, hP, hT, hsc⟩

end FlockDeploy.KS
